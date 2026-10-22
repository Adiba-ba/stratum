"""Flight-corridor evaluation subcommand for the STRATUM command line.

Two modes share one geometry:

- **great-circle route profile** (``--route VIDP-VABB``): 25 waypoints at
  ``--flight-level`` spaced 6 minutes apart from the departure time. This
  is a *route profile*, not a real flight track; the command says so when
  it prints and records ``trajectory_source: "great_circle_profile"``.
- **flown trajectory** (``--trajectory CSV``): cleaned ADS-B points as
  ingested by :mod:`stratum.io.adsb`, optionally selected by
  ``--callsign`` or ``--icao24``.

Without ``--model`` the command runs in physics-only mode: ERA5 state,
Schmidt-Appleman formation and ice supersaturation per waypoint, no
persistence probability and no ATC decision. With ``--model`` it runs the
full calibrated pipeline (:func:`evaluate_corridor_flight`) and prints the
waypoint audit plus the dispatcher advisory. The ERA5 file is required
either way: a missing file or a missing model artifact exits with code 2
instead of falling back to synthetic meteorology.

With ``--model`` the ``training_envelope`` recorded in the artifact
metadata is passed to the decision layer, so a recommended level whose ISA
pressure lies outside the range covered by the training rows is rejected as
an extrapolation. When the baseline lies outside a known envelope the
command warns that its probabilities extrapolate, and when the artifact
records no usable envelope it warns that the candidates are unchecked.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stratum.io.adsb import load_adsb_csv, select_trajectory
from stratum.io.era5 import load_era5_dataset
from stratum.model.artifacts import load_model
from stratum.model.envelope import pressure_envelope_bounds
from stratum.model.features import FEATURE_NAMES
from stratum.physics.thermo import fl_to_p
from stratum.pipeline.build_features import (
    extract_features_for_point,
    extract_features_for_trajectory,
)
from stratum.pipeline.runner import evaluate_corridor_flight
from stratum.reporting.html import probe_rhi_evidence, render_corridor_report
from stratum.routes.corridor import AIRPORTS, great_circle_waypoints

__all__ = ["corridor"]

#: Great-circle waypoint count and spacing of the route profile.
_DEFAULT_WAYPOINTS: int = 25
_STEP_MINUTES: float = 6.0

#: Time label offset used next to UTC departure times [minutes].
_IST_OFFSET_MINUTES: int = 330

#: Joules per gigajoule for energy-forcing formatting.
_GJ_IN_J: float = 1.0e9


def _parse_route(value: str) -> tuple[str, str]:
    """Parse ``VIDP-VABB`` into an ICAO ``(origin, destination)`` pair."""
    parts = [part.strip().upper() for part in value.split("-")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"--route {value!r} must be an ORIGIN-DESTINATION ICAO pair "
            "(for example VIDP-VABB)"
        )
    for code in parts:
        if code not in AIRPORTS:
            known = ", ".join(sorted(AIRPORTS))
            raise ValueError(
                f"unknown ICAO location indicator {code!r} in --route "
                f"{value!r}; known airports: {known}"
            )
    return parts[0], parts[1]


def _parse_levels(value: str) -> list[float]:
    """Parse ``330,350,370`` into positive flight levels."""
    levels: list[float] = []
    for chunk in value.split(","):
        text = chunk.strip()
        if not text:
            raise ValueError(f"empty flight level in --levels value {value!r}")
        try:
            level = float(text)
        except ValueError as error:
            raise ValueError(
                f"flight level {text!r} is not a number; expected e.g. 330,350,370"
            ) from error
        if not math.isfinite(level) or level <= 0.0:
            raise ValueError(f"flight level {text!r} must be positive")
        levels.append(level)
    if not levels:
        raise ValueError("--levels must contain at least one flight level")
    return levels


def _parse_departure(value: str) -> pd.Timestamp:
    """Parse an ISO-8601 departure time into a UTC pandas timestamp."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"--departure {value!r} is not a valid timestamp")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _time_axis(ds_met: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last ERA5 time step as UTC timestamps.

    Raises
    ------
    ValueError
        When the dataset has neither a ``time`` nor a ``valid_time``
        coordinate, or the axis is empty.
    """
    for name in ("time", "valid_time"):
        if name in ds_met.coords or name in ds_met.dims:
            values = np.asarray(ds_met[name].values)
            if values.size == 0:
                raise ValueError("ERA5 dataset contains no time steps")
            return (
                pd.Timestamp(values.min()).tz_localize("UTC"),
                pd.Timestamp(values.max()).tz_localize("UTC"),
            )
    raise ValueError("ERA5 dataset is missing the time coordinate")


def _format_ist(timestamp: pd.Timestamp) -> str:
    """``HH:MM`` IST label for a UTC timestamp."""
    return (timestamp + pd.Timedelta(minutes=_IST_OFFSET_MINUTES)).strftime("%H:%M")


def _format_time(time_utc: str) -> str:
    """``HH:MM`` UTC from an ISO-8601 timestamp string."""
    return pd.Timestamp(time_utc).strftime("%H:%M")


def _fmt(value: float, spec: str) -> str:
    """Format ``value`` or ``n/a`` when it is not finite."""
    number = float(value)
    return format(number, spec) if math.isfinite(number) else "n/a"


def _great_circle_frame(
    origin: str,
    destination: str,
    departure: pd.Timestamp,
    flight_level: float,
    num_points: int,
    *,
    callsign: str,
    icao24: str,
) -> pd.DataFrame:
    """Build the canonical trajectory frame of a great-circle route profile."""
    waypoints = great_circle_waypoints(
        AIRPORTS[origin],
        AIRPORTS[destination],
        num_points=num_points,
        fl=flight_level,
    )
    records = [
        {
            "icao24": icao24,
            "callsign": callsign,
            "time": departure + pd.Timedelta(minutes=_STEP_MINUTES * index),
            "lat": float(waypoint["lat"]),
            "lon": float(waypoint["lon"]),
            "baro_alt_ft": float(flight_level) * 100.0,
            "on_ground": False,
            "source": "great_circle_profile",
        }
        for index, waypoint in enumerate(waypoints)
    ]
    return pd.DataFrame(records)


def _load_trajectory(
    path: Path,
    callsign: str | None,
    icao24: str | None,
) -> pd.DataFrame:
    """Load a cleaned ADS-B trajectory, optionally selecting one flight.

    Raises
    ------
    ValueError
        For a missing/invalid CSV, an empty selection, or an ambiguous
        frame that carries several flights without an identifier.
    """
    frame = load_adsb_csv(path)
    if frame.empty:
        raise ValueError(
            f"trajectory {path} contains no usable airborne cruise points "
            "after cleaning"
        )
    if callsign is not None or icao24 is not None:
        selected = select_trajectory(frame, callsign=callsign, icao24=icao24)
        if selected.empty:
            wanted = callsign if callsign is not None else icao24
            raise ValueError(
                f"no trajectory points matched the requested flight {wanted!r}"
            )
        return selected
    if "icao24" in frame.columns and frame["icao24"].nunique(dropna=False) > 1:
        raise ValueError(
            f"trajectory {path} carries {frame['icao24'].nunique()} flights; "
            "select one with --callsign or --icao24"
        )
    return frame


def _physics_segments(
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Waypoint records for physics-only mode (no decisions)."""
    segments: list[dict[str, Any]] = []
    for index, info in enumerate(metadata):
        segments.append(
            {
                "segment_index": index,
                "status": "ok" if info["data_available"] else info["unavailable_reason"],
                "data_available": bool(info["data_available"]),
                "time_utc": info["time_utc"],
                "lat": float(info["lat"]),
                "lon": float(info["lon"]),
                "flight_level": float(info["flight_level"]),
                "pressure_pa": float(info["pressure_pa"]),
                "temperature_c": float(info["temperature_k"]) - 273.15,
                "rhi": float(info["rhi"]),
                "tlm_c": float(info["tlm_k"]) - 273.15,
                "tlc_c": float(info["tlc_k"]) - 273.15,
                "sac_forms": bool(info["sac_forms"]),
                "ice_supersaturated": bool(info["ambient_ice_supersaturated"]),
            }
        )
    return segments


def _screen_levels(
    ds_met: Any,
    origin: str,
    destination: str,
    departure: pd.Timestamp,
    levels: list[float],
    num_points: int,
    aircraft_type: str,
) -> list[dict[str, Any]]:
    """Sample SAC/ice statistics at extra great-circle cruise levels."""
    screen: list[dict[str, Any]] = []
    for level in levels:
        waypoints = great_circle_waypoints(
            AIRPORTS[origin],
            AIRPORTS[destination],
            num_points=num_points,
            fl=level,
        )
        available = 0
        sac_forming = 0
        supersaturated = 0
        for index, waypoint in enumerate(waypoints):
            _, info = extract_features_for_point(
                float(waypoint["lat"]),
                float(waypoint["lon"]),
                float(waypoint["p_Pa"]),
                departure + pd.Timedelta(minutes=_STEP_MINUTES * index),
                aircraft_type,
                ds_met=ds_met,
            )
            if not info["data_available"]:
                continue
            available += 1
            sac_forming += int(bool(info["sac_forms"]))
            supersaturated += int(bool(info["ambient_ice_supersaturated"]))
        screen.append(
            {
                "flight_level": float(level),
                "waypoints": len(waypoints),
                "data_available": available,
                "sac_forming": sac_forming,
                "ice_supersaturated": supersaturated,
            }
        )
    return screen


def _print_waypoint_table(segments: list[dict[str, Any]]) -> None:
    """Print the physics-only waypoint audit table."""
    header = (
        f"{'#':>3}  {'UTC':>5}  {'lat':>7}  {'lon':>7}  {'FL':>3}  "
        f"{'T[C]':>6}  {'RHi':>5}  {'SAC':>3}  {'ice':>3}  status"
    )
    print(header)
    for segment in segments:
        print(
            f"{int(segment['segment_index']) + 1:>3}  "
            f"{_format_time(str(segment['time_utc'])):>5}  "
            f"{float(segment['lat']):>7.2f}  {float(segment['lon']):>7.2f}  "
            f"{float(segment['flight_level']):>3.0f}  "
            f"{_fmt(float(segment['temperature_c']), '>6.1f')}  "
            f"{_fmt(float(segment['rhi']), '>5.2f')}  "
            f"{'yes' if segment['sac_forms'] else 'no':>3}  "
            f"{'yes' if segment['ice_supersaturated'] else 'no':>3}  "
            f"{segment['status']}"
        )


def _print_model_table(segments: list[dict[str, Any]]) -> None:
    """Print the calibrated waypoint audit table."""
    header = (
        f"{'#':>3}  {'UTC':>5}  {'lat':>7}  {'lon':>7}  {'FL':>3}  "
        f"{'T[C]':>6}  {'RHi':>5}  {'P(pers)':>7}  {'SAC':>4}  "
        f"{'ice':>3}  decision"
    )
    print(header)
    for segment in segments:
        decision = segment["decision_status"] or "-"
        if segment["decision_status"] == "eligible":
            decision = f"eligible -> FL{float(segment['recommended_fl']):.0f}"
        elif segment["decision_status"] == "no_robust_candidate":
            decision = "no robust candidate"
        print(
            f"{int(segment['segment_index']) + 1:>3}  "
            f"{_format_time(str(segment['time_utc'])):>5}  "
            f"{float(segment['lat']):>7.2f}  {float(segment['lon']):>7.2f}  "
            f"{float(segment['flight_level']):>3.0f}  "
            f"{float(segment['temperature_c']):>6.1f}  "
            f"{float(segment['rhi']):>5.2f}  "
            f"{float(segment['p_persist']):>7.3f}  "
            f"{'yes' if segment['sac_forms'] else 'no':>4}  "
            f"{'yes' if segment['ambient_ice_supersaturated'] else 'no':>3}  "
            f"{decision}"
        )


def _print_advisory(result: dict[str, Any]) -> None:
    """Print the ATC dispatcher advisory block."""
    reroutes = result["recommended_reroutes"]
    if not reroutes:
        print(
            "  No advisories: calibrated persistence risk stayed below the "
            "activation threshold (or no robust candidate) on all segments; "
            "flying the filed profile costs nothing extra."
        )
        return
    for reroute in reroutes:
        print(
            f"  Event {int(reroute['event_index']) + 1}: segment "
            f"#{int(reroute['segment_index']) + 1} at "
            f"{_format_time(str(reroute['time_utc']))} UTC, "
            f"FL{float(reroute['baseline_fl']):.0f} -> "
            f"FL{float(reroute['recommended_fl']):.0f} "
            f"({reroute['direction']})"
        )
        print(
            f"    Calibrated P(persist): {float(reroute['p_persist']):.3f} | "
            f"{int(reroute['segments_in_event'])} contiguous flagged segment(s)"
        )
        print(
            f"    Delta fuel: +{float(reroute['delta_fuel_kg']):.2f} kg | "
            f"delta CO2: +{float(reroute['delta_co2_kg']):.2f} kg"
        )
        print(
            f"    Avoided contrail warming: "
            f"{float(reroute['avoided_contrail_ef_j']) / _GJ_IN_J:.2f} GJ | "
            f"benefit-to-cost ratio "
            f"{float(reroute['benefit_to_cost_ratio']):.3e}"
        )
    count = len(reroutes)
    advisories = f"{count} advisory event" + ("s" if count != 1 else "")
    print(
        f"  Totals: {advisories} | "
        f"+{float(result['total_extra_fuel_kg']):.2f} kg fuel | "
        f"+{float(result['total_extra_co2_kg']):.2f} kg CO2 | "
        f"{float(result['total_avoidable_warming_j']) / _GJ_IN_J:.2f} GJ "
        "avoided warming"
    )


def _json_sanitize(value: Any) -> Any:
    """Recursively convert report values into standards-compliant JSON.

    Non-finite floats become ``None`` so the file never carries the bare
    ``NaN``/``Infinity`` tokens; numpy scalars become plain Python values.
    """
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_sanitize(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) or value is None:
        return value
    return str(value)


_PHYSICS_BANNER = (
    "PHYSICS-ONLY MODE: no calibrated model was supplied (--model), so no "
    "persistence probability and no ATC advisory is produced. The waypoints "
    "show ERA5 state, Schmidt-Appleman formation and ice supersaturation only."
)


def _render_physics_report(
    path: Path,
    corridor_data: dict[str, Any],
    source: dict[str, Any],
    flight: dict[str, Any],
) -> None:
    """Write a standalone HTML report for the physics-only mode.

    The library corridor report is not used here: it renders a calibrated
    advisory card and decision column, which would misrepresent a run
    without a model. This renderer prints the explicit no-model banner
    instead.
    """
    segments = corridor_data["segments"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rows = []
    for segment in segments:
        rows.append(
            "<tr>"
            f'<td class="num">{int(segment["segment_index"]) + 1}</td>'
            f'<td class="num">{_format_time(str(segment["time_utc"]))}</td>'
            f'<td class="num">{float(segment["lat"]):.2f}</td>'
            f'<td class="num">{float(segment["lon"]):.2f}</td>'
            f'<td class="num">{float(segment["flight_level"]):.0f}</td>'
            f'<td class="num">{_fmt(float(segment["temperature_c"]), ".1f")}</td>'
            f'<td class="num">{_fmt(float(segment["rhi"]), ".2f")}</td>'
            f'<td>{"yes" if segment["sac_forms"] else "no"}</td>'
            f'<td>{"yes" if segment["ice_supersaturated"] else "no"}</td>'
            f'<td>{segment["status"]}</td>'
            "</tr>"
        )
    summary = corridor_data["summary"]
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STRATUM // Corridor Physics Screen (no model)</title>
<style>
  body {{ margin: 0; color: #111; font-family: -apple-system, BlinkMacSystemFont,
    "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px; }}
  .page {{ max-width: 920px; margin: 0 auto; padding: 48px 28px 64px; }}
  h1 {{ font-size: 24px; margin: 6px 0 2px; }}
  .kicker {{ font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
    color: #666; }}
  .banner {{ border: 2px solid #111; padding: 14px 18px; margin-top: 22px;
    font-weight: 600; }}
  .meta {{ margin-top: 18px; color: #444; }}
  table {{ width: 100%; border-collapse: collapse; border: 1px solid #111;
    margin-top: 26px; }}
  th {{ text-align: left; font-size: 10.5px; letter-spacing: 0.1em;
    text-transform: uppercase; color: #666; padding: 9px 12px;
    border-bottom: 1px solid #111; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e5e5e5; }}
  td.num {{ text-align: right; font-family: ui-monospace, SFMono-Regular,
    Menlo, monospace; }}
  footer {{ margin-top: 40px; border-top: 1px solid #111; padding-top: 14px;
    font-size: 11.5px; color: #666; }}
</style>
</head>
<body>
<div class="page">
  <div class="kicker">STRATUM // Flight corridor physics screen</div>
  <h1>{corridor_data["origin_label"]} &#8594; {corridor_data["destination_label"]}</h1>
  <div class="banner">{_PHYSICS_BANNER}</div>
  <div class="meta">
    {corridor_data["num_waypoints"]} waypoints &#183; trajectory source
    <strong>{corridor_data["trajectory_source"]}</strong> &#183; departure
    {corridor_data["departure_utc"]} &#183; aircraft {flight["aircraft_type"]} &#183;
    weather {source["label"]}
  </div>
  <p>
    {summary["data_available_waypoints"]} of {summary["total_waypoints"]}
    waypoints carry ERA5 data; {summary["sac_forming_waypoints"]} are
    SAC-forming and {summary["ice_supersaturated_waypoints"]} are
    ice-supersaturated at the sampled level.
  </p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>UTC</th><th>Lat</th><th>Lon</th><th>FL</th>
        <th>Temp [&#176;C]</th><th>RHi</th><th>SAC forms</th>
        <th>Ice supersaturated</th><th>Status</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
  <footer>
    Physics screen only: this report contains no persistence probability and
    no reroute decision. Generated on {generated_at} from source
    {source["label"]}.
  </footer>
</div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_html, encoding="utf-8")


def corridor(argv: list[str]) -> int:
    """Evaluate one flight corridor against real ERA5 weather.

    Parameters
    ----------
    argv : list of str
        Arguments after ``corridor``: ``--era5`` and one of ``--route`` /
        ``--trajectory`` are required; ``--model`` switches from
        physics-only mode to the calibrated evaluation.

    Returns
    -------
    int
        0 on success, 2 for a missing ERA5 file or model artifact, an
        unknown route, an empty/ambiguous trajectory, or malformed
        options.
    """
    parser = argparse.ArgumentParser(
        prog="stratum corridor",
        description=(
            "Evaluate a great-circle route profile or a flown ADS-B "
            "trajectory against real ERA5 weather; with --model the "
            "calibrated persistence pipeline runs end to end."
        ),
    )
    parser.add_argument(
        "--era5",
        required=True,
        type=Path,
        metavar="PATH",
        help="ERA5 NetCDF file with the canonical variables",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--route",
        metavar="A-B",
        help="ICAO route pair for a great-circle cruise profile, e.g. VIDP-VABB",
    )
    target.add_argument(
        "--trajectory",
        type=Path,
        metavar="CSV",
        help="cleaned OpenSky-style ADS-B CSV of one flown trajectory",
    )
    parser.add_argument(
        "--callsign",
        default=None,
        help="callsign of the trajectory to select (implies --trajectory)",
    )
    parser.add_argument(
        "--icao24",
        default=None,
        help="ICAO 24-bit address of the trajectory to select",
    )
    parser.add_argument(
        "--departure",
        default=None,
        metavar="ISO8601",
        help="departure time of the route profile (default: earliest ERA5 step)",
    )
    parser.add_argument(
        "--aircraft",
        default="A320",
        help="engine designator for the feature assembly (default A320)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="MODEL.joblib",
        help="calibrated model artifact; without it the run is physics-only",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="HTML",
        help="write the corridor HTML report to this path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="JSON",
        help="write the corridor result dictionary to this JSON path",
    )
    parser.add_argument(
        "--flight-level",
        type=float,
        default=350.0,
        help="cruise flight level of the route profile (default 350)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=_DEFAULT_WAYPOINTS,
        metavar="N",
        help=f"waypoints of the route profile (default {_DEFAULT_WAYPOINTS})",
    )
    parser.add_argument(
        "--levels",
        default=None,
        metavar="FL[,FL...]",
        help=(
            "additional great-circle cruise levels to screen for context "
            "(route profiles only)"
        ),
    )
    args = parser.parse_args(argv)

    if not args.era5.is_file():
        print(
            f"ERROR: ERA5 file not found: {args.era5}\n"
            "Download the real reanalysis first (see `stratum fetch`) or pass "
            "an existing NetCDF file; no synthetic meteorology is substituted.",
            file=sys.stderr,
        )
        return 2
    if args.model is not None and not args.model.is_file():
        print(
            f"ERROR: model artifact not found: {args.model}\n"
            "Train one with `stratum train`; without --model the command runs "
            "in physics-only mode.",
            file=sys.stderr,
        )
        return 2
    if args.flight_level <= 0.0 or not math.isfinite(args.flight_level):
        print(
            f"ERROR: --flight-level must be positive, got {args.flight_level}",
            file=sys.stderr,
        )
        return 2
    if args.points < 2:
        print(f"ERROR: --points must be >= 2, got {args.points}", file=sys.stderr)
        return 2

    route: tuple[str, str] | None = None
    if args.route is not None:
        try:
            route = _parse_route(args.route)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    extra_levels: list[float] | None = None
    if args.levels is not None:
        try:
            extra_levels = _parse_levels(args.levels)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    departure: pd.Timestamp | None = None
    if args.departure is not None:
        try:
            departure = _parse_departure(args.departure)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        ds_met = load_era5_dataset(str(args.era5))
    except FileNotFoundError:
        print(f"ERROR: ERA5 file not found: {args.era5}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        first_time, last_time = _time_axis(ds_met)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    origin = ""
    destination = ""
    origin_label = ""
    destination_label = ""
    cruise_fl: float | None = None
    departure_assumed = False

    if route is not None:
        origin, destination = route
        origin_label = origin
        destination_label = destination
        cruise_fl = float(args.flight_level)
        if departure is None:
            departure = first_time
            departure_assumed = True
        print(
            f"Trajectory: great-circle route profile {origin} -> {destination} "
            f"at FL{cruise_fl:.0f} ({args.points} waypoints at "
            f"{_STEP_MINUTES:.0f}-minute spacing)."
        )
        print(
            "This is a route profile, NOT a real flight track; pass "
            "--trajectory with an ADS-B CSV for a flown track "
            "(trajectory_source=great_circle_profile recorded)."
        )
        trajectory = _great_circle_frame(
            origin,
            destination,
            departure,
            cruise_fl,
            args.points,
            callsign=args.callsign or f"{origin}-{destination}",
            icao24=args.icao24 or "",
        )
        window_end = departure + pd.Timedelta(
            minutes=_STEP_MINUTES * (args.points - 1)
        )
        if departure < first_time or window_end > last_time:
            print(
                "WARNING: the profile window "
                f"{departure.isoformat()} .. {window_end.isoformat()} extends "
                "beyond the ERA5 time coverage "
                f"{first_time.isoformat()} .. {last_time.isoformat()}; the "
                "affected waypoints report out_of_domain."
            )
        trajectory_source = "great_circle_profile"
    else:
        try:
            trajectory = _load_trajectory(
                args.trajectory, args.callsign, args.icao24
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        trajectory_source = "adsb_trajectory"
        trajectory_times = pd.to_datetime(trajectory["time"], utc=True)
        departure = trajectory_times.min()
        departure_assumed = False
        cruise_altitudes = pd.to_numeric(trajectory["baro_alt_ft"], errors="coerce")
        cruise_fl = float(cruise_altitudes.median() / 100.0)
        if not math.isfinite(cruise_fl):
            cruise_fl = None
        identifier = args.callsign or args.icao24 or "all flights"
        origin_label = f"ADS-B trajectory ({identifier})"
        destination_label = ""
        print(
            f"Trajectory: cleaned ADS-B flight track ({identifier}), "
            f"{len(trajectory)} points, "
            f"{departure.isoformat()} .. {trajectory_times.max().isoformat()}"
        )
        if args.departure is not None:
            print(
                "NOTE: --departure applies to great-circle route profiles; a "
                "flown trajectory keeps its own timestamps."
            )

    print(
        f"Departure: {departure.strftime('%Y-%m-%d %H:%M')} UTC "
        f"({_format_ist(departure)} IST)"
        + (" [assumed: earliest ERA5 time step]" if departure_assumed else "")
    )
    print(f"Weather data: ECMWF ERA5 reanalysis ({args.era5.name})")

    if extra_levels is not None and trajectory_source == "adsb_trajectory":
        print(
            "NOTE: --levels applies to great-circle route profiles only; it is "
            "ignored for --trajectory."
        )
        extra_levels = None

    level_screen: list[dict[str, Any]] | None = None
    if extra_levels is not None and route is not None:
        level_screen = _screen_levels(
            ds_met,
            origin,
            destination,
            departure,
            extra_levels,
            args.points,
            args.aircraft,
        )
        print("Additional great-circle cruise-level screen (route context):")
        for entry in level_screen:
            print(
                f"  FL{float(entry['flight_level']):.0f}: "
                f"{int(entry['data_available'])}/{int(entry['waypoints'])} with "
                f"ERA5 data | {int(entry['sac_forming'])} SAC-forming | "
                f"{int(entry['ice_supersaturated'])} ice-supersaturated"
            )

    model = None
    model_metadata: dict[str, Any] = {}
    if args.model is not None:
        try:
            model, model_metadata = load_model(args.model)
        except FileNotFoundError:
            print(f"ERROR: model artifact not found: {args.model}", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        artifact_features = model_metadata.get("feature_names")
        if artifact_features is not None and list(artifact_features) != list(
            FEATURE_NAMES
        ):
            print(
                "ERROR: model artifact feature contract does not match this "
                f"build for {args.model}; expected the frozen "
                f"{len(FEATURE_NAMES)}-feature order. Retrain with `stratum train`.",
                file=sys.stderr,
            )
            return 2

    envelope = model_metadata.get("training_envelope")
    pressure_envelope = pressure_envelope_bounds(envelope)
    if model is not None:
        if pressure_envelope is None:
            print(
                "WARNING: the model artifact records no usable training "
                "envelope; candidate levels are not checked against the "
                "training range and may extrapolate beyond the training data."
            )
        elif cruise_fl is not None and math.isfinite(cruise_fl):
            baseline_pressure_pa = fl_to_p(cruise_fl) * 100.0
            low_pa, high_pa = pressure_envelope
            if not low_pa <= baseline_pressure_pa <= high_pa:
                print(
                    f"WARNING: baseline FL{cruise_fl:.0f} "
                    f"({baseline_pressure_pa:.0f} Pa) lies outside the model "
                    f"training envelope ({low_pa:.0f}..{high_pa:.0f} Pa); its "
                    "calibrated persistence probabilities extrapolate beyond "
                    "the training data."
                )

    source = {
        "label": f"ECMWF ERA5 reanalysis ({args.era5.name})",
        "era5_filename": args.era5.name,
    }
    flight = {
        "callsign": args.callsign or "",
        "icao24": args.icao24 or "",
        "aircraft_type": args.aircraft,
    }

    if model is None:
        try:
            _, metadata = extract_features_for_trajectory(
                trajectory, ds_met, args.aircraft
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        segments = _physics_segments(metadata)
        summary = {
            "total_waypoints": len(segments),
            "data_available_waypoints": sum(
                1 for segment in segments if segment["data_available"]
            ),
            "sac_forming_waypoints": sum(
                1 for segment in segments if segment["sac_forms"]
            ),
            "ice_supersaturated_waypoints": sum(
                1 for segment in segments if segment["ice_supersaturated"]
            ),
        }

        print()
        print(_PHYSICS_BANNER)
        print()
        _print_waypoint_table(segments)
        print()
        print("Corridor summary")
        print("-" * 78)
        print(
            f"  Waypoints: {summary['total_waypoints']} total | "
            f"{summary['data_available_waypoints']} with ERA5 data | "
            f"{summary['sac_forming_waypoints']} SAC-forming | "
            f"{summary['ice_supersaturated_waypoints']} ice-supersaturated"
        )
        print("  Decisions: none (physics-only mode)")

        corridor_data: dict[str, Any] = {
            "trajectory_source": trajectory_source,
            "mode": "physics_only",
            "model": None,
            "origin": origin,
            "destination": destination,
            "origin_label": origin_label,
            "destination_label": destination_label,
            "num_waypoints": len(segments),
            "cruise_fl": cruise_fl,
            "departure_utc": departure.isoformat(),
            "departure_assumed": departure_assumed,
            "departure_local_label": f"{_format_ist(departure)} IST",
            "weather_source": source["label"],
            "segments": segments,
            "summary": summary,
            "level_screen": level_screen,
        }
        if args.out is not None:
            _render_physics_report(args.out, corridor_data, source, flight)
            print(f"HTML report written: {args.out}")
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(
                    _json_sanitize(corridor_data), indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"JSON written: {args.json}")
        return 0

    warnings.filterwarnings(
        "ignore",
        message=(
            "X does not have valid feature names, but LGBMClassifier was "
            "fitted with feature names"
        ),
    )
    try:
        result = evaluate_corridor_flight(
            trajectory,
            ds_met,
            model,
            aircraft_type=args.aircraft,
            pressure_envelope=pressure_envelope,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print()
    print("Segment-by-segment atmospheric and contrail risk breakdown")
    print("-" * 78)
    _print_model_table(result["segments"])
    print()
    print("ATC Dispatcher Advisory")
    print("-" * 78)
    _print_advisory(result)
    print()
    print("Corridor summary")
    print("-" * 78)
    print(
        f"  Segments: {int(result['total_segments'])} total | "
        f"{int(result['contrail_forming_segments'])} SAC-forming | "
        f"{int(result['contrail_persistent_segments'])} predicted persistent | "
        f"{int(result['flagged_segments'])} flagged for avoidance"
    )

    corridor_data = {
        "trajectory_source": trajectory_source,
        "mode": "calibrated_model",
        "model": {
            "path": str(args.model),
            "feature_names": list(FEATURE_NAMES),
            "metadata": model_metadata,
        },
        "origin": origin,
        "destination": destination,
        "origin_label": origin_label,
        "destination_label": destination_label,
        "num_waypoints": int(result["total_segments"]),
        "cruise_fl": cruise_fl,
        "departure_utc": departure.isoformat(),
        "departure_assumed": departure_assumed,
        "departure_local_label": f"{_format_ist(departure)} IST",
        "weather_source": source["label"],
        "level_screen": level_screen,
        "result": result,
    }

    reroutes = result["recommended_reroutes"]
    if reroutes:
        primary = reroutes[0]
        corridor_data["rhi_probe"] = probe_rhi_evidence(
            ds_met,
            result["segments"],
            float(primary["baseline_fl"]),
            float(primary["recommended_fl"]),
        )

    if args.out is not None:
        render_corridor_report(
            args.out,
            corridor=corridor_data,
            flight=flight,
            source=source,
            model=model_metadata,
            test_summary=None,
        )
        print(f"HTML report written: {args.out}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(_json_sanitize(corridor_data), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"JSON written: {args.json}")
    return 0
