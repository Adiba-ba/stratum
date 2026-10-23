"""Vertical contrail-risk profile per waypoint for the STRATUM command line.

``stratum profile`` samples one or more cruise flight levels at every
waypoint of a route and answers a dispatcher question: *which cruise level
minimises the calibrated contrail-persistence risk at this point?*

Input handling mirrors :mod:`stratum.cli.corridor`: real ERA5 reanalysis is
required (no synthetic fallback) and the geometry is either a great-circle
**route profile** (``--route VIDP-VABB``) or a flown ADS-B **track**
(``--trajectory CSV``). Unlike ``corridor``, the geometry is evaluated at
every requested flight level, so each waypoint gets a vertical risk profile.

Definitions
-----------
- The ambient ``pressure_pa`` of a requested flight level follows the ISA
  relation :func:`stratum.physics.thermo.fl_to_p`: ``fl_to_p(level) * 100``;
  this is the same conversion the feature assembler applies internally.
- ``p_persist`` is the calibrated probability of contrail persistence from
  the loaded model artifact. It is predicted only when ERA5 data is
  available at the sample.
- ``combined_risk`` is ``p_persist`` when ERA5 data is available *and* the
  Schmidt-Appleman criterion forms at the sample, and ``0.0`` otherwise:
  a level where no contrail can form carries no persistence risk, and a
  level without weather data is never scored as risky.
- ``risk_band`` maps ``combined_risk`` to ``"low"`` (``< 0.10``),
  ``"moderate"`` (``0.10 <= risk < 0.40``) and ``"high"`` (``>= 0.40``,
  the ``ABSTAIN_THRESHOLD`` of the decision layer).
- The recommended level per waypoint is the level with the lowest
  ``combined_risk`` among levels that carry ERA5 data *and* lie inside the
  **training envelope** recorded in the model artifact metadata (the
  pressure range the labeled training rows covered, written by
  ``stratum train``). When no such level exists, or the artifact records
  no envelope, the choice falls back to all levels with ERA5 data and a
  warning states that the recommendation lies outside the training
  envelope. Ties break towards the level nearest the baseline
  (``--baseline-fl``, default FL350), then towards the lower level. The
  runner-up is reported as well.
- ``training_envelope`` on every level record is ``True`` when the level's
  ambient pressure lies inside the recorded training envelope (inclusive
  bounds, :data:`_ENVELOPE_TOLERANCE_PA` tolerance), ``False`` outside it
  and ``None`` when the artifact records no envelope.
- The fuel delta of a recommendation is
  :func:`stratum.decide.fuel.estimate_fuel_penalty` against the baseline
  flight level over a 20-minute window. It is the extra fuel burned during
  that window after the level change, not a whole-flight fuel figure.

Requested levels outside ``MIN_ALLOWED_FL..MAX_ALLOWED_FL`` are rejected
with a clear :class:`ValueError`. The command exits with 0 on success, 2
for bad inputs (missing ERA5/model file, unknown airports, empty
trajectory, out-of-range levels, no waypoint with ERA5 data) and 1 when
the model artifact fails the feature-name sanity check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stratum.cli.corridor import (
    _DEFAULT_WAYPOINTS,
    _STEP_MINUTES,
    _fmt,
    _format_time,
    _json_sanitize,
    _load_trajectory,
    _parse_departure,
    _parse_levels,
    _parse_route,
    _time_axis,
)
from stratum.decide.counterfactual import (
    ABSTAIN_THRESHOLD,
    MAX_ALLOWED_FL,
    MIN_ALLOWED_FL,
)
from stratum.decide.fuel import estimate_fuel_penalty
from stratum.io.era5 import load_era5_dataset
from stratum.model.artifacts import load_model
from stratum.model.envelope import pressure_envelope_bounds
from stratum.model.features import FEATURE_NAMES
from stratum.physics.thermo import FT_TO_M, fl_to_p
from stratum.pipeline.build_features import (
    extract_features_for_point,
    pressure_to_altitude_m,
)
from stratum.routes.corridor import AIRPORTS, great_circle_waypoints

__all__ = ["profile", "risk_band"]

#: Upper bound of the ``"low"`` risk band (exclusive).
LOW_RISK_MAX: float = 0.10

#: Lower bound of the ``"high"`` risk band; the decision-layer abstain
#: threshold, reused so the band and the ATC gate cannot drift apart.
HIGH_RISK_MIN: float = ABSTAIN_THRESHOLD

#: Fuel penalty analysis window per level change [min].
_FUEL_WINDOW_MINUTES: float = 20.0

#: IST label offset next to UTC departure times [minutes].
_IST_OFFSET_MINUTES: int = 330

#: Metres per flight level (100 ft), used to report a pressure as a level.
_METERS_PER_FLIGHT_LEVEL: float = 100.0 * FT_TO_M

#: Absolute pressure tolerance [Pa] of the training-envelope comparison.
_ENVELOPE_TOLERANCE_PA: float = 1.0e-6

#: Flight-level quantum [hundreds of ft] used in envelope labels.
_FLIGHT_LEVEL_QUANTUM: int = 10


def _flight_level_of_pressure(pressure_pa: float) -> float:
    """ISA flight level [hundreds of ft] of an ambient pressure [Pa]."""
    return pressure_to_altitude_m(pressure_pa) / _METERS_PER_FLIGHT_LEVEL


def _rounded_flight_level(pressure_pa: float) -> int:
    """Flight level of a pressure, rounded to the nearest quantum [100 ft]."""
    level = _flight_level_of_pressure(pressure_pa) / _FLIGHT_LEVEL_QUANTUM
    return int(round(level)) * _FLIGHT_LEVEL_QUANTUM


def _envelope_bounds(envelope: Any) -> tuple[float, float] | None:
    """Return the ``(pressure_pa_min, pressure_pa_max)`` pair of an envelope.

    The envelope is the ``training_envelope`` mapping written by
    ``stratum train``. ``None`` is returned when it is missing, malformed,
    non-finite or inverted, so a hand-edited artifact degrades to "range
    unknown" instead of breaking the command. This is a thin wrapper around
    :func:`stratum.model.envelope.pressure_envelope_bounds`, shared with the
    corridor command.
    """
    return pressure_envelope_bounds(envelope)


def _within_envelope(
    pressure_pa: float, bounds: tuple[float, float] | None
) -> bool | None:
    """Whether ``pressure_pa`` lies inside the training envelope.

    Bounds are inclusive on both sides with an absolute
    :data:`_ENVELOPE_TOLERANCE_PA` tolerance. ``None`` means the envelope
    is unknown, so the level cannot be classified.
    """
    if bounds is None:
        return None
    low, high = bounds
    return bool(
        low - _ENVELOPE_TOLERANCE_PA
        <= pressure_pa
        <= high + _ENVELOPE_TOLERANCE_PA
    )


def _envelope_label(envelope: Any, bounds: tuple[float, float]) -> str:
    """Readable level range of a training envelope, e.g. ``FL330..FL370``."""
    levels: Any = None
    if isinstance(envelope, Mapping):
        levels = envelope.get("flight_levels")
    if isinstance(levels, (list, tuple)) and len(levels) > 0:
        try:
            numbers = sorted({int(level) for level in levels})
        except (TypeError, ValueError):
            numbers = []
        if numbers:
            return f"FL{numbers[0]}..FL{numbers[-1]}"
    low_pressure, high_pressure = bounds
    lower_level, upper_level = sorted(
        (
            _rounded_flight_level(low_pressure),
            _rounded_flight_level(high_pressure),
        )
    )
    return f"FL{lower_level}..FL{upper_level}"


def risk_band(combined_risk: float) -> str:
    """Bucket a combined persistence risk into low/moderate/high.

    Bands are ``"low"`` below :data:`LOW_RISK_MAX`, ``"moderate"`` up to
    :data:`HIGH_RISK_MIN` (exclusive) and ``"high"`` at or above it.
    """
    risk = float(combined_risk)
    if risk < LOW_RISK_MAX:
        return "low"
    if risk < HIGH_RISK_MIN:
        return "moderate"
    return "high"


def _validate_levels(levels: list[float]) -> None:
    """Reject flight levels outside the allowed cruise band.

    Raises
    ------
    ValueError
        When a level lies outside ``MIN_ALLOWED_FL..MAX_ALLOWED_FL``.
    """
    for level in levels:
        if level < MIN_ALLOWED_FL or level > MAX_ALLOWED_FL:
            raise ValueError(
                f"flight level FL{level:.0f} is outside the allowed cruise "
                f"band FL{MIN_ALLOWED_FL:.0f}..FL{MAX_ALLOWED_FL:.0f}; "
                "request levels inside that band"
            )


def _parse_point_region(value: str) -> tuple[float, float, float, float]:
    """Parse ``LAT_MIN,LON_MIN,LAT_MAX,LON_MAX`` into a bounding box.

    Raises
    ------
    ValueError
        For a malformed box or one whose minimum spans exceed the maxima.
    """
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4 or not all(parts):
        raise ValueError(
            f"--point-region {value!r} must be LAT_MIN,LON_MIN,LAT_MAX,LON_MAX"
        )
    try:
        south, west, north, east = (float(part) for part in parts)
    except ValueError as error:
        raise ValueError(
            f"--point-region {value!r} must contain four numbers"
        ) from error
    if not all(math.isfinite(number) for number in (south, west, north, east)):
        raise ValueError(f"--point-region {value!r} must contain finite numbers")
    if south >= north or west >= east:
        raise ValueError(
            f"--point-region {value!r} must satisfy LAT_MIN < LAT_MAX and "
            "LON_MIN < LON_MAX"
        )
    return south, west, north, east


def _region_contains(
    region: tuple[float, float, float, float], lat: float, lon: float
) -> bool:
    """Whether ``(lat, lon)`` lies inside the ``--point-region`` box."""
    south, west, north, east = region
    return south <= lat <= north and west <= lon <= east


def _route_waypoints(
    origin: str,
    destination: str,
    departure: pd.Timestamp,
    num_points: int,
    baseline_fl: float,
) -> list[dict[str, Any]]:
    """Waypoints of a great-circle route profile at the baseline level."""
    waypoints = great_circle_waypoints(
        AIRPORTS[origin],
        AIRPORTS[destination],
        num_points=num_points,
        fl=baseline_fl,
    )
    return [
        {
            "index": index,
            "time": departure + pd.Timedelta(minutes=_STEP_MINUTES * index),
            "lat": float(waypoint["lat"]),
            "lon": float(waypoint["lon"]),
        }
        for index, waypoint in enumerate(waypoints)
    ]


def _track_waypoints(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Waypoints of a cleaned ADS-B trajectory in frame order."""
    times = pd.to_datetime(frame["time"], utc=True)
    return [
        {
            "index": index,
            "time": times.iloc[index],
            "lat": float(frame["lat"].iloc[index]),
            "lon": float(frame["lon"].iloc[index]),
        }
        for index in range(len(frame))
    ]


def _level_record(
    model: Any,
    ds_met: Any,
    aircraft_type: str,
    waypoint: dict[str, Any],
    level: float,
    envelope: tuple[float, float] | None,
) -> dict[str, Any]:
    """Evaluate one flight level at one waypoint.

    ``p_persist`` is ``None`` when ERA5 data is unavailable; the combined
    risk is zero whenever the Schmidt-Appleman criterion does not form.
    ``training_envelope`` classifies the level pressure against the
    training envelope recorded in the model artifact (``None`` when the
    artifact carries none).
    """
    pressure_pa = fl_to_p(level) * 100.0
    vector, info = extract_features_for_point(
        float(waypoint["lat"]),
        float(waypoint["lon"]),
        float(pressure_pa),
        waypoint["time"],
        aircraft_type,
        ds_met=ds_met,
    )
    data_available = bool(info["data_available"])
    sac_forms = bool(info["sac_forms"])
    p_persist: float | None = None
    if data_available:
        probabilities = np.asarray(
            model.predict_proba(np.asarray(vector, dtype=float).reshape(1, -1)),
            dtype=float,
        )
        probability = float(probabilities[0, 1])
        if math.isfinite(probability):
            p_persist = min(max(probability, 0.0), 1.0)
    combined_risk = float(p_persist) if (p_persist is not None and sac_forms) else 0.0
    temperature_k = float(info["temperature_k"])
    return {
        "flight_level": float(level),
        "pressure_pa": float(pressure_pa),
        "data_available": data_available,
        "unavailable_reason": info["unavailable_reason"],
        "training_envelope": _within_envelope(float(pressure_pa), envelope),
        "temperature_k": temperature_k,
        "temperature_c": temperature_k - 273.15,
        "rhi": float(info["rhi"]),
        "sac_forms": sac_forms,
        "ambient_ice_supersaturated": bool(info["ambient_ice_supersaturated"]),
        "p_persist": p_persist,
        "combined_risk": combined_risk,
        "risk_band": risk_band(combined_risk),
    }


def _recommend(
    records: list[dict[str, Any]],
    baseline_fl: float,
    aircraft_type: str,
    envelope: tuple[float, float] | None,
) -> dict[str, Any] | None:
    """Pick the recommended level of one waypoint, or ``None`` without data.

    Candidates are the levels with ERA5 data. When the training envelope
    is known, preference goes to the candidates inside it; if none
    qualifies, every level with data is considered again and
    ``envelope_fallback`` is set so the caller can warn that the
    recommendation lies outside the training envelope. The ordering key is
    ``(combined_risk, distance to baseline, level)`` so ties break towards
    the baseline and then towards the lower level.
    """
    candidates = [record for record in records if record["data_available"]]
    if not candidates:
        return None
    inside = [
        record for record in candidates if record["training_envelope"] is True
    ]
    envelope_fallback = bool(envelope is not None and not inside)
    if inside:
        candidates = inside
    ordered = sorted(
        candidates,
        key=lambda record: (
            float(record["combined_risk"]),
            abs(float(record["flight_level"]) - baseline_fl),
            float(record["flight_level"]),
        ),
    )
    best = ordered[0]
    runner_up = ordered[1] if len(ordered) > 1 else None
    fuel = estimate_fuel_penalty(
        aircraft_type,
        baseline_fl,
        float(best["flight_level"]),
        duration_min=_FUEL_WINDOW_MINUTES,
    )
    return {
        "recommended_level": float(best["flight_level"]),
        "combined_risk": float(best["combined_risk"]),
        "risk_band": best["risk_band"],
        "training_envelope": best["training_envelope"],
        "envelope_fallback": envelope_fallback,
        "runner_up_level": (
            float(runner_up["flight_level"]) if runner_up is not None else None
        ),
        "runner_up_combined_risk": (
            float(runner_up["combined_risk"]) if runner_up is not None else None
        ),
        "fuel_delta": {
            "delta_fuel_kg": float(fuel["delta_fuel_kg"]),
            "delta_co2_kg": float(fuel["delta_co2_kg"]),
            "penalty_pct": float(fuel["penalty_pct"]),
            "baseline_fl": float(baseline_fl),
            "window_minutes": _FUEL_WINDOW_MINUTES,
        },
    }


def _print_table(
    waypoints: list[dict[str, Any]],
    recommendations: list[dict[str, Any] | None],
) -> None:
    """Print the per-waypoint, per-level risk table and recommendations."""
    header = (
        f"{'#':>3}  {'UTC':>5}  {'lat':>7}  {'lon':>7}  {'FL':>3}  "
        f"{'T[C]':>6}  {'RHi':>5}  {'SAC':>3}  {'p_persist':>9}  "
        f"{'combined':>8}  band"
    )
    print(header)
    for waypoint, recommendation in zip(waypoints, recommendations):
        for position, record in enumerate(waypoint["levels"]):
            if position == 0:
                lead = (
                    f"{int(waypoint['index']) + 1:>3}  "
                    f"{_format_time(str(waypoint['time_utc'])):>5}  "
                    f"{float(waypoint['lat']):>7.2f}  "
                    f"{float(waypoint['lon']):>7.2f}  "
                )
            else:
                lead = " " * 35
            p_persist = record["p_persist"]
            p_text = "n/a" if p_persist is None else _fmt(p_persist, ".3f")
            print(
                f"{lead}"
                f"{float(record['flight_level']):>3.0f}  "
                f"{_fmt(float(record['temperature_c']), '>6.1f'):>6}  "
                f"{_fmt(float(record['rhi']), '>5.2f'):>5}  "
                f"{'yes' if record['sac_forms'] else 'no':>3}  "
                f"{p_text:>9}  "
                f"{float(record['combined_risk']):>8.3f}  "
                f"{record['risk_band']}"
            )
        if recommendation is None:
            print("      -> no level with ERA5 data at this waypoint")
            continue
        summary = (
            f"      -> FL{float(recommendation['recommended_level']):.0f} | "
            f"combined risk {float(recommendation['combined_risk']):.3f} "
            f"({recommendation['risk_band']})"
        )
        if recommendation["runner_up_level"] is not None:
            summary += (
                f" | runner-up FL{float(recommendation['runner_up_level']):.0f} "
                f"{float(recommendation['runner_up_combined_risk']):.3f}"
            )
        else:
            summary += " | no runner-up"
        print(summary)
        fuel = recommendation["fuel_delta"]
        print(
            f"         fuel delta vs baseline "
            f"FL{float(fuel['baseline_fl']):.0f} over "
            f"{_FUEL_WINDOW_MINUTES:.0f} min: "
            f"+{float(fuel['delta_fuel_kg']):.2f} kg, "
            f"+{float(fuel['delta_co2_kg']):.2f} kg CO2"
        )
        if recommendation["envelope_fallback"]:
            print(
                "         WARNING: no requested level here lies inside the "
                "training envelope; the recommended FL"
                f"{float(recommendation['recommended_level']):.0f} is outside "
                "the training envelope."
            )


def _print_summary(
    waypoints: list[dict[str, Any]],
    recommendations: list[dict[str, Any] | None],
    baseline_fl: float,
) -> None:
    """Print the route-level recommendation summary."""
    with_data = sum(
        1
        for waypoint in waypoints
        if any(record["data_available"] for record in waypoint["levels"])
    )
    counts: dict[float, int] = {}
    total_fuel = 0.0
    total_co2 = 0.0
    for recommendation in recommendations:
        if recommendation is None:
            continue
        level = float(recommendation["recommended_level"])
        counts[level] = counts.get(level, 0) + 1
        total_fuel += float(recommendation["fuel_delta"]["delta_fuel_kg"])
        total_co2 += float(recommendation["fuel_delta"]["delta_co2_kg"])
    print()
    print("Recommendation summary")
    print("-" * 78)
    print(
        f"  Waypoints: {len(waypoints)} total | {with_data} with ERA5 data at "
        "at least one level"
    )
    if counts:
        distribution = " | ".join(
            f"FL{level:.0f} x{count}"
            for level, count in sorted(counts.items())
        )
        print(f"  Recommended levels: {distribution}")
        print(
            f"  Fuel delta vs baseline FL{baseline_fl:.0f}: "
            f"+{total_fuel:.2f} kg, +{total_co2:.2f} kg CO2 "
            f"({_FUEL_WINDOW_MINUTES:.0f}-minute window per waypoint)"
        )
    else:
        print("  Recommended levels: none (no waypoint carries ERA5 data)")


def _write_csv(path: Path, waypoints: list[dict[str, Any]]) -> None:
    """Write one CSV row per (waypoint, level) pair."""
    fieldnames = [
        "waypoint_index",
        "time_utc",
        "lat",
        "lon",
        "flight_level",
        "pressure_pa",
        "temperature_c",
        "rhi",
        "sac_forms",
        "data_available",
        "unavailable_reason",
        "training_envelope",
        "p_persist",
        "combined_risk",
        "risk_band",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for waypoint in waypoints:
            for record in waypoint["levels"]:
                writer.writerow(
                    {
                        "waypoint_index": int(waypoint["index"]) + 1,
                        "time_utc": waypoint["time_utc"],
                        "lat": float(waypoint["lat"]),
                        "lon": float(waypoint["lon"]),
                        "flight_level": float(record["flight_level"]),
                        "pressure_pa": float(record["pressure_pa"]),
                        "temperature_c": float(record["temperature_c"]),
                        "rhi": float(record["rhi"]),
                        "sac_forms": bool(record["sac_forms"]),
                        "data_available": bool(record["data_available"]),
                        "unavailable_reason": record["unavailable_reason"] or "",
                        "training_envelope": (
                            ""
                            if record["training_envelope"] is None
                            else bool(record["training_envelope"])
                        ),
                        "p_persist": (
                            "" if record["p_persist"] is None
                            else float(record["p_persist"])
                        ),
                        "combined_risk": float(record["combined_risk"]),
                        "risk_band": record["risk_band"],
                    }
                )


def profile(argv: list[str]) -> int:
    """Build the vertical contrail-risk profile of a route or flight track.

    Parameters
    ----------
    argv : list of str
        Arguments after ``profile``: ``--era5``, ``--model``, ``--levels``
        and one of ``--route`` / ``--trajectory`` are required.

    Returns
    -------
    int
        0 on success, 2 for bad inputs (missing ERA5/model file, unknown
        airports, empty trajectory, out-of-range levels, no waypoint with
        ERA5 data), 1 when the model artifact fails the feature-name
        sanity check.

    Notes
    -----
    The model artifact metadata may carry a ``training_envelope`` written
    by ``stratum train``. When present, every level record is flagged
    against it, one warning line lists requested levels outside it, and
    the recommendation prefers levels inside it; without it (or when no
    level with data lies inside it) the recommendation falls back to all
    levels with data and states that it lies outside the training
    envelope.
    """
    parser = argparse.ArgumentParser(
        prog="stratum profile",
        description=(
            "Evaluate calibrated contrail-persistence risk at every cruise "
            "level of a great-circle route profile or a flown ADS-B track, "
            "and recommend a cruise level per waypoint."
        ),
    )
    parser.add_argument(
        "--era5",
        required=True,
        type=Path,
        metavar="PATH",
        help="ERA5 NetCDF file with the canonical variables",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        metavar="MODEL.joblib",
        help="calibrated model artifact from `stratum train`",
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
        "--levels",
        required=True,
        metavar="FL[,FL...]",
        help="cruise levels to profile, e.g. 290,310,330,350,370,390",
    )
    parser.add_argument(
        "--baseline-fl",
        type=float,
        default=350.0,
        metavar="FL",
        help="filed cruise level used for tie-breaks and fuel deltas (default 350)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=_DEFAULT_WAYPOINTS,
        metavar="N",
        help=f"waypoints of the route profile (default {_DEFAULT_WAYPOINTS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="JSON",
        help="write the full profile dictionary to this JSON path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="CSV",
        help="write one CSV row per waypoint and level to this path",
    )
    parser.add_argument(
        "--point-region",
        default=None,
        metavar="LAT_MIN,LON_MIN,LAT_MAX,LON_MAX",
        help="keep only waypoints inside this bounding box",
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
    if not args.model.is_file():
        print(
            f"ERROR: model artifact not found: {args.model}\n"
            "Train one with `stratum train`; the vertical risk profile needs a "
            "calibrated persistence model.",
            file=sys.stderr,
        )
        return 2

    try:
        origin, destination = _parse_route(args.route) if args.route else ("", "")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        levels = _parse_levels(args.levels)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        _validate_levels(levels)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not math.isfinite(args.baseline_fl) or args.baseline_fl <= 0.0:
        print(
            f"ERROR: --baseline-fl must be positive, got {args.baseline_fl}",
            file=sys.stderr,
        )
        return 2
    if args.points < 2:
        print(f"ERROR: --points must be >= 2, got {args.points}", file=sys.stderr)
        return 2

    region: tuple[float, float, float, float] | None = None
    if args.point_region is not None:
        try:
            region = _parse_point_region(args.point_region)
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
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        first_time, last_time = _time_axis(ds_met)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    origin_label = ""
    destination_label = ""
    departure_assumed = False
    if args.route is not None:
        trajectory_source = "great_circle_profile"
        origin_label = origin
        destination_label = destination
        if departure is None:
            departure = first_time
            departure_assumed = True
        print(
            f"Trajectory: great-circle route profile {origin} -> {destination} "
            f"({args.points} waypoints at {_STEP_MINUTES:.0f}-minute spacing, "
            f"baseline FL{args.baseline_fl:.0f})."
        )
        print(
            "This is a route profile, NOT a real flight track; pass "
            "--trajectory with an ADS-B CSV for a flown track "
            "(trajectory_source=great_circle_profile recorded)."
        )
        waypoints = _route_waypoints(
            origin,
            destination,
            departure,
            args.points,
            args.baseline_fl,
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
    else:
        trajectory_source = "adsb_trajectory"
        try:
            frame = _load_trajectory(args.trajectory, args.callsign, args.icao24)
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        waypoints = _track_waypoints(frame)
        departure = waypoints[0]["time"]
        identifier = args.callsign or args.icao24 or "all flights"
        origin_label = f"ADS-B trajectory ({identifier})"
        print(
            f"Trajectory: cleaned ADS-B flight track ({identifier}), "
            f"{len(waypoints)} points, "
            f"{departure.isoformat()} .. {waypoints[-1]['time'].isoformat()}"
        )
        if args.departure is not None:
            print(
                "NOTE: --departure applies to great-circle route profiles; a "
                "flown trajectory keeps its own timestamps."
            )

    if region is not None:
        kept = [
            waypoint
            for waypoint in waypoints
            if _region_contains(region, waypoint["lat"], waypoint["lon"])
        ]
        print(
            f"--point-region kept {len(kept)} of {len(waypoints)} waypoints "
            f"(LAT_MIN={region[0]:.3f}, LON_MIN={region[1]:.3f}, "
            f"LAT_MAX={region[2]:.3f}, LON_MAX={region[3]:.3f})."
        )
        waypoints = kept
        if not waypoints:
            print(
                "ERROR: no waypoint lies inside --point-region; widen the box "
                "or drop the filter.",
                file=sys.stderr,
            )
            return 2

    ist_label = (departure + pd.Timedelta(minutes=_IST_OFFSET_MINUTES)).strftime(
        "%H:%M"
    )
    departure_label = (
        f"{departure.strftime('%Y-%m-%d %H:%M')} UTC ({ist_label} IST)"
    )
    print(
        "Departure: "
        + departure_label
        + (" [assumed: earliest ERA5 time step]" if departure_assumed else "")
    )
    print(f"Weather data: ECMWF ERA5 reanalysis ({args.era5.name})")

    try:
        model, model_metadata = load_model(args.model)
    except FileNotFoundError:
        print(f"ERROR: model artifact not found: {args.model}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    artifact_features = model_metadata.get("feature_names")
    if artifact_features is None:
        print(
            f"WARNING: model artifact {args.model} carries no feature_names; "
            "the frozen feature contract cannot be verified.",
            file=sys.stderr,
        )
    elif list(artifact_features) != list(FEATURE_NAMES):
        print(
            "ERROR: model artifact feature contract does not match this build "
            f"for {args.model}; expected the frozen {len(FEATURE_NAMES)}-feature "
            "order. Retrain with `stratum train`.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Model: {args.model} | aircraft {args.aircraft} | levels "
        + ", ".join(f"FL{level:.0f}" for level in levels)
    )

    envelope = model_metadata.get("training_envelope")
    envelope_bounds = _envelope_bounds(envelope)
    if envelope_bounds is None:
        print(
            "WARNING: the model artifact records no training envelope; the "
            "recommendation is not checked against it and may lie outside "
            "the training envelope."
        )

    warnings.filterwarnings(
        "ignore",
        message=(
            "X does not have valid feature names, but LGBMClassifier was "
            "fitted with feature names"
        ),
    )
    try:
        for waypoint in waypoints:
            waypoint["time_utc"] = waypoint["time"].isoformat()
            waypoint["levels"] = [
                _level_record(
                    model,
                    ds_met,
                    args.aircraft,
                    waypoint,
                    level,
                    envelope_bounds,
                )
                for level in levels
            ]
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    recommendations = [
        _recommend(
            waypoint["levels"],
            float(args.baseline_fl),
            args.aircraft,
            envelope_bounds,
        )
        for waypoint in waypoints
    ]
    if not any(
        record["data_available"]
        for waypoint in waypoints
        for record in waypoint["levels"]
    ):
        print(
            "ERROR: no waypoint carries ERA5 data at any requested level; "
            "check the route geometry, the profile time window and the ERA5 "
            "domain before retrying. No files were written.",
            file=sys.stderr,
        )
        return 2

    if envelope_bounds is not None:
        outside_levels = [
            level
            for level in levels
            if _within_envelope(fl_to_p(level) * 100.0, envelope_bounds)
            is False
        ]
        if outside_levels:
            print(
                "WARNING: requested level(s) "
                + ", ".join(f"FL{level:.0f}" for level in outside_levels)
                + " lie outside the training envelope "
                + _envelope_label(envelope, envelope_bounds)
                + "; their scores extrapolate beyond the training data."
            )

    print()
    _print_table(waypoints, recommendations)
    _print_summary(waypoints, recommendations, float(args.baseline_fl))

    metadata = {
        "command": "stratum profile",
        "trajectory_source": trajectory_source,
        "origin": origin,
        "destination": destination,
        "origin_label": origin_label,
        "destination_label": destination_label,
        "departure_utc": departure.isoformat(),
        "departure_assumed": departure_assumed,
        "aircraft_type": args.aircraft,
        "model_path": str(args.model),
        "model_metadata": model_metadata,
        "training_envelope": envelope,
        "baseline_fl": float(args.baseline_fl),
        "levels": [float(level) for level in levels],
        "num_waypoints": len(waypoints),
        "era5_filename": args.era5.name,
        "point_region": list(region) if region is not None else None,
        "risk_bands": {
            "low": f"< {LOW_RISK_MAX:.2f}",
            "moderate": f"{LOW_RISK_MAX:.2f} <= risk < {HIGH_RISK_MIN:.2f}",
            "high": f">= {HIGH_RISK_MIN:.2f} (ABSTAIN_THRESHOLD)",
        },
        "combined_risk_definition": (
            "p_persist when ERA5 data is available and the Schmidt-Appleman "
            "criterion forms at the sample; 0.0 otherwise"
        ),
        "fuel_delta_assumption": (
            "estimate_fuel_penalty against the baseline flight level over a "
            f"{_FUEL_WINDOW_MINUTES:.0f}-minute window after the level change"
        ),
    }
    payload = {
        "metadata": metadata,
        "levels": [float(level) for level in levels],
        "waypoints": [
            {
                "index": int(waypoint["index"]),
                "time_utc": waypoint["time_utc"],
                "lat": float(waypoint["lat"]),
                "lon": float(waypoint["lon"]),
                "levels": waypoint["levels"],
            }
            for waypoint in waypoints
        ],
        "recommendations": [
            {
                "waypoint_index": int(waypoint["index"]),
                "time_utc": waypoint["time_utc"],
                "lat": float(waypoint["lat"]),
                "lon": float(waypoint["lon"]),
                "baseline_fl": float(args.baseline_fl),
                "recommended_level": (
                    None if recommendation is None
                    else recommendation["recommended_level"]
                ),
                "combined_risk": (
                    None if recommendation is None
                    else recommendation["combined_risk"]
                ),
                "risk_band": (
                    None if recommendation is None else recommendation["risk_band"]
                ),
                "runner_up_level": (
                    None if recommendation is None
                    else recommendation["runner_up_level"]
                ),
                "runner_up_combined_risk": (
                    None if recommendation is None
                    else recommendation["runner_up_combined_risk"]
                ),
                "training_envelope": (
                    None if recommendation is None
                    else recommendation["training_envelope"]
                ),
                "envelope_fallback": (
                    None if recommendation is None
                    else recommendation["envelope_fallback"]
                ),
                "fuel_delta": (
                    None if recommendation is None
                    else recommendation["fuel_delta"]
                ),
            }
            for waypoint, recommendation in zip(waypoints, recommendations)
        ],
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_json_sanitize(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"JSON written: {args.out}")
    if args.csv is not None:
        _write_csv(args.csv, waypoints)
        print(f"CSV written: {args.csv}")
    return 0
