"""Label-building subcommands for the STRATUM command line.

Two datasets can be produced from real inputs only:

- ``build-physics-labels`` samples great-circle route profiles inside a
  real ERA5 file and labels every waypoint with the Schmidt-Appleman
  criterion plus ice supersaturation (the Step 5 protocol in
  :mod:`stratum.labels.physics`);
- ``build-observed-labels`` labels ADS-B trajectory segments against a
  real satellite scene and the same ERA5 state (the Step 6 protocol in
  :mod:`stratum.labels.observed`).

Both commands refuse to run without their inputs: a missing ERA5 file, an
unknown route, or a scene without channels exits with code 2 instead of
substituting synthetic meteorology or a synthetic scene.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stratum.cli.scene import _parse_bbox, load_scene
from stratum.io.adsb import load_adsb_csv, select_trajectory
from stratum.io.era5 import load_era5_dataset
from stratum.labels import observed as observed_labels
from stratum.labels import physics as physics_labels
from stratum.routes.corridor import AIRPORTS

__all__ = ["build_observed_labels", "build_physics_labels"]


def _parse_routes(value: str) -> list[tuple[str, str]]:
    """Parse ``VIDP-VABB,VABB-VOBL`` into ICAO ``(origin, destination)`` pairs.

    Raises
    ------
    ValueError
        For empty entries, malformed pairs, or ICAO codes absent from
        :data:`stratum.routes.corridor.AIRPORTS`.
    """
    routes: list[tuple[str, str]] = []
    for chunk in value.split(","):
        text = chunk.strip()
        if not text:
            raise ValueError(f"empty route entry in --routes value {value!r}")
        parts = [part.strip().upper() for part in text.split("-")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                f"route {text!r} must be an ORIGIN-DESTINATION ICAO pair "
                "(for example VIDP-VABB)"
            )
        for code in parts:
            if code not in AIRPORTS:
                known = ", ".join(sorted(AIRPORTS))
                raise ValueError(
                    f"unknown ICAO location indicator {code!r} in route "
                    f"{text!r}; known airports: {known}"
                )
        routes.append((parts[0], parts[1]))
    if not routes:
        raise ValueError("--routes must contain at least one route")
    return routes


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


def _parse_hours(value: str) -> list[int]:
    """Parse ``0-5,18-23`` into de-duplicated time-axis indices.

    Every comma-separated entry is either one integer index or an
    inclusive ``A-B`` range of integer indices into the ERA5 time axis.
    """
    indices: list[int] = []
    for chunk in value.split(","):
        text = chunk.strip()
        if not text:
            raise ValueError(f"empty entry in --hours value {value!r}")
        bounds = text.split("-")
        if len(bounds) not in (1, 2):
            raise ValueError(
                f"--hours entry {text!r} must be an integer index or an "
                "inclusive A-B range"
            )
        try:
            numbers = [int(part) for part in bounds]
        except ValueError as error:
            raise ValueError(
                f"--hours entry {text!r} must be an integer index or an "
                "inclusive A-B range (for example 0-5,18-23)"
            ) from error
        if len(numbers) == 1:
            indices.append(numbers[0])
        else:
            start, stop = numbers
            if stop < start:
                raise ValueError(
                    f"--hours range {text!r} is decreasing; use A-B with A <= B"
                )
            indices.extend(range(start, stop + 1))
    if not indices:
        raise ValueError("--hours must contain at least one time index")
    if any(index < 0 for index in indices):
        raise ValueError("--hours indices must be >= 0")
    # Preserve the requested order while sampling each time step once.
    return list(dict.fromkeys(indices))


def _dataset_time_count(ds_met: Any) -> int:
    """Return the number of ERA5 time steps.

    Raises
    ------
    ValueError
        When the dataset has neither a ``time`` nor a ``valid_time``
        coordinate.
    """
    for name in ("time", "valid_time"):
        if name in ds_met.coords or name in ds_met.dims:
            return int(np.asarray(ds_met[name].values).size)
    raise ValueError("ERA5 dataset is missing the time coordinate")


def _label_counts(labels: np.ndarray) -> tuple[int, int, int]:
    """Return ``(persistent, subsaturated, censored)`` label counts."""
    values = np.asarray(labels, dtype=float)
    persistent = int(np.sum(values == physics_labels.LABEL_FORMING_PERSISTENT))
    subsaturated = int(np.sum(values == physics_labels.LABEL_FORMING_SUBSATURATED))
    censored = int(np.isnan(values).sum())
    return persistent, subsaturated, censored


def _observed_time_range(metadata: list[dict[str, object]]) -> tuple[str, str] | None:
    """First and last ``time_utc`` of the labeled segments, when present."""
    stamps = [
        pd.Timestamp(str(entry["time_utc"]))
        for entry in metadata
        if entry.get("time_utc")
    ]
    if not stamps:
        return None
    return min(stamps).isoformat(), max(stamps).isoformat()


def build_physics_labels(argv: list[str]) -> int:
    """Build a physics-labeled dataset from a real ERA5 file.

    Parameters
    ----------
    argv : list of str
        Arguments after ``build-physics-labels``: ``--era5``, ``--routes``,
        ``--levels`` and ``--out`` are required; ``--aircraft``,
        ``--points``, ``--hours`` and ``--max-times`` are optional.

    Returns
    -------
    int
        0 on success, 2 for a missing or invalid ERA5 file, an unknown
        route, or malformed sampling options.
    """
    parser = argparse.ArgumentParser(
        prog="stratum build-physics-labels",
        description=(
            "Sample great-circle route profiles in a real ERA5 file and "
            "label every waypoint with the Schmidt-Appleman criterion plus "
            "ice supersaturation. No synthetic meteorology is allowed."
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
        "--routes",
        required=True,
        metavar="A-B[,C-D...]",
        help="route pairs of ICAO location indicators, e.g. VIDP-VABB,VABB-VOBL",
    )
    parser.add_argument(
        "--levels",
        required=True,
        metavar="FL[,FL...]",
        help="flight levels to sample [hundreds of feet], e.g. 330,350,370",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="CSV",
        help="output CSV with features, label and provenance columns",
    )
    parser.add_argument(
        "--aircraft",
        default="A320",
        help="engine designator for the SAC thresholds (default A320)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=25,
        metavar="N",
        help="waypoints per route (default 25)",
    )
    parser.add_argument(
        "--hours",
        default=None,
        metavar="0-5,18-23",
        help=(
            "time-axis indices to sample as single indices or inclusive "
            "A-B ranges (default: every ERA5 time step)"
        ),
    )
    parser.add_argument(
        "--max-times",
        type=int,
        default=None,
        metavar="N",
        help="cap the number of sampled time steps after --hours selection",
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
    if args.points < 2:
        print(
            f"ERROR: --points must be >= 2, got {args.points}",
            file=sys.stderr,
        )
        return 2
    if args.max_times is not None and args.max_times < 1:
        print(
            f"ERROR: --max-times must be >= 1, got {args.max_times}",
            file=sys.stderr,
        )
        return 2

    try:
        routes = _parse_routes(args.routes)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        levels = _parse_levels(args.levels)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        requested_hours = None if args.hours is None else _parse_hours(args.hours)
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
        n_times = _dataset_time_count(ds_met)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if requested_hours is None:
        selected_hours: list[int] = list(range(n_times))
    else:
        out_of_range = [index for index in requested_hours if index >= n_times]
        if out_of_range:
            print(
                f"ERROR: --hours indices {out_of_range} are outside the ERA5 "
                f"time axis of {n_times} step(s)",
                file=sys.stderr,
            )
            return 2
        selected_hours = requested_hours

    if args.max_times is not None and len(selected_hours) > args.max_times:
        print(
            f"NOTE: --max-times caps the sample to the first {args.max_times} "
            f"of {len(selected_hours)} selected time step(s)."
        )
        selected_hours = selected_hours[: args.max_times]

    try:
        samples = physics_labels.sample_route_profiles(
            ds_met,
            routes,
            levels,
            selected_hours,
            points_per_route=args.points,
            aircraft_type=args.aircraft,
        )
        features, labels, metadata = physics_labels.build_physics_labels(
            ds_met, samples, aircraft_type=args.aircraft
        )
        physics_labels.write_dataset(args.out, features, labels, metadata)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    persistent, subsaturated, censored = _label_counts(labels)
    data_available = sum(1 for entry in metadata if entry["data_available"])
    censored_reasons: dict[str, int] = {}
    for entry in metadata:
        reason = entry.get("censored_reason")
        if reason is not None:
            censored_reasons[str(reason)] = censored_reasons.get(str(reason), 0) + 1

    route_text = ", ".join(f"{origin}-{destination}" for origin, destination in routes)
    level_text = ", ".join(f"FL{level:.0f}" for level in levels)
    print(f"Physics label dataset written: {args.out}")
    print(f"  Routes                 : {route_text}")
    print(f"  Flight levels          : {level_text}")
    print(f"  Time steps used        : {len(selected_hours)} of {n_times}")
    print(f"  Rows                   : {len(labels)}")
    print(f"  Label 1 (persistent)   : {persistent}")
    print(f"  Label 0 (subsaturated) : {subsaturated}")
    print(f"  Censored (NaN)         : {censored}")
    print(f"  ERA5 data available    : {data_available}")
    if censored_reasons:
        reasons_text = ", ".join(
            f"{reason}={count}" for reason, count in sorted(censored_reasons.items())
        )
        print(f"  Censoring reasons      : {reasons_text}")
    return 0


def build_observed_labels(argv: list[str]) -> int:
    """Build an observed-label dataset from a real scene and trajectory.

    Parameters
    ----------
    argv : list of str
        Arguments after ``build-observed-labels``: ``--satellite``,
        ``--trajectory``, ``--era5`` and ``--out`` are required; the
        identifier, detector and crop options are optional.

    Returns
    -------
    int
        0 on success, 2 for a missing input file, a scene without
        navigation or observation time, or an empty trajectory selection.
    """
    parser = argparse.ArgumentParser(
        prog="stratum build-observed-labels",
        description=(
            "Match detector output on a real satellite scene against ADS-B "
            "trajectory segments with ERA5 state and label each segment "
            "1.0 (matched), 0.0 (observable miss) or NaN (unobservable)."
        ),
    )
    parser.add_argument(
        "--satellite",
        required=True,
        type=Path,
        metavar="NC",
        help="NetCDF scene with the IR_108/IR_120 split-window pair",
    )
    parser.add_argument(
        "--trajectory",
        required=True,
        type=Path,
        metavar="CSV",
        help="OpenSky-style ADS-B CSV with the canonical columns",
    )
    parser.add_argument(
        "--era5",
        required=True,
        type=Path,
        metavar="PATH",
        help="ERA5 NetCDF file with the canonical variables",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="CSV",
        help="output CSV with features, label and provenance columns",
    )
    identifiers = parser.add_mutually_exclusive_group()
    identifiers.add_argument(
        "--callsign",
        default=None,
        help="select one flight by callsign (case-insensitive)",
    )
    identifiers.add_argument(
        "--icao24",
        default=None,
        help="select one flight by ICAO 24-bit transponder address",
    )
    parser.add_argument(
        "--aircraft",
        default="A320",
        help="engine designator for the feature assembly (default A320)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="detector response threshold [robust sigma] (default 2.0)",
    )
    parser.add_argument(
        "--min-length-px",
        type=int,
        default=8,
        help="minimum accepted detection length [px] (default 8)",
    )
    parser.add_argument(
        "--bbox",
        default=None,
        metavar="N,W,S,E",
        help="optional crop box in degrees before detection",
    )
    parser.add_argument(
        "--observation-time",
        default=None,
        metavar="ISO8601",
        help=(
            "scene observation time; required only when the scene file "
            "carries none"
        ),
    )
    args = parser.parse_args(argv)

    for label, path in (
        ("satellite scene", args.satellite),
        ("ADS-B trajectory", args.trajectory),
        ("ERA5 file", args.era5),
    ):
        if not path.is_file():
            print(
                f"ERROR: {label} not found: {path}\n"
                "No synthetic input is substituted; fetch or point at real data.",
                file=sys.stderr,
            )
            return 2

    bbox: tuple[float, float, float, float] | None = None
    if args.bbox is not None:
        try:
            bbox = _parse_bbox(args.bbox)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        scene = load_scene(args.satellite, bbox=bbox)
    except FileNotFoundError:
        print(f"ERROR: satellite scene not found: {args.satellite}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if scene.georef is None:
        print(
            "ERROR: the scene carries no navigation (neither "
            "goes_imager_projection nor latitude/longitude coordinates), so "
            "detections cannot be matched against trajectory segments.",
            file=sys.stderr,
        )
        return 2

    observation_time: object | None = args.observation_time
    if observation_time is None:
        observation_time = scene.observation_time
    if observation_time is None:
        print(
            "ERROR: the scene carries no observation time; pass "
            "--observation-time ISO8601.",
            file=sys.stderr,
        )
        return 2

    try:
        trajectories = load_adsb_csv(args.trajectory)
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if trajectories.empty:
        print(
            "ERROR: the trajectory file contains no usable airborne cruise "
            "points after cleaning.",
            file=sys.stderr,
        )
        return 2
    if args.callsign is not None or args.icao24 is not None:
        try:
            trajectories = select_trajectory(
                trajectories, callsign=args.callsign, icao24=args.icao24
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        if trajectories.empty:
            wanted = args.callsign if args.callsign is not None else args.icao24
            print(
                f"ERROR: no trajectory points matched the requested flight "
                f"{wanted!r}.",
                file=sys.stderr,
            )
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
        features, labels, metadata = observed_labels.build_observed_labels(
            bt_108=scene.bt_108,
            bt_120=scene.bt_120,
            georef=scene.georef,
            observation_time=observation_time,
            trajectories=trajectories,
            ds_met=ds_met,
            aircraft_type=args.aircraft,
            threshold=args.threshold,
            min_length_px=args.min_length_px,
        )
        physics_labels.write_dataset(args.out, features, labels, metadata)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    matched, misses, censored = _label_counts(labels)
    time_range = _observed_time_range(metadata)
    print(f"Observed label dataset written: {args.out}")
    print(f"  Satellite scene          : {args.satellite}")
    print(f"  Observation time         : {pd.Timestamp(observation_time).isoformat()}")
    print(f"  Georeferencing           : {scene.georeferencing}")
    print(f"  Trajectory segments      : {len(labels)}")
    print(f"  Label 1 (matched)        : {matched}")
    print(f"  Label 0 (observable miss): {misses}")
    print(f"  Censored (NaN)           : {censored}")
    if time_range is not None:
        print(f"  Segment time range       : {time_range[0]} .. {time_range[1]}")
    return 0
