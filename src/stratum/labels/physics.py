"""Physics-based label builders from real ERA5 reanalysis (Step 5).

The physics label answers "will this waypoint hold an ice-supersaturated
contrail?" using only reanalysis state and the Schmidt-Appleman
criterion:

- ``1.0`` when the Schmidt-Appleman criterion (SAC) forms and the ambient
  air is ice-supersaturated (RHi > 1.0);
- ``0.0`` when the SAC forms but the ambient air is not supersaturated
  (RHi <= 1.0);
- ``nan`` when the SAC does not form, or when no ERA5 state backs the
  sample.

Non-forming waypoints are censored to ``nan`` on purpose: persistence is
only defined for a contrail that exists. A waypoint that cannot form a
contrail carries no information about persistence, so it must neither be
trained on as a negative nor silently become a zero; NaN keeps those rows
out of the loss while preserving them for audits.

Every value comes from an ERA5 lookup through
:func:`stratum.pipeline.build_features.extract_features_for_point`; this
module draws no random numbers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from stratum.model.features import FEATURE_NAMES
from stratum.physics import fl_to_p
from stratum.pipeline.build_features import extract_features_for_point
from stratum.routes.corridor import AIRPORTS, great_circle_waypoints

__all__ = [
    "PhysicsSample",
    "LABEL_FORMING_PERSISTENT",
    "LABEL_FORMING_SUBSATURATED",
    "CENSORED_SAC_NOT_FORMED",
    "CENSORED_DATA_UNAVAILABLE",
    "build_physics_labels",
    "sample_route_profiles",
    "write_dataset",
    "read_dataset",
]

#: Label for a SAC-forming, ice-supersaturated waypoint.
LABEL_FORMING_PERSISTENT: float = 1.0
#: Label for a SAC-forming waypoint in subsaturated air.
LABEL_FORMING_SUBSATURATED: float = 0.0

#: Censoring reason for a waypoint where the SAC does not form.
CENSORED_SAC_NOT_FORMED: str = "sac_not_formed"
#: Censoring reason for a waypoint without usable ERA5 data.
CENSORED_DATA_UNAVAILABLE: str = "data_unavailable"

#: Relative humidity over ice above which a contrail persists.
RHI_PERSISTENCE_THRESHOLD: float = 1.0

#: Provenance columns written next to the frozen feature block.
PROVENANCE_COLUMNS: tuple[str, ...] = ("lat", "lon", "pressure_pa", "time")

_LABEL_COLUMN: str = "label"


class PhysicsSample(NamedTuple):
    """One reanalysis lookup point ready for :func:`build_physics_labels`.

    Attributes
    ----------
    lat, lon : float
        Sample position [deg N, deg E].
    pressure_pa : float
        Ambient pressure [Pa].
    time : datetime-like
        Sample time; naive values are interpreted as UTC.
    """

    lat: float
    lon: float
    pressure_pa: float
    time: Any


def _unpack_sample(sample: Any) -> tuple[float, float, float, Any]:
    """Return ``(lat, lon, pressure_pa, time)`` from a sample entry.

    Accepts :class:`PhysicsSample`, any 4-element sequence, or a mapping
    with ``lat``/``lon``/``pressure_pa``/``time`` keys.
    """
    if isinstance(sample, Mapping):
        try:
            return (
                float(sample["lat"]),
                float(sample["lon"]),
                float(sample["pressure_pa"]),
                sample["time"],
            )
        except KeyError as error:
            raise ValueError(
                f"sample mapping is missing key {error.args[0]!r}; expected "
                "lat, lon, pressure_pa and time"
            ) from error
    values = tuple(sample)
    if len(values) != 4:
        raise ValueError(
            f"sample must carry (lat, lon, pressure_pa, time), got {sample!r}"
        )
    lat, lon, pressure_pa, when = values
    return float(lat), float(lon), float(pressure_pa), when


def build_physics_labels(
    ds_met: xr.Dataset,
    samples: Iterable[Any],
    aircraft_type: str = "A320",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Label reanalysis samples with the SAC + supersaturation rule.

    Parameters
    ----------
    ds_met : xarray.Dataset
        ERA5 dataset with the canonical variables.
    samples : iterable
        One entry per sample: :class:`PhysicsSample`, a 4-element
        ``(lat_deg, lon_deg, pressure_pa, datetime)`` sequence, or a
        mapping with those keys. :func:`sample_route_profiles` produces
        ready-made entries.
    aircraft_type : str, optional
        Engine designator used for the SAC thresholds, by default
        ``"A320"``.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, list of dict)
        - feature matrix ``X`` of shape ``(N, len(FEATURE_NAMES))`` in
          frozen order, ``nan`` rows where ERA5 data is unavailable;
        - label vector ``y`` in ``{0.0, 1.0, nan}`` as documented in the
          module docstring;
        - one metadata dictionary per sample: the per-point diagnostics of
          :func:`stratum.pipeline.build_features.extract_features_for_point`
          plus ``label`` and ``censored_reason`` (``None`` for accepted
          rows).

    Raises
    ------
    ValueError
        If a sample cannot be unpacked or carries non-finite geometry.
    """
    vectors: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []

    for sample in samples:
        lat, lon, pressure_pa, when = _unpack_sample(sample)
        vector, info = extract_features_for_point(
            lat,
            lon,
            pressure_pa,
            when,
            aircraft_type,
            ds_met=ds_met,
        )

        censored_reason: str | None = None
        if not info["data_available"]:
            label = float("nan")
            censored_reason = CENSORED_DATA_UNAVAILABLE
        elif not info["sac_forms"]:
            # No contrail forms here, so persistence is undefined. The row
            # is censored (NaN) instead of labeled 0.0: a non-forming
            # waypoint says nothing about whether a contrail would
            # persist, and training it as a negative would teach the
            # model about formation instead of persistence.
            label = float("nan")
            censored_reason = CENSORED_SAC_NOT_FORMED
        elif info["rhi"] > RHI_PERSISTENCE_THRESHOLD:
            label = LABEL_FORMING_PERSISTENT
        else:
            label = LABEL_FORMING_SUBSATURATED

        vectors.append(np.asarray(vector, dtype=float))
        labels.append(label)
        entry = dict(info)
        entry["label"] = label
        entry["censored_reason"] = censored_reason
        metadata.append(entry)

    if vectors:
        features = np.vstack(vectors)
    else:
        features = np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return features, np.asarray(labels, dtype=float), metadata


def _dataset_times(ds_met: xr.Dataset) -> np.ndarray:
    """Return the ERA5 time axis as ``datetime64[ns]`` values.

    Raises
    ------
    ValueError
        When the dataset has neither a ``time`` nor a ``valid_time``
        coordinate.
    """
    for name in ("time", "valid_time"):
        if name in ds_met.coords or name in ds_met.dims:
            return np.asarray(ds_met[name].values).astype("datetime64[ns]")
    raise ValueError("ERA5 dataset is missing the time coordinate")


def _resolve_sample_time(entry: Any, available: np.ndarray) -> pd.Timestamp:
    """Resolve one ``hours`` entry to a timestamp present in the dataset.

    Integer entries are indices into the dataset time axis; datetime-like
    entries must match an available timestamp exactly.
    """
    if isinstance(entry, (int, np.integer)) and not isinstance(entry, bool):
        index = int(entry)
        if not 0 <= index < available.size:
            raise ValueError(
                f"hour index {index} is outside the dataset time axis of "
                f"{available.size} step(s)"
            )
        return pd.Timestamp(available[index]).tz_localize("UTC")
    requested = pd.Timestamp(entry)
    if pd.isna(requested):
        raise ValueError(f"hours entry {entry!r} is not a valid time index")
    if requested.tzinfo is not None:
        requested = requested.tz_convert("UTC").tz_localize(None)
    matches = np.flatnonzero(available == np.datetime64(requested, "ns"))
    if matches.size == 0:
        raise ValueError(
            f"requested time {requested.isoformat()} is not present in the ERA5 "
            f"dataset; available times start at "
            f"{pd.Timestamp(available[0]).isoformat()}"
        )
    return pd.Timestamp(available[int(matches[0])]).tz_localize("UTC")


def _route_endpoints(route: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resolve one route entry to two ``(lat, lon)`` coordinate pairs.

    Accepts a pair of ICAO location indicators (resolved through
    :data:`stratum.routes.corridor.AIRPORTS`) or a pair of coordinate
    pairs.
    """
    origin, destination = route
    if isinstance(origin, str) or isinstance(destination, str):
        try:
            return AIRPORTS[str(origin).upper()], AIRPORTS[str(destination).upper()]
        except KeyError as error:
            raise ValueError(
                f"unknown ICAO location indicator {error.args[0]!r} in route "
                f"{route!r}"
            ) from error
    return (
        (float(origin[0]), float(origin[1])),
        (float(destination[0]), float(destination[1])),
    )


def sample_route_profiles(
    ds_met: xr.Dataset,
    routes: Iterable[Any],
    flight_levels: Sequence[float],
    hours: Sequence[Any],
    points_per_route: int = 25,
    aircraft_type: str = "A320",
) -> list[PhysicsSample]:
    """Sample real ERA5 points along great-circle route profiles.

    For every ``(route, flight level, time)`` combination the function
    interpolates ``points_per_route`` great-circle waypoints with
    :func:`stratum.routes.corridor.great_circle_waypoints` and pairs each
    waypoint with the ISA pressure of its flight level and a timestamp
    that really exists on the ERA5 time axis. Nothing is synthesized: the
    values are produced later by
    :func:`build_physics_labels`, which only performs ERA5 lookups.

    Parameters
    ----------
    ds_met : xarray.Dataset
        ERA5 dataset; its ``time`` (or ``valid_time``) coordinate
        provides the real sample times.
    routes : iterable
        Route entries as pairs of ICAO location indicators (e.g.
        ``("VIDP", "VABB")``) or pairs of ``(lat, lon)`` coordinates.
    flight_levels : sequence of float
        Flight levels to sample [hundreds of feet].
    hours : sequence
        Integer indices into the dataset time axis, or datetime-like
        values that must match an available timestamp exactly.
    points_per_route : int, optional
        Waypoints per route, by default 25.
    aircraft_type : str, optional
        Accepted for symmetry with :func:`build_physics_labels`; the
        sampling itself is aircraft independent.

    Returns
    -------
    list of PhysicsSample
        Samples in ``(route, flight level, time)`` order, ready to pass
        to :func:`build_physics_labels`.

    Raises
    ------
    ValueError
        For an unknown airport, an out-of-range hour index, a requested
        time absent from the dataset, or too few points per route.
    """
    available = _dataset_times(ds_met)
    if available.size == 0:
        raise ValueError("ERA5 dataset contains no time steps to sample")
    resolved_times = [_resolve_sample_time(entry, available) for entry in hours]

    samples: list[PhysicsSample] = []
    for route in routes:
        origin, destination = _route_endpoints(route)
        for flight_level in flight_levels:
            waypoints = great_circle_waypoints(
                origin,
                destination,
                num_points=points_per_route,
                fl=float(flight_level),
            )
            for timestamp in resolved_times:
                for waypoint in waypoints:
                    samples.append(
                        PhysicsSample(
                            lat=float(waypoint["lat"]),
                            lon=float(waypoint["lon"]),
                            pressure_pa=float(waypoint["p_Pa"]),
                            time=timestamp,
                        )
                    )
    return samples


def _provenance(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the provenance columns of one metadata entry.

    The sample time is taken from ``time`` when present and from
    ``time_utc`` otherwise, so entries produced directly by
    :func:`build_physics_labels` can be written without adaptation.
    """
    missing = [name for name in PROVENANCE_COLUMNS if name not in entry]
    if missing == ["time"] and "time_utc" in entry:
        missing = []
    if missing:
        raise ValueError(f"metadata entry is missing provenance keys: {missing}")
    timestamp = entry["time"] if "time" in entry else entry["time_utc"]
    return {
        "lat": float(entry["lat"]),
        "lon": float(entry["lon"]),
        "pressure_pa": float(entry["pressure_pa"]),
        "time": str(timestamp),
    }


def write_dataset(
    path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Mapping[str, Any]],
) -> None:
    """Write a labeled dataset to CSV (features + label + provenance).

    The feature columns are the frozen :data:`FEATURE_NAMES` order, the
    label column is ``label``, and the provenance columns are ``lat``,
    ``lon``, ``pressure_pa`` and ``time`` taken from ``meta``.

    Parameters
    ----------
    path : str or pathlib.Path
        Output CSV path.
    X : numpy.ndarray
        Feature matrix of shape ``(N, len(FEATURE_NAMES))``.
    y : numpy.ndarray
        Label vector of shape ``(N,)``; ``nan`` marks censored rows.
    meta : sequence of mapping
        Metadata entries carrying the provenance keys.

    Raises
    ------
    ValueError
        For mismatched shapes or metadata missing provenance keys.
    """
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"X must have shape (N, {len(FEATURE_NAMES)}), got {features.shape!r}"
        )
    if labels.shape != (features.shape[0],):
        raise ValueError(
            f"y must have shape ({features.shape[0]},), got {labels.shape!r}"
        )
    if len(meta) != features.shape[0]:
        raise ValueError(
            f"meta has {len(meta)} entries for {features.shape[0]} rows"
        )

    frame = pd.DataFrame(features, columns=list(FEATURE_NAMES))
    frame[_LABEL_COLUMN] = labels
    provenance = [_provenance(entry) for entry in meta]
    for column in PROVENANCE_COLUMNS:
        frame[column] = [entry[column] for entry in provenance]
    frame.to_csv(path, index=False)


def read_dataset(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Read a CSV written by :func:`write_dataset`.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV path.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, list of dict)
        Feature matrix in frozen order, label vector (``nan`` preserved)
        and one provenance dictionary per row.

    Raises
    ------
    ValueError
        When the file misses feature, label or provenance columns.
    """
    frame = pd.read_csv(path)
    missing = [
        name
        for name in (*FEATURE_NAMES, _LABEL_COLUMN, *PROVENANCE_COLUMNS)
        if name not in frame.columns
    ]
    if missing:
        raise ValueError(f"dataset {path} is missing columns: {missing}")

    features = frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    labels = frame[_LABEL_COLUMN].to_numpy(dtype=float)
    metadata: list[dict[str, Any]] = []
    for record in frame.loc[:, list(PROVENANCE_COLUMNS)].to_dict("records"):
        entry = {
            "lat": float(record["lat"]),
            "lon": float(record["lon"]),
            "pressure_pa": float(record["pressure_pa"]),
            "time": str(record["time"]),
        }
        if math.isnan(entry["lat"]) or math.isnan(entry["pressure_pa"]):
            raise ValueError(f"dataset {path} contains non-finite provenance")
        metadata.append(entry)
    return features, labels, metadata
