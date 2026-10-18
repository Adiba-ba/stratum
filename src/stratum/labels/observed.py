"""Observed-label path: satellite contrail detections against ADS-B.

The observed label answers "did the satellite see a contrail line on this
flight segment?" for a concrete scene. Per consecutive trajectory point
pair:

1. the ERA5 state at the segment midpoint drives the shared feature
   assembly (:func:`stratum.pipeline.build_features.extract_features_for_point`);
2. the midpoint is advected to the observation time with the
   segment-mean horizontal wind and displaced by the Meteosat parallax
   of its altitude, so it is compared with the scene in image space;
3. the detector
   (:func:`stratum.detect.filters.detect_contrail_lines`) runs once on
   the scene; each detected object centroid is georeferenced and its
   image-space angle is converted to a compass heading;
4. the segment matches a detection when both the centroid distance and
   the orientation agree with
   :func:`stratum.match.label.match_segment_to_detections`.

The tri-state rule of :func:`stratum.match.label.assign_tristate_label`
turns a match into ``1.0``. A miss becomes ``0.0`` only when the
trajectory was observable: ``observable_slots`` is the number of segments
of that trajectory whose midpoint passes the observability gate
(:func:`stratum.match.observe.is_scene_observable` with the ERA5 high
cloud fraction and the satellite viewing zenith angle) at the observation
time. With fewer than two observable segments the scene provides no
statement, so the label stays ``nan``.

The module performs no file IO except the optional
:func:`load_bt_pair` helper, which the CLI can use to feed a scene into
:func:`build_observed_labels`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from stratum.detect.filters import detect_contrail_lines
from stratum.geo.angles import pixel_angle_to_heading_deg, initial_bearing_deg
from stratum.io.era5 import interpolate_met_to_segment
from stratum.match.advect import advect_segment
from stratum.match.label import assign_tristate_label, match_segment_to_detections
from stratum.match.observe import is_scene_observable
from stratum.match.parallax import (
    METEOSAT9_LATITUDE_DEG,
    METEOSAT9_LONGITUDE_DEG,
    calculate_parallax_displacement,
)
from stratum.model.features import FEATURE_NAMES
from stratum.physics import fl_to_p
from stratum.pipeline.build_features import (
    extract_features_for_point,
    pressure_to_altitude_m,
)

__all__ = [
    "SatelliteScene",
    "EARTH_RADIUS_KM",
    "GEOSTATIONARY_ALTITUDE_KM",
    "load_bt_pair",
    "viewing_zenith_angle_deg",
    "build_observed_labels",
]

#: Spherical Earth radius used by the viewing-zenith approximation [km].
EARTH_RADIUS_KM: float = 6371.0
#: Geostationary altitude above the surface [km].
GEOSTATIONARY_ALTITUDE_KM: float = 35786.0

_REQUIRED_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "time",
    "lat",
    "lon",
    "baro_alt_ft",
)

#: Longitude of the Meteosat sub-satellite point used as the default [deg E].
_DEFAULT_SUBSATELLITE_LON_DEG: float = METEOSAT9_LONGITUDE_DEG

#: Reason recorded when a segment is newer than the observation.
_REASON_AFTER_OBSERVATION: str = "after_observation"


@dataclass(frozen=True)
class SatelliteScene:
    """A satellite scene the CLI can load on demand.

    Attributes
    ----------
    path : str
        Path to the NetCDF scene (see :func:`load_bt_pair` for the
        expected layout).
    time : datetime-like
        Observation time of the scene; naive values are interpreted as
        UTC.
    """

    path: str
    time: Any


def load_bt_pair(
    path: str | Path,
    channel_108: str = "IR_108",
    channel_120: str = "IR_120",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a split-window brightness-temperature pair from a NetCDF file.

    Expected layout: a NetCDF file with two two-dimensional variables,
    named ``IR_108`` and ``IR_120`` by default, holding the 10.8 um and
    12.0 um brightness temperatures [K] on the image grid
    (``array[row, col]``, row increasing southward). The file carries no
    georeferencing contract here: the caller supplies the matching
    :class:`stratum.geo.angles.ArrayGeoreferencer`.

    Parameters
    ----------
    path : str or pathlib.Path
        NetCDF file path.
    channel_108, channel_120 : str, optional
        Variable names of the two channels.

    Returns
    -------
    tuple of numpy.ndarray
        ``(bt_108, bt_120)`` as float arrays of identical shape.

    Raises
    ------
    ValueError
        When a channel is missing or is not a 2-D array.
    """
    with xr.open_dataset(path) as dataset:
        arrays: list[np.ndarray] = []
        for name in (channel_108, channel_120):
            if name not in dataset.data_vars:
                raise ValueError(
                    f"scene {path} is missing channel {name!r}; available "
                    f"variables: {sorted(dataset.data_vars)}"
                )
            values = np.asarray(dataset[name].values, dtype=float)
            if values.ndim != 2:
                raise ValueError(
                    f"channel {name!r} in {path} must be 2-D, got shape "
                    f"{values.shape!r}"
                )
            arrays.append(values.copy())
    bt_108, bt_120 = arrays
    if bt_108.shape != bt_120.shape:
        raise ValueError(
            f"channel shapes differ: {bt_108.shape!r} vs {bt_120.shape!r}"
        )
    return bt_108, bt_120


def viewing_zenith_angle_deg(
    lat: float,
    lon: float,
    subsatellite_lon_deg: float,
    subsatellite_lat_deg: float = 0.0,
) -> float:
    """Satellite viewing zenith angle of a surface target [deg].

    The angular distance ``s`` between the target and the sub-satellite
    point follows from the spherical law of cosines; the viewing zenith
    angle is then the exact geostationary geometry

    ``VZA = atan2((r + h) sin(s), (r + h) cos(s) - r)``

    with ``r = EARTH_RADIUS_KM`` (6371 km) and
    ``h_geo = GEOSTATIONARY_ALTITUDE_KM`` (35786 km). The expression is
    the on-disk geometry of a spherical Earth, used to drive the
    observability gate, not to do navigation. A target beyond the
    geostationary limb (``s >= acos(r / (r + h))``) has no line of sight
    and returns ``nan``.

    Parameters
    ----------
    lat, lon : float
        Target position [deg N, deg E].
    subsatellite_lon_deg : float
        Sub-satellite longitude [deg E].
    subsatellite_lat_deg : float, optional
        Sub-satellite latitude [deg N], by default 0.0.

    Returns
    -------
    float
        Viewing zenith angle [deg], or ``nan`` for a non-finite target or
        a target beyond the limb.
    """
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return math.nan
    lat_rad = math.radians(lat)
    subsatellite_lat_rad = math.radians(subsatellite_lat_deg)
    delta_lon_rad = math.radians(lon - subsatellite_lon_deg)
    cos_s = math.sin(lat_rad) * math.sin(subsatellite_lat_rad) + math.cos(
        lat_rad
    ) * math.cos(subsatellite_lat_rad) * math.cos(delta_lon_rad)
    cos_s = min(1.0, max(-1.0, cos_s))
    s = math.acos(cos_s)
    cos_limb = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + GEOSTATIONARY_ALTITUDE_KM)
    if cos_s < cos_limb:
        return math.nan
    r = EARTH_RADIUS_KM
    h = GEOSTATIONARY_ALTITUDE_KM
    return math.degrees(
        math.atan2((r + h) * math.sin(s), (r + h) * math.cos(s) - r)
    )


def _coerce_utc(when: Any) -> pd.Timestamp:
    """Return ``when`` as a UTC pandas timestamp (naive means UTC)."""
    timestamp = pd.Timestamp(when)
    if pd.isna(timestamp):
        raise ValueError(f"expected a valid datetime, got {when!r}")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _resolve_subsatellite_lon(georef: Any, override: float | None) -> float:
    """Longitude of the sub-satellite point for a georeferencer [deg E].

    An explicit ``override`` wins; otherwise the georeferencer's
    ``subsatellite_lon_deg`` attribute is used when present, falling back
    to the Meteosat-9 nominal position.
    """
    if override is not None:
        return float(override)
    return float(getattr(georef, "subsatellite_lon_deg", _DEFAULT_SUBSATELLITE_LON_DEG))


def _georeference_detections(
    objects: Sequence[dict[str, Any]], georef: Any
) -> list[dict[str, Any]]:
    """Convert image-space detections to lat/lon centroids and headings.

    Objects whose centroid or heading cannot be georeferenced are dropped.
    """
    detections: list[dict[str, Any]] = []
    for obj in objects:
        row, col = (float(value) for value in obj["centroid"])
        lat, lon = georef.to_latlon(row, col)
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        heading = pixel_angle_to_heading_deg(
            float(obj["angle_deg"]), row, col, georef.to_latlon
        )
        if not math.isfinite(heading):
            continue
        detections.append(
            {
                "centroid": (lat, lon),
                "angle_deg": heading,
                "length_px": float(obj.get("length_px", float("nan"))),
                "mean_btd": float(obj.get("mean_btd", float("nan"))),
            }
        )
    return detections


def _trajectory_groups(trajectories: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split a trajectory frame by ``icao24`` (single group without it)."""
    missing = [
        name for name in _REQUIRED_TRAJECTORY_COLUMNS if name not in trajectories.columns
    ]
    if missing:
        raise ValueError(f"trajectory frame is missing columns: {missing}")
    if trajectories.empty:
        raise ValueError("trajectory frame is empty")
    if "icao24" not in trajectories.columns:
        return [("", trajectories)]
    groups: list[tuple[str, pd.DataFrame]] = []
    for icao24, group in trajectories.groupby("icao24", sort=True):
        groups.append((str(icao24), group))
    return groups


def build_observed_labels(
    *,
    bt_108: np.ndarray,
    bt_120: np.ndarray,
    georef: Any,
    observation_time: Any,
    trajectories: pd.DataFrame,
    ds_met: xr.Dataset,
    aircraft_type: str = "A320",
    threshold: float = 2.0,
    min_length_px: int = 8,
    distance_tolerance_km: float = 10.0,
    angle_tolerance_deg: float = 20.0,
    subsatellite_lon_deg: float | None = None,
    subsatellite_lat_deg: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Label ADS-B segments from one satellite scene and ERA5 state.

    Parameters
    ----------
    bt_108, bt_120 : numpy.ndarray
        10.8 um and 12.0 um brightness temperatures [K] of the scene,
        shape ``(n_rows, n_cols)``; see :func:`load_bt_pair`.
    georef : object
        Georeferencer with ``to_latlon(row, col) -> (lat, lon)`` (e.g.
        :class:`stratum.geo.angles.ArrayGeoreferencer`). A
        ``subsatellite_lon_deg`` attribute is honoured when
        ``subsatellite_lon_deg`` is not passed explicitly.
    observation_time : datetime-like
        Time of the scene; naive values are interpreted as UTC.
    trajectories : pandas.DataFrame
        Cleaned ADS-B points, split into flights by ``icao24`` (required
        unless the frame holds a single flight); requires ``time``,
        ``lat``, ``lon`` and ``baro_alt_ft``.
    ds_met : xarray.Dataset
        ERA5 dataset with the canonical variables.
    aircraft_type : str, optional
        Engine designator for the feature assembly, by default ``"A320"``.
    threshold : float, optional
        Detector response threshold [robust sigma], by default 2.0.
    min_length_px : int, optional
        Minimum accepted detection length [px], by default 8.
    distance_tolerance_km : float, optional
        Maximum centroid distance for a match [km], by default 10.0.
    angle_tolerance_deg : float, optional
        Maximum orientation difference for a match [deg], by default 20.0.
    subsatellite_lon_deg, subsatellite_lat_deg : float, optional
        Sub-satellite point [deg] for the viewing-zenith computation;
        defaults are read from the georeferencer and then from Meteosat-9
        (45.5 E, 0 N).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, list of dict)
        - feature matrix ``X`` of shape ``(N, len(FEATURE_NAMES))``, one
          row per consecutive trajectory point pair;
        - tri-state label vector ``y`` (``1.0`` matched, ``0.0``
          observable miss, ``nan`` unobservable/unavailable);
        - one metadata dictionary per segment with the provenance
          (``lat``, ``lon``, ``pressure_pa``, ``time_utc``), the
          ``matched``/``observable`` flags, ``observable_slots``, the
          viewing zenith angle, the advected/parallax-corrected
          coordinate, the segment heading and the ERA5 wind used.

    Raises
    ------
    ValueError
        For malformed scenes, trajectories or non-finite segment
        geometry.
    """
    bt_108 = np.asarray(bt_108, dtype=float)
    bt_120 = np.asarray(bt_120, dtype=float)
    if bt_108.ndim != 2 or bt_120.ndim != 2 or bt_108.shape != bt_120.shape:
        raise ValueError(
            "bt_108 and bt_120 must be 2-D arrays of the same shape, got "
            f"{bt_108.shape!r} and {bt_120.shape!r}"
        )
    observation = _coerce_utc(observation_time)
    subsatellite_lon = _resolve_subsatellite_lon(georef, subsatellite_lon_deg)
    sub_lat = (
        METEOSAT9_LATITUDE_DEG
        if subsatellite_lat_deg is None
        else float(subsatellite_lat_deg)
    )

    detection = detect_contrail_lines(
        bt_108, bt_120, threshold=threshold, min_length_px=min_length_px
    )
    detections = _georeference_detections(detection["objects"], georef)

    vectors: list[np.ndarray] = []
    labels: list[float] = []
    metadata: list[dict[str, Any]] = []

    for icao24, group in _trajectory_groups(trajectories):
        group = group.sort_values("time")
        times = pd.to_datetime(group["time"], utc=True)
        records: list[dict[str, Any]] = []

        for index in range(len(group) - 1):
            row_start = group.iloc[index]
            row_end = group.iloc[index + 1]
            lat_start = float(row_start["lat"])
            lon_start = float(row_start["lon"])
            lat_end = float(row_end["lat"])
            lon_end = float(row_end["lon"])
            altitude_start_ft = float(row_start["baro_alt_ft"])
            altitude_end_ft = float(row_end["baro_alt_ft"])
            if not all(
                math.isfinite(value)
                for value in (
                    lat_start,
                    lon_start,
                    lat_end,
                    lon_end,
                    altitude_start_ft,
                    altitude_end_ft,
                )
            ):
                raise ValueError(
                    f"trajectory {icao24!r} segment {index} has non-finite "
                    "position or altitude"
                )
            time_start = times.iloc[index]
            time_end = times.iloc[index + 1]
            if pd.isna(time_start) or pd.isna(time_end):
                raise ValueError(
                    f"trajectory {icao24!r} segment {index} has a missing time"
                )

            # Local planar midpoint; segment lengths on a corridor are far
            # below the scale where the approximation matters.
            mid_lat = 0.5 * (lat_start + lat_end)
            mid_lon = 0.5 * (lon_start + lon_end)
            mid_time = time_start + (time_end - time_start) / 2

            pressure_start = fl_to_p(altitude_start_ft / 100.0) * 100.0
            pressure_end = fl_to_p(altitude_end_ft / 100.0) * 100.0
            pressure_mid = 0.5 * (pressure_start + pressure_end)

            vector, info = extract_features_for_point(
                mid_lat,
                mid_lon,
                pressure_mid,
                mid_time,
                aircraft_type,
                ds_met=ds_met,
            )

            met_start = interpolate_met_to_segment(
                ds_met, lat_start, lon_start, pressure_start, time_start
            )
            met_end = interpolate_met_to_segment(
                ds_met, lat_end, lon_end, pressure_end, time_end
            )
            u_wind = 0.5 * (
                met_start["u_component_of_wind"] + met_end["u_component_of_wind"]
            )
            v_wind = 0.5 * (
                met_start["v_component_of_wind"] + met_end["v_component_of_wind"]
            )

            elapsed_seconds = (observation - mid_time).total_seconds()
            heading_deg = initial_bearing_deg(lat_start, lon_start, lat_end, lon_end)
            altitude_m = pressure_to_altitude_m(pressure_mid)

            record: dict[str, Any] = {
                "icao24": icao24,
                "segment_index": index,
                "lat": mid_lat,
                "lon": mid_lon,
                "pressure_pa": pressure_mid,
                "time_utc": mid_time.isoformat(),
                "data_available": bool(info["data_available"]),
                "unavailable_reason": info["unavailable_reason"],
                "high_cloud_fraction": info["high_cloud_fraction"],
                "viewing_zenith_angle_deg": viewing_zenith_angle_deg(
                    mid_lat, mid_lon, subsatellite_lon, sub_lat
                ),
                "heading_deg": heading_deg,
                "u_wind_mps": u_wind,
                "v_wind_mps": v_wind,
                "elapsed_seconds": elapsed_seconds,
                "altitude_m": altitude_m,
                "vector": np.asarray(vector, dtype=float),
                "matched": False,
                "observable": False,
                "advected_lat": float("nan"),
                "advected_lon": float("nan"),
                "parallax_delta_lat_deg": float("nan"),
                "parallax_delta_lon_deg": float("nan"),
                "label": float("nan"),
                "censored_reason": None,
            }

            if not info["data_available"]:
                record["censored_reason"] = info["unavailable_reason"]
                records.append(record)
                continue
            if elapsed_seconds < 0.0:
                # The segment happens after the satellite overpass; the
                # scene cannot say anything about it.
                record["censored_reason"] = _REASON_AFTER_OBSERVATION
                records.append(record)
                continue

            advected_lat, advected_lon = advect_segment(
                mid_lat, mid_lon, u_wind, v_wind, elapsed_seconds
            )
            delta_lat, delta_lon = calculate_parallax_displacement(
                advected_lat, advected_lon, altitude_m
            )
            record["advected_lat"] = advected_lat
            record["advected_lon"] = advected_lon
            record["parallax_delta_lat_deg"] = delta_lat
            record["parallax_delta_lon_deg"] = delta_lon

            record["observable"] = bool(
                math.isfinite(record["viewing_zenith_angle_deg"])
                and is_scene_observable(
                    float(info["high_cloud_fraction"]),
                    float(record["viewing_zenith_angle_deg"]),
                )
            )
            record["matched"] = bool(
                math.isfinite(heading_deg)
                and match_segment_to_detections(
                    (advected_lat + delta_lat, advected_lon + delta_lon),
                    heading_deg,
                    detections,
                    distance_tolerance_km=distance_tolerance_km,
                    angle_tolerance_deg=angle_tolerance_deg,
                )
            )
            records.append(record)

        observable_slots = sum(1 for record in records if record["observable"])
        for record in records:
            if record["censored_reason"] is not None:
                label = float("nan")
            else:
                label = assign_tristate_label(record["matched"], observable_slots)
            record["observable_slots"] = observable_slots
            record["label"] = label
            vectors.append(record.pop("vector"))
            labels.append(label)
            record["detection_count"] = int(detection["num_detected"])
            metadata.append(record)

    if vectors:
        features = np.vstack(vectors)
    else:
        features = np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return features, np.asarray(labels, dtype=float), metadata
