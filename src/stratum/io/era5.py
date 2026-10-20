"""ERA5 reanalysis ingestion and point interpolation for STRATUM.

The ingestion layer targets the Indian subdomain used by the pipeline:

- bounding box ``(North, West, South, East) = (32.0, 68.0, 15.0, 82.0)``,
- pressure levels ``[200, 225, 250, 300, 350, 400]`` hPa,
- the six meteorological variables required by the correction model.

:func:`load_era5_dataset` validates a NetCDF file,
:func:`interpolate_met_to_segment` extracts local ambient values at an
arbitrary ``(lat, lon, pressure, time)`` point through multi-dimensional
linear interpolation, and :func:`is_in_domain` reports whether an
arbitrary point falls inside that domain.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "ERA5_BBOX",
    "ERA5_PRESSURE_LEVELS_HPA",
    "ERA5_VARIABLES",
    "is_in_domain",
    "load_era5_dataset",
    "interpolate_met_to_segment",
]

#: Domain bounding box (North, West, South, East) [deg].
#: Operating domain of the pipeline: north, west, south, east [deg].
#: Covers the Indian subcontinent, the Arabian Sea and the Bay of Bengal,
#: which is the area the downloaded ERA5 extract spans.
ERA5_BBOX: tuple[float, float, float, float] = (40.0, 60.0, 5.0, 95.0)

#: Pressure levels [hPa] stored for the pipeline.
ERA5_PRESSURE_LEVELS_HPA: tuple[int, ...] = (200, 225, 250, 300, 350, 400)

#: Meteorological variables required by the correction-model features.
ERA5_VARIABLES: tuple[str, ...] = (
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "fraction_of_cloud_cover",
)

_LATITUDE_NAMES: tuple[str, ...] = ("latitude", "lat")
_LONGITUDE_NAMES: tuple[str, ...] = ("longitude", "lon")
_LEVEL_NAMES: tuple[str, ...] = ("level", "pressure_level")
_TIME_NAMES: tuple[str, ...] = ("time", "valid_time")

#: CDS NetCDF exports may use short variable names; canonical name first.
_VARIABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "temperature": ("temperature", "t"),
    "specific_humidity": ("specific_humidity", "q"),
    "u_component_of_wind": ("u_component_of_wind", "u"),
    "v_component_of_wind": ("v_component_of_wind", "v"),
    "vertical_velocity": ("vertical_velocity", "w"),
    "fraction_of_cloud_cover": ("fraction_of_cloud_cover", "cc", "tcc"),
}

#: Absolute position tolerance [deg] used by :func:`is_in_domain`.
_DOMAIN_TOLERANCE_DEG: float = 1.0e-6

#: Absolute pressure tolerance [Pa] used by :func:`is_in_domain`.
_DOMAIN_TOLERANCE_PA: float = 1.0e-3


def _resolve_coordinate(ds: xr.Dataset, names: tuple[str, ...], role: str) -> str:
    """Return the first coordinate name in ``names`` present in ``ds``."""
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(
        f"ERA5 dataset is missing the {role} coordinate; expected one of {list(names)}"
    )


def _to_nanosecond_utc(value: Any) -> np.datetime64:
    """Convert a datetime-like value to a tz-naive UTC ``datetime64[ns]``."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return np.datetime64(timestamp, "ns")


def _in_closed_range(
    value: float, lower: float, upper: float, tolerance: float
) -> bool:
    """Return whether ``value`` lies in the closed ``[lower, upper]`` band."""
    return lower - tolerance <= value <= upper + tolerance


def load_era5_dataset(file_path: str) -> xr.Dataset:
    """Load and validate an ERA5 NetCDF file into memory.

    Validation checks the presence of the latitude, longitude, pressure
    level and time coordinates and of all :data:`ERA5_VARIABLES`. Both
    the classic ERA5 naming (``time``, ``level``, long variable names)
    and the newer CDS export naming (``valid_time``, ``pressure_level``,
    short variable names such as ``t``/``q``/``cc``) are accepted; the
    returned dataset uses canonical names. Latitude is sorted ascending
    when the file stores it descending. The dataset is fully loaded and
    the source file handle is closed before it is returned, so no
    descriptor outlives the call.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file.

    Returns
    -------
    xarray.Dataset
        Validated in-memory dataset with canonical variable names,
        latitude ascending and the time coordinate renamed to ``time``.

    Raises
    ------
    ValueError
        If a required coordinate or variable is missing.
    FileNotFoundError
        If ``file_path`` does not exist.
    """
    with xr.open_dataset(file_path) as source:
        latitude_name = _resolve_coordinate(source, _LATITUDE_NAMES, "latitude")
        _resolve_coordinate(source, _LONGITUDE_NAMES, "longitude")
        _resolve_coordinate(source, _LEVEL_NAMES, "pressure level")
        time_name = _resolve_coordinate(source, _TIME_NAMES, "time")

        rename_map: dict[str, str] = {}
        if time_name != "time":
            rename_map[time_name] = "time"

        missing_variables: list[str] = []
        for canonical, aliases in _VARIABLE_ALIASES.items():
            actual = next(
                (name for name in aliases if name in source.data_vars), None
            )
            if actual is None:
                missing_variables.append(canonical)
            elif actual != canonical:
                rename_map[actual] = canonical
        if missing_variables:
            raise ValueError(
                f"ERA5 dataset {file_path!r} is missing variables: "
                f"{missing_variables}"
            )
        ds = source.rename(rename_map) if rename_map else source

        latitude = ds[latitude_name]
        if latitude.size > 1 and float(latitude[0]) > float(latitude[-1]):
            ds = ds.sortby(latitude_name)
        ds.load()
    return ds


def is_in_domain(lat_deg: float, lon_deg: float, pressure_pa: float) -> bool:
    """Report whether a query point lies inside the ERA5 ingestion domain.

    The domain spans :data:`ERA5_BBOX` in latitude/longitude and the
    closed interval covered by :data:`ERA5_PRESSURE_LEVELS_HPA`
    (``20000.0`` to ``40000.0`` Pa). All bounds are inclusive, with an
    absolute tolerance to absorb floating-point representation error:
    ``1e-6`` deg for position and ``1e-3`` Pa for pressure. A value
    exactly on a bound (e.g. ``lat_deg=32.0`` or ``pressure_pa=40000.0``)
    counts as inside; a value more than one tolerance outside does not.
    Non-finite inputs are outside.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Query position [deg N, deg E].
    pressure_pa : float
        Query pressure [Pa].

    Returns
    -------
    bool
        ``True`` when the point lies inside the closed domain.
    """
    if not (
        math.isfinite(lat_deg)
        and math.isfinite(lon_deg)
        and math.isfinite(pressure_pa)
    ):
        return False
    north, west, south, east = ERA5_BBOX
    pressure_min_pa = min(ERA5_PRESSURE_LEVELS_HPA) * 100.0
    pressure_max_pa = max(ERA5_PRESSURE_LEVELS_HPA) * 100.0
    return (
        _in_closed_range(lat_deg, south, north, _DOMAIN_TOLERANCE_DEG)
        and _in_closed_range(lon_deg, west, east, _DOMAIN_TOLERANCE_DEG)
        and _in_closed_range(
            pressure_pa,
            pressure_min_pa,
            pressure_max_pa,
            _DOMAIN_TOLERANCE_PA,
        )
    )


def interpolate_met_to_segment(
    ds_met: xr.Dataset,
    lat: float,
    lon: float,
    pressure_pa: float,
    time_dt: Any,
) -> dict[str, float]:
    """Interpolate local ambient values at a segment point.

    Performs linear multi-dimensional interpolation over latitude,
    longitude, pressure level and time. Points outside the dataset domain
    (or pressure range) return ``nan`` values, matching xarray semantics.

    Parameters
    ----------
    ds_met : xarray.Dataset
        Validated ERA5 dataset, e.g. from :func:`load_era5_dataset`.
    lat, lon : float
        Query position [deg N, deg E].
    pressure_pa : float
        Query pressure [Pa]; converted to hPa for the level axis.
    time_dt : datetime-like
        Query time; tz-aware values are converted to UTC.

    Returns
    -------
    dict[str, float]
        One entry per :data:`ERA5_VARIABLES` present in ``ds_met``.

    Raises
    ------
    ValueError
        For non-finite coordinates, non-positive pressure, or a dataset
        missing required coordinates.
    """
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError(f"lat and lon must be finite, got {lat!r} and {lon!r}")
    if not math.isfinite(pressure_pa) or pressure_pa <= 0.0:
        raise ValueError(f"pressure_pa must be positive, got {pressure_pa!r}")

    latitude_name = _resolve_coordinate(ds_met, _LATITUDE_NAMES, "latitude")
    longitude_name = _resolve_coordinate(ds_met, _LONGITUDE_NAMES, "longitude")
    level_name = _resolve_coordinate(ds_met, _LEVEL_NAMES, "pressure level")
    time_name = _resolve_coordinate(ds_met, _TIME_NAMES, "time")

    available = [name for name in ERA5_VARIABLES if name in ds_met.data_vars]
    if not available:
        raise ValueError(
            "dataset contains none of the ERA5_VARIABLES under canonical names; "
            "run load_era5_dataset first"
        )
    subset = ds_met[available]
    interp_coordinate_names = {
        latitude_name,
        longitude_name,
        level_name,
        time_name,
    }
    # Drop auxiliary non-dimension coordinates (e.g. the string-valued
    # CDS ``expver`` track) that xarray cannot interpolate.
    auxiliary_coordinates = [
        name
        for name in subset.coords
        if name not in interp_coordinate_names and name not in subset.dims
    ]
    if auxiliary_coordinates:
        subset = subset.drop_vars(auxiliary_coordinates)

    # Interpolating the whole time axis for every query costs the same
    # work for a one-day and a one-week extract, so the two bracketing
    # hours are selected first and the space interpolation runs on those
    # slices only. The result matches a full four-dimensional
    # interpolation, including NaN outside the time range.
    times = np.asarray(subset[time_name].values).astype("datetime64[ns]")
    target_ns = np.datetime64(int(_to_nanosecond_utc(time_dt)), "ns")
    if target_ns < times[0] or target_ns > times[-1]:
        return {name: float("nan") for name in available}

    upper = int(np.searchsorted(times, target_ns, side="left"))
    if upper >= times.size:
        upper = times.size - 1
    lower = upper - 1 if times[upper] != target_ns else upper

    def _slice_values(index: int) -> dict[str, float]:
        one = subset.isel({time_name: index})
        point = one.interp(
            {
                latitude_name: float(lat),
                longitude_name: float(lon),
                level_name: pressure_pa / 100.0,
            }
        )
        values_at_time: dict[str, float] = {}
        for name in available:
            scalar = point[name].squeeze()
            if scalar.ndim > 0:
                # Collapse auxiliary dimensions (e.g. ERA5 ``expver``) by
                # taking the first index; documented schema has none.
                scalar = scalar.isel({dim: 0 for dim in scalar.dims})
            values_at_time[name] = float(scalar.values)
        return values_at_time

    lower_values = _slice_values(lower)
    if lower == upper:
        return lower_values
    upper_values = _slice_values(upper)
    span = float((times[upper] - times[lower]) / np.timedelta64(1, "ns"))
    weight = float((target_ns - times[lower]) / np.timedelta64(1, "ns")) / span
    return {
        name: (1.0 - weight) * lower_values[name] + weight * upper_values[name]
        for name in available
    }
