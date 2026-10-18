"""OpenSky-style ADS-B CSV ingestion into the canonical trajectory schema.

Real trajectory exports (OpenSky Network CSV downloads and equivalent
files) carry header names and units that differ from the canonical
schema in :mod:`stratum.io.adsb_schema`. This module bridges them:

- :data:`OPENSKY_COLUMN_MAP` maps recognized raw header names to the
  :data:`~stratum.io.adsb_schema.ADSB_COLUMNS` names;
- :func:`load_adsb_csv` reads a CSV with case-insensitive headers,
  converts units, normalizes timestamps to UTC, and applies
  :func:`~stratum.io.adsb_schema.clean_trajectory_points`;
- :func:`select_trajectory` extracts one flight by ``callsign`` or
  ``icao24``, ordered in time.

Unit conversion rules
---------------------
Exactly two source units differ from the canonical schema; each is
converted with the exact constant shown:

- metre altitudes to feet: multiply by ``METRES_TO_FEET``
  (``3.280839895013123``);
- metre-per-second vertical rates to feet per minute: multiply by
  ``MPS_TO_FPM`` (``196.8503937007874``).

Conversion is keyed on the raw header name, because the same unit
appears under several names:

- metre altitudes: ``baro_altitude`` and ``geo_altitude``
  (``geo_altitude`` is also the ``baro_alt_ft`` fallback when no
  barometric altitude column exists);
- metre-per-second vertical rates: ``vertical_rate`` and ``baro_rate``.

Every other recognized header is taken as already canonical:
``baro_alt_ft`` and ``altitude`` in feet, ``vrate``/``vrate_fpm`` in
feet per minute, ``velocity``/``ground_speed``/``spd`` in knots,
``track``/``heading``/``true_track`` in degrees, ``lat``/``latitude``
and ``lon``/``longitude`` in degrees. A custom header mapped through
the ``column_map`` argument is copied without conversion.

Time normalization
------------------
``time_unit="auto"`` inspects the time column. Numeric values greater
than ``1e11`` are milliseconds since the Unix epoch; smaller numeric
values are seconds. ISO 8601 strings are parsed with
``pandas.to_datetime(..., utc=True)``. The returned ``time`` column is
tz-naive UTC (``datetime64[ns]``), the convention used by the advection
and ERA5 callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from stratum.io.adsb_schema import (
    ADSB_COLUMNS,
    REQUIRED_COLUMNS,
    clean_trajectory_points,
)

__all__ = [
    "OPENSKY_COLUMN_MAP",
    "METRES_TO_FEET",
    "MPS_TO_FPM",
    "load_adsb_csv",
    "select_trajectory",
]

#: Metres to feet conversion factor.
METRES_TO_FEET: float = 3.280839895013123

#: Metres per second to feet per minute conversion factor.
MPS_TO_FPM: float = 196.8503937007874

#: Numeric epoch values above this bound are read as milliseconds.
_MS_EPOCH_THRESHOLD: float = 1e11

_TIME_UNITS: tuple[str, ...] = ("auto", "s", "ms")

#: Raw header name to canonical column. Insertion order is the
#: resolution priority: when a CSV carries several aliases of the same
#: canonical column, the first listed wins. Header matching is
#: case-insensitive and whitespace-insensitive; see
#: :func:`_normalize_header`.
OPENSKY_COLUMN_MAP: dict[str, str] = {
    "icao24": "icao24",
    "callsign": "callsign",
    "call_sign": "callsign",
    "time": "time",
    "timestamp": "time",
    "snapshot_time": "time",
    "position_time": "time",
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "longitude": "lon",
    "baro_alt_ft": "baro_alt_ft",
    "baro_altitude": "baro_alt_ft",
    "altitude": "baro_alt_ft",
    "geo_alt_ft": "geo_alt_ft",
    "geo_altitude": "geo_alt_ft",
    "gs_kt": "gs_kt",
    "velocity": "gs_kt",
    "ground_speed": "gs_kt",
    "spd": "gs_kt",
    "track_deg": "track_deg",
    "track": "track_deg",
    "heading": "track_deg",
    "true_track": "track_deg",
    "vrate_fpm": "vrate_fpm",
    "vertical_rate": "vrate_fpm",
    "vrate": "vrate_fpm",
    "baro_rate": "vrate_fpm",
    "on_ground": "on_ground",
    "source": "source",
}

#: Raw header names whose altitude values are metres.
_METRE_ALTITUDE_SOURCES: frozenset[str] = frozenset({"baro_altitude", "geo_altitude"})

#: Raw header names whose vertical rates are metres per second.
_MPS_VERTICAL_RATE_SOURCES: frozenset[str] = frozenset({"vertical_rate", "baro_rate"})

#: Extra ``baro_alt_ft`` source, tried after the mapped barometric names.
_BARO_ALTITUDE_FALLBACKS: tuple[str, ...] = ("geo_altitude",)

#: String values read as ``True`` in the boolean ``on_ground`` column.
_TRUTHY_VALUES: frozenset[str] = frozenset({"true", "t", "1", "1.0", "yes", "y"})


def _normalize_header(name: object) -> str:
    """Normalize a header name for case- and whitespace-insensitive matching."""
    text = str(name).strip().lstrip("\ufeff").lower()
    return "_".join(text.split())


def _candidate_names(target: str, column_map: Mapping[str, str]) -> tuple[str, ...]:
    """Return the raw header names that resolve to ``target``, in priority order."""
    names = [raw for raw, canonical in column_map.items() if canonical == target]
    if target == "baro_alt_ft":
        names.extend(
            name for name in _BARO_ALTITUDE_FALLBACKS if name not in names
        )
    return tuple(names)


def _resolve_sources(
    columns: pd.Index, column_map: Mapping[str, str]
) -> dict[str, str]:
    """Map canonical columns to the actual CSV headers present in the file."""
    lookup: dict[str, str] = {}
    for column in columns:
        lookup.setdefault(_normalize_header(column), str(column))

    resolved: dict[str, str] = {}
    for target in ADSB_COLUMNS:
        for candidate in _candidate_names(target, column_map):
            actual = lookup.get(_normalize_header(candidate))
            if actual is not None:
                resolved[target] = actual
                break
    return resolved


def _as_text(series: pd.Series) -> pd.Series:
    """Return a whitespace-stripped string view of ``series``."""
    return series.astype(str).str.strip()


def _as_numeric(series: pd.Series) -> pd.Series:
    """Coerce ``series`` to numeric, mapping blanks to ``NaN``."""
    text = series.astype(str).str.strip()
    return pd.to_numeric(text.where(text.ne(""), np.nan), errors="coerce")


def _as_boolean(series: pd.Series) -> pd.Series:
    """Coerce ``series`` to bool; unknown or blank values become ``False``."""
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(_TRUTHY_VALUES)


def _altitude_to_feet(series: pd.Series, raw_name: str) -> pd.Series:
    """Convert an altitude column to feet using the metre-source rules."""
    values = _as_numeric(series)
    if _normalize_header(raw_name) in _METRE_ALTITUDE_SOURCES:
        values = values * METRES_TO_FEET
    return values


def _vertical_rate_to_fpm(series: pd.Series, raw_name: str) -> pd.Series:
    """Convert a vertical-rate column to feet per minute."""
    values = _as_numeric(series)
    if _normalize_header(raw_name) in _MPS_VERTICAL_RATE_SOURCES:
        values = values * MPS_TO_FPM
    return values


def _normalize_time(
    series: pd.Series, time_unit: str, raw_name: str
) -> pd.Series:
    """Parse a time column into tz-naive UTC ``datetime64[ns]``.

    Numeric columns are read as epoch values; values above
    ``_MS_EPOCH_THRESHOLD`` mean milliseconds, the rest seconds.
    Non-numeric columns are parsed as ISO 8601 via ``pandas.to_datetime``.
    """
    if time_unit not in _TIME_UNITS:
        raise ValueError(
            f"time_unit must be one of {list(_TIME_UNITS)}, got {time_unit!r}"
        )

    text = series.astype(str).str.strip()
    non_blank = text.ne("")
    numeric = pd.to_numeric(text.where(non_blank, np.nan), errors="coerce")
    use_numeric = time_unit in ("s", "ms") or (
        bool(non_blank.any()) and int(numeric.notna().sum()) == int(non_blank.sum())
    )

    if use_numeric:
        unit = time_unit
        if unit == "auto":
            unit = "ms" if float(numeric.max()) > _MS_EPOCH_THRESHOLD else "s"
        stamps = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        stamps = pd.to_datetime(
            text.where(non_blank, np.nan), utc=True, errors="coerce"
        )

    if bool(non_blank.any()) and bool(stamps.isna().all()):
        raise ValueError(
            f"could not parse the time column {raw_name!r} with "
            f"time_unit={time_unit!r}; expected epoch seconds/milliseconds "
            "or ISO 8601 timestamps"
        )
    return stamps.dt.tz_convert("UTC").dt.tz_localize(None)


def load_adsb_csv(
    path: str | Path,
    *,
    column_map: Mapping[str, str] | None = None,
    time_unit: str = "auto",
) -> pd.DataFrame:
    """Load an OpenSky-style ADS-B CSV into the canonical schema.

    Headers are matched case-insensitively after stripping whitespace and
    normalizing internal whitespace to underscores. Recognized names and
    their units are listed in :data:`OPENSKY_COLUMN_MAP` and the module
    docstring. Absent nullable or synthesized columns are filled in
    (``geo_alt_ft`` with ``NaN``, ``source`` with ``"opensky"``). The
    frame is then cleaned with
    :func:`~stratum.io.adsb_schema.clean_trajectory_points`: airborne
    cruise points only, inside the Indian domain, kinematically
    plausible and de-duplicated.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file to read.
    column_map : mapping of str to str, optional
        Extra raw-header to canonical-name mappings, merged over
        :data:`OPENSKY_COLUMN_MAP`; explicit entries add aliases and
        their values are copied without unit conversion.
    time_unit : {"auto", "s", "ms"}, optional
        Unit of the time column. ``"auto"`` (default) treats numeric
        values greater than ``1e11`` as milliseconds and smaller values
        as seconds, and parses non-numeric values as ISO 8601. ``"s"``
        and ``"ms"`` force numeric epoch interpretation.

    Returns
    -------
    pandas.DataFrame
        Cleaned points with exactly :data:`ADSB_COLUMNS`, sorted by
        ``(icao24, time)`` and re-indexed from 0. ``time`` is tz-naive
        UTC (``datetime64[ns]``); ``clean_trajectory_points`` localizes
        to UTC, and the loader drops the timezone to honor the tz-naive
        ingestion contract.

    Raises
    ------
    ValueError
        If a required column is missing, ``time_unit`` is unknown, or no
        timestamp in the time column could be parsed.
    """
    csv_path = Path(path)
    merged_map: dict[str, str] = dict(OPENSKY_COLUMN_MAP)
    if column_map is not None:
        merged_map.update(dict(column_map))

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    sources = _resolve_sources(frame.columns, merged_map)

    # ``source`` is synthesized when absent, so it is never "missing".
    required = [name for name in REQUIRED_COLUMNS if name != "source"]
    missing = [name for name in required if name not in sources]
    if missing:
        recognized = {
            name: list(_candidate_names(name, merged_map)) for name in missing
        }
        raise ValueError(
            f"ADS-B CSV {str(csv_path)!r} is missing required columns: "
            f"{missing}; recognized header names per column: {recognized}"
        )

    columns: dict[str, pd.Series] = {}
    columns["icao24"] = _as_text(frame[sources["icao24"]])
    columns["time"] = _normalize_time(
        frame[sources["time"]], time_unit, sources["time"]
    )
    columns["lat"] = _as_numeric(frame[sources["lat"]])
    columns["lon"] = _as_numeric(frame[sources["lon"]])
    columns["baro_alt_ft"] = _altitude_to_feet(
        frame[sources["baro_alt_ft"]], sources["baro_alt_ft"]
    )
    if "geo_alt_ft" in sources:
        columns["geo_alt_ft"] = _altitude_to_feet(
            frame[sources["geo_alt_ft"]], sources["geo_alt_ft"]
        )
    else:
        columns["geo_alt_ft"] = pd.Series(np.nan, index=frame.index, dtype=float)
    columns["gs_kt"] = _as_numeric(frame[sources["gs_kt"]])
    columns["track_deg"] = _as_numeric(frame[sources["track_deg"]])
    columns["vrate_fpm"] = _vertical_rate_to_fpm(
        frame[sources["vrate_fpm"]], sources["vrate_fpm"]
    )
    columns["callsign"] = _as_text(frame[sources["callsign"]])
    columns["on_ground"] = _as_boolean(frame[sources["on_ground"]])
    if "source" in sources:
        columns["source"] = _as_text(frame[sources["source"]])
    else:
        columns["source"] = pd.Series("opensky", index=frame.index)

    points = pd.DataFrame(columns)[list(ADSB_COLUMNS)]
    cleaned = clean_trajectory_points(points)
    cleaned["time"] = cleaned["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    return cleaned


def select_trajectory(
    df: pd.DataFrame,
    *,
    callsign: str | None = None,
    icao24: str | None = None,
) -> pd.DataFrame:
    """Select one flight from a trajectory frame and sort it in time.

    Parameters
    ----------
    df : pandas.DataFrame
        Trajectory points; must contain ``time`` and the column matching
        the chosen identifier.
    callsign : str, optional
        Callsign to match, compared after stripping whitespace and
        upper-casing.
    icao24 : str, optional
        ICAO 24-bit address to match, compared after stripping
        whitespace and lower-casing.

    Returns
    -------
    pandas.DataFrame
        Matching rows sorted by ``time`` ascending and re-indexed from 0.

    Raises
    ------
    ValueError
        If not exactly one identifier is given or a required column is
        missing.
    """
    if (callsign is None) == (icao24 is None):
        raise ValueError("provide exactly one of callsign or icao24")

    if callsign is not None:
        wanted = str(callsign).strip().upper()
        if not wanted:
            raise ValueError("callsign must be a non-empty string")
        if "callsign" not in df.columns:
            raise ValueError(
                "ADS-B frame is missing required columns: ['callsign']"
            )
        keys = df["callsign"].astype(str).str.strip().str.upper()
        selected = df[keys == wanted]
    else:
        wanted = str(icao24).strip().lower()
        if not wanted:
            raise ValueError("icao24 must be a non-empty string")
        if "icao24" not in df.columns:
            raise ValueError("ADS-B frame is missing required columns: ['icao24']")
        keys = df["icao24"].astype(str).str.strip().str.lower()
        selected = df[keys == wanted]

    if "time" not in selected.columns:
        raise ValueError("ADS-B frame is missing required columns: ['time']")

    selected = selected.sort_values("time", kind="mergesort")
    return selected.reset_index(drop=True)
