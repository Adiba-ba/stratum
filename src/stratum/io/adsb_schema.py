"""Canonical ADS-B trajectory schema and cleaning for Project STRATUM.

Section 6.4 defines the canonical trajectory-point schema ingested from
OpenSky (`source='opensky'`) and ADS-B Exchange (`source='adsbx'`):

- ``icao24`` (str, lowercase), ``time`` (UTC), ``lat``/``lon`` (deg),
- ``baro_alt_ft`` and nullable ``geo_alt_ft`` [ft],
- ``gs_kt`` [kt], ``track_deg`` [deg], ``vrate_fpm`` [ft/min],
- ``callsign`` (str, stripped), ``on_ground`` (bool), ``source`` (str).

:func:`clean_trajectory_points` applies the ingestion filters: airborne
rows only, cruise retention (baro altitude >= FL250), the Indian domain
bounding box, kinematic plausibility, and de-duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "ADSB_COLUMNS",
    "REQUIRED_COLUMNS",
    "CRUISE_MIN_BARO_ALT_FT",
    "LATITUDE_BOUNDS_DEG",
    "LONGITUDE_BOUNDS_DEG",
    "MAX_ABS_VRATE_FPM",
    "MAX_GROUND_SPEED_JUMP_KT",
    "clean_trajectory_points",
]

#: Canonical column order of the cleaned trajectory frame.
ADSB_COLUMNS: tuple[str, ...] = (
    "icao24",
    "time",
    "lat",
    "lon",
    "baro_alt_ft",
    "geo_alt_ft",
    "gs_kt",
    "track_deg",
    "vrate_fpm",
    "callsign",
    "on_ground",
    "source",
)

#: Columns that must be present in the raw input; ``geo_alt_ft`` is the
#: only nullable column and is created when absent.
REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    name for name in ADSB_COLUMNS if name != "geo_alt_ft"
)

#: Cruise retention floor [ft]: keep FL250 and above.
CRUISE_MIN_BARO_ALT_FT: float = 25000.0

#: Indian domain bounding box for latitude and longitude [deg].
LATITUDE_BOUNDS_DEG: tuple[float, float] = (6.0, 37.0)
LONGITUDE_BOUNDS_DEG: tuple[float, float] = (68.0, 98.0)

#: Kinematic plausibility gates.
MAX_ABS_VRATE_FPM: float = 6000.0
MAX_GROUND_SPEED_JUMP_KT: float = 100.0

_NUMERIC_COLUMNS: tuple[str, ...] = (
    "lat",
    "lon",
    "baro_alt_ft",
    "geo_alt_ft",
    "gs_kt",
    "track_deg",
    "vrate_fpm",
)


def clean_trajectory_points(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw ADS-B points into the canonical schema (Section 6.4).

    Filters applied in order:

    1. type normalization (UTC time, lowercase ``icao24``, stripped
       ``callsign``, numeric columns coerced, lowercased ``source``);
    2. drop ``on_ground`` rows and rows missing ``time``/``lat``/``lon``/
       ``baro_alt_ft``;
    3. cruise retention: ``baro_alt_ft >= CRUISE_MIN_BARO_ALT_FT``;
    4. Indian domain bounding box;
    5. sort by ``(icao24, time)`` and drop duplicate fixes;
    6. kinematic plausibility: drop a point when its vertical rate
       exceeds ``MAX_ABS_VRATE_FPM`` or its ground speed jumps by more
       than ``MAX_GROUND_SPEED_JUMP_KT`` relative to the previous fix of
       the same aircraft.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw points containing at least :data:`REQUIRED_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        Cleaned frame with exactly :data:`ADSB_COLUMNS`, sorted by
        ``(icao24, time)`` and re-indexed from 0.

    Raises
    ------
    ValueError
        If a required column is missing.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"ADS-B frame is missing required columns: {missing}")

    cleaned = df.copy()
    if "geo_alt_ft" not in cleaned.columns:
        cleaned["geo_alt_ft"] = np.nan

    cleaned["time"] = pd.to_datetime(cleaned["time"], utc=True)
    for name in _NUMERIC_COLUMNS:
        cleaned[name] = pd.to_numeric(cleaned[name], errors="coerce")
    cleaned["icao24"] = cleaned["icao24"].astype(str).str.strip().str.lower()
    cleaned["callsign"] = cleaned["callsign"].astype(str).str.strip()
    cleaned["source"] = cleaned["source"].astype(str).str.strip().str.lower()

    cleaned = cleaned.dropna(subset=["time", "lat", "lon", "baro_alt_ft"])
    on_ground = cleaned["on_ground"].fillna(False).astype(bool)
    cleaned = cleaned[~on_ground]
    cleaned = cleaned[cleaned["baro_alt_ft"] >= CRUISE_MIN_BARO_ALT_FT]
    cleaned = cleaned[
        cleaned["lat"].between(*LATITUDE_BOUNDS_DEG)
        & cleaned["lon"].between(*LONGITUDE_BOUNDS_DEG)
    ]

    cleaned = cleaned.sort_values(["icao24", "time"], kind="mergesort")
    cleaned = cleaned.drop_duplicates(subset=["icao24", "time"], keep="first")

    previous_gs = cleaned.groupby("icao24", sort=False)["gs_kt"].shift(1)
    ground_speed_jump = (cleaned["gs_kt"] - previous_gs).abs()
    plausible = (cleaned["vrate_fpm"].abs() <= MAX_ABS_VRATE_FPM) & ~(
        ground_speed_jump > MAX_GROUND_SPEED_JUMP_KT
    )
    cleaned = cleaned[plausible]

    return cleaned.loc[:, list(ADSB_COLUMNS)].reset_index(drop=True)
