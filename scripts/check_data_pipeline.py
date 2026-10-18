#!/usr/bin/env python
"""Check the STRATUM real-data ingestion pipeline end to end.

Looks for the ERA5 analysis file ``era5_del_bom_20250101.nc`` in the
project root, a ``data/`` directory, and the current working directory.
When the file is present, ambient temperature and humidity are
interpolated along a simulated FL350 Delhi-Mumbai route and the
Schmidt-Appleman criterion is evaluated at every waypoint. When the file
is missing, the same check runs on an in-memory synthetic mock dataset
and prints download guidance.

Exit code is 0 when the route evaluation completes.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# The netCDF4 1.7.2 wheel can emit a spurious NumPy binary-compatibility
# RuntimeWarning on first engine use; functionality is exercised below.
warnings.filterwarnings(
    "ignore",
    message="numpy.ndarray size changed, may indicate binary incompatibility",
    category=RuntimeWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratum.io.era5 import (  # noqa: E402
    ERA5_PRESSURE_LEVELS_HPA,
    interpolate_met_to_segment,
    load_era5_dataset,
)
from stratum.physics import e_i, evaluate_sac  # noqa: E402
from stratum.routes.corridor import AIRPORTS, great_circle_waypoints  # noqa: E402

ERA5_FILENAME = "era5_del_bom_20250101.nc"
CRUISE_FL = 350.0
N_WAYPOINTS = 7
EPS = 0.622


def _candidate_paths() -> list[Path]:
    """Locations searched for the real ERA5 file, in order."""
    return [
        PROJECT_ROOT / ERA5_FILENAME,
        PROJECT_ROOT / "data" / ERA5_FILENAME,
        Path.cwd() / ERA5_FILENAME,
    ]


def _find_era5_file() -> Path | None:
    """Return the first existing candidate path, or ``None``."""
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate
    return None


def _rhi_from_specific_humidity(
    specific_humidity: float, pressure_pa: float, temperature_k: float
) -> float:
    """Convert ERA5 specific humidity to relative humidity over ice.

    Vapor pressure follows ``e = q p / (eps + (1 - eps) q)``; RHi divides
    by the Murphy & Koop ice saturation pressure.
    """
    vapor_pressure = specific_humidity * pressure_pa / (
        EPS + (1.0 - EPS) * specific_humidity
    )
    return vapor_pressure / e_i(temperature_k)


def _synthetic_era5() -> xr.Dataset:
    """Build a physically plausible mock ERA5 dataset over the domain.

    Temperature decreases with pressure along a fixed lapse; the specific
    humidity field is built by inverting the RHi definition for a
    wavenumber-like winter field between RHi 0.15 and 1.4, so the mock
    spans both forming and non-forming (and supersaturated) points.
    """
    lats = np.array([15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 32.0])
    lons = np.array([68.0, 71.0, 74.0, 77.0, 80.0, 82.0])
    levels = np.array(ERA5_PRESSURE_LEVELS_HPA, dtype=float)
    times = np.array(
        ["2025-01-01T00:00", "2025-01-01T06:00"], dtype="datetime64[ns]"
    )

    level = levels[:, None, None, None]
    lat = lats[None, :, None, None]
    lon = lons[None, None, :, None]
    t_index = np.arange(times.size, dtype=float)[None, None, None, :]

    temperature = (
        216.65
        + 0.11 * (level - 200.0)
        + 0.0 * lat
        + 0.0 * lon
        + 0.0 * t_index
    )
    target_rhi = np.clip(
        0.5
        + 0.6
        * np.sin(np.pi * (lat - 15.0) / 17.0)
        * np.cos(np.pi * (lon - 68.0) / 14.0),
        0.15,
        1.4,
    )
    saturation_pressure = np.array(
        [e_i(float(t)) for t in (216.65 + 0.11 * (levels - 200.0))]
    )[:, None, None, None]
    specific_humidity = (
        target_rhi * saturation_pressure * EPS / (level * 100.0)
    ) * (1.0 + 0.05 * t_index)

    u_wind = 8.0 + 0.05 * (lat - 15.0) + 0.0 * t_index + 0.0 * level + 0.0 * lon
    v_wind = -3.0 + 0.1 * (lon - 68.0) + 0.0 * t_index + 0.0 * level + 0.0 * lat
    omega = 0.002 + 0.0 * level + 0.0 * lat + 0.0 * lon + 0.0 * t_index
    cloud = 0.3 + 0.0 * level + 0.0 * lat + 0.0 * lon + 0.0 * t_index

    fields = {
        "temperature": temperature,
        "specific_humidity": specific_humidity,
        "u_component_of_wind": u_wind,
        "v_component_of_wind": v_wind,
        "vertical_velocity": omega,
        "fraction_of_cloud_cover": cloud,
    }
    data_vars = {
        name: (("time", "level", "latitude", "longitude"), field.transpose(3, 0, 1, 2))
        for name, field in fields.items()
    }
    return xr.Dataset(
        data_vars,
        coords={
            "time": times,
            "level": levels,
            "latitude": lats,
            "longitude": lons,
        },
    )


def _evaluate_route(
    ds_met: xr.Dataset,
    source_label: str,
    time_value: Any,
) -> int:
    """Evaluate the SAC along the simulated route; return the exit code."""
    waypoints = great_circle_waypoints(
        AIRPORTS["VIDP"], AIRPORTS["VABB"], num_points=N_WAYPOINTS, fl=CRUISE_FL
    )
    query_time = np.datetime_as_string(np.datetime64(time_value, "s"))
    print(f"Data source:  {source_label}")
    print(
        f"Route:        VIDP (Delhi) -> VABB (Mumbai), cruise FL{CRUISE_FL:.0f}, "
        f"{len(waypoints)} waypoints"
    )
    print(f"Query time:   {query_time} UTC")
    print()
    header = (
        f"{'#':>2} {'lat':>7} {'lon':>7} {'p [hPa]':>8} {'T [C]':>7} "
        f"{'RHi':>6} {'TLM [C]':>8} {'TLC [C]':>8} {'forms':>6} {'persists':>8}"
    )
    print(header)
    print("-" * len(header))

    sac_true = 0
    persists_count = 0
    for index, waypoint in enumerate(waypoints, start=1):
        values = interpolate_met_to_segment(
            ds_met,
            lat=waypoint["lat"],
            lon=waypoint["lon"],
            pressure_pa=waypoint["p_Pa"],
            time_dt=time_value,
        )
        temperature = values["temperature"]
        humidity = values["specific_humidity"]
        if not (math.isfinite(temperature) and math.isfinite(humidity)):
            print(f"{index:>2}  no data inside the dataset domain")
            continue
        rhi = _rhi_from_specific_humidity(humidity, waypoint["p_Pa"], temperature)
        sac = evaluate_sac(waypoint["p_Pa"], temperature, rhi)
        sac_true += int(sac["forms"])
        persists_count += int(sac["persists"])
        print(
            f"{index:>2} {waypoint['lat']:>7.2f} {waypoint['lon']:>7.2f} "
            f"{waypoint['p_Pa'] / 100.0:>8.1f} {temperature - 273.15:>7.2f} "
            f"{rhi:>6.2f} {sac['TLM_C']:>8.2f} {sac['TLC_C']:>8.2f} "
            f"{str(sac['forms']):>6} {str(sac['persists']):>8}"
        )

    print()
    print(
        f"SAC holds at {sac_true}/{len(waypoints)} waypoints; "
        f"ice-supersaturated at {persists_count}/{len(waypoints)}."
    )
    return 0


def main() -> int:
    real_path = _find_era5_file()
    if real_path is not None:
        ds_met = load_era5_dataset(str(real_path))
        return _evaluate_route(
            ds_met,
            f"real ERA5 file {real_path}",
            ds_met["time"].values[0],
        )

    print(f"NOTE: real ERA5 file '{ERA5_FILENAME}' not found.")
    print("Searched:")
    for candidate in _candidate_paths():
        print(f"  - {candidate}")
    print()
    print("Running the same check against an in-memory synthetic ERA5 mock instead.")
    print("To use real data, download the ERA5 file with the CDS API and place it at:")
    print(f"  {PROJECT_ROOT / 'data' / ERA5_FILENAME}")
    print()
    ds_met = _synthetic_era5()
    return _evaluate_route(
        ds_met,
        "synthetic mock (no ERA5 file on disk)",
        np.datetime64("2025-01-01T03:00"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
