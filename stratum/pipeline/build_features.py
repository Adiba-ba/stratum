"""Feature assembly along cleaned ADS-B trajectories (Step 8).

For every trajectory point the assembler converts barometric altitude to a
pressure level, interpolates the ambient ERA5 state at the target level
and at +/- 25 hPa humidity offsets, evaluates the Schmidt-Appleman
thresholds, derives the remaining atmospheric features, and produces one
row in the canonical feature order of :mod:`stratum.model.features`.

Reference assumptions (documented, not measured):

- the humidity offsets sample the vertical structure of humidity: the
  "above" sample uses the lower pressure ``p - 25 hPa``, the "below"
  sample the higher pressure ``p + 25 hPa``;
- the tropopause is taken at 16 km (the standard tropical/subtropical
  level near 100 hPa), so ``dist_to_tropopause_m`` is signed relative to
  it;
- ``cloud_ice`` is a proxy flag (cloud cover above 0.5) because the
  canonical ERA5 variable set does not carry cloud ice content;
- turbulence enters through the gradient Richardson number computed on
  the two bounding layers, radiative geometry through the solar zenith
  angle, and engine microphysics through the designator lookup in
  :mod:`stratum.physics.diagnostics`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stratum.io.era5 import interpolate_met_to_segment
from stratum.model.features import extract_features_row, feature_vector
from stratum.physics import G, critical_T, e_i, e_w, eps, fl_to_p, solve_TLM
from stratum.physics import thermo
from stratum.physics.diagnostics import (
    compute_richardson_number,
    compute_solar_zenith_angle,
    get_aircraft_params,
)

__all__ = [
    "TROPOPAUSE_ALTITUDE_M",
    "HUMIDITY_OFFSET_HPA",
    "CLOUD_ICE_COVER_THRESHOLD",
    "specific_humidity_to_rhi",
    "pressure_to_altitude_m",
    "extract_features_for_trajectory",
]

#: Assumed tropical/subtropical tropopause altitude [m] (~100 hPa).
TROPOPAUSE_ALTITUDE_M: float = 16000.0

#: Vertical humidity sampling offset [hPa] for the structure features.
HUMIDITY_OFFSET_HPA: float = 25.0

#: Cloud cover threshold used as the ``cloud_ice`` proxy flag.
CLOUD_ICE_COVER_THRESHOLD: float = 0.5

_REQUIRED_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "time",
    "lat",
    "lon",
    "baro_alt_ft",
)

_KELVIN_OFFSET: float = 273.15


def specific_humidity_to_rhi(
    specific_humidity: float,
    pressure_pa: float,
    temperature_k: float,
) -> float:
    """Convert specific humidity to relative humidity over ice [-].

    Vapor pressure follows ``e = q p / (eps + (1 - eps) q)`` and RHi is
    ``e / e_i(T)`` with the Murphy & Koop ice curve. Returns ``nan`` for
    non-finite inputs.
    """
    if not (
        math.isfinite(specific_humidity)
        and math.isfinite(pressure_pa)
        and math.isfinite(temperature_k)
    ) or pressure_pa <= 0.0:
        return float("nan")
    vapor_pressure = specific_humidity * pressure_pa / (
        eps + (1.0 - eps) * specific_humidity
    )
    return vapor_pressure / e_i(temperature_k)


def pressure_to_altitude_m(pressure_pa: float) -> float:
    """Invert the ISA pressure-altitude relation [m] for ``pressure_pa``.

    Uses the tropospheric lapse below the 11 km tropopause and the
    isothermal layer above it (ICAO standard atmosphere).

    Raises
    ------
    ValueError
        For non-positive or non-finite pressure.
    """
    if not math.isfinite(pressure_pa) or pressure_pa <= 0.0:
        raise ValueError(f"pressure_pa must be positive, got {pressure_pa!r}")

    exponent = thermo.G0_M_PER_S2 / (
        thermo.R_AIR_J_PER_KG_K * thermo.LAPSE_K_PER_M
    )
    pressure_tropopause = thermo.P0_PA * (
        thermo.T_TROPOPAUSE_K / thermo.T0_K
    ) ** exponent
    if pressure_pa >= pressure_tropopause:
        temperature_k = thermo.T0_K * (pressure_pa / thermo.P0_PA) ** (
            1.0 / exponent
        )
        return (thermo.T0_K - temperature_k) / thermo.LAPSE_K_PER_M
    return thermo.H_TROPOPAUSE_M + (
        thermo.R_AIR_J_PER_KG_K
        * thermo.T_TROPOPAUSE_K
        / thermo.G0_M_PER_S2
    ) * math.log(pressure_tropopause / pressure_pa)


def extract_features_for_trajectory(
    df_trajectory: pd.DataFrame,
    ds_met: xr.Dataset,
    aircraft_type: str = "A320",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Assemble the canonical feature matrix along a trajectory.

    Parameters
    ----------
    df_trajectory : pandas.DataFrame
        Cleaned ADS-B points; requires ``time``, ``lat``, ``lon`` and
        ``baro_alt_ft``. Optional ``icao24``/``callsign`` columns are
        echoed into the metadata.
    ds_met : xarray.Dataset
        ERA5 dataset with the canonical variables, e.g. from
        :func:`stratum.io.era5.load_era5_dataset`.
    aircraft_type : str, optional
        ICAO designator used to look up the engine parameters
        (propulsion efficiency ``eta`` and nvPM number emission index)
        from :data:`stratum.physics.diagnostics.AIRCRAFT_ENGINE_DATABASE`,
        by default ``"A320"``.

    Returns
    -------
    tuple of (numpy.ndarray, list of dict)
        - feature matrix ``X`` of shape ``(N, len(FEATURE_NAMES))`` in
          canonical order;
        - one metadata dictionary per row (time, position, flight level,
          ambient state, SAC and diagnostic values) for reporting and
          audits.

    Raises
    ------
    ValueError
        If required columns are missing, the frame is empty, or a row
        carries non-finite position/altitude or a missing time.
    """
    missing = [
        name
        for name in _REQUIRED_TRAJECTORY_COLUMNS
        if name not in df_trajectory.columns
    ]
    if missing:
        raise ValueError(f"trajectory frame is missing columns: {missing}")
    if df_trajectory.empty:
        raise ValueError("trajectory frame is empty")

    engine = get_aircraft_params(aircraft_type)
    engine_eta = float(engine["eta"])
    log_soot_ei_n = math.log10(float(engine["ei_n"]))

    times = pd.to_datetime(df_trajectory["time"], utc=True)
    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for position in range(len(df_trajectory)):
        record = df_trajectory.iloc[position]
        lat = float(record["lat"])
        lon = float(record["lon"])
        altitude_ft = float(record["baro_alt_ft"])
        if not all(math.isfinite(value) for value in (lat, lon, altitude_ft)):
            raise ValueError(
                f"row {position} has non-finite lat/lon/baro_alt_ft; clean the "
                "trajectory first"
            )
        timestamp = times.iloc[position]
        if pd.isna(timestamp):
            raise ValueError(f"row {position} has a missing timestamp")

        flight_level = altitude_ft / 100.0
        pressure_pa = fl_to_p(flight_level) * 100.0
        altitude_m = pressure_to_altitude_m(pressure_pa)
        offset_pa = HUMIDITY_OFFSET_HPA * 100.0

        met = interpolate_met_to_segment(ds_met, lat, lon, pressure_pa, timestamp)
        met_above = interpolate_met_to_segment(
            ds_met, lat, lon, pressure_pa - offset_pa, timestamp
        )
        met_below = interpolate_met_to_segment(
            ds_met, lat, lon, pressure_pa + offset_pa, timestamp
        )

        temperature_k = met["temperature"]
        rhi = specific_humidity_to_rhi(
            met["specific_humidity"], pressure_pa, temperature_k
        )
        rhi_above = specific_humidity_to_rhi(
            met_above["specific_humidity"],
            pressure_pa - offset_pa,
            met_above["temperature"],
        )
        rhi_below = specific_humidity_to_rhi(
            met_below["specific_humidity"],
            pressure_pa + offset_pa,
            met_below["temperature"],
        )
        rhi_grad = rhi_above - rhi_below

        tlm_k = solve_TLM(pressure_pa, engine_eta)
        if math.isfinite(temperature_k) and math.isfinite(rhi):
            rhw = max(rhi * e_i(temperature_k) / e_w(temperature_k), 1.0e-9)
            tlc_k = critical_T(rhw, tlm_k, G(pressure_pa, engine_eta))
            sac_forms = temperature_k < tlc_k
        else:
            tlc_k = float("nan")
            sac_forms = False

        delta_u = met_above["u_component_of_wind"] - met_below["u_component_of_wind"]
        delta_v = met_above["v_component_of_wind"] - met_below["v_component_of_wind"]
        altitude_above_m = pressure_to_altitude_m(pressure_pa - offset_pa)
        altitude_below_m = pressure_to_altitude_m(pressure_pa + offset_pa)
        delta_z = abs(altitude_below_m - altitude_above_m)
        wind_shear = math.hypot(delta_u, delta_v) / delta_z

        richardson_inputs = (
            met_above["temperature"],
            met_below["temperature"],
            met_above["u_component_of_wind"],
            met_below["u_component_of_wind"],
            met_above["v_component_of_wind"],
            met_below["v_component_of_wind"],
        )
        if all(math.isfinite(value) for value in richardson_inputs):
            richardson_number = compute_richardson_number(
                temp_above_k=met_above["temperature"],
                temp_below_k=met_below["temperature"],
                u_above=met_above["u_component_of_wind"],
                u_below=met_below["u_component_of_wind"],
                v_above=met_above["v_component_of_wind"],
                v_below=met_below["v_component_of_wind"],
                alt_above_m=altitude_above_m,
                alt_below_m=altitude_below_m,
                p_ambient_pa=pressure_pa,
            )
        else:
            # Neutral laminar fallback keeps the feature matrix NaN-free
            # when a bounding layer is outside the ERA5 domain.
            richardson_number = 1.0

        solar_zenith_deg = compute_solar_zenith_angle(
            timestamp.tz_convert("UTC").to_pydatetime(), lat, lon
        )

        cloud_cover = met["fraction_of_cloud_cover"]
        if math.isfinite(cloud_cover):
            cloud_ice = 1.0 if cloud_cover > CLOUD_ICE_COVER_THRESHOLD else 0.0
        else:
            cloud_ice = float("nan")

        local_solar_time = (
            float(timestamp.hour)
            + float(timestamp.minute) / 60.0
            + lon / 15.0
        ) % 24.0

        row = extract_features_row(
            rhi=rhi,
            rhi_above=rhi_above,
            rhi_below=rhi_below,
            rhi_grad=rhi_grad,
            temp_k=temperature_k,
            tlm_k=tlm_k,
            tlc_k=tlc_k,
            pressure_pa=pressure_pa,
            cloud_ice=cloud_ice,
            high_cloud_fraction=cloud_cover,
            vertical_velocity_omega=met["vertical_velocity"],
            vertical_wind_shear=wind_shear,
            dist_to_tropopause_m=altitude_m - TROPOPAUSE_ALTITUDE_M,
            lat=lat,
            lon=lon,
            month=float(timestamp.month),
            local_solar_time_hours=local_solar_time,
            richardson_number=richardson_number,
            solar_zenith_deg=solar_zenith_deg,
            engine_efficiency_eta=engine_eta,
            log_soot_ei_n=log_soot_ei_n,
        )
        vectors.append(feature_vector(row))
        metadata.append(
            {
                "row_index": position,
                "icao24": str(record["icao24"]) if "icao24" in df_trajectory else "",
                "callsign": (
                    str(record["callsign"]) if "callsign" in df_trajectory else ""
                ),
                "time_utc": timestamp.isoformat(),
                "lat": lat,
                "lon": lon,
                "flight_level": flight_level,
                "pressure_pa": pressure_pa,
                "altitude_m": altitude_m,
                "temperature_k": temperature_k,
                "rhi": rhi,
                "rhi_above": rhi_above,
                "rhi_below": rhi_below,
                "tlm_c": tlm_k - _KELVIN_OFFSET,
                "tlc_c": tlc_k - _KELVIN_OFFSET,
                "richardson_number": richardson_number,
                "solar_zenith_deg": solar_zenith_deg,
                "engine_efficiency_eta": engine_eta,
                "log_soot_ei_n": log_soot_ei_n,
                "sac_forms": bool(sac_forms),
                "ambient_ice_supersaturated": bool(rhi > 1.0),
            }
        )

    return np.vstack(vectors), metadata
