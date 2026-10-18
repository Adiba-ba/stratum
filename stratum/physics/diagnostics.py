"""Atmospheric diagnostics and aircraft parameter lookups."""

from __future__ import annotations

import math
from datetime import datetime
import numpy as np

__all__ = [
    "AIRCRAFT_ENGINE_DATABASE",
    "get_aircraft_params",
    "compute_richardson_number",
    "compute_solar_zenith_angle",
]

# Gas constants and thermal parameters
G = 9.80665              # Gravitational acceleration [m/s^2]
R_D = 287.05             # Specific gas constant for dry air [J/(kg K)]
C_P = 1004.0             # Specific heat capacity at constant pressure [J/(kg K)]
KAPPA = R_D / C_P        # 2/7 ≈ 0.286

# Certified default parameters by aircraft ICAO designator
# nvPM number emissions index (EI_n) [particles / kg fuel] and cruise efficiency (eta)
AIRCRAFT_ENGINE_DATABASE: dict[str, dict[str, float]] = {
    "A320": {"eta": 0.32, "ei_n": 1.5e15, "q_fuel": 43.1e6},
    "A20N": {"eta": 0.38, "ei_n": 4.0e14, "q_fuel": 43.2e6},  # A320neo (LEAP-1A)
    "B738": {"eta": 0.31, "ei_n": 1.8e15, "q_fuel": 43.1e6},  # 737-800 (CFM56-7B)
    "B38M": {"eta": 0.37, "ei_n": 5.0e14, "q_fuel": 43.2e6},  # 737 MAX 8
    "B77W": {"eta": 0.35, "ei_n": 8.0e14, "q_fuel": 43.0e6},  # 777-300ER (GE90)
    "DEFAULT": {"eta": 0.33, "ei_n": 1.0e15, "q_fuel": 43.1e6},
}


def get_aircraft_params(aircraft_type: str) -> dict[str, float]:
    """Retrieve certified propulsion efficiency and soot emission parameters."""
    return AIRCRAFT_ENGINE_DATABASE.get(
        aircraft_type.upper(), AIRCRAFT_ENGINE_DATABASE["DEFAULT"]
    )


def compute_richardson_number(
    temp_above_k: float,
    temp_below_k: float,
    u_above: float,
    u_below: float,
    v_above: float,
    v_below: float,
    alt_above_m: float,
    alt_below_m: float,
    p_ambient_pa: float,
) -> float:
    """Calculate the Gradient Richardson Number (Ri) between adjacent flight levels.

    Ri = (g / theta) * (d_theta / dz) / ((du/dz)^2 + (dv/dz)^2)
    Ri < 0.25 indicates dynamic instability/turbulence that shears contrails apart.
    """
    dz = alt_above_m - alt_below_m
    if abs(dz) < 1.0:
        return 1.0  # Stable fallback if layers collapse

    # Potential temperature: theta = T * (100000 / p)^kappa
    theta_above = temp_above_k * (100000.0 / (p_ambient_pa - 2500.0)) ** KAPPA
    theta_below = temp_below_k * (100000.0 / (p_ambient_pa + 2500.0)) ** KAPPA
    theta_mid = 0.5 * (theta_above + theta_below)

    d_theta_dz = (theta_above - theta_below) / dz
    du_dz = (u_above - u_below) / dz
    dv_dz = (v_above - v_below) / dz

    shear_squared = du_dz**2 + dv_dz**2
    if shear_squared < 1e-8:
        return 10.0  # High stability, virtually zero mechanical shear

    ri = (G / theta_mid) * d_theta_dz / shear_squared
    # Clamp Ri to avoid numerical overflow in extreme gradients
    return float(np.clip(ri, -5.0, 10.0))


def compute_solar_zenith_angle(dt_utc: datetime, lat_deg: float, lon_deg: float) -> float:
    """Calculate the solar zenith angle in degrees (0° = directly overhead, >90° = night)."""
    # Day of year and fractional hour
    day_of_year = dt_utc.timetuple().tm_yday
    hour_utc = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

    # Solar declination approximation (radians)
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour_utc - 12.0) / 24.0)
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
    )

    # Equation of time in minutes
    eq_time_min = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # Solar hour angle (radians)
    time_offset = eq_time_min + 4.0 * lon_deg
    solar_time_min = (hour_utc * 60.0 + time_offset) % 1440.0
    hour_angle = math.radians((solar_time_min / 4.0) - 180.0)

    lat_rad = math.radians(lat_deg)
    cos_zenith = math.sin(lat_rad) * math.sin(declination) + math.cos(
        lat_rad
    ) * math.cos(declination) * math.cos(hour_angle)

    zenith_rad = math.acos(np.clip(cos_zenith, -1.0, 1.0))
    return math.degrees(zenith_rad)
