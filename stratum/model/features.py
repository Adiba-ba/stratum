"""Frozen feature schema and row assembly for the STRATUM correction model.

The canonical order in ``FEATURE_NAMES`` is the model contract: engine
microphysics (propulsion efficiency, soot number emission index),
atmospheric turbulence (gradient Richardson number), and radiative
geometry (solar zenith angle) extend the previous physics/time block. The
order must not change without retraining. Helper :func:`encode_cyclical`
remains available for callers that build their own encodings.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "encode_cyclical",
    "extract_features_row",
    "feature_vector",
]

#: Canonical feature order for the correction model (Section 12.2).
FEATURE_NAMES: tuple[str, ...] = (
    # Relative humidity over ice: point value and vertical structure.
    "rhi",
    "rhi_above",
    "rhi_below",
    "rhi_grad",
    # Temperature and pressure state.
    "temp_k",
    "tlm_k",
    "tlc_k",
    "pressure_pa",
    # Cloud and dynamics.
    "cloud_ice",
    "high_cloud_fraction",
    "vertical_velocity_omega",
    "vertical_wind_shear",
    # Vertical distance to the tropopause and horizontal position.
    "dist_to_tropopause_m",
    "lat",
    "lon",
    # Time.
    "month",
    "local_solar_time_hours",
    # Atmospheric diagnostics.
    "richardson_number",
    "solar_zenith_deg",
    # Engine microphysics.
    "engine_efficiency_eta",
    "log_soot_ei_n",
)

#: Neutral defaults for the diagnostic/engine features.
DEFAULT_RICHARDSON_NUMBER: float = 1.0
DEFAULT_SOLAR_ZENITH_DEG: float = 100.0
DEFAULT_ENGINE_EFFICIENCY_ETA: float = 0.32
DEFAULT_LOG_SOOT_EI_N: float = 15.0


def encode_cyclical(val: float, period: float) -> tuple[float, float]:
    """Encode a cyclical value as a unit-circle ``(sin, cos)`` pair.

    ``val`` is mapped onto the angle ``2 * pi * val / period`` so that
    values one full period apart coincide and the encoding is continuous
    across the wrap-around (e.g. month 12 and month 1).
    """
    if period <= 0.0:
        raise ValueError(f"period must be positive, got {period!r}")
    angle = 2.0 * math.pi * val / period
    return math.sin(angle), math.cos(angle)


def extract_features_row(
    *,
    rhi: float,
    rhi_above: float,
    rhi_below: float,
    rhi_grad: float,
    temp_k: float,
    tlm_k: float,
    tlc_k: float,
    pressure_pa: float,
    cloud_ice: float,
    high_cloud_fraction: float,
    vertical_velocity_omega: float,
    vertical_wind_shear: float,
    dist_to_tropopause_m: float,
    lat: float,
    lon: float,
    month: float,
    local_solar_time_hours: float,
    richardson_number: float = DEFAULT_RICHARDSON_NUMBER,
    solar_zenith_deg: float = DEFAULT_SOLAR_ZENITH_DEG,
    engine_efficiency_eta: float = DEFAULT_ENGINE_EFFICIENCY_ETA,
    log_soot_ei_n: float = DEFAULT_LOG_SOOT_EI_N,
) -> dict[str, float]:
    """Assemble one sample as a feature dictionary in canonical order.

    ``tlm_k`` and ``tlc_k`` are the Schmidt-Appleman threshold
    temperatures [K]. ``month`` (1-12) and ``local_solar_time_hours``
    (0-24) enter as raw values; ``richardson_number`` (gradient Ri),
    ``solar_zenith_deg`` (0-180), ``engine_efficiency_eta`` (0-1) and
    ``log_soot_ei_n`` (log10 of the nvPM number emission index) default to
    the neutral values used when a diagnostic is unavailable.

    Returns
    -------
    dict[str, float]
        Mapping keyed by :data:`FEATURE_NAMES`, in canonical order.
    """
    return {
        "rhi": float(rhi),
        "rhi_above": float(rhi_above),
        "rhi_below": float(rhi_below),
        "rhi_grad": float(rhi_grad),
        "temp_k": float(temp_k),
        "tlm_k": float(tlm_k),
        "tlc_k": float(tlc_k),
        "pressure_pa": float(pressure_pa),
        "cloud_ice": float(cloud_ice),
        "high_cloud_fraction": float(high_cloud_fraction),
        "vertical_velocity_omega": float(vertical_velocity_omega),
        "vertical_wind_shear": float(vertical_wind_shear),
        "dist_to_tropopause_m": float(dist_to_tropopause_m),
        "lat": float(lat),
        "lon": float(lon),
        "month": float(month),
        "local_solar_time_hours": float(local_solar_time_hours),
        "richardson_number": float(richardson_number),
        "solar_zenith_deg": float(solar_zenith_deg),
        "engine_efficiency_eta": float(engine_efficiency_eta),
        "log_soot_ei_n": float(log_soot_ei_n),
    }


def feature_vector(row: Mapping[str, float]) -> np.ndarray:
    """Convert a feature dictionary into a vector in canonical order.

    Raises
    ------
    KeyError
        If ``row`` misses any feature from :data:`FEATURE_NAMES`.
    """
    missing = [name for name in FEATURE_NAMES if name not in row]
    if missing:
        raise KeyError(f"row is missing features: {missing}")
    return np.array([float(row[name]) for name in FEATURE_NAMES], dtype=float)
