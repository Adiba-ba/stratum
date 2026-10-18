"""Thermodynamic baselines for Project STRATUM.

Implements the Murphy & Koop (2005) saturation vapor pressure
formulations for liquid water and ice, the exact analytic temperature
derivative of the liquid saturation curve, and ISA conversion from
flight level to ambient pressure.

References
----------
Murphy, D. M., & Koop, T. (2005). Review of the vapour pressures of ice
and supercooled water for atmospheric applications. Quarterly Journal of
the Royal Meteorological Society, 131(608), 1539-1565.

International Civil Aviation Organization (1993). Manual of the ICAO
Standard Atmosphere, 3rd ed., Doc 7488.
"""

from __future__ import annotations

import math

__all__ = ["e_w", "e_i", "d_e_w_dT", "fl_to_p"]

# ---------------------------------------------------------------------------
# International Standard Atmosphere (ISA) constants
# ---------------------------------------------------------------------------

P0_PA: float = 101325.0  # sea-level pressure [Pa]
T0_K: float = 288.15  # sea-level temperature [K]
LAPSE_K_PER_M: float = 0.0065  # tropospheric lapse rate [K/m]
G0_M_PER_S2: float = 9.80665  # standard gravity [m/s^2]
R_UNIVERSAL_J_PER_MOL_K: float = 8.31446  # universal gas constant [J/(mol*K)]
M_AIR_KG_PER_MOL: float = 0.0289644  # molar mass of dry air [kg/mol]
R_AIR_J_PER_KG_K: float = R_UNIVERSAL_J_PER_MOL_K / M_AIR_KG_PER_MOL

H_TROPOPAUSE_M: float = 11000.0  # ISA tropopause altitude [m]
T_TROPOPAUSE_K: float = 216.65  # ISA tropopause temperature [K]

FT_TO_M: float = 0.3048  # feet to metres


def _ln_e_w_and_slope(T: float) -> tuple[float, float]:
    """Return ``(ln e_w, d ln e_w / dT)`` for liquid water at ``T`` [K]."""
    tanh_term = math.tanh(0.0415 * (T - 218.8))
    inner = 53.878 - 1331.22 / T - 9.44523 * math.log(T) + 0.014025 * T
    d_inner = 1331.22 / T**2 - 9.44523 / T + 0.014025
    d_tanh = 0.0415 * (1.0 - tanh_term**2)

    ln_e = (
        54.842763
        - 6763.22 / T
        - 4.210 * math.log(T)
        + 0.000367 * T
        + tanh_term * inner
    )
    d_ln_e = (
        6763.22 / T**2
        - 4.210 / T
        + 0.000367
        + d_tanh * inner
        + tanh_term * d_inner
    )
    return ln_e, d_ln_e


def e_w(T: float) -> float:
    """Saturation vapor pressure over liquid water [Pa] at ``T`` [K].

    Implements Murphy & Koop (2005), eq. (10).
    """
    ln_e, _ = _ln_e_w_and_slope(T)
    return math.exp(ln_e)


def d_e_w_dT(T: float) -> float:
    """Exact analytic derivative d(e_w)/dT [Pa/K] at ``T`` [K].

    Differentiates the Murphy & Koop (2005) liquid curve analytically;
    no finite differences are used.
    """
    ln_e, d_ln_e = _ln_e_w_and_slope(T)
    return math.exp(ln_e) * d_ln_e


def e_i(T: float) -> float:
    """Saturation vapor pressure over ice [Pa] at ``T`` [K].

    Implements Murphy & Koop (2005), eq. (7).
    """
    ln_e = 9.550426 - 5723.265 / T + 3.53068 * math.log(T) - 0.00728332 * T
    return math.exp(ln_e)


def fl_to_p(fl_hundred_ft: float) -> float:
    """Convert a flight level to ISA ambient pressure [hPa].

    ``fl_hundred_ft`` is the flight level in hundreds of feet (e.g. 250
    for FL250). Uses the ISA tropospheric lapse rate up to 11 km and the
    isothermal layer above it.
    """
    h_m = fl_hundred_ft * 100.0 * FT_TO_M
    exponent = G0_M_PER_S2 / (R_AIR_J_PER_KG_K * LAPSE_K_PER_M)

    if h_m <= H_TROPOPAUSE_M:
        T_K = T0_K - LAPSE_K_PER_M * h_m
        p_pa = P0_PA * (T_K / T0_K) ** exponent
    else:
        p_trop_pa = P0_PA * (T_TROPOPAUSE_K / T0_K) ** exponent
        p_pa = p_trop_pa * math.exp(
            -G0_M_PER_S2 * (h_m - H_TROPOPAUSE_M) / (R_AIR_J_PER_KG_K * T_TROPOPAUSE_K)
        )
    return p_pa / 100.0
