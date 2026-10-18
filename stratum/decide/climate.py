"""Climate-benefit accounting for counterfactual altitude changes.

Contrail energy forcing (EF) is reported in Joules. To compare it against
the CO2 penalty of an extra-fuel maneuver, both sides are expressed as an
Earth-mean integrated radiative impact [J/m^2]:

- the avoided contrail term is ``EF / Earth surface area``;
- the CO2 term uses the Section 14.4 proxy that 1 kg CO2 carries an
  integrated radiative impact of ~3.5e-14 W/m^2 on a 20-year horizon,
  scaled linearly with the metric horizon (100-year reporting from the
  same proxy).

The linear horizon scaling is a documented proxy, not a full impulse
response model.
"""

from __future__ import annotations

import math

__all__ = [
    "CO2_INTEGRATED_FORCING_PER_KG_20YR",
    "EARTH_SURFACE_AREA_M2",
    "calculate_net_climate_benefit",
]

#: Integrated radiative impact proxy per kg CO2 on a 20-year horizon
#: [W/m^2 per kg] (Section 14.4).
CO2_INTEGRATED_FORCING_PER_KG_20YR: float = 3.5e-14

#: Earth surface area [m^2] used to convert contrail energy to a flux.
EARTH_SURFACE_AREA_M2: float = 5.10072e14

_BASE_HORIZON_YEARS: float = 20.0


def calculate_net_climate_benefit(
    avoided_contrail_ef_j: float,
    delta_co2_kg: float,
    metric_horizon_years: int = 20,
) -> dict[str, float | bool]:
    """Compare avoided contrail warming against the CO2 cost of a maneuver.

    Parameters
    ----------
    avoided_contrail_ef_j : float
        Contrail energy forcing avoided by the maneuver [J], >= 0.
    delta_co2_kg : float
        Extra CO2 emitted by the maneuver [kg], >= 0.
    metric_horizon_years : int, optional
        Reporting horizon for the CO2 impact, by default 20. The 20-year
        proxy from Section 14.4 is scaled linearly with the horizon
        (e.g. 5x at 100 years).

    Returns
    -------
    dict[str, float | bool]
        - ``avoided_ef_j``: echoed avoided energy forcing [J].
        - ``delta_co2_kg``: echoed CO2 penalty [kg].
        - ``net_benefit_positive``: True when the avoided contrail flux
          exceeds the CO2 flux over the metric horizon.
        - ``benefit_to_cost_ratio``: avoided contrail flux divided by the
          CO2 flux; ``inf`` when the CO2 cost is zero and the benefit is
          positive, ``0.0`` when both sides are zero.

    Raises
    ------
    ValueError
        For negative energy or mass inputs, or a non-positive horizon.
    """
    if avoided_contrail_ef_j < 0.0:
        raise ValueError(
            f"avoided_contrail_ef_j must be non-negative, got {avoided_contrail_ef_j!r}"
        )
    if delta_co2_kg < 0.0:
        raise ValueError(f"delta_co2_kg must be non-negative, got {delta_co2_kg!r}")
    if metric_horizon_years <= 0:
        raise ValueError(
            f"metric_horizon_years must be positive, got {metric_horizon_years!r}"
        )

    horizon_scale = metric_horizon_years / _BASE_HORIZON_YEARS
    avoided_flux = avoided_contrail_ef_j / EARTH_SURFACE_AREA_M2
    co2_flux = delta_co2_kg * CO2_INTEGRATED_FORCING_PER_KG_20YR * horizon_scale

    if co2_flux > 0.0:
        ratio = avoided_flux / co2_flux
    elif avoided_flux > 0.0:
        ratio = math.inf
    else:
        ratio = 0.0

    return {
        "avoided_ef_j": float(avoided_contrail_ef_j),
        "delta_co2_kg": float(delta_co2_kg),
        "net_benefit_positive": bool(avoided_flux > co2_flux),
        "benefit_to_cost_ratio": float(ratio),
    }
