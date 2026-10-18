"""Climate-benefit accounting for counterfactual altitude changes.

Contrail energy forcing (EF) is reported in Joules. To compare it against
the CO2 penalty of an extra-fuel maneuver, both sides are expressed as an
Earth-mean integrated radiative impact [J/m^2]:

- the avoided contrail term is ``EF / Earth surface area``;
- the CO2 term takes the absolute global warming potential (AGWP) of CO2
  [W*yr/m^2 per kg], converts it to J/m^2 with :data:`SECONDS_PER_YEAR`,
  and evaluates it at the requested metric horizon.

The AGWP endpoints are the Teoh et al. (2024) contrail-inventory values
for the 20-year and 100-year horizons (2.39e-14 and 8.80e-14
W*yr/m^2 per kg; the same numbers adopted in IPCC AR6). Horizons between
20 and 100 years are obtained by linear interpolation between the two
endpoints. This interpolation is a documented approximation: the true
AGWP curve is a smooth impulse-response integral, not a straight line,
so an intermediate horizon can be off by a few percent of the curve
value. Horizons outside [20, 100] raise :class:`ValueError` because no
cited endpoint bounds the extrapolation.

References
----------
Teoh, R., Engberg, Z., Shapiro, M., Dray, L., & Stettler, M. E. J.
(2024). Global aviation contrail climate effects from 2019 to 2021.
Atmospheric Chemistry and Physics, 24, 6071-6093.
https://doi.org/10.5194/acp-24-6071-2024
"""

from __future__ import annotations

import math

__all__ = [
    "AGWP_CO2_20YR_W_YR_PER_M2_PER_KG",
    "AGWP_CO2_100YR_W_YR_PER_M2_PER_KG",
    "CO2_AGWP_W_YR_PER_M2_PER_KG_20YR",
    "MAX_METRIC_HORIZON_YEARS",
    "MIN_METRIC_HORIZON_YEARS",
    "EARTH_SURFACE_AREA_M2",
    "SECONDS_PER_YEAR",
    "co2_agwp_w_yr_per_m2_per_kg",
    "calculate_net_climate_benefit",
]

#: Integrated radiative forcing (AGWP) per kg CO2 over a 20-year horizon
#: [W*yr/m^2 per kg] (Teoh et al. 2024, ACP 24, 6071-6093; IPCC AR6).
AGWP_CO2_20YR_W_YR_PER_M2_PER_KG: float = 2.39e-14

#: Integrated radiative forcing (AGWP) per kg CO2 over a 100-year
#: horizon [W*yr/m^2 per kg] (Teoh et al. 2024, ACP 24, 6071-6093; IPCC
#: AR6).
AGWP_CO2_100YR_W_YR_PER_M2_PER_KG: float = 8.80e-14

#: Backwards-compatible alias of the 20-year AGWP.
CO2_AGWP_W_YR_PER_M2_PER_KG_20YR: float = AGWP_CO2_20YR_W_YR_PER_M2_PER_KG

#: Shortest metric horizon with a cited AGWP endpoint [yr].
MIN_METRIC_HORIZON_YEARS: int = 20

#: Longest metric horizon with a cited AGWP endpoint [yr].
MAX_METRIC_HORIZON_YEARS: int = 100

#: Julian year [s/yr], used to convert the AGWP unit W*yr/m^2 to J/m^2.
SECONDS_PER_YEAR: float = 3.15576e7

#: Earth surface area [m^2] used to convert contrail energy to a flux.
EARTH_SURFACE_AREA_M2: float = 5.10072e14


def co2_agwp_w_yr_per_m2_per_kg(metric_horizon_years: float) -> float:
    """AGWP of CO2 at ``metric_horizon_years`` [W*yr/m^2 per kg].

    The value is linearly interpolated between the cited 20-year and
    100-year endpoints of Teoh et al. (2024). The interpolation is a
    documented approximation, not a full impulse-response integration.

    Parameters
    ----------
    metric_horizon_years : float
        Metric horizon [yr], between :data:`MIN_METRIC_HORIZON_YEARS` and
        :data:`MAX_METRIC_HORIZON_YEARS` inclusive.

    Returns
    -------
    float
        Interpolated AGWP [W*yr/m^2 per kg].

    Raises
    ------
    ValueError
        When the horizon is non-finite or outside the cited range.
    """
    if not math.isfinite(metric_horizon_years):
        raise ValueError(
            f"metric_horizon_years must be finite, got {metric_horizon_years!r}"
        )
    if not (
        MIN_METRIC_HORIZON_YEARS
        <= metric_horizon_years
        <= MAX_METRIC_HORIZON_YEARS
    ):
        raise ValueError(
            f"metric_horizon_years must lie between "
            f"{MIN_METRIC_HORIZON_YEARS} and {MAX_METRIC_HORIZON_YEARS} years "
            f"(cited Teoh et al. 2024 AGWP endpoints), got "
            f"{metric_horizon_years!r}"
        )
    span = MAX_METRIC_HORIZON_YEARS - MIN_METRIC_HORIZON_YEARS
    fraction = (metric_horizon_years - MIN_METRIC_HORIZON_YEARS) / span
    return AGWP_CO2_20YR_W_YR_PER_M2_PER_KG + fraction * (
        AGWP_CO2_100YR_W_YR_PER_M2_PER_KG - AGWP_CO2_20YR_W_YR_PER_M2_PER_KG
    )


def calculate_net_climate_benefit(
    avoided_contrail_ef_j: float,
    delta_co2_kg: float,
    metric_horizon_years: int = 20,
) -> dict[str, float | bool]:
    """Compare avoided contrail warming against the CO2 cost of a maneuver.

    Both sides are computed as Earth-mean integrated radiative impacts
    [J/m^2]: the avoided contrail energy forcing [J] divided by the Earth
    surface area [m^2], and the CO2 mass [kg] times the AGWP of CO2 at
    ``metric_horizon_years`` [W*yr/m^2 per kg, Teoh et al. 2024] times
    :data:`SECONDS_PER_YEAR` [s/yr].

    Parameters
    ----------
    avoided_contrail_ef_j : float
        Contrail energy forcing avoided by the maneuver [J], >= 0.
    delta_co2_kg : float
        Extra CO2 emitted by the maneuver [kg], >= 0.
    metric_horizon_years : int, optional
        Reporting horizon for the CO2 impact, by default 20. Horizons
        from 20 to 100 years are supported; intermediate values use the
        linear AGWP interpolation of
        :func:`co2_agwp_w_yr_per_m2_per_kg`.

    Returns
    -------
    dict[str, float | bool]
        - ``avoided_ef_j``: echoed avoided energy forcing [J].
        - ``delta_co2_kg``: echoed CO2 penalty [kg].
        - ``avoided_flux_j_per_m2``: avoided contrail flux [J/m^2].
        - ``co2_flux_j_per_m2``: CO2 integrated impact [J/m^2] over the
          metric horizon.
        - ``metric_horizon_years``: echoed horizon [yr].
        - ``co2_agwp_w_yr_per_m2_per_kg``: AGWP used for the horizon
          [W*yr/m^2 per kg].
        - ``net_benefit_positive``: True when the avoided contrail flux
          exceeds the CO2 flux over the metric horizon.
        - ``benefit_to_cost_ratio``: avoided contrail flux divided by the
          CO2 flux; ``inf`` when the CO2 cost is zero and the benefit is
          positive, ``0.0`` when both sides are zero.

    Raises
    ------
    ValueError
        For negative energy or mass inputs, or a horizon outside
        ``[20, 100]`` years.
    """
    if avoided_contrail_ef_j < 0.0:
        raise ValueError(
            f"avoided_contrail_ef_j must be non-negative, got {avoided_contrail_ef_j!r}"
        )
    if delta_co2_kg < 0.0:
        raise ValueError(f"delta_co2_kg must be non-negative, got {delta_co2_kg!r}")

    agwp = co2_agwp_w_yr_per_m2_per_kg(metric_horizon_years)
    avoided_flux_j_per_m2 = avoided_contrail_ef_j / EARTH_SURFACE_AREA_M2
    co2_flux_j_per_m2 = delta_co2_kg * agwp * SECONDS_PER_YEAR

    if co2_flux_j_per_m2 > 0.0:
        ratio = avoided_flux_j_per_m2 / co2_flux_j_per_m2
    elif avoided_flux_j_per_m2 > 0.0:
        ratio = math.inf
    else:
        ratio = 0.0

    return {
        "avoided_ef_j": float(avoided_contrail_ef_j),
        "delta_co2_kg": float(delta_co2_kg),
        "avoided_flux_j_per_m2": float(avoided_flux_j_per_m2),
        "co2_flux_j_per_m2": float(co2_flux_j_per_m2),
        "metric_horizon_years": float(metric_horizon_years),
        "co2_agwp_w_yr_per_m2_per_kg": float(agwp),
        "net_benefit_positive": bool(avoided_flux_j_per_m2 > co2_flux_j_per_m2),
        "benefit_to_cost_ratio": float(ratio),
    }
