"""Decision-layer subpackage for Project STRATUM.

Re-exports the fuel-penalty model, the climate-benefit accounting, and the
counterfactual altitude evaluator.
"""

from stratum.decide.climate import (
    AGWP_CO2_20YR_W_YR_PER_M2_PER_KG,
    AGWP_CO2_100YR_W_YR_PER_M2_PER_KG,
    CO2_AGWP_W_YR_PER_M2_PER_KG_20YR,
    EARTH_SURFACE_AREA_M2,
    MAX_METRIC_HORIZON_YEARS,
    MIN_METRIC_HORIZON_YEARS,
    SECONDS_PER_YEAR,
    calculate_net_climate_benefit,
    co2_agwp_w_yr_per_m2_per_kg,
)
from stratum.decide.counterfactual import (
    DEFAULT_SEGMENT_LENGTH_KM,
    MEDIAN_PERSISTENT_EF_J_PER_M,
    AltitudeDecision,
    CandidateEvaluation,
    evaluate_altitude_candidates,
)
from stratum.decide.fuel import CO2_PER_KG_FUEL, estimate_fuel_penalty

__all__ = [
    "CO2_PER_KG_FUEL",
    "estimate_fuel_penalty",
    "AGWP_CO2_20YR_W_YR_PER_M2_PER_KG",
    "AGWP_CO2_100YR_W_YR_PER_M2_PER_KG",
    "CO2_AGWP_W_YR_PER_M2_PER_KG_20YR",
    "EARTH_SURFACE_AREA_M2",
    "MAX_METRIC_HORIZON_YEARS",
    "MIN_METRIC_HORIZON_YEARS",
    "SECONDS_PER_YEAR",
    "calculate_net_climate_benefit",
    "co2_agwp_w_yr_per_m2_per_kg",
    "DEFAULT_SEGMENT_LENGTH_KM",
    "MEDIAN_PERSISTENT_EF_J_PER_M",
    "AltitudeDecision",
    "CandidateEvaluation",
    "evaluate_altitude_candidates",
]
