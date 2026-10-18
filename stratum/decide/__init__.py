"""Decision-layer subpackage for Project STRATUM.

Re-exports the fuel-penalty model, the climate-benefit accounting, and the
counterfactual altitude evaluator.
"""

from stratum.decide.climate import (
    CO2_INTEGRATED_FORCING_PER_KG_20YR,
    calculate_net_climate_benefit,
)
from stratum.decide.counterfactual import (
    AltitudeDecision,
    CandidateEvaluation,
    evaluate_altitude_candidates,
)
from stratum.decide.fuel import CO2_PER_KG_FUEL, estimate_fuel_penalty

__all__ = [
    "CO2_PER_KG_FUEL",
    "estimate_fuel_penalty",
    "CO2_INTEGRATED_FORCING_PER_KG_20YR",
    "calculate_net_climate_benefit",
    "AltitudeDecision",
    "CandidateEvaluation",
    "evaluate_altitude_candidates",
]
