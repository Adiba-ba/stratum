#!/usr/bin/env python
"""Demo: counterfactual decision layer (Project STRATUM, Step 6).

Simulates a flagged Delhi-Mumbai (VIDP-VABB) flight at FL350 with a
calibrated contrail persistence probability of 0.78. Evaluates the two
primary counterfactual flight levels (FL370 step climb, FL330 descent)
and prints an auditable decision card with fuel, CO2, avoided contrail
energy forcing, final status, and a dispatcher recommendation.

Exit code is 0 when the scenario yields an eligible recommendation.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratum.decide import (  # noqa: E402
    AltitudeDecision,
    CandidateEvaluation,
    evaluate_altitude_candidates,
)

FLIGHT_LABEL = "VIDP (Delhi) -> VABB (Mumbai)"
AIRCRAFT_TYPE = "A320"
BASELINE_FL = 350.0
BASELINE_P_PERSIST = 0.78


def _format_candidate(candidate: CandidateEvaluation) -> str:
    """Render one auditable candidate row."""
    label = f"FL{candidate['target_fl']:.0f}"
    change = f"{candidate['direction']} {candidate['altitude_change_ft']:+.0f} ft"
    if candidate["rejected"]:
        return f"{label}  {change:<18}: REJECT ({candidate['rejection_reason']})"
    verdict = "ELIGIBLE" if candidate["net_benefit_positive"] else "REJECT"
    avoided = candidate["avoided_ef_j"]
    return (
        f"{label}  {change:<18}: fuel +{candidate['delta_fuel_kg']:.2f} kg | "
        f"CO2 +{candidate['delta_co2_kg']:.2f} kg | "
        f"avoided warming {avoided:.2e} J ({avoided / 1.0e9:.2f} GJ) | "
        f"benefit/cost {candidate['benefit_to_cost_ratio']:.2e} | {verdict}"
    )


def _dispatcher_recommendation(decision: AltitudeDecision) -> str:
    """Human-readable recommendation for the dispatcher."""
    baseline_fl = decision["baseline_fl"]
    status = decision["status"]
    if status == "abstain":
        return (
            f"No action at FL{baseline_fl:.0f}: persistence probability "
            f"{decision['baseline_p_persist']:.2f} is below the 0.40 activation "
            "threshold, so extra fuel is not justified."
        )
    if status == "no_robust_candidate":
        return (
            f"Keep FL{baseline_fl:.0f}: no counterfactual flight level passes the "
            "benefit test under the current constraints."
        )

    recommended = decision["recommended_fl"]
    delta_fuel = decision["delta_fuel_kg"]
    delta_co2 = decision["delta_co2_kg"]
    avoided = decision["avoided_contrail_ef_j"]
    if recommended is None or delta_fuel is None or delta_co2 is None or avoided is None:
        raise RuntimeError("eligible decision is missing recommendation fields")
    direction = "step climb" if recommended > baseline_fl else "descent"
    return (
        f"Recommend {direction} from FL{baseline_fl:.0f} to FL{recommended:.0f} for the "
        f"flagged segment: avoids {avoided:.2e} J of contrail energy forcing for an "
        f"extra {delta_fuel:.1f} kg of fuel ({delta_co2:.1f} kg CO2). Coordinate with "
        "ATC and confirm fuel reserves before the level change."
    )


def main() -> int:
    decision = evaluate_altitude_candidates(
        baseline_p_persist=BASELINE_P_PERSIST,
        baseline_fl=BASELINE_FL,
        aircraft_type=AIRCRAFT_TYPE,
    )

    print("STRATUM Step 6 - Counterfactual Decision Layer demo")
    print("=" * 66)
    print(f"Flight: {FLIGHT_LABEL}, aircraft {AIRCRAFT_TYPE}")
    print(
        f"Flagged segment at FL{BASELINE_FL:.0f} with calibrated "
        f"P(persist) = {BASELINE_P_PERSIST:.2f} (activation threshold 0.40)"
    )
    print()
    print("Counterfactual candidates")
    print("-" * 66)
    for candidate in decision["candidates"]:
        print(_format_candidate(candidate))
    print()
    print("Decision card")
    print("-" * 66)
    print(f"Baseline flight level:      FL{decision['baseline_fl']:.0f}")
    print(f"Calibrated P(persist):      {decision['baseline_p_persist']:.2f}")
    recommended = decision["recommended_fl"]
    if recommended is not None:
        delta_fl = recommended - decision["baseline_fl"]
        direction = "step climb" if delta_fl > 0.0 else "descent"
        print(f"Recommended flight level:   FL{recommended:.0f} ({direction})")
        print(f"Delta fuel:                 +{decision['delta_fuel_kg']:.2f} kg")
        print(f"Delta CO2:                  +{decision['delta_co2_kg']:.2f} kg")
        avoided = decision["avoided_contrail_ef_j"]
        print(f"Avoided warming:            {avoided:.2e} J ({avoided / 1.0e9:.2f} GJ)")
        print(f"Benefit-to-cost ratio:      {decision['benefit_to_cost_ratio']:.2e}")
    else:
        print("Recommended flight level:   none")
    print(f"Final status:               {decision['status'].upper()}")
    print(f"Reason: {decision['reason']}")
    print()
    print(f"Dispatcher recommendation: {_dispatcher_recommendation(decision)}")
    return 0 if decision["status"] == "eligible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
