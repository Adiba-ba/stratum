"""Counterfactual altitude evaluation for contrail-avoidance decisions.

For a flagged flight segment with a calibrated persistence probability,
this module evaluates the two primary counterfactual flight levels
(baseline +/- 2,000 ft), rejects physically infeasible changes, prices the
extra fuel with :mod:`stratum.decide.fuel`, prices the avoided contrail
energy forcing with :mod:`stratum.decide.climate`, and returns an
auditable decision record.

Decision policy:

- ``abstain``: probability below the 0.40 activation threshold; extra
  fuel is not justified.
- ``eligible``: at least one candidate has a positive net climate
  benefit; the best benefit-to-cost ratio is recommended.
- ``no_robust_candidate``: nothing passes the benefit test.
"""

from __future__ import annotations

import math
from typing import TypedDict

from stratum.decide.climate import calculate_net_climate_benefit
from stratum.decide.fuel import AIRCRAFT_CRUISE_BURN_KG_S, estimate_fuel_penalty

__all__ = [
    "ABSTAIN_THRESHOLD",
    "MAX_ALLOWED_FL",
    "MIN_ALLOWED_FL",
    "ALTITUDE_STEP_FL",
    "REFERENCE_SEGMENT_EF_J",
    "CandidateEvaluation",
    "AltitudeDecision",
    "evaluate_altitude_candidates",
]

#: Below this calibrated P(persist) the decision layer abstains.
ABSTAIN_THRESHOLD: float = 0.40

#: Standard ceiling and minimum cruise flight levels [hundreds of feet].
MAX_ALLOWED_FL: float = 390.0
MIN_ALLOWED_FL: float = 270.0

#: Primary alternative offset [hundreds of feet] (+/- 2,000 ft).
ALTITUDE_STEP_FL: float = 20.0

#: Reference contrail energy forcing of a flagged persistent segment [J].
#: Used to translate a persistence-probability reduction into an avoided
#: energy forcing; an audit constant, not a per-flight measurement.
REFERENCE_SEGMENT_EF_J: float = 1.0e10

#: Escape effectiveness of each direction (fraction of the flagged
#: segment's contrail energy forcing removed) when the layer is left.
_CLIMB_EFFECTIVENESS: float = 0.85
_DESCENT_EFFECTIVENESS: float = 0.80


class CandidateEvaluation(TypedDict):
    """Auditable evaluation record for one counterfactual flight level."""

    target_fl: float
    direction: str
    altitude_change_ft: float
    rejected: bool
    rejection_reason: str | None
    delta_fuel_kg: float
    delta_co2_kg: float
    penalty_pct: float
    avoided_ef_j: float
    benefit_to_cost_ratio: float
    net_benefit_positive: bool


class AltitudeDecision(TypedDict):
    """Decision record returned by :func:`evaluate_altitude_candidates`."""

    status: str
    baseline_fl: float
    baseline_p_persist: float
    recommended_fl: float | None
    delta_fuel_kg: float | None
    delta_co2_kg: float | None
    avoided_contrail_ef_j: float | None
    benefit_to_cost_ratio: float | None
    candidates: list[CandidateEvaluation]
    reason: str


def _evaluate_candidate(
    baseline_p_persist: float,
    baseline_fl: float,
    delta_fl: float,
    aircraft_type: str,
) -> CandidateEvaluation:
    """Evaluate one +/- ``ALTITUDE_STEP_FL`` candidate against all gates."""
    target_fl = baseline_fl + delta_fl
    change_ft = delta_fl * 100.0
    direction = "step climb" if delta_fl > 0.0 else "descent"

    rejection_reason: str | None = None
    if target_fl > MAX_ALLOWED_FL:
        rejection_reason = (
            f"target FL{target_fl:.0f} exceeds standard ceiling FL{MAX_ALLOWED_FL:.0f}"
        )
    elif target_fl < MIN_ALLOWED_FL:
        rejection_reason = (
            f"target FL{target_fl:.0f} below minimum cruise FL{MIN_ALLOWED_FL:.0f}"
        )
    if rejection_reason is not None:
        return CandidateEvaluation(
            target_fl=target_fl,
            direction=direction,
            altitude_change_ft=change_ft,
            rejected=True,
            rejection_reason=rejection_reason,
            delta_fuel_kg=0.0,
            delta_co2_kg=0.0,
            penalty_pct=0.0,
            avoided_ef_j=0.0,
            benefit_to_cost_ratio=0.0,
            net_benefit_positive=False,
        )

    penalty = estimate_fuel_penalty(aircraft_type, baseline_fl, target_fl)
    effectiveness = (
        _CLIMB_EFFECTIVENESS if delta_fl > 0.0 else _DESCENT_EFFECTIVENESS
    )
    avoided_ef_j = baseline_p_persist * effectiveness * REFERENCE_SEGMENT_EF_J
    benefit = calculate_net_climate_benefit(avoided_ef_j, penalty["delta_co2_kg"])
    return CandidateEvaluation(
        target_fl=target_fl,
        direction=direction,
        altitude_change_ft=change_ft,
        rejected=False,
        rejection_reason=None,
        delta_fuel_kg=penalty["delta_fuel_kg"],
        delta_co2_kg=penalty["delta_co2_kg"],
        penalty_pct=penalty["penalty_pct"],
        avoided_ef_j=avoided_ef_j,
        benefit_to_cost_ratio=float(benefit["benefit_to_cost_ratio"]),
        net_benefit_positive=bool(benefit["net_benefit_positive"]),
    )


def evaluate_altitude_candidates(
    baseline_p_persist: float,
    baseline_fl: float,
    aircraft_type: str = "A320",
) -> AltitudeDecision:
    """Evaluate the primary counterfactual flight levels for a segment.

    Parameters
    ----------
    baseline_p_persist : float
        Calibrated probability of contrail persistence for the flagged
        segment, in ``[0.0, 1.0]``.
    baseline_fl : float
        Current flight level, in hundreds of feet (FL350 -> ``350.0``).
    aircraft_type : str, optional
        Aircraft type for the fuel penalty model, by default ``"A320"``.

    Returns
    -------
    AltitudeDecision
        Decision record with ``status`` one of ``"abstain"``,
        ``"eligible"``, ``"no_robust_candidate"``, the baseline state,
        the recommendation (when eligible), and the per-candidate
        evaluation records.

    Raises
    ------
    ValueError
        For a probability outside ``[0.0, 1.0]``, a non-positive baseline
        flight level, or an unsupported aircraft type.
    """
    p_persist = float(baseline_p_persist)
    if not math.isfinite(p_persist) or not 0.0 <= p_persist <= 1.0:
        raise ValueError(
            f"baseline_p_persist must be in [0.0, 1.0], got {baseline_p_persist!r}"
        )
    if baseline_fl <= 0.0:
        raise ValueError(f"baseline_fl must be positive, got {baseline_fl!r}")
    key = aircraft_type.strip().upper()
    if key not in AIRCRAFT_CRUISE_BURN_KG_S:
        supported = ", ".join(sorted(AIRCRAFT_CRUISE_BURN_KG_S))
        raise ValueError(
            f"unsupported aircraft_type {aircraft_type!r}; expected one of {supported}"
        )

    if p_persist < ABSTAIN_THRESHOLD:
        return AltitudeDecision(
            status="abstain",
            baseline_fl=baseline_fl,
            baseline_p_persist=p_persist,
            recommended_fl=None,
            delta_fuel_kg=None,
            delta_co2_kg=None,
            avoided_contrail_ef_j=None,
            benefit_to_cost_ratio=None,
            candidates=[],
            reason=(
                f"P(persist)={p_persist:.2f} is below the "
                f"{ABSTAIN_THRESHOLD:.2f} activation threshold; extra fuel "
                "is not justified."
            ),
        )

    candidates = [
        _evaluate_candidate(p_persist, baseline_fl, +ALTITUDE_STEP_FL, aircraft_type),
        _evaluate_candidate(p_persist, baseline_fl, -ALTITUDE_STEP_FL, aircraft_type),
    ]
    eligible = [
        candidate
        for candidate in candidates
        if not candidate["rejected"] and candidate["net_benefit_positive"]
    ]
    if not eligible:
        return AltitudeDecision(
            status="no_robust_candidate",
            baseline_fl=baseline_fl,
            baseline_p_persist=p_persist,
            recommended_fl=None,
            delta_fuel_kg=None,
            delta_co2_kg=None,
            avoided_contrail_ef_j=None,
            benefit_to_cost_ratio=None,
            candidates=candidates,
            reason="No counterfactual flight level passes the net-benefit test.",
        )

    best = max(eligible, key=lambda candidate: candidate["benefit_to_cost_ratio"])
    return AltitudeDecision(
        status="eligible",
        baseline_fl=baseline_fl,
        baseline_p_persist=p_persist,
        recommended_fl=best["target_fl"],
        delta_fuel_kg=best["delta_fuel_kg"],
        delta_co2_kg=best["delta_co2_kg"],
        avoided_contrail_ef_j=best["avoided_ef_j"],
        benefit_to_cost_ratio=best["benefit_to_cost_ratio"],
        candidates=candidates,
        reason=(
            f"FL{best['target_fl']:.0f} ({best['direction']}) passes the "
            "net-benefit test with the best benefit-to-cost ratio."
        ),
    )
