"""Tests for the counterfactual decision layer (Step 6)."""

from __future__ import annotations

import pytest

from stratum.decide import (
    CO2_PER_KG_FUEL,
    calculate_net_climate_benefit,
    estimate_fuel_penalty,
    evaluate_altitude_candidates,
)

WINDOW_SECONDS = 20.0 * 60.0
CRUISE_BURN_KG_S = 0.65


def test_co2_conversion_100_kg_fuel() -> None:
    assert CO2_PER_KG_FUEL * 100.0 == pytest.approx(316.0)


def test_fuel_penalty_co2_is_consistent() -> None:
    penalty = estimate_fuel_penalty("A320", 350.0, 370.0)
    assert penalty["delta_co2_kg"] == pytest.approx(
        penalty["delta_fuel_kg"] * CO2_PER_KG_FUEL
    )


def test_climb_penalty_midpoint() -> None:
    penalty = estimate_fuel_penalty("A320", 350.0, 370.0)
    assert penalty["penalty_pct"] == pytest.approx(2.0)
    assert penalty["delta_fuel_kg"] == pytest.approx(
        0.020 * CRUISE_BURN_KG_S * WINDOW_SECONDS
    )


def test_descent_penalty_midpoint() -> None:
    penalty = estimate_fuel_penalty("B737", 350.0, 330.0)
    assert penalty["penalty_pct"] == pytest.approx(1.15)
    assert penalty["delta_fuel_kg"] == pytest.approx(
        0.0115 * CRUISE_BURN_KG_S * WINDOW_SECONDS
    )


def test_level_change_is_free() -> None:
    penalty = estimate_fuel_penalty("A320", 350.0, 350.0)
    assert penalty == {"delta_fuel_kg": 0.0, "delta_co2_kg": 0.0, "penalty_pct": 0.0}


def test_unknown_aircraft_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_fuel_penalty("A380", 350.0, 370.0)
    with pytest.raises(ValueError):
        evaluate_altitude_candidates(0.85, 350.0, aircraft_type="A380")


def test_net_climate_benefit_positive_when_contrail_dominates() -> None:
    result = calculate_net_climate_benefit(
        avoided_contrail_ef_j=6.0e9, delta_co2_kg=30.0
    )
    assert result["net_benefit_positive"] is True
    assert result["benefit_to_cost_ratio"] > 1.0
    assert result["avoided_ef_j"] == pytest.approx(6.0e9)
    assert result["delta_co2_kg"] == pytest.approx(30.0)


def test_net_climate_benefit_negative_without_avoided_contrail() -> None:
    result = calculate_net_climate_benefit(
        avoided_contrail_ef_j=0.0, delta_co2_kg=30.0
    )
    assert result["net_benefit_positive"] is False
    assert result["benefit_to_cost_ratio"] == pytest.approx(0.0)


def test_horizon_scaling_grows_co2_cost() -> None:
    short = calculate_net_climate_benefit(6.0e9, 30.0, metric_horizon_years=20)
    long = calculate_net_climate_benefit(6.0e9, 30.0, metric_horizon_years=100)
    assert long["benefit_to_cost_ratio"] == pytest.approx(
        short["benefit_to_cost_ratio"] / 5.0
    )
    assert long["net_benefit_positive"] is True


def test_climate_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        calculate_net_climate_benefit(1.0e9, 10.0, metric_horizon_years=0)
    with pytest.raises(ValueError):
        calculate_net_climate_benefit(-1.0, 10.0)


def test_low_probability_abstains() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.20, baseline_fl=350.0)
    assert decision["status"] == "abstain"
    assert decision["recommended_fl"] is None
    assert decision["candidates"] == []
    assert decision["delta_fuel_kg"] is None


def test_high_probability_recommends_altitude() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.85, baseline_fl=350.0)
    assert decision["status"] == "eligible"
    assert decision["recommended_fl"] == pytest.approx(330.0)
    assert len(decision["candidates"]) == 2
    assert decision["delta_fuel_kg"] is not None
    assert decision["delta_fuel_kg"] > 0.0
    assert decision["delta_co2_kg"] is not None
    assert decision["delta_co2_kg"] == pytest.approx(
        decision["delta_fuel_kg"] * CO2_PER_KG_FUEL
    )
    assert decision["avoided_contrail_ef_j"] is not None
    assert decision["avoided_contrail_ef_j"] > 0.0
    for candidate in decision["candidates"]:
        assert candidate["rejected"] is False
        assert candidate["net_benefit_positive"] is True
    assert decision["candidates"][0]["target_fl"] == pytest.approx(370.0)
    assert decision["candidates"][1]["target_fl"] == pytest.approx(330.0)


def test_climb_above_ceiling_rejected() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.85, baseline_fl=390.0)
    climb = decision["candidates"][0]
    assert climb["target_fl"] == pytest.approx(410.0)
    assert climb["rejected"] is True
    assert climb["rejection_reason"] is not None
    assert "ceiling" in climb["rejection_reason"]
    assert climb["delta_fuel_kg"] == pytest.approx(0.0)
    assert climb["net_benefit_positive"] is False
    assert decision["status"] == "eligible"
    assert decision["recommended_fl"] == pytest.approx(370.0)


def test_descent_below_minimum_rejected() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.85, baseline_fl=270.0)
    descent = decision["candidates"][1]
    assert descent["target_fl"] == pytest.approx(250.0)
    assert descent["rejected"] is True
    assert descent["rejection_reason"] is not None
    assert decision["recommended_fl"] == pytest.approx(290.0)


def test_activation_threshold_boundary_is_evaluated() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.40, baseline_fl=350.0)
    assert decision["status"] == "eligible"


def test_probability_validation() -> None:
    with pytest.raises(ValueError):
        evaluate_altitude_candidates(baseline_p_persist=1.5, baseline_fl=350.0)
    with pytest.raises(ValueError):
        evaluate_altitude_candidates(baseline_p_persist=-0.1, baseline_fl=350.0)
