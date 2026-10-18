"""Tests for the counterfactual decision layer (Step 6)."""

from __future__ import annotations

import pytest

from stratum.decide import (
    AGWP_CO2_20YR_W_YR_PER_M2_PER_KG,
    AGWP_CO2_100YR_W_YR_PER_M2_PER_KG,
    CO2_AGWP_W_YR_PER_M2_PER_KG_20YR,
    CO2_PER_KG_FUEL,
    EARTH_SURFACE_AREA_M2,
    MAX_METRIC_HORIZON_YEARS,
    MIN_METRIC_HORIZON_YEARS,
    SECONDS_PER_YEAR,
    calculate_net_climate_benefit,
    co2_agwp_w_yr_per_m2_per_kg,
    estimate_fuel_penalty,
    evaluate_altitude_candidates,
)
from stratum.decide.counterfactual import (
    DEFAULT_SEGMENT_LENGTH_KM,
    MEDIAN_PERSISTENT_EF_J_PER_M,
)

WINDOW_SECONDS = 20.0 * 60.0
CRUISE_BURN_KG_S = 0.65

# Independent copies of the cited climate constants (Teoh et al. 2024,
# ACP 24, 6071-6093), used to recompute expected ratios from the
# documented J/m^2 conversion instead of trusting the module output.
AGWP_CO2_20 = 2.39e-14
AGWP_CO2_100 = 8.80e-14
MEDIAN_PERSISTENT_EF_J_PER_M_REFERENCE = 1.74e7

SUPPORTED_CRUISE_BURN_KG_S = {
    "A320": 0.65,
    "A20N": 0.58,
    "B737": 0.65,
    "B738": 0.65,
    "B38M": 0.60,
    "B77W": 2.44,
    "DEFAULT": 0.65,
}


def _reference_agwp(metric_horizon_years: float) -> float:
    """Recompute the linearly interpolated AGWP from the cited endpoints."""
    fraction = (metric_horizon_years - 20.0) / 80.0
    return AGWP_CO2_20 + fraction * (AGWP_CO2_100 - AGWP_CO2_20)


def _expected_ratio(
    avoided_contrail_ef_j: float,
    delta_co2_kg: float,
    metric_horizon_years: float = 20.0,
) -> float:
    """Recompute the benefit-to-cost ratio from the documented units."""
    avoided_flux_j_per_m2 = avoided_contrail_ef_j / EARTH_SURFACE_AREA_M2
    co2_flux_j_per_m2 = (
        delta_co2_kg * _reference_agwp(metric_horizon_years) * SECONDS_PER_YEAR
    )
    return avoided_flux_j_per_m2 / co2_flux_j_per_m2


def _expected_avoided_ef_j(
    p_persist: float, effectiveness: float, segment_length_km: float
) -> float:
    """Recompute the avoided forcing from the cited per-metre median."""
    return (
        p_persist
        * effectiveness
        * MEDIAN_PERSISTENT_EF_J_PER_M_REFERENCE
        * segment_length_km
        * 1000.0
    )


def test_co2_conversion_100_kg_fuel() -> None:
    assert CO2_PER_KG_FUEL * 100.0 == pytest.approx(316.0)


def test_agwp_endpoints_are_cited_values() -> None:
    assert AGWP_CO2_20YR_W_YR_PER_M2_PER_KG == pytest.approx(2.39e-14, rel=1e-12)
    assert AGWP_CO2_100YR_W_YR_PER_M2_PER_KG == pytest.approx(8.80e-14, rel=1e-12)
    # The old name is kept as an alias of the 20-year value.
    assert CO2_AGWP_W_YR_PER_M2_PER_KG_20YR == pytest.approx(
        AGWP_CO2_20YR_W_YR_PER_M2_PER_KG, rel=1e-12
    )
    assert MIN_METRIC_HORIZON_YEARS == 20
    assert MAX_METRIC_HORIZON_YEARS == 100


def test_agwp_interpolates_between_cited_endpoints() -> None:
    assert co2_agwp_w_yr_per_m2_per_kg(20) == pytest.approx(AGWP_CO2_20, rel=1e-12)
    assert co2_agwp_w_yr_per_m2_per_kg(100) == pytest.approx(AGWP_CO2_100, rel=1e-12)
    assert co2_agwp_w_yr_per_m2_per_kg(60) == pytest.approx(
        0.5 * (AGWP_CO2_20 + AGWP_CO2_100), rel=1e-12
    )
    assert co2_agwp_w_yr_per_m2_per_kg(40) == pytest.approx(
        _reference_agwp(40.0), rel=1e-12
    )
    # Monotone in the horizon.
    assert co2_agwp_w_yr_per_m2_per_kg(30) < co2_agwp_w_yr_per_m2_per_kg(70)


def test_agwp_rejects_horizons_outside_cited_range() -> None:
    for horizon in (19, 19.9, 100.1, 101, 0, -20, float("nan")):
        with pytest.raises(ValueError):
            co2_agwp_w_yr_per_m2_per_kg(horizon)


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


def test_fuel_penalty_supports_registry_aircraft_types() -> None:
    for aircraft_type, cruise_burn_kg_s in SUPPORTED_CRUISE_BURN_KG_S.items():
        penalty = estimate_fuel_penalty(aircraft_type, 350.0, 370.0)
        assert penalty["penalty_pct"] == pytest.approx(2.0)
        assert penalty["delta_fuel_kg"] == pytest.approx(
            0.020 * cruise_burn_kg_s * WINDOW_SECONDS
        )
    assert estimate_fuel_penalty("b38m", 350.0, 370.0) == pytest.approx(
        estimate_fuel_penalty("B38M", 350.0, 370.0)
    )


def test_unknown_aircraft_rejected_by_fuel_model() -> None:
    with pytest.raises(ValueError, match="B77W"):
        estimate_fuel_penalty("A380", 350.0, 370.0)


def test_unknown_aircraft_falls_back_to_default_in_decision() -> None:
    unknown = evaluate_altitude_candidates(0.85, 350.0, aircraft_type="A388")
    default = evaluate_altitude_candidates(0.85, 350.0, aircraft_type="DEFAULT")
    assert unknown["aircraft_type"] == "DEFAULT"
    assert unknown["status"] == "eligible"
    assert unknown["recommended_fl"] == pytest.approx(default["recommended_fl"])
    assert unknown["delta_fuel_kg"] == pytest.approx(default["delta_fuel_kg"])
    assert unknown["benefit_to_cost_ratio"] == pytest.approx(
        default["benefit_to_cost_ratio"]
    )


def test_unit_conversion_locks_seconds_per_year() -> None:
    result = calculate_net_climate_benefit(5.56e9, 28.35)
    expected = _expected_ratio(5.56e9, 28.35)
    assert result["benefit_to_cost_ratio"] == pytest.approx(expected, rel=1e-6)
    assert result["net_benefit_positive"] is False
    assert result["avoided_flux_j_per_m2"] == pytest.approx(
        5.56e9 / EARTH_SURFACE_AREA_M2, rel=1e-9
    )
    # 28.35 kg CO2 at the cited 20-year AGWP (2.39e-14 W*yr/m^2 per kg).
    assert result["co2_flux_j_per_m2"] == pytest.approx(
        28.35 * AGWP_CO2_20 * SECONDS_PER_YEAR, rel=1e-9
    )
    assert result["co2_flux_j_per_m2"] == pytest.approx(2.138e-05, rel=1e-3)
    assert result["metric_horizon_years"] == pytest.approx(20.0)
    assert result["co2_agwp_w_yr_per_m2_per_kg"] == pytest.approx(AGWP_CO2_20)


def test_net_climate_benefit_positive_when_contrail_dominates() -> None:
    # Sanity case locked by the task: P(persist)=0.695, descent
    # effectiveness 0.80, 48 km segment -> 4.64e11 J avoided against
    # 28.35 kg CO2.
    avoided_ef_j = _expected_avoided_ef_j(0.695, 0.80, 48.0)
    assert avoided_ef_j == pytest.approx(4.64e11, rel=1e-3)
    result = calculate_net_climate_benefit(
        avoided_contrail_ef_j=avoided_ef_j, delta_co2_kg=28.35
    )
    assert result["net_benefit_positive"] is True
    assert result["avoided_ef_j"] == pytest.approx(avoided_ef_j)
    assert result["delta_co2_kg"] == pytest.approx(28.35)
    assert result["benefit_to_cost_ratio"] == pytest.approx(
        _expected_ratio(avoided_ef_j, 28.35), rel=1e-6
    )
    assert result["benefit_to_cost_ratio"] == pytest.approx(42.6, rel=1e-2)


def test_net_climate_benefit_negative_without_avoided_contrail() -> None:
    result = calculate_net_climate_benefit(
        avoided_contrail_ef_j=0.0, delta_co2_kg=30.0
    )
    assert result["net_benefit_positive"] is False
    assert result["benefit_to_cost_ratio"] == pytest.approx(0.0)
    assert result["avoided_flux_j_per_m2"] == pytest.approx(0.0)
    assert result["co2_flux_j_per_m2"] > 0.0


def test_horizon_scaling_grows_co2_cost() -> None:
    short = calculate_net_climate_benefit(4.64e11, 28.35, metric_horizon_years=20)
    long = calculate_net_climate_benefit(4.64e11, 28.35, metric_horizon_years=100)
    mid = calculate_net_climate_benefit(4.64e11, 28.35, metric_horizon_years=60)

    assert long["benefit_to_cost_ratio"] == pytest.approx(
        _expected_ratio(4.64e11, 28.35, 100.0), rel=1e-9
    )
    assert mid["benefit_to_cost_ratio"] == pytest.approx(
        _expected_ratio(4.64e11, 28.35, 60.0), rel=1e-9
    )
    assert long["benefit_to_cost_ratio"] < mid["benefit_to_cost_ratio"]
    assert mid["benefit_to_cost_ratio"] < short["benefit_to_cost_ratio"]
    # The 100-year AGWP is 8.80e-14 / 2.39e-14 = 3.68x the 20-year value,
    # so the ratio shrinks by that factor (not the old 5x proxy).
    assert short["benefit_to_cost_ratio"] / long["benefit_to_cost_ratio"] == (
        pytest.approx(AGWP_CO2_100 / AGWP_CO2_20, rel=1e-9)
    )
    assert short["net_benefit_positive"] is True
    assert long["net_benefit_positive"] is True


def test_climate_rejects_invalid_inputs() -> None:
    for horizon in (19, 101):
        with pytest.raises(ValueError):
            calculate_net_climate_benefit(1.0e9, 10.0, metric_horizon_years=horizon)
    with pytest.raises(ValueError):
        calculate_net_climate_benefit(-1.0, 10.0)
    with pytest.raises(ValueError):
        calculate_net_climate_benefit(1.0e9, -10.0)


def test_low_probability_abstains() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.20, baseline_fl=350.0)
    assert decision["status"] == "abstain"
    assert decision["aircraft_type"] == "A320"
    assert decision["recommended_fl"] is None
    assert decision["candidates"] == []
    assert decision["delta_fuel_kg"] is None
    assert decision["segment_length_km"] == pytest.approx(DEFAULT_SEGMENT_LENGTH_KM)


def test_high_probability_recommends_altitude() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.85, baseline_fl=350.0)
    assert decision["status"] == "eligible"
    assert decision["aircraft_type"] == "A320"
    assert decision["recommended_fl"] == pytest.approx(330.0)
    assert decision["segment_length_km"] == pytest.approx(48.0)
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

    # Best candidate is the descent: 1.15% of the 20-minute window fuel
    # becomes CO2, and the 0.80-effective descent avoids 80% of the
    # median persistent forcing over the 48 km segment at P=0.85.
    expected_delta_fuel_kg = 0.0115 * CRUISE_BURN_KG_S * WINDOW_SECONDS
    expected_avoided_ef_j = _expected_avoided_ef_j(0.85, 0.80, 48.0)
    assert expected_avoided_ef_j == pytest.approx(5.67936e11, rel=1e-9)
    assert decision["delta_fuel_kg"] == pytest.approx(expected_delta_fuel_kg)
    assert decision["avoided_contrail_ef_j"] == pytest.approx(expected_avoided_ef_j)
    assert decision["benefit_to_cost_ratio"] == pytest.approx(
        _expected_ratio(
            expected_avoided_ef_j, expected_delta_fuel_kg * CO2_PER_KG_FUEL
        ),
        rel=1e-9,
    )
    # The climb keeps a higher escape effectiveness (0.85) but its 2.0%
    # fuel penalty outweighs that advantage at this probability.
    climb = decision["candidates"][0]
    assert climb["avoided_ef_j"] == pytest.approx(
        _expected_avoided_ef_j(0.85, 0.85, 48.0)
    )
    assert climb["benefit_to_cost_ratio"] < decision["benefit_to_cost_ratio"]


def test_avoided_forcing_scales_with_segment_length() -> None:
    short = evaluate_altitude_candidates(0.85, 350.0, segment_length_km=24.0)
    long = evaluate_altitude_candidates(0.85, 350.0, segment_length_km=48.0)
    assert short["status"] == long["status"] == "eligible"
    assert short["avoided_contrail_ef_j"] * 2.0 == pytest.approx(
        long["avoided_contrail_ef_j"], rel=1e-12
    )
    assert short["benefit_to_cost_ratio"] * 2.0 == pytest.approx(
        long["benefit_to_cost_ratio"], rel=1e-12
    )


def test_short_segment_cannot_justify_widebody_maneuver() -> None:
    # A 5 km segment carries little avoidable forcing while the B77W fuel
    # penalty (1.15% of 2.44 kg/s over 20 min -> >100 kg CO2) dominates,
    # so no candidate passes the net-benefit test.
    decision = evaluate_altitude_candidates(
        0.40, 350.0, aircraft_type="B77W", segment_length_km=5.0
    )
    assert decision["status"] == "no_robust_candidate"
    assert decision["aircraft_type"] == "B77W"
    assert decision["recommended_fl"] is None
    for candidate in decision["candidates"]:
        assert candidate["rejected"] is False
        assert candidate["net_benefit_positive"] is False
    # The same probability on a full-length segment does pass.
    full = evaluate_altitude_candidates(
        0.40, 350.0, aircraft_type="B77W", segment_length_km=48.0
    )
    assert full["status"] == "eligible"


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
    assert decision["recommended_fl"] < decision["baseline_fl"]


def test_descent_below_minimum_rejected() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.85, baseline_fl=270.0)
    descent = decision["candidates"][1]
    assert descent["target_fl"] == pytest.approx(250.0)
    assert descent["rejected"] is True
    assert descent["rejection_reason"] is not None
    assert decision["recommended_fl"] == pytest.approx(290.0)
    assert decision["recommended_fl"] > decision["baseline_fl"]


def test_activation_threshold_boundary_is_evaluated() -> None:
    decision = evaluate_altitude_candidates(baseline_p_persist=0.40, baseline_fl=350.0)
    assert decision["status"] == "eligible"
    assert decision["recommended_fl"] == pytest.approx(330.0)


def test_probability_validation() -> None:
    with pytest.raises(ValueError):
        evaluate_altitude_candidates(baseline_p_persist=1.5, baseline_fl=350.0)
    with pytest.raises(ValueError):
        evaluate_altitude_candidates(baseline_p_persist=-0.1, baseline_fl=350.0)
    with pytest.raises(ValueError):
        evaluate_altitude_candidates(
            baseline_p_persist=0.85, baseline_fl=350.0, segment_length_km=0.0
        )
    with pytest.raises(ValueError):
        evaluate_altitude_candidates(
            baseline_p_persist=0.85, baseline_fl=350.0, segment_length_km=float("nan")
        )
