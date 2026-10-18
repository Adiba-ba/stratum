"""Tests for atmospheric diagnostics, aircraft parameters, and engine-aware SAC.

Covers the aircraft engine database lookup, the gradient Richardson
number (stability, turbulent shear, clamps, degenerate layers), the solar
zenith angle (equinox noon, night, seasonal high-latitude sun), and the
engine-dependent Schmidt-Appleman slope.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from stratum.physics import compute_sac_slope
from stratum.physics.diagnostics import (
    AIRCRAFT_ENGINE_DATABASE,
    C_P,
    KAPPA,
    R_D,
    compute_richardson_number,
    compute_solar_zenith_angle,
    get_aircraft_params,
)

REFERENCE_PRESSURE_PA = 23842.0


def test_aircraft_database_lookup_is_case_insensitive() -> None:
    assert get_aircraft_params("a320") == AIRCRAFT_ENGINE_DATABASE["A320"]
    assert get_aircraft_params("A20N") == AIRCRAFT_ENGINE_DATABASE["A20N"]
    assert get_aircraft_params("B738")["ei_n"] == pytest.approx(1.8e15)


def test_aircraft_database_fallback_to_default() -> None:
    params = get_aircraft_params("ZZZZ")
    assert params == AIRCRAFT_ENGINE_DATABASE["DEFAULT"]
    assert params["eta"] == pytest.approx(0.33)
    assert params["q_fuel"] == pytest.approx(43.1e6)
    # Unknown types must not silently create database entries.
    assert "ZZZZ" not in AIRCRAFT_ENGINE_DATABASE


def test_aircraft_database_entries_are_physical() -> None:
    for designator, params in AIRCRAFT_ENGINE_DATABASE.items():
        assert 0.0 < params["eta"] < 1.0, designator
        assert 1.0e13 < params["ei_n"] < 1.0e16, designator
        assert 42.0e6 < params["q_fuel"] < 45.0e6, designator
    assert KAPPA == pytest.approx(R_D / C_P)
    assert KAPPA == pytest.approx(2.0 / 7.0, abs=1e-3)


def test_aircraft_database_lookup_returns_a_copy() -> None:
    """Mutating a lookup result must not re-tune later lookups."""
    params = get_aircraft_params("a320")
    params["eta"] = 0.99
    params["ei_n"] = 0.0

    fresh = get_aircraft_params("A320")
    assert fresh == AIRCRAFT_ENGINE_DATABASE["A320"]
    assert fresh["eta"] == pytest.approx(0.32)
    assert fresh["ei_n"] == pytest.approx(1.5e15)


def test_aircraft_database_fallback_returns_a_copy() -> None:
    """The DEFAULT fallback result is a copy too."""
    fallback = get_aircraft_params("ZZZZ")
    fallback["q_fuel"] = 0.0

    assert get_aircraft_params("ZZZZ") == AIRCRAFT_ENGINE_DATABASE["DEFAULT"]
    assert AIRCRAFT_ENGINE_DATABASE["DEFAULT"]["q_fuel"] == pytest.approx(43.1e6)


def test_richardson_number_zero_shear_is_very_stable() -> None:
    ri = compute_richardson_number(
        temp_above_k=250.0,
        temp_below_k=240.0,
        u_above=25.0,
        u_below=25.0,
        v_above=-5.0,
        v_below=-5.0,
        alt_above_m=11000.0,
        alt_below_m=10400.0,
        p_ambient_pa=REFERENCE_PRESSURE_PA,
    )
    assert ri == pytest.approx(10.0)


def test_richardson_number_collapsed_layers_fall_back() -> None:
    ri = compute_richardson_number(
        temp_above_k=250.0,
        temp_below_k=250.0,
        u_above=40.0,
        u_below=0.0,
        v_above=0.0,
        v_below=0.0,
        alt_above_m=11000.0,
        alt_below_m=11000.5,
        p_ambient_pa=REFERENCE_PRESSURE_PA,
    )
    assert ri == pytest.approx(1.0)


def test_richardson_number_stable_stratification_is_positive() -> None:
    ri = compute_richardson_number(
        temp_above_k=221.0,
        temp_below_k=221.0,
        u_above=40.0,
        u_below=10.0,
        v_above=0.0,
        v_below=0.0,
        alt_above_m=11400.0,
        alt_below_m=10800.0,
        p_ambient_pa=REFERENCE_PRESSURE_PA,
    )
    # Equal temperatures still give d(theta)/dz > 0 thanks to the pressure
    # term; moderate shear keeps Ri positive and below the clamp.
    assert 0.0 < ri < 10.0


def test_richardson_number_strong_shear_is_turbulent() -> None:
    ri = compute_richardson_number(
        temp_above_k=221.0,
        temp_below_k=221.0,
        u_above=60.0,
        u_below=5.0,
        v_above=20.0,
        v_below=-10.0,
        alt_above_m=11000.0,
        alt_below_m=10700.0,
        p_ambient_pa=REFERENCE_PRESSURE_PA,
    )
    assert 0.0 < ri < 0.25


def test_richardson_number_clamps_extreme_instability() -> None:
    ri = compute_richardson_number(
        temp_above_k=200.0,
        temp_below_k=240.0,
        u_above=0.5,
        u_below=0.0,
        v_above=0.0,
        v_below=0.0,
        alt_above_m=11200.0,
        alt_below_m=10400.0,
        p_ambient_pa=REFERENCE_PRESSURE_PA,
    )
    assert ri == pytest.approx(-5.0)


def test_solar_zenith_angle_equinox_noon_is_overhead() -> None:
    zenith = compute_solar_zenith_angle(
        datetime(2025, 3, 20, 12, 0, 0), lat_deg=0.0, lon_deg=0.0
    )
    assert 0.0 < zenith < 5.0


def test_solar_zenith_angle_night_is_below_horizon() -> None:
    zenith = compute_solar_zenith_angle(
        datetime(2025, 3, 20, 0, 0, 0), lat_deg=0.0, lon_deg=0.0
    )
    assert zenith > 90.0
    assert zenith <= 180.0


def test_solar_zenith_angle_winter_high_latitude() -> None:
    zenith = compute_solar_zenith_angle(
        datetime(2025, 12, 21, 12, 0, 0), lat_deg=60.0, lon_deg=0.0
    )
    # Solar altitude = 90 - 60 - 23.4 = 6.6 deg, so zenith ~ 83.4 deg.
    assert 78.0 < zenith < 88.0


def test_solar_zenith_angle_bounds_over_day() -> None:
    for hour in range(24):
        for lat_deg, lon_deg in ((19.09, 72.87), (28.57, 77.10), (-40.0, 150.0)):
            zenith = compute_solar_zenith_angle(
                datetime(2025, 6, 15, hour, 30, 0), lat_deg, lon_deg
            )
            assert math.isfinite(zenith)
            assert 0.0 <= zenith <= 180.0


def test_sac_slope_increases_with_engine_efficiency() -> None:
    """A more efficient engine leaves less waste heat: G rises with eta."""
    low_eta = compute_sac_slope(REFERENCE_PRESSURE_PA, eta=0.30)
    high_eta = compute_sac_slope(REFERENCE_PRESSURE_PA, eta=0.40)
    assert high_eta > low_eta
    assert high_eta / low_eta == pytest.approx(0.70 / 0.60, rel=1e-12)


def test_sac_slope_scales_linearly_with_pressure() -> None:
    low_pressure = compute_sac_slope(20000.0)
    high_pressure = compute_sac_slope(40000.0)
    assert high_pressure == pytest.approx(2.0 * low_pressure, rel=1e-12)
