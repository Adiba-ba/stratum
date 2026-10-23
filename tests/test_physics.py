"""Baseline physics verification tests for Project STRATUM.

Asserts the STRATUM verification quantities for the Schmidt-Appleman
criterion at 250 hPa / eta = 0.30 and the ISA pressure conversion.
"""

from __future__ import annotations

import math

import pytest

from stratum.physics import sac, thermo

P250_PA = 25000.0
ETA = 0.30


def test_mixing_slope_250hpa() -> None:
    """G at 250 hPa / eta = 0.30 equals 1.6413706426 Pa/K."""
    g = sac.G(P250_PA, ETA)
    assert round(g, 6) == 1.641371
    assert g == pytest.approx(1.6413706426, abs=1e-9)


def test_tlm_250hpa() -> None:
    """T_LM at 250 hPa / eta = 0.30 equals -41.9128742 deg C."""
    tlm_c = sac.solve_TLM(P250_PA, ETA) - 273.15
    assert tlm_c == pytest.approx(-41.9128742, abs=0.01)


def test_analytic_liquid_derivative() -> None:
    """Analytic d(e_w)/dT matches a central finite difference."""
    t_k = 231.0
    h = 1e-5
    finite_difference = (thermo.e_w(t_k + h) - thermo.e_w(t_k - h)) / (2.0 * h)
    assert thermo.d_e_w_dT(t_k) == pytest.approx(finite_difference, rel=1e-8)


@pytest.mark.parametrize(
    ("fl", "expected_hpa"),
    [(250.0, 376.015), (340.0, 249.996)],
)
def test_fl_to_p(fl: float, expected_hpa: float) -> None:
    """ISA pressure conversion checks for FL250 and FL340."""
    assert thermo.fl_to_p(fl) == pytest.approx(expected_hpa, abs=0.05)


@pytest.mark.parametrize(
    ("t_c", "rhi", "forms", "persists"),
    [
        (-52.0, 0.60, True, False),  # Point A
        (-50.0, 1.12, True, True),  # Point B
        (-38.0, 0.58, False, False),  # Point C
    ],
)
def test_sac_points(t_c: float, rhi: float, forms: bool, persists: bool) -> None:
    """SAC evaluation at verification points A, B and C (250 hPa)."""
    result = sac.evaluate_sac(P250_PA, t_c + 273.15, rhi, eta=ETA)
    assert result["forms"] is forms, result
    assert result["persists"] is persists, result


def test_critical_t_bracket_failure_raises() -> None:
    """An inconsistent (TLM, G) pair fails the root bracket loudly."""
    with pytest.raises(ValueError, match="bracket failed") as excinfo:
        sac.critical_T(rhw=0.5, TLM=231.0, G=0.01)
    message = str(excinfo.value)
    assert "rhw=0.5" in message
    assert "TLM=231.0" in message
    assert "G=0.01" in message


def test_evaluate_sac_zero_rhi_matches_linear_root() -> None:
    """RHi = 0.0 solves the mixing-line equation without a special case."""
    result = sac.evaluate_sac(P250_PA, 223.15, rhi=0.0, eta=ETA)
    tlm_k = sac.solve_TLM(P250_PA, ETA)
    expected_tlc_c = tlm_k - thermo.e_w(tlm_k) / sac.G(P250_PA, ETA) - 273.15

    assert math.isfinite(result["TLC_C"])
    assert result["TLC_C"] == pytest.approx(expected_tlc_c, abs=1e-9)
    assert result["TLC_C"] < result["TLM_C"]
    assert result["persists"] is False


@pytest.mark.parametrize("rhw", [-0.1, float("nan"), float("inf")])
def test_critical_t_rejects_negative_or_non_finite_rhw(rhw: float) -> None:
    """Only non-negative, finite water relative humidities are accepted."""
    with pytest.raises(ValueError, match="non-negative and finite"):
        sac.critical_T(rhw=rhw, TLM=231.0, G=0.5)


def test_evaluate_sac_raises_on_impossible_mixing_slope() -> None:
    """SAC evaluation fails loudly instead of returning a bogus ``forms``."""
    with pytest.raises(ValueError):
        sac.evaluate_sac(p_Pa=10.0, T_K=220.0, rhi=0.5)
