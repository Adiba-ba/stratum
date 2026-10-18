"""Schema tests for the extended 21-feature correction-model contract.

The engine-microphysics upgrade replaces the cyclical encodings and
derived temperature margins with the raw ``month``,
``local_solar_time_hours``, Schmidt-Appleman threshold temperatures
(``tlm_k``, ``tlc_k``) and the four diagnostic/engine features
(Richardson number, solar zenith, propulsion efficiency, log soot EI).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stratum.model.features import (
    DEFAULT_ENGINE_EFFICIENCY_ETA,
    DEFAULT_LOG_SOOT_EI_N,
    DEFAULT_RICHARDSON_NUMBER,
    DEFAULT_SOLAR_ZENITH_DEG,
    FEATURE_NAMES,
    encode_cyclical,
    extract_features_row,
    feature_vector,
)

EXPECTED_FEATURE_NAMES = (
    "rhi",
    "rhi_above",
    "rhi_below",
    "rhi_grad",
    "temp_k",
    "tlm_k",
    "tlc_k",
    "pressure_pa",
    "cloud_ice",
    "high_cloud_fraction",
    "vertical_velocity_omega",
    "vertical_wind_shear",
    "dist_to_tropopause_m",
    "lat",
    "lon",
    "month",
    "local_solar_time_hours",
    "richardson_number",
    "solar_zenith_deg",
    "engine_efficiency_eta",
    "log_soot_ei_n",
)


def _base_kwargs() -> dict[str, float]:
    return {
        "rhi": 0.9,
        "rhi_above": 1.1,
        "rhi_below": 0.7,
        "rhi_grad": 0.4,
        "temp_k": 225.0,
        "tlm_k": 231.0,
        "tlc_k": 226.0,
        "pressure_pa": 25000.0,
        "cloud_ice": 1.0,
        "high_cloud_fraction": 0.3,
        "vertical_velocity_omega": -0.05,
        "vertical_wind_shear": 6.0,
        "dist_to_tropopause_m": 1500.0,
        "lat": 28.6,
        "lon": 77.2,
        "month": 12.0,
        "local_solar_time_hours": 1.0,
    }


def test_feature_schema_is_exact_and_ordered() -> None:
    assert FEATURE_NAMES == EXPECTED_FEATURE_NAMES
    assert len(FEATURE_NAMES) == 21


def test_extract_features_row_uses_diagnostic_defaults() -> None:
    row = extract_features_row(**_base_kwargs())
    assert tuple(row) == FEATURE_NAMES
    assert row["richardson_number"] == pytest.approx(DEFAULT_RICHARDSON_NUMBER)
    assert row["solar_zenith_deg"] == pytest.approx(DEFAULT_SOLAR_ZENITH_DEG)
    assert row["engine_efficiency_eta"] == pytest.approx(DEFAULT_ENGINE_EFFICIENCY_ETA)
    assert row["log_soot_ei_n"] == pytest.approx(DEFAULT_LOG_SOOT_EI_N)
    # Raw time and threshold features are carried verbatim.
    assert row["tlm_k"] == pytest.approx(231.0)
    assert row["tlc_k"] == pytest.approx(226.0)
    assert row["month"] == pytest.approx(12.0)
    assert row["local_solar_time_hours"] == pytest.approx(1.0)


def test_extract_features_row_accepts_explicit_diagnostics() -> None:
    row = extract_features_row(
        **_base_kwargs(),
        richardson_number=0.35,
        solar_zenith_deg=145.0,
        engine_efficiency_eta=0.38,
        log_soot_ei_n=math.log10(4.0e14),
    )
    assert row["richardson_number"] == pytest.approx(0.35)
    assert row["solar_zenith_deg"] == pytest.approx(145.0)
    assert row["engine_efficiency_eta"] == pytest.approx(0.38)
    assert row["log_soot_ei_n"] == pytest.approx(math.log10(4.0e14))


def test_feature_vector_round_trips_in_canonical_order() -> None:
    row = extract_features_row(**_base_kwargs(), richardson_number=2.5)
    vector = feature_vector(row)
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector[FEATURE_NAMES.index("rhi")] == pytest.approx(0.9)
    assert vector[FEATURE_NAMES.index("richardson_number")] == pytest.approx(2.5)
    assert vector[FEATURE_NAMES.index("log_soot_ei_n")] == pytest.approx(
        DEFAULT_LOG_SOOT_EI_N
    )


def test_feature_vector_rejects_missing_keys() -> None:
    row = {name: 0.0 for name in FEATURE_NAMES if name != "rhi"}
    with pytest.raises(KeyError):
        feature_vector(row)


def test_encode_cyclical_helper_is_kept() -> None:
    start = encode_cyclical(0.0, 12.0)
    wrap = encode_cyclical(12.0, 12.0)
    assert start[0] == pytest.approx(wrap[0], abs=1e-12)
    assert start[1] == pytest.approx(wrap[1], abs=1e-12)
    sin_jan, cos_jan = encode_cyclical(1.0, 12.0)
    assert sin_jan == pytest.approx(math.sin(math.pi / 6.0))
    assert cos_jan == pytest.approx(math.cos(math.pi / 6.0))
    with pytest.raises(ValueError):
        encode_cyclical(1.0, 0.0)
