"""Tests for the physics-constrained correction model (Step 5, extended schema).

The schema now carries raw threshold temperatures and the diagnostic and
engine features; dataset generators below build labels from the latent
temperature margin (``temp_k - tlm_k``) so the model must recover it from
the raw features.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from stratum.model import (
    FEATURE_NAMES,
    compute_brier_score,
    encode_cyclical,
    extract_features_row,
    feature_vector,
    fit_calibrator,
    train_contrail_model,
)

# LightGBM's sklearn wrapper validates feature names even when predictions
# receive plain ndarrays fitted from ndarrays; silence its spurious warning.
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names, "
    "but LGBMClassifier was fitted with feature names"
)

RHI_INDEX = FEATURE_NAMES.index("rhi")
TLM_K_INDEX = FEATURE_NAMES.index("tlm_k")
TEMP_K_INDEX = FEATURE_NAMES.index("temp_k")
RICHARDSON_INDEX = FEATURE_NAMES.index("richardson_number")
ETA_INDEX = FEATURE_NAMES.index("engine_efficiency_eta")
LOG_SOOT_INDEX = FEATURE_NAMES.index("log_soot_ei_n")

_CONSTRAINED_INDICES = {
    RHI_INDEX: 1,
    TLM_K_INDEX: -1,
    RICHARDSON_INDEX: -1,
    ETA_INDEX: 1,
    LOG_SOOT_INDEX: 1,
}


def _synthetic_dataset(
    n_samples: int = 1500,
    seed: int = 7,
    nan_fraction: float = 0.0,
    label_noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Physically shaped synthetic data consistent with the feature schema.

    Labels are drawn from a latent risk that increases with ice
    supersaturation and with a colder temperature margin
    (``temp_k - tlm_k``); the model sees ``temp_k`` and ``tlm_k``
    separately and must reconstruct the margin.
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 1.0, size=(n_samples, len(FEATURE_NAMES)))
    rhi = rng.uniform(0.1, 1.5, n_samples)
    margin = rng.uniform(-12.0, 4.0, n_samples)
    tlm_k = rng.uniform(226.0, 234.0, n_samples)
    features[:, RHI_INDEX] = rhi
    features[:, TLM_K_INDEX] = tlm_k
    features[:, TEMP_K_INDEX] = tlm_k + margin
    logit = 2.5 * (rhi - 0.8) - 0.35 * margin
    prob = 1.0 / (1.0 + np.exp(-logit))
    labels = (rng.uniform(size=n_samples) < prob).astype(float)
    flipped = rng.uniform(size=n_samples) < label_noise
    labels[flipped] = 1.0 - labels[flipped]
    if nan_fraction > 0.0:
        n_nan = int(round(n_samples * nan_fraction))
        labels[rng.choice(n_samples, size=n_nan, replace=False)] = np.nan
    return features, labels


def _circle_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two unit-circle ``(sin, cos)`` points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_encode_cyclical_month_wrap_around() -> None:
    january = encode_cyclical(1.0, 12.0)
    december = encode_cyclical(12.0, 12.0)
    july = encode_cyclical(7.0, 12.0)
    expected = 2.0 * math.sin(math.pi / 12.0)
    assert _circle_distance(january, december) == pytest.approx(expected, abs=1e-12)
    assert _circle_distance(january, december) < _circle_distance(january, july)
    assert _circle_distance(january, july) == pytest.approx(2.0, abs=1e-12)


def test_encode_cyclical_rejects_nonpositive_period() -> None:
    with pytest.raises(ValueError):
        encode_cyclical(1.0, 0.0)


def test_extract_features_row_assembles_schema_vector() -> None:
    row = extract_features_row(
        rhi=0.9,
        rhi_above=1.1,
        rhi_below=0.7,
        rhi_grad=0.4,
        temp_k=225.0,
        tlm_k=231.0,
        tlc_k=226.0,
        pressure_pa=25000.0,
        cloud_ice=1.0,
        high_cloud_fraction=0.3,
        vertical_velocity_omega=-0.05,
        vertical_wind_shear=6.0,
        dist_to_tropopause_m=1500.0,
        lat=28.6,
        lon=77.2,
        month=12.0,
        local_solar_time_hours=1.0,
        richardson_number=0.8,
        solar_zenith_deg=120.0,
        engine_efficiency_eta=0.38,
        log_soot_ei_n=14.6,
    )
    assert tuple(row) == FEATURE_NAMES
    assert row["tlm_k"] == pytest.approx(231.0)
    assert row["tlc_k"] == pytest.approx(226.0)
    assert row["month"] == pytest.approx(12.0)
    assert row["local_solar_time_hours"] == pytest.approx(1.0)
    assert row["richardson_number"] == pytest.approx(0.8)
    assert row["solar_zenith_deg"] == pytest.approx(120.0)
    assert row["engine_efficiency_eta"] == pytest.approx(0.38)
    assert row["log_soot_ei_n"] == pytest.approx(14.6)
    vector = feature_vector(row)
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector[RHI_INDEX] == pytest.approx(0.9)
    assert vector[TLM_K_INDEX] == pytest.approx(231.0)
    assert vector[RICHARDSON_INDEX] == pytest.approx(0.8)


def test_extract_features_row_defaults_diagnostics() -> None:
    row = extract_features_row(
        rhi=0.9,
        rhi_above=1.1,
        rhi_below=0.7,
        rhi_grad=0.4,
        temp_k=225.0,
        tlm_k=231.0,
        tlc_k=226.0,
        pressure_pa=25000.0,
        cloud_ice=1.0,
        high_cloud_fraction=0.3,
        vertical_velocity_omega=-0.05,
        vertical_wind_shear=6.0,
        dist_to_tropopause_m=1500.0,
        lat=28.6,
        lon=77.2,
        month=12.0,
        local_solar_time_hours=1.0,
    )
    assert row["richardson_number"] == pytest.approx(1.0)
    assert row["solar_zenith_deg"] == pytest.approx(100.0)
    assert row["engine_efficiency_eta"] == pytest.approx(0.32)
    assert row["log_soot_ei_n"] == pytest.approx(15.0)


def test_feature_vector_rejects_missing_keys() -> None:
    row = {name: 0.0 for name in FEATURE_NAMES if name != "rhi"}
    with pytest.raises(KeyError):
        feature_vector(row)


def test_training_strictly_masks_nan_labels() -> None:
    features, labels = _synthetic_dataset(n_samples=600, seed=11, nan_fraction=0.3)
    assert int(np.isnan(labels).sum()) == 180
    model = train_contrail_model(features, labels, list(FEATURE_NAMES))
    constraints = list(model.get_params()["monotone_constraints"])
    assert len(constraints) == len(FEATURE_NAMES)
    for index in range(len(FEATURE_NAMES)):
        assert constraints[index] == _CONSTRAINED_INDICES.get(index, 0)
    probabilities = model.predict_proba(features)[:, 1]
    assert probabilities.shape == (600,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_training_rejects_all_nan_labels() -> None:
    features, _ = _synthetic_dataset(n_samples=50, seed=5)
    with pytest.raises(ValueError):
        train_contrail_model(features, np.full(50, np.nan), list(FEATURE_NAMES))


def test_training_rejects_non_binary_labels() -> None:
    features, _ = _synthetic_dataset(n_samples=60, seed=13)
    labels = np.zeros(60)
    labels[0] = 0.5
    with pytest.raises(ValueError):
        train_contrail_model(features, labels, list(FEATURE_NAMES))


def test_calibrated_probability_is_monotone_in_physics_features() -> None:
    features, labels = _synthetic_dataset(n_samples=1500, seed=23, nan_fraction=0.1)
    split = 1000
    model = train_contrail_model(features[:split], labels[:split], list(FEATURE_NAMES))
    calibrated = fit_calibrator(model, features[split:], labels[split:])
    baseline = features[split].copy()

    def sweep(index: int, values: np.ndarray) -> np.ndarray:
        probabilities = np.empty(values.size, dtype=float)
        for position, value in enumerate(values):
            row = baseline.copy()
            row[index] = value
            probabilities[position] = calibrated.predict_proba(row)[0, 1]
        return probabilities

    rhi_probs = sweep(RHI_INDEX, np.linspace(0.1, 1.5, 40))
    assert np.all(np.diff(rhi_probs) >= -1e-9)

    tlm_probs = sweep(TLM_K_INDEX, np.linspace(224.0, 236.0, 40))
    assert np.all(np.diff(tlm_probs) <= 1e-9)

    richardson_probs = sweep(RICHARDSON_INDEX, np.linspace(0.1, 4.0, 40))
    assert np.all(np.diff(richardson_probs) <= 1e-9)

    eta_probs = sweep(ETA_INDEX, np.linspace(0.30, 0.40, 40))
    assert np.all(np.diff(eta_probs) >= -1e-9)

    soot_probs = sweep(LOG_SOOT_INDEX, np.linspace(14.0, 16.0, 40))
    assert np.all(np.diff(soot_probs) >= -1e-9)


def test_brier_score_matches_sklearn() -> None:
    y_true = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    y_prob = np.array([0.10, 0.80, 0.65, 0.30, 0.90, 0.20, 0.55, 0.99])
    assert compute_brier_score(y_true, y_prob) == pytest.approx(
        brier_score_loss(y_true, y_prob), abs=1e-12
    )
    masked_true = y_true.copy()
    masked_true[0] = np.nan
    assert compute_brier_score(masked_true, y_prob) == pytest.approx(
        brier_score_loss(y_true[1:], y_prob[1:]), abs=1e-12
    )


def test_brier_score_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_brier_score(np.array([0.0, 1.0]), np.array([0.5]))


def test_calibrated_probabilities_are_valid_and_skip_nan_rows() -> None:
    features, labels = _synthetic_dataset(n_samples=400, seed=3, nan_fraction=0.2)
    model = train_contrail_model(features[:300], labels[:300], list(FEATURE_NAMES))
    calibrated = fit_calibrator(model, features[300:], labels[300:])
    proba = calibrated.predict_proba(features[300:])
    assert proba.shape == (100, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba[:, 0] + proba[:, 1], 1.0)


def test_fit_calibrator_rejects_all_nan_labels() -> None:
    features, labels = _synthetic_dataset(n_samples=200, seed=17)
    model = train_contrail_model(features[:150], labels[:150], list(FEATURE_NAMES))
    with pytest.raises(ValueError):
        fit_calibrator(model, features[150:], np.full(50, np.nan))
