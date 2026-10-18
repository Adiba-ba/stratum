#!/usr/bin/env python
"""Demo: physics-constrained correction model (Project STRATUM, Step 5).

Generates a synthetic dataset of flight segments with physically plausible
atmospheric states, injects tri-state labels (0/1 with sensor noise, NaN
for unobservable scenes), trains a monotone LightGBM classifier, fits
isotonic calibration on the validation split, and reports train/validation
counts, raw versus calibrated Brier scores, a monotonicity verification,
and sample predictions for cold/humid versus warm/dry states.

Exit code is 0 when both monotonicity checks pass.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratum.model import (  # noqa: E402
    CalibratedModel,
    FEATURE_NAMES,
    compute_brier_score,
    extract_features_row,
    feature_vector,
    fit_calibrator,
    train_contrail_model,
)
from stratum.physics import (  # noqa: E402
    G,
    critical_T,
    e_i,
    e_w,
    fl_to_p,
    solve_TLM,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "X does not have valid feature names, but LGBMClassifier was "
        "fitted with feature names"
    ),
)

N_SEGMENTS = 2000
TRAIN_FRACTION = 0.7
UNOBSERVABLE_FRACTION = 0.2
LABEL_NOISE_FRACTION = 0.05
RANDOM_SEED = 42
TROPOPAUSE_ALTITUDE_M = 11000.0

RHI_INDEX = FEATURE_NAMES.index("rhi")
TLM_K_INDEX = FEATURE_NAMES.index("tlm_k")


def _generate_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic segments with physics-based tri-state labels.

    Each segment samples a flight level and derives pressure and the
    Schmidt-Appleman thresholds from :mod:`stratum.physics`. The label is
    drawn from a latent risk that increases with ice supersaturation and
    decreases with the temperature margin, then flipped with sensor noise.
    Unobservable segments (opaque cloud) get a NaN label.
    """
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for _ in range(N_SEGMENTS):
        flight_level = rng.uniform(300.0, 400.0)
        pressure_pa = fl_to_p(flight_level) * 100.0
        altitude_m = flight_level * 100.0 * 0.3048
        tlm_k = solve_TLM(pressure_pa)
        temp_margin_tlm = rng.uniform(-12.0, 4.0)
        temp_k = tlm_k + temp_margin_tlm
        rhi = float(rng.beta(2.0, 2.0)) * 1.7
        rhw = rhi * e_i(temp_k) / e_w(temp_k)
        tlc_k = critical_T(rhw, tlm_k, G(pressure_pa))

        rhi_above = float(np.clip(rhi + rng.normal(0.0, 0.12), 0.0, 2.0))
        rhi_below = float(np.clip(rhi + rng.normal(0.0, 0.12), 0.0, 2.0))

        unobservable = rng.random() < UNOBSERVABLE_FRACTION
        if unobservable:
            high_cloud_fraction = float(rng.uniform(0.6, 0.95))
        else:
            high_cloud_fraction = float(rng.uniform(0.0, 0.6))

        richardson_number = float(rng.uniform(0.1, 4.0))
        solar_zenith_deg = float(rng.uniform(0.0, 180.0))
        engine_efficiency_eta = float(rng.uniform(0.30, 0.40))
        log_soot_ei_n = float(rng.uniform(14.0, 16.0))

        logit = (
            2.4 * (rhi - 0.85)
            - 0.35 * temp_margin_tlm
            - 1.5 * high_cloud_fraction
            - 0.5 * (richardson_number - 1.0)
            + 4.0 * (engine_efficiency_eta - 0.32)
        )
        persistent = rng.random() < 1.0 / (1.0 + math.exp(-logit))
        if rng.random() < LABEL_NOISE_FRACTION:
            persistent = not persistent
        label = float("nan") if unobservable else float(persistent)

        lon = float(rng.uniform(68.0, 90.0))
        utc_hours = float(rng.uniform(0.0, 24.0))
        dist_to_tropopause_m = (
            altitude_m - TROPOPAUSE_ALTITUDE_M + float(rng.normal(0.0, 150.0))
        )
        row = extract_features_row(
            rhi=rhi,
            rhi_above=rhi_above,
            rhi_below=rhi_below,
            rhi_grad=rhi_above - rhi_below,
            temp_k=temp_k,
            tlm_k=tlm_k,
            tlc_k=tlc_k,
            pressure_pa=pressure_pa,
            cloud_ice=float(rng.random() < 0.25),
            high_cloud_fraction=high_cloud_fraction,
            vertical_velocity_omega=float(rng.normal(0.0, 0.08)),
            vertical_wind_shear=float(rng.uniform(0.0, 15.0)),
            dist_to_tropopause_m=dist_to_tropopause_m,
            lat=float(rng.uniform(8.0, 30.0)),
            lon=lon,
            month=float(rng.integers(1, 13)),
            local_solar_time_hours=(utc_hours + lon / 15.0) % 24.0,
            richardson_number=richardson_number,
            solar_zenith_deg=solar_zenith_deg,
            engine_efficiency_eta=engine_efficiency_eta,
            log_soot_ei_n=log_soot_ei_n,
        )
        rows.append(feature_vector(row))
        labels.append(label)
    return np.vstack(rows), np.asarray(labels, dtype=float)


def _sweep(
    model: CalibratedModel,
    baseline: np.ndarray,
    index: int,
    values: np.ndarray,
) -> np.ndarray:
    """Calibrated positive-class probabilities along one feature sweep."""
    probabilities = np.empty(values.size, dtype=float)
    for position, value in enumerate(values):
        row = baseline.copy()
        row[index] = value
        probabilities[position] = model.predict_proba(row)[0, 1]
    return probabilities


def main() -> int:
    rng = np.random.default_rng(RANDOM_SEED)
    features, labels = _generate_dataset(rng)
    n_samples = features.shape[0]
    n_train = int(round(n_samples * TRAIN_FRACTION))
    permutation = rng.permutation(n_samples)
    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:]
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]

    model = train_contrail_model(X_train, y_train, list(FEATURE_NAMES))
    calibrated = fit_calibrator(model, X_val, y_val)

    valid_val = ~np.isnan(y_val)
    X_val_valid = X_val[valid_val]
    y_val_valid = y_val[valid_val]
    raw_probs = model.predict_proba(X_val_valid)[:, 1]
    calibrated_probs = calibrated.predict_proba(X_val_valid)[:, 1]
    raw_brier = compute_brier_score(y_val_valid, raw_probs)
    calibrated_brier = compute_brier_score(y_val_valid, calibrated_probs)

    baseline = np.nanmedian(X_train[~np.isnan(y_train)], axis=0)
    rhi_probs = _sweep(calibrated, baseline, RHI_INDEX, np.linspace(0.1, 1.5, 50))
    margin_probs = _sweep(
        calibrated,
        baseline,
        TLM_K_INDEX,
        np.linspace(baseline[TLM_K_INDEX] - 10.0, baseline[TLM_K_INDEX] + 10.0, 50),
    )
    rhi_monotone = bool(np.all(np.diff(rhi_probs) >= -1e-9))
    margin_monotone = bool(np.all(np.diff(margin_probs) <= 1e-9))

    cold_humid = baseline.copy()
    cold_humid[RHI_INDEX] = 1.25
    cold_humid[TLM_K_INDEX] = baseline[TLM_K_INDEX] - 6.0
    warm_dry = baseline.copy()
    warm_dry[RHI_INDEX] = 0.35
    warm_dry[TLM_K_INDEX] = baseline[TLM_K_INDEX] + 6.0

    train_masked = int(np.isnan(y_train).sum())
    val_masked = int(np.isnan(y_val).sum())
    print("STRATUM Step 5 - Physics-Constrained Correction Model demo")
    print("=" * 64)
    print(f"Synthetic segments: {n_samples} (seed={RANDOM_SEED})")
    print(f"Feature schema: {len(FEATURE_NAMES)} features (frozen Section 12.2 order)")
    print(
        f"Train split:      {y_train.size} rows -> "
        f"{y_train.size - train_masked} labeled, "
        f"{train_masked} NaN-masked (unobservable)"
    )
    print(
        f"Validation split: {y_val.size} rows -> "
        f"{y_val.size - val_masked} labeled, "
        f"{val_masked} NaN-masked (unobservable)"
    )
    print()
    print(f"Raw validation Brier score:          {raw_brier:.5f}")
    print(f"Calibrated validation Brier score:   {calibrated_brier:.5f}")
    print(f"Brier improvement after calibration: {raw_brier - calibrated_brier:+.5f}")
    print()
    rhi_status = "PASS" if rhi_monotone else "FAIL"
    margin_status = "PASS" if margin_monotone else "FAIL"
    print(f"Monotonicity (RHi increasing -> P(persist) non-decreasing):            {rhi_status}")
    print(f"Monotonicity (tlm_k increasing -> P(persist) non-increasing):          {margin_status}")
    print()
    print("Sample predictions (all other features held at the training median):")
    for name, row in (("cold/humid", cold_humid), ("warm/dry", warm_dry)):
        raw_p = float(calibrated.predict_raw_proba(row)[0])
        calibrated_p = float(calibrated.predict_proba(row)[0, 1])
        print(
            f"  {name:>10}: rhi={row[RHI_INDEX]:.2f}, "
            f"tlm_k={row[TLM_K_INDEX]:.1f} K "
            f"-> raw {raw_p:.3f}, calibrated {calibrated_p:.3f}"
        )
    return 0 if (rhi_monotone and margin_monotone) else 1


if __name__ == "__main__":
    raise SystemExit(main())
