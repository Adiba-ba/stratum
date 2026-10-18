"""Monotone LightGBM training for the STRATUM correction model.

The classifier is constrained to stay physically consistent: ice
supersaturation (``rhi``) and warmer ``tlm_k`` thresholds both raise the
persistence risk (warmer mixing-line thresholds raise the critical
temperature), per :data:`MONOTONE_CONSTRAINTS`, while lower Richardson
numbers (turbulent shear), higher engine efficiency, and higher soot
number emission indices enter the engine and turbulence microphysics.
Rows whose label is NaN (unobservable scenes, e.g. opaque cloud) are
strictly masked out and never treated as negatives.
"""

from __future__ import annotations

from typing import Sequence

import lightgbm as lgb
import numpy as np

__all__ = [
    "MONOTONE_CONSTRAINTS",
    "monotone_constraints_for",
    "train_contrail_model",
    "compute_brier_score",
]

#: Native LightGBM monotone constraint per feature: +1 increasing, -1 decreasing.
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "rhi": 1,
    "tlm_k": 1,
    "richardson_number": -1,
    "engine_efficiency_eta": 1,
    "log_soot_ei_n": 1,
}

#: Fixed LightGBM hyperparameters; ``random_state`` fixes reproducibility.
_MODEL_PARAMS: dict[str, object] = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "monotone_constraints_method": "intermediate",
    "random_state": 42,
    "verbosity": -1,
}


def monotone_constraints_for(feature_names: Sequence[str]) -> list[int]:
    """Order LightGBM monotone constraints to match ``feature_names``.

    Features absent from :data:`MONOTONE_CONSTRAINTS` are unconstrained
    (entry ``0``).
    """
    return [MONOTONE_CONSTRAINTS.get(name, 0) for name in feature_names]


def train_contrail_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> lgb.LGBMClassifier:
    """Fit a monotone binary LightGBM classifier on labelled segments.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, n_features)`` in the
        canonical order of ``feature_names``. NaN feature values are
        supported natively by LightGBM.
    y : np.ndarray
        Tri-state labels: ``1.0`` (persistent), ``0.0`` (not persistent)
        or ``NaN`` (unobservable). Rows with NaN labels are masked out;
        they are never coerced to ``0.0``.
    feature_names : list[str]
        Feature names in the same column order as ``X``.

    Returns
    -------
    lgb.LGBMClassifier
        Fitted classifier using binary logloss, ``random_state=42``, and
        native monotone constraints from :data:`MONOTONE_CONSTRAINTS`.
    """
    features = np.asarray(X, dtype=float)
    labels = np.asarray(y, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {features.shape!r}")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(
            "y must be 1-D with one label per row of X, got shapes "
            f"{labels.shape!r} and {features.shape!r}"
        )
    if len(feature_names) != features.shape[1]:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X has "
            f"{features.shape[1]} columns"
        )

    valid = ~np.isnan(labels)
    if not valid.any():
        raise ValueError("all labels are NaN; nothing to train on")
    labels_valid = labels[valid]
    if not np.all(np.isin(labels_valid, (0.0, 1.0))):
        raise ValueError("labels must be 0.0, 1.0, or NaN")

    model = lgb.LGBMClassifier(
        monotone_constraints=monotone_constraints_for(feature_names),
        **_MODEL_PARAMS,
    )
    model.fit(features[valid], labels_valid)
    return model


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between outcomes and predicted probabilities.

    Non-finite entries of either array (e.g. masked NaN labels) are
    excluded from the mean. This matches
    ``sklearn.metrics.brier_score_loss`` on clean binary inputs.
    """
    outcomes = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(y_prob, dtype=float)
    if outcomes.shape != probabilities.shape:
        raise ValueError(
            "y_true and y_prob must have the same shape, got "
            f"{outcomes.shape!r} and {probabilities.shape!r}"
        )
    valid = np.isfinite(outcomes) & np.isfinite(probabilities)
    if not valid.any():
        raise ValueError("no finite (label, probability) pairs to score")
    return float(np.mean((probabilities[valid] - outcomes[valid]) ** 2))
