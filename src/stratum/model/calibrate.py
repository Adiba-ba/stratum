"""Isotonic probability calibration for the STRATUM correction model.

Wraps a fitted binary classifier together with an isotonic regression map
fitted on validation predictions, so the calibrated output is a true
probability in ``[0.0, 1.0]``. Isotonic regression is monotonically
non-decreasing, therefore the physical monotone constraints of the base
model are preserved by calibration.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression

__all__ = ["CalibratedModel", "fit_calibrator"]


class PredictProbaModel(Protocol):
    """Protocol for estimators exposing sklearn-style ``predict_proba``."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape ``(n_samples, 2)``."""
        ...


class CalibratedModel:
    """Isotonic-calibrated wrapper around a binary classifier.

    Parameters
    ----------
    model : PredictProbaModel
        Fitted base classifier.
    calibrator : IsotonicRegression
        Isotonic map fitted on the base model's positive-class
        probabilities.
    """

    def __init__(
        self,
        model: PredictProbaModel,
        calibrator: IsotonicRegression,
    ) -> None:
        self._model = model
        self._calibrator = calibrator

    @property
    def model(self) -> PredictProbaModel:
        """The wrapped base classifier."""
        return self._model

    @property
    def calibrator(self) -> IsotonicRegression:
        """The fitted isotonic calibration map."""
        return self._calibrator

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability of the base model, shape ``(n,)``."""
        features = np.atleast_2d(np.asarray(X, dtype=float))
        raw = np.asarray(self._model.predict_proba(features), dtype=float)
        return raw[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated class probabilities of shape ``(n_samples, 2)``.

        Column 1 holds the calibrated probability of persistence.
        Predictions outside the fitted range are clipped, so the output
        always lies within ``[0.0, 1.0]``.
        """
        positive = self._calibrator.predict(self.predict_raw_proba(X))
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary decisions from the calibrated positive probability."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def fit_calibrator(
    model: PredictProbaModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> CalibratedModel:
    """Fit isotonic calibration on non-NaN validation rows.

    Parameters
    ----------
    model : PredictProbaModel
        Fitted base classifier.
    X_val : np.ndarray
        Validation features, shape ``(n_samples, n_features)``.
    y_val : np.ndarray
        Validation labels; NaN rows (unobservable scenes) are excluded
        from the isotonic fit.

    Returns
    -------
    CalibratedModel
        Wrapper whose ``predict_proba`` returns calibrated probabilities
        clipped to ``[0.0, 1.0]``.
    """
    features = np.asarray(X_val, dtype=float)
    labels = np.asarray(y_val, dtype=float)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError(
            "X_val and y_val shapes do not line up: "
            f"{features.shape!r} and {labels.shape!r}"
        )
    valid = ~np.isnan(labels)
    if not valid.any():
        raise ValueError("all validation labels are NaN; nothing to calibrate on")

    raw = np.asarray(model.predict_proba(features[valid]), dtype=float)[:, 1]
    calibrator = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=0.0,
        y_max=1.0,
    )
    calibrator.fit(raw, labels[valid])
    return CalibratedModel(model, calibrator)
