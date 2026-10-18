"""Correction-model subpackage for Project STRATUM.

Re-exports the frozen feature schema, the monotone LightGBM training
helper, and the isotonic calibration wrapper.
"""

from stratum.model.calibrate import CalibratedModel, fit_calibrator
from stratum.model.features import (
    FEATURE_NAMES,
    encode_cyclical,
    extract_features_row,
    feature_vector,
)
from stratum.model.train import compute_brier_score, train_contrail_model

__all__ = [
    "FEATURE_NAMES",
    "encode_cyclical",
    "extract_features_row",
    "feature_vector",
    "train_contrail_model",
    "compute_brier_score",
    "CalibratedModel",
    "fit_calibrator",
]
