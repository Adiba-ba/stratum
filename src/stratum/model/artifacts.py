"""Model artifact IO for Project STRATUM.

A trained :class:`stratum.model.calibrate.CalibratedModel` is persisted
together with its metadata (frozen feature names, training information and
validation metrics) in a single joblib payload, so a loaded artifact is
self-describing: the consumer can check the feature contract before
feeding rows into ``predict_proba``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib

from stratum.model.calibrate import CalibratedModel

__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "save_model",
    "load_model",
]

#: Payload format version written by :func:`save_model`.
ARTIFACT_FORMAT_VERSION: int = 1

_PAYLOAD_MODEL_KEY: str = "model"
_PAYLOAD_METADATA_KEY: str = "metadata"
_PAYLOAD_VERSION_KEY: str = "format_version"


def _validate_model(model: Any) -> CalibratedModel:
    """Return ``model`` when it is a calibrated classifier.

    Raises
    ------
    TypeError
        When the object does not expose a ``predict_proba`` method.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "model must expose predict_proba(X) -> (N, 2); got "
            f"{type(model).__name__}"
        )
    return model


def save_model(
    model: CalibratedModel,
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Persist a calibrated model and its metadata to one joblib file.

    Parameters
    ----------
    model : CalibratedModel
        Fitted calibrated classifier.
    path : str or pathlib.Path
        Destination file (conventionally ``*.joblib``). Missing parent
        directories are created.
    metadata : dict
        Artifact metadata stored next to the model: frozen feature names
        (``"feature_names"``), training information and validation metrics
        are the expected keys; the dictionary is stored as given.

    Raises
    ------
    TypeError
        When ``model`` has no ``predict_proba`` or ``metadata`` is not a
        mapping.
    ValueError
        When ``path`` is empty.
    """
    _validate_model(model)
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")
    destination = Path(path)
    if not str(destination):
        raise ValueError("path must be a non-empty file path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        _PAYLOAD_VERSION_KEY: ARTIFACT_FORMAT_VERSION,
        _PAYLOAD_MODEL_KEY: model,
        _PAYLOAD_METADATA_KEY: dict(metadata),
    }
    joblib.dump(payload, destination)


def load_model(path: str | Path) -> tuple[CalibratedModel, dict[str, Any]]:
    """Load a model artifact written by :func:`save_model`.

    Parameters
    ----------
    path : str or pathlib.Path
        Artifact path.

    Returns
    -------
    tuple of (CalibratedModel, dict)
        The model and a copy of its metadata. A bare model payload (no
        metadata wrapper) is accepted and returns an empty metadata
        dictionary.

    Raises
    ------
    FileNotFoundError
        When ``path`` does not exist.
    ValueError
        When the payload is not a model artifact or was written by a
        newer format version.
    """
    payload = joblib.load(path)
    if not isinstance(payload, Mapping) or _PAYLOAD_MODEL_KEY not in payload:
        # Tolerate a bare estimator payload written outside save_model.
        return _validate_model(payload), {}
    version = payload.get(_PAYLOAD_VERSION_KEY, ARTIFACT_FORMAT_VERSION)
    if int(version) > ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"artifact {path} uses format version {version}, but this build "
            f"understands at most {ARTIFACT_FORMAT_VERSION}"
        )
    model = _validate_model(payload[_PAYLOAD_MODEL_KEY])
    metadata = payload.get(_PAYLOAD_METADATA_KEY) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"artifact {path} carries a non-mapping metadata field")
    return model, dict(metadata)
