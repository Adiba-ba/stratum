"""Model-training subcommand for the STRATUM command line.

Trains the monotone LightGBM correction model with isotonic calibration on
a labeled dataset produced by ``build-physics-labels`` or
``build-observed-labels``. The split is time-ordered: rows are sorted by
their provenance timestamp and the last ``--val-fraction`` of them become
the validation set, so calibration and validation metrics are measured on
the newest weather the labels cover instead of a random reshuffle.

The artifact is self-describing: the joblib payload carries the frozen
feature names, the training configuration, the validation metrics, the
training envelope (the pressure range and flight levels the labeled
training rows cover, used by ``stratum profile`` to flag extrapolation)
and the dataset provenance (path, row counts, modification time and
SHA-256). A metrics JSON is written next to the artifact. Nothing is
synthetic here: the dataset is the user's file, and a missing or
unreadable dataset exits with code 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stratum.labels.physics import read_dataset
from stratum.model.artifacts import save_model
from stratum.model.calibrate import CalibratedModel, fit_calibrator
from stratum.model.features import FEATURE_NAMES
from stratum.model.train import (
    MONOTONE_CONSTRAINTS,
    compute_brier_score,
    train_contrail_model,
)
from stratum.physics import thermo
from stratum.pipeline.build_features import pressure_to_altitude_m

__all__ = ["train"]

#: Chunk size [bytes] used when hashing the dataset for provenance.
_HASH_CHUNK_BYTES: int = 1 << 20

#: Features whose monotone response is verified after calibration.
_MONOTONE_CHECK_FEATURES: tuple[str, ...] = ("rhi", "tlm_k")

#: Points per feature sweep used by the monotonicity check.
_SWEEP_POINTS: int = 50

#: Flight-level quantum [hundreds of ft] used by the training envelope.
_FLIGHT_LEVEL_QUANTUM: int = 10


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sweep(
    model: CalibratedModel,
    baseline: np.ndarray,
    index: int,
    values: np.ndarray,
) -> np.ndarray:
    """Calibrated positive-class probabilities along one feature sweep.

    The sweep holds every other feature at the training median and moves
    one feature across ``values``, the same probe used by
    ``scripts/demo_model.py``.
    """
    probabilities = np.empty(values.size, dtype=float)
    for position, value in enumerate(values):
        row = baseline.copy()
        row[index] = value
        probabilities[position] = model.predict_proba(row)[0, 1]
    return probabilities


def _monotone_checks(
    model: CalibratedModel, features_labeled: np.ndarray
) -> dict[str, bool]:
    """Verify the constrained feature directions on the calibrated model.

    ``rhi`` and ``tlm_k`` both carry a ``+1`` monotone constraint in
    :data:`stratum.model.train.MONOTONE_CONSTRAINTS`, so the calibrated
    probability must be non-decreasing in both. Isotonic calibration is
    non-decreasing, so the check is meaningful on the calibrated output.
    """
    baseline = np.nanmedian(features_labeled, axis=0)
    checks: dict[str, bool] = {}
    for name in _MONOTONE_CHECK_FEATURES:
        index = FEATURE_NAMES.index(name)
        if name == "rhi":
            values = np.linspace(0.1, 1.5, _SWEEP_POINTS)
        else:
            values = np.linspace(
                baseline[index] - 10.0, baseline[index] + 10.0, _SWEEP_POINTS
            )
        probabilities = _sweep(model, baseline, index, values)
        if MONOTONE_CONSTRAINTS.get(name, 0) >= 0:
            checks[name] = bool(np.all(np.diff(probabilities) >= -1.0e-9))
        else:
            checks[name] = bool(np.all(np.diff(probabilities) <= 1.0e-9))
    return checks


def _flight_level_of_pressure(pressure_pa: float) -> float:
    """ISA flight level [hundreds of ft] of an ambient pressure [Pa]."""
    return pressure_to_altitude_m(pressure_pa) / (100.0 * thermo.FT_TO_M)


def _rounded_flight_level(pressure_pa: float) -> int:
    """Flight level of a pressure, rounded to the nearest quantum [100 ft]."""
    level = _flight_level_of_pressure(pressure_pa) / _FLIGHT_LEVEL_QUANTUM
    return int(round(level)) * _FLIGHT_LEVEL_QUANTUM


def _mean_probability(probabilities: np.ndarray, mask: np.ndarray) -> float:
    """Mean predicted probability over the rows selected by ``mask``."""
    selected = np.asarray(probabilities)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        return float("nan")
    return float(np.mean(selected))


def _training_envelope(
    metadata: list[dict[str, Any]],
    train_idx: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
) -> dict[str, Any] | None:
    """Summarise the pressure range the model's training rows cover.

    The envelope is taken from the ``pressure_pa`` provenance of every
    training-split row that is backed by data, i.e. rows with a finite
    label or, when the label is censored (``nan``), rows whose feature
    vector is fully finite. Censored rows with finite features still
    describe the model's input domain, because the physics censoring
    (no Schmidt-Appleman formation, ``RHi <= 1``) says nothing about
    data availability. Rows without data and validation rows never
    widen the envelope.

    Parameters
    ----------
    metadata : list of dict
        One provenance entry per dataset row, as returned by
        :func:`stratum.labels.physics.read_dataset`.
    train_idx : numpy.ndarray
        Indices of the training split.
    labels : numpy.ndarray
        Label vector in dataset row order.
    features : numpy.ndarray
        Feature matrix in dataset row order, with ``nan`` rows for
        samples that no ERA5 state backs.

    Returns
    -------
    dict or None
        ``{"pressure_pa_min": float, "pressure_pa_max": float,
        "flight_levels": sorted list of int}`` with the flight levels
        rounded to the nearest :data:`_FLIGHT_LEVEL_QUANTUM`. ``None``
        when no data-backed training row carries a finite
        ``pressure_pa``, e.g. when the dataset was written without the
        provenance column.
    """
    if len(metadata) != labels.shape[0]:
        return None
    pressures = np.full(labels.shape[0], float("nan"))
    for index, entry in enumerate(metadata):
        try:
            pressures[index] = float(entry.get("pressure_pa"))
        except (TypeError, ValueError):
            pressures[index] = float("nan")

    backed = np.isfinite(labels[train_idx]) | np.all(
        np.isfinite(features[train_idx]), axis=1
    )
    selected = pressures[train_idx][backed]
    selected = selected[np.isfinite(selected) & (selected > 0.0)]
    if selected.size == 0:
        return None

    levels = sorted({_rounded_flight_level(pressure) for pressure in selected})
    return {
        "pressure_pa_min": float(selected.min()),
        "pressure_pa_max": float(selected.max()),
        "flight_levels": [int(level) for level in levels],
    }


def train(argv: list[str]) -> int:
    """Train and calibrate the correction model on a labeled CSV.

    Parameters
    ----------
    argv : list of str
        Arguments after ``train``: ``--dataset`` and ``--out`` are
        required; ``--val-fraction`` and ``--min-train-rows`` are
        optional.

    Returns
    -------
    int
        0 when the model trains and the monotonicity checks pass, 1 when
        the split has too little labeled data or a check fails, 2 for a
        missing or unreadable dataset and malformed options.

    Notes
    -----
    The artifact metadata and the metrics JSON record a
    ``training_envelope`` derived from the labeled training rows: the
    pressure range and the ISA flight levels they cover. ``stratum
    profile`` reads it to flag levels that extrapolate beyond the
    training data. Datasets without a usable ``pressure_pa`` provenance
    column record ``training_envelope: None`` together with a warning.
    """
    parser = argparse.ArgumentParser(
        prog="stratum train",
        description=(
            "Train the monotone LightGBM correction model with isotonic "
            "calibration on a labeled dataset; the time-ordered tail of "
            "the rows becomes the validation split."
        ),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        metavar="CSV",
        help="labeled CSV from build-physics-labels or build-observed-labels",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="MODEL.joblib",
        help="output model artifact (metrics JSON is written next to it)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.3,
        help="fraction of the time-ordered rows used for validation (default 0.3)",
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=50,
        help="minimum labeled training rows required (default 50)",
    )
    parser.add_argument(
        "--min-val-positive",
        type=int,
        default=20,
        help=(
            "minimum persistent rows the validation block must hold "
            "(default 20); the block is the last contiguous stretch of "
            "time steps with enough labeled and persistent rows"
        ),
    )
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        print(
            f"ERROR: dataset not found: {args.dataset}\n"
            "Build one first with `stratum build-physics-labels` or "
            "`stratum build-observed-labels`; nothing is synthesized.",
            file=sys.stderr,
        )
        return 2
    if not 0.0 < args.val_fraction < 1.0:
        print(
            f"ERROR: --val-fraction must lie strictly inside (0, 1), got "
            f"{args.val_fraction}",
            file=sys.stderr,
        )
        return 2
    if args.min_train_rows < 1:
        print(
            f"ERROR: --min-train-rows must be >= 1, got {args.min_train_rows}",
            file=sys.stderr,
        )
        return 2

    try:
        features, labels, metadata = read_dataset(args.dataset)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    n_rows = int(labels.shape[0])
    if n_rows < 2:
        print(
            f"ERROR: dataset {args.dataset} has {n_rows} row(s); at least 2 "
            "are needed for a train/validation split.",
            file=sys.stderr,
        )
        return 2

    try:
        times = pd.to_datetime(
            [str(entry["time"]) for entry in metadata], utc=True
        )
    except (TypeError, ValueError) as exc:
        print(
            f"ERROR: dataset {args.dataset} carries unparsable provenance "
            f"times: {exc}",
            file=sys.stderr,
        )
        return 2
    if bool(pd.isna(times).any()):
        print(
            f"ERROR: dataset {args.dataset} carries missing provenance times.",
            file=sys.stderr,
        )
        return 2

    single_time_step = bool(times.nunique() == 1)
    if single_time_step:
        print(
            "WARNING: the dataset contains a single unique time step, so a "
            "time-ordered split cannot order anything; falling back to a "
            "deterministic row-order split (the file's last rows validate).",
            file=sys.stderr,
        )
        order = np.arange(n_rows)
    else:
        order = times.argsort(kind="stable")

    n_val = int(round(args.val_fraction * n_rows))
    n_val = max(1, min(n_rows - 1, n_val))
    min_val_labeled = max(int(args.min_train_rows), 50)
    min_val_positive = max(int(args.min_val_positive), 5)

    # A blind time-ordered tail can hold almost no labeled rows when the
    # physics censors most of the period: contrail formation is
    # synoptic, so a cold spell leaves whole days without a single
    # labeled sample. The validation block is therefore the last
    # contiguous stretch of time steps that carries enough labeled rows
    # and at least a few persistent ones, and training uses the rows
    # strictly before it. No later row ever enters training, so the
    # split stays free of time leakage.
    unique_times = np.unique(times[order])
    labeled_by_time = np.array(
        [int(np.isfinite(labels[order[times[order] == value]]).sum()) for value in unique_times]
    )
    positive_by_time = np.array(
        [int((labels[order[times[order] == value]] == 1.0).sum()) for value in unique_times]
    )
    suffix_labeled = 0
    suffix_positive = 0
    val_start = len(unique_times)
    for position in range(len(unique_times) - 1, -1, -1):
        suffix_labeled += int(labeled_by_time[position])
        suffix_positive += int(positive_by_time[position])
        if suffix_labeled >= min_val_labeled and suffix_positive >= min_val_positive:
            val_start = position
            break
    if val_start >= len(unique_times):
        print(
            f"ERROR: no trailing block of time steps carries both "
            f"{min_val_labeled} labeled and {min_val_positive} persistent "
            f"rows; the dataset has {int(np.isfinite(labels).sum())} labeled "
            f"and {int((labels == 1.0).sum())} persistent rows in total. "
            "Lower --min-val-positive, label a longer period, or use a "
            "dataset with more formation.",
            file=sys.stderr,
        )
        return 1

    val_times = set(unique_times[val_start:])
    in_val = np.array([value in val_times for value in times[order]])
    val_idx = np.asarray(order[in_val], dtype=int)
    train_idx = np.asarray(order[~in_val], dtype=int)

    train_labeled = np.isfinite(labels[train_idx])
    n_train_labeled = int(train_labeled.sum())
    n_val_labeled = int(np.isfinite(labels[val_idx]).sum())
    n_masked = int(np.isnan(labels).sum())

    if n_train_labeled < args.min_train_rows:
        print(
            f"ERROR: only {n_train_labeled} labeled training row(s); "
            f"--min-train-rows is {args.min_train_rows}. Label more waypoints "
            "or lower --min-train-rows.",
            file=sys.stderr,
        )
        return 1
    if n_val_labeled == 0:
        print(
            "ERROR: the validation split contains no finite labels; nothing "
            "to calibrate or score. Rebuild the dataset or adjust "
            "--val-fraction.",
            file=sys.stderr,
        )
        return 1

    warnings.filterwarnings(
        "ignore",
        message=(
            "X does not have valid feature names, but LGBMClassifier was "
            "fitted with feature names"
        ),
    )

    try:
        model = train_contrail_model(
            features[train_idx], labels[train_idx], list(FEATURE_NAMES)
        )
        calibrated = fit_calibrator(model, features[val_idx], labels[val_idx])
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    valid_val = np.isfinite(labels[val_idx])
    features_val = features[val_idx][valid_val]
    labels_val = labels[val_idx][valid_val]
    raw_probabilities = calibrated.predict_raw_proba(features_val)
    calibrated_probabilities = calibrated.predict_proba(features_val)[:, 1]
    metrics: dict[str, Any] = {
        "n_samples": n_rows,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_train_labeled": n_train_labeled,
        "n_val_labeled": n_val_labeled,
        "n_masked": n_masked,
        "brier_raw": compute_brier_score(labels_val, raw_probabilities),
        "brier_calibrated": compute_brier_score(labels_val, calibrated_probabilities),
        "n_val_positive": int((labels_val == 1.0).sum()),
        "n_val_negative": int((labels_val == 0.0).sum()),
        "mean_probability_positive": _mean_probability(
            calibrated_probabilities, labels_val == 1.0
        ),
        "mean_probability_negative": _mean_probability(
            calibrated_probabilities, labels_val == 0.0
        ),
    }

    monotone = _monotone_checks(calibrated, features[train_idx][train_labeled])
    monotone_passed = bool(all(monotone.values()))

    training_envelope = _training_envelope(metadata, train_idx, labels, features)
    if training_envelope is None:
        print(
            "WARNING: the dataset carries no usable pressure_pa provenance in "
            "its data-backed training rows, so training_envelope is not "
            "recorded; `stratum profile` cannot check requested levels against "
            "the training range.",
            file=sys.stderr,
        )

    stats = args.dataset.stat()
    metadata_payload: dict[str, Any] = {
        "feature_names": list(FEATURE_NAMES),
        "dataset": str(args.dataset),
        "training_envelope": training_envelope,
        "training": {
            "val_fraction": float(args.val_fraction),
            "min_train_rows": int(args.min_train_rows),
            "time_ordered_split": not single_time_step,
            "single_time_step": single_time_step,
            "n_samples": n_rows,
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
            "n_masked": n_masked,
            "monotone_checks": monotone,
        },
        "metrics": metrics,
        "provenance": {
            "dataset_path": str(args.dataset),
            "dataset_size_bytes": int(stats.st_size),
            "dataset_mtime_utc": datetime.fromtimestamp(
                stats.st_mtime, timezone.utc
            ).isoformat(),
            "dataset_sha256": _sha256(args.dataset),
        },
    }

    save_model(calibrated, args.out, metadata_payload)
    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Model artifact written: {args.out}")
    print(f"Metrics JSON written  : {metrics_path}")
    print(
        f"Rows: {n_rows} total | {int(train_idx.size)} train "
        f"({n_train_labeled} labeled) | {int(val_idx.size)} val "
        f"({n_val_labeled} labeled) | {n_masked} NaN-masked"
    )
    if training_envelope is None:
        print("Training envelope: not recorded (no pressure_pa provenance)")
    else:
        envelope_levels = ", ".join(
            f"FL{level}" for level in training_envelope["flight_levels"]
        )
        print(
            "Training envelope: pressure "
            f"{training_envelope['pressure_pa_min']:.0f}.."
            f"{training_envelope['pressure_pa_max']:.0f} Pa | "
            f"{envelope_levels}"
        )
    print(
        f"Validation Brier: raw {metrics['brier_raw']:.5f} -> calibrated "
        f"{metrics['brier_calibrated']:.5f}"
    )
    print(
        f"Validation classes: {metrics['n_val_positive']} persistent, "
        f"{metrics['n_val_negative']} subsaturated | mean probability "
        f"{metrics['mean_probability_positive']:.3f} on persistent rows, "
        f"{metrics['mean_probability_negative']:.3f} on subsaturated rows"
    )
    check_text = " | ".join(
        f"{name} {'PASS' if passed else 'FAIL'}" for name, passed in monotone.items()
    )
    print(f"Monotonicity (calibrated): {check_text}")

    if not monotone_passed:
        print(
            "ERROR: at least one monotone constraint was violated by the "
            "calibrated model; the artifact and metrics were written anyway "
            "for inspection.",
            file=sys.stderr,
        )
        return 1
    return 0
