"""Smoke tests for the core STRATUM CLI workflows.

Every fixture is synthetic and every command runs offline on temporary
files: a synthetic ERA5 field feeds ``build-physics-labels``, ``train``
and ``corridor``, and a synthetic split-window scene feeds ``detect-scene``
through the registered command line entry point.

The ERA5 field geometry mirrors ``tests/test_io.py`` (levels 200/250/300
hPa, a 15-30 N / 70-80 E grid, two time steps). Its humidity field is
tuned so the VIDP-VABB cruise levels split into both label classes. The
FL370 point sample (~217 hPa) stays inside the file's pressure range, so
those rows are labeled too even though their upper ``-25 hPa`` probe
leaves the topmost stored level (200 hPa) and turns the probe-derived
features into NaN.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stratum.cli import corridor as corridor_cli
from stratum.cli import labels as labels_cli
from stratum.cli import scene as scene_cli
from stratum.cli import train as train_cli
from stratum.cli.main import main
from stratum.model.artifacts import load_model
from stratum.model.features import FEATURE_NAMES
from stratum.physics import e_i

# The netCDF4 wheel can emit a spurious NumPy binary-compatibility
# RuntimeWarning on first engine use (see src/stratum/pipeline/runner.py).
pytestmark = pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed.*:RuntimeWarning"
)

ERA5_LATS = np.array([30.0, 25.0, 20.0, 15.0])  # descending; loader must flip
ERA5_LONS = np.array([70.0, 75.0, 80.0])
ERA5_LEVELS = np.array([200.0, 250.0, 300.0])
ERA5_TIMES = np.array(
    ["2025-01-01T00:00", "2025-01-01T06:00"], dtype="datetime64[ns]"
)

#: Specific-humidity conversion constant used by the synthetic field.
_EPS = 0.622

_E_I_VECTORIZED = np.vectorize(e_i)


class PhysicsDataset(NamedTuple):
    """Paths and output of the one-time ``build-physics-labels`` fixture."""

    era5: Path
    labels: Path
    stdout: str


def _temperature(level: float, lat: float, lon: float, t: float) -> np.ndarray:
    """Linear temperature field monotone in pressure level."""
    return (
        200.0
        + 0.30 * (level - 200.0)
        + 0.2 * (lat - 15.0)
        + 0.05 * (lon - 70.0)
        + 0.3 * t
    )


def _target_rhi(lon: float, t: float) -> np.ndarray:
    """Target ice-relative humidity spanning both label classes."""
    return 0.55 + 0.05 * (lon - 70.0) + 0.02 * t


def _specific_humidity(
    level: float, lat: float, lon: float, t: float
) -> np.ndarray:
    """Moisture that realizes :func:`_target_rhi` over the ice curve."""
    temperature = _temperature(level, lat, lon, t)
    return _target_rhi(lon, t) * _E_I_VECTORIZED(temperature) * _EPS / (level * 100.0)


def _synthetic_era5() -> xr.Dataset:
    """Build a linear-in-all-coordinates ERA5-like dataset for testing."""
    level = ERA5_LEVELS[:, None, None, None]
    lat = ERA5_LATS[None, :, None, None]
    lon = ERA5_LONS[None, None, :, None]
    t_index = np.arange(ERA5_TIMES.size, dtype=float)[None, None, None, :]
    fields = {
        "temperature": _temperature(level, lat, lon, t_index),
        "specific_humidity": _specific_humidity(level, lat, lon, t_index),
        "u_component_of_wind": (
            5.0 + 0.5 * (lat - 15.0) - 0.1 * (level - 200.0) + 0.2 * t_index
        ),
        "v_component_of_wind": (
            -2.0 + 0.1 * (lon - 70.0) + 0.05 * (lat - 15.0) + 0.1 * t_index
        ),
        "vertical_velocity": (
            0.01 * (level - 200.0) - 0.002 * (lat - 15.0) + 0.001 * t_index
        ),
        "fraction_of_cloud_cover": (
            0.1 + 0.01 * (lat - 15.0) + 0.02 * (lon - 70.0)
            + 0.005 * (level - 200.0)
        ),
    }
    shape = (
        ERA5_LEVELS.size,
        ERA5_LATS.size,
        ERA5_LONS.size,
        ERA5_TIMES.size,
    )
    data_vars = {
        name: (
            ("time", "level", "latitude", "longitude"),
            np.broadcast_to(field, shape).transpose(3, 0, 1, 2),
        )
        for name, field in fields.items()
    }
    return xr.Dataset(
        data_vars,
        coords={
            "time": ERA5_TIMES,
            "level": ERA5_LEVELS,
            "latitude": ERA5_LATS,
            "longitude": ERA5_LONS,
        },
    )


def _scene_dataset() -> xr.Dataset:
    """Split-window scene with 2-D navigation and one warm line."""
    rows, cols = 12, 14
    lats = np.linspace(25.0, 23.0, rows)[:, None] * np.ones((1, cols))
    lons = np.linspace(70.0, 73.0, cols)[None, :] * np.ones((rows, 1))
    bt_108 = np.full((rows, cols), 250.0)
    bt_120 = np.full((rows, cols), 250.0)
    bt_108[5, :] += 4.0  # a horizontal contrail-like BTD line
    return xr.Dataset(
        data_vars={
            "IR_108": (("row", "col"), bt_108),
            "IR_120": (("row", "col"), bt_120),
            "latitude": (("row", "col"), lats),
            "longitude": (("row", "col"), lons),
        },
        coords={"row": np.arange(rows), "col": np.arange(cols)},
    )


@pytest.fixture(scope="module")
def physics_dataset(tmp_path_factory: pytest.TempPathFactory) -> PhysicsDataset:
    """Synthetic ERA5 file plus the labeled CSV built from it by the CLI."""
    directory = tmp_path_factory.mktemp("stratum_cli_core")
    era5_path = directory / "synthetic_era5.nc"
    _synthetic_era5().to_netcdf(era5_path)

    labels_path = directory / "physics_labels.csv"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = labels_cli.build_physics_labels(
            [
                "--era5",
                str(era5_path),
                "--routes",
                "VIDP-VABB",
                "--levels",
                "330,350,370",
                "--out",
                str(labels_path),
                "--max-times",
                "2",
            ]
        )
    assert code == 0, buffer.getvalue()
    assert labels_path.is_file()
    return PhysicsDataset(era5=era5_path, labels=labels_path, stdout=buffer.getvalue())


# ---------------------------------------------------------------------------
# build-physics-labels
# ---------------------------------------------------------------------------


def test_build_physics_labels_writes_dataset(physics_dataset: PhysicsDataset) -> None:
    frame = pd.read_csv(physics_dataset.labels)
    assert set(FEATURE_NAMES) <= set(frame.columns)
    assert {"label", "lat", "lon", "pressure_pa", "time"} <= set(frame.columns)
    assert len(frame) == 150  # 25 waypoints x 3 levels x 2 time steps

    # Every level carries a usable point sample now: FL370 (~217 hPa) is
    # inside the 200-300 hPa file range, only its upper probe is not. So
    # all 150 rows are labeled and none is censored; the previous
    # all-or-nothing probe rule produced 59/41/50 with 50 censored rows.
    labels = frame["label"].to_numpy(dtype=float)
    assert int((labels == 1.0).sum()) == 109
    assert int((labels == 0.0).sum()) == 41
    assert int(np.isnan(labels).sum()) == 0

    stdout = physics_dataset.stdout
    assert "Label 1 (persistent)   : 109" in stdout
    assert "Label 0 (subsaturated) : 41" in stdout
    assert "Censored (NaN)         : 0" in stdout
    assert "ERA5 data available    : 150" in stdout
    assert "Censoring reasons" not in stdout


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_writes_artifact_and_metrics(
    physics_dataset: PhysicsDataset, tmp_path: Path
) -> None:
    artifact = tmp_path / "model.joblib"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = train_cli.train(
            ["--dataset", str(physics_dataset.labels), "--out", str(artifact)]
        )
    assert code == 0, buffer.getvalue()
    assert artifact.is_file()

    metrics_path = artifact.with_suffix(".metrics.json")
    assert metrics_path.is_file()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["feature_names"] == list(FEATURE_NAMES)
    assert payload["dataset"] == str(physics_dataset.labels)
    assert payload["training"]["time_ordered_split"] is True
    assert payload["training"]["monotone_checks"] == {"rhi": True, "tlm_k": True}
    assert len(payload["provenance"]["dataset_sha256"]) == 64

    metrics = payload["metrics"]
    assert math.isfinite(metrics["brier_raw"])
    assert math.isfinite(metrics["brier_calibrated"])
    # The validation block is the last contiguous stretch of time steps
    # that carries enough labeled and persistent rows; the fixture holds
    # two time steps, so the block is the later one and training keeps
    # the earlier one.
    assert metrics["n_train"] + metrics["n_val"] == metrics["n_samples"]
    assert metrics["n_val"] == 75
    assert metrics["n_train"] == 75
    assert metrics["n_masked"] == 0
    assert metrics["n_val_labeled"] >= 50
    assert metrics["n_val_positive"] >= 20
    assert metrics["n_val_positive"] + metrics["n_val_negative"] == metrics[
        "n_val_labeled"
    ]

    # The training envelope records the range the labeled training rows
    # cover: FL330/FL350/FL370, i.e. the ISA pressures of those levels.
    envelope = payload["training_envelope"]
    assert envelope is not None
    assert math.isfinite(envelope["pressure_pa_min"])
    assert math.isfinite(envelope["pressure_pa_max"])
    assert envelope["pressure_pa_min"] < envelope["pressure_pa_max"]
    assert envelope["flight_levels"] == [330, 350, 370]

    model, metadata = load_model(artifact)
    assert hasattr(model, "predict_proba")
    assert metadata["training_envelope"] == envelope
    assert metadata["metrics"]["brier_calibrated"] == metrics["brier_calibrated"]
    assert "rhi PASS | tlm_k PASS" in buffer.getvalue()


def test_training_envelope_ignores_rows_without_pressure_provenance() -> None:
    labels = np.array([1.0, float("nan")])
    train_idx = np.array([0, 1])
    # Row 0 is labeled, row 1 is censored but backed by finite features,
    # so both describe the model's input domain.
    features = np.array([[0.1, 0.2], [0.3, 0.4]])
    unbacked_features = np.array([[0.1, 0.2], [float("nan"), float("nan")]])

    # No pressure_pa provenance at all: the envelope is unknown, so the
    # caller records None and warns (the CSV column provides it in the
    # production datasets).
    assert (
        train_cli._training_envelope(
            [{"lat": 28.0}, {"lat": 28.0}], train_idx, labels, features
        )
        is None
    )

    # Censored rows without data must not widen the envelope: only the
    # labeled row contributes, so the bounds are that single pressure.
    envelope = train_cli._training_envelope(
        [{"lat": 28.0, "pressure_pa": 23846.0}, {"lat": 28.0}],
        train_idx,
        labels,
        unbacked_features,
    )
    assert envelope is not None
    assert envelope["pressure_pa_min"] == pytest.approx(23846.0)
    assert envelope["pressure_pa_max"] == pytest.approx(23846.0)
    assert envelope["flight_levels"] == [350]

    # A censored row with finite features still widens the envelope.
    envelope = train_cli._training_envelope(
        [
            {"lat": 28.0, "pressure_pa": 23846.0},
            {"lat": 20.0, "pressure_pa": 26201.0},
        ],
        train_idx,
        labels,
        features,
    )
    assert envelope is not None
    assert envelope["pressure_pa_min"] == pytest.approx(23846.0)
    assert envelope["pressure_pa_max"] == pytest.approx(26201.0)
    assert envelope["flight_levels"] == [330, 350]


def test_train_missing_dataset_exits_2(tmp_path: Path) -> None:
    code = train_cli.train(
        ["--dataset", str(tmp_path / "absent.csv"), "--out", str(tmp_path / "m.joblib")]
    )
    assert code == 2


# ---------------------------------------------------------------------------
# corridor
# ---------------------------------------------------------------------------


def test_corridor_physics_only_route(
    physics_dataset: PhysicsDataset, tmp_path: Path
) -> None:
    json_path = tmp_path / "corridor.json"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = corridor_cli.corridor(
            [
                "--era5",
                str(physics_dataset.era5),
                "--route",
                "VIDP-VABB",
                "--json",
                str(json_path),
            ]
        )
    assert code == 0, buffer.getvalue()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["mode"] == "physics_only"
    assert payload["model"] is None
    assert payload["trajectory_source"] == "great_circle_profile"
    assert payload["num_waypoints"] == 25
    assert payload["departure_assumed"] is True
    segments = payload["segments"]
    assert len(segments) == 25
    # Physics-only mode must not fabricate persistence probabilities or
    # decisions: no calibrated model ran.
    assert all("p_persist" not in segment for segment in segments)
    assert all("decision_status" not in segment for segment in segments)
    assert "result" not in payload
    assert payload["summary"]["total_waypoints"] == 25

    stdout = buffer.getvalue()
    assert "NOT a real flight track" in stdout
    assert "PHYSICS-ONLY MODE" in stdout
    assert "JSON written" in stdout


def test_corridor_missing_era5_exits_2(tmp_path: Path) -> None:
    code = corridor_cli.corridor(
        ["--era5", str(tmp_path / "absent.nc"), "--route", "VIDP-VABB"]
    )
    assert code == 2


# ---------------------------------------------------------------------------
# detect-scene
# ---------------------------------------------------------------------------


def test_detect_scene_writes_summary_via_main(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.nc"
    _scene_dataset().to_netcdf(scene_path)
    out_dir = tmp_path / "detections"

    assert (
        main(["detect-scene", "--satellite", str(scene_path), "--out-dir", str(out_dir)])
        == 0
    )

    summary_path = out_dir / "summary.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(payload["num_detected"], int)
    assert payload["georeferencing"] == "latlon_arrays"
    assert payload["num_detected"] >= 1
    assert len(payload["objects"]) == payload["num_detected"]
    first = payload["objects"][0]
    assert math.isfinite(first["lat"]) and math.isfinite(first["lon"])
    assert first["heading_deg"] is None or math.isfinite(first["heading_deg"])


def test_detect_scene_missing_file_exits_2(tmp_path: Path) -> None:
    assert (
        scene_cli.detect_scene(["--satellite", str(tmp_path / "absent.nc")]) == 2
    )


# ---------------------------------------------------------------------------
# build-observed-labels
# ---------------------------------------------------------------------------


def test_build_observed_labels_missing_inputs_exits_2(tmp_path: Path) -> None:
    code = labels_cli.build_observed_labels(
        [
            "--satellite",
            str(tmp_path / "absent.nc"),
            "--trajectory",
            str(tmp_path / "absent.csv"),
            "--era5",
            str(tmp_path / "absent_era5.nc"),
            "--out",
            str(tmp_path / "observed.csv"),
        ]
    )
    assert code == 2


def test_train_reports_when_no_validation_block_can_be_scored(
    physics_dataset: PhysicsDataset, tmp_path: Path
) -> None:
    """A tail without persistent rows cannot be validated; say so plainly."""
    frame = pd.read_csv(physics_dataset.labels)
    latest = pd.to_datetime(frame["time"]).max()
    frame.loc[pd.to_datetime(frame["time"]) == latest, "label"] = float("nan")
    dataset = tmp_path / "no_tail_labels.csv"
    frame.to_csv(dataset, index=False)

    artifact = tmp_path / "model.joblib"
    code = train_cli.train(["--dataset", str(dataset), "--out", str(artifact)])
    assert code == 1
    assert not artifact.exists()


def test_validation_block_holds_later_times_than_training(
    physics_dataset: PhysicsDataset, tmp_path: Path
) -> None:
    """The split must not leak later times into training."""
    artifact = tmp_path / "model.joblib"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = train_cli.train(
            ["--dataset", str(physics_dataset.labels), "--out", str(artifact)]
        )
    assert code == 0, buffer.getvalue()
    payload = json.loads(
        artifact.with_suffix(".metrics.json").read_text(encoding="utf-8")
    )
    assert payload["training"]["time_ordered_split"] is True
    assert payload["metrics"]["n_val_labeled"] >= 50
    assert payload["metrics"]["n_val_positive"] >= 20
