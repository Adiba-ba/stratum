"""Offline tests for the ``stratum profile`` subcommand.

A small synthetic ERA5 field (the geometry of ``tests/test_io.py``) feeds
the handler together with a tiny calibrated model artifact. The artifact is
trained from a hand-made feature matrix in the frozen
:data:`stratum.model.features.FEATURE_NAMES` order, so the test needs no
real data and no network. The humidity field increases eastwards: at FL330
the western waypoints are subsaturated (no Schmidt-Appleman formation)
while the eastern ones are not, which exercises the zero-risk rule.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import xarray as xr

from stratum.cli import profile as profile_cli
from stratum.cli.main import main
from stratum.model.artifacts import save_model
from stratum.model.calibrate import fit_calibrator
from stratum.model.features import FEATURE_NAMES
from stratum.model.train import train_contrail_model
from stratum.physics import e_i, fl_to_p

# The netCDF4 wheel can emit a spurious NumPy binary-compatibility
# RuntimeWarning on first engine use (see tests/test_cli_core.py).
pytestmark = pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed.*:RuntimeWarning"
)

ERA5_LATS = np.array([30.0, 25.0, 20.0, 15.0])  # descending; loader must flip
ERA5_LONS = np.array([70.0, 75.0, 80.0])
ERA5_LEVELS = np.array([200.0, 250.0, 300.0])
ERA5_TIMES = np.array(
    ["2025-01-01T00:00", "2025-01-01T06:00"], dtype="datetime64[ns]"
)

#: Training envelope of the fixture model: FL330..FL350.
_TRAINING_ENVELOPE = {
    "pressure_pa_min": fl_to_p(350.0) * 100.0,
    "pressure_pa_max": fl_to_p(330.0) * 100.0,
    "flight_levels": [330, 350],
}

#: Envelope that excludes every level the synthetic file can serve.
_OUT_OF_RANGE_ENVELOPE = {
    "pressure_pa_min": fl_to_p(390.0) * 100.0,
    "pressure_pa_max": fl_to_p(370.0) * 100.0,
    "flight_levels": [370, 390],
}

#: Specific-humidity conversion constant used by the synthetic field.
_EPS = 0.622

_E_I_VECTORIZED = np.vectorize(e_i)


class ProfileRun(NamedTuple):
    """Synthetic inputs and the output of one completed profile run."""

    era5: Path
    model: Path
    code: int
    payload: dict
    csv: Path
    stdout: str


def _temperature(level: float, lat: float, lon: float, t: float) -> np.ndarray:
    """Linear temperature field monotone in pressure level."""
    return (
        200.0
        + 0.4 * (level - 200.0)
        + 0.2 * (lat - 15.0)
        + 0.05 * (lon - 70.0)
        + 0.3 * t
    )


def _target_rhi(lon: float) -> np.ndarray:
    """Ice-relative humidity increasing eastwards within [0.05, 1.6]."""
    return np.clip(0.05 + 0.16 * (lon - 70.0), 0.05, 1.6)


def _specific_humidity(
    level: float, lat: float, lon: float, t: float
) -> np.ndarray:
    """Moisture that realizes :func:`_target_rhi` over the ice curve."""
    temperature = _temperature(level, lat, lon, t)
    return _target_rhi(lon) * _E_I_VECTORIZED(temperature) * _EPS / (level * 100.0)


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
            0.1
            + 0.01 * (lat - 15.0)
            + 0.02 * (lon - 70.0)
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


def _tiny_model(
    path: Path,
    feature_names: list[str] | None = None,
    training_envelope: dict | None = None,
) -> Path:
    """Train and persist a tiny calibrated artifact.

    The hand-made matrix follows the frozen feature order; ``rhi`` is the
    separable column so the monotone LightGBM fit converges in milliseconds.
    ``training_envelope`` is stored verbatim in the artifact metadata, so
    tests control the range the profile command checks against.
    """
    generator = np.random.default_rng(7)
    features = generator.normal(size=(40, len(FEATURE_NAMES)))
    rhi_index = FEATURE_NAMES.index("rhi")
    features[:, rhi_index] = np.linspace(-1.0, 1.0, features.shape[0])
    labels = (features[:, rhi_index] > 0.0).astype(float)
    base = train_contrail_model(features, labels, list(FEATURE_NAMES))
    calibrated = fit_calibrator(base, features, labels)
    metadata: dict = {
        "feature_names": (
            list(FEATURE_NAMES) if feature_names is None else feature_names
        ),
        "source": "tests.test_cli_profile",
    }
    if training_envelope is not None:
        metadata["training_envelope"] = training_envelope
    save_model(calibrated, path, metadata)
    return path


@pytest.fixture(scope="module")
def profile_run(tmp_path_factory: pytest.TempPathFactory) -> ProfileRun:
    """Run the handler once on the synthetic route and cache the output."""
    directory = tmp_path_factory.mktemp("stratum_cli_profile")
    era5_path = directory / "synthetic_era5.nc"
    _synthetic_era5().to_netcdf(era5_path)
    model_path = _tiny_model(
        directory / "model.joblib", training_envelope=_TRAINING_ENVELOPE
    )
    json_path = directory / "profile.json"
    csv_path = directory / "profile.csv"

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = profile_cli.profile(
            [
                "--era5",
                str(era5_path),
                "--model",
                str(model_path),
                "--route",
                "VIDP-VABB",
                "--levels",
                "330,350",
                "--out",
                str(json_path),
                "--csv",
                str(csv_path),
            ]
        )
    assert code == 0, buffer.getvalue()
    return ProfileRun(
        era5=era5_path,
        model=model_path,
        code=code,
        payload=json.loads(json_path.read_text(encoding="utf-8")),
        csv=csv_path,
        stdout=buffer.getvalue(),
    )


def test_profile_route_writes_levels_and_recommendations(
    profile_run: ProfileRun,
) -> None:
    payload = profile_run.payload
    assert payload["metadata"]["trajectory_source"] == "great_circle_profile"
    assert payload["metadata"]["departure_assumed"] is True
    assert payload["levels"] == [330.0, 350.0]
    assert payload["metadata"]["levels"] == [330.0, 350.0]

    waypoints = payload["waypoints"]
    recommendations = payload["recommendations"]
    assert len(waypoints) == 25
    assert len(recommendations) == 25
    assert payload["metadata"]["training_envelope"] == _TRAINING_ENVELOPE
    for waypoint in waypoints:
        assert len(waypoint["levels"]) == 2
        for record in waypoint["levels"]:
            assert record["flight_level"] in (330.0, 350.0)
            assert record["risk_band"] in {"low", "moderate", "high"}
            assert 0.0 <= record["combined_risk"] <= 1.0
            # The fixture envelope covers both requested levels.
            assert record["training_envelope"] is True
    for recommendation in recommendations:
        assert recommendation["recommended_level"] in (330.0, 350.0)
        assert 0.0 <= recommendation["combined_risk"] <= 1.0
        assert recommendation["training_envelope"] is True
        assert recommendation["envelope_fallback"] is False
        assert recommendation["fuel_delta"]["delta_fuel_kg"] >= 0.0

    stdout = profile_run.stdout
    assert "route profile" in stdout
    assert "NOT a real flight track" in stdout
    assert "FL330" in stdout and "FL350" in stdout
    assert "JSON written" in stdout
    assert "CSV written" in stdout
    assert "outside the training envelope" not in stdout

    csv_lines = profile_run.csv.read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 1 + 25 * 2  # header + one row per (waypoint, level)
    assert "training_envelope" in csv_lines[0]


def _run_profile(
    profile_run: ProfileRun,
    tmp_path: Path,
    levels: str,
    model: Path | None = None,
) -> tuple[int, dict | None, str]:
    """Run the handler on the fixture ERA5 field and return its output."""
    json_path = tmp_path / f"profile_{levels.replace(',', '_')}.json"
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = profile_cli.profile(
            [
                "--era5",
                str(profile_run.era5),
                "--model",
                str(profile_run.model if model is None else model),
                "--route",
                "VIDP-VABB",
                "--levels",
                levels,
                "--out",
                str(json_path),
            ]
        )
    payload = (
        json.loads(json_path.read_text(encoding="utf-8"))
        if json_path.is_file()
        else None
    )
    return code, payload, buffer.getvalue()


def test_profile_flags_levels_outside_training_envelope(
    tmp_path: Path, profile_run: ProfileRun
) -> None:
    code, payload, stdout = _run_profile(profile_run, tmp_path, "290,330,390")
    assert code == 0, stdout
    assert payload is not None
    for waypoint in payload["waypoints"]:
        records = {
            record["flight_level"]: record for record in waypoint["levels"]
        }
        assert records[290.0]["training_envelope"] is False
        assert records[390.0]["training_envelope"] is False
        assert records[330.0]["training_envelope"] is True
    for recommendation in payload["recommendations"]:
        # Only FL330 both lies inside the envelope and carries ERA5 data.
        assert recommendation["recommended_level"] == 330.0
        assert recommendation["training_envelope"] is True
        assert recommendation["envelope_fallback"] is False
    assert (
        "FL290, FL390 lie outside the training envelope FL330..FL350" in stdout
    )


def test_profile_falls_back_outside_envelope_and_warns(
    tmp_path: Path, profile_run: ProfileRun
) -> None:
    model = _tiny_model(
        tmp_path / "out_of_range_envelope.joblib",
        training_envelope=_OUT_OF_RANGE_ENVELOPE,
    )
    code, payload, stdout = _run_profile(
        profile_run, tmp_path, "330,350", model=model
    )
    assert code == 0, stdout
    assert payload is not None
    for waypoint in payload["waypoints"]:
        for record in waypoint["levels"]:
            assert record["training_envelope"] is False
    for recommendation in payload["recommendations"]:
        assert recommendation["recommended_level"] in (330.0, 350.0)
        assert recommendation["training_envelope"] is False
        assert recommendation["envelope_fallback"] is True
    assert (
        "FL330, FL350 lie outside the training envelope FL370..FL390" in stdout
    )
    assert (
        "no requested level here lies inside the training envelope" in stdout
    )


def test_profile_without_envelope_marks_records_unknown(
    tmp_path: Path, profile_run: ProfileRun
) -> None:
    model = _tiny_model(tmp_path / "no_envelope.joblib")
    code, payload, stdout = _run_profile(
        profile_run, tmp_path, "330,350", model=model
    )
    assert code == 0, stdout
    assert payload is not None
    for waypoint in payload["waypoints"]:
        for record in waypoint["levels"]:
            assert record["training_envelope"] is None
    for recommendation in payload["recommendations"]:
        assert recommendation["training_envelope"] is None
        assert recommendation["envelope_fallback"] is False
    assert "records no training envelope" in stdout


def test_envelope_bounds_rejects_malformed_metadata() -> None:
    assert profile_cli._envelope_bounds(None) is None
    assert profile_cli._envelope_bounds({}) is None
    assert profile_cli._envelope_bounds({"pressure_pa_min": 1.0}) is None
    assert (
        profile_cli._envelope_bounds(
            {"pressure_pa_min": 2.0, "pressure_pa_max": 1.0}
        )
        is None
    )
    assert (
        profile_cli._envelope_bounds(
            {"pressure_pa_min": float("nan"), "pressure_pa_max": 1.0}
        )
        is None
    )
    assert profile_cli._envelope_bounds(
        {"pressure_pa_min": 1.0, "pressure_pa_max": 2.0}
    ) == (1.0, 2.0)


def test_profile_zero_risk_when_sac_does_not_form(profile_run: ProfileRun) -> None:
    records = [
        record
        for waypoint in profile_run.payload["waypoints"]
        for record in waypoint["levels"]
    ]
    non_forming = [record for record in records if not record["sac_forms"]]
    assert non_forming, "synthetic field must contain subsaturated samples"
    for record in non_forming:
        assert record["combined_risk"] == 0.0
        assert record["risk_band"] == "low"


def test_profile_registered_in_main(profile_run: ProfileRun) -> None:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = main(
            [
                "profile",
                "--era5",
                str(profile_run.era5),
                "--model",
                str(profile_run.model),
                "--route",
                "VIDP-VABB",
                "--levels",
                "330,350",
            ]
        )
    assert code == 0, buffer.getvalue()
    assert "Recommendation summary" in buffer.getvalue()


def test_profile_missing_era5_exits_2_and_writes_nothing(
    tmp_path: Path, profile_run: ProfileRun
) -> None:
    out = tmp_path / "missing.json"
    csv_out = tmp_path / "missing.csv"
    code = profile_cli.profile(
        [
            "--era5",
            str(tmp_path / "absent.nc"),
            "--model",
            str(profile_run.model),
            "--route",
            "VIDP-VABB",
            "--levels",
            "330,350",
            "--out",
            str(out),
            "--csv",
            str(csv_out),
        ]
    )
    assert code == 2
    assert not out.exists()
    assert not csv_out.exists()


def test_profile_levels_outside_cruise_band_exit_2(
    tmp_path: Path, profile_run: ProfileRun
) -> None:
    assert (
        profile_cli.profile(
            [
                "--era5",
                str(profile_run.era5),
                "--model",
                str(profile_run.model),
                "--route",
                "VIDP-VABB",
                "--levels",
                "250,350",
            ]
        )
        == 2
    )
    with pytest.raises(ValueError, match="outside the allowed cruise band"):
        profile_cli._validate_levels([250.0])


def test_profile_all_levels_out_of_domain_exit_2(
    tmp_path: Path, profile_run: ProfileRun
) -> None:
    out = tmp_path / "out_of_domain.json"
    code = profile_cli.profile(
        [
            "--era5",
            str(profile_run.era5),
            "--model",
            str(profile_run.model),
            "--route",
            "VIDP-VABB",
            "--levels",
            "390",  # ~197 hPa, above the ERA5 pressure floor of 200 hPa
            "--out",
            str(out),
        ]
    )
    assert code == 2
    assert not out.exists()


def test_profile_feature_name_mismatch_exits_1(
    tmp_path: Path, profile_run: ProfileRun
) -> None:
    artifact = _tiny_model(tmp_path / "mismatched.joblib", ["rhi", "tlm_k"])
    out = tmp_path / "mismatch.json"
    code = profile_cli.profile(
        [
            "--era5",
            str(profile_run.era5),
            "--model",
            str(artifact),
            "--route",
            "VIDP-VABB",
            "--levels",
            "330,350",
            "--out",
            str(out),
        ]
    )
    assert code == 1
    assert not out.exists()
