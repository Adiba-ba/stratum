"""Tests for the end-to-end pipeline (Step 8).

Covers ISA pressure-altitude inversion, specific humidity to RHi
conversion, feature assembly along a synthetic Delhi-Mumbai trajectory
against an in-memory mock ERA5 dataset, and the corridor runner
(flagship decision cards, aggregation, and the no-risk path).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stratum.model.calibrate import CalibratedModel, fit_calibrator
from stratum.model.features import FEATURE_NAMES
from stratum.model.train import train_contrail_model
from stratum.physics import e_i, fl_to_p
from stratum.physics import thermo
from stratum.physics.diagnostics import (
    compute_richardson_number,
    compute_solar_zenith_angle,
)
from stratum.pipeline import (
    HUMIDITY_OFFSET_HPA,
    evaluate_corridor_flight,
    extract_features_for_point,
    extract_features_for_trajectory,
    pressure_to_altitude_m,
    specific_humidity_to_rhi,
)
from stratum.routes.corridor import AIRPORTS, great_circle_waypoints

# LightGBM's sklearn wrapper validates feature names even when predictions
# receive plain ndarrays fitted from ndarrays; silence its spurious warning.
pytestmark = pytest.mark.filterwarnings(
    "ignore:X does not have valid feature names, "
    "but LGBMClassifier was fitted with feature names"
)

MOCK_LATS = np.linspace(15.0, 32.0, 6)
MOCK_LONS = np.linspace(68.0, 82.0, 6)
MOCK_LEVELS = np.array([200.0, 225.0, 250.0, 300.0, 350.0, 400.0])
MOCK_TIMES = np.array(
    [
        "2025-01-01T00:00",
        "2025-01-01T01:00",
        "2025-01-01T02:00",
        "2025-01-01T03:00",
    ],
    dtype="datetime64[ns]",
)

_CRUISE_FL = 350.0
_NUM_POINTS = 25
_STEP_MINUTES = 6

#: Ground-truth logit used for the synthetic training set and the
#: scenarios (monotone increasing in RHi, decreasing in TLM margin).
_RHI_CENTRE = 0.85
_MARGIN_CENTRE = -4.0


def _mock_met(
    target_rhi: float,
    temperature_offset: float = 0.0,
    levels: np.ndarray | None = None,
) -> xr.Dataset:
    """Build an in-memory ERA5-like dataset with controlled mid-level RHi.

    Temperature is linear in pressure level (so interpolation is exact);
    specific humidity is prescribed from the ice saturation curve so that
    the vapor pressure corresponds to ``target_rhi`` at each grid level.
    ``levels`` overrides the default pressure axis, e.g. to place the
    topmost stored level just below a queried point.
    """
    pressure_levels = (
        MOCK_LEVELS if levels is None else np.asarray(levels, dtype=float)
    )
    shape = (
        pressure_levels.size,
        MOCK_LATS.size,
        MOCK_LONS.size,
        MOCK_TIMES.size,
    )
    level = pressure_levels[:, None, None, None]
    lat = MOCK_LATS[None, :, None, None]
    lon = MOCK_LONS[None, None, :, None]
    time_index = np.arange(MOCK_TIMES.size, dtype=float)[None, None, None, :] * 0.0

    temperature = (
        216.65 + 0.11 * (level - 200.0) + temperature_offset + 0.0 * time_index
    )
    saturation = np.array(
        [
            e_i(216.65 + 0.11 * (value - 200.0) + temperature_offset)
            for value in pressure_levels
        ]
    )[:, None, None, None]
    specific_humidity = target_rhi * saturation * 0.622 / (level * 100.0)

    u_wind = 8.0 + 0.05 * (lat - 15.0) + 0.004 * (level - 200.0)
    v_wind = -3.0 + 0.1 * (lon - 68.0) + 0.001 * (level - 200.0)

    dataset = xr.Dataset(
        data_vars={
            "temperature": (
                ("level", "latitude", "longitude", "time"),
                np.broadcast_to(temperature, shape),
            ),
            "specific_humidity": (
                ("level", "latitude", "longitude", "time"),
                np.broadcast_to(specific_humidity, shape),
            ),
            "u_component_of_wind": (
                ("level", "latitude", "longitude", "time"),
                np.broadcast_to(u_wind, shape),
            ),
            "v_component_of_wind": (
                ("level", "latitude", "longitude", "time"),
                np.broadcast_to(v_wind, shape),
            ),
            "vertical_velocity": (
                ("level", "latitude", "longitude", "time"),
                np.full(shape, 0.002),
            ),
            "fraction_of_cloud_cover": (
                ("level", "latitude", "longitude", "time"),
                np.full(shape, 0.20),
            ),
        },
        coords={
            "level": pressure_levels,
            "latitude": MOCK_LATS,
            "longitude": MOCK_LONS,
            "time": MOCK_TIMES,
        },
    )
    return dataset.transpose("time", "level", "latitude", "longitude")


def _trajectory(
    num_points: int = _NUM_POINTS,
    fl: float = _CRUISE_FL,
    start: str = "2025-01-01T00:00:00Z",
) -> pd.DataFrame:
    """Build a cleaned ADS-B shaped VIDP -> VABB cruise trajectory."""
    waypoints = great_circle_waypoints(
        AIRPORTS["VIDP"], AIRPORTS["VABB"], num_points=num_points, fl=fl
    )
    start_time = pd.Timestamp(start)
    records = []
    for index, waypoint in enumerate(waypoints):
        records.append(
            {
                "icao24": "800abc",
                "callsign": "AIC866",
                "time": start_time + pd.Timedelta(minutes=_STEP_MINUTES * index),
                "lat": waypoint["lat"],
                "lon": waypoint["lon"],
                "baro_alt_ft": fl * 100.0,
                "gs_kt": 455.0,
                "on_ground": False,
                "source": "opensky",
            }
        )
    return pd.DataFrame(records)


def _trained_model(seed: int = 11) -> CalibratedModel:
    """Train a small calibrated baseline consistent with the scenario model.

    The latent label depends on the temperature margin ``temp_k - tlm_k``;
    the model sees ``temp_k`` and ``tlm_k`` separately.
    """
    rng = np.random.default_rng(seed)
    n_train, n_val = 500, 300
    n_total = n_train + n_val
    features = np.zeros((n_total, len(FEATURE_NAMES)), dtype=float)
    rhi = rng.uniform(0.2, 1.6, n_total)
    margin = rng.uniform(-12.0, 4.0, n_total)
    features[:, FEATURE_NAMES.index("rhi")] = rhi
    features[:, FEATURE_NAMES.index("tlm_k")] = 231.0
    features[:, FEATURE_NAMES.index("temp_k")] = 231.0 + margin
    logit = 3.0 * (rhi - _RHI_CENTRE) - 0.15 * (margin - _MARGIN_CENTRE)
    labels = (rng.uniform(size=n_total) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    model = train_contrail_model(
        features[:n_train], labels[:n_train], list(FEATURE_NAMES)
    )
    return fit_calibrator(model, features[n_train:], labels[n_train:])


def _contiguous_flag_runs(segments: list[dict]) -> int:
    """Count contiguous runs of segments carrying a decision record."""
    runs = 0
    previous_flagged = False
    for segment in segments:
        flagged = segment["decision_status"] is not None
        if flagged and not previous_flagged:
            runs += 1
        previous_flagged = flagged
    return runs


class _FixedProbabilityModel:
    """CalibratedModel stand-in returning a constant persistence probability."""

    def __init__(self, probability: float) -> None:
        self._probability = float(probability)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probability = np.full(len(features), self._probability, dtype=float)
        return np.column_stack([1.0 - probability, probability])


def test_pressure_to_altitude_inverts_isa() -> None:
    assert pressure_to_altitude_m(thermo.P0_PA) == pytest.approx(0.0, abs=1e-6)
    # FL350 in the ISA is 10 668 m (35 000 ft).
    altitude = pressure_to_altitude_m(fl_to_p(_CRUISE_FL) * 100.0)
    assert altitude == pytest.approx(10668.0, abs=5.0)
    # Monotone: lower pressure -> higher altitude.
    assert pressure_to_altitude_m(20000.0) > pressure_to_altitude_m(30000.0)
    with pytest.raises(ValueError):
        pressure_to_altitude_m(-1.0)


def test_specific_humidity_to_rhi_roundtrip() -> None:
    temperature_k = 221.0
    pressure_pa = 23842.0
    for target in (0.3, 1.0, 1.4):
        humidity = target * e_i(temperature_k) * 0.622 / pressure_pa
        assert specific_humidity_to_rhi(
            humidity, pressure_pa, temperature_k
        ) == pytest.approx(target, rel=1e-3)
    assert math.isnan(specific_humidity_to_rhi(float("nan"), pressure_pa, temperature_k))


def test_extract_features_shapes_and_finiteness() -> None:
    features, metadata = extract_features_for_trajectory(
        _trajectory(), _mock_met(target_rhi=1.3)
    )
    assert features.shape == (_NUM_POINTS, len(FEATURE_NAMES))
    assert len(metadata) == _NUM_POINTS
    assert np.all(np.isfinite(features))

    pressure_pa = fl_to_p(_CRUISE_FL) * 100.0
    pressure_column = features[:, FEATURE_NAMES.index("pressure_pa")]
    assert np.allclose(pressure_column, pressure_pa, rtol=1e-9)

    rhi_column = features[:, FEATURE_NAMES.index("rhi")]
    assert np.allclose(rhi_column, 1.3, rtol=0.1)
    gradient = features[:, FEATURE_NAMES.index("rhi_grad")]
    assert np.allclose(
        gradient,
        features[:, FEATURE_NAMES.index("rhi_above")]
        - features[:, FEATURE_NAMES.index("rhi_below")],
        atol=1e-12,
    )

    eta_column = features[:, FEATURE_NAMES.index("engine_efficiency_eta")]
    assert np.allclose(eta_column, 0.32)
    soot_column = features[:, FEATURE_NAMES.index("log_soot_ei_n")]
    assert np.allclose(soot_column, math.log10(1.5e15))
    richardson_column = features[:, FEATURE_NAMES.index("richardson_number")]
    assert np.all(np.isfinite(richardson_column))
    assert np.all(richardson_column > 0.0)
    zenith_column = features[:, FEATURE_NAMES.index("solar_zenith_deg")]
    assert np.all((zenith_column >= 0.0) & (zenith_column <= 180.0))

    for info in metadata:
        assert info["rhi"] == pytest.approx(1.3, rel=0.1)
        assert info["flight_level"] == pytest.approx(_CRUISE_FL)
        assert info["pressure_pa"] == pytest.approx(pressure_pa, rel=1e-9)
        assert info["rhi_above"] - info["rhi_below"] != 0.0
        assert info["richardson_number"] >= 0.0
        assert 0.0 <= info["solar_zenith_deg"] <= 180.0
        assert info["engine_efficiency_eta"] == pytest.approx(0.32)
        assert info["log_soot_ei_n"] == pytest.approx(math.log10(1.5e15))
    assert any(info["sac_forms"] for info in metadata)
    assert all(info["ambient_ice_supersaturated"] for info in metadata)


def test_extract_features_wind_shear_matches_analytic_gradient() -> None:
    features, metadata = extract_features_for_trajectory(
        _trajectory(), _mock_met(target_rhi=0.6)
    )
    pressure_pa = fl_to_p(_CRUISE_FL) * 100.0
    offset_pa = HUMIDITY_OFFSET_HPA * 100.0
    delta_z = abs(
        pressure_to_altitude_m(pressure_pa + offset_pa)
        - pressure_to_altitude_m(pressure_pa - offset_pa)
    )
    delta_u = 0.004 * (-2.0 * HUMIDITY_OFFSET_HPA)
    delta_v = 0.001 * (-2.0 * HUMIDITY_OFFSET_HPA)
    expected_shear = math.hypot(delta_u, delta_v) / delta_z

    shear_column = features[:, FEATURE_NAMES.index("vertical_wind_shear")]
    assert np.allclose(shear_column, expected_shear, rtol=1e-6)
    assert shear_column[0] > 0.0

    # Temperature confirms the linear-level interpolation used above.
    info = metadata[0]
    level_hpa = pressure_pa / 100.0
    assert info["temperature_k"] == pytest.approx(
        216.65 + 0.11 * (level_hpa - 200.0), rel=1e-6
    )
    assert info["tlm_c"] == pytest.approx(-42.5, abs=5.0)

    # Richardson number matches the analytic bounding-layer value.
    level_above_hpa = level_hpa - HUMIDITY_OFFSET_HPA
    level_below_hpa = level_hpa + HUMIDITY_OFFSET_HPA
    lat = float(metadata[0]["lat"])
    lon = float(metadata[0]["lon"])
    u_above = 8.0 + 0.05 * (lat - 15.0) + 0.004 * (level_above_hpa - 200.0)
    u_below = 8.0 + 0.05 * (lat - 15.0) + 0.004 * (level_below_hpa - 200.0)
    v_above = -3.0 + 0.1 * (lon - 68.0) + 0.001 * (level_above_hpa - 200.0)
    v_below = -3.0 + 0.1 * (lon - 68.0) + 0.001 * (level_below_hpa - 200.0)
    expected_richardson = compute_richardson_number(
        temp_above_k=216.65 + 0.11 * (level_above_hpa - 200.0),
        temp_below_k=216.65 + 0.11 * (level_below_hpa - 200.0),
        u_above=u_above,
        u_below=u_below,
        v_above=v_above,
        v_below=v_below,
        alt_above_m=pressure_to_altitude_m(pressure_pa - offset_pa),
        alt_below_m=pressure_to_altitude_m(pressure_pa + offset_pa),
        p_ambient_pa=pressure_pa,
    )
    richardson_column = features[:, FEATURE_NAMES.index("richardson_number")]
    assert np.allclose(richardson_column, expected_richardson, rtol=1e-9)

    # Solar zenith angle matches the astronomical computation.
    expected_zenith = compute_solar_zenith_angle(
        pd.Timestamp(metadata[0]["time_utc"]).to_pydatetime(), lat, lon
    )
    zenith_column = features[:, FEATURE_NAMES.index("solar_zenith_deg")]
    assert zenith_column[0] == pytest.approx(expected_zenith, rel=1e-12)
    assert 0.0 <= expected_zenith <= 180.0


def test_extract_features_validates_input() -> None:
    dataset = _mock_met(target_rhi=1.0)
    with pytest.raises(ValueError):
        extract_features_for_trajectory(_trajectory().drop(columns=["lon"]), dataset)
    with pytest.raises(ValueError):
        extract_features_for_trajectory(_trajectory().iloc[0:0], dataset)
    broken = _trajectory()
    broken.loc[0, "lat"] = float("nan")
    with pytest.raises(ValueError):
        extract_features_for_trajectory(broken, dataset)


def test_runner_flags_moist_corridor() -> None:
    result = evaluate_corridor_flight(
        _trajectory(), _mock_met(target_rhi=1.3), _trained_model()
    )

    assert result["total_segments"] == _NUM_POINTS
    assert result["contrail_forming_segments"] >= 1
    assert result["contrail_persistent_segments"] >= 1
    assert result["flagged_segments"] >= 1
    assert len(result["recommended_reroutes"]) == _contiguous_flag_runs(
        result["segments"]
    )
    assert len(result["recommended_reroutes"]) >= 1

    for segment in result["segments"]:
        assert 0.0 <= segment["p_persist"] <= 1.0
        if segment["decision_status"] is not None:
            assert segment["decision_status"] in {
                "abstain",
                "eligible",
                "no_robust_candidate",
            }

    reroute = result["recommended_reroutes"][0]
    for key in (
        "event_index",
        "segment_index",
        "time_utc",
        "segments_in_event",
        "p_persist",
        "baseline_fl",
        "recommended_fl",
        "direction",
        "delta_fuel_kg",
        "delta_co2_kg",
        "avoided_contrail_ef_j",
        "benefit_to_cost_ratio",
    ):
        assert key in reroute
    assert reroute["p_persist"] >= 0.40
    assert reroute["baseline_fl"] == pytest.approx(_CRUISE_FL)
    assert reroute["recommended_fl"] != pytest.approx(reroute["baseline_fl"])
    assert reroute["delta_fuel_kg"] > 0.0
    assert reroute["delta_co2_kg"] > 0.0
    assert reroute["avoided_contrail_ef_j"] > 0.0
    assert reroute["benefit_to_cost_ratio"] > 0.0
    assert 1 <= reroute["segments_in_event"] <= _NUM_POINTS

    total = sum(
        entry["avoided_contrail_ef_j"] for entry in result["recommended_reroutes"]
    )
    assert result["total_avoidable_warming_j"] == pytest.approx(total, rel=1e-12)
    assert result["total_avoidable_warming_j"] > 0.0
    assert result["total_extra_fuel_kg"] == pytest.approx(
        sum(entry["delta_fuel_kg"] for entry in result["recommended_reroutes"]),
        rel=1e-12,
    )


def test_runner_dry_corridor_produces_no_advisories() -> None:
    result = evaluate_corridor_flight(
        _trajectory(),
        _mock_met(target_rhi=0.2, temperature_offset=10.0),
        _trained_model(),
    )

    assert result["total_segments"] == _NUM_POINTS
    assert result["contrail_forming_segments"] == 0
    assert result["contrail_persistent_segments"] == 0
    assert result["flagged_segments"] == 0
    assert result["recommended_reroutes"] == []
    assert result["total_avoidable_warming_j"] == 0.0
    assert all(segment["decision_status"] is None for segment in result["segments"])


def _trajectory_with_out_of_domain_dip(
    outside: list[int],
) -> pd.DataFrame:
    """Return the cruise trajectory with a southern dip outside ERA5_BBOX."""
    trajectory = _trajectory()
    for index in outside:
        # 2 deg N is inside the ADS-B bounds (6-37 N) but south of the
        # ERA5 domain (5-40 N).
        trajectory.loc[index, "lat"] = 2.0
    return trajectory


def test_extract_features_marks_out_of_domain_rows() -> None:
    outside = [4, 5, 6]
    features, metadata = extract_features_for_trajectory(
        _trajectory_with_out_of_domain_dip(outside), _mock_met(target_rhi=1.3)
    )

    assert features.shape == (_NUM_POINTS, len(FEATURE_NAMES))
    assert len(metadata) == _NUM_POINTS
    for index in outside:
        assert metadata[index]["data_available"] is False
        assert metadata[index]["unavailable_reason"] == "out_of_domain"
        # The row is kept in place with nan values, schema unchanged.
        assert np.all(np.isnan(features[index]))
    for index in range(_NUM_POINTS):
        if index not in outside:
            assert metadata[index]["data_available"] is True
            assert metadata[index]["unavailable_reason"] is None


def test_extract_features_marks_missing_met_rows() -> None:
    dataset = _mock_met(target_rhi=1.3)
    dataset["vertical_velocity"] = xr.full_like(
        dataset["vertical_velocity"], float("nan")
    )
    features, metadata = extract_features_for_trajectory(_trajectory(), dataset)

    assert all(info["data_available"] is False for info in metadata)
    assert all(info["unavailable_reason"] == "missing_met" for info in metadata)
    assert np.all(np.isnan(features))


def test_extract_features_fl370_keeps_row_with_nan_upper_probe() -> None:
    """A point inside the file keeps its row when the upper probe leaves it.

    The bundled real ERA5 file tops out at 200 hPa, so at FL370 (~217 hPa)
    the ``-25 hPa`` probe lands at ~192 hPa and is undefined. Only the
    point sample gates ``data_available``; the probe-derived features must
    be NaN instead of an invented value, while the point-derived features
    (RHi, SAC) stay usable.
    """
    features, metadata = extract_features_for_trajectory(
        _trajectory(fl=370.0), _mock_met(target_rhi=1.3)
    )
    assert features.shape == (_NUM_POINTS, len(FEATURE_NAMES))
    assert all(info["data_available"] is True for info in metadata)
    assert all(info["unavailable_reason"] is None for info in metadata)

    for name in (
        "rhi_above",
        "rhi_grad",
        "vertical_wind_shear",
        "richardson_number",
    ):
        column = features[:, FEATURE_NAMES.index(name)]
        assert np.all(np.isnan(column)), name
    # The lower probe is still inside the file, so only the upper one is
    # undefined at FL370; rhi (the point sample) stays finite.
    assert np.all(np.isfinite(features[:, FEATURE_NAMES.index("rhi_below")]))
    assert np.all(np.isfinite(features[:, FEATURE_NAMES.index("rhi")]))
    # The SAC was evaluated on the point sample, not skipped.
    assert np.all(np.isfinite(features[:, FEATURE_NAMES.index("tlc_k")]))
    assert all(isinstance(info["sac_forms"], bool) for info in metadata)


def test_extract_features_keeps_row_when_probes_leave_range() -> None:
    """Both bounding-layer samples can leave the file; the row still stands."""
    dataset = _mock_met(target_rhi=1.3, levels=np.array([250.0, 280.0]))
    pressure_pa = 27000.0  # 270 hPa; the probes at 245/295 hPa are outside
    vector, info = extract_features_for_point(
        28.5, 77.0, pressure_pa, "2025-01-01T01:00", ds_met=dataset
    )

    assert info["data_available"] is True
    assert info["unavailable_reason"] is None
    assert math.isfinite(info["rhi"])
    assert math.isfinite(vector[FEATURE_NAMES.index("rhi")])
    for name in (
        "rhi_above",
        "rhi_below",
        "rhi_grad",
        "vertical_wind_shear",
        "richardson_number",
    ):
        assert math.isnan(vector[FEATURE_NAMES.index(name)]), name
    # The SAC still runs on the point sample.
    assert math.isfinite(info["tlc_k"])
    assert isinstance(info["sac_forms"], bool)


def test_runner_skips_out_of_domain_segments() -> None:
    outside = list(range(8, 13))
    # A constant high probability would flag every segment if the guard
    # were absent, so the counts below pin the skip behaviour.
    result = evaluate_corridor_flight(
        _trajectory_with_out_of_domain_dip(outside),
        _mock_met(target_rhi=1.3),
        _FixedProbabilityModel(0.9),
    )

    assert result["total_segments"] == _NUM_POINTS
    assert result["out_of_domain_segments"] == len(outside)
    assert result["contrail_persistent_segments"] == _NUM_POINTS - len(outside)
    assert result["flagged_segments"] == _NUM_POINTS - len(outside)

    for segment in result["segments"]:
        if segment["segment_index"] in outside:
            assert segment["status"] == "out_of_domain"
            assert segment["decision_status"] is None
            assert segment["recommended_fl"] is None
            assert segment["delta_fuel_kg"] is None
            assert segment["delta_co2_kg"] is None
            assert segment["avoided_contrail_ef_j"] is None
        else:
            assert segment["status"] == "ok"
            assert segment["decision_status"] is not None

    # No advisory covers an out-of-domain segment, and the data gap
    # splits the flagged run into two events of at most 20 segments.
    for reroute in result["recommended_reroutes"]:
        assert reroute["segment_index"] not in outside
        assert reroute["segments_in_event"] <= _NUM_POINTS - len(outside)

    total = sum(
        entry["avoided_contrail_ef_j"] for entry in result["recommended_reroutes"]
    )
    assert result["total_avoidable_warming_j"] == pytest.approx(total, rel=1e-12)


def test_runner_reports_missing_met_segments_without_advisories() -> None:
    dataset = _mock_met(target_rhi=1.3)
    dataset["fraction_of_cloud_cover"] = xr.full_like(
        dataset["fraction_of_cloud_cover"], float("nan")
    )
    result = evaluate_corridor_flight(
        _trajectory(), dataset, _FixedProbabilityModel(0.9)
    )

    assert result["flagged_segments"] == 0
    assert result["contrail_persistent_segments"] == 0
    assert result["recommended_reroutes"] == []
    assert all(segment["status"] == "missing_met" for segment in result["segments"])
    assert all(segment["decision_status"] is None for segment in result["segments"])
