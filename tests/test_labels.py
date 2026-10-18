"""Tests for the real-data label builders (Step 5).

Covers the physics label path (SAC formation plus ice supersaturation
against an in-memory ERA5-like dataset), the CSV dataset round-trip, the
route-profile sampler, the viewing-zenith geometry, the NetCDF scene
loader, and the observed label path (detector output matched against
advected, parallax-corrected trajectory segments on a synthetic scene).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stratum.detect.filters import detect_contrail_lines
from stratum.geo.angles import ArrayGeoreferencer
from stratum.io.era5 import interpolate_met_to_segment
from stratum.labels.observed import (
    build_observed_labels,
    load_bt_pair,
    viewing_zenith_angle_deg,
)
from stratum.labels.physics import (
    CENSORED_DATA_UNAVAILABLE,
    CENSORED_SAC_NOT_FORMED,
    LABEL_FORMING_PERSISTENT,
    LABEL_FORMING_SUBSATURATED,
    RHI_PERSISTENCE_THRESHOLD,
    PhysicsSample,
    build_physics_labels,
    read_dataset,
    sample_route_profiles,
    write_dataset,
)
from stratum.match.advect import advect_segment
from stratum.match.parallax import calculate_parallax_displacement
from stratum.model.features import FEATURE_NAMES
from stratum.physics import e_i, fl_to_p
from stratum.pipeline.build_features import pressure_to_altitude_m

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
_PRESSURE_PA = fl_to_p(_CRUISE_FL) * 100.0
_SAMPLE_TIME = "2025-01-01T01:00"

#: Latitude above which :func:`_mock_met` reports a high-cloud fraction of
#: 0.90 instead of 0.20, used to make the observed-label scene unobservable.
_HIGH_CLOUD_BOUNDARY_LAT = 29.5


def _mock_met(
    target_rhi: float,
    temperature_offset: float = 0.0,
    high_cloud_boundary_lat: float | None = None,
) -> xr.Dataset:
    """Build an in-memory ERA5-like dataset with a controlled RHi.

    Temperature is linear in pressure level so interpolation is exact;
    specific humidity is prescribed from the ice saturation curve so the
    vapor pressure corresponds to ``target_rhi`` at every grid level.
    """
    shape = (MOCK_LEVELS.size, MOCK_LATS.size, MOCK_LONS.size, MOCK_TIMES.size)
    level = MOCK_LEVELS[:, None, None, None]
    lat = MOCK_LATS[None, :, None, None]
    lon = MOCK_LONS[None, None, :, None]

    temperature = 216.65 + 0.11 * (level - 200.0) + temperature_offset
    saturation = np.array(
        [
            e_i(216.65 + 0.11 * (value - 200.0) + temperature_offset)
            for value in MOCK_LEVELS
        ]
    )[:, None, None, None]
    specific_humidity = target_rhi * saturation * 0.622 / (level * 100.0)

    u_wind = 8.0 + 0.05 * (lat - 15.0) + 0.004 * (level - 200.0)
    v_wind = -3.0 + 0.1 * (lon - 68.0) + 0.001 * (level - 200.0)

    if high_cloud_boundary_lat is None:
        cloud_cover = np.full(shape, 0.20)
    else:
        column = np.where(
            MOCK_LATS < high_cloud_boundary_lat, 0.20, 0.90
        )[None, :, None, None]
        cloud_cover = np.broadcast_to(column, shape).copy()

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
                cloud_cover,
            ),
        },
        coords={
            "level": MOCK_LEVELS,
            "latitude": MOCK_LATS,
            "longitude": MOCK_LONS,
            "time": MOCK_TIMES,
        },
    )
    return dataset.transpose("time", "level", "latitude", "longitude")


# ---------------------------------------------------------------------------
# Physics labels
# ---------------------------------------------------------------------------


def test_build_physics_labels_persistent_and_subsaturated() -> None:
    sample = (28.5, 77.0, _PRESSURE_PA, _SAMPLE_TIME)

    features, labels, meta = build_physics_labels(
        _mock_met(1.3), [PhysicsSample(*sample)]
    )
    assert features.shape == (1, len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))
    assert labels[0] == LABEL_FORMING_PERSISTENT
    assert meta[0]["sac_forms"] is True
    assert meta[0]["rhi"] > 1.0
    assert meta[0]["censored_reason"] is None
    assert meta[0]["label"] == LABEL_FORMING_PERSISTENT

    _, labels_dry, meta_dry = build_physics_labels(_mock_met(0.9), [sample])
    assert labels_dry[0] == LABEL_FORMING_SUBSATURATED
    assert meta_dry[0]["sac_forms"] is True
    assert meta_dry[0]["rhi"] <= 1.0
    assert meta_dry[0]["censored_reason"] is None


def test_build_physics_labels_censors_non_forming_and_missing_data() -> None:
    sample = (28.5, 77.0, _PRESSURE_PA, _SAMPLE_TIME)

    # +10 K warms the ambient air above the critical temperature: the SAC
    # cannot form, so persistence is undefined and the row is censored.
    features, labels, meta = build_physics_labels(
        _mock_met(0.9, temperature_offset=10.0), [sample]
    )
    assert math.isnan(labels[0])
    assert meta[0]["sac_forms"] is False
    assert meta[0]["censored_reason"] == CENSORED_SAC_NOT_FORMED
    assert np.all(np.isfinite(features))

    # 10 N lies south of the ERA5 domain (15-32 N): the point sample has
    # no reanalysis state, and only such a genuinely missing point sample
    # censors a row. An out-of-range +/-25 hPa probe does not (see the
    # probe test below).
    out_of_domain = (10.0, 77.0, _PRESSURE_PA, _SAMPLE_TIME)
    features_missing, labels_missing, meta_missing = build_physics_labels(
        _mock_met(1.3), [out_of_domain]
    )
    assert math.isnan(labels_missing[0])
    assert meta_missing[0]["data_available"] is False
    assert meta_missing[0]["unavailable_reason"] == "out_of_domain"
    assert meta_missing[0]["censored_reason"] == CENSORED_DATA_UNAVAILABLE
    assert np.all(np.isnan(features_missing))


def test_build_physics_labels_keeps_rows_whose_probes_leave_range() -> None:
    """Only a missing point sample censors; a missing probe sample does not.

    The level axis is truncated to 225-400 hPa and the sample sits at
    FL360 (~227 hPa): the point interpolation is inside the file, but the
    upper ``-25 hPa`` probe (~202 hPa) leaves it. The old all-or-nothing
    availability rule censored such rows as ``data_unavailable``; now the
    row is labeled (here 1 persistent) and only the probe-derived
    features are NaN.
    """
    dataset = _mock_met(1.3).sel(level=[225.0, 250.0, 300.0, 350.0, 400.0])
    sample = (28.5, 77.0, fl_to_p(360.0) * 100.0, _SAMPLE_TIME)
    features, labels, meta = build_physics_labels(dataset, [sample])

    assert meta[0]["data_available"] is True
    assert meta[0]["unavailable_reason"] is None
    assert meta[0]["censored_reason"] is None
    assert labels[0] == LABEL_FORMING_PERSISTENT
    assert meta[0]["rhi"] > RHI_PERSISTENCE_THRESHOLD
    assert np.isfinite(features[0, FEATURE_NAMES.index("rhi")])
    # The lower probe (~252 hPa) is still inside the file.
    assert np.isfinite(features[0, FEATURE_NAMES.index("rhi_below")])
    for name in (
        "rhi_above",
        "rhi_grad",
        "vertical_wind_shear",
        "richardson_number",
    ):
        assert math.isnan(features[0, FEATURE_NAMES.index(name)]), name


def test_build_physics_labels_accepts_sample_forms() -> None:
    dataset = _mock_met(1.3)
    as_tuple = (28.5, 77.0, _PRESSURE_PA, _SAMPLE_TIME)
    as_mapping = {
        "lat": 28.5,
        "lon": 77.0,
        "pressure_pa": _PRESSURE_PA,
        "time": _SAMPLE_TIME,
    }
    as_named = PhysicsSample(28.5, 77.0, _PRESSURE_PA, _SAMPLE_TIME)

    _, labels_tuple, _ = build_physics_labels(dataset, [as_tuple])
    _, labels_mapping, _ = build_physics_labels(dataset, [as_mapping])
    _, labels_named, _ = build_physics_labels(dataset, [as_named])
    np.testing.assert_array_equal(labels_tuple, labels_mapping)
    np.testing.assert_array_equal(labels_tuple, labels_named)

    with pytest.raises(ValueError):
        build_physics_labels(dataset, [(28.5, 77.0, _PRESSURE_PA)])
    with pytest.raises(ValueError):
        build_physics_labels(
            dataset, [{"lat": 28.5, "lon": 77.0, "pressure_pa": _PRESSURE_PA}]
        )


# ---------------------------------------------------------------------------
# Route-profile sampling
# ---------------------------------------------------------------------------


def test_sample_route_profiles_uses_real_dataset_times() -> None:
    dataset = _mock_met(1.3)
    samples = sample_route_profiles(
        dataset, [("VIDP", "VABB")], [_CRUISE_FL], [0], points_per_route=5
    )

    assert len(samples) == 5
    expected_time = pd.Timestamp("2025-01-01T00:00", tz="UTC")
    for sample in samples:
        assert isinstance(sample, PhysicsSample)
        assert 15.0 <= sample.lat <= 32.0
        assert 68.0 <= sample.lon <= 82.0
        assert sample.pressure_pa == pytest.approx(_PRESSURE_PA)
        assert pd.Timestamp(sample.time) == expected_time

    _, labels, _ = build_physics_labels(dataset, samples)
    assert labels.shape == (5,)
    assert np.all(labels == LABEL_FORMING_PERSISTENT)


def test_sample_route_profiles_orders_combinations() -> None:
    dataset = _mock_met(1.3)
    samples = sample_route_profiles(
        dataset,
        [("VIDP", "VABB")],
        [330.0, 350.0],
        [0, 1],
        points_per_route=3,
    )
    assert len(samples) == 2 * 2 * 3

    # Order is (route, flight level, time, waypoint).
    assert samples[0].pressure_pa == pytest.approx(fl_to_p(330.0) * 100.0)
    assert samples[5].pressure_pa == pytest.approx(fl_to_p(330.0) * 100.0)
    assert pd.Timestamp(samples[5].time) == pd.Timestamp(
        "2025-01-01T01:00", tz="UTC"
    )
    assert samples[11].pressure_pa == pytest.approx(_PRESSURE_PA)


def test_sample_route_profiles_accepts_renamed_time_axis() -> None:
    dataset = _mock_met(1.3).rename({"time": "valid_time"})
    samples = sample_route_profiles(
        dataset, [("VIDP", "VABB")], [_CRUISE_FL], [2], points_per_route=4
    )
    assert len(samples) == 4
    assert pd.Timestamp(samples[0].time) == pd.Timestamp(
        "2025-01-01T02:00", tz="UTC"
    )


def test_sample_route_profiles_validates_input() -> None:
    dataset = _mock_met(1.3)
    with pytest.raises(ValueError):
        sample_route_profiles(dataset, [("XXXX", "VABB")], [_CRUISE_FL], [0])
    with pytest.raises(ValueError):
        sample_route_profiles(dataset, [("VIDP", "VABB")], [_CRUISE_FL], [9])
    with pytest.raises(ValueError):
        sample_route_profiles(
            dataset, [("VIDP", "VABB")], [_CRUISE_FL], ["2025-06-01T00:00"]
        )
    without_time = xr.Dataset({"temperature": ("x", [1.0])})
    with pytest.raises(ValueError):
        sample_route_profiles(without_time, [("VIDP", "VABB")], [_CRUISE_FL], [0])


# ---------------------------------------------------------------------------
# Dataset CSV round-trip
# ---------------------------------------------------------------------------


def test_write_read_dataset_roundtrip(tmp_path) -> None:
    dataset = _mock_met(1.3)
    samples = [
        (28.5, 77.0, _PRESSURE_PA, _SAMPLE_TIME),
        (10.0, 77.0, _PRESSURE_PA, _SAMPLE_TIME),
    ]
    features, labels, meta = build_physics_labels(dataset, samples)

    path = tmp_path / "labels.csv"
    write_dataset(path, features, labels, meta)
    read_features, read_labels, read_meta = read_dataset(path)

    # In-domain rows round-trip exactly.
    np.testing.assert_allclose(read_features[0], features[0], rtol=1e-12)
    assert read_labels[0] == LABEL_FORMING_PERSISTENT
    assert read_meta[0]["lat"] == pytest.approx(28.5)
    assert read_meta[0]["lon"] == pytest.approx(77.0)
    assert read_meta[0]["pressure_pa"] == pytest.approx(_PRESSURE_PA)
    assert read_meta[0]["time"].startswith("2025-01-01T01:00")

    # The censored row keeps its NaN features, but the shared
    # lat/lon/pressure columns carry the provenance of the sample.
    provenance_features = {
        FEATURE_NAMES.index("lat"): 10.0,
        FEATURE_NAMES.index("lon"): 77.0,
        FEATURE_NAMES.index("pressure_pa"): _PRESSURE_PA,
    }
    for index, value in enumerate(read_features[1]):
        if index in provenance_features:
            assert value == pytest.approx(provenance_features[index])
        else:
            assert math.isnan(value)
    assert math.isnan(read_labels[1])
    assert math.isnan(labels[1])
    assert read_meta[1]["lat"] == pytest.approx(10.0)


def test_write_dataset_validates_shapes(tmp_path) -> None:
    features = np.zeros((2, len(FEATURE_NAMES)))
    labels = np.zeros(2)
    meta = [
        {"lat": 28.0, "lon": 77.0, "pressure_pa": _PRESSURE_PA, "time": _SAMPLE_TIME},
        {"lat": 28.0, "lon": 77.0, "pressure_pa": _PRESSURE_PA, "time": _SAMPLE_TIME},
    ]
    path = tmp_path / "dataset.csv"
    with pytest.raises(ValueError):
        write_dataset(path, features[:, :-1], labels, meta)
    with pytest.raises(ValueError):
        write_dataset(path, features, labels[:1], meta)
    with pytest.raises(ValueError):
        write_dataset(path, features, labels, meta[:1])
    with pytest.raises(ValueError):
        write_dataset(
            path,
            features,
            labels,
            [
                {"lat": 28.0, "lon": 77.0, "pressure_pa": _PRESSURE_PA},
                {"lat": 28.0, "lon": 77.0, "pressure_pa": _PRESSURE_PA},
            ],
        )


def test_read_dataset_validates_columns_and_provenance(tmp_path) -> None:
    incomplete = tmp_path / "incomplete.csv"
    pd.DataFrame({"lat": [28.0], "lon": [77.0]}).to_csv(incomplete, index=False)
    with pytest.raises(ValueError):
        read_dataset(incomplete)

    frame = pd.DataFrame(np.zeros((1, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    frame["label"] = 1.0
    frame["lat"] = np.nan
    frame["lon"] = 77.0
    frame["pressure_pa"] = _PRESSURE_PA
    frame["time"] = _SAMPLE_TIME
    non_finite = tmp_path / "non_finite.csv"
    frame.to_csv(non_finite, index=False)
    with pytest.raises(ValueError):
        read_dataset(non_finite)


# ---------------------------------------------------------------------------
# Satellite scene helpers
# ---------------------------------------------------------------------------


def test_load_bt_pair_roundtrip(tmp_path) -> None:
    bt_108 = np.arange(12.0).reshape(3, 4) + 240.0
    bt_120 = bt_108 - 2.0
    path = tmp_path / "scene.nc"
    xr.Dataset(
        {
            "IR_108": (("row", "col"), bt_108),
            "IR_120": (("row", "col"), bt_120),
        }
    ).to_netcdf(path)

    loaded_108, loaded_120 = load_bt_pair(path)
    np.testing.assert_array_equal(loaded_108, bt_108)
    np.testing.assert_array_equal(loaded_120, bt_120)

    renamed = tmp_path / "renamed.nc"
    xr.Dataset(
        {"a": (("y", "x"), bt_108), "b": (("y", "x"), bt_120)}
    ).to_netcdf(renamed)
    custom_108, custom_120 = load_bt_pair(renamed, channel_108="a", channel_120="b")
    np.testing.assert_array_equal(custom_108, bt_108)
    np.testing.assert_array_equal(custom_120, bt_120)


def test_load_bt_pair_rejects_malformed_scenes(tmp_path) -> None:
    missing = tmp_path / "missing.nc"
    xr.Dataset({"IR_108": (("row", "col"), np.zeros((3, 3)))}).to_netcdf(missing)
    with pytest.raises(ValueError):
        load_bt_pair(missing)

    three_dimensional = tmp_path / "three_dimensional.nc"
    xr.Dataset(
        {
            "IR_108": (("t", "row", "col"), np.zeros((2, 3, 3))),
            "IR_120": (("t", "row", "col"), np.zeros((2, 3, 3))),
        }
    ).to_netcdf(three_dimensional)
    with pytest.raises(ValueError):
        load_bt_pair(three_dimensional)


def test_viewing_zenith_angle_known_geometry() -> None:
    assert viewing_zenith_angle_deg(0.0, 45.5, 45.5) == 0.0
    assert viewing_zenith_angle_deg(
        28.5, 77.0, 45.5
    ) == pytest.approx(47.90795804503825, rel=1e-9)
    # Beyond the geostationary limb there is no line of sight.
    assert math.isnan(viewing_zenith_angle_deg(0.0, 130.0, 45.5))
    assert math.isnan(viewing_zenith_angle_deg(float("nan"), 77.0, 45.5))
    # The angle grows with the distance from the sub-satellite point.
    assert viewing_zenith_angle_deg(20.0, 60.0, 45.5) < viewing_zenith_angle_deg(
        28.5, 77.0, 45.5
    )


# ---------------------------------------------------------------------------
# Observed labels
# ---------------------------------------------------------------------------

_SCENE_SIZE = 80
_SCENE_NORTH_LAT = 29.0
_SCENE_WEST_LON = 76.0
_SCENE_SPAN_DEG = 2.0
_PLANTED_LINE_PX = 21
_PLANTED_ANOMALY_K = 2.5


def _scene_georeferencer() -> ArrayGeoreferencer:
    """North-up 80x80 georeferencer over 27-29 N, 76-78 E."""
    lats = np.linspace(_SCENE_NORTH_LAT, _SCENE_NORTH_LAT - _SCENE_SPAN_DEG, _SCENE_SIZE)
    lons = np.linspace(_SCENE_WEST_LON, _SCENE_WEST_LON + _SCENE_SPAN_DEG, _SCENE_SIZE)
    lats_grid, lons_grid = np.meshgrid(lats, lons, indexing="ij")
    return ArrayGeoreferencer(lats_grid, lons_grid)


def _scene_rowcol(lat: float, lon: float) -> tuple[float, float]:
    """Invert the linear scene grid: ``(lat, lon) -> (row, col)``."""
    scale = _SCENE_SPAN_DEG / (_SCENE_SIZE - 1)
    return (_SCENE_NORTH_LAT - lat) / scale, (lon - _SCENE_WEST_LON) / scale


def _apparent_segment_position(
    ds_met: xr.Dataset,
    observation_time: pd.Timestamp,
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> tuple[float, float]:
    """Image-space apparent position of one cruise segment.

    Mirrors the production chain (segment-mean wind advection, then the
    Meteosat parallax of the ISA altitude of the cruise pressure) so the
    fixture can plant a detection where the module under test expects
    one. Only the fixture uses these formulas; the labels themselves are
    produced by :func:`build_observed_labels`.
    """
    mid_lat = 0.5 * (lat_start + lat_end)
    mid_lon = 0.5 * (lon_start + lon_end)
    mid_time = t_start + (t_end - t_start) / 2

    met_start = interpolate_met_to_segment(ds_met, lat_start, lon_start, _PRESSURE_PA, t_start)
    met_end = interpolate_met_to_segment(ds_met, lat_end, lon_end, _PRESSURE_PA, t_end)
    u_wind = 0.5 * (met_start["u_component_of_wind"] + met_end["u_component_of_wind"])
    v_wind = 0.5 * (met_start["v_component_of_wind"] + met_end["v_component_of_wind"])

    elapsed = (observation_time - mid_time).total_seconds()
    advected_lat, advected_lon = advect_segment(
        mid_lat, mid_lon, u_wind, v_wind, elapsed
    )
    altitude_m = pressure_to_altitude_m(_PRESSURE_PA)
    delta_lat, delta_lon = calculate_parallax_displacement(
        advected_lat, advected_lon, altitude_m
    )
    return advected_lat + delta_lat, advected_lon + delta_lon


def _planted_scene() -> tuple[np.ndarray, np.ndarray]:
    """Split-window scene with a BTD line where the matched track appears."""
    observation = pd.Timestamp("2025-01-01T02:30:00Z")
    apparent_lat, apparent_lon = _apparent_segment_position(
        _mock_met(1.3, high_cloud_boundary_lat=_HIGH_CLOUD_BOUNDARY_LAT),
        observation,
        28.0,
        76.70,
        28.0,
        76.74,
        pd.Timestamp("2025-01-01T01:00Z"),
        pd.Timestamp("2025-01-01T01:30Z"),
    )
    row, col = _scene_rowcol(apparent_lat, apparent_lon)
    assert 0 <= row <= _SCENE_SIZE - 1
    assert 0 <= col <= _SCENE_SIZE - 1

    rng = np.random.default_rng(3)
    bt_108 = 240.0 + rng.normal(0.0, 0.2, (_SCENE_SIZE, _SCENE_SIZE))
    bt_120 = 240.0 + rng.normal(0.0, 0.2, (_SCENE_SIZE, _SCENE_SIZE))

    half = (_PLANTED_LINE_PX - 1) // 2
    start_col = int(round(col)) - half
    bt_108[int(round(row)), start_col : start_col + _PLANTED_LINE_PX] += (
        _PLANTED_ANOMALY_K
    )
    return bt_108, bt_120


def _observed_trajectories() -> pd.DataFrame:
    """Three flights: matched track, observable miss, unobservable track."""
    matched_times = [
        pd.Timestamp("2025-01-01T01:00Z"),
        pd.Timestamp("2025-01-01T01:30Z"),
    ]
    miss_times = [
        pd.Timestamp("2025-01-01T01:00Z"),
        pd.Timestamp("2025-01-01T01:30Z"),
        pd.Timestamp("2025-01-01T02:00Z"),
    ]
    records: list[dict[str, object]] = []
    for lat, lon, timestamp in zip(
        [28.0, 28.0], [76.70, 76.74], matched_times
    ):
        records.append(
            {
                "icao24": "aaa111",
                "time": timestamp,
                "lat": lat,
                "lon": lon,
                "baro_alt_ft": 35000.0,
            }
        )
    for lat, lon, timestamp in zip(
        [29.3, 29.3, 29.3], [76.70, 76.74, 76.78], miss_times
    ):
        records.append(
            {
                "icao24": "bbb222",
                "time": timestamp,
                "lat": lat,
                "lon": lon,
                "baro_alt_ft": 35000.0,
            }
        )
    for lat, lon, timestamp in zip(
        [30.5, 30.5, 30.5], [76.70, 76.74, 76.78], miss_times
    ):
        records.append(
            {
                "icao24": "ccc333",
                "time": timestamp,
                "lat": lat,
                "lon": lon,
                "baro_alt_ft": 35000.0,
            }
        )
    return pd.DataFrame(records)


def test_build_observed_labels_matches_misses_and_censors() -> None:
    georef = _scene_georeferencer()
    dataset = _mock_met(1.3, high_cloud_boundary_lat=_HIGH_CLOUD_BOUNDARY_LAT)
    bt_108, bt_120 = _planted_scene()
    observation = pd.Timestamp("2025-01-01T02:30:00Z")

    # The planted line must survive the detector's threshold and length
    # gates before the association can be exercised.
    detection = detect_contrail_lines(bt_108, bt_120, threshold=2.0, min_length_px=8)
    assert detection["num_detected"] >= 1

    features, labels, meta = build_observed_labels(
        bt_108=bt_108,
        bt_120=bt_120,
        georef=georef,
        observation_time=observation,
        trajectories=_observed_trajectories(),
        ds_met=dataset,
    )

    assert features.shape == (5, len(FEATURE_NAMES))
    assert np.all(np.isfinite(features))

    matched = meta[0]
    assert matched["icao24"] == "aaa111"
    assert matched["matched"] is True
    assert matched["observable"] is True
    assert labels[0] == 1.0
    assert matched["label"] == 1.0
    assert matched["censored_reason"] is None
    assert matched["detection_count"] >= 1
    assert matched["elapsed_seconds"] > 0.0
    assert math.isfinite(matched["advected_lat"])
    assert math.isfinite(matched["parallax_delta_lat_deg"])

    for index, entry in enumerate(meta[1:3], start=1):
        assert entry["icao24"] == "bbb222"
        assert entry["matched"] is False
        assert entry["observable"] is True
        assert entry["observable_slots"] == 2
        assert entry["label"] == 0.0
        assert labels[index] == 0.0
        assert entry["censored_reason"] is None

    for index, entry in enumerate(meta[3:5], start=3):
        assert entry["icao24"] == "ccc333"
        assert entry["observable"] is False
        assert entry["observable_slots"] == 0
        assert math.isnan(entry["label"])
        assert math.isnan(labels[index])


def test_build_observed_labels_censors_after_observation() -> None:
    georef = _scene_georeferencer()
    dataset = _mock_met(1.3)
    bt_108, bt_120 = _planted_scene()
    trajectories = pd.DataFrame(
        [
            {
                "icao24": "ddd444",
                "time": pd.Timestamp("2025-01-01T02:30Z"),
                "lat": 28.0,
                "lon": 76.70,
                "baro_alt_ft": 35000.0,
            },
            {
                "icao24": "ddd444",
                "time": pd.Timestamp("2025-01-01T02:45Z"),
                "lat": 28.0,
                "lon": 76.74,
                "baro_alt_ft": 35000.0,
            },
        ]
    )

    _, labels, meta = build_observed_labels(
        bt_108=bt_108,
        bt_120=bt_120,
        georef=georef,
        observation_time=pd.Timestamp("2025-01-01T02:30Z"),
        trajectories=trajectories,
        ds_met=dataset,
    )

    assert len(meta) == 1
    assert meta[0]["elapsed_seconds"] < 0.0
    assert meta[0]["censored_reason"] == "after_observation"
    assert math.isnan(labels[0])


def test_build_observed_labels_validates_scene_and_trajectories() -> None:
    georef = _scene_georeferencer()
    dataset = _mock_met(1.3)
    observation = pd.Timestamp("2025-01-01T02:30:00Z")
    trajectories = _observed_trajectories()

    with pytest.raises(ValueError):
        build_observed_labels(
            bt_108=np.zeros((4, 4)),
            bt_120=np.zeros((4, 5)),
            georef=georef,
            observation_time=observation,
            trajectories=trajectories,
            ds_met=dataset,
        )
    with pytest.raises(ValueError):
        build_observed_labels(
            bt_108=np.zeros((4, 4)),
            bt_120=np.zeros((4, 4)),
            georef=georef,
            observation_time=observation,
            trajectories=trajectories.drop(columns=["baro_alt_ft"]),
            ds_met=dataset,
        )
    with pytest.raises(ValueError):
        build_observed_labels(
            bt_108=np.zeros((4, 4)),
            bt_120=np.zeros((4, 4)),
            georef=georef,
            observation_time=observation,
            trajectories=trajectories.iloc[0:0],
            ds_met=dataset,
        )
