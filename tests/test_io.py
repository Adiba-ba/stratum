"""Tests for the real-data ingestion layer (Step 7)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from stratum.io import (
    ADSB_COLUMNS,
    ERA5_VARIABLES,
    SEVIRI_CHANNELS,
    clean_trajectory_points,
    crop_to_domain,
    interpolate_met_to_segment,
    load_era5_dataset,
)

# The netCDF4 1.7.2 wheel can emit a spurious NumPy binary-compatibility
# RuntimeWarning on first engine use under pytest; functionality is
# covered by the assertions themselves.
pytestmark = pytest.mark.filterwarnings(
    "ignore:numpy.ndarray size changed, may indicate binary incompatibility"
)

BASE_TIME = pd.Timestamp("2025-01-01T06:00:00Z")


# ---------------------------------------------------------------------------
# ADS-B schema and cleaning
# ---------------------------------------------------------------------------


def _record(**overrides: object) -> dict[str, object]:
    """Build one raw ADS-B record with sensible cruise defaults."""
    record: dict[str, object] = {
        "icao24": "ABC123",
        "time": BASE_TIME,
        "lat": 20.0,
        "lon": 75.0,
        "baro_alt_ft": 35000.0,
        "geo_alt_ft": 35200.0,
        "gs_kt": 450.0,
        "track_deg": 90.0,
        "vrate_fpm": 0.0,
        "callsign": " AIC123 ",
        "on_ground": False,
        "source": "OpenSky",
    }
    record.update(overrides)
    return record


def _frame_from(overrides_list: list[dict[str, object]]) -> pd.DataFrame:
    """Build a frame whose rows have distinct timestamps by default."""
    records = []
    for index, overrides in enumerate(overrides_list):
        merged: dict[str, object] = {
            "time": BASE_TIME + pd.Timedelta(minutes=index),
        }
        merged.update(overrides)
        records.append(_record(**merged))
    return pd.DataFrame(records)


def test_cleaning_drops_ground_low_altitude_and_missing() -> None:
    frame = _frame_from(
        [
            {},
            {"on_ground": True},
            {"baro_alt_ft": 10000.0},
            {"lat": None},
            {"lon": None},
            {"time": None},
            {"baro_alt_ft": None},
        ]
    )
    cleaned = clean_trajectory_points(frame)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "icao24"] == "abc123"


def test_cleaning_enforces_indian_domain() -> None:
    frame = _frame_from(
        [
            {"lat": 45.0},
            {"lat": 5.5},
            {"lon": 100.0},
            {"lon": 67.9},
            {"lat": 6.0, "lon": 98.0},
            {"lat": 37.0, "lon": 68.0},
        ]
    )
    cleaned = clean_trajectory_points(frame)
    assert len(cleaned) == 2
    assert list(cleaned["lat"]) == [6.0, 37.0]


def test_cleaning_flags_implausible_vrate_and_speed_jump() -> None:
    t0 = BASE_TIME
    frame = pd.DataFrame(
        [
            _record(time=t0, icao24="a", gs_kt=450.0, vrate_fpm=0.0),
            _record(
                time=t0 + pd.Timedelta(minutes=1),
                icao24="a",
                gs_kt=620.0,
                vrate_fpm=0.0,
            ),
            _record(
                time=t0 + pd.Timedelta(minutes=2),
                icao24="a",
                gs_kt=450.0,
                vrate_fpm=7000.0,
            ),
            _record(
                time=t0 + pd.Timedelta(minutes=3),
                icao24="a",
                gs_kt=470.0,
                vrate_fpm=-500.0,
            ),
            _record(
                time=t0 + pd.Timedelta(minutes=1),
                icao24="b",
                gs_kt=620.0,
                vrate_fpm=0.0,
            ),
        ]
    )
    cleaned = clean_trajectory_points(frame)
    assert list(cleaned["icao24"]) == ["a", "a", "b"]
    assert list(cleaned["gs_kt"]) == [450.0, 470.0, 620.0]


def test_cleaning_normalizes_sorts_and_deduplicates() -> None:
    frame = pd.DataFrame(
        [
            _record(icao24="ABC123", time=BASE_TIME + pd.Timedelta(minutes=5)),
            _record(icao24="abc123", time=BASE_TIME),
            _record(icao24="ABC123", time=BASE_TIME),
            _record(
                icao24="DEF456",
                time=BASE_TIME - pd.Timedelta(minutes=1),
                callsign="  IGO999  ",
            ),
        ]
    )
    cleaned = clean_trajectory_points(frame)
    assert list(cleaned["icao24"]) == ["abc123", "abc123", "def456"]
    assert cleaned["time"].tolist() == [
        BASE_TIME,
        BASE_TIME + pd.Timedelta(minutes=5),
        BASE_TIME - pd.Timedelta(minutes=1),
    ]
    for _, group in cleaned.groupby("icao24", sort=False):
        assert group["time"].tolist() == sorted(group["time"])
    assert str(cleaned["time"].dtype) == "datetime64[ns, UTC]"
    assert cleaned.loc[cleaned["icao24"] == "def456", "callsign"].iloc[0] == "IGO999"
    assert set(cleaned["source"]) == {"opensky"}
    assert list(cleaned.columns) == list(ADSB_COLUMNS)


def test_cleaning_missing_required_column_raises() -> None:
    frame = pd.DataFrame([_record()]).drop(columns=["baro_alt_ft"])
    with pytest.raises(ValueError, match="missing required columns"):
        clean_trajectory_points(frame)


def test_cleaning_empty_result_keeps_canonical_columns() -> None:
    frame = _frame_from([{"on_ground": True}, {"baro_alt_ft": 9000.0}])
    cleaned = clean_trajectory_points(frame)
    assert cleaned.empty
    assert list(cleaned.columns) == list(ADSB_COLUMNS)


# ---------------------------------------------------------------------------
# ERA5 ingestion and interpolation
# ---------------------------------------------------------------------------

ERA5_LATS = np.array([30.0, 25.0, 20.0, 15.0])  # descending; loader must flip
ERA5_LONS = np.array([70.0, 75.0, 80.0])
ERA5_LEVELS = np.array([200.0, 250.0, 300.0])
ERA5_TIMES = np.array(["2025-01-01T00:00", "2025-01-01T06:00"], dtype="datetime64[ns]")


def _temperature(level: float, lat: float, lon: float, t: float) -> float:
    return (
        200.0
        + 0.4 * (level - 200.0)
        + 0.2 * (lat - 15.0)
        + 0.05 * (lon - 70.0)
        + 0.3 * t
    )


def _specific_humidity(level: float, lat: float, lon: float, t: float) -> float:
    return (
        1.0e-4
        + 1.0e-6 * (level - 200.0)
        + 1.0e-7 * (lat - 15.0)
        + 1.0e-7 * (lon - 70.0)
        + 2.0e-7 * t
    )


def _u_wind(level: float, lat: float, lon: float, t: float) -> float:
    return 5.0 + 0.5 * (lat - 15.0) - 0.1 * (level - 200.0) + 0.2 * t


def _v_wind(level: float, lat: float, lon: float, t: float) -> float:
    return -2.0 + 0.1 * (lon - 70.0) + 0.05 * (lat - 15.0) + 0.1 * t


def _omega(level: float, lat: float, lon: float, t: float) -> float:
    return 0.01 * (level - 200.0) - 0.002 * (lat - 15.0) + 0.001 * t


def _cloud_fraction(level: float, lat: float, lon: float, t: float) -> float:
    return (
        0.1
        + 0.01 * (lat - 15.0)
        + 0.02 * (lon - 70.0)
        + 0.005 * (level - 200.0)
        + 0.01 * t
    )


def _expected_values(
    level: float, lat: float, lon: float, t_index: float
) -> dict[str, float]:
    """Analytic field values used to verify exact linear interpolation."""
    return {
        "temperature": _temperature(level, lat, lon, t_index),
        "specific_humidity": _specific_humidity(level, lat, lon, t_index),
        "u_component_of_wind": _u_wind(level, lat, lon, t_index),
        "v_component_of_wind": _v_wind(level, lat, lon, t_index),
        "vertical_velocity": _omega(level, lat, lon, t_index),
        "fraction_of_cloud_cover": _cloud_fraction(level, lat, lon, t_index),
    }


def _synthetic_era5() -> xr.Dataset:
    """Build a linear-in-all-coordinates ERA5-like dataset for testing."""
    level = ERA5_LEVELS[:, None, None, None]
    lat = ERA5_LATS[None, :, None, None]
    lon = ERA5_LONS[None, None, :, None]
    t_index = np.arange(ERA5_TIMES.size, dtype=float)[None, None, None, :]
    fields = {
        "temperature": _temperature(level, lat, lon, t_index),
        "specific_humidity": _specific_humidity(level, lat, lon, t_index),
        "u_component_of_wind": _u_wind(level, lat, lon, t_index),
        "v_component_of_wind": _v_wind(level, lat, lon, t_index),
        "vertical_velocity": _omega(level, lat, lon, t_index),
        "fraction_of_cloud_cover": _cloud_fraction(level, lat, lon, t_index),
    }
    target_shape = (
        ERA5_LEVELS.size,
        ERA5_LATS.size,
        ERA5_LONS.size,
        ERA5_TIMES.size,
    )
    data_vars = {
        name: (
            ("time", "level", "latitude", "longitude"),
            np.broadcast_to(field, target_shape).transpose(3, 0, 1, 2),
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


@pytest.fixture()
def era5_file(tmp_path) -> str:
    """Write the synthetic ERA5 dataset to a temporary NetCDF file."""
    path = tmp_path / "synthetic_era5.nc"
    _synthetic_era5().to_netcdf(path)
    return str(path)


def test_load_era5_dataset_validates_and_sorts_latitude(era5_file: str) -> None:
    ds = load_era5_dataset(era5_file)
    assert list(ds["latitude"].values) == pytest.approx([15.0, 20.0, 25.0, 30.0])
    assert set(ERA5_VARIABLES) <= set(ds.data_vars)


def test_interpolate_met_to_segment_matches_analytic_fields(era5_file: str) -> None:
    ds = load_era5_dataset(era5_file)
    values = interpolate_met_to_segment(
        ds, lat=22.5, lon=77.5, pressure_pa=22500.0, time_dt="2025-01-01T03:00"
    )
    expected = _expected_values(225.0, 22.5, 77.5, 0.5)
    assert set(values) == set(ERA5_VARIABLES)
    for name, expected_value in expected.items():
        assert values[name] == pytest.approx(expected_value, rel=1e-12, abs=1e-12)


def test_interpolate_accepts_tz_aware_datetimes(era5_file: str) -> None:
    ds = load_era5_dataset(era5_file)
    naive = interpolate_met_to_segment(
        ds, lat=20.0, lon=75.0, pressure_pa=25000.0, time_dt="2025-01-01T03:00"
    )
    aware = interpolate_met_to_segment(
        ds, lat=20.0, lon=75.0, pressure_pa=25000.0,
        time_dt=pd.Timestamp("2025-01-01T03:00:00Z"),
    )
    assert aware["temperature"] == pytest.approx(naive["temperature"], rel=1e-12)


def test_interpolate_outside_domain_returns_nan(era5_file: str) -> None:
    ds = load_era5_dataset(era5_file)
    values = interpolate_met_to_segment(
        ds, lat=5.0, lon=60.0, pressure_pa=22500.0, time_dt="2025-01-01T03:00"
    )
    assert math.isnan(values["temperature"])


def test_interpolate_rejects_bad_pressure(era5_file: str) -> None:
    ds = load_era5_dataset(era5_file)
    with pytest.raises(ValueError):
        interpolate_met_to_segment(ds, 20.0, 75.0, -1.0, "2025-01-01T03:00")


def test_load_era5_dataset_rejects_missing_variable(tmp_path) -> None:
    path = tmp_path / "missing_variable.nc"
    _synthetic_era5().drop_vars("temperature").to_netcdf(path)
    with pytest.raises(ValueError, match="missing variables"):
        load_era5_dataset(str(path))


def test_load_era5_dataset_rejects_missing_coordinate(tmp_path) -> None:
    path = tmp_path / "missing_coordinate.nc"
    _synthetic_era5().rename({"level": "unexpected_level"}).to_netcdf(path)
    with pytest.raises(ValueError, match="pressure level"):
        load_era5_dataset(str(path))


def test_load_era5_dataset_normalizes_cds_short_names(tmp_path) -> None:
    cds_style = _synthetic_era5().rename(
        {
            "time": "valid_time",
            "temperature": "t",
            "specific_humidity": "q",
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
            "vertical_velocity": "w",
            "fraction_of_cloud_cover": "cc",
        }
    )
    path = tmp_path / "cds_style.nc"
    cds_style.to_netcdf(path)
    ds = load_era5_dataset(str(path))
    assert "time" in ds.coords
    assert "valid_time" not in ds.coords
    assert set(ERA5_VARIABLES) <= set(ds.data_vars)
    values = interpolate_met_to_segment(
        ds, lat=22.5, lon=77.5, pressure_pa=22500.0, time_dt="2025-01-01T03:00"
    )
    expected = _expected_values(225.0, 22.5, 77.5, 0.5)
    assert values["temperature"] == pytest.approx(
        expected["temperature"], rel=1e-12
    )
    assert values["fraction_of_cloud_cover"] == pytest.approx(
        expected["fraction_of_cloud_cover"], rel=1e-12
    )


def test_load_era5_dataset_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_era5_dataset(str(tmp_path / "does_not_exist.nc"))


# ---------------------------------------------------------------------------
# Meteosat SEVIRI cropping
# ---------------------------------------------------------------------------


def test_seviri_channels_contract() -> None:
    assert SEVIRI_CHANNELS == ("IR_108", "IR_120", "IR_087")


def test_crop_to_domain_selects_subwindow() -> None:
    lats = np.array([30.0, 25.0, 20.0, 15.0])
    lons = np.array([68.0, 72.0, 76.0, 80.0])
    image = np.arange(16, dtype=float).reshape(4, 4)
    cropped, cropped_lats, cropped_lons = crop_to_domain(
        image, lats, lons, (26.0, 71.0, 19.0, 77.0)
    )
    assert list(cropped_lats) == [25.0, 20.0]
    assert list(cropped_lons) == [72.0, 76.0]
    assert cropped.shape == (2, 2)
    assert cropped == pytest.approx(np.array([[5.0, 6.0], [9.0, 10.0]]))


def test_crop_to_domain_preserves_descending_longitudes() -> None:
    lats = np.array([10.0, 20.0])
    lons = np.array([80.0, 76.0, 72.0, 68.0])
    image = np.arange(8, dtype=float).reshape(2, 4)
    cropped, _, cropped_lons = crop_to_domain(
        image, lats, lons, (25.0, 71.0, 5.0, 77.0)
    )
    assert list(cropped_lons) == [76.0, 72.0]
    assert cropped == pytest.approx(image[:, 1:3])


def test_crop_to_domain_empty_box_raises() -> None:
    lats = np.array([10.0, 20.0])
    lons = np.array([70.0, 80.0])
    with pytest.raises(ValueError, match="no pixel centers"):
        crop_to_domain(np.zeros((2, 2)), lats, lons, (15.0, 90.0, 12.0, 95.0))


def test_crop_to_domain_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="does not match"):
        crop_to_domain(
            np.zeros((3, 3)),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            (3.0, 0.0, 0.0, 3.0),
        )


def test_crop_to_domain_rejects_malformed_bbox() -> None:
    with pytest.raises(ValueError):
        crop_to_domain(
            np.zeros((2, 2)),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            (0.0, 0.0, 1.0, 1.0),
        )
