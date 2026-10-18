"""Tests for OpenSky-style ADS-B CSV ingestion (``stratum.io.adsb``)."""

from __future__ import annotations

import pandas as pd
import pytest

from stratum.io import ADSB_COLUMNS, load_adsb_csv, select_trajectory
from stratum.io.adsb import METRES_TO_FEET, MPS_TO_FPM, OPENSKY_COLUMN_MAP

#: 2025-01-01T06:00:00Z as Unix seconds.
BASE_EPOCH_S = 1735711200
BASE_TIME = pd.Timestamp("2025-01-01T06:00:00")


def _opensky_row(*, drop: tuple[str, ...] = (), **overrides: object) -> dict[str, object]:
    """Build one OpenSky-style CSV row (metre altitude, m/s vertical rate)."""
    row: dict[str, object] = {
        "time": BASE_EPOCH_S,
        "icao24": "abc123",
        "callsign": "AIC123",
        "lat": 20.0,
        "lon": 75.0,
        "baro_altitude": 10668.0,  # metres -> 35000 ft
        "geo_altitude": 10700.0,  # metres
        "velocity": 450.0,  # knots, taken as-is
        "true_track": 90.0,
        "vertical_rate": 5.0,  # m/s -> 984.25 ft/min
        "on_ground": "False",
    }
    for name in drop:
        row.pop(name)
    row.update(overrides)
    return row


def _write_csv(tmp_path, rows: list[dict[str, object]], name: str = "adsb.csv"):
    """Write rows to a CSV file and return its path."""
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _point(**overrides: object) -> dict[str, object]:
    """Build one canonical-schema point with airborne cruise defaults."""
    point: dict[str, object] = {
        "icao24": "abc123",
        "time": BASE_TIME,
        "lat": 20.0,
        "lon": 75.0,
        "baro_alt_ft": 35000.0,
        "geo_alt_ft": 35200.0,
        "gs_kt": 450.0,
        "track_deg": 90.0,
        "vrate_fpm": 0.0,
        "callsign": "AIC123",
        "on_ground": False,
        "source": "opensky",
    }
    point.update(overrides)
    return point


# ---------------------------------------------------------------------------
# Column mapping contract
# ---------------------------------------------------------------------------


def test_opensky_column_map_contract() -> None:
    aliases = {
        "callsign": "callsign",
        "call_sign": "callsign",
        "time": "time",
        "timestamp": "time",
        "snapshot_time": "time",
        "position_time": "time",
        "lat": "lat",
        "latitude": "lat",
        "lon": "lon",
        "longitude": "lon",
        "baro_alt_ft": "baro_alt_ft",
        "baro_altitude": "baro_alt_ft",
        "altitude": "baro_alt_ft",
        "geo_altitude": "geo_alt_ft",
        "velocity": "gs_kt",
        "ground_speed": "gs_kt",
        "spd": "gs_kt",
        "track": "track_deg",
        "heading": "track_deg",
        "true_track": "track_deg",
        "vertical_rate": "vrate_fpm",
        "vrate": "vrate_fpm",
        "baro_rate": "vrate_fpm",
        "on_ground": "on_ground",
    }
    for raw, canonical in aliases.items():
        assert OPENSKY_COLUMN_MAP[raw] == canonical
    assert set(OPENSKY_COLUMN_MAP.values()) <= set(ADSB_COLUMNS)


# ---------------------------------------------------------------------------
# CSV loading: units, timestamps, headers, errors
# ---------------------------------------------------------------------------


def test_load_opensky_style_converts_units_and_cleans(tmp_path) -> None:
    path = _write_csv(
        tmp_path,
        [
            _opensky_row(),
            _opensky_row(
                time=BASE_EPOCH_S + 60,
                icao24="def456",
                callsign="IGO999",
                on_ground="True",  # dropped: on the ground
            ),
            _opensky_row(
                time=BASE_EPOCH_S + 120,
                icao24="aaa111",
                callsign="SEJ111",
                baro_altitude=6000.0,  # ~19685 ft, dropped: below FL250
            ),
        ],
    )

    points = load_adsb_csv(path)

    assert list(points.columns) == list(ADSB_COLUMNS)
    assert len(points) == 1
    row = points.iloc[0]
    assert row["icao24"] == "abc123"
    assert row["baro_alt_ft"] == pytest.approx(35000.0, rel=1e-12)
    assert row["geo_alt_ft"] == pytest.approx(10700.0 * METRES_TO_FEET, rel=1e-12)
    assert row["vrate_fpm"] == pytest.approx(5.0 * MPS_TO_FPM, rel=1e-12)
    assert row["gs_kt"] == pytest.approx(450.0)
    assert row["track_deg"] == pytest.approx(90.0)
    assert row["callsign"] == "AIC123"
    assert bool(row["on_ground"]) is False
    assert row["source"] == "opensky"


def test_load_parses_iso_times_and_messy_headers(tmp_path) -> None:
    row = _opensky_row(drop=("time",))
    row[" Time "] = "2025-01-01T06:00:00Z"
    row["ICAO24"] = row.pop("icao24")
    row["Call_Sign"] = row.pop("callsign")
    row["LATITUDE"] = row.pop("lat")
    row["Longitude"] = row.pop("lon")
    path = _write_csv(tmp_path, [row])

    points = load_adsb_csv(path)

    assert points.loc[0, "time"] == BASE_TIME
    assert str(points["time"].dtype) == "datetime64[ns]"
    assert points["time"].dt.tz is None
    assert points.loc[0, "callsign"] == "AIC123"
    assert points.loc[0, "lat"] == pytest.approx(20.0)
    assert points.loc[0, "lon"] == pytest.approx(75.0)


def test_time_unit_auto_detects_seconds_and_milliseconds(tmp_path) -> None:
    seconds_path = _write_csv(tmp_path, [_opensky_row()], name="seconds.csv")
    millis_row = _opensky_row(drop=("time",))
    millis_row["timestamp"] = BASE_EPOCH_S * 1000
    millis_path = _write_csv(tmp_path, [millis_row], name="millis.csv")

    seconds = load_adsb_csv(seconds_path)
    millis = load_adsb_csv(millis_path)
    explicit_millis = load_adsb_csv(millis_path, time_unit="ms")

    for points in (seconds, millis, explicit_millis):
        assert points.loc[0, "time"] == BASE_TIME


def test_unknown_time_unit_raises(tmp_path) -> None:
    path = _write_csv(tmp_path, [_opensky_row()])
    with pytest.raises(ValueError, match="time_unit"):
        load_adsb_csv(path, time_unit="fortnight")


def test_missing_required_columns_raise_with_names(tmp_path) -> None:
    path = _write_csv(tmp_path, [_opensky_row(drop=("lon",))])
    with pytest.raises(ValueError) as excinfo:
        load_adsb_csv(path)
    message = str(excinfo.value)
    assert "missing required columns" in message
    assert "'lon'" in message

    no_altitude = _write_csv(
        tmp_path,
        [_opensky_row(drop=("baro_altitude", "geo_altitude"))],
        name="no_altitude.csv",
    )
    with pytest.raises(ValueError) as excinfo:
        load_adsb_csv(no_altitude)
    assert "'baro_alt_ft'" in str(excinfo.value)


def test_load_accepts_feet_and_fpm_headers(tmp_path) -> None:
    path = _write_csv(
        tmp_path,
        [
            {
                "time": BASE_EPOCH_S,
                "icao24": "abc123",
                "callsign": "AIC123",
                "lat": 20.0,
                "lon": 75.0,
                "baro_alt_ft": 35000.0,
                "gs_kt": 450.0,
                "track_deg": 90.0,
                "vrate": 900.0,
                "on_ground": "False",
            }
        ],
    )

    points = load_adsb_csv(path)

    assert points.loc[0, "baro_alt_ft"] == pytest.approx(35000.0)
    assert points.loc[0, "vrate_fpm"] == pytest.approx(900.0)
    assert pd.isna(points.loc[0, "geo_alt_ft"])
    assert points.loc[0, "source"] == "opensky"


def test_geo_altitude_used_when_baro_altitude_absent(tmp_path) -> None:
    path = _write_csv(tmp_path, [_opensky_row(drop=("baro_altitude",))])

    points = load_adsb_csv(path)

    expected = 10700.0 * METRES_TO_FEET
    assert points.loc[0, "baro_alt_ft"] == pytest.approx(expected, rel=1e-12)
    assert points.loc[0, "geo_alt_ft"] == pytest.approx(expected, rel=1e-12)


def test_column_map_override_maps_custom_headers(tmp_path) -> None:
    row = _opensky_row(drop=("callsign",))
    row["ac_id"] = "AIC123"
    path = _write_csv(tmp_path, [row])

    points = load_adsb_csv(path, column_map={"ac_id": "callsign"})

    assert points.loc[0, "callsign"] == "AIC123"


# ---------------------------------------------------------------------------
# Trajectory selection
# ---------------------------------------------------------------------------


def test_select_trajectory_by_callsign_case_and_whitespace() -> None:
    frame = pd.DataFrame(
        [
            _point(time=BASE_TIME + pd.Timedelta(minutes=2), callsign=" AIC123 "),
            _point(time=BASE_TIME, callsign="aic123"),
            _point(icao24="def456", callsign="IGO999"),
        ]
    )

    selected = select_trajectory(frame, callsign="  aIc123  ")

    assert len(selected) == 2
    assert list(selected["time"]) == [BASE_TIME, BASE_TIME + pd.Timedelta(minutes=2)]
    assert list(selected.index) == [0, 1]


def test_select_trajectory_by_icao24() -> None:
    frame = pd.DataFrame(
        [
            _point(time=BASE_TIME + pd.Timedelta(minutes=5), icao24="ABC123"),
            _point(time=BASE_TIME, icao24="abc123"),
            _point(icao24="def456", callsign="IGO999"),
        ]
    )

    selected = select_trajectory(frame, icao24=" ABc123 ")

    assert list(selected["time"]) == [BASE_TIME, BASE_TIME + pd.Timedelta(minutes=5)]
    assert set(selected["icao24"]) == {"abc123", "ABC123"}


def test_select_trajectory_requires_exactly_one_identifier() -> None:
    frame = pd.DataFrame([_point()])

    with pytest.raises(ValueError, match="exactly one"):
        select_trajectory(frame)
    with pytest.raises(ValueError, match="exactly one"):
        select_trajectory(frame, callsign="AIC123", icao24="abc123")


def test_select_trajectory_missing_identifier_column_raises() -> None:
    frame = pd.DataFrame([_point()]).drop(columns=["callsign"])

    with pytest.raises(ValueError, match="missing required columns"):
        select_trajectory(frame, callsign="AIC123")


def test_load_then_select_returns_sorted_flight(tmp_path) -> None:
    rows = [
        _opensky_row(time=BASE_EPOCH_S + 60, callsign=" IGO999 "),
        _opensky_row(time=BASE_EPOCH_S),
        _opensky_row(time=BASE_EPOCH_S + 120, icao24="def456", callsign="igO999"),
    ]
    points = load_adsb_csv(_write_csv(tmp_path, rows))

    flight = select_trajectory(points, callsign="igo999")

    assert len(flight) == 2
    assert list(flight["time"]) == [
        BASE_TIME + pd.Timedelta(minutes=1),
        BASE_TIME + pd.Timedelta(minutes=2),
    ]
    assert list(flight.index) == [0, 1]
