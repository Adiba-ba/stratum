"""Smoke tests for the STRATUM command line subcommands.

The network-facing paths (``check-live``, ``goes``, ``adsb-live``) are not
executed here. Their pure helpers are tested with fixture payloads instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from stratum.cli import demos, fetch, verify
from stratum.cli.main import main


def test_main_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "0.1.0" in capsys.readouterr().out


def test_main_dispatches_verify_baseline_json() -> None:
    assert main(["verify-baseline", "--json"]) == 0


def test_verify_baseline_json_reports_all_checks_passed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert verify.verify_baseline(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["checks_total"] > 0
    assert payload["checks_passed"] == payload["checks_total"]
    assert all(check["passed"] for check in payload["checks"])
    assert any(
        check["name"].startswith("G @ 250 hPa") for check in payload["checks"]
    )


def test_screen_routes_writes_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "screening.md"
    assert verify.screen_routes(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "| Route | Season/Regime | Formation % | Persistence % |" in text
    assert "VIDP-VABB (Delhi-Mumbai)" in text
    assert "Indian corridor contrail screening" in capsys.readouterr().out


def test_screen_routes_honours_eta(capsys: pytest.CaptureFixture[str]) -> None:
    assert verify.screen_routes(["--eta", "0.35"]) == 0
    assert "eta = 0.35" in capsys.readouterr().out


def test_demo_detector_prints_banner_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert demos.demo(["detector"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == demos.SYNTHETIC_BANNER
    assert "Contrail detected: YES" in out


def test_demo_matching_prints_banner_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert demos.demo(["matching"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == demos.SYNTHETIC_BANNER
    assert "Tri-state labels as expected: YES" in out


def test_demo_decision_prints_banner_first(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert demos.demo(["decision"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == demos.SYNTHETIC_BANNER


def _goes_item(name: str) -> dict:
    """Minimal GCS JSON API object entry."""
    return {"name": name, "size": "1"}


def _goes_granule(band: int, start: str, scan_type: str | None = None) -> str:
    """Build a plausible full-disk ABI granule name for one scan start."""
    field = "" if scan_type is None else f"_t{scan_type}"
    return (
        "ABI-L2-CMIPF/2024/001/00/"
        f"OR_ABI-L2-CMIPF-M6C{band:02d}_G16{field}_s{start}.nc"
    )


def test_select_goes_object_picks_earliest_start_per_band() -> None:
    listing = [
        _goes_item(_goes_granule(14, "20240010030205")),
        _goes_item(_goes_granule(14, "20240010000205")),
        _goes_item(_goes_granule(15, "20240010010205")),
        _goes_item(_goes_granule(14, "20240010020205")),
    ]
    assert fetch.select_goes_object(listing, 14) == _goes_granule(
        14, "20240010000205"
    )
    assert fetch.select_goes_object(listing, 15) == _goes_granule(
        15, "20240010010205"
    )


def test_select_goes_object_skips_non_full_scan_type() -> None:
    listing = [
        _goes_item(_goes_granule(14, "20240010000205", scan_type="B")),
        _goes_item(_goes_granule(14, "20240010010205", scan_type="F")),
    ]
    assert fetch.select_goes_object(listing, 14) == _goes_granule(
        14, "20240010010205", scan_type="F"
    )


def test_select_goes_object_returns_none_without_match() -> None:
    listing = [
        _goes_item(_goes_granule(1, "20240010000205")),
        _goes_item("ABI-L2-CMIPF/2024/001/00/index.html"),
    ]
    assert fetch.select_goes_object(listing, 14) is None


def _opensky_state(
    *,
    time_position: float | None = 1700000000,
    callsign: str | None = "KLM102 ",
) -> list:
    """One OpenSky state vector with the 2024 field order."""
    return [
        "abc123",
        callsign,
        "Netherlands",
        time_position,
        1700000010,
        76.1,
        24.2,
        10668.0,
        False,
        250.0,
        271.0,
        -2.5,
        None,
        10820.0,
        "1234",
        False,
        0,
    ]


def test_opensky_states_to_frame_converts_units_and_headers() -> None:
    frame = fetch.opensky_states_to_frame([_opensky_state()])
    assert list(frame.columns) == [
        "icao24",
        "callsign",
        "time",
        "lat",
        "lon",
        "baro_alt_ft",
        "velocity",
        "track_deg",
        "vrate_fpm",
        "on_ground",
        "geo_alt_ft",
    ]
    row = frame.iloc[0]
    assert row["icao24"] == "abc123"
    assert row["callsign"] == "KLM102"
    assert row["time"] == 1700000000
    assert row["lat"] == pytest.approx(24.2)
    assert row["lon"] == pytest.approx(76.1)
    assert row["baro_alt_ft"] == pytest.approx(10668.0 * 3.280839895013123)
    assert row["velocity"] == pytest.approx(250.0)
    assert row["track_deg"] == pytest.approx(271.0)
    assert row["vrate_fpm"] == pytest.approx(-2.5 * 196.8503937007874)
    assert bool(row["on_ground"]) is False
    assert row["geo_alt_ft"] == pytest.approx(10820.0 * 3.280839895013123)


def test_opensky_states_to_frame_tolerates_missing_fields() -> None:
    short_state = ["def456", None, "France", None, 1700000010, 2.0, 48.0, None]
    frame = fetch.opensky_states_to_frame([short_state, _opensky_state()])
    assert len(frame) == 2
    first = frame.iloc[0]
    assert first["callsign"] == ""
    assert first["time"] == 1700000010
    assert pd.isna(first["baro_alt_ft"])
    assert pd.isna(first["vrate_fpm"])
    assert pd.isna(first["geo_alt_ft"])
    assert bool(first["on_ground"]) is False


def test_fetch_requires_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    assert fetch.fetch([]) == 2
    assert "goes" in capsys.readouterr().out


def test_parse_doy_accepts_both_forms() -> None:
    assert fetch._parse_doy("2024-001") == (2024, 1)
    assert fetch._parse_doy("2024001") == (2024, 1)
    with pytest.raises(ValueError):
        fetch._parse_doy("2024-400")
