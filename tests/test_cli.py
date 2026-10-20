"""Smoke tests for the STRATUM command line subcommands.

The network-facing paths (``check-live``, ``goes``, ``adsb-live``,
``adsb-history``, ``seviri``, ``era5``) are not executed here. Their pure
helpers and injected HTTP/CDS clients are tested with fixture payloads.
"""

from __future__ import annotations

import io
import os
import json
import time
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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
    assert row["velocity"] == pytest.approx(250.0 / fetch.MPS_TO_KT)
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


# --- OpenSky authenticated fetch -------------------------------------------------


def _clear_opensky_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_OPENSKY_TOKEN_CACHE", {})


def test_opensky_token_request_body_and_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENSKY_CLIENT_ID", "cid")
    monkeypatch.setenv("OPENSKY_CLIENT_SECRET", "csecret")
    _clear_opensky_cache(monkeypatch)
    calls: dict[str, object] = {}

    def fake_post(url: str, data: dict[str, str]) -> bytes:
        calls["url"] = url
        calls["data"] = data
        return json.dumps({"access_token": "tok123", "expires_in": 1800}).encode()

    monkeypatch.setattr(fetch, "_http_post_form", fake_post)
    assert fetch._opensky_token() == "tok123"
    assert calls["url"] == fetch._OPENSKY_TOKEN_URL
    assert calls["data"] == {
        "grant_type": "client_credentials",
        "client_id": "cid",
        "client_secret": "csecret",
    }


def test_http_post_form_sets_form_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request: urllib.request.Request, timeout: float = 30.0) -> object:
        captured["headers"] = dict(request.header_items())
        captured["data"] = request.data
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    fetch._http_post_form("https://example.test/token", {"grant_type": "client_credentials"})
    assert captured["headers"].get("Content-type") == "application/x-www-form-urlencoded"
    assert captured["data"] == b"grant_type=client_credentials"


def test_opensky_token_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENSKY_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENSKY_CLIENT_SECRET", raising=False)
    # The .env fallback must not repopulate the variables in this test.
    monkeypatch.setattr(fetch, "_load_local_env", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="OPENSKY_CLIENT_ID"):
        fetch._opensky_token()


def test_opensky_request_sends_bearer_and_refreshes_on_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENSKY_CLIENT_ID", "cid")
    monkeypatch.setenv("OPENSKY_CLIENT_SECRET", "csecret")
    _clear_opensky_cache(monkeypatch)
    posts: list[dict[str, str]] = []

    def fake_post(url: str, data: dict[str, str]) -> bytes:
        posts.append(data)
        return json.dumps({"access_token": "tok", "expires_in": 1800}).encode()

    monkeypatch.setattr(fetch, "_http_post_form", fake_post)
    gets: list[tuple[str, dict[str, str] | None]] = []

    def fake_get(
        url: str, *, timeout: float = 30.0, headers: dict[str, str] | None = None
    ) -> bytes:
        gets.append((url, headers))
        if len(gets) == 1:
            raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)
        return b'{"states":[["abc","CS","X",1,2,1.0,2.0,1000.0,false,100.0,90.0,0.0,null,1000.0]]}'

    monkeypatch.setattr(fetch, "_http_get", fake_get)
    body = fetch._opensky_request("/states/all", params={"time": 123})
    assert json.loads(body)["states"]
    assert gets[0][1]["Authorization"] == "Bearer tok"
    assert "time=123" in gets[0][0]
    assert len(posts) == 2  # token re-fetched after the 401


def test_opensky_request_429_quotes_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENSKY_CLIENT_ID", "cid")
    monkeypatch.setenv("OPENSKY_CLIENT_SECRET", "csecret")
    _clear_opensky_cache(monkeypatch)
    monkeypatch.setattr(
        fetch,
        "_http_post_form",
        lambda url, data: json.dumps({"access_token": "tok"}).encode(),
    )

    def fake_get(url: str, *, timeout: float = 30.0, headers: dict[str, str] | None = None) -> bytes:
        raise urllib.error.HTTPError(
            url, 429, "Too Many Requests", {"X-Rate-Limit-Retry-After-Seconds": "7"}, None
        )

    monkeypatch.setattr(fetch, "_http_get", fake_get)
    with pytest.raises(ValueError, match="retry after 7 s"):
        fetch._opensky_request("/states/all")


def _adsb_history_args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    # OpenSky serves flight tracks for the last 30 days only, so the test
    # window sits a few hours before now.
    end = datetime.now(timezone.utc) - timedelta(hours=2)
    begin = end - timedelta(minutes=10)
    values: dict[str, object] = {
        "begin": begin.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "icao24": None,
        "max_flights": 25,
        "out": str(tmp_path / "adsb.csv"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_adsb_history_maps_flights_to_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "_opensky_token", lambda: "tok")
    calls: list[tuple[str, dict[str, object] | None]] = []

    def fake_request(path: str, *, params: dict[str, object] | None = None, **kw: object) -> bytes:
        calls.append((path, params))
        if path == "/flights/all":
            return json.dumps(
                [{"icao24": "abc123", "callsign": "TEST1 ", "firstSeen": 1000, "lastSeen": 2000}]
            ).encode()
        return json.dumps(
            {
                "icao24": "abc123",
                "callsign": "TEST1 ",
                "path": [
                    [1000, 10.0, 20.0, 1000.0, 90.0, False],
                    [1060, 10.1, 20.1, 1000.0, 90.0, True],
                    [1120, 10.2, 20.2, 1000.0, 90.0, False],
                ],
            }
        ).encode()

    monkeypatch.setattr(fetch, "_opensky_request", fake_request)
    out = tmp_path / "adsb.csv"
    args = _adsb_history_args(tmp_path, out=str(out))
    assert fetch._adsb_history(args) == 0
    frame = pd.read_csv(out)
    assert list(frame.columns) == list(fetch._OPENSKY_COLUMNS)
    assert len(frame) == 2  # the on_ground row is dropped
    assert frame.iloc[0]["icao24"] == "abc123"
    assert frame.iloc[0]["callsign"] == "TEST1"
    assert frame.iloc[0]["baro_alt_ft"] == pytest.approx(1000.0 * 3.280839895013123)
    assert bool(frame.iloc[0]["on_ground"]) is False
    assert calls[0][0] == "/flights/all"
    assert calls[0][1] == {
        "begin": int(fetch._parse_iso8601(args.begin).timestamp()),
        "end": int(fetch._parse_iso8601(args.end).timestamp()),
    }


def test_adsb_history_tracks_falls_back_to_tracks_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "_opensky_token", lambda: "tok")
    paths: list[str] = []

    def fake_request(path: str, *, params: dict[str, object] | None = None, **kw: object) -> bytes:
        paths.append(path)
        if path == "/flights/all":
            return json.dumps(
                [{"icao24": "abc123", "callsign": "T", "firstSeen": 1000, "lastSeen": 2000}]
            ).encode()
        if path == "/tracks":
            raise urllib.error.HTTPError(path, 404, "Not Found", {}, None)
        return json.dumps(
            {"icao24": "abc123", "callsign": "T", "path": [[1000, 1.0, 2.0, 100.0, 0.0, False]]}
        ).encode()

    monkeypatch.setattr(fetch, "_opensky_request", fake_request)
    out = tmp_path / "adsb.csv"
    assert fetch._adsb_history(_adsb_history_args(tmp_path, out=str(out))) == 0
    assert paths == ["/flights/all", "/tracks", "/tracks/all"]
    assert len(pd.read_csv(out)) == 1


def test_adsb_history_interval_validation(tmp_path: Path) -> None:
    out = tmp_path / "adsb.csv"
    assert (
        fetch._adsb_history(
            _adsb_history_args(
                tmp_path,
                begin="2025-01-01T00:00:00Z",
                end="2025-01-01T03:00:00Z",
                out=str(out),
            )
        )
        == 2
    )
    assert (
        fetch._adsb_history(
            _adsb_history_args(
                tmp_path,
                begin="2025-01-01T00:00:00Z",
                end="2025-01-04T00:00:00Z",
                icao24="abc123",
                out=str(out),
            )
        )
        == 2
    )


def test_adsb_history_zero_flights_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fetch, "_opensky_token", lambda: "tok")
    monkeypatch.setattr(
        fetch,
        "_opensky_request",
        lambda path, *, params=None, **kw: b"[]",
    )
    out = tmp_path / "adsb.csv"
    assert fetch._adsb_history(_adsb_history_args(tmp_path, out=str(out))) == 0
    assert "No OpenSky flights" in capsys.readouterr().out
    assert not out.exists()


def test_adsb_live_authenticated_time_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSKY_CLIENT_ID", "cid")
    monkeypatch.setenv("OPENSKY_CLIENT_SECRET", "csecret")
    args = SimpleNamespace(out="x.csv", bbox=None, time=int(time.time()) - 7200)
    assert fetch._adsb_live(args) == 2


def test_adsb_live_anonymous_notes_history(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENSKY_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENSKY_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(
        fetch,
        "_http_get",
        lambda url, *, timeout=30.0, headers=None: json.dumps({"states": []}).encode(),
    )
    out = Path("snapshot.csv")
    args = SimpleNamespace(out=str(out), bbox=None, time=None)
    assert fetch._adsb_live(args) == 2
    assert "adsb-history" in capsys.readouterr().out


def test_adsb_live_writes_anonymous_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENSKY_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENSKY_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(
        fetch,
        "_http_get",
        lambda url, *, timeout=30.0, headers=None: json.dumps(
            {"states": [_opensky_state()]}
        ).encode(),
    )
    out = tmp_path / "snapshot.csv"
    args = SimpleNamespace(out=str(out), bbox=None, time=None)
    assert fetch._adsb_live(args) == 0
    frame = pd.read_csv(out)
    assert len(frame) == 1
    assert frame.iloc[0]["icao24"] == "abc123"


def test_adsb_live_authenticated_uses_request_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENSKY_CLIENT_ID", "cid")
    monkeypatch.setenv("OPENSKY_CLIENT_SECRET", "csecret")
    captured: dict[str, object] = {}

    def fake_request(path: str, *, params: dict[str, object] | None = None, **kw: object) -> bytes:
        captured["path"] = path
        captured["params"] = params
        return json.dumps({"states": [_opensky_state()]}).encode()

    monkeypatch.setattr(fetch, "_opensky_request", fake_request)
    out = tmp_path / "snapshot.csv"
    epoch = int(time.time()) - 600
    args = SimpleNamespace(out=str(out), bbox=None, time=epoch)
    assert fetch._adsb_live(args) == 0
    assert captured["path"] == "/states/all"
    assert captured["params"] == {"time": epoch}
    assert len(pd.read_csv(out)) == 1


# --- EUMETSAT SEVIRI -------------------------------------------------------------


def _zip_bytes(member: str, payload: bytes = b"hrIT") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(member, payload)
    return buffer.getvalue()


class _FakeProduct:
    def __init__(self, pid: str, start: datetime, end: datetime, payload: bytes) -> None:
        self.pid = pid
        self.sensing_start = start
        self.sensing_end = end
        self.payload = payload

    def __str__(self) -> str:
        return self.pid

    def open(self):  # noqa: ANN201 - context manager
        @contextmanager
        def stream():
            yield io.BytesIO(self.payload)

        return stream()


class _FakeCollection:
    def __init__(self, products: list[_FakeProduct]) -> None:
        self.products = products
        self.query: dict[str, object] = {}

    def search(self, **query: object) -> list[_FakeProduct]:
        self.query = query
        return self.products


class _FakeDataStore:
    def __init__(self, collection: _FakeCollection) -> None:
        self.collection = collection
        self.collection_id: str | None = None

    def get_collection(self, collection_id: str) -> _FakeCollection:
        self.collection_id = collection_id
        return self.collection


def test_select_seviri_product_picks_closest_slot() -> None:
    requested = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    products = [
        _FakeProduct(
            "P15",
            datetime(2025, 1, 1, 0, 15, 10),
            datetime(2025, 1, 1, 0, 27, 40),
            b"",
        ),
        _FakeProduct(
            "P00",
            datetime(2025, 1, 1, 0, 0, 10),
            datetime(2025, 1, 1, 0, 12, 40),
            b"",
        ),
    ]
    assert fetch._select_seviri_product(products, requested).pid == "P00"


def test_seviri_downloads_granule_and_extracts_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    products = [
        _FakeProduct(
            "P00",
            requested + timedelta(seconds=10),
            requested + timedelta(minutes=12),
            _zip_bytes("MSG2-SEVI-MSG15-0100-NA-20250101000010.nat"),
        )
    ]
    collection = _FakeCollection(products)
    store = _FakeDataStore(collection)
    monkeypatch.setattr(fetch, "_eumdac_datastore", lambda: store)
    monkeypatch.setenv("EUMETSAT_CONSUMER_KEY", "key")
    monkeypatch.setenv("EUMETSAT_CONSUMER_SECRET", "secret")

    out = tmp_path / "seviri"
    args = SimpleNamespace(
        date="2025-01-01",
        hour=0,
        channels="108,120",
        collection="EO:EUM:DAT:MSG:HRSEVIRI-IODC",
        out=out,
    )
    assert fetch._seviri(args) == 0
    assert store.collection_id == "EO:EUM:DAT:MSG:HRSEVIRI-IODC"
    assert collection.query["set"] == "brief"
    assert collection.query["dtstart"].startswith("2024-12-31T23:45")
    assert collection.query["dtend"].startswith("2025-01-01T00:15")
    members = list(out.glob("*.nat"))
    assert [member.name for member in members] == [
        "MSG2-SEVI-MSG15-0100-NA-20250101000010.nat"
    ]
    assert not list(out.glob("*.zip"))


def test_seviri_missing_env_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EUMETSAT_CONSUMER_KEY", raising=False)
    monkeypatch.delenv("EUMETSAT_CONSUMER_SECRET", raising=False)
    monkeypatch.delenv("EUMETSAT_API_TOKEN", raising=False)
    args = SimpleNamespace(
        date="2025-01-01",
        hour=0,
        channels="108,120",
        collection=fetch.DEFAULT_SEVIRI_COLLECTION,
        out=tmp_path,
    )
    assert fetch._seviri(args) == 2


def test_seviri_invalid_date_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EUMETSAT_CONSUMER_KEY", "key")
    monkeypatch.setenv("EUMETSAT_CONSUMER_SECRET", "secret")
    args = SimpleNamespace(
        date="01-01-2025",
        hour=0,
        channels="108,120",
        collection=fetch.DEFAULT_SEVIRI_COLLECTION,
        out=tmp_path,
    )
    assert fetch._seviri(args) == 2


# --- ERA5 ------------------------------------------------------------------------


def test_era5_request_matches_era5_constants() -> None:
    from stratum.io.era5 import ERA5_PRESSURE_LEVELS_HPA

    request = fetch._era5_request("2025-01-01", 2)
    assert request["variable"] == list(fetch._ERA5_VARIABLES)
    assert request["pressure_level"] == [str(level) for level in ERA5_PRESSURE_LEVELS_HPA]
    assert request["area"] == [40.0, 60.0, 5.0, 95.0]
    assert request["data_format"] == "netcdf"
    assert request["year"] == ["2025"]
    assert request["month"] == ["01"]
    assert request["day"] == ["01", "02"]
    assert len(request["time"]) == 24
    assert request["time"][0] == "00:00"
    assert request["time"][-1] == "23:00"


def test_era5_client_honours_env_and_submits_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CDSAPI_URL", "https://cds.example.test")
    monkeypatch.setenv("CDSAPI_KEY", "cds-key")
    monkeypatch.setattr(fetch, "_probe_cds", lambda url: None)
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def retrieve(self, name: str, request: dict, target: str) -> None:
            captured["name"] = name
            captured["request"] = request
            captured["target"] = target

    monkeypatch.setattr(fetch, "_cdsapi_client", lambda **kwargs: FakeClient(**kwargs))
    out = tmp_path / "era5.nc"
    args = SimpleNamespace(date="2025-01-01", days=1, out=out)
    assert fetch._era5(args) == 0
    assert captured["init"] == {"url": "https://cds.example.test", "key": "cds-key"}
    assert captured["name"] == fetch._ERA5_DATASET
    assert captured["target"] == str(out)
    assert captured["request"] == fetch._era5_request("2025-01-01", 1)


def test_era5_missing_creds_exits_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_client(**kwargs: object) -> object:
        raise RuntimeError("no ~/.cdsapirc")

    monkeypatch.setattr(fetch, "_probe_cds", lambda url: None)
    monkeypatch.setattr(fetch, "_cdsapi_client", broken_client)
    args = SimpleNamespace(date="2025-01-01", days=1, out=tmp_path / "era5.nc")
    assert fetch._era5(args) == 2


def test_era5_unreachable_host_exits_two_without_queueing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def broken_client(**kwargs: object) -> object:
        raise AssertionError("the client must not be created")

    monkeypatch.setattr(fetch, "_probe_cds", lambda url: "connection refused")
    monkeypatch.setattr(fetch, "_cdsapi_client", broken_client)
    args = SimpleNamespace(date="2025-01-01", days=1, out=tmp_path / "era5.nc")
    assert fetch._era5(args) == 2
    assert "not reachable" in capsys.readouterr().err


def test_load_local_env_fills_missing_variables_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nSTRATUM_TEST_KEY=from_file\nSTRATUM_TEST_KEEP=from_file\n\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("STRATUM_TEST_KEY", raising=False)
    monkeypatch.setenv("STRATUM_TEST_KEEP", "from_environment")
    fetch._load_local_env(env_file)
    assert os.environ["STRATUM_TEST_KEY"] == "from_file"
    assert os.environ["STRATUM_TEST_KEEP"] == "from_environment"
    monkeypatch.delenv("STRATUM_TEST_KEY", raising=False)
    monkeypatch.delenv("STRATUM_TEST_KEEP", raising=False)


def test_load_local_env_ignores_missing_file(tmp_path: Path) -> None:
    fetch._load_local_env(tmp_path / "absent.env")


def test_adsb_history_rejects_interval_older_than_track_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(fetch, "_opensky_token", lambda: "tok")
    args = _adsb_history_args(
        tmp_path,
        begin="2025-01-01T00:10:00Z",
        end="2025-01-01T00:20:00Z",
    )
    assert fetch._adsb_history(args) == 2
    assert "30 days" in capsys.readouterr().err


def test_opensky_request_reports_track_window_on_429_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "_opensky_token", lambda: "tok")

    def raise_429(url: str, *, timeout: float = 30.0, headers: dict[str, str] | None = None) -> bytes:
        raise urllib.error.HTTPError(
            url,
            429,
            "Too many requests",
            {"X-Rate-Limit-Retry-After-Seconds": "9223372036"},
            None,
        )

    monkeypatch.setattr(fetch, "_http_get", raise_429)
    with pytest.raises(ValueError, match="30 days"):
        fetch._opensky_request("/tracks/all", params={"icao24": "abc123", "time": 1})


def test_track_records_derive_speed_and_vertical_rate() -> None:
    track = {
        "icao24": "abc123",
        "callsign": "TEST1 ",
        "path": [
            [1000, 10.0, 20.0, 1000.0, 90.0, False],
            [1060, 10.1, 20.1, 1100.0, 90.0, False],
            [1120, 10.2, 20.2, 1200.0, 90.0, False],
        ],
    }
    records = fetch._opensky_track_to_records(track)
    assert [r["time"] for r in records] == [1000, 1060, 1120]
    # Ground speed comes from the great-circle distance over the time step.
    assert records[1]["velocity"] == pytest.approx(
        fetch.haversine_km((10.0, 20.0), (10.1, 20.1)) * 1000.0 / 60.0 / fetch.MPS_TO_KT
    )
    # Vertical rate comes from the barometric altitude change per minute.
    assert records[1]["vrate_fpm"] == pytest.approx(
        (1100.0 - 1000.0) * fetch.METRES_TO_FEET / 60.0 * 60.0
    )
    assert records[2]["vrate_fpm"] == pytest.approx(
        (1200.0 - 1100.0) * fetch.METRES_TO_FEET / 60.0 * 60.0
    )
    # The first fix borrows the difference to the second fix.
    assert records[0]["vrate_fpm"] == pytest.approx(records[1]["vrate_fpm"])
