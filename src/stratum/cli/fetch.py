"""Data fetch subcommands for the STRATUM command line.

Five downloaders are provided:

``goes``
    Downloads real GOES-16 ABI Level 2 CMI NetCDF files from the public
    ``gcp-public-data-goes-16`` bucket. No authentication is required.
``adsb-live``
    Snapshots the OpenSky ``/api/states/all`` endpoint into a CSV that
    :mod:`stratum.io.adsb` can load. Anonymous when no credentials are set;
    authenticated with ``--time`` when ``OPENSKY_CLIENT_ID``/``SECRET`` are.
``adsb-history``
    Fetches OpenSky flight tracks over an interval into a loader-compatible
    CSV. Requires OpenSky credentials.
``seviri``
    Downloads EUMETSAT MSG SEVIRI Level 1.5 granules as native HRIT files.
    Requires EUMETSAT credentials.
``era5``
    Submits an ERA5 pressure-level request to the Copernicus Climate Data
    Store. Requires CDS credentials.

None of these commands writes into the repository's training data; ``goes``
caches under ``data/cache/goes16``, ``seviri`` under ``data/cache/seviri``,
and the others write exactly the file named by ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stratum.match.label import haversine_km

__all__ = ["fetch", "opensky_states_to_frame", "select_goes_object"]

_GOES_BUCKET = "gcp-public-data-goes-16"
_GOES_OBJECT_URL = f"https://storage.googleapis.com/{_GOES_BUCKET}"
_GOES_LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{_GOES_BUCKET}/o"
_OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"

#: OpenSky OAuth2 token endpoint (client-credentials grant).
_OPENSKY_TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network/"
    "protocol/openid-connect/token"
)
#: OpenSky REST API root.
_OPENSKY_API_ROOT = "https://opensky-network.org/api"

#: Refresh the cached token when it is older than this many seconds; the
#: server issues tokens valid for ~1800 s, so 25 min leaves a safety margin.
_OPENSKY_TOKEN_REFRESH_S = 25 * 60

#: Module-level token cache: ``{"token": str, "expires": monotonic float}``.
_OPENSKY_TOKEN_CACHE: dict[str, Any] = {}

#: OpenSky flight and track endpoint paths.
_OPENSKY_FLIGHTS_ALL = "/flights/all"
_OPENSKY_FLIGHTS_AIRCRAFT = "/flights/aircraft"
_OPENSKY_TRACKS = "/tracks"
_OPENSKY_TRACKS_ALL = "/tracks/all"

#: OpenSky interval limits: 2 h for all flights, 2 d for one aircraft.
_OPENSKY_MAX_INTERVAL_S = 2 * 3600
_OPENSKY_MAX_AIRCRAFT_INTERVAL_S = 2 * 24 * 3600

#: EUMETSAT MSG SEVIRI Level 1.5 collection for the Indian Ocean domain.
_EUMETSAT_SEVIRI_IODC = "EO:EUM:DAT:MSG:HRSEVIRI-IODC"
#: EUMETSAT MSG SEVIRI Level 1.5 collection at 0 degree.
_EUMETSAT_SEVIRI_MSG0 = "EO:EUM:DAT:MSG:HRSEVIRI"

#: Default SEVIRI collection; the Indian Ocean product covers the Indian
#: domain (60-95 E) that the 0-degree product clips at its eastern edge.
DEFAULT_SEVIRI_COLLECTION = _EUMETSAT_SEVIRI_IODC

#: Default cache directory for downloaded SEVIRI granules.
DEFAULT_SEVIRI_OUT = Path("data/cache/seviri")

#: Default SEVIRI channels: IR_108 and IR_120, the BTD pair used by the
#: detector in :mod:`stratum.io.meteosat`.
DEFAULT_SEVIRI_CHANNELS = "108,120"

#: Native SEVIRI channel names accepted by ``--channels``.
_SEVIRI_CHANNELS = (
    "016",
    "039",
    "062",
    "073",
    "087",
    "097",
    "108",
    "120",
    "134",
    "HRV",
)

#: ERA5 request variables. This is a superset of the six loader variables
#: (:data:`stratum.io.era5.ERA5_VARIABLES`); the extra humidity and cloud
#: ice fields are fetched for downstream analysis.
_ERA5_VARIABLES: tuple[str, ...] = (
    "fraction_of_cloud_cover",
    "relative_humidity",
    "specific_cloud_ice_water_content",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
)

#: ERA5 request domain (North, West, South, East) [deg].
_ERA5_DOMAIN = (40.0, 60.0, 5.0, 95.0)

#: CDS dataset id for pressure-level ERA5 reanalysis.
_ERA5_DATASET = "reanalysis-era5-pressure-levels"

#: Default number of days requested by ``fetch era5``.
_DEFAULT_ERA5_DAYS = 7

#: OpenSky serves flight tracks for the last 30 days only.
_OPENSKY_TRACK_MAX_AGE_DAYS = 30

#: Default cache directory for downloaded GOES-16 granules.
DEFAULT_GOES_OUT = Path("data/cache/goes16")

#: Default ABI bands: C14 (11.2 um) and C15 (12.3 um), a SEVIRI-like BTD pair.
DEFAULT_GOES_BANDS = "14,15"

#: Metres to feet conversion factor, mirrors :mod:`stratum.io.adsb`.
METRES_TO_FEET: float = 3.280839895013123

#: Metres per second to feet per minute conversion factor.
MPS_TO_FPM: float = 196.8503937007874

#: Metres per second per knot; OpenSky reports speeds in metres per second.
MPS_TO_KT: float = 0.5144444444444445

#: Matches an ABI L2 CMIPF granule; captures band, optional scan type and start.
_GOES_NAME_PATTERN = re.compile(
    r"^OR_ABI-L2-CMIPF-M6C(?P<band>\d{2})_G16"
    r"(?:_t(?P<type>[A-Za-z]))?_s(?P<start>\d{14})"
)

#: Column order of the snapshot CSV; headers resolve in :mod:`stratum.io.adsb`.
#: ``track_deg`` is not optional for that loader, so it is always written.
_OPENSKY_COLUMNS: tuple[str, ...] = (
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
)

#: OpenSky state-vector field indices (2024 schema).
_OPENSKY_ICAO24 = 0
_OPENSKY_CALLSIGN = 1
_OPENSKY_TIME_POSITION = 3
_OPENSKY_LAST_CONTACT = 4
_OPENSKY_LON = 5
_OPENSKY_LAT = 6
_OPENSKY_BARO_ALTITUDE_M = 7
_OPENSKY_ON_GROUND = 8
_OPENSKY_VELOCITY_MPS = 9
_OPENSKY_TRUE_TRACK_DEG = 10
_OPENSKY_VERTICAL_RATE_MPS = 11
_OPENSKY_GEO_ALTITUDE_M = 13
_OPENSKY_MIN_FIELDS = 14


def select_goes_object(prefix_listing: list[dict], band: int) -> str | None:
    """Return the granule to download for ``band`` from a bucket listing.

    Parameters
    ----------
    prefix_listing : list of dict
        Objects returned by the GCS JSON API for one ``YYYY/DOY/HH`` prefix.
        Only the ``name`` entry of each object is read.
    band : int
        ABI channel number, e.g. ``14`` or ``15``.

    Returns
    -------
    str or None
        Object name of the earliest scan starting at or after the hour
        boundary, or ``None`` when the listing holds no matching granule.
        Granules whose optional ``t`` scan-type field is not ``F`` are
        skipped when that field is present.
    """
    pattern = _GOES_NAME_PATTERN
    best_name: str | None = None
    best_start: str | None = None
    for item in prefix_listing:
        name = str(item.get("name", ""))
        basename = name.rsplit("/", 1)[-1]
        match = pattern.match(basename)
        if match is None or int(match.group("band")) != band:
            continue
        if match.group("type") is not None and match.group("type") != "F":
            continue
        start = match.group("start")
        if best_start is None or start < best_start:
            best_start = start
            best_name = name
    return best_name


def opensky_states_to_frame(states: list[list]) -> pd.DataFrame:
    """Convert OpenSky state vectors into a loader-compatible frame.

    Parameters
    ----------
    states : list of list
        The ``states`` array of an OpenSky ``/api/states/all`` response.
        Short or partially ``None``-filled vectors are tolerated.

    Returns
    -------
    pandas.DataFrame
        One row per state vector with the columns
        :data:`_OPENSKY_COLUMNS`. ``baro_alt_ft`` and ``geo_alt_ft`` are
        converted from metres to feet, ``vrate_fpm`` from metres per
        second to feet per minute, and ``velocity`` from metres per
        second to knots, so the values are already in the units their
        headers advertise when :func:`stratum.io.adsb.load_adsb_csv`
        reads the file.
    """

    def at(state: list, index: int) -> object:
        return state[index] if index < len(state) else None

    def number(value: object) -> float | None:
        if value is None:
            return None
        return float(value)

    records: list[dict[str, object]] = []
    for state in states:
        if len(state) < _OPENSKY_MIN_FIELDS:
            state = list(state) + [None] * (_OPENSKY_MIN_FIELDS - len(state))
        timestamp = at(state, _OPENSKY_TIME_POSITION)
        if timestamp is None:
            timestamp = at(state, _OPENSKY_LAST_CONTACT)
        icao24 = at(state, _OPENSKY_ICAO24)
        callsign = at(state, _OPENSKY_CALLSIGN)
        baro_m = number(at(state, _OPENSKY_BARO_ALTITUDE_M))
        geo_m = number(at(state, _OPENSKY_GEO_ALTITUDE_M))
        vertical_rate_mps = number(at(state, _OPENSKY_VERTICAL_RATE_MPS))
        records.append(
            {
                "icao24": "" if icao24 is None else str(icao24).strip(),
                "callsign": "" if callsign is None else str(callsign).strip(),
                "time": timestamp,
                "lat": number(at(state, _OPENSKY_LAT)),
                "lon": number(at(state, _OPENSKY_LON)),
                "baro_alt_ft": None if baro_m is None else baro_m * METRES_TO_FEET,
                "velocity": (
                    None
                    if number(at(state, _OPENSKY_VELOCITY_MPS)) is None
                    else float(at(state, _OPENSKY_VELOCITY_MPS)) / MPS_TO_KT
                ),
                "track_deg": number(at(state, _OPENSKY_TRUE_TRACK_DEG)),
                "vrate_fpm": (
                    None
                    if vertical_rate_mps is None
                    else vertical_rate_mps * MPS_TO_FPM
                ),
                "on_ground": bool(at(state, _OPENSKY_ON_GROUND)),
                "geo_alt_ft": None if geo_m is None else geo_m * METRES_TO_FEET,
            }
        )
    return pd.DataFrame(records, columns=list(_OPENSKY_COLUMNS))


def _load_local_env(path: Path = Path(".env")) -> None:
    """Load ``KEY=VALUE`` pairs from a local ``.env`` file into the environment.

    Variables already set in the environment win, so an exported value
    overrides the file. A missing or unreadable file is ignored, which
    keeps the credential-free commands working without any setup.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        # Empty values are placeholders: setting them would override the
        # defaults that libraries such as cdsapi read from the environment.
        if key and value and key not in os.environ:
            os.environ[key] = value


def _opensky_token() -> str:
    """Return a cached OpenSky access token, requesting one when needed.

    Reads ``OPENSKY_CLIENT_ID`` and ``OPENSKY_CLIENT_SECRET`` from the
    environment and exchanges them for a client-credentials token. The
    token is cached module-wide and refreshed when it is older than
    :data:`_OPENSKY_TOKEN_REFRESH_S` seconds.

    Returns
    -------
    str
        A Bearer access token.

    Raises
    ------
    ValueError
        If either environment variable is missing or the token response
        carries no ``access_token``.
    """
    client_id = os.environ.get("OPENSKY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("OPENSKY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        _load_local_env()
        client_id = os.environ.get("OPENSKY_CLIENT_ID", "").strip()
        client_secret = os.environ.get("OPENSKY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise ValueError("set OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET in .env")

    now = time.monotonic()
    token = _OPENSKY_TOKEN_CACHE.get("token")
    expires = float(_OPENSKY_TOKEN_CACHE.get("expires", 0.0))
    if isinstance(token, str) and token and expires > now:
        return token

    body = _http_post_form(
        _OPENSKY_TOKEN_URL,
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    payload = _json_loads(body, _OPENSKY_TOKEN_URL)
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise ValueError("OpenSky token response did not include an access_token")
    ttl = float(payload.get("expires_in", 1800))
    _OPENSKY_TOKEN_CACHE.clear()
    _OPENSKY_TOKEN_CACHE["token"] = token
    _OPENSKY_TOKEN_CACHE["expires"] = now + min(ttl, _OPENSKY_TOKEN_REFRESH_S)
    return token


def _opensky_request(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    _retry_on_401: bool = True,
) -> bytes:
    """GET an OpenSky API path with a Bearer token.

    A 401 clears the cached token and retries once; a 429 raises a
    ``ValueError`` that quotes the ``X-Rate-Limit-Retry-After-Seconds``
    header when the server sends it.
    """
    url = _OPENSKY_API_ROOT + path
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    headers = {"Authorization": f"Bearer {_opensky_token()}"}
    try:
        return _http_get(url, timeout=30.0, headers=headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 401 and _retry_on_401:
            _OPENSKY_TOKEN_CACHE.clear()
            return _opensky_request(path, params=params, _retry_on_401=False)
        if exc.code == 429:
            retry = exc.headers.get("X-Rate-Limit-Retry-After-Seconds")
            try:
                seconds = float(retry) if retry else 0.0
            except ValueError:
                seconds = 0.0
            if seconds >= 1_000_000_000:
                # The API sends a sentinel when the request can never be
                # served, which is what an over-age track request gets.
                raise ValueError(
                    "OpenSky refused the request (HTTP 429): flight tracks are "
                    f"served for the last {_OPENSKY_TRACK_MAX_AGE_DAYS} days "
                    "only. Older trajectories need the Trino/MinIO historical "
                    "interface."
                ) from exc
            detail = f"; retry after {seconds:.0f} s" if seconds else ""
            raise ValueError(f"OpenSky rate limited (HTTP 429){detail}") from exc
        raise


def _opensky_track_to_records(track: dict) -> list[dict[str, object]]:
    """Convert one OpenSky track payload into canonical CSV records.

    OpenSky track points are ``[time, lat, lon, baro_alt_m, track_deg,
    on_ground]`` with optional trailing velocity (m/s), vertical rate
    (m/s) and geometric altitude (m) fields; absent trailing fields are
    written as ``None``.

    The track endpoint reports neither ground speed nor vertical rate, so
    both are derived from consecutive fixes: ground speed from the
    great-circle distance over the time step, vertical rate from the
    barometric altitude change. The first fix borrows the difference to
    the second fix so that no row keeps a missing kinematic value.
    """
    callsign = track.get("callsign")
    records: list[dict[str, object]] = []
    for point in track.get("path") or []:
        if len(point) < 6:
            continue
        timestamp, lat, lon, baro_m, track_deg, on_ground = point[:6]
        velocity = point[6] if len(point) > 6 else None
        vrate = point[7] if len(point) > 7 else None
        geo_m = point[8] if len(point) > 8 else None
        records.append(
            {
                "icao24": str(track.get("icao24", "")).strip(),
                "callsign": "" if callsign is None else str(callsign).strip(),
                "time": timestamp,
                "lat": None if lat is None else float(lat),
                "lon": None if lon is None else float(lon),
                "baro_alt_ft": None if baro_m is None else float(baro_m) * METRES_TO_FEET,
                "velocity": None if velocity is None else float(velocity) / MPS_TO_KT,
                "track_deg": None if track_deg is None else float(track_deg),
                "vrate_fpm": None if vrate is None else float(vrate) * MPS_TO_FPM,
                "on_ground": bool(on_ground),
                "geo_alt_ft": None if geo_m is None else float(geo_m) * METRES_TO_FEET,
            }
        )
    return _derive_kinematics(records)


def _derive_kinematics(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Fill missing ground speed and vertical rate from consecutive fixes.

    ``velocity`` holds ground speed in knots and ``vrate_fpm`` the
    vertical rate in feet per minute, matching the canonical ADS-B
    schema. Values the source already reported are kept.
    """
    usable = [
        index
        for index, record in enumerate(records)
        if record.get("time") is not None
        and record.get("lat") is not None
        and record.get("lon") is not None
    ]
    for position, index in enumerate(usable):
        record = records[index]
        if record.get("velocity") is not None and record.get("vrate_fpm") is not None:
            continue
        if position == 0:
            other = usable[1] if len(usable) > 1 else None
            sign = -1.0
        else:
            other = usable[position - 1]
            sign = 1.0
        if other is None:
            continue
        neighbour = records[other]
        dt = float(record["time"]) - float(neighbour["time"])
        if dt == 0:
            continue
        if record.get("velocity") is None:
            distance_km = haversine_km(
                (float(neighbour["lat"]), float(neighbour["lon"])),
                (float(record["lat"]), float(record["lon"])),
            )
            metres_per_second = distance_km * 1000.0 / abs(dt)
            record["velocity"] = metres_per_second / MPS_TO_KT
        if record.get("vrate_fpm") is None:
            if record.get("baro_alt_ft") is None or neighbour.get("baro_alt_ft") is None:
                continue
            change_ft = float(record["baro_alt_ft"]) - float(neighbour["baro_alt_ft"])
            record["vrate_fpm"] = sign * change_ft / abs(dt) * 60.0
    return records


def _http_get(
    url: str, *, timeout: float = 30.0, headers: dict[str, str] | None = None
) -> bytes:
    """Fetch ``url`` with the STRATUM user agent and return the body.

    ``headers`` are merged over the default ``User-Agent`` so callers can
    attach e.g. an ``Authorization`` header without losing the user agent.
    """
    request_headers = {"User-Agent": "StratumEngine/1.0"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _http_post_form(url: str, data: dict[str, str]) -> bytes:
    """POST an ``application/x-www-form-urlencoded`` body and return the body."""
    body = urllib.parse.urlencode(data).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "User-Agent": "StratumEngine/1.0",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        return response.read()


def _list_goes_prefix(prefix: str) -> list[dict]:
    """List every object under ``prefix`` using the GCS JSON API."""
    items: list[dict] = []
    page_token: str | None = None
    for _ in range(50):  # hard page cap; one hour never needs this many
        query: dict[str, object] = {"prefix": prefix, "maxResults": 1000}
        if page_token is not None:
            query["pageToken"] = page_token
        url = f"{_GOES_LIST_URL}?{urllib.parse.urlencode(query)}"
        payload = _json_loads(_http_get(url), url)
        items.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return items


def _json_loads(body: bytes, source: str) -> dict:
    """Parse a JSON object response, raising ValueError with context."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{source} did not return valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source} did not return a JSON object")
    return payload


def _parse_doy(value: str) -> tuple[int, int]:
    """Parse ``YYYY-DDD`` or ``YYYYDDD`` into (year, day of year)."""
    text = value.strip()
    if "-" in text:
        year_text, doy_text = text.split("-", 1)
    else:
        if len(text) != 7 or not text.isdigit():
            raise ValueError(f"--doy must be YYYY-DDD or YYYYDDD, got {value!r}")
        year_text, doy_text = text[:4], text[4:]
    year, doy = int(year_text), int(doy_text)
    if not 1 <= doy <= 366:
        raise ValueError(f"day of year out of range in --doy {value!r}")
    return year, doy


def _goes(args: argparse.Namespace) -> int:
    """Download one GOES-16 ABI granule per requested band."""
    import datetime as _dt

    if args.date is not None:
        try:
            parsed = _dt.date.fromisoformat(args.date)
        except ValueError:
            print(f"--date must be YYYY-MM-DD, got {args.date!r}", file=sys.stderr)
            return 2
        year, doy = parsed.year, parsed.timetuple().tm_yday
    else:
        try:
            year, doy = _parse_doy(args.doy)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if not 0 <= args.hour <= 23:
        print(f"--hour must be in 0-23, got {args.hour}", file=sys.stderr)
        return 2

    try:
        bands = [int(part) for part in str(args.band).split(",") if part.strip() != ""]
    except ValueError:
        print(
            f"--band must be a comma-separated list, got {args.band!r}",
            file=sys.stderr,
        )
        return 2
    if not bands or any(not 1 <= band <= 16 for band in bands):
        print(f"--band entries must be in 1-16, got {args.band!r}", file=sys.stderr)
        return 2

    prefix = f"ABI-L2-CMIPF/{year:04d}/{doy:03d}/{args.hour:02d}/"
    print("GOES-16 ABI L2 CMIPF fetch")
    print(f"Bucket:  gs://{_GOES_BUCKET}/{prefix}")
    print(f"Bands:   {', '.join(f'C{band:02d}' for band in bands)}")

    try:
        listing = _list_goes_prefix(prefix)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            f"failed to list {_GOES_LIST_URL} for prefix {prefix!r}: {exc}. "
            "Check network access; the bucket is public and needs no auth.",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not listing:
        print(
            f"no objects under {prefix!r}; the hour may be outside the "
            "GOES-16 record (2017 onward).",
            file=sys.stderr,
        )
        return 2

    out_dir: Path = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for band in bands:
        name = select_goes_object(listing, band)
        if name is None:
            print(
                f"no OR_ABI-L2-CMIPF-M6C{band:02d}_G16 granule under {prefix!r}",
                file=sys.stderr,
            )
            return 2
        url = f"{_GOES_OBJECT_URL}/{urllib.parse.quote(name)}"
        destination = out_dir / name.rsplit("/", 1)[-1]
        try:
            body = _http_get(url, timeout=120.0)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"download failed for {url}: {exc}. The bucket is public; "
                "retry when the network is available.",
                file=sys.stderr,
            )
            return 2
        destination.write_bytes(body)
        downloaded.append(destination)
        size_mib = len(body) / (1024.0 * 1024.0)
        print(f"C{band:02d}: {name}")
        print(f"     -> {destination.resolve()} ({size_mib:.1f} MiB)")

    print()
    print(f"Downloaded {len(downloaded)} file(s) to {out_dir.resolve()}:")
    for path in downloaded:
        print(f"  {path.resolve()}")
    return 0


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Parse ``N,W,S,E`` into (south, west, north, east) for OpenSky."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox must be N,W,S,E, got {value!r}")
    north, west, south, east = (float(part) for part in parts)
    if not south < north or not west < east:
        raise ValueError(f"--bbox has an empty span: {value!r}")
    return south, west, north, east


def _adsb_live(args: argparse.Namespace) -> int:
    """Snapshot the OpenSky states endpoint into a CSV.

    Anonymous by default. When OpenSky credentials are present ``--time
    EPOCH`` selects a state vector at most one hour back, which is the
    authenticated window; older times are rejected in favour of
    ``fetch adsb-history``.
    """
    query: dict[str, object] = {}
    authenticated = bool(
        os.environ.get("OPENSKY_CLIENT_ID", "").strip()
        and os.environ.get("OPENSKY_CLIENT_SECRET", "").strip()
    )
    if args.time is not None:
        if not authenticated:
            print(
                "--time needs OpenSky credentials; use 'stratum fetch "
                "adsb-history' for authenticated history",
                file=sys.stderr,
            )
            return 2
        if int(time.time()) - int(args.time) > 3600:
            print(
                "OpenSky authenticated --time is limited to 1 hour back; use "
                "'stratum fetch adsb-history' for older intervals",
                file=sys.stderr,
            )
            return 2
        query["time"] = int(args.time)

    if args.bbox is not None:
        try:
            south, west, north, east = _parse_bbox(args.bbox)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        query.update(
            {
                "lamin": south,
                "lomin": west,
                "lamax": north,
                "lomax": east,
            }
        )

    url = _OPENSKY_STATES_URL
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    if authenticated:
        try:
            body = _opensky_request("/states/all", params=query)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"OpenSky request failed for {url}: {exc}. The endpoint is "
                "rate limited; retry later or narrow --bbox.",
                file=sys.stderr,
            )
            return 2
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    else:
        print(
            "Note: OPENSKY_CLIENT_ID/OPENSKY_CLIENT_SECRET unset; anonymous "
            "snapshot only. Use 'stratum fetch adsb-history' for authenticated "
            "history."
        )
        try:
            body = _http_get(url)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"OpenSky request failed for {url}: {exc}. The anonymous "
                "endpoint is rate limited; retry later or narrow --bbox.",
                file=sys.stderr,
            )
            return 2

    try:
        payload = _json_loads(body, url)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    states = payload.get("states")
    if not states:
        print(
            f"OpenSky response from {url} has no 'states' array "
            f"(keys: {sorted(payload)}). Nothing written.",
            file=sys.stderr,
        )
        return 2

    frame = opensky_states_to_frame(states)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    print("OpenSky live snapshot")
    print(f"Source: {url}")
    print(f"States: {len(frame)}")
    print(f"CSV:    {out_path.resolve()}")
    print()
    print(
        "WARNING: this is a real ADS-B snapshot, but it is NOT time-aligned "
        "with historical ERA5 fields. Do not pair it with ERA5 reanalysis "
        "labels without an explicit time match."
    )
    return 0


def _parse_iso8601(value: str) -> datetime:
    """Parse an ISO 8601 timestamp into an aware UTC ``datetime``."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"expected ISO 8601 timestamp, got {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fetch_opensky_track(icao24: str, time_epoch: int) -> dict | None:
    """Fetch one aircraft track, falling back to ``/tracks/all`` on 404."""
    for path in (_OPENSKY_TRACKS, _OPENSKY_TRACKS_ALL):
        try:
            body = _opensky_request(
                path, params={"icao24": icao24, "time": time_epoch}
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and path == _OPENSKY_TRACKS:
                continue
            raise
        payload = _json_loads(body, path)
        if isinstance(payload, dict):
            return payload
    return None


def _adsb_history(args: argparse.Namespace) -> int:
    """Fetch OpenSky flight tracks over an interval into a canonical CSV.

    The interval is limited to 2 hours without ``--icao24`` and to 2 days
    with it, matching the OpenSky flight endpoints.
    """
    try:
        begin = _parse_iso8601(args.begin)
        end = _parse_iso8601(args.end)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if end <= begin:
        print("--end must be after --begin", file=sys.stderr)
        return 2

    icao24 = args.icao24.strip().lower() if args.icao24 else None
    span = (end - begin).total_seconds()
    limit = _OPENSKY_MAX_AIRCRAFT_INTERVAL_S if icao24 else _OPENSKY_MAX_INTERVAL_S
    if span > limit:
        hint = (
            "with --icao24 the limit is 2 days"
            if icao24
            else "without --icao24 the limit is 2 hours; add --icao24 to widen it"
        )
        print(
            f"--begin/--end span {span / 3600:.2f} h exceeds the OpenSky "
            f"limit ({hint})",
            file=sys.stderr,
        )
        return 2

    age_days = (datetime.now(timezone.utc) - end).total_seconds() / 86400.0
    if age_days > _OPENSKY_TRACK_MAX_AGE_DAYS:
        print(
            f"the interval ends {age_days:.1f} days ago; OpenSky serves "
            f"flight tracks for the last {_OPENSKY_TRACK_MAX_AGE_DAYS} days "
            "only. Older trajectories need the Trino/MinIO historical "
            "interface.",
            file=sys.stderr,
        )
        return 2

    out_path = Path(args.out)
    try:
        _opensky_token()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    query: dict[str, object] = {"begin": int(begin.timestamp()), "end": int(end.timestamp())}
    if icao24:
        query["icao24"] = icao24
    flights_path = _OPENSKY_FLIGHTS_AIRCRAFT if icao24 else _OPENSKY_FLIGHTS_ALL
    try:
        body = _opensky_request(flights_path, params=query)
        flights = json.loads(body.decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        print(f"OpenSky flights request failed: {exc}", file=sys.stderr)
        return 2
    if not isinstance(flights, list):
        print("OpenSky flights response was not a JSON array", file=sys.stderr)
        return 2

    max_flights = max(int(args.max_flights), 0)
    flights = flights[:max_flights]
    if not flights:
        print(
            f"No OpenSky flights between {begin.isoformat()} and "
            f"{end.isoformat()}; nothing written to {out_path}."
        )
        return 0

    records: list[dict[str, object]] = []
    fetched = 0
    for flight in flights:
        flight_icao = str(flight.get("icao24", "")).strip().lower()
        start = flight.get("firstSeen")
        if not flight_icao or start is None:
            continue
        try:
            track = _fetch_opensky_track(flight_icao, int(start))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"OpenSky track request failed for {flight_icao}: {exc}", file=sys.stderr)
            return 2
        if track is None:
            continue
        records.extend(_opensky_track_to_records(track))
        fetched += 1

    frame = pd.DataFrame(records, columns=list(_OPENSKY_COLUMNS))
    if not frame.empty:
        frame = frame[~frame["on_ground"].astype(bool)]
        frame = frame.sort_values("time", kind="mergesort").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    print("OpenSky flight history")
    print(f"Interval: {begin.isoformat()} .. {end.isoformat()}")
    print(f"Flights:  {fetched} of {len(flights)} requested")
    print(f"Points:   {len(frame)} (on_ground rows dropped)")
    print(f"CSV:      {out_path.resolve()}")
    return 0


# EUMETSAT probe (2026-10-20): the Data Store browse/search endpoints are
# public, but downloads are not. The working credential path is
# EUMETSAT_CONSUMER_KEY/EUMETSAT_CONSUMER_SECRET exchanged by eumdac for an
# OAuth2 token (POST https://api.eumetsat.int/token with HTTP Basic auth).
# EUMETSAT_API_TOKEN is not a Data Store Bearer token: it returns HTTP 401 on
# the browse, search and download endpoints, so the bearer fallback below is
# kept for spec compliance but is not expected to authenticate. Products are
# application/zip archives of native HRIT ``.nat`` files, and the probe
# account returned HTTP 403 "GeneralLicense required" until the EUMETSAT Data
# Store licence is accepted.
def _resolve_eumetsat_auth() -> tuple[str, str]:
    """Return ``("consumer", "")`` or ``("bearer", token)`` from the environment."""
    key = os.environ.get("EUMETSAT_CONSUMER_KEY", "").strip()
    secret = os.environ.get("EUMETSAT_CONSUMER_SECRET", "").strip()
    if not (key and secret):
        _load_local_env()
        key = os.environ.get("EUMETSAT_CONSUMER_KEY", "").strip()
        secret = os.environ.get("EUMETSAT_CONSUMER_SECRET", "").strip()
    if key and secret:
        return "consumer", ""
    token = os.environ.get("EUMETSAT_API_TOKEN", "").strip()
    if not token:
        _load_local_env()
        token = os.environ.get("EUMETSAT_API_TOKEN", "").strip()
    if token:
        return "bearer", token
    raise ValueError(
        "set EUMETSAT_CONSUMER_KEY and EUMETSAT_CONSUMER_SECRET (or "
        "EUMETSAT_API_TOKEN) in .env"
    )


class _BearerToken:
    """Minimal eumdac token backed by a pre-issued Bearer string."""

    def __init__(self, token: str) -> None:
        from eumdac.token import URLs

        self._token = token
        self.urls = URLs()

    @property
    def access_token(self) -> str:
        return self._token

    @property
    def auth(self):  # pragma: no cover - thin wrapper around eumdac
        from eumdac.token import HTTPBearerAuth

        return HTTPBearerAuth(self._token)


def _eumdac_datastore():
    """Build an eumdac :class:`DataStore` from the environment credentials."""
    mode, token = _resolve_eumetsat_auth()
    from eumdac import DataStore

    if mode == "consumer":
        from eumdac.token import AccessToken

        credentials = (
            os.environ["EUMETSAT_CONSUMER_KEY"],
            os.environ["EUMETSAT_CONSUMER_SECRET"],
        )
        return DataStore(AccessToken(credentials))
    return DataStore(_BearerToken(token))


def _select_seviri_product(products, requested: datetime):
    """Pick the granule whose slot is closest to ``requested``.

    A granule that covers ``requested`` (``sensing_start <= requested <
    sensing_end``) wins; otherwise the closest ``sensing_start`` does.
    """
    best = None
    best_key = None
    for product in products:
        start = product.sensing_start
        end = product.sensing_end
        start_utc = (
            start.replace(tzinfo=timezone.utc)
            if start.tzinfo is None
            else start.astimezone(timezone.utc)
        )
        end_utc = (
            end.replace(tzinfo=timezone.utc)
            if end.tzinfo is None
            else end.astimezone(timezone.utc)
        )
        covers = start_utc <= requested < end_utc
        distance = abs((start_utc - requested).total_seconds())
        key = (0 if covers else 1, distance)
        if best_key is None or key < best_key:
            best_key = key
            best = product
    return best


def _download_seviri_product(product, out_dir: Path) -> list[Path]:
    """Stream a product zip into ``out_dir`` and extract its members."""
    archive = out_dir / f"{product}.zip"
    with archive.open("wb") as handle:
        with product.open() as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            target = out_dir / Path(name).name
            target.write_bytes(bundle.read(name))
            extracted.append(target)
    archive.unlink()
    return extracted


def _seviri(args: argparse.Namespace) -> int:
    """Download the MSG SEVIRI granule nearest a requested UTC time."""
    try:
        day = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"--date must be YYYY-MM-DD, got {args.date!r}", file=sys.stderr)
        return 2
    if not 0 <= args.hour <= 23:
        print(f"--hour must be in 0-23, got {args.hour}", file=sys.stderr)
        return 2
    channels = [part.strip() for part in str(args.channels).split(",") if part.strip()]
    if not channels or any(channel not in _SEVIRI_CHANNELS for channel in channels):
        print(
            f"--channels entries must be SEVIRI channel names, got {args.channels!r}",
            file=sys.stderr,
        )
        return 2

    try:
        mode, _ = _resolve_eumetsat_auth()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    requested = day.replace(hour=args.hour, minute=0, second=0, microsecond=0)
    window_start = requested - timedelta(minutes=15)
    window_end = requested + timedelta(minutes=15)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("EUMETSAT MSG SEVIRI Level 1.5 fetch")
    print(f"Collection: {args.collection}")
    print(f"Auth:       {mode}")
    print(
        f"Time:       {requested.isoformat()} "
        f"(search {window_start.isoformat()} .. {window_end.isoformat()})"
    )
    print(f"Channels:   {', '.join(channels)}")
    print(f"Out:        {out_dir.resolve()}")

    try:
        datastore = _eumdac_datastore()
        collection = datastore.get_collection(args.collection)
        results = collection.search(
            set="brief",
            dtstart=window_start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            dtend=window_end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        )
        products = list(results)
    except Exception as exc:  # noqa: BLE001 - network/auth/collection errors
        print(
            f"EUMETSAT search failed for {args.collection}: {exc}. Check "
            "credentials, the collection id, and that the EUMETSAT Data "
            "Store licence is accepted.",
            file=sys.stderr,
        )
        return 2

    if not products:
        print(
            f"no SEVIRI granule near {requested.isoformat()} in {args.collection}",
            file=sys.stderr,
        )
        return 2

    product = _select_seviri_product(products, requested)
    try:
        members = _download_seviri_product(product, out_dir)
    except Exception as exc:  # noqa: BLE001 - network/auth/download errors
        print(
            f"EUMETSAT download failed for {product}: {exc}. The product is a "
            "zip of native HRIT files; check credentials and licence "
            "acceptance.",
            file=sys.stderr,
        )
        return 2

    print(f"Granule:    {product}")
    print(
        f"Sensing:    {product.sensing_start.isoformat()} .. "
        f"{product.sensing_end.isoformat()}"
    )
    print()
    print(f"Downloaded {len(members)} file(s) to {out_dir.resolve()}:")
    for path in members:
        size = path.stat().st_size
        print(f"  {path.resolve()} ({size / (1024.0 * 1024.0):.1f} MiB)")
    print()
    print(
        "NOTE: SEVIRI granules are native HRIT (.nat) archives, not NetCDF. "
        "stratum detect-scene expects NetCDF IR_108/IR_120 arrays; convert "
        "with a HRIT reader (e.g. satpy) before use."
    )
    return 0


def _era5_request(date_text: str, days: int) -> dict[str, Any]:
    """Build the CDS request dict.

    Variables and domain are the fetch specification; the pressure levels
    are imported from :mod:`stratum.io.era5` so the request stays in sync
    with the loader.
    """
    from stratum.io.era5 import ERA5_PRESSURE_LEVELS_HPA

    try:
        start = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(f"--date must be YYYY-MM-DD, got {date_text!r}") from exc
    if days < 1:
        raise ValueError(f"--days must be >= 1, got {days}")
    north, west, south, east = _ERA5_DOMAIN
    return {
        "product_type": "reanalysis",
        "variable": list(_ERA5_VARIABLES),
        "pressure_level": [str(level) for level in ERA5_PRESSURE_LEVELS_HPA],
        "year": [start.strftime("%Y")],
        "month": [start.strftime("%m")],
        "day": [
            (start + timedelta(days=offset)).strftime("%d") for offset in range(days)
        ],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [north, west, south, east],
    }


def _cdsapi_client(**kwargs):
    """Return a cdsapi client honouring ``CDSAPI_URL``/``CDSAPI_KEY``."""
    import cdsapi

    return cdsapi.Client(**kwargs)


def _probe_cds(url: str) -> str | None:
    """Return ``None`` when the Climate Data Store answers, else the reason.

    The CDS client retries a failing host hundreds of times, so the fetch
    command probes the endpoint first and fails fast with a readable
    message instead of hanging.
    """
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            response.read(1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return str(exc)
    return None


def _era5(args: argparse.Namespace) -> int:
    """Submit an ERA5 pressure-level request to the Climate Data Store."""
    try:
        request = _era5_request(args.date, int(args.days))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    client_kwargs: dict[str, str] = {}
    url = os.environ.get("CDSAPI_URL", "").strip()
    key = os.environ.get("CDSAPI_KEY", "").strip()
    if not (url or key):
        _load_local_env()
        url = os.environ.get("CDSAPI_URL", "").strip()
        key = os.environ.get("CDSAPI_KEY", "").strip()
    if url:
        client_kwargs["url"] = url
    if key:
        client_kwargs["key"] = key

    # The CDS client retries a failing host hundreds of times, so probe the
    # endpoint first and fail fast with a readable message.
    probe_url = url or "https://cds.climate.copernicus.eu/api"
    problem = _probe_cds(probe_url)
    if problem is not None:
        print(
            f"the Climate Data Store is not reachable at {probe_url}: {problem}. "
            "Retry when the host answers; no request was queued.",
            file=sys.stderr,
        )
        return 2

    try:
        client = _cdsapi_client(**client_kwargs)
    except Exception as exc:  # noqa: BLE001 - missing ~/.cdsapirc
        print(
            f"could not create a CDS client: {exc}. Set CDSAPI_URL/CDSAPI_KEY "
            "or configure ~/.cdsapirc.",
            file=sys.stderr,
        )
        return 2

    out_path = (
        Path(args.out)
        if args.out is not None
        else Path(f"data/cache/era5/era5_{args.date}.nc")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    north, west, south, east = _ERA5_DOMAIN
    print("ERA5 pressure-level request")
    print(f"Days:      {args.days}")
    print(f"Levels:    {', '.join(request['pressure_level'])}")
    print(f"Variables: {', '.join(request['variable'])}")
    print(f"Domain:    {north}N {west}W {south}N {east}E")
    print(f"Output:    {out_path.resolve()}")

    try:
        client.retrieve(_ERA5_DATASET, request, str(out_path))
    except Exception as exc:  # noqa: BLE001 - network/auth/queue errors
        print(f"CDS retrieve failed: {exc}", file=sys.stderr)
        return 2
    return 0


def fetch(argv: list[str]) -> int:
    """Download external data: GOES, OpenSky, SEVIRI, or ERA5.

    Parameters
    ----------
    argv : list of str
        Arguments after ``fetch``; the first value selects the subcommand:
        ``goes``, ``adsb-live``, ``adsb-history``, ``seviri``, or ``era5``.

    Returns
    -------
    int
        0 on success, 2 on invalid arguments, network failure, or a
        missing/empty upstream response.
    """
    parser = argparse.ArgumentParser(
        prog="stratum fetch",
        description="Download external STRATUM data sets (GOES needs none).",
    )
    subparsers = parser.add_subparsers(dest="source", metavar="SOURCE")

    goes = subparsers.add_parser(
        "goes",
        help="download GOES-16 ABI L2 CMI NetCDF granules from the public bucket",
        description=(
            "Download real GOES-16 ABI L2 CMIPF granules from "
            "gs://gcp-public-data-goes-16 (public, no authentication)."
        ),
    )
    date_group = goes.add_mutually_exclusive_group()
    date_group.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="UTC date of the granule",
    )
    date_group.add_argument(
        "--doy",
        default=None,
        metavar="YYYY-DDD",
        help="UTC year and day of year instead of --date",
    )
    goes.add_argument(
        "--hour",
        type=int,
        default=0,
        help="UTC hour of the granule, 0-23 (default 0)",
    )
    goes.add_argument(
        "--band",
        default=DEFAULT_GOES_BANDS,
        help=(
            "comma-separated ABI bands; default "
            f"{DEFAULT_GOES_BANDS} (C14 11.2 um and C15 12.3 um, the "
            "SEVIRI-like BTD pair)"
        ),
    )
    goes.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_GOES_OUT,
        metavar="DIR",
        help=f"output directory (default {DEFAULT_GOES_OUT})",
    )
    goes.set_defaults(handler=_goes)

    adsb = subparsers.add_parser(
        "adsb-live",
        help="snapshot the anonymous OpenSky /api/states/all endpoint",
        description=(
            "Fetch a live OpenSky state snapshot into a CSV that "
            "stratum.io.adsb can load. The snapshot is real but is not "
            "aligned with historical ERA5 fields."
        ),
    )
    adsb.add_argument(
        "--out",
        required=True,
        metavar="FILE",
        help="CSV file to write",
    )
    adsb.add_argument(
        "--bbox",
        default=None,
        metavar="N,W,S,E",
        help="optional latitude/longitude box",
    )
    adsb.add_argument(
        "--time",
        type=int,
        default=None,
        metavar="EPOCH",
        help="authenticated state-vector time, at most 1 h back",
    )
    adsb.set_defaults(handler=_adsb_live)

    history = subparsers.add_parser(
        "adsb-history",
        help="fetch OpenSky flight tracks over an interval",
        description=(
            "Fetch OpenSky flight tracks between --begin and --end into a CSV "
            "that stratum.io.adsb can load. Requires OPENSKY_CLIENT_ID and "
            "OPENSKY_CLIENT_SECRET. Interval limit: 2 h without --icao24, "
            "2 d with."
        ),
    )
    history.add_argument(
        "--begin", required=True, metavar="ISO8601", help="start time (UTC)"
    )
    history.add_argument(
        "--end", required=True, metavar="ISO8601", help="end time (UTC)"
    )
    history.add_argument(
        "--icao24",
        default=None,
        metavar="HEX",
        help="limit to one aircraft (lowercase)",
    )
    history.add_argument(
        "--max-flights",
        type=int,
        default=25,
        metavar="N",
        help="cap on flights (default 25)",
    )
    history.add_argument(
        "--out", required=True, metavar="FILE", help="CSV file to write"
    )
    history.set_defaults(handler=_adsb_history)

    seviri = subparsers.add_parser(
        "seviri",
        help="download a MSG SEVIRI Level 1.5 granule",
        description=(
            "Download the EUMETSAT MSG SEVIRI Level 1.5 granule nearest a UTC "
            "time as native HRIT files. Requires EUMETSAT_CONSUMER_KEY and "
            "EUMETSAT_CONSUMER_SECRET."
        ),
    )
    seviri.add_argument(
        "--date", required=True, metavar="YYYY-MM-DD", help="UTC date"
    )
    seviri.add_argument(
        "--hour", type=int, default=0, help="UTC hour, 0-23 (default 0)"
    )
    seviri.add_argument(
        "--channels",
        default=DEFAULT_SEVIRI_CHANNELS,
        help=(
            "comma-separated SEVIRI channels (informational; the granule is "
            f"full-resolution, default {DEFAULT_SEVIRI_CHANNELS})"
        ),
    )
    seviri.add_argument(
        "--collection",
        default=DEFAULT_SEVIRI_COLLECTION,
        metavar="ID",
        help="EUMETSAT collection id (default "
        f"{DEFAULT_SEVIRI_COLLECTION})",
    )
    seviri.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SEVIRI_OUT,
        metavar="DIR",
        help=f"output directory (default {DEFAULT_SEVIRI_OUT})",
    )
    seviri.set_defaults(handler=_seviri)

    era5 = subparsers.add_parser(
        "era5",
        help="request ERA5 pressure-level data from the Climate Data Store",
        description=(
            "Submit an ERA5 pressure-level NetCDF request to CDS. Requires "
            "CDSAPI_URL/CDSAPI_KEY or ~/.cdsapirc."
        ),
    )
    era5.add_argument(
        "--date", required=True, metavar="YYYY-MM-DD", help="first UTC day"
    )
    era5.add_argument(
        "--days",
        type=int,
        default=_DEFAULT_ERA5_DAYS,
        help=f"number of days (default {_DEFAULT_ERA5_DAYS})",
    )
    era5.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="NetCDF output file (default data/cache/era5/era5_<date>.nc)",
    )
    era5.set_defaults(handler=_era5)

    args = parser.parse_args(argv)
    if args.source is None:
        parser.print_help()
        return 2

    if args.source == "goes" and args.date is None and args.doy is None:
        print("goes needs --date YYYY-MM-DD or --doy YYYY-DDD", file=sys.stderr)
        return 2

    return int(args.handler(args))
