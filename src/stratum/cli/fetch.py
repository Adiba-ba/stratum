"""Data fetch subcommands for the STRATUM command line.

Two downloaders are provided:

``goes``
    Downloads real GOES-16 ABI Level 2 CMI NetCDF files from the public
    ``gcp-public-data-goes-16`` bucket. No authentication is required.
``adsb-live``
    Snapshots the anonymous OpenSky ``/api/states/all`` endpoint into a CSV
    that :mod:`stratum.io.adsb` can load.

Neither command writes into the repository's training data; ``goes``
caches under ``data/cache/goes16`` by default, and ``adsb-live`` writes
exactly the file named by ``--out``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

__all__ = ["fetch", "opensky_states_to_frame", "select_goes_object"]

_GOES_BUCKET = "gcp-public-data-goes-16"
_GOES_OBJECT_URL = f"https://storage.googleapis.com/{_GOES_BUCKET}"
_GOES_LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{_GOES_BUCKET}/o"
_OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"

#: Default cache directory for downloaded GOES-16 granules.
DEFAULT_GOES_OUT = Path("data/cache/goes16")

#: Default ABI bands: C14 (11.2 um) and C15 (12.3 um), a SEVIRI-like BTD pair.
DEFAULT_GOES_BANDS = "14,15"

#: Metres to feet conversion factor, mirrors :mod:`stratum.io.adsb`.
METRES_TO_FEET: float = 3.280839895013123

#: Metres per second to feet per minute conversion factor.
MPS_TO_FPM: float = 196.8503937007874

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
        converted from metres to feet, and ``vrate_fpm`` from metres per
        second to feet per minute, so the values are already in the units
        their headers advertise when :func:`stratum.io.adsb.load_adsb_csv`
        reads the file. ``velocity`` is passed through in OpenSky's native
        metres per second because the loader maps it to ``gs_kt`` unchanged.
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
                "velocity": number(at(state, _OPENSKY_VELOCITY_MPS)),
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


def _http_get(url: str, *, timeout: float = 30.0) -> bytes:
    """Fetch ``url`` with the STRATUM user agent and return the body."""
    request = urllib.request.Request(url, headers={"User-Agent": "StratumEngine/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
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
    """Snapshot the anonymous OpenSky states endpoint into a CSV."""
    query: dict[str, object] = {}
    if args.bbox is not None:
        try:
            south, west, north, east = _parse_bbox(args.bbox)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        query = {
            "lamin": south,
            "lomin": west,
            "lamax": north,
            "lomax": east,
        }

    url = _OPENSKY_STATES_URL
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    try:
        payload = _json_loads(_http_get(url), url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(
            f"OpenSky request failed for {url}: {exc}. The anonymous endpoint "
            "is rate limited; retry later or narrow --bbox.",
            file=sys.stderr,
        )
        return 2
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


def fetch(argv: list[str]) -> int:
    """Download external data: GOES-16 granules or an OpenSky snapshot.

    Parameters
    ----------
    argv : list of str
        Arguments after ``fetch``; the first value selects the subcommand,
        ``goes`` or ``adsb-live``.

    Returns
    -------
    int
        0 on success, 2 on invalid arguments, network failure, or a
        missing/empty upstream response.
    """
    parser = argparse.ArgumentParser(
        prog="stratum fetch",
        description="Download external STRATUM data sets (no credentials).",
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
    adsb.set_defaults(handler=_adsb_live)

    args = parser.parse_args(argv)
    if args.source is None:
        parser.print_help()
        return 2

    if args.source == "goes" and args.date is None and args.doy is None:
        print("goes needs --date YYYY-MM-DD or --doy YYYY-DDD", file=sys.stderr)
        return 2

    return int(args.handler(args))
