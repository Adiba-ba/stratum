"""Live weather check subcommand for the STRATUM command line.

Samples the Open-Meteo hourly pressure-level forecast along the built-in
RJAA (Narita) to RJBB (Kansai) track, converts water relative humidity to
ice relative humidity, and runs the exact Schmidt-Appleman criterion at
FL260 for each waypoint. This is a real forecast query, but it is a
single-hour diagnostic, not a validated prediction of the operational
product.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from stratum.physics.diagnostics import get_aircraft_params
from stratum.physics.sac import evaluate_sac
from stratum.physics.thermo import e_i, e_w

__all__ = ["check_live"]

#: RJAA (Narita) -> RJBB (Kansai) great-circle track used by the check.
WAYPOINTS = [
    (35.76, 140.38), (35.63, 139.81), (35.50, 139.24),
    (35.36, 138.67), (35.22, 138.10), (35.07, 137.53),
    (34.92, 136.96), (34.76, 136.39), (34.60, 135.82),
    (34.44, 135.24),
]

#: Route identity of the built-in track; no other pair has geometry here.
DEFAULT_DEP = "RJAA"
DEFAULT_ARR = "RJBB"
DEFAULT_HOUR = 3  # 12:00 JST is 03:00 UTC

P_FL260 = 36000.0  # ~FL260 in Pa
AIRCRAFT = get_aircraft_params("A320")

_OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
    "&hourly=temperature_400hPa,temperature_300hPa,"
    "relative_humidity_400hPa,relative_humidity_300hPa"
)


def _fetch_hourly(lat: float, lon: float, date: str | None) -> dict:
    """Fetch the hourly pressure-level series for one waypoint."""
    url = _OPEN_METEO_URL.format(lat=lat, lon=lon)
    if date is None:
        url += "&forecast_days=1"
    else:
        url += f"&start_date={date}&end_date={date}"
    request = urllib.request.Request(url, headers={"User-Agent": "StratumEngine/1.0"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def check_live(argv: list[str]) -> int:
    """Sample Open-Meteo along RJAA-RJBB and evaluate the SAC at FL260.

    Parameters
    ----------
    argv : list of str
        Arguments after ``check-live``: ``--dep``, ``--arr``, ``--hour`` and
        ``--date``. Only the built-in ``RJAA`` to ``RJBB`` track has
        geometry, so any other pair is rejected.

    Returns
    -------
    int
        0 on success, 2 on invalid arguments or a network/response failure.
    """
    parser = argparse.ArgumentParser(
        prog="stratum check-live",
        description=(
            "Query Open-Meteo live weather along the RJAA-RJBB track and "
            "evaluate the Schmidt-Appleman criterion at FL260."
        ),
    )
    parser.add_argument(
        "--dep", default=DEFAULT_DEP, help="departure ICAO (default RJAA)"
    )
    parser.add_argument(
        "--arr", default=DEFAULT_ARR, help="arrival ICAO (default RJBB)"
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=DEFAULT_HOUR,
        help=f"UTC hour to sample, 0-23 (default {DEFAULT_HOUR})",
    )
    parser.add_argument(
        "--date",
        default=None,
        metavar="YYYY-MM-DD",
        help="UTC date to sample (default: today's forecast window)",
    )
    args = parser.parse_args(argv)

    dep = str(args.dep).strip().upper()
    arr = str(args.arr).strip().upper()
    if (dep, arr) != (DEFAULT_DEP, DEFAULT_ARR):
        print(
            f"check-live only ships geometry for the {DEFAULT_DEP} -> "
            f"{DEFAULT_ARR} track; got {dep} -> {arr}. Add the route "
            "waypoints to stratum.cli.live to support another pair.",
            file=sys.stderr,
        )
        return 2
    if not 0 <= args.hour <= 23:
        print(f"--hour must be in 0-23, got {args.hour}", file=sys.stderr)
        return 2

    print()
    print("=" * 90)
    print("PROJECT STRATUM - LIVE FLIGHT CONTRAIL PREDICTION (REAL WEATHER DATA)")
    print(
        f"Flight: GK215 | Route: {dep} -> {arr} | Level: FL260 (~360 hPa) | "
        "Aircraft: A320"
    )
    print("=" * 90)
    header = (
        f"{'#':>2} | {'Lat':>6} | {'Lon':>6} | {'Temp [C]':>8} | "
        f"{'T_crit [C]':>10} | {'RHi':>6} | {'Forms?':>7} | "
        f"{'Persists?':>9} | Recommendation"
    )
    print(header)
    print("-" * 90)

    for idx, (lat, lon) in enumerate(WAYPOINTS, 1):
        try:
            data = _fetch_hourly(lat, lon, args.date)
            hourly = data["hourly"]
            t_400 = hourly["temperature_400hPa"][args.hour]
            t_300 = hourly["temperature_300hPa"][args.hour]
            rh_400 = hourly["relative_humidity_400hPa"][args.hour]
            rh_300 = hourly["relative_humidity_300hPa"][args.hour]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"Open-Meteo request failed at waypoint {idx} "
                f"({lat:.2f}, {lon:.2f}): {exc}. Check network access and retry.",
                file=sys.stderr,
            )
            return 2
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            print(
                f"Open-Meteo returned an unusable response at waypoint {idx} "
                f"({lat:.2f}, {lon:.2f}): {exc!r}. Retry later or pick a "
                "different --date/--hour within the forecast window.",
                file=sys.stderr,
            )
            return 2

        # Linear interpolation to 360 hPa (40% weight to 300 hPa, 60% to 400 hPa)
        temp_c = t_400 + 0.4 * (t_300 - t_400)
        rh_water = (rh_400 + 0.4 * (rh_300 - rh_400)) / 100.0
        temp_k = temp_c + 273.15

        # Convert water relative humidity to ice relative humidity (RHi)
        rhi = rh_water * e_w(temp_k) / e_i(temp_k)

        # Evaluate exact Schmidt-Appleman physics
        sac = evaluate_sac(
            p_Pa=P_FL260,
            T_K=temp_k,
            rhi=rhi,
            eta=AIRCRAFT["eta"],
            q_fuel=AIRCRAFT["q_fuel"],
        )

        action = (
            "ABSTAIN (Clear sky)"
            if not sac["forms"]
            else ("MONITOR" if not sac["persists"] else "DESCENT")
        )
        print(
            f"{idx:>2} | {lat:>6.2f} | {lon:>6.2f} | {temp_c:>8.1f} | "
            f"{sac['TLC_C']:>10.1f} | {rhi * 100.0:>5.0f}% | "
            f"{str(sac['forms']):>7} | {str(sac['persists']):>9} | {action}"
        )

    print("-" * 90)
    print("Data Source: Open-Meteo Live GFS/ECMWF atmospheric pressure-level feed.")
    print()
    return 0
