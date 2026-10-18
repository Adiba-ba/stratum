import json
import math
import urllib.request
from datetime import datetime, timezone
from stratum.physics.sac import evaluate_sac
from stratum.physics.diagnostics import get_aircraft_params
from stratum.physics.thermo import e_w, e_i

# RJAA (Narita) -> RJBB (Kansai)
WAYPOINTS = [
    (35.76, 140.38), (35.63, 139.81), (35.50, 139.24),
    (35.36, 138.67), (35.22, 138.10), (35.07, 137.53),
    (34.92, 136.96), (34.76, 136.39), (34.60, 135.82),
    (34.44, 135.24)
]

P_FL260 = 36000.0  # ~FL260 in Pa
AIRCRAFT = get_aircraft_params("A320")
UTC_HOUR = 3  # 12:00 JST is 03:00 UTC

print("\n" + "=" * 90)
print("PROJECT STRATUM — LIVE FLIGHT CONTRAIL PREDICTION (REAL WEATHER DATA)")
print("Flight: GK215 | Route: RJAA -> RJBB | Level: FL260 (~360 hPa) | Aircraft: A320")
print("=" * 90)
header = f"{'#':>2} | {'Lat':>6} | {'Lon':>6} | {'Temp [C]':>8} | {'T_crit [C]':>10} | {'RHi':>6} | {'Forms?':>7} | {'Persists?':>9} | Recommendation"
print(header)
print("-" * 90)

for idx, (lat, lon) in enumerate(WAYPOINTS, 1):
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&hourly=temperature_400hPa,temperature_300hPa,relative_humidity_400hPa,relative_humidity_300hPa"
        "&forecast_days=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "StratumEngine/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    hourly = data["hourly"]
    t_400 = hourly["temperature_400hPa"][UTC_HOUR]
    t_300 = hourly["temperature_300hPa"][UTC_HOUR]
    rh_400 = hourly["relative_humidity_400hPa"][UTC_HOUR]
    rh_300 = hourly["relative_humidity_300hPa"][UTC_HOUR]

    # Linear interpolation to 360 hPa (40% weight to 300 hPa, 60% weight to 400 hPa)
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
        q_fuel=AIRCRAFT["q_fuel"]
    )

    action = "ABSTAIN (Clear sky)" if not sac["forms"] else ("MONITOR" if not sac["persists"] else "DESCENT")
    print(
        f"{idx:>2} | {lat:>6.2f} | {lon:>6.2f} | {temp_c:>8.1f} | {sac['TLC_C']:>10.1f} | "
        f"{rhi * 100.0:>5.0f}% | {str(sac['forms']):>7} | {str(sac['persists']):>9} | {action}"
    )

print("-" * 90)
print("Data Source: Open-Meteo Live GFS/ECMWF atmospheric pressure-level feed.\n")
