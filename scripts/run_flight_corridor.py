#!/usr/bin/env python
"""Run the STRATUM end-to-end pipeline on a Delhi-Mumbai corridor flight.

Synthesizes a realistic VIDP -> VABB cruise profile with 25 waypoints,
loads the real ERA5 analysis file ``era5_del_bom_20250101.nc`` (falling
back to an in-memory synthetic dataset with download guidance), trains a
calibrated LightGBM baseline, runs
:func:`stratum.pipeline.evaluate_corridor_flight`, and prints an executive
flight report: flight metadata, segment-by-segment atmospheric and
contrail risk, and an ATC dispatcher advisory with recommended altitude
deviations, fuel deltas and avoided climate impact.

Exit code is 0 when the report completes.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

# The netCDF4 1.7.2 wheel can emit a spurious NumPy binary-compatibility
# RuntimeWarning on first engine use; functionality is exercised below.
warnings.filterwarnings(
    "ignore",
    message="numpy.ndarray size changed, may indicate binary incompatibility",
    category=RuntimeWarning,
)

# LightGBM's sklearn wrapper validates feature names even when predictions
# receive plain ndarrays fitted from ndarrays; silence its spurious warning.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, "
    "but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratum.io.era5 import (  # noqa: E402
    interpolate_met_to_segment,
    load_era5_dataset,
)
from stratum.model.calibrate import CalibratedModel, fit_calibrator  # noqa: E402
from stratum.model.features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features_row,
    feature_vector,
)
from stratum.model.train import compute_brier_score, train_contrail_model  # noqa: E402
from stratum.physics import e_i, fl_to_p, get_aircraft_params, solve_TLM  # noqa: E402
from stratum.pipeline import evaluate_corridor_flight  # noqa: E402
from stratum.routes.corridor import AIRPORTS, great_circle_waypoints  # noqa: E402

ERA5_FILENAME = "era5_del_bom_20250101.nc"
CALLSIGN = "AIC866"
ICAO24 = "800abc"
ROUTE = ("VIDP", "VABB")
AIRCRAFT_TYPE = "A320"
CRUISE_FL = 350.0
N_WAYPOINTS = 25
STEP_MINUTES = 6
TRAIN_SAMPLES = 3000
RNG_SEED = 42
IST_OFFSET_MINUTES = 330
RHI_CENTRE = 0.90
MARGIN_CENTRE = -4.0
RHI_SLOPE = 3.0
MARGIN_SLOPE = 0.15
RICHARDSON_CENTRE = 1.0
RICHARDSON_SLOPE = 0.5
ETA_CENTRE = 0.32
ETA_SLOPE = 4.0
LABEL_FLIP_PROBABILITY = 0.03
MASKED_LABEL_FRACTION = 0.25
TRAIN_FRACTION = 0.65
EPS = 0.622
GJ_IN_J = 1.0e9


def _candidate_paths() -> list[Path]:
    """Locations searched for the real ERA5 file, in order."""
    return [
        PROJECT_ROOT / ERA5_FILENAME,
        PROJECT_ROOT / "data" / ERA5_FILENAME,
        Path.cwd() / ERA5_FILENAME,
    ]


def _find_era5_file() -> Path | None:
    """Return the first existing candidate path, or ``None``."""
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate
    return None


def _build_mock_era5() -> xr.Dataset:
    """Fallback synthetic dataset when the real ERA5 file is absent.

    Mid-route humidity is tuned so the corridor contains an
    ice-supersaturated pocket, which keeps the fallback run meaningful.
    """
    levels = np.array([200.0, 225.0, 250.0, 300.0, 350.0, 400.0])
    latitudes = np.linspace(15.0, 32.0, 10)
    longitudes = np.linspace(68.0, 82.0, 10)
    times = np.arange(
        "2025-01-01T00:00", "2025-01-01T09:00", dtype="datetime64[h]"
    ).astype("datetime64[ns]")
    shape = (levels.size, latitudes.size, longitudes.size, times.size)

    level = levels[:, None, None, None]
    latitude = latitudes[None, :, None, None]
    longitude = longitudes[None, None, :, None]

    temperature = 216.65 + 0.11 * (level - 200.0)
    moisture_pocket = np.exp(-((longitude - 74.5) ** 2) / 8.0) * np.exp(
        -((level - 250.0) ** 2) / 4000.0
    )
    target_rhi = 0.35 + 1.15 * moisture_pocket
    saturation = np.array(
        [e_i(216.65 + 0.11 * (value - 200.0)) for value in levels]
    )[:, None, None, None]
    specific_humidity = target_rhi * saturation * EPS / (level * 100.0)

    u_wind = 8.0 + 0.05 * (latitude - 15.0) + 0.004 * (level - 200.0)
    v_wind = -3.0 + 0.001 * (level - 200.0)

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
                np.full(shape, 0.25),
            ),
        },
        coords={
            "level": levels,
            "latitude": latitudes,
            "longitude": longitudes,
            "time": times,
        },
    )
    return dataset.transpose("time", "level", "latitude", "longitude")


def _load_met() -> tuple[xr.Dataset, str]:
    """Load the real ERA5 file, or the synthetic fallback with a note."""
    path = _find_era5_file()
    if path is not None:
        return load_era5_dataset(str(path)), f"real ERA5 file {path}"
    return _build_mock_era5(), (
        "synthetic mock dataset - place "
        f"{ERA5_FILENAME} in the project root or data/ to use real analysis"
    )


def _rhi_at(
    ds_met: xr.Dataset, latitude: float, longitude: float, pressure_pa: float, time: Any
) -> float:
    """Ice-relative humidity from an interpolated ERA5 state."""
    met = interpolate_met_to_segment(ds_met, latitude, longitude, pressure_pa, time)
    humidity = met["specific_humidity"]
    vapor_pressure = humidity * pressure_pa / (EPS + (1.0 - EPS) * humidity)
    return vapor_pressure / e_i(met["temperature"])


def _select_departure_time(
    ds_met: xr.Dataset,
    waypoints: list[dict[str, Any]],
    cruise_fl: float,
) -> tuple[pd.Timestamp, float, int]:
    """Pick the worst-case departure hour for the corridor at cruise level.

    Scans every available ERA5 time whose flight window fits inside the
    dataset, probes a subsampled set of waypoints, and returns the hour
    with the highest route-mean RHi (plus that mean and the number of
    scanned candidate hours).
    """
    times = np.asarray(ds_met["time"].values)
    window = np.timedelta64((N_WAYPOINTS - 1) * STEP_MINUTES, "m")
    pressure_pa = fl_to_p(cruise_fl) * 100.0
    probes = waypoints[::3]

    best_time: pd.Timestamp | None = None
    best_mean = float("nan")
    scanned = 0
    for index in range(times.size):
        candidate = times[index]
        if candidate + window > times[-1]:
            break
        scanned += 1
        values = [
            _rhi_at(ds_met, waypoint["lat"], waypoint["lon"], pressure_pa, candidate)
            for waypoint in probes
        ]
        mean_rhi = float(np.nanmean(values))
        if best_time is None or mean_rhi > best_mean:
            best_time = pd.Timestamp(candidate).tz_localize("UTC")
            best_mean = mean_rhi
    if best_time is None:
        fallback = pd.Timestamp(times[0]).tz_localize("UTC")
        return fallback, float("nan"), 0
    return best_time, best_mean, scanned


def _synthesize_flight(departure: pd.Timestamp) -> pd.DataFrame:
    """Cleaned ADS-B shaped cruise profile for VIDP -> VABB."""
    waypoints = great_circle_waypoints(
        AIRPORTS[ROUTE[0]], AIRPORTS[ROUTE[1]], num_points=N_WAYPOINTS, fl=CRUISE_FL
    )
    records = []
    for index, waypoint in enumerate(waypoints):
        records.append(
            {
                "icao24": ICAO24,
                "callsign": CALLSIGN,
                "time": departure + pd.Timedelta(minutes=STEP_MINUTES * index),
                "lat": waypoint["lat"],
                "lon": waypoint["lon"],
                "baro_alt_ft": CRUISE_FL * 100.0,
                "gs_kt": 455.0,
                "track_deg": 214.0,
                "vrate_fpm": 0.0,
                "on_ground": False,
                "source": "opensky",
            }
        )
    return pd.DataFrame(records)


def _train_baseline() -> tuple[CalibratedModel, dict[str, Any]]:
    """Train the calibrated LightGBM baseline on a synthetic risk population.

    The ground-truth logit is monotone in the canonical features (RHi up,
    TLM margin down, Richardson number down, engine efficiency up) and
    the labels carry a small flip probability plus a quarter of
    non-observable NaNs, mirroring the Step 5 protocol.
    """
    rng = np.random.default_rng(RNG_SEED)
    engine_eta = float(get_aircraft_params(AIRCRAFT_TYPE)["eta"])
    vectors: list[np.ndarray] = []
    labels: list[float] = []

    for _ in range(TRAIN_SAMPLES):
        flight_level = float(rng.uniform(300.0, 400.0))
        pressure_pa = fl_to_p(flight_level) * 100.0
        tlm_k = solve_TLM(pressure_pa, engine_eta)
        margin = float(rng.uniform(-12.0, 4.0))
        rhi = float(rng.uniform(0.1, 1.6))
        rhi_above = max(0.0, rhi + float(rng.normal(0.0, 0.1)))
        rhi_below = max(0.0, rhi + float(rng.normal(0.0, 0.1)))
        richardson_number = float(rng.uniform(0.1, 4.0))
        solar_zenith_deg = float(rng.uniform(0.0, 180.0))
        engine_efficiency_eta = float(rng.uniform(0.30, 0.40))
        log_soot_ei_n = float(rng.uniform(14.0, 16.0))

        row = extract_features_row(
            rhi=rhi,
            rhi_above=rhi_above,
            rhi_below=rhi_below,
            rhi_grad=rhi_above - rhi_below,
            temp_k=tlm_k + margin,
            tlm_k=tlm_k,
            tlc_k=tlm_k - float(rng.uniform(0.5, 8.0)),
            pressure_pa=pressure_pa,
            cloud_ice=float(rng.random() < 0.2),
            high_cloud_fraction=float(rng.uniform(0.0, 0.8)),
            vertical_velocity_omega=float(rng.normal(0.0, 0.05)),
            vertical_wind_shear=float(rng.uniform(0.0, 6.0e-4)),
            dist_to_tropopause_m=float(rng.uniform(9000.0, 12500.0)) - 16000.0,
            lat=float(rng.uniform(15.0, 32.0)),
            lon=float(rng.uniform(68.0, 82.0)),
            month=1.0,
            local_solar_time_hours=float(rng.uniform(0.0, 24.0)),
            richardson_number=richardson_number,
            solar_zenith_deg=solar_zenith_deg,
            engine_efficiency_eta=engine_efficiency_eta,
            log_soot_ei_n=log_soot_ei_n,
        )
        vectors.append(feature_vector(row))

        logit = (
            RHI_SLOPE * (rhi - RHI_CENTRE)
            - MARGIN_SLOPE * (margin - MARGIN_CENTRE)
            - RICHARDSON_SLOPE * (richardson_number - RICHARDSON_CENTRE)
            + ETA_SLOPE * (engine_efficiency_eta - ETA_CENTRE)
        )
        probability = 1.0 / (1.0 + math.exp(-logit))
        label = 1.0 if rng.random() < probability else 0.0
        if rng.random() < LABEL_FLIP_PROBABILITY:
            label = 1.0 - label
        if rng.random() < MASKED_LABEL_FRACTION:
            label = float("nan")
        labels.append(label)

    features = np.vstack(vectors)
    label_array = np.asarray(labels, dtype=float)
    observed = np.isfinite(label_array)
    split = int(TRAIN_FRACTION * TRAIN_SAMPLES)
    indices = np.arange(TRAIN_SAMPLES)
    train_indices = indices[(indices < split) & observed]
    val_indices = indices[(indices >= split) & observed]

    model = train_contrail_model(
        features[train_indices], label_array[train_indices], list(FEATURE_NAMES)
    )
    calibrated = fit_calibrator(
        model, features[val_indices], label_array[val_indices]
    )
    raw = calibrated.predict_raw_proba(features[val_indices])
    calibrated_probability = calibrated.predict_proba(features[val_indices])[:, 1]
    metrics = {
        "n_samples": TRAIN_SAMPLES,
        "n_train": int(train_indices.size),
        "n_val": int(val_indices.size),
        "n_masked": int(np.sum(~observed)),
        "brier_raw": compute_brier_score(label_array[val_indices], raw),
        "brier_calibrated": compute_brier_score(
            label_array[val_indices], calibrated_probability
        ),
    }
    return calibrated, metrics


def _format_time(time_utc: str) -> str:
    """``HH:MM`` UTC from an ISO-8601 timestamp string."""
    return pd.Timestamp(time_utc).strftime("%H:%M")


def _print_report(
    flight: pd.DataFrame,
    departure: pd.Timestamp,
    scan_mean_rhi: float,
    scan_hours: int,
    source_label: str,
    metrics: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Print the executive flight report."""
    origin, destination = ROUTE
    origin_lat, origin_lon = AIRPORTS[origin]
    destination_lat, destination_lon = AIRPORTS[destination]
    arrival = departure + pd.Timedelta(minutes=STEP_MINUTES * (N_WAYPOINTS - 1))
    ist_departure = departure + pd.Timedelta(minutes=IST_OFFSET_MINUTES)

    print("=" * 78)
    print("STRATUM Step 8 - End-to-End Corridor Flight Report")
    print("=" * 78)
    print("Flight metadata")
    print("-" * 78)
    print(f"  Aircraft         : {CALLSIGN} (icao24 {ICAO24}), type {AIRCRAFT_TYPE}")
    print(
        f"  Route            : {origin} ({origin_lat:.2f}N, {origin_lon:.2f}E) -> "
        f"{destination} ({destination_lat:.2f}N, {destination_lon:.2f}E)"
    )
    print(
        f"  Profile          : great-circle, {N_WAYPOINTS} cruise waypoints, "
        f"FL{CRUISE_FL:.0f}, {STEP_MINUTES} min spacing"
    )
    print(
        f"  Departure        : {departure.strftime('%Y-%m-%d %H:%M')} UTC "
        f"({ist_departure.strftime('%H:%M')} IST), arrival "
        f"{arrival.strftime('%H:%M')} UTC"
    )
    print(f"  Trajectory rows  : {len(flight)}")
    print(f"  Weather data     : {source_label}")
    if scan_hours > 0:
        print(
            f"  Departure scan   : worst-case hour for FL{CRUISE_FL:.0f} "
            f"(route-mean RHi {scan_mean_rhi:.3f} over {scan_hours} candidate "
            "hours)"
        )
    print(
        f"  Model baseline   : LightGBM monotone + isotonic | "
        f"{metrics['n_train']} train / {metrics['n_val']} val rows "
        f"({metrics['n_masked']} NaN-masked) | val Brier "
        f"{metrics['brier_raw']:.4f} -> {metrics['brier_calibrated']:.4f}"
    )

    print()
    print("Segment-by-segment atmospheric and contrail risk breakdown")
    print("-" * 78)
    header = (
        f"{'#':>3}  {'UTC':>5}  {'lat':>7}  {'lon':>7}  {'FL':>3}  "
        f"{'T[C]':>6}  {'RHi':>5}  {'P(pers)':>7}  {'SAC':>4}  "
        f"{'ice':>3}  decision"
    )
    print(header)
    for segment in result["segments"]:
        temperature = segment["temperature_c"]
        rhi = segment["rhi"]
        decision = segment["decision_status"] or "-"
        if segment["decision_status"] == "eligible":
            decision = f"eligible -> FL{segment['recommended_fl']:.0f}"
        elif segment["decision_status"] == "no_robust_candidate":
            decision = "no robust candidate"
        print(
            f"{segment['segment_index'] + 1:>3}  "
            f"{_format_time(segment['time_utc']):>5}  "
            f"{segment['lat']:>7.2f}  {segment['lon']:>7.2f}  "
            f"{segment['flight_level']:>3.0f}  "
            f"{temperature:>6.1f}  {rhi:>5.2f}  "
            f"{segment['p_persist']:>7.3f}  "
            f"{'yes' if segment['sac_forms'] else 'no':>4}  "
            f"{'yes' if segment['ambient_ice_supersaturated'] else 'no':>3}  "
            f"{decision}"
        )

    print()
    print("ATC Dispatcher Advisory")
    print("-" * 78)
    reroutes = result["recommended_reroutes"]
    if not reroutes:
        print(
            "  No advisories: calibrated persistence risk stayed below the "
            "0.40 activation threshold (or no robust candidate) on all "
            "segments; flying the filed profile costs nothing extra."
        )
    else:
        for reroute in reroutes:
            print(
                f"  Event {reroute['event_index'] + 1}: segment "
                f"#{reroute['segment_index'] + 1} at "
                f"{_format_time(reroute['time_utc'])} UTC, "
                f"FL{reroute['baseline_fl']:.0f} -> "
                f"FL{reroute['recommended_fl']:.0f} ({reroute['direction']})"
            )
            print(
                f"    Calibrated P(persist): {reroute['p_persist']:.3f} | "
                f"{reroute['segments_in_event']} contiguous flagged segment(s)"
            )
            print(
                f"    Delta fuel: +{reroute['delta_fuel_kg']:.2f} kg | "
                f"delta CO2: +{reroute['delta_co2_kg']:.2f} kg"
            )
            print(
                f"    Avoided contrail warming: "
                f"{reroute['avoided_contrail_ef_j'] / GJ_IN_J:.2f} GJ | "
                f"benefit-to-cost ratio "
                f"{reroute['benefit_to_cost_ratio']:.3e}"
            )
        advisories = f"{len(reroutes)} advisory event" + (
            "s" if len(reroutes) != 1 else ""
        )
        print(
            f"  Totals: {advisories} | "
            f"+{result['total_extra_fuel_kg']:.2f} kg fuel | "
            f"+{result['total_extra_co2_kg']:.2f} kg CO2 | "
            f"{result['total_avoidable_warming_j'] / GJ_IN_J:.2f} GJ avoided "
            "warming"
        )

    print()
    print("Corridor summary")
    print("-" * 78)
    print(
        f"  Segments: {result['total_segments']} total | "
        f"{result['contrail_forming_segments']} SAC-forming | "
        f"{result['contrail_persistent_segments']} predicted persistent | "
        f"{result['flagged_segments']} flagged for avoidance"
    )
    print()


def main() -> int:
    """Run the corridor flight report and return the process exit code."""
    ds_met, source_label = _load_met()
    waypoints = great_circle_waypoints(
        AIRPORTS[ROUTE[0]], AIRPORTS[ROUTE[1]], num_points=N_WAYPOINTS, fl=CRUISE_FL
    )
    departure, scan_mean_rhi, scan_hours = _select_departure_time(
        ds_met, waypoints, CRUISE_FL
    )
    flight = _synthesize_flight(departure)
    model, metrics = _train_baseline()
    result = evaluate_corridor_flight(flight, ds_met, model, aircraft_type=AIRCRAFT_TYPE)
    _print_report(
        flight, departure, scan_mean_rhi, scan_hours, source_label, metrics, result
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
