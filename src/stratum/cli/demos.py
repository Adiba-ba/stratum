"""Demo subcommands for the STRATUM command line.

Ports ``scripts/demo_detector.py``, ``scripts/demo_matching.py``,
``scripts/demo_decide.py`` and ``scripts/demo_model.py``. Every demo runs on
synthetic fixtures only and prints the banner
:data:`SYNTHETIC_BANNER` as its first line, so no output can be mistaken
for a real observation or a trained production model.
"""

from __future__ import annotations

import argparse
import math
import warnings
from typing import Callable

import numpy as np

from stratum.decide import (
    AltitudeDecision,
    CandidateEvaluation,
    evaluate_altitude_candidates,
)
from stratum.detect.filters import detect_contrail_lines
from stratum.match.advect import advect_segment
from stratum.match.label import (
    assign_tristate_label,
    haversine_km,
    match_segment_to_detections,
)
from stratum.match.observe import is_scene_observable
from stratum.match.parallax import calculate_parallax_displacement
from stratum.model import (
    CalibratedModel,
    FEATURE_NAMES,
    compute_brier_score,
    extract_features_row,
    feature_vector,
    fit_calibrator,
    train_contrail_model,
)
from stratum.model.train import MONOTONE_CONSTRAINTS
from stratum.physics import (
    G,
    critical_T,
    e_i,
    e_w,
    fl_to_p,
    solve_TLM,
)
from stratum.routes.corridor import AIRPORTS, great_circle_waypoints

__all__ = ["demo"]

#: First output line of every demo; fixtures are synthetic, not observations.
SYNTHETIC_BANNER = "Synthetic fixture demo - not real data."

# ---------------------------------------------------------------------------
# detector demo (scripts/demo_detector.py)
# ---------------------------------------------------------------------------

SEED = 0
SCENE_SIZE = 160
BACKGROUND_K = 240.0
NOISE_STD_K = 0.25
CONTRAIL_ANGLE_DEG = 35.0
CONTRAIL_LENGTH_PX = 60
CONTRAIL_ANOMALY_K = 2.5
BLOB_CENTRE = (120.0, 110.0)
BLOB_RADIUS_PX = 10
BLOB_ANOMALY_K = 3.0
ANGLE_TOLERANCE_DEG = 15.0


def build_scene(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a synthetic (bt108, bt120) scene with a contrail and a blob."""
    rng = np.random.default_rng(seed)
    bt108 = BACKGROUND_K + rng.normal(0.0, NOISE_STD_K, (SCENE_SIZE, SCENE_SIZE))
    bt120 = BACKGROUND_K + rng.normal(0.0, NOISE_STD_K, (SCENE_SIZE, SCENE_SIZE))

    start_col = 30
    cols = np.arange(start_col, start_col + CONTRAIL_LENGTH_PX)
    rows = np.round(
        30.0 + (cols - start_col) * math.tan(math.radians(CONTRAIL_ANGLE_DEG))
    ).astype(int)
    bt108[np.clip(rows, 0, SCENE_SIZE - 1), cols] += CONTRAIL_ANOMALY_K

    grid_y, grid_x = np.mgrid[0:SCENE_SIZE, 0:SCENE_SIZE]
    blob_mask = (
        (grid_y - BLOB_CENTRE[0]) ** 2 + (grid_x - BLOB_CENTRE[1]) ** 2
        <= BLOB_RADIUS_PX**2
    )
    bt108[blob_mask] += BLOB_ANOMALY_K
    return bt108, bt120


def _angular_distance(angle_a: float, angle_b: float) -> float:
    """Return the smallest angle difference in [0, 90] degrees."""
    difference = abs(angle_a - angle_b) % 180.0
    return min(difference, 180.0 - difference)


def _demo_detector() -> int:
    """Run the detector demo and print the summary report."""
    bt108, bt120 = build_scene(SEED)
    result = detect_contrail_lines(bt108, bt120)
    objects = result["objects"]

    print(SYNTHETIC_BANNER)
    print("Project STRATUM - classical satellite contrail detector demo")
    print(
        f"Scene {SCENE_SIZE}x{SCENE_SIZE} px, {BACKGROUND_K:.0f} K background, "
        f"Gaussian noise sigma={NOISE_STD_K:.2f} K (seed {SEED})"
    )
    print(
        f"Planted contrail: {CONTRAIL_LENGTH_PX} px at {CONTRAIL_ANGLE_DEG:.1f} deg, "
        f"+{CONTRAIL_ANOMALY_K:.1f} K BTD"
    )
    print(
        f"Planted cloud blob: radius {BLOB_RADIUS_PX} px at {BLOB_CENTRE}, "
        f"+{BLOB_ANOMALY_K:.1f} K BTD"
    )
    print()

    if objects:
        print("| Object | Centroid (row, col) | Length px | Angle deg | Mean BTD K |")
        print("| --- | --- | --- | --- | --- |")
        for index, obj in enumerate(objects, start=1):
            centroid = obj["centroid"]
            print(
                f"| {index} | ({centroid[0]:.1f}, {centroid[1]:.1f}) | "
                f"{obj['length_px']:.1f} | {obj['angle_deg']:.1f} | "
                f"{obj['mean_btd']:.3f} |"
            )
    else:
        print("No objects detected.")
    print()

    contrail_hits = [
        obj
        for obj in objects
        if _angular_distance(obj["angle_deg"], CONTRAIL_ANGLE_DEG)
        <= ANGLE_TOLERANCE_DEG
    ]
    blob_hits = [
        obj
        for obj in objects
        if math.hypot(
            obj["centroid"][0] - BLOB_CENTRE[0], obj["centroid"][1] - BLOB_CENTRE[1]
        )
        <= BLOB_RADIUS_PX + 6.0
    ]
    contrail_detected = len(contrail_hits) >= 1
    blob_rejected = len(blob_hits) == 0

    print(
        f"Contrail detected: {'YES' if contrail_detected else 'NO'} "
        f"({len(contrail_hits)} object(s) within +/-{ANGLE_TOLERANCE_DEG:.0f} deg "
        f"of {CONTRAIL_ANGLE_DEG:.1f} deg)"
    )
    print(
        f"Round cloud blob rejected: {'YES' if blob_rejected else 'NO'} "
        f"({len(blob_hits)} detection(s) near the blob centre)"
    )
    print(f"Total objects after length/aspect gates: {result['num_detected']}")

    return 0 if (contrail_detected and blob_rejected) else 1


# ---------------------------------------------------------------------------
# matching demo (scripts/demo_matching.py)
# ---------------------------------------------------------------------------

MATCH_FL = 350.0
SEGMENT_WAYPOINT_INDEX = 10
ADVECTION_SECONDS = 45.0 * 60.0
U_JET_MPS = 30.0  # westerly jet: eastward
V_JET_MPS = 0.0
MATCH_ALTITUDE_M = 35000.0 * 0.3048
DISTANCE_TOLERANCE_KM = 10.0
MATCH_ANGLE_TOLERANCE_DEG = 20.0
TOTAL_SLOTS = 3
DETECTION_OFFSET_DEG = 0.005  # ~0.5 km north of the apparent position


def _initial_bearing_deg(
    coord1: tuple[float, float], coord2: tuple[float, float]
) -> float:
    """Initial great-circle bearing from coord1 to coord2 [deg, 0-360]."""
    lat1, lat2 = math.radians(coord1[0]), math.radians(coord2[0])
    delta_lon = math.radians(coord2[1] - coord1[1])
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        delta_lon
    )
    return math.degrees(math.atan2(x, y)) % 360.0


def _demo_matching() -> int:
    """Run the matching demo and print the summary tables."""
    waypoints = great_circle_waypoints(
        AIRPORTS["VIDP"], AIRPORTS["VABB"], num_points=25, fl=MATCH_FL
    )
    current = waypoints[SEGMENT_WAYPOINT_INDEX]
    following = waypoints[SEGMENT_WAYPOINT_INDEX + 1]
    segment_coord = (current["lat"], current["lon"])
    heading_deg = _initial_bearing_deg(
        segment_coord, (following["lat"], following["lon"])
    )

    advected = advect_segment(
        segment_coord[0], segment_coord[1], U_JET_MPS, V_JET_MPS, ADVECTION_SECONDS
    )
    drift_km = haversine_km(segment_coord, advected)
    delta_lat, delta_lon = calculate_parallax_displacement(
        advected[0], advected[1], MATCH_ALTITUDE_M
    )
    apparent = (advected[0] + delta_lat, advected[1] + delta_lon)
    parallax_km = haversine_km(advected, apparent)

    detection = {
        "centroid": (apparent[0] + DETECTION_OFFSET_DEG, apparent[1]),
        "angle_deg": heading_deg % 180.0,
        "length_px": 42.0,
        "mean_btd": 1.8,
    }

    print(SYNTHETIC_BANNER)
    print("Project STRATUM - flight-contrail association demo")
    print(
        f"Segment: VIDP-VABB waypoint {SEGMENT_WAYPOINT_INDEX} at FL{MATCH_FL:.0f} "
        f"(lat {segment_coord[0]:.4f}, lon {segment_coord[1]:.4f}), "
        f"heading {heading_deg:.1f} deg"
    )
    print(
        f"Advection: 45 min at {U_JET_MPS:.0f} m/s westerly -> drift {drift_km:.1f} km "
        f"(lat {advected[0]:.4f}, lon {advected[1]:.4f})"
    )
    print(
        f"Meteosat-9 parallax: delta_lat {delta_lat:+.4f} deg, "
        f"delta_lon {delta_lon:+.4f} deg ({parallax_km:.1f} km) -> "
        f"apparent (lat {apparent[0]:.4f}, lon {apparent[1]:.4f})"
    )
    print(
        f"Synthetic detection: centroid offset "
        f"{haversine_km(apparent, detection['centroid']):.1f} km, "
        f"angle {detection['angle_deg']:.1f} deg"
    )
    print()

    scenarios: list[tuple[str, str, float, float, list[dict[str, object]], bool]] = [
        (
            "1",
            "Clear sky, matching line",
            0.05,
            32.0,
            [detection],
            True,
        ),
        (
            "2",
            "Clear sky, no line",
            0.05,
            32.0,
            [],
            True,
        ),
        (
            "3",
            "Stormy/cloudy sky, no line",
            0.85,
            32.0,
            [],
            False,
        ),
    ]

    print(
        "| Scenario | Sky condition | Observable slots | Matched | Label | "
        "Interpretation |"
    )
    print("| --- | --- | --- | --- | --- | --- |")

    expected_labels: list[float] = []
    actual_labels: list[float] = []
    for name, sky, cloud_fraction, zenith, detections, expected_observable in scenarios:
        observable = is_scene_observable(cloud_fraction, zenith)
        assert observable is expected_observable
        observable_slots = TOTAL_SLOTS if observable else 0
        matched = match_segment_to_detections(
            apparent,
            heading_deg,
            detections,
            distance_tolerance_km=DISTANCE_TOLERANCE_KM,
            angle_tolerance_deg=MATCH_ANGLE_TOLERANCE_DEG,
        )
        label_value = assign_tristate_label(matched, observable_slots)
        if math.isnan(label_value):
            interpretation = "Unknown/unobservable (NaN, not a negative)"
            label_text = "NaN"
        elif label_value == 1.0:
            interpretation = "Positive: contrail detected on segment"
            label_text = "1.0"
        else:
            interpretation = "Observable non-detection"
            label_text = "0.0"
        print(
            f"| {name} | {sky} | {observable_slots} | {matched} | "
            f"{label_text} | {interpretation} |"
        )
        expected = 1.0 if name == "1" else (0.0 if name == "2" else float("nan"))
        expected_labels.append(expected)
        actual_labels.append(label_value)

    ok = all(
        (math.isnan(exp) and math.isnan(act)) or exp == act
        for exp, act in zip(expected_labels, actual_labels)
    )
    print()
    print(
        "Tri-state labels as expected: "
        f"{'YES' if ok else 'NO'} (1.0 positive, 0.0 observable non-detection, "
        "NaN unobservable/unknown)"
    )
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# decision demo (scripts/demo_decide.py)
# ---------------------------------------------------------------------------

FLIGHT_LABEL = "VIDP (Delhi) -> VABB (Mumbai)"
AIRCRAFT_TYPE = "A320"
BASELINE_FL = 350.0
BASELINE_P_PERSIST = 0.78


def _format_candidate(candidate: CandidateEvaluation) -> str:
    """Render one auditable candidate row."""
    label = f"FL{candidate['target_fl']:.0f}"
    change = f"{candidate['direction']} {candidate['altitude_change_ft']:+.0f} ft"
    if candidate["rejected"]:
        return f"{label}  {change:<18}: REJECT ({candidate['rejection_reason']})"
    verdict = "ELIGIBLE" if candidate["net_benefit_positive"] else "REJECT"
    avoided = candidate["avoided_ef_j"]
    return (
        f"{label}  {change:<18}: fuel +{candidate['delta_fuel_kg']:.2f} kg | "
        f"CO2 +{candidate['delta_co2_kg']:.2f} kg | "
        f"avoided warming {avoided:.2e} J ({avoided / 1.0e9:.2f} GJ) | "
        f"benefit/cost {candidate['benefit_to_cost_ratio']:.2e} | {verdict}"
    )


def _dispatcher_recommendation(decision: AltitudeDecision) -> str:
    """Human-readable recommendation for the dispatcher."""
    baseline_fl = decision["baseline_fl"]
    status = decision["status"]
    if status == "abstain":
        return (
            f"No action at FL{baseline_fl:.0f}: persistence probability "
            f"{decision['baseline_p_persist']:.2f} is below the 0.40 activation "
            "threshold, so extra fuel is not justified."
        )
    if status == "no_robust_candidate":
        return (
            f"Keep FL{baseline_fl:.0f}: no counterfactual flight level passes the "
            "benefit test under the current constraints."
        )

    recommended = decision["recommended_fl"]
    delta_fuel = decision["delta_fuel_kg"]
    delta_co2 = decision["delta_co2_kg"]
    avoided = decision["avoided_contrail_ef_j"]
    if (
        recommended is None
        or delta_fuel is None
        or delta_co2 is None
        or avoided is None
    ):
        raise RuntimeError("eligible decision is missing recommendation fields")
    direction = "step climb" if recommended > baseline_fl else "descent"
    return (
        f"Recommend {direction} from FL{baseline_fl:.0f} to "
        f"FL{recommended:.0f} for the "
        f"flagged segment: avoids {avoided:.2e} J of contrail energy forcing for an "
        f"extra {delta_fuel:.1f} kg of fuel ({delta_co2:.1f} kg CO2). Coordinate with "
        "ATC and confirm fuel reserves before the level change."
    )


def _demo_decision() -> int:
    """Run the counterfactual decision layer demo."""
    decision = evaluate_altitude_candidates(
        baseline_p_persist=BASELINE_P_PERSIST,
        baseline_fl=BASELINE_FL,
        aircraft_type=AIRCRAFT_TYPE,
    )

    print(SYNTHETIC_BANNER)
    print("STRATUM Step 6 - Counterfactual Decision Layer demo")
    print("=" * 66)
    print(f"Flight: {FLIGHT_LABEL}, aircraft {AIRCRAFT_TYPE}")
    print(
        f"Flagged segment at FL{BASELINE_FL:.0f} with calibrated "
        f"P(persist) = {BASELINE_P_PERSIST:.2f} (activation threshold 0.40)"
    )
    print()
    print("Counterfactual candidates")
    print("-" * 66)
    for candidate in decision["candidates"]:
        print(_format_candidate(candidate))
    print()
    print("Decision card")
    print("-" * 66)
    print(f"Baseline flight level:      FL{decision['baseline_fl']:.0f}")
    print(f"Calibrated P(persist):      {decision['baseline_p_persist']:.2f}")
    recommended = decision["recommended_fl"]
    if recommended is not None:
        delta_fl = recommended - decision["baseline_fl"]
        direction = "step climb" if delta_fl > 0.0 else "descent"
        print(f"Recommended flight level:   FL{recommended:.0f} ({direction})")
        print(f"Delta fuel:                 +{decision['delta_fuel_kg']:.2f} kg")
        print(f"Delta CO2:                  +{decision['delta_co2_kg']:.2f} kg")
        avoided = decision["avoided_contrail_ef_j"]
        print(f"Avoided warming:            {avoided:.2e} J ({avoided / 1.0e9:.2f} GJ)")
        print(f"Benefit-to-cost ratio:      {decision['benefit_to_cost_ratio']:.2e}")
    else:
        print("Recommended flight level:   none")
    print(f"Final status:               {decision['status'].upper()}")
    print(f"Reason: {decision['reason']}")
    print()
    print(f"Dispatcher recommendation: {_dispatcher_recommendation(decision)}")
    return 0 if decision["status"] == "eligible" else 1


# ---------------------------------------------------------------------------
# model fixture demo (scripts/demo_model.py)
# ---------------------------------------------------------------------------

N_SEGMENTS = 2000
TRAIN_FRACTION = 0.7
UNOBSERVABLE_FRACTION = 0.2
LABEL_NOISE_FRACTION = 0.05
RANDOM_SEED = 42
TROPOPAUSE_ALTITUDE_M = 11000.0

RHI_INDEX = FEATURE_NAMES.index("rhi")
TLM_K_INDEX = FEATURE_NAMES.index("tlm_k")


def _generate_dataset(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic segments with physics-based tri-state labels.

    Each segment samples a flight level and derives pressure and the
    Schmidt-Appleman thresholds from :mod:`stratum.physics`. The label is
    drawn from a latent risk that increases with ice supersaturation and
    decreases with the temperature margin, then flipped with sensor noise.
    Unobservable segments (opaque cloud) get a NaN label.
    """
    rows: list[np.ndarray] = []
    labels: list[float] = []
    for _ in range(N_SEGMENTS):
        flight_level = rng.uniform(300.0, 400.0)
        pressure_pa = fl_to_p(flight_level) * 100.0
        altitude_m = flight_level * 100.0 * 0.3048
        tlm_k = solve_TLM(pressure_pa)
        temp_margin_tlm = rng.uniform(-12.0, 4.0)
        temp_k = tlm_k + temp_margin_tlm
        rhi = float(rng.beta(2.0, 2.0)) * 1.7
        rhw = rhi * e_i(temp_k) / e_w(temp_k)
        tlc_k = critical_T(rhw, tlm_k, G(pressure_pa))

        rhi_above = float(np.clip(rhi + rng.normal(0.0, 0.12), 0.0, 2.0))
        rhi_below = float(np.clip(rhi + rng.normal(0.0, 0.12), 0.0, 2.0))

        unobservable = rng.random() < UNOBSERVABLE_FRACTION
        if unobservable:
            high_cloud_fraction = float(rng.uniform(0.6, 0.95))
        else:
            high_cloud_fraction = float(rng.uniform(0.0, 0.6))

        richardson_number = float(rng.uniform(0.1, 4.0))
        solar_zenith_deg = float(rng.uniform(0.0, 180.0))
        engine_efficiency_eta = float(rng.uniform(0.30, 0.40))
        log_soot_ei_n = float(rng.uniform(14.0, 16.0))

        logit = (
            2.4 * (rhi - 0.85)
            - 0.35 * temp_margin_tlm
            - 1.5 * high_cloud_fraction
            - 0.5 * (richardson_number - 1.0)
            + 4.0 * (engine_efficiency_eta - 0.32)
        )
        persistent = rng.random() < 1.0 / (1.0 + math.exp(-logit))
        if rng.random() < LABEL_NOISE_FRACTION:
            persistent = not persistent
        label = float("nan") if unobservable else float(persistent)

        lon = float(rng.uniform(68.0, 90.0))
        utc_hours = float(rng.uniform(0.0, 24.0))
        dist_to_tropopause_m = (
            altitude_m - TROPOPAUSE_ALTITUDE_M + float(rng.normal(0.0, 150.0))
        )
        row = extract_features_row(
            rhi=rhi,
            rhi_above=rhi_above,
            rhi_below=rhi_below,
            rhi_grad=rhi_above - rhi_below,
            temp_k=temp_k,
            tlm_k=tlm_k,
            tlc_k=tlc_k,
            pressure_pa=pressure_pa,
            cloud_ice=float(rng.random() < 0.25),
            high_cloud_fraction=high_cloud_fraction,
            vertical_velocity_omega=float(rng.normal(0.0, 0.08)),
            vertical_wind_shear=float(rng.uniform(0.0, 15.0)),
            dist_to_tropopause_m=dist_to_tropopause_m,
            lat=float(rng.uniform(8.0, 30.0)),
            lon=lon,
            month=float(rng.integers(1, 13)),
            local_solar_time_hours=(utc_hours + lon / 15.0) % 24.0,
            richardson_number=richardson_number,
            solar_zenith_deg=solar_zenith_deg,
            engine_efficiency_eta=engine_efficiency_eta,
            log_soot_ei_n=log_soot_ei_n,
        )
        rows.append(feature_vector(row))
        labels.append(label)
    return np.vstack(rows), np.asarray(labels, dtype=float)


def _sweep(
    model: CalibratedModel,
    baseline: np.ndarray,
    index: int,
    values: np.ndarray,
) -> np.ndarray:
    """Calibrated positive-class probabilities along one feature sweep."""
    probabilities = np.empty(values.size, dtype=float)
    for position, value in enumerate(values):
        row = baseline.copy()
        row[index] = value
        probabilities[position] = model.predict_proba(row)[0, 1]
    return probabilities


def _demo_model_fixture() -> int:
    """Train the correction model on synthetic data and report the checks.

    The monotonicity self-check asserts the constraint directions actually
    enforced by :data:`stratum.model.train.MONOTONE_CONSTRAINTS`: ``rhi`` and
    ``tlm_k`` both carry ``+1``, so P(persist) must be non-decreasing in both.
    The synthetic generator drives its latent label with the temperature
    margin ``temp_k - tlm_k``, so the learned response to ``tlm_k`` comes from
    the constraint, not from the generator.
    """
    warnings.filterwarnings(
        "ignore",
        message=(
            "X does not have valid feature names, but LGBMClassifier was "
            "fitted with feature names"
        ),
    )

    rng = np.random.default_rng(RANDOM_SEED)
    features, labels = _generate_dataset(rng)
    n_samples = features.shape[0]
    n_train = int(round(n_samples * TRAIN_FRACTION))
    permutation = rng.permutation(n_samples)
    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:]
    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]

    model = train_contrail_model(X_train, y_train, list(FEATURE_NAMES))
    calibrated = fit_calibrator(model, X_val, y_val)

    valid_val = ~np.isnan(y_val)
    X_val_valid = X_val[valid_val]
    y_val_valid = y_val[valid_val]
    raw_probs = model.predict_proba(X_val_valid)[:, 1]
    calibrated_probs = calibrated.predict_proba(X_val_valid)[:, 1]
    raw_brier = compute_brier_score(y_val_valid, raw_probs)
    calibrated_brier = compute_brier_score(y_val_valid, calibrated_probs)

    baseline = np.nanmedian(X_train[~np.isnan(y_train)], axis=0)
    rhi_probs = _sweep(calibrated, baseline, RHI_INDEX, np.linspace(0.1, 1.5, 50))
    tlm_probs = _sweep(
        calibrated,
        baseline,
        TLM_K_INDEX,
        np.linspace(baseline[TLM_K_INDEX] - 10.0, baseline[TLM_K_INDEX] + 10.0, 50),
    )
    # MONOTONE_CONSTRAINTS sets both features to +1 (increasing), so the
    # calibrated probability must be non-decreasing along both sweeps.
    assert MONOTONE_CONSTRAINTS["rhi"] == 1
    assert MONOTONE_CONSTRAINTS["tlm_k"] == 1
    rhi_monotone = bool(np.all(np.diff(rhi_probs) >= -1e-9))
    tlm_monotone = bool(np.all(np.diff(tlm_probs) >= -1e-9))

    cold_humid = baseline.copy()
    cold_humid[RHI_INDEX] = 1.25
    cold_humid[TLM_K_INDEX] = baseline[TLM_K_INDEX] - 6.0
    warm_dry = baseline.copy()
    warm_dry[RHI_INDEX] = 0.35
    warm_dry[TLM_K_INDEX] = baseline[TLM_K_INDEX] + 6.0

    train_masked = int(np.isnan(y_train).sum())
    val_masked = int(np.isnan(y_val).sum())
    print(SYNTHETIC_BANNER)
    print("STRATUM Step 5 - Physics-Constrained Correction Model demo")
    print("=" * 64)
    print(f"Synthetic segments: {n_samples} (seed={RANDOM_SEED})")
    print(f"Feature schema: {len(FEATURE_NAMES)} features (frozen Section 12.2 order)")
    print(
        f"Train split:      {y_train.size} rows -> "
        f"{y_train.size - train_masked} labeled, "
        f"{train_masked} NaN-masked (unobservable)"
    )
    print(
        f"Validation split: {y_val.size} rows -> "
        f"{y_val.size - val_masked} labeled, "
        f"{val_masked} NaN-masked (unobservable)"
    )
    print()
    print(f"Raw validation Brier score:          {raw_brier:.5f}")
    print(f"Calibrated validation Brier score:   {calibrated_brier:.5f}")
    print(f"Brier improvement after calibration: {raw_brier - calibrated_brier:+.5f}")
    print()
    rhi_status = "PASS" if rhi_monotone else "FAIL"
    tlm_status = "PASS" if tlm_monotone else "FAIL"
    print(
        "Monotonicity (RHi increasing -> P(persist) non-decreasing):"
        f"            {rhi_status}"
    )
    print(
        "Monotonicity (tlm_k increasing -> P(persist) non-decreasing):"
        f"          {tlm_status}"
    )
    print()
    print("Sample predictions (all other features held at the training median):")
    for name, row in (("cold/humid", cold_humid), ("warm/dry", warm_dry)):
        raw_p = float(calibrated.predict_raw_proba(row)[0])
        calibrated_p = float(calibrated.predict_proba(row)[0, 1])
        print(
            f"  {name:>10}: rhi={row[RHI_INDEX]:.2f}, "
            f"tlm_k={row[TLM_K_INDEX]:.1f} K "
            f"-> raw {raw_p:.3f}, calibrated {calibrated_p:.3f}"
        )
    return 0 if (rhi_monotone and tlm_monotone) else 1


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

#: Demo name -> implementation. Names are the CLI subcommand names.
_DEMOS: dict[str, Callable[[], int]] = {
    "detector": _demo_detector,
    "matching": _demo_matching,
    "decision": _demo_decision,
    "model-fixture": _demo_model_fixture,
}


def demo(argv: list[str]) -> int:
    """Run one synthetic-fixture demo selected by the first argument.

    Parameters
    ----------
    argv : list of str
        Arguments after ``demo``; the first value is one of ``detector``,
        ``matching``, ``decision`` or ``model-fixture``.

    Returns
    -------
    int
        The exit code of the selected demo; 2 when no demo is named.
    """
    parser = argparse.ArgumentParser(
        prog="stratum demo",
        description=(
            "Run one standalone demo of a STRATUM subsystem. All demos use "
            "synthetic fixtures and are not real data."
        ),
    )
    parser.add_argument(
        "demo",
        choices=sorted(_DEMOS),
        help="demo to run",
    )
    if not argv:
        parser.print_help()
        return 2
    args = parser.parse_args(argv)

    runner = _DEMOS[args.demo]
    return int(runner())
