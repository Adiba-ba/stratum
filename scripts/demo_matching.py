#!/usr/bin/env python3
"""Demo of flight-contrail association with parallax and tri-state labels.

Simulates a flight segment on the Delhi-Mumbai corridor at FL350,
advects it 45 minutes downwind with a 30 m/s westerly jet, applies the
Meteosat-9 parallax correction, matches it against a synthetic satellite
detection, and demonstrates the tri-state labels for a clear-sky match,
a clear-sky non-detection and an unobservable storm scene.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratum.match.advect import advect_segment  # noqa: E402
from stratum.match.label import (  # noqa: E402
    assign_tristate_label,
    haversine_km,
    match_segment_to_detections,
)
from stratum.match.observe import is_scene_observable  # noqa: E402
from stratum.match.parallax import calculate_parallax_displacement  # noqa: E402
from stratum.routes.corridor import AIRPORTS, great_circle_waypoints  # noqa: E402

FL = 350.0
SEGMENT_WAYPOINT_INDEX = 10
ADVECTION_SECONDS = 45.0 * 60.0
U_JET_MPS = 30.0  # westerly jet: eastward
V_JET_MPS = 0.0
ALTITUDE_M = 35000.0 * 0.3048
DISTANCE_TOLERANCE_KM = 10.0
ANGLE_TOLERANCE_DEG = 20.0
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


def main() -> int:
    """Run the matching demo and print the summary tables."""
    waypoints = great_circle_waypoints(
        AIRPORTS["VIDP"], AIRPORTS["VABB"], num_points=25, fl=FL
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
        advected[0], advected[1], ALTITUDE_M
    )
    apparent = (advected[0] + delta_lat, advected[1] + delta_lon)
    parallax_km = haversine_km(advected, apparent)

    detection = {
        "centroid": (apparent[0] + DETECTION_OFFSET_DEG, apparent[1]),
        "angle_deg": heading_deg % 180.0,
        "length_px": 42.0,
        "mean_btd": 1.8,
    }

    print("Project STRATUM - flight-contrail association demo")
    print(
        f"Segment: VIDP-VABB waypoint {SEGMENT_WAYPOINT_INDEX} at FL{FL:.0f} "
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
        "| Scenario | Sky condition | Observable slots | Matched | Label | Interpretation |"
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
            angle_tolerance_deg=ANGLE_TOLERANCE_DEG,
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


if __name__ == "__main__":
    raise SystemExit(main())
