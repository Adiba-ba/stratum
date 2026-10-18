#!/usr/bin/env python3
"""Demo of the classical satellite contrail detector on a synthetic scene.

Builds a 160x160 split-window scene at 240 K with Gaussian noise that
contains a clear-sky background, one straight contrail (60 pixels at
+2.5 K BTD) and one round cirrus blob (radius 10 pixels at +3.0 K BTD),
runs :func:`stratum.detect.filters.detect_contrail_lines`, and prints a
short auditable report confirming detection of the contrail and
rejection of the blob.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratum.detect.filters import detect_contrail_lines  # noqa: E402

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
    rows = np.round(30.0 + (cols - start_col) * math.tan(math.radians(CONTRAIL_ANGLE_DEG))).astype(int)
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


def main() -> int:
    """Run the detector demo and print the summary report."""
    bt108, bt120 = build_scene(SEED)
    result = detect_contrail_lines(bt108, bt120)
    objects = result["objects"]

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
                f"{obj['length_px']:.1f} | {obj['angle_deg']:.1f} | {obj['mean_btd']:.3f} |"
            )
    else:
        print("No objects detected.")
    print()

    contrail_hits = [
        obj
        for obj in objects
        if _angular_distance(obj["angle_deg"], CONTRAIL_ANGLE_DEG) <= ANGLE_TOLERANCE_DEG
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


if __name__ == "__main__":
    raise SystemExit(main())
