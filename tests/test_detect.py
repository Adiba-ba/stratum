"""Tests for the classical satellite contrail detector.

Scenes are synthetic 100x100 split-window fields at 240 K with Gaussian
noise. A contrail is planted as a straight 20-pixel line with a +2.5 K
BTD anomaly; a cirrus-like cloud is planted as a round 20x20 blob with a
+3.0 K BTD anomaly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stratum.detect import btd as btd_module
from stratum.detect import filters

BACKGROUND_K = 240.0
SCENE_SIZE = 100
NOISE_STD_K = 0.2
PLANTED_ANGLE_DEG = 30.0
PLANTED_LINE_LENGTH_PX = 20
LINE_ANOMALY_K = 2.5
BLOB_ANOMALY_K = 3.0
BLOB_RADIUS_PX = 10


def _background_scene(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a 240 K split-window background with independent Gaussian noise."""
    rng = np.random.default_rng(seed)
    bt108 = BACKGROUND_K + rng.normal(0.0, NOISE_STD_K, (SCENE_SIZE, SCENE_SIZE))
    bt120 = BACKGROUND_K + rng.normal(0.0, NOISE_STD_K, (SCENE_SIZE, SCENE_SIZE))
    return bt108, bt120


def _plant_line(
    bt108: np.ndarray, angle_deg: float, length_px: int, anomaly_k: float
) -> np.ndarray:
    """Add a straight line BTD anomaly of ``length_px`` pixels to ``bt108``."""
    start_col = (SCENE_SIZE - length_px) // 2
    cols = np.arange(start_col, start_col + length_px)
    rows = np.round(
        SCENE_SIZE / 2.0 + (cols - start_col) * math.tan(math.radians(angle_deg))
    ).astype(int)
    rows = np.clip(rows, 0, SCENE_SIZE - 1)
    bt108[rows, cols] += anomaly_k
    return bt108


def _plant_blob(
    bt108: np.ndarray, radius_px: float, anomaly_k: float
) -> np.ndarray:
    """Add a round circular blob BTD anomaly to ``bt108``."""
    grid_y, grid_x = np.mgrid[0:SCENE_SIZE, 0:SCENE_SIZE]
    mask = (grid_y - SCENE_SIZE / 2.0) ** 2 + (grid_x - SCENE_SIZE / 2.0) ** 2 <= radius_px**2
    bt108[mask] += anomaly_k
    return bt108


def _angular_distance(angle_a: float, angle_b: float) -> float:
    """Return the smallest angle difference in [0, 90] degrees."""
    difference = abs(angle_a - angle_b) % 180.0
    return min(difference, 180.0 - difference)


def test_compute_btd_is_temperature_difference() -> None:
    """BTD equals bt108 - bt120."""
    bt120 = np.full((4, 4), 240.0)
    bt108 = bt120 + 2.0
    btd = btd_module.compute_btd(bt108, bt120)
    assert btd.shape == (4, 4)
    np.testing.assert_allclose(btd, 2.0)


def test_compute_btd_rejects_shape_mismatch() -> None:
    """Mismatched channel shapes raise ValueError."""
    with pytest.raises(ValueError):
        btd_module.compute_btd(np.zeros((4, 4)), np.zeros((4, 5)))


def test_normalize_btd_constant_field_is_finite_zero() -> None:
    """A constant field normalizes to zero without division-by-zero artifacts."""
    normalized = btd_module.normalize_btd(np.full((30, 30), 240.0))
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(normalized, 0.0)


def test_normalize_btd_amplifies_spike() -> None:
    """A BTD spike stands out after background removal and robust scaling."""
    btd = np.zeros((31, 31))
    btd[15, 15] = 3.0
    normalized = btd_module.normalize_btd(btd)
    assert normalized[15, 15] > 10.0


def test_oriented_line_kernels_properties() -> None:
    """Eight zero-mean, odd-sized kernels with positive cores and negative flanks."""
    kernels = filters.build_oriented_line_kernels(length=11, width=3, num_angles=8)
    assert len(kernels) == 8
    for kernel in kernels:
        assert kernel.shape == (11, 11)
        assert kernel.sum() == pytest.approx(0.0, abs=1e-12)
        assert kernel.max() > 0.0
        assert kernel.min() < 0.0
    assert np.array_equal(kernels[0], np.flip(kernels[0]))  # 0 deg is symmetric


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_clear_sky_no_detections(seed: int) -> None:
    """Uniform 240 K background plus Gaussian noise yields no detections."""
    bt108, bt120 = _background_scene(seed)
    result = filters.detect_contrail_lines(bt108, bt120)
    assert result["num_detected"] == 0
    assert result["objects"] == []
    assert result["max_response_map"].shape == (SCENE_SIZE, SCENE_SIZE)


def test_straight_contrail_detected() -> None:
    """A 20-pixel +2.5 K line is detected with the correct orientation."""
    bt108, bt120 = _background_scene(seed=1)
    _plant_line(bt108, PLANTED_ANGLE_DEG, PLANTED_LINE_LENGTH_PX, LINE_ANOMALY_K)

    result = filters.detect_contrail_lines(bt108, bt120)

    assert result["num_detected"] >= 1
    assert result["num_detected"] == len(result["objects"])
    best = max(result["objects"], key=lambda obj: obj["length_px"])
    assert _angular_distance(best["angle_deg"], PLANTED_ANGLE_DEG) <= 15.0
    assert best["length_px"] >= 8
    assert best["mean_btd"] > 0.2
    assert abs(best["centroid"][0] - SCENE_SIZE / 2.0) < 10.0
    assert abs(best["centroid"][1] - SCENE_SIZE / 2.0) < 10.0


def test_round_blob_rejected_by_aspect_gate() -> None:
    """A round 20x20 +3.0 K blob fails the minimum aspect-ratio gate."""
    bt108, bt120 = _background_scene(seed=2)
    _plant_blob(bt108, BLOB_RADIUS_PX, BLOB_ANOMALY_K)

    result = filters.detect_contrail_lines(bt108, bt120)

    assert result["num_detected"] == 0, result["objects"]
