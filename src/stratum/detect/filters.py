"""Oriented matched filters and line extraction for contrail detection.

Builds elongated zero-mean line kernels at discrete orientations,
convolves the normalized BTD field with each kernel, keeps the maximum
response per pixel, applies hysteresis thresholding, and extracts
connected line objects that pass length and aspect-ratio gates.

Angles are measured counter-clockwise from the column (x) axis of the
array, with rows representing the y axis. All reported angles lie in
[0, 180).
"""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
from scipy import ndimage

from stratum.detect.btd import compute_btd, normalize_btd

__all__ = [
    "DetectedObject",
    "DetectionResult",
    "build_oriented_line_kernels",
    "detect_contrail_lines",
]

#: Hysteresis weak-threshold as a fraction of the primary threshold.
HYSTERESIS_WEAK_FRACTION: float = 0.7
#: Minimum object aspect ratio (length / width) accepted as a line.
MIN_ASPECT_RATIO: float = 3.0
#: Negative flank width of each kernel, in pixels.
FLANK_WIDTH_PX: int = 1
#: 8-connectivity structure for connected-component labelling.
_CONNECTIVITY: np.ndarray = np.ones((3, 3), dtype=bool)


class DetectedObject(TypedDict):
    """A detected contrail-like line object.

    Attributes
    ----------
    centroid : (float, float)
        Mean (row, col) position of the object mask [pixels].
    length_px : float
        Extent along the principal axis [pixels].
    angle_deg : float
        Principal-axis orientation in [0, 180) [deg].
    mean_btd : float
        Mean raw BTD of the object mask [K].
    """

    centroid: tuple[float, float]
    length_px: float
    angle_deg: float
    mean_btd: float


class DetectionResult(TypedDict):
    """Result of :func:`detect_contrail_lines`.

    Attributes
    ----------
    num_detected : int
        Number of accepted line objects.
    objects : list of DetectedObject
        Accepted line objects.
    max_response_map : np.ndarray
        Per-pixel maximum matched-filter response [robust sigma units].
    """

    num_detected: int
    objects: list[DetectedObject]
    max_response_map: np.ndarray


def build_oriented_line_kernels(
    length: int = 11, width: int = 3, num_angles: int = 8
) -> list[np.ndarray]:
    """Build elongated zero-mean matched-filter kernels.

    Each kernel has a positive line core of the requested ``width`` and
    negative flanks (``FLANK_WIDTH_PX`` pixels each side) along a line of
    the requested ``length``. Core and flank weights are normalized by
    their pixel counts so that every kernel sums to zero, making the
    response vanish on any uniform background.

    Parameters
    ----------
    length : int, optional
        Line length in pixels, by default 11. Rounded up to an odd
        support size so a pixel centre exists.
    width : int, optional
        Positive core width in pixels, by default 3.
    num_angles : int, optional
        Number of orientations spanning [0, 180), by default 8.

    Returns
    -------
    list of np.ndarray
        ``num_angles`` kernels of shape ``(support, support)``.
    """
    if length < 3:
        raise ValueError(f"length must be >= 3, got {length}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    if num_angles < 1:
        raise ValueError(f"num_angles must be >= 1, got {num_angles}")

    support = length if length % 2 == 1 else length + 1
    half_length = length / 2.0
    half_width = width / 2.0
    grid_y, grid_x = np.mgrid[0:support, 0:support] - support // 2

    kernels: list[np.ndarray] = []
    for index in range(num_angles):
        theta = math.pi * index / num_angles
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        along = grid_x * cos_t + grid_y * sin_t
        perpendicular = -grid_x * sin_t + grid_y * cos_t

        core = (np.abs(perpendicular) <= half_width) & (np.abs(along) <= half_length)
        flank = (
            (np.abs(perpendicular) > half_width)
            & (np.abs(perpendicular) <= half_width + FLANK_WIDTH_PX)
            & (np.abs(along) <= half_length)
        )
        if not core.any() or not flank.any():
            raise ValueError(
                f"kernel geometry is degenerate for index {index} "
                f"(length={length}, width={width})"
            )

        kernel = np.zeros((support, support), dtype=float)
        kernel[core] = 1.0 / float(core.sum())
        kernel[flank] -= 1.0 / float(flank.sum())
        kernels.append(kernel)
    return kernels


def _object_from_mask(
    mask: np.ndarray, btd: np.ndarray, min_length_px: int
) -> DetectedObject | None:
    """Build a DetectedObject from a component mask, applying shape gates.

    Returns ``None`` when the component fails the minimum length or
    aspect-ratio gate.
    """
    coords = np.argwhere(mask)
    if coords.shape[0] < 2:
        return None

    centre = coords.mean(axis=0)
    centred = coords - centre
    covariance = centred.T @ centred / float(len(centred))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_axis = eigenvectors[:, -1]
    minor_axis = eigenvectors[:, 0]

    along = centred @ principal_axis
    length_px = float(along.max() - along.min() + 1.0)
    # 4 * sigma of the minor axis approximates the object width for a
    # uniform bar (sqrt(12) / 4 = 1.155 * width) and is robust to lone
    # outlier pixels inflating a raw extent.
    width_px = max(4.0 * math.sqrt(max(float(eigenvalues[0]), 0.0)), 1.0)

    if length_px < min_length_px or length_px / width_px < MIN_ASPECT_RATIO:
        return None

    angle_deg = math.degrees(math.atan2(principal_axis[0], principal_axis[1])) % 180.0
    return {
        "centroid": (float(centre[0]), float(centre[1])),
        "length_px": length_px,
        "angle_deg": angle_deg,
        "mean_btd": float(btd[mask].mean()),
    }


def detect_contrail_lines(
    bt108: np.ndarray, bt120: np.ndarray, threshold: float = 2.0, min_length_px: int = 8
) -> DetectionResult:
    """Detect contrail-like lines in a split-window scene.

    Pipeline: compute and normalize BTD, convolve with the eight oriented
    line kernels while retaining the maximum response and best angle per
    pixel, apply hysteresis thresholding (weak = ``0.7 * threshold``),
    then label connected components and keep objects whose principal
    length is at least ``min_length_px`` and whose aspect ratio is at
    least ``MIN_ASPECT_RATIO``.

    Parameters
    ----------
    bt108 : np.ndarray
        Brightness temperature at 10.8 um [K].
    bt120 : np.ndarray
        Brightness temperature at 12.0 um [K], same shape as ``bt108``.
    threshold : float, optional
        Primary response threshold [robust sigma units], by default 2.0.
    min_length_px : int, optional
        Minimum accepted object length in pixels, by default 8.

    Returns
    -------
    DetectionResult
        ``num_detected``, per-object descriptors and the maximum-response
        map.
    """
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if min_length_px < 1:
        raise ValueError(f"min_length_px must be >= 1, got {min_length_px}")

    btd = compute_btd(bt108, bt120)
    normalized = normalize_btd(btd)

    response = np.full(btd.shape, -np.inf, dtype=float)
    angle_index = np.zeros(btd.shape, dtype=int)
    for index, kernel in enumerate(build_oriented_line_kernels()):
        kernel_response = ndimage.convolve(normalized, kernel, mode="reflect")
        better = kernel_response > response
        response = np.where(better, kernel_response, response)
        angle_index = np.where(better, index, angle_index)

    strong = response >= threshold
    weak = response >= HYSTERESIS_WEAK_FRACTION * threshold
    labels, num_labels = ndimage.label(weak, structure=_CONNECTIVITY)

    objects: list[DetectedObject] = []
    if num_labels:
        strong_labels = np.unique(labels[strong])
        for label in strong_labels:
            if label == 0:
                continue
            obj = _object_from_mask(labels == label, btd, min_length_px)
            if obj is not None:
                objects.append(obj)

    return {
        "num_detected": len(objects),
        "objects": objects,
        "max_response_map": response,
    }
