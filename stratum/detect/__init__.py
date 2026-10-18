"""Contrail-detection subpackage for Project STRATUM."""

from stratum.detect.btd import compute_btd, normalize_btd
from stratum.detect.filters import (
    DetectedObject,
    DetectionResult,
    build_oriented_line_kernels,
    detect_contrail_lines,
)

__all__ = [
    "compute_btd",
    "normalize_btd",
    "build_oriented_line_kernels",
    "detect_contrail_lines",
    "DetectedObject",
    "DetectionResult",
]
