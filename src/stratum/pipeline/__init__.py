"""End-to-end pipeline package (Step 8).

Assembles the canonical feature matrix along cleaned ADS-B trajectories
and evaluates a full flight corridor against the calibrated persistence
model plus the counterfactual decision layer.
"""

from __future__ import annotations

from stratum.pipeline.build_features import (
    CLOUD_ICE_COVER_THRESHOLD,
    HUMIDITY_OFFSET_HPA,
    TROPOPAUSE_ALTITUDE_M,
    extract_features_for_point,
    extract_features_for_trajectory,
    pressure_to_altitude_m,
    specific_humidity_to_rhi,
)
from stratum.pipeline.runner import (
    PERSISTENT_PROBABILITY_THRESHOLD,
    evaluate_corridor_flight,
)

__all__ = [
    "CLOUD_ICE_COVER_THRESHOLD",
    "HUMIDITY_OFFSET_HPA",
    "PERSISTENT_PROBABILITY_THRESHOLD",
    "TROPOPAUSE_ALTITUDE_M",
    "evaluate_corridor_flight",
    "extract_features_for_point",
    "extract_features_for_trajectory",
    "pressure_to_altitude_m",
    "specific_humidity_to_rhi",
]
