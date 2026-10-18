"""Flight-contrail association subpackage for Project STRATUM."""

from stratum.match.advect import advect_segment
from stratum.match.label import (
    assign_tristate_label,
    haversine_km,
    match_segment_to_detections,
)
from stratum.match.observe import is_scene_observable
from stratum.match.parallax import calculate_parallax_displacement

__all__ = [
    "advect_segment",
    "calculate_parallax_displacement",
    "is_scene_observable",
    "match_segment_to_detections",
    "assign_tristate_label",
    "haversine_km",
]
