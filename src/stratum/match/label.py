"""Flight-contrail association and tri-state labeling.

A flight segment (after wind advection and parallax correction) is
matched against satellite detections by centroid distance and line
orientation. The outcome is encoded as a tri-state label:

- ``1.0``  positive: a detection is associated with the segment,
- ``0.0``  observable non-detection: the scene was observed in at least
  two satellite slots and no line was found,
- ``nan``  unknown/unobservable: too few observable slots, so no
  statement can be made. NaN is never coerced to 0.0.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

__all__ = [
    "match_segment_to_detections",
    "assign_tristate_label",
    "haversine_km",
    "EARTH_RADIUS_M",
]

#: Spherical Earth radius used for great-circle distances [m].
EARTH_RADIUS_M: float = 6_371_000.0


def haversine_km(
    coord1: tuple[float, float], coord2: tuple[float, float]
) -> float:
    """Great-circle distance between two (lat, lon) coordinates [km]."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a))) / 1000.0


def _angle_difference_deg(angle_a: float, angle_b: float) -> float:
    """Smallest orientation difference in [0, 90] degrees (mod 180)."""
    difference = abs(angle_a - angle_b) % 180.0
    return min(difference, 180.0 - difference)


def match_segment_to_detections(
    advected_parallax_coord: tuple[float, float],
    segment_heading_deg: float,
    detected_objects: Sequence[Mapping[str, Any]],
    distance_tolerance_km: float = 10.0,
    angle_tolerance_deg: float = 20.0,
) -> bool:
    """Match an advected, parallax-corrected segment to detections.

    A match requires at least one detected object whose centroid lies
    within ``distance_tolerance_km`` of the segment coordinate and whose
    orientation agrees with ``segment_heading_deg`` within
    ``angle_tolerance_deg`` (mod 180 degrees).

    Parameters
    ----------
    advected_parallax_coord : (float, float)
        Segment coordinate after wind advection and parallax correction,
        as ``(lat, lon)`` [deg].
    segment_heading_deg : float
        Segment flight heading [deg] (any 360-degree convention).
    detected_objects : sequence of mapping
        Detections, each with keys ``centroid`` (``(lat, lon)`` [deg])
        and ``angle_deg`` (line orientation in any 180-degree
        convention).
    distance_tolerance_km : float, optional
        Maximum centroid distance for a match [km], by default 10.0.
    angle_tolerance_deg : float, optional
        Maximum orientation difference for a match [deg], by default 20.0.

    Returns
    -------
    bool
        True if any detection satisfies both tolerances.
    """
    for obj in detected_objects:
        centroid = obj["centroid"]
        object_lat = float(centroid[0])
        object_lon = float(centroid[1])
        if (
            haversine_km(advected_parallax_coord, (object_lat, object_lon))
            > distance_tolerance_km
        ):
            continue
        object_angle = float(obj["angle_deg"])
        if (
            _angle_difference_deg(object_angle, segment_heading_deg)
            <= angle_tolerance_deg
        ):
            return True
    return False


def assign_tristate_label(matched: bool, observable_slots: int) -> float:
    """Assign the tri-state flight-contrail label.

    Parameters
    ----------
    matched : bool
        True when a satellite detection was associated with the segment.
    observable_slots : int
        Number of satellite slots in which the scene passed the
        observability gate.

    Returns
    -------
    float
        ``1.0`` when matched; ``0.0`` when not matched but the scene was
        observable in at least two slots (a genuine observable
        non-detection); otherwise ``nan`` (unknown/unobservable).
        NaN is never coerced to 0.0.
    """
    if matched:
        return 1.0
    if observable_slots >= 2:
        return 0.0
    return float("nan")
