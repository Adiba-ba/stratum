"""Route-generation subpackage for Project STRATUM."""

from stratum.routes.corridor import (
    AIRPORTS,
    RouteProfileResult,
    Waypoint,
    evaluate_route_profile,
    great_circle_waypoints,
    piecewise_profile,
)

__all__ = [
    "AIRPORTS",
    "Waypoint",
    "RouteProfileResult",
    "piecewise_profile",
    "great_circle_waypoints",
    "evaluate_route_profile",
]
