"""Indian flight-corridor generation and route-level contrail screening.

Provides ICAO airport coordinates, great-circle waypoint interpolation at
a prescribed flight level, and a reduction of per-waypoint
Schmidt-Appleman criterion (SAC) evaluations to route-level statistics.

Waypoints carry ambient pressure in Pa (converted from the ISA pressure
in hPa returned by :func:`stratum.physics.thermo.fl_to_p`) so they can be
fed directly to :func:`stratum.physics.sac.evaluate_sac`.
"""

from __future__ import annotations

import math
from typing import Sequence, TypedDict

from stratum.physics import thermo
from stratum.physics.sac import evaluate_sac

__all__ = [
    "AIRPORTS",
    "Waypoint",
    "RouteProfileResult",
    "piecewise_profile",
    "great_circle_waypoints",
    "evaluate_route_profile",
]

#: ICAO location indicator -> (latitude [deg N], longitude [deg E]).
AIRPORTS: dict[str, tuple[float, float]] = {
    "VIDP": (28.5665, 77.1031),  # Delhi, Indira Gandhi International
    "VABB": (19.0896, 72.8656),  # Mumbai, Chhatrapati Shivaji Maharaj International
    "VOBL": (13.1986, 77.7066),  # Bengaluru, Kempegowda International
    "VECC": (22.6547, 88.4467),  # Kolkata, Netaji Subhas Chandra Bose International
    "VAAH": (23.0772, 72.6347),  # Ahmedabad, Sardar Vallabhbhai Patel International
}

_KELVIN_OFFSET: float = 273.15
_HPA_PER_PA: float = 100.0
_RHI_MAX: float = 2.0


class Waypoint(TypedDict):
    """A single corridor waypoint.

    Attributes
    ----------
    lat : float
        Latitude [deg N].
    lon : float
        Longitude [deg E].
    fl : float
        Flight level [hundreds of feet].
    p_Pa : float
        ISA ambient pressure [Pa].
    """

    lat: float
    lon: float
    fl: float
    p_Pa: float


class RouteProfileResult(TypedDict):
    """Route-level summary of per-waypoint SAC evaluations.

    Attributes
    ----------
    total_waypoints : int
        Number of evaluated waypoints.
    formation_count : int
        Waypoints where the SAC holds (contrail forms).
    persistence_count : int
        Waypoints that are ice-supersaturated (RHi > 1).
    formation_pct : float
        ``formation_count`` as a percentage of ``total_waypoints``.
    persistence_pct : float
        ``persistence_count`` as a percentage of ``total_waypoints``.
    details : list of dict
        Per-waypoint SAC dictionaries as returned by
        :func:`stratum.physics.sac.evaluate_sac`.
    """

    total_waypoints: int
    formation_count: int
    persistence_count: int
    formation_pct: float
    persistence_pct: float
    details: list[dict[str, float | bool]]


def piecewise_profile(
    knots: Sequence[tuple[float, float]], num_points: int
) -> list[float]:
    """Sample a piecewise-linear profile given ``(fraction, value)`` knots.

    Parameters
    ----------
    knots : sequence of (float, float)
        Profile control points. Fractions are route positions in [0, 1]
        and must be strictly increasing; values are linearly
        interpolated between knots and clamped outside the knot range.
    num_points : int
        Number of equally spaced samples over [0, 1].

    Returns
    -------
    list of float
        Sampled profile values, starting at the first knot value and
        ending at the last knot value.
    """
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")
    if not knots:
        raise ValueError("knots must contain at least one (fraction, value) pair")
    fractions = [fraction for fraction, _ in knots]
    if any(b <= a for a, b in zip(fractions, fractions[1:])):
        raise ValueError(f"knot fractions must be strictly increasing, got {fractions}")

    values: list[float] = []
    segment = 0
    for i in range(num_points):
        t = i / (num_points - 1)
        while segment < len(knots) - 2 and t > knots[segment + 1][0]:
            segment += 1
        f0, v0 = knots[segment]
        if segment + 1 < len(knots):
            f1, v1 = knots[segment + 1]
        else:
            f1, v1 = f0, v0
        if t <= f0:
            values.append(v0)
        elif t >= f1:
            values.append(v1)
        else:
            values.append(v0 + (v1 - v0) * (t - f0) / (f1 - f0))
    return values


def _unit_vector(lat_rad: float, lon_rad: float) -> tuple[float, float, float]:
    """Return the Cartesian unit vector for spherical angles [rad]."""
    cos_lat = math.cos(lat_rad)
    return (
        cos_lat * math.cos(lon_rad),
        cos_lat * math.sin(lon_rad),
        math.sin(lat_rad),
    )


def _to_latlon(x: float, y: float, z: float) -> tuple[float, float]:
    """Convert a Cartesian direction to (latitude, longitude) [deg]."""
    lat_deg = math.degrees(math.atan2(z, math.sqrt(x * x + y * y)))
    lon_deg = math.degrees(math.atan2(y, x))
    return lat_deg, lon_deg


def great_circle_waypoints(
    coord1: tuple[float, float],
    coord2: tuple[float, float],
    num_points: int = 25,
    fl: float = 340,
) -> list[Waypoint]:
    """Interpolate ``num_points`` great-circle waypoints between two points.

    Endpoints are reproduced exactly; intermediate points use spherical
    linear interpolation (slerp) of the unit direction vectors. Every
    waypoint carries the ISA pressure for flight level ``fl``.

    Parameters
    ----------
    coord1, coord2 : (float, float)
        Start and end coordinates as (latitude [deg N], longitude [deg E]).
    num_points : int, optional
        Total number of waypoints including both endpoints, by default 25.
    fl : float, optional
        Flight level in hundreds of feet, by default 340.

    Returns
    -------
    list of Waypoint
        ``num_points`` dictionaries with keys ``lat``, ``lon``, ``fl`` and
        ``p_Pa`` (ISA pressure [Pa]).
    """
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}")

    lat1_rad, lon1_rad = math.radians(coord1[0]), math.radians(coord1[1])
    lat2_rad, lon2_rad = math.radians(coord2[0]), math.radians(coord2[1])
    v1 = _unit_vector(lat1_rad, lon1_rad)
    v2 = _unit_vector(lat2_rad, lon2_rad)

    dot = min(1.0, max(-1.0, sum(a * b for a, b in zip(v1, v2))))
    omega = math.acos(dot)
    p_pa = thermo.fl_to_p(fl) * _HPA_PER_PA

    waypoints: list[Waypoint] = []
    for i in range(num_points):
        if i == 0:
            lat_deg, lon_deg = coord1[0], coord1[1]
        elif i == num_points - 1:
            lat_deg, lon_deg = coord2[0], coord2[1]
        else:
            t = i / (num_points - 1)
            if omega < 1e-12:
                x = (1.0 - t) * v1[0] + t * v2[0]
                y = (1.0 - t) * v1[1] + t * v2[1]
                z = (1.0 - t) * v1[2] + t * v2[2]
            else:
                scale = math.sin(omega)
                a = math.sin((1.0 - t) * omega) / scale
                b = math.sin(t * omega) / scale
                x = a * v1[0] + b * v2[0]
                y = a * v1[1] + b * v2[1]
                z = a * v1[2] + b * v2[2]
            norm = math.sqrt(x * x + y * y + z * z)
            lat_deg, lon_deg = _to_latlon(x / norm, y / norm, z / norm)
        waypoints.append(
            {"lat": lat_deg, "lon": lon_deg, "fl": float(fl), "p_Pa": p_pa}
        )
    return waypoints


def evaluate_route_profile(
    waypoints: Sequence[Waypoint],
    temp_c_list: Sequence[float],
    rhi_list: Sequence[float],
    eta: float = 0.30,
) -> RouteProfileResult:
    """Evaluate the Schmidt-Appleman criterion along a route profile.

    Parameters
    ----------
    waypoints : sequence of Waypoint
        Waypoints from :func:`great_circle_waypoints`.
    temp_c_list : sequence of float
        Ambient temperatures [deg C], one per waypoint.
    rhi_list : sequence of float
        Ambient relative humidities over ice [-], one per waypoint,
        each within [0.0, 2.0].
    eta : float, optional
        Engine efficiency (0-1), by default 0.30.

    Returns
    -------
    RouteProfileResult
        Route-level counts, percentages and per-waypoint SAC details.

    Raises
    ------
    ValueError
        If the profile list lengths differ from the waypoint count or a
        RHi value lies outside [0.0, 2.0].
    """
    total = len(waypoints)
    if len(temp_c_list) != total or len(rhi_list) != total:
        raise ValueError(
            "profile lengths must match waypoint count: "
            f"{total} waypoints, {len(temp_c_list)} temperatures, "
            f"{len(rhi_list)} RHi values"
        )

    details: list[dict[str, float | bool]] = []
    formation_count = 0
    persistence_count = 0
    for waypoint, temp_c, rhi in zip(waypoints, temp_c_list, rhi_list):
        if not 0.0 <= rhi <= _RHI_MAX:
            raise ValueError(f"RHi must lie in [0.0, {_RHI_MAX}], got {rhi!r}")
        detail = evaluate_sac(
            float(waypoint["p_Pa"]), float(temp_c) + _KELVIN_OFFSET, float(rhi), eta=eta
        )
        details.append(detail)
        if detail["forms"]:
            formation_count += 1
        if detail["persists"]:
            persistence_count += 1

    return {
        "total_waypoints": total,
        "formation_count": formation_count,
        "persistence_count": persistence_count,
        "formation_pct": 100.0 * formation_count / total,
        "persistence_pct": 100.0 * persistence_count / total,
        "details": details,
    }
