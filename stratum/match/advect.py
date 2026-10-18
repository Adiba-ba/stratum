"""Wind advection of flight-segment coordinates on a spherical Earth."""

from __future__ import annotations

import math

__all__ = ["advect_segment", "EARTH_RADIUS_M"]

#: Spherical Earth radius used by the advection model [m].
EARTH_RADIUS_M: float = 6_371_000.0


def advect_segment(
    lat: float,
    lon: float,
    u_wind_mps: float,
    v_wind_mps: float,
    elapsed_seconds: float,
) -> tuple[float, float]:
    """Drift a coordinate forward with horizontal wind components.

    Uses a first-order local-tangent-plane approximation on a sphere:
    ``d_lat = v * t / R`` and ``d_lon = u * t / (R * cos(lat))``.

    Parameters
    ----------
    lat, lon : float
        Start coordinates [deg N, deg E].
    u_wind_mps : float
        Eastward wind component [m/s] (positive = toward east).
    v_wind_mps : float
        Northward wind component [m/s] (positive = toward north).
    elapsed_seconds : float
        Drift time [s], non-negative.

    Returns
    -------
    (float, float)
        Advected ``(lat, lon)`` [deg N, deg E]; longitude is wrapped to
        [-180, 180).

    Raises
    ------
    ValueError
        If ``elapsed_seconds`` is negative or the start latitude is too
        close to a pole for the local approximation.
    """
    if elapsed_seconds < 0.0:
        raise ValueError(f"elapsed_seconds must be non-negative, got {elapsed_seconds}")

    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-12:
        raise ValueError(f"latitude {lat} is too close to a pole for advection")

    new_lat = lat + math.degrees(v_wind_mps * elapsed_seconds / EARTH_RADIUS_M)
    new_lon = lon + math.degrees(
        u_wind_mps * elapsed_seconds / (EARTH_RADIUS_M * cos_lat)
    )
    wrapped_lon = (new_lon + 180.0) % 360.0 - 180.0
    return new_lat, wrapped_lon
