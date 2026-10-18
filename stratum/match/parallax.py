"""Meteosat-9 parallax displacement for elevated targets.

A geostationary satellite images a target at altitude along a slanted
line of sight. The satellite navigation usually geolocates the pixel at
the Earth ellipsoid surface, so the apparent ground position is
displaced away from the sub-satellite point relative to the true
sub-target position. The displacement is roughly ``alt_m * tan(zeta)``,
where ``zeta`` is the satellite zenith angle at the target.

The ray from the satellite through the target is intersected with a
spherical Earth; the near-side entry point (just beyond the target) is
the apparent surface location. Coordinates are geocentric on a sphere.
"""

from __future__ import annotations

import math

__all__ = [
    "calculate_parallax_displacement",
    "EARTH_RADIUS_M",
    "METEOSAT9_ALTITUDE_M",
    "METEOSAT9_LONGITUDE_DEG",
]

#: Spherical Earth radius used by the parallax model [m].
EARTH_RADIUS_M: float = 6_371_000.0
#: Meteosat-9 geostationary altitude above the surface [m].
METEOSAT9_ALTITUDE_M: float = 35_786_000.0
#: Meteosat-9 nominal sub-satellite longitude [deg E].
METEOSAT9_LONGITUDE_DEG: float = 45.5
#: Meteosat-9 sub-satellite latitude [deg N].
METEOSAT9_LATITUDE_DEG: float = 0.0


def _to_cartesian(
    lat_deg: float, lon_deg: float, radius_m: float
) -> tuple[float, float, float]:
    """Convert geocentric (lat, lon, radius) to Cartesian ECEF [m]."""
    lat_rad, lon_rad = math.radians(lat_deg), math.radians(lon_deg)
    cos_lat = math.cos(lat_rad)
    return (
        radius_m * cos_lat * math.cos(lon_rad),
        radius_m * cos_lat * math.sin(lon_rad),
        radius_m * math.sin(lat_rad),
    )


def _to_geodetic(x: float, y: float, z: float) -> tuple[float, float]:
    """Convert Cartesian ECEF [m] to geocentric (lat, lon) [deg]."""
    radius = math.sqrt(x * x + y * y + z * z)
    lat_deg = math.degrees(math.asin(z / radius))
    lon_deg = math.degrees(math.atan2(y, x))
    return lat_deg, lon_deg


def calculate_parallax_displacement(
    lat: float, lon: float, alt_m: float
) -> tuple[float, float]:
    """Apparent Meteosat-9 displacement of an elevated target.

    Parameters
    ----------
    lat, lon : float
        True target coordinates [deg N, deg E].
    alt_m : float
        Target altitude above the spherical Earth [m].

    Returns
    -------
    (float, float)
        Apparent minus true displacement ``(delta_lat_deg, delta_lon_deg)``,
        pointing away from the sub-satellite point (0 deg N, 45.5 deg E).
        Add it to the true coordinate to obtain the location that the
        satellite image reports.
    """
    satellite = _to_cartesian(
        METEOSAT9_LATITUDE_DEG,
        METEOSAT9_LONGITUDE_DEG,
        EARTH_RADIUS_M + METEOSAT9_ALTITUDE_M,
    )
    target = _to_cartesian(lat, lon, EARTH_RADIUS_M + alt_m)
    direction = tuple(t - s for t, s in zip(target, satellite))

    # Solve |satellite + t * direction|^2 = EARTH_RADIUS_M^2 for the
    # surface intersection beyond the target.
    a = sum(component * component for component in direction)
    b = 2.0 * sum(s * d for s, d in zip(satellite, direction))
    c = sum(s * s for s in satellite) - EARTH_RADIUS_M**2
    discriminant = math.sqrt(b * b - 4.0 * a * c)
    # The near-side entry root (minus sign) is the first surface crossing
    # beyond the target; the plus root is the exit point on the far side
    # of the planet.
    t = (-b - discriminant) / (2.0 * a)
    surface = tuple(s + t * d for s, d in zip(satellite, direction))

    apparent_lat, apparent_lon = _to_geodetic(*surface)
    return apparent_lat - lat, apparent_lon - lon
