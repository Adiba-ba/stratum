"""Geostationary imager navigation on an oblate ellipsoid.

The equations are the "ABI fixed grid" navigation equations of the GOES-R
Series Product Definition and Users' Guide (PUG), Volume 3: Level 1b
Products, paragraph 5.1.2.8 "Navigation of Image Data": 5.1.2.8.1 for the
fixed grid to geodetic direction and 5.1.2.8.2 for the geodetic to fixed
grid direction.

    https://www.ospo.noaa.gov/resources/documents/PUG/GS%20Series%20416-R-PUG-L1B-0347%20Vol%203%20Rev%202.3.pdf

The PUG defines the fixed grid through the E/W scanning angle ``x`` and the
N/S elevation angle ``y`` (both in radians), the perspective point height
``H = perspective_point_height + semi_major_axis`` and the GRS80 semi-axes.
Two sweep geometries are supported:

* ``sweep_angle_axis="x"`` -- the GOES ABI. The N/S elevation angle is
  ``arctan(sz / sx)`` and the E/W scanning angle is ``arcsin(-sy / r)``,
  with ``s = (sx, sy, sz)`` the vector from the point to the satellite.
* ``sweep_angle_axis="y"`` -- the Meteosat SEVIRI convention, where the
  two angles exchange roles: ``x = arctan(-sy / sx)`` and
  ``y = arcsin(sz / r)``.

Both conventions match the PROJ ``geos`` projection (``+sweep=x`` and
``+sweep=y``); the test suite cross-validates against ``pyproj`` when it is
installed, and reproduces the numerical example printed in PUG paragraph
5.1.2.8.1. Points that are not visible from the satellite yield NaN instead
of raising.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["GeostationaryNavigation", "from_goes_attrs"]

#: GRS80 semi-major axis [m], the GOES PUG earth model.
GRS80_SEMI_MAJOR_M: float = 6378137.0
#: GRS80 semi-minor axis [m] (= (1 - f) * semi-major, f = 1 / 298.257222096).
GRS80_SEMI_MINOR_M: float = 6356752.31414
#: Attribute names carrying the satellite height above the ellipsoid.
_HEIGHT_ATTRS: tuple[str, ...] = ("perspective_point_height", "satellite_height")


class GeostationaryNavigation:
    """Fixed-grid navigation for a geostationary imager.

    Parameters
    ----------
    satellite_height_m : float
        Satellite height above the ellipsoid [m]; the
        ``perspective_point_height`` attribute of a GOES product.
    longitude_of_projection_deg : float
        Sub-satellite longitude [deg E].
    sweep_angle_axis : {"x", "y"}
        Axis about which the instrument sweep is measured; ``"x"`` for the
        GOES ABI fixed grid, ``"y"`` for Meteosat SEVIRI style products.
    semi_major_m, semi_minor_m : float
        Ellipsoid semi-axes [m]. Defaults are the GRS80 values of the PUG.

    Raises
    ------
    ValueError
        If ``sweep_angle_axis`` is neither ``"x"`` nor ``"y"``, or if
        ``satellite_height_m`` is not positive.
    """

    def __init__(
        self,
        satellite_height_m: float,
        longitude_of_projection_deg: float,
        sweep_angle_axis: str = "x",
        semi_major_m: float = GRS80_SEMI_MAJOR_M,
        semi_minor_m: float = GRS80_SEMI_MINOR_M,
    ) -> None:
        if isinstance(sweep_angle_axis, bytes):
            sweep_angle_axis = sweep_angle_axis.decode()
        if sweep_angle_axis not in ("x", "y"):
            raise ValueError(
                f"sweep_angle_axis must be 'x' or 'y', got {sweep_angle_axis!r}"
            )
        if not satellite_height_m > 0.0:
            raise ValueError(
                f"satellite_height_m must be positive, got {satellite_height_m!r}"
            )
        self.satellite_height_m = float(satellite_height_m)
        self.longitude_of_projection_deg = float(longitude_of_projection_deg)
        self.sweep_angle_axis = str(sweep_angle_axis)
        self.semi_major_m = float(semi_major_m)
        self.semi_minor_m = float(semi_minor_m)
        #: Satellite distance from the earth centre H [m] (PUG paragraph 5.1.2.8).
        self._perspective_point_m = self.satellite_height_m + self.semi_major_m
        self._axis_ratio_sq = (self.semi_major_m / self.semi_minor_m) ** 2
        self._eccentricity_sq = (
            self.semi_major_m**2 - self.semi_minor_m**2
        ) / self.semi_major_m**2

    def scan_angles_to_latlon(
        self, x_rad: Any, y_rad: Any
    ) -> tuple[Any, Any]:
        """Geodetic coordinates of the fixed grid point ``(x, y)``.

        PUG paragraph 5.1.2.8.1: the ray from the satellite along the scan
        angles is intersected with the ellipsoid and the near-side solution
        is converted to geodetic latitude and longitude.

        Parameters
        ----------
        x_rad, y_rad : float or numpy.ndarray
            Fixed grid E/W scanning angle and N/S elevation angle [rad].

        Returns
        -------
        tuple
            ``(lat_deg, lon_deg)`` [deg N, deg E], NaN where the ray misses
            the earth (the point is not visible from the satellite). Floats
            are returned for scalar input, arrays for array input.
        """
        x = np.asarray(x_rad, dtype=float)
        y = np.asarray(y_rad, dtype=float)
        h = self._perspective_point_m
        k = self._axis_ratio_sq
        cos_x, sin_x = np.cos(x), np.sin(x)
        cos_y, sin_y = np.cos(y), np.sin(y)

        if self.sweep_angle_axis == "x":
            quad_a = sin_x**2 + cos_x**2 * (cos_y**2 + k * sin_y**2)
        else:
            quad_a = cos_y**2 + k * sin_y**2
        quad_b = -2.0 * h * cos_x * cos_y
        quad_c = h**2 - self.semi_major_m**2
        discriminant = quad_b**2 - 4.0 * quad_a * quad_c
        # Ray/ellipsoid intersection distance; the minus root is the
        # near-side crossing (PUG "rs").
        distance = (
            -quad_b - np.sqrt(np.where(discriminant >= 0.0, discriminant, np.nan))
        ) / (2.0 * quad_a)

        # PUG s-vector and the corresponding point P = (H - sx, -sy, sz)
        # in the satellite frame, whose centre is the satellite and whose
        # first axis points at the earth centre.
        s_x = distance * cos_x * cos_y
        if self.sweep_angle_axis == "x":
            s_y = -distance * sin_x
            s_z = distance * cos_x * sin_y
        else:
            s_y = -distance * sin_x * cos_y
            s_z = distance * sin_y
        p_x = h - s_x
        p_y = -s_y
        p_z = s_z

        lat = np.degrees(np.arctan(k * p_z / np.hypot(p_x, p_y)))
        lon = self.longitude_of_projection_deg + np.degrees(np.arctan2(p_y, p_x))
        return _as_output(lat, lon, scalar=(x.ndim == 0 and y.ndim == 0))

    def latlon_to_scan_angles(
        self, lat_deg: Any, lon_deg: Any
    ) -> tuple[Any, Any]:
        """Fixed grid scan angles of the geodetic point ``(lat, lon)``.

        PUG paragraph 5.1.2.8.2: the ellipsoid point is expressed in the
        satellite frame and the scan angles are recovered from the vector
        to the point.

        Parameters
        ----------
        lat_deg, lon_deg : float or numpy.ndarray
            GRS80 geodetic latitude and longitude [deg N, deg E].

        Returns
        -------
        tuple
            ``(x_rad, y_rad)`` [rad], NaN where the point is not visible
            from the satellite. Floats are returned for scalar input,
            arrays for array input.
        """
        lat = np.asarray(lat_deg, dtype=float)
        lon = np.asarray(lon_deg, dtype=float)
        h = self._perspective_point_m
        k = self._axis_ratio_sq
        lam = np.radians(lon - self.longitude_of_projection_deg)

        # PUG: geocentric latitude and geocentric distance of the point.
        phi_c = np.arctan(np.tan(np.radians(lat)) / k)
        cos_phi_c = np.cos(phi_c)
        r_c = self.semi_minor_m / np.sqrt(
            1.0 - self._eccentricity_sq * cos_phi_c**2
        )
        s_x = h - r_c * cos_phi_c * np.cos(lam)
        s_y = -r_c * cos_phi_c * np.sin(lam)
        s_z = r_c * np.sin(phi_c)
        distance = np.sqrt(s_x**2 + s_y**2 + s_z**2)

        # PUG visibility test: the point lies on the far side of the earth
        # when this quadratic form in the satellite-to-point vector is not
        # positive.
        visible = s_x * (h - s_x) - s_y**2 - k * s_z**2 > 0.0
        if self.sweep_angle_axis == "x":
            x = np.arcsin(np.clip(-s_y / distance, -1.0, 1.0))
            y = np.arctan(s_z / s_x)
        else:
            x = np.arctan(-s_y / s_x)
            y = np.arcsin(np.clip(s_z / distance, -1.0, 1.0))
        x = np.where(visible, x, np.nan)
        y = np.where(visible, y, np.nan)
        return _as_output(x, y, scalar=(lat.ndim == 0 and lon.ndim == 0))


def from_goes_attrs(attrs: Mapping[str, Any]) -> GeostationaryNavigation:
    """Navigation object from a GOES ``goes_imager_projection`` attribute dict.

    Parameters
    ----------
    attrs : Mapping
        Attributes of the netCDF ``goes_imager_projection`` variable, or an
        equivalent global-attribute mapping: ``perspective_point_height``
        (or ``satellite_height`` as a fallback), ``longitude_of_projection_origin``,
        ``sweep_angle_axis`` and the optional ellipsoid axes
        ``semi_major_axis`` / ``semi_minor_axis``.

    Returns
    -------
    GeostationaryNavigation

    Raises
    ------
    ValueError
        If the satellite height or the projection longitude is missing or
        unusable, or the sweep axis is unknown.
    """
    height: float | None = None
    for key in _HEIGHT_ATTRS:
        if key in attrs:
            try:
                height = float(attrs[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"attribute {key!r} is not a number: {attrs[key]!r}"
                ) from exc
            break
    if height is None:
        raise ValueError(
            "missing satellite height attribute; expected one of "
            f"{', '.join(_HEIGHT_ATTRS)}"
        )
    try:
        longitude = float(attrs["longitude_of_projection_origin"])
    except KeyError as exc:
        raise ValueError(
            "missing longitude_of_projection_origin attribute"
        ) from exc

    sweep = attrs.get("sweep_angle_axis", "x")
    if isinstance(sweep, bytes):
        sweep = sweep.decode()
    return GeostationaryNavigation(
        satellite_height_m=height,
        longitude_of_projection_deg=longitude,
        sweep_angle_axis=str(sweep),
        semi_major_m=float(attrs.get("semi_major_axis", GRS80_SEMI_MAJOR_M)),
        semi_minor_m=float(attrs.get("semi_minor_axis", GRS80_SEMI_MINOR_M)),
    )


def _as_output(
    first: np.ndarray, second: np.ndarray, scalar: bool
) -> tuple[Any, Any]:
    """Return floats for scalar input, arrays otherwise."""
    if scalar:
        return float(first), float(second)
    return first, second
