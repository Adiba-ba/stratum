"""Image-space geometry helpers: detection headings and coordinate lookups.

A detection is a line segment in image index space, where row indices
increase downward (southward) and column indices increase rightward
(eastward). Detectors report the segment angle as
``atan2(delta_row, delta_col)`` in degrees, which is an image-space angle,
not a compass heading. Converting it to a heading requires the map
geometry, so the endpoints of the segment are georeferenced and the
great-circle initial bearing between them is returned.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = [
    "ArrayGeoreferencer",
    "initial_bearing_deg",
    "pixel_angle_to_heading_deg",
]

LatLon = tuple[float, float]
ToLatLon = Callable[[float, float], LatLon]


def initial_bearing_deg(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle initial bearing from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Coordinates of the two points [deg N, deg E].

    Returns
    -------
    float
        Compass bearing [deg]: 0 is north, 90 is east, increasing clockwise.
        NaN if both points coincide.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(
        delta_lambda
    )
    if x == 0.0 and y == 0.0:
        return math.nan
    return math.degrees(math.atan2(y, x)) % 360.0


def pixel_angle_to_heading_deg(
    angle_px_deg: float,
    row: float,
    col: float,
    to_latlon: ToLatLon,
    step_px: float = 5.0,
) -> float:
    """Compass heading of a detection line reported in image index space.

    The detection angle follows the ``atan2(delta_row, delta_col)``
    convention: 0 deg points east along increasing columns, 90 deg points
    south along increasing rows, so 270 deg points north on a north-up
    image. Two points are sampled at ``+/- step_px`` along the line and
    georeferenced; the heading is the initial bearing from the negative to
    the positive end.

    Parameters
    ----------
    angle_px_deg : float
        Detection angle [deg] from ``atan2(delta_row, delta_col)``.
    row, col : float
        Central detection position in image index coordinates.
    to_latlon : callable
        ``to_latlon(row, col) -> (lat_deg, lon_deg)``, for example the
        method of an :class:`ArrayGeoreferencer`. It must return NaN for
        positions outside the image.
    step_px : float, optional
        Half-length of the sampled segment [pixel]. Defaults to 5.

    Returns
    -------
    float
        Compass heading [deg] in ``[0, 360)``, or NaN when either sampled
        endpoint is outside the georeferenced area or the two endpoints
        coincide.

    Raises
    ------
    ValueError
        If ``step_px`` is not positive.
    """
    if step_px <= 0.0:
        raise ValueError(f"step_px must be positive, got {step_px!r}")
    angle_rad = math.radians(angle_px_deg)
    delta_row = math.sin(angle_rad) * step_px
    delta_col = math.cos(angle_rad) * step_px
    tail_lat, tail_lon = to_latlon(row - delta_row, col - delta_col)
    head_lat, head_lon = to_latlon(row + delta_row, col + delta_col)
    if not all(math.isfinite(value) for value in (tail_lat, tail_lon, head_lat, head_lon)):
        return math.nan
    return initial_bearing_deg(tail_lat, tail_lon, head_lat, head_lon)


class ArrayGeoreferencer:
    """Bilinear georeferencing of a 2-D latitude/longitude array pair.

    Parameters
    ----------
    lats_deg, lons_deg : numpy.ndarray
        Two-dimensional coordinate arrays of identical shape, indexed as
        ``[row, col]``. Row indices increase downward; the arrays may be
        north-up (decreasing latitude) or south-up (increasing latitude).
        Longitudes are interpolated as given, so they must be unwrapped
        around the region of interest (no +/-180 deg adjustment).

    Raises
    ------
    ValueError
        If the arrays are not two-dimensional or their shapes differ.
    """

    def __init__(self, lats_deg: np.ndarray, lons_deg: np.ndarray) -> None:
        lats = np.asarray(lats_deg, dtype=float)
        lons = np.asarray(lons_deg, dtype=float)
        if lats.ndim != 2 or lons.ndim != 2:
            raise ValueError(
                f"coordinate arrays must be 2-D, got {lats.ndim}-D and {lons.ndim}-D"
            )
        if lats.shape != lons.shape:
            raise ValueError(
                f"coordinate shapes differ: {lats.shape!r} vs {lons.shape!r}"
            )
        if lats.size == 0:
            raise ValueError("coordinate arrays must not be empty")
        self.lats_deg = lats
        self.lons_deg = lons

    @property
    def shape(self) -> tuple[int, int]:
        """Coordinate array shape ``(n_rows, n_cols)``."""
        return self.lats_deg.shape

    def to_latlon(self, row: float, col: float) -> LatLon:
        """Interpolate coordinates at fractional image index ``(row, col)``.

        Parameters
        ----------
        row, col : float
            Image index coordinates. Positions inside
            ``[0, n_rows - 1] x [0, n_cols - 1]`` are interpolated
            bilinearly, with the border cell weights clamped to the outer
            pixel centres.

        Returns
        -------
        tuple of float
            ``(lat_deg, lon_deg)``; NaN when the position is outside the
            array or the inputs are not finite.
        """
        if not (math.isfinite(row) and math.isfinite(col)):
            return (math.nan, math.nan)
        n_rows, n_cols = self.shape
        if row < 0.0 or row > n_rows - 1 or col < 0.0 or col > n_cols - 1:
            return (math.nan, math.nan)
        row = min(row, float(n_rows - 1))
        col = min(col, float(n_cols - 1))
        row_0 = min(int(row), max(n_rows - 2, 0))
        col_0 = min(int(col), max(n_cols - 2, 0))
        row_weight = row - row_0
        col_weight = col - col_0
        lat = _bilinear(self.lats_deg, row_0, col_0, row_weight, col_weight)
        lon = _bilinear(self.lons_deg, row_0, col_0, row_weight, col_weight)
        return (float(lat), float(lon))


def _bilinear(
    values: np.ndarray, row_0: int, col_0: int, row_weight: float, col_weight: float
) -> float:
    """Bilinear interpolation of ``values`` in the cell at ``(row_0, col_0)``."""
    row_1 = min(row_0 + 1, values.shape[0] - 1)
    col_1 = min(col_0 + 1, values.shape[1] - 1)
    top = (
        values[row_0, col_0] * (1.0 - col_weight) + values[row_0, col_1] * col_weight
    )
    bottom = (
        values[row_1, col_0] * (1.0 - col_weight) + values[row_1, col_1] * col_weight
    )
    return top * (1.0 - row_weight) + bottom * row_weight
