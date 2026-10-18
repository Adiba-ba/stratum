"""Tests for geostationary navigation, detection headings and georeferencing."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pytest

from stratum.geo import (
    ArrayGeoreferencer,
    EARTH_RADIUS_M,
    GeostationaryNavigation,
    from_goes_attrs,
    initial_bearing_deg,
    pixel_angle_to_heading_deg,
)

GOES16_ATTRS = {
    "perspective_point_height": 35786023.0,
    "longitude_of_projection_origin": -75.2,
    "sweep_angle_axis": "x",
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
}
METEOSAT_ATTRS = {
    "perspective_point_height": 35785831.0,
    "longitude_of_projection_origin": 0.0,
    "sweep_angle_axis": "y",
    "semi_major_axis": 6378137.0,
    "semi_minor_axis": 6356752.31414,
}
#: Latitude band and longitude span safely inside the visible disk.
LAT_HALF_DEG = 50.0
LON_HALF_DEG = 60.0


def _visible_grid(
    attrs: Mapping[str, Any], steps: int = 21
) -> tuple[np.ndarray, np.ndarray]:
    """Latitude/longitude mesh of points inside the satellite view."""
    lat_axis = np.linspace(-LAT_HALF_DEG, LAT_HALF_DEG, steps)
    lon_axis = float(attrs["longitude_of_projection_origin"]) + np.linspace(
        -LON_HALF_DEG, LON_HALF_DEG, steps
    )
    return np.meshgrid(lat_axis, lon_axis, indexing="ij")


# ---------------------------------------------------------------------------
# Fixed grid navigation (GOES-R PUG Volume 3, paragraph 5.1.2.8)
# ---------------------------------------------------------------------------


def test_constants_use_project_mean_radius() -> None:
    """The geo constant matches the spherical-Earth convention of the repo."""
    from stratum.match import parallax

    assert EARTH_RADIUS_M == 6_371_000.0
    assert EARTH_RADIUS_M == parallax.EARTH_RADIUS_M


def test_goes16_roundtrip_over_visible_grid() -> None:
    """lat/lon -> scan angles -> lat/lon stays within 1e-6 deg (GOES-16)."""
    nav = from_goes_attrs(GOES16_ATTRS)
    lat, lon = _visible_grid(GOES16_ATTRS)
    x, y = nav.latlon_to_scan_angles(lat, lon)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    lat_back, lon_back = nav.scan_angles_to_latlon(x, y)
    assert np.max(np.abs(lat_back - lat)) < 1e-6
    assert np.max(np.abs(lon_back - lon)) < 1e-6


def test_sub_satellite_point_maps_to_zero_scan_angles() -> None:
    """The nadir point sits at the fixed grid origin."""
    nav = from_goes_attrs(GOES16_ATTRS)
    x, y = nav.latlon_to_scan_angles(0.0, -75.2)
    assert isinstance(x, float) and isinstance(y, float)
    assert abs(x) < 1e-12
    assert abs(y) < 1e-12
    lat, lon = nav.scan_angles_to_latlon(0.0, 0.0)
    assert abs(lat) < 1e-12
    assert abs(lon - (-75.2)) < 1e-12


def test_invisible_points_return_nan() -> None:
    """Far-side points and rays missing the earth yield NaN, not exceptions."""
    nav = from_goes_attrs(GOES16_ATTRS)
    x, y = nav.latlon_to_scan_angles(0.0, 105.0)
    assert math.isnan(x) and math.isnan(y)
    assert math.isnan(nav.latlon_to_scan_angles(-20.0, 40.0)[0])
    assert math.isnan(nav.scan_angles_to_latlon(0.30, 0.0)[0])
    assert math.isnan(nav.scan_angles_to_latlon(0.0, -0.30)[1])
    # points just inside the limb remain finite (limb at ~0.1519 rad)
    assert math.isfinite(nav.scan_angles_to_latlon(0.14, -0.05)[0])
    # NaN only in the invisible slots of an array input
    x_arr, y_arr = nav.latlon_to_scan_angles(
        np.array([0.0, 0.0]), np.array([-75.2, 105.0])
    )
    assert np.isfinite(x_arr[0]) and math.isnan(x_arr[1])
    assert np.isfinite(y_arr[0]) and math.isnan(y_arr[1])


def test_meteosat_sweep_axis_y_roundtrip() -> None:
    """Meteosat-style sweep_angle_axis='y' round-trips as well."""
    nav = from_goes_attrs(METEOSAT_ATTRS)
    lat, lon = _visible_grid(METEOSAT_ATTRS)
    x, y = nav.latlon_to_scan_angles(lat, lon)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    lat_back, lon_back = nav.scan_angles_to_latlon(x, y)
    assert np.max(np.abs(lat_back - lat)) < 1e-6
    assert np.max(np.abs(lon_back - lon)) < 1e-6
    x0, y0 = nav.latlon_to_scan_angles(0.0, 0.0)
    assert abs(x0) < 1e-12
    assert abs(y0) < 1e-12
    assert math.isnan(nav.latlon_to_scan_angles(0.0, 100.0)[0])


def test_pug_worked_example() -> None:
    """Reproduce the GOES-R PUG Volume 3 worked example (paragraph 5.1.2.8)."""
    nav = GeostationaryNavigation(35786023.0, -75.0, "x")
    lat, lon = nav.scan_angles_to_latlon(-0.024052, 0.095340)
    assert lat == pytest.approx(33.846162, abs=1e-6)
    assert lon == pytest.approx(-84.690932, abs=1e-6)
    x, y = nav.latlon_to_scan_angles(33.846162, -84.690932)
    assert x == pytest.approx(-0.024052, abs=1e-6)
    assert y == pytest.approx(0.095340, abs=1e-6)


def test_navigation_rejects_unknown_sweep_axis_and_height() -> None:
    """Constructor validates the sweep axis and the satellite height."""
    with pytest.raises(ValueError):
        GeostationaryNavigation(35786023.0, -75.2, sweep_angle_axis="z")
    with pytest.raises(ValueError):
        GeostationaryNavigation(0.0, -75.2)
    with pytest.raises(ValueError):
        GeostationaryNavigation(-1.0, -75.2)


def test_from_goes_attrs_reads_projection_attributes() -> None:
    """The helper maps goes_imager_projection attributes onto the class."""
    nav = from_goes_attrs(GOES16_ATTRS)
    assert nav.satellite_height_m == pytest.approx(35786023.0)
    assert nav.longitude_of_projection_deg == pytest.approx(-75.2)
    assert nav.sweep_angle_axis == "x"
    assert nav.semi_major_m == pytest.approx(6378137.0)
    assert nav.semi_minor_m == pytest.approx(6356752.31414)


def test_from_goes_attrs_accepts_satellite_height_and_bytes_sweep() -> None:
    """satellite_height substitutes for perspective_point_height."""
    attrs = dict(GOES16_ATTRS)
    attrs["satellite_height"] = attrs.pop("perspective_point_height")
    attrs["sweep_angle_axis"] = b"x"
    nav = from_goes_attrs(attrs)
    assert nav.satellite_height_m == pytest.approx(35786023.0)
    assert nav.sweep_angle_axis == "x"
    assert nav.latlon_to_scan_angles(0.0, -75.2) == pytest.approx((0.0, 0.0), abs=1e-12)


def test_from_goes_attrs_accepts_numpy_scalars() -> None:
    """netCDF4 hands attributes back as numpy scalars."""
    attrs = {
        "perspective_point_height": np.float32(35786023.0),
        "longitude_of_projection_origin": np.float64(-75.2),
        "sweep_angle_axis": np.str_("x"),
    }
    nav = from_goes_attrs(attrs)
    assert nav.satellite_height_m == pytest.approx(35786023.0)
    assert nav.sweep_angle_axis == "x"
    assert nav.semi_major_m == pytest.approx(6378137.0)
    assert nav.semi_minor_m == pytest.approx(6356752.31414)


def test_from_goes_attrs_requires_height_and_longitude() -> None:
    """Missing projection attributes raise ValueError."""
    with pytest.raises(ValueError):
        from_goes_attrs({"longitude_of_projection_origin": -75.2})
    with pytest.raises(ValueError):
        from_goes_attrs({"perspective_point_height": 35786023.0})
    with pytest.raises(ValueError):
        from_goes_attrs({"perspective_point_height": "high"})


@pytest.mark.parametrize("attrs", [GOES16_ATTRS, METEOSAT_ATTRS])
def test_pyproj_geos_cross_validation(attrs: Mapping[str, Any]) -> None:
    """Cross-validate against PROJ's geos projection (skipped without pyproj)."""
    pyproj = pytest.importorskip("pyproj")
    nav = from_goes_attrs(attrs)
    height = float(attrs["perspective_point_height"])
    lon_0 = float(attrs["longitude_of_projection_origin"])
    projection = pyproj.Proj(
        proj="geos",
        h=height,
        lon_0=lon_0,
        sweep=str(attrs["sweep_angle_axis"]),
        a=float(attrs["semi_major_axis"]),
        b=float(attrs["semi_minor_axis"]),
    )
    rng = np.random.default_rng(20260818)
    lat = rng.uniform(-60.0, 60.0, 40)
    lon = lon_0 + rng.uniform(-60.0, 60.0, 40)
    x_m, y_m = projection(lon, lat)
    visible = np.isfinite(x_m) & np.isfinite(y_m) & (np.abs(x_m) < 1e8)
    assert visible.sum() >= 20

    # lat/lon -> scan angles against PROJ's forward projection
    x, y = nav.latlon_to_scan_angles(lat[visible], lon[visible])
    assert np.isfinite(x).all()
    assert np.allclose(x, x_m[visible] / height, atol=1e-12)
    assert np.allclose(y, y_m[visible] / height, atol=1e-12)

    # scan angles -> lat/lon against PROJ's inverse projection
    lat_back, lon_back = nav.scan_angles_to_latlon(x, y)
    lon_proj, lat_proj = projection(x * height, y * height, inverse=True)
    assert np.allclose(lat_back, lat_proj, atol=1e-9)
    assert np.allclose(lon_back, lon_proj, atol=1e-9)

    # PROJ flags invisible lat/lon with non-finite output; we flag it with NaN
    lat_far = np.array([0.0, 0.0, 45.0])
    lon_far = np.array([lon_0 + 105.0, lon_0 - 105.0, lon_0 + 100.0])
    x_far, y_far = projection(lon_far, lat_far)
    assert not np.isfinite(x_far).any()
    assert not np.isfinite(y_far).any()
    assert np.isnan(nav.latlon_to_scan_angles(lat_far, lon_far)[0]).all()
    assert np.isnan(nav.latlon_to_scan_angles(lat_far, lon_far)[1]).all()


# ---------------------------------------------------------------------------
# Array georeferencing and detection headings
# ---------------------------------------------------------------------------


def _affine_georeferencer(
    n_rows: int = 9,
    n_cols: int = 11,
    lat_north: float = 50.0,
    d_lat: float = -0.25,
    lon_west: float = 10.0,
    d_lon: float = 0.5,
) -> ArrayGeoreferencer:
    """North-up affine grid: latitude falls southward, longitude rises eastward."""
    rows = np.arange(n_rows, dtype=float)[:, None]
    cols = np.arange(n_cols, dtype=float)[None, :]
    lats = lat_north + d_lat * rows + np.zeros_like(cols)
    lons = lon_west + d_lon * cols + np.zeros_like(rows)
    return ArrayGeoreferencer(lats, lons)


def _expected_heading_deg(
    delta_row: float, delta_col: float, lat_deg: float, d_lat: float, d_lon: float
) -> float:
    """Tangent-plane heading of a grid-direction step [compass deg]."""
    delta_lat = d_lat * delta_row
    delta_lon = d_lon * delta_col
    return (
        math.degrees(
            math.atan2(delta_lon * math.cos(math.radians(lat_deg)), delta_lat)
        )
        % 360.0
    )


def test_array_georeferencer_interpolates_affine_field() -> None:
    """Bilinear interpolation reproduces an affine coordinate field exactly."""
    georef = _affine_georeferencer()
    for row, col in ((0.0, 0.0), (3.25, 7.5), (8.0, 10.0), (4.9, 2.1)):
        lat, lon = georef.to_latlon(row, col)
        assert lat == pytest.approx(50.0 - 0.25 * row)
        assert lon == pytest.approx(10.0 + 0.5 * col)


def test_array_georeferencer_bilinear_accuracy_on_curved_field() -> None:
    """Cross terms are interpolated, not extrapolated from a local plane."""
    rows = np.arange(6, dtype=float)[:, None]
    cols = np.arange(7, dtype=float)[None, :]
    lats = 40.0 - 0.1 * rows + 0.02 * rows * cols
    lons = -5.0 + 0.3 * cols + 0.01 * rows * cols
    georef = ArrayGeoreferencer(lats, lons)
    lat, lon = georef.to_latlon(2.5, 3.25)
    assert lat == pytest.approx(40.0 - 0.1 * 2.5 + 0.02 * 2.5 * 3.25)
    assert lon == pytest.approx(-5.0 + 0.3 * 3.25 + 0.01 * 2.5 * 3.25)
    # half-cell inside the last row still interpolates, one step outside is NaN
    assert math.isfinite(georef.to_latlon(5.0, 6.0)[0])
    assert math.isnan(georef.to_latlon(5.01, 6.0)[0])
    assert math.isnan(georef.to_latlon(-0.01, 3.0)[0])
    assert math.isnan(georef.to_latlon(3.0, math.nan)[1])


def test_array_georeferencer_handles_latitude_flipped_grid() -> None:
    """South-up grids (latitude increasing with row) interpolate correctly."""
    rows = np.arange(8, dtype=float)[:, None]
    cols = np.arange(6, dtype=float)[None, :]
    lats = 20.0 + 0.4 * rows + np.zeros_like(cols)
    lons = 70.0 + 0.25 * cols + np.zeros_like(rows)
    georef = ArrayGeoreferencer(lats, lons)
    assert georef.shape == (8, 6)
    lat, lon = georef.to_latlon(3.5, 2.25)
    assert lat == pytest.approx(20.0 + 0.4 * 3.5)
    assert lon == pytest.approx(70.0 + 0.25 * 2.25)
    assert math.isnan(georef.to_latlon(8.0, 0.0)[0])


def test_array_georeferencer_rejects_bad_shapes() -> None:
    """Shape mismatches and non-2-D inputs raise ValueError."""
    with pytest.raises(ValueError):
        ArrayGeoreferencer(np.zeros((3, 4)), np.zeros((4, 3)))
    with pytest.raises(ValueError):
        ArrayGeoreferencer(np.zeros(4), np.zeros(4))


def test_initial_bearing_cardinal_directions() -> None:
    """North, east, south and west bearings follow the compass convention."""
    assert initial_bearing_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0)
    assert initial_bearing_deg(0.0, 0.0, 0.0, 10.0) == pytest.approx(90.0)
    assert initial_bearing_deg(10.0, 0.0, 0.0, 0.0) == pytest.approx(180.0)
    assert initial_bearing_deg(0.0, 10.0, 0.0, 0.0) == pytest.approx(270.0)
    # along a parallel the great-circle bearing is not exactly cardinal
    # (hand-computed with the standard spherical bearing formula)
    assert initial_bearing_deg(10.0, 0.0, 10.0, -10.0) == pytest.approx(
        270.87, abs=0.01
    )
    assert initial_bearing_deg(45.0, 20.0, 45.0, 30.0) == pytest.approx(
        86.46, abs=0.01
    )
    assert math.isnan(initial_bearing_deg(5.0, 5.0, 5.0, 5.0))


def test_pixel_heading_north_up_grid() -> None:
    """On a north-up grid the image-space 'up' direction is compass north."""
    georef = _affine_georeferencer()
    row, col = 4.0, 5.0
    lat = 50.0 - 0.25 * row
    # pixel up (decreasing row) -> north; down -> south; the east and west
    # segment follows a parallel, so the great-circle bearing at ~49 deg N
    # is a few tenths of a degree off the cardinal value.
    cases = (
        (270.0, 0.0, 0.1),
        (0.0, 90.0, 0.5),
        (90.0, 180.0, 0.1),
        (180.0, 270.0, 0.5),
        (315.0, _expected_heading_deg(-1.0, 1.0, lat, -0.25, 0.5), 0.5),
        (45.0, _expected_heading_deg(1.0, 1.0, lat, -0.25, 0.5), 0.5),
        (225.0, _expected_heading_deg(-1.0, -1.0, lat, -0.25, 0.5), 0.5),
    )
    for angle_px, expected, tolerance in cases:
        heading = pixel_angle_to_heading_deg(
            angle_px, row, col, georef.to_latlon, step_px=1.0
        )
        assert heading == pytest.approx(expected, abs=tolerance), angle_px


def test_pixel_heading_latitude_flipped_grid() -> None:
    """On a south-up grid pixel-space 'up' points south."""
    rows = np.arange(9, dtype=float)[:, None]
    cols = np.arange(11, dtype=float)[None, :]
    lats = 20.0 + 0.4 * rows + np.zeros_like(cols)
    lons = 70.0 + 0.25 * cols + np.zeros_like(rows)
    georef = ArrayGeoreferencer(lats, lons)
    row, col = 4.0, 5.0
    lat = 20.0 + 0.4 * row
    headings = {
        270.0: 180.0,  # pixel up -> south
        0.0: 90.0,  # pixel east -> east
        315.0: _expected_heading_deg(-1.0, 1.0, lat, 0.4, 0.25),
    }
    for angle_px, expected in headings.items():
        heading = pixel_angle_to_heading_deg(
            angle_px, row, col, georef.to_latlon, step_px=1.0
        )
        assert heading == pytest.approx(expected, abs=0.5), angle_px


def test_pixel_heading_nan_when_endpoint_leaves_grid() -> None:
    """Detection endpoints outside the georeferenced area give NaN."""
    georef = _affine_georeferencer(n_rows=5, n_cols=5)
    assert math.isnan(pixel_angle_to_heading_deg(270.0, 0.5, 2.0, georef.to_latlon))
    assert math.isfinite(
        pixel_angle_to_heading_deg(270.0, 2.0, 2.0, georef.to_latlon, step_px=2.0)
    )
    with pytest.raises(ValueError):
        pixel_angle_to_heading_deg(270.0, 2.0, 2.0, georef.to_latlon, step_px=0.0)


def test_pixel_heading_on_geostationary_grid() -> None:
    """End-to-end: a north-up GOES grid turns pixel-up into compass north."""
    nav = from_goes_attrs(GOES16_ATTRS)
    lat_axis = np.linspace(35.0, 30.0, 7)
    lon_axis = np.linspace(-80.0, -75.0, 7)
    lats, lons = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    x, y = nav.latlon_to_scan_angles(lats, lons)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    # scan angles are on the fixed grid; recover coordinates through the
    # navigation object to build the georeferencer
    grid_lats, grid_lons = nav.scan_angles_to_latlon(x, y)
    assert np.allclose(grid_lats, lats, atol=1e-9)
    assert np.allclose(grid_lons, lons, atol=1e-9)
    georef = ArrayGeoreferencer(grid_lats, grid_lons)
    heading = pixel_angle_to_heading_deg(270.0, 3.0, 3.0, georef.to_latlon, step_px=2.0)
    assert abs(heading) < 1.0
