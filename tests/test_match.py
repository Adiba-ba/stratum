"""Tests for flight-contrail association, parallax, advection and labels."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stratum.match import advect, label, observe, parallax

DELHI = (28.5, 77.0)
MUMBAI = (19.0, 73.0)
FL360_ALT_M = 36000.0 * 0.3048


def _haversine_km(
    coord1: tuple[float, float], coord2: tuple[float, float]
) -> float:
    """Reference great-circle distance between (lat, lon) pairs [km]."""
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    a = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 2.0 * parallax.EARTH_RADIUS_M * math.asin(math.sqrt(a)) / 1000.0


def test_parallax_fl360_delhi_displaced_north_east() -> None:
    """FL360 over Delhi shifts ~10-14 km away from the sub-satellite point."""
    delta_lat, delta_lon = parallax.calculate_parallax_displacement(
        DELHI[0], DELHI[1], FL360_ALT_M
    )
    assert delta_lat > 0.0  # northward, away from 0 deg N
    assert delta_lon > 0.0  # eastward, away from 45.5 deg E
    distance_km = _haversine_km(DELHI, (DELHI[0] + delta_lat, DELHI[1] + delta_lon))
    assert 10.0 <= distance_km <= 14.0, distance_km
    assert 0.07 <= distance_km / 111.32 <= 0.14


def test_parallax_mumbai_11000m_within_bounds() -> None:
    """11000 m over Mumbai shifts 8-15 km northeast as well."""
    delta_lat, delta_lon = parallax.calculate_parallax_displacement(
        MUMBAI[0], MUMBAI[1], 11000.0
    )
    assert delta_lat > 0.0
    assert delta_lon > 0.0
    distance_km = _haversine_km(MUMBAI, (MUMBAI[0] + delta_lat, MUMBAI[1] + delta_lon))
    assert 8.0 <= distance_km <= 15.0, distance_km


def test_parallax_zero_at_sub_satellite_point() -> None:
    """A target below the satellite has no parallax displacement."""
    delta_lat, delta_lon = parallax.calculate_parallax_displacement(0.0, 45.5, 11000.0)
    assert delta_lat == pytest.approx(0.0, abs=1e-9)
    assert delta_lon == pytest.approx(0.0, abs=1e-9)


def test_advect_eastward_wind_drifts_36km() -> None:
    """20 m/s eastward for 1800 s drifts the point ~36 km east."""
    new_lat, new_lon = advect.advect_segment(DELHI[0], DELHI[1], 20.0, 0.0, 1800.0)
    assert new_lat == pytest.approx(DELHI[0])
    assert new_lon > DELHI[1]
    assert new_lon - DELHI[1] == pytest.approx(0.3684, abs=0.002)
    distance_km = _haversine_km(DELHI, (new_lat, new_lon))
    assert 35.5 <= distance_km <= 36.5, distance_km


def test_advect_zero_wind_is_identity_and_negative_time_rejected() -> None:
    """Zero wind leaves the point unchanged; negative time raises."""
    assert advect.advect_segment(DELHI[0], DELHI[1], 0.0, 0.0, 3600.0) == DELHI
    with pytest.raises(ValueError):
        advect.advect_segment(DELHI[0], DELHI[1], 20.0, 0.0, -1.0)


@pytest.mark.parametrize(
    ("high_cloud_fraction", "viewing_zenith_angle_deg", "opaque_cloud", "expected"),
    [
        (0.6, 40.0, False, False),  # too much high cloud
        (0.2, 65.0, False, False),  # viewing zenith angle too large
        (0.2, 40.0, True, False),  # opaque cloud screens the scene
        (0.2, 40.0, False, True),  # all gates pass
        (0.49, 59.9, False, True),  # just inside the thresholds
    ],
)
def test_observability_gate(
    high_cloud_fraction: float,
    viewing_zenith_angle_deg: float,
    opaque_cloud: bool,
    expected: bool,
) -> None:
    """Observability requires cloud < 0.5, VZA < 60 and no opaque cloud."""
    assert (
        observe.is_scene_observable(
            high_cloud_fraction, viewing_zenith_angle_deg, opaque_cloud
        )
        is expected
    )


def test_tristate_label_matched_is_positive() -> None:
    """A matched segment is labeled 1.0 regardless of slot count."""
    assert label.assign_tristate_label(True, 0) == 1.0
    assert label.assign_tristate_label(True, 3) == 1.0


def test_tristate_label_observable_non_detection_is_zero() -> None:
    """Not matched with at least two observable slots is a hard 0.0."""
    assert label.assign_tristate_label(False, 3) == 0.0
    assert label.assign_tristate_label(False, 2) == 0.0


def test_tristate_label_unobservable_is_nan_not_zero() -> None:
    """Not matched with fewer than two observable slots is NaN, never 0.0."""
    for observable_slots in (0, 1):
        result = label.assign_tristate_label(False, observable_slots)
        assert np.isnan(result)
        assert result != 0.0


def test_match_requires_distance_and_angle() -> None:
    """Both centroid distance and line orientation must agree."""
    segment = (28.5, 77.0)
    heading = 175.0
    nearby_parallel = {"centroid": (28.52, 77.0), "angle_deg": 5.0}
    nearby_crossing = {"centroid": (28.52, 77.0), "angle_deg": 80.0}
    distant_parallel = {"centroid": (28.6, 77.0), "angle_deg": 175.0}

    assert label.match_segment_to_detections(segment, heading, [nearby_parallel])
    assert not label.match_segment_to_detections(segment, heading, [nearby_crossing])
    assert label.match_segment_to_detections(
        segment, heading, [nearby_parallel, nearby_crossing]
    )
    assert not label.match_segment_to_detections(segment, heading, [distant_parallel])
    assert label.match_segment_to_detections(
        segment, heading, [distant_parallel], distance_tolerance_km=12.0
    )


def test_match_angle_is_modulo_180() -> None:
    """Heading 350 deg and object angle 10 deg are 20 deg apart mod 180."""
    segment = (28.5, 77.0)
    obj = {"centroid": (28.5, 77.01), "angle_deg": 10.0}
    assert label.match_segment_to_detections(
        segment, 350.0, [obj], angle_tolerance_deg=20.0
    )
    assert not label.match_segment_to_detections(
        segment, 350.0, [obj], angle_tolerance_deg=19.0
    )
