"""Tests for Indian corridor generation and route-level SAC screening."""

from __future__ import annotations

import math

import pytest

from stratum.physics import thermo
from stratum.routes import corridor

FL340: float = 340.0
NUM_POINTS: int = 25


def test_airport_coordinates_registered() -> None:
    """ICAO coordinates match the STRATUM reference values."""
    assert corridor.AIRPORTS["VIDP"] == (28.5665, 77.1031)
    assert corridor.AIRPORTS["VABB"] == (19.0896, 72.8656)
    assert corridor.AIRPORTS["VOBL"] == (13.1986, 77.7066)
    assert corridor.AIRPORTS["VECC"] == (22.6547, 88.4467)
    assert corridor.AIRPORTS["VAAH"] == (23.0772, 72.6347)


def test_great_circle_endpoints_length_and_pressure() -> None:
    """Default track has 25 points, exact endpoints and ISA FL340 pressure."""
    coord1 = corridor.AIRPORTS["VIDP"]
    coord2 = corridor.AIRPORTS["VABB"]
    waypoints = corridor.great_circle_waypoints(coord1, coord2)

    assert len(waypoints) == NUM_POINTS
    assert (waypoints[0]["lat"], waypoints[0]["lon"]) == coord1
    assert (waypoints[-1]["lat"], waypoints[-1]["lon"]) == coord2

    p_expected = thermo.fl_to_p(FL340) * 100.0
    for waypoint in waypoints:
        assert waypoint["fl"] == FL340
        assert waypoint["p_Pa"] == pytest.approx(p_expected, rel=1e-12)
        assert 19.0 <= waypoint["lat"] <= 29.0
        assert 72.0 <= waypoint["lon"] <= 78.0


def test_great_circle_custom_num_points_and_fl() -> None:
    """Custom point count and flight level are honoured."""
    coord1 = corridor.AIRPORTS["VABB"]
    coord2 = corridor.AIRPORTS["VOBL"]
    waypoints = corridor.great_circle_waypoints(coord1, coord2, num_points=2, fl=380)

    assert len(waypoints) == 2
    assert (waypoints[0]["lat"], waypoints[0]["lon"]) == coord1
    assert (waypoints[1]["lat"], waypoints[1]["lon"]) == coord2
    assert all(waypoint["fl"] == 380.0 for waypoint in waypoints)
    assert all(
        waypoint["p_Pa"] == pytest.approx(thermo.fl_to_p(380.0) * 100.0, rel=1e-12)
        for waypoint in waypoints
    )


def test_great_circle_latitude_decreases_delhi_to_mumbai() -> None:
    """A southbound great-circle leg tracks monotonically decreasing latitude."""
    waypoints = corridor.great_circle_waypoints(
        corridor.AIRPORTS["VIDP"], corridor.AIRPORTS["VABB"]
    )
    latitudes = [waypoint["lat"] for waypoint in waypoints]
    assert all(a > b for a, b in zip(latitudes, latitudes[1:]))


def test_great_circle_requires_two_points() -> None:
    """Fewer than two waypoints is rejected."""
    with pytest.raises(ValueError):
        corridor.great_circle_waypoints(
            corridor.AIRPORTS["VIDP"], corridor.AIRPORTS["VABB"], num_points=1
        )


def test_great_circle_rejects_antipodal_endpoints() -> None:
    """Antipodal endpoints make the slerp singular and are rejected."""
    with pytest.raises(ValueError, match="antipodal"):
        corridor.great_circle_waypoints((0.0, 0.0), (0.0, 180.0))


def test_great_circle_identical_endpoints_still_work() -> None:
    """Identical endpoints keep the straight-line fill behaviour."""
    coord = corridor.AIRPORTS["VIDP"]
    waypoints = corridor.great_circle_waypoints(coord, coord, num_points=3)

    assert len(waypoints) == 3
    for waypoint in waypoints:
        assert waypoint["lat"] == pytest.approx(coord[0], abs=1e-9)
        assert waypoint["lon"] == pytest.approx(coord[1], abs=1e-9)


def test_great_circle_normal_route_still_works() -> None:
    """A non-antipodal route keeps producing finite interior waypoints."""
    coord1 = corridor.AIRPORTS["VIDP"]
    coord2 = corridor.AIRPORTS["VABB"]
    waypoints = corridor.great_circle_waypoints(coord1, coord2)

    assert len(waypoints) == NUM_POINTS
    assert (waypoints[0]["lat"], waypoints[0]["lon"]) == coord1
    assert (waypoints[-1]["lat"], waypoints[-1]["lon"]) == coord2
    for waypoint in waypoints[1:-1]:
        assert math.isfinite(waypoint["lat"])
        assert math.isfinite(waypoint["lon"])
        assert 19.0 <= waypoint["lat"] <= 29.0
        assert 72.0 <= waypoint["lon"] <= 78.0


def test_evaluate_route_profile_validates_lengths() -> None:
    """Profile lists must match the waypoint count."""
    waypoints = corridor.great_circle_waypoints(
        corridor.AIRPORTS["VIDP"], corridor.AIRPORTS["VABB"]
    )
    with pytest.raises(ValueError):
        corridor.evaluate_route_profile(waypoints, [-50.0, -49.0], [0.5] * NUM_POINTS)


def test_vabb_vobl_winter_zero_persistence() -> None:
    """Warm tropical leg: T > T_LC everywhere, exactly 0% persistence."""
    waypoints = corridor.great_circle_waypoints(
        corridor.AIRPORTS["VABB"], corridor.AIRPORTS["VOBL"]
    )
    temps = corridor.piecewise_profile(((0.0, -38.0), (1.0, -34.0)), NUM_POINTS)
    rhi_values = corridor.piecewise_profile(((0.0, 0.30), (1.0, 0.75)), NUM_POINTS)

    result = corridor.evaluate_route_profile(waypoints, temps, rhi_values)

    assert result["persistence_count"] == 0
    assert result["persistence_pct"] == 0.0
    assert result["formation_count"] == 0
    for detail, temp_c in zip(result["details"], temps):
        assert detail["forms"] is False
        assert float(detail["TLC_C"]) < temp_c  # ambient warmer than T_LC


def test_vidp_vecc_winter_persistence_above_40pct() -> None:
    """Moist winter jet leg: ice-supersaturated over 40% of waypoints."""
    waypoints = corridor.great_circle_waypoints(
        corridor.AIRPORTS["VIDP"], corridor.AIRPORTS["VECC"]
    )
    temps = corridor.piecewise_profile(((0.0, -54.0), (1.0, -47.0)), NUM_POINTS)
    rhi_values = corridor.piecewise_profile(
        ((0.0, 0.65), (0.55, 1.00), (1.0, 1.25)), NUM_POINTS
    )

    result = corridor.evaluate_route_profile(waypoints, temps, rhi_values)

    assert result["persistence_pct"] > 40.0
    assert result["formation_count"] >= result["persistence_count"]
    assert result["total_waypoints"] == NUM_POINTS
