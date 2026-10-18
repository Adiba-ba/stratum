"""Tests for the standalone corridor report (Step 9).

Covers the three direction states of the report (climb, descent, no
recommendation), the candidate rationale wording, the optional
verification KPI, and the ERA5 RHi probe evidence paragraph.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from stratum.decide.counterfactual import evaluate_altitude_candidates
from stratum.physics import e_i
from stratum.reporting import probe_rhi_evidence, render_corridor_report

_AIRCRAFT = "A320"
_PERSIST_THRESHOLD = 0.85

_PROBE_LEVELS_HPA = np.array([200.0, 250.0, 300.0, 350.0, 400.0])
_PROBE_LATS = np.array([27.0, 29.0])
_PROBE_LONS = np.array([76.0, 78.0])
_PROBE_TIMES = np.array(
    ["2025-01-01T01:00", "2025-01-01T02:00"], dtype="datetime64[ns]"
)
#: Prescribed RHi per pressure level: the upper level is wetter than the
#: level 4,000 ft below it, so a descent dries the sampled air.
_PROBE_RHI_BY_LEVEL = {200.0: 1.40, 250.0: 0.60, 300.0: 0.90, 350.0: 0.90, 400.0: 1.10}


def _probe_dataset() -> xr.Dataset:
    """In-memory ERA5-like dataset with a prescribed RHi profile."""
    shape = (
        _PROBE_LEVELS_HPA.size,
        _PROBE_LATS.size,
        _PROBE_LONS.size,
        _PROBE_TIMES.size,
    )
    level = _PROBE_LEVELS_HPA[:, None, None, None]
    temperature = 216.65 + 0.11 * (level - 200.0)
    saturation = np.array(
        [e_i(216.65 + 0.11 * (value - 200.0)) for value in _PROBE_LEVELS_HPA]
    )[:, None, None, None]
    rhi = np.array(
        [_PROBE_RHI_BY_LEVEL[value] for value in _PROBE_LEVELS_HPA]
    )[:, None, None, None]
    specific_humidity = rhi * saturation * 0.622 / (level * 100.0)

    return xr.Dataset(
        data_vars={
            "temperature": (
                ("level", "latitude", "longitude", "time"),
                np.broadcast_to(temperature, shape),
            ),
            "specific_humidity": (
                ("level", "latitude", "longitude", "time"),
                np.broadcast_to(specific_humidity, shape),
            ),
            "u_component_of_wind": (
                ("level", "latitude", "longitude", "time"),
                np.full(shape, 5.0),
            ),
            "v_component_of_wind": (
                ("level", "latitude", "longitude", "time"),
                np.zeros(shape),
            ),
            "vertical_velocity": (
                ("level", "latitude", "longitude", "time"),
                np.zeros(shape),
            ),
            "fraction_of_cloud_cover": (
                ("level", "latitude", "longitude", "time"),
                np.full(shape, 0.20),
            ),
        },
        coords={
            "level": _PROBE_LEVELS_HPA,
            "latitude": _PROBE_LATS,
            "longitude": _PROBE_LONS,
            "time": _PROBE_TIMES,
        },
    ).transpose("time", "level", "latitude", "longitude")


def _segment(
    index: int,
    decision_status: str | None = None,
    recommended_fl: float | None = None,
    flight_level: float = 350.0,
) -> dict:
    """One runner-shaped waypoint record for the audit table."""
    return {
        "segment_index": index,
        "time_utc": f"2025-01-01T0{index + 1}:00:00+00:00",
        "lat": 28.0 + 0.1 * index,
        "lon": 77.0,
        "temperature_c": -40.0,
        "rhi": 1.2,
        "p_persist": _PERSIST_THRESHOLD,
        "sac_forms": True,
        "decision_status": decision_status,
        "recommended_fl": recommended_fl,
        "flight_level": flight_level,
    }


def _primary(baseline_fl: float) -> dict:
    """Advisory record consistent with the real decision layer."""
    decision = evaluate_altitude_candidates(
        _PERSIST_THRESHOLD, baseline_fl, _AIRCRAFT
    )
    assert decision["status"] == "eligible"
    return {
        "baseline_fl": baseline_fl,
        "recommended_fl": decision["recommended_fl"],
        "p_persist": _PERSIST_THRESHOLD,
        "delta_fuel_kg": decision["delta_fuel_kg"],
        "delta_co2_kg": decision["delta_co2_kg"],
        "avoided_contrail_ef_j": decision["avoided_contrail_ef_j"],
        "benefit_to_cost_ratio": decision["benefit_to_cost_ratio"],
    }


def _corridor(
    baseline_fl: float,
    segments: list[dict],
    reroutes: list[dict] | None = None,
) -> dict:
    return {
        "origin": "VIDP",
        "destination": "VABB",
        "origin_label": "Delhi",
        "destination_label": "Mumbai",
        "num_waypoints": len(segments),
        "cruise_fl": baseline_fl,
        "segments": segments,
        "recommended_reroutes": reroutes or [],
        "total_segments": len(segments),
        "contrail_persistent_segments": sum(
            1 for segment in segments if segment["p_persist"] >= 0.5
        ),
    }


def _render(tmp_path, corridor: dict, name: str, **kwargs) -> str:
    path = tmp_path / name
    render_corridor_report(
        path,
        corridor=corridor,
        flight={"callsign": "AIC866", "icao24": "800abc", "aircraft_type": _AIRCRAFT},
        source={"label": "ECMWF ERA5 4D Reanalysis"},
        model={},
        **kwargs,
    )
    return path.read_text(encoding="utf-8")


def test_report_climb_direction_comes_from_decision_data(tmp_path) -> None:
    corridor = _corridor(
        baseline_fl=270.0,
        segments=[
            _segment(0, decision_status="eligible", recommended_fl=290.0, flight_level=270.0),
            _segment(1, decision_status=None, flight_level=270.0),
        ],
        reroutes=[_primary(270.0)],
    )

    html = _render(tmp_path, corridor, "climb.html")

    assert "CLIMB &#8594; FL290" in html
    assert "STEP CLIMB TO FL290" in html
    assert "DESCEND" not in html
    # Exactly one candidate is the recommendation; the other is rejected.
    assert html.count("SELECTED") == 1
    assert "SELECTED &#8212; climb to FL290" in html
    assert "REJECTED &#8212; target FL250 below minimum cruise FL270" in html


def test_report_descent_direction_comes_from_decision_data(tmp_path) -> None:
    corridor = _corridor(
        baseline_fl=390.0,
        segments=[
            _segment(0, decision_status="eligible", recommended_fl=370.0, flight_level=390.0),
            _segment(1, decision_status=None, flight_level=390.0),
        ],
        reroutes=[_primary(390.0)],
    )

    html = _render(tmp_path, corridor, "descent.html")

    assert "DESCEND &#8594; FL370" in html
    assert "STEP DESCENT TO FL370" in html
    assert "CLIMB" not in html
    assert html.count("SELECTED") == 1
    assert "SELECTED &#8212; descent to FL370" in html
    # The ceiling candidate and the two-step descent are both rejected.
    assert html.count("REJECTED") == 2
    assert "exceeds standard ceiling FL390" in html


def test_report_without_recommendation_omits_directions(tmp_path) -> None:
    corridor = _corridor(
        baseline_fl=350.0,
        segments=[
            _segment(0, decision_status="abstain"),
            _segment(1, decision_status="no_robust_candidate"),
            _segment(2, decision_status=None),
        ],
    )

    html = _render(tmp_path, corridor, "no_recommendation.html")

    assert "NO ACTION REQUIRED" in html
    assert "ABSTAIN" in html
    assert "NO CANDIDATE" in html
    assert "CLIMB" not in html
    assert "DESCEND" not in html
    assert "No advisory" in html


def test_report_eligible_without_level_change_is_not_a_direction(tmp_path) -> None:
    corridor = _corridor(
        baseline_fl=350.0,
        segments=[
            _segment(0, decision_status="eligible", recommended_fl=350.0),
        ],
    )

    html = _render(tmp_path, corridor, "no_level_change.html")

    assert "CLIMB" not in html
    assert "DESCEND" not in html
    assert "&#8212;" in html


def test_report_verification_kpi_is_caller_supplied(tmp_path) -> None:
    corridor = _corridor(
        baseline_fl=270.0,
        segments=[_segment(0, decision_status="eligible", recommended_fl=290.0, flight_level=270.0)],
        reroutes=[_primary(270.0)],
    )

    without = _render(tmp_path, corridor, "without_summary.html")
    assert "Verification" not in without
    assert "pytest tests/" not in without

    with_summary = _render(
        tmp_path, corridor, "with_summary.html", test_summary="179 PASSED"
    )
    assert "179 PASSED" in with_summary
    assert "Verification" in with_summary


def test_report_probe_paragraph_describes_recommended_direction(tmp_path) -> None:
    probe = {
        "n_waypoints": 3.0,
        "baseline_fl": 390.0,
        "proposed_fl": 370.0,
        "mean_baseline": 1.20,
        "mean_proposed": 0.70,
        "n_drier_at_proposed": 3.0,
        "n_subsaturated_proposed": 2.0,
    }
    corridor = _corridor(
        baseline_fl=390.0,
        segments=[_segment(0, decision_status="eligible", recommended_fl=370.0, flight_level=390.0)],
        reroutes=[_primary(390.0)],
    )
    corridor["rhi_probe"] = probe

    html = _render(tmp_path, corridor, "probe.html")

    assert "ERA5 probe over the 3 flagged waypoints" in html
    assert "the step descent dries the air" in html
    assert "1.20 at FL390" in html
    assert "0.70 at FL370" in html

    # Without a recommendation the probe numbers must not be rendered.
    without = _corridor(baseline_fl=350.0, segments=[_segment(0)])
    without["rhi_probe"] = probe
    html_without = _render(tmp_path, without, "probe_without.html")
    assert "ERA5 probe over" not in html_without


def test_probe_rhi_evidence_samples_both_levels() -> None:
    dataset = _probe_dataset()
    segments = [
        {"lat": 28.0, "lon": 77.0, "time_utc": "2025-01-01T01:30:00+00:00"},
        {"lat": 28.5, "lon": 76.5, "time_utc": "2025-01-01T01:45:00+00:00"},
        {"lat": 28.0, "lon": 77.0},  # missing time: skipped
        {"lat": 40.0, "lon": 77.0, "time_utc": "2025-01-01T01:30:00+00:00"},
    ]

    probe = probe_rhi_evidence(dataset, segments, baseline_fl=370.0, proposed_fl=330.0)

    assert probe is not None
    assert probe["n_waypoints"] == 2.0
    assert probe["baseline_fl"] == 370.0
    assert probe["proposed_fl"] == 330.0
    # The prescribed profile is wetter at FL370 than at FL330.
    assert probe["mean_baseline"] > probe["mean_proposed"]
    assert probe["n_drier_at_proposed"] == 2.0
    assert probe["n_subsaturated_proposed"] == 2.0

    assert probe_rhi_evidence(dataset, [], 370.0, 330.0) is None
