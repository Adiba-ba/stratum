"""Corridor-level pipeline orchestration (Step 8).

Ties the whole STRATUM chain together for a single flight: feature
assembly along the cleaned ADS-B trajectory, calibrated persistence
probability, per-segment counterfactual evaluation, and aggregation into
an actionable reroute advisory. Consecutive flagged segments are grouped
into one *event* so the same avoided contrail is not counted twice.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stratum.decide.counterfactual import (
    ABSTAIN_THRESHOLD,
    DEFAULT_SEGMENT_LENGTH_KM,
    AltitudeDecision,
    evaluate_altitude_candidates,
)
from stratum.match.label import haversine_km
from stratum.model.calibrate import CalibratedModel
from stratum.pipeline.build_features import extract_features_for_trajectory

__all__ = [
    "PERSISTENT_PROBABILITY_THRESHOLD",
    "evaluate_corridor_flight",
]

#: Calibrated probability at or above which a segment counts as
#: "persistent" in the flight statistics (the decision layer activation
#: uses :data:`stratum.decide.counterfactual.ABSTAIN_THRESHOLD` instead).
PERSISTENT_PROBABILITY_THRESHOLD: float = 0.50


def _segment_lengths_km(metadata: list[dict[str, Any]]) -> list[float]:
    """Per-point segment length [km] from the haversine of neighbours.

    The length of a point's segment is the great-circle distance to the
    next point, except for the last point, which uses the distance to the
    previous point. This is the ``segment_length_km`` handed to
    :func:`stratum.decide.counterfactual.evaluate_altitude_candidates`
    so the avoided energy forcing scales with the real flight distance
    instead of a fixed audit constant.
    """
    coordinates = [
        (float(info["lat"]), float(info["lon"])) for info in metadata
    ]
    lengths: list[float] = []
    for index, coordinate in enumerate(coordinates):
        if index + 1 < len(coordinates):
            neighbour = coordinates[index + 1]
        elif coordinates:
            neighbour = coordinates[index - 1]
        else:  # pragma: no cover - metadata is never empty here
            lengths.append(DEFAULT_SEGMENT_LENGTH_KM)
            continue
        length_km = haversine_km(coordinate, neighbour)
        if not math.isfinite(length_km) or length_km <= 0.0:
            length_km = DEFAULT_SEGMENT_LENGTH_KM
        lengths.append(length_km)
    return lengths


def evaluate_corridor_flight(
    df_trajectory: pd.DataFrame,
    ds_met: xr.Dataset,
    model: CalibratedModel,
    aircraft_type: str = "A320",
) -> dict[str, Any]:
    """Evaluate one flight corridor end to end.

    Parameters
    ----------
    df_trajectory : pandas.DataFrame
        Cleaned ADS-B trajectory (see
        :func:`stratum.pipeline.build_features.extract_features_for_trajectory`).
    ds_met : xarray.Dataset
        ERA5 dataset with the canonical variables.
    model : CalibratedModel
        Trained LightGBM model with isotonic calibration; must expose
        ``predict_proba(X) -> (N, 2)``.
    aircraft_type : str, optional
        Aircraft type for the fuel penalty model, by default ``"A320"``.

    Returns
    -------
    dict[str, Any]
        Aggregated flight report with at least:

        - ``total_segments`` : number of trajectory points evaluated;
        - ``contrail_forming_segments`` : segments where the SAC criterion
          is satisfied (physics);
        - ``contrail_persistent_segments`` : segments with calibrated
          ``P(persist) >= 0.50``;
        - ``out_of_domain_segments`` : segments whose ERA5 data is
          unavailable (outside the ingestion domain, or interpolation
          gaps); they never receive a decision and never enter an event;
        - ``total_avoidable_warming_j`` : avoided contrail effective
          forcing summed over reroute events [J];
        - ``recommended_reroutes`` : one structured advisory per
          contiguous flagged run (level change, the real segment length
          used for the avoided-forcing accounting, fuel/CO2 delta,
          avoided warming, benefit-to-cost ratio);
        - ``segments`` : per-segment atmospheric and decision records;
          entries carry ``status`` (``"ok"``, ``"out_of_domain"`` or
          ``"missing_met"``), ``segment_length_km`` (haversine distance
          to the neighbouring waypoint) and a decision record only when
          ERA5 data is available.

    Raises
    ------
    ValueError
        Propagated from feature extraction for malformed trajectories.
    """
    features, metadata = extract_features_for_trajectory(
        df_trajectory, ds_met, aircraft_type
    )
    probabilities = np.asarray(model.predict_proba(features), dtype=float)[:, 1]
    segment_lengths_km = _segment_lengths_km(metadata)

    forming_segments = 0
    persistent_segments = 0
    out_of_domain_segments = 0
    flagged_indices: list[int] = []
    decisions: dict[int, AltitudeDecision] = {}
    segments: list[dict[str, Any]] = []

    for index, info in enumerate(metadata):
        probability = float(probabilities[index])
        if not info["data_available"]:
            out_of_domain_segments += 1
        else:
            if info["sac_forms"]:
                forming_segments += 1
            if probability >= PERSISTENT_PROBABILITY_THRESHOLD:
                persistent_segments += 1

        decision: AltitudeDecision | None = None
        if info["data_available"] and probability >= ABSTAIN_THRESHOLD:
            decision = evaluate_altitude_candidates(
                probability,
                float(info["flight_level"]),
                aircraft_type,
                segment_length_km=segment_lengths_km[index],
            )
            decisions[index] = decision
            flagged_indices.append(index)

        segments.append(
            {
                "segment_index": index,
                "status": (
                    "ok" if info["data_available"] else info["unavailable_reason"]
                ),
                "data_available": info["data_available"],
                "time_utc": info["time_utc"],
                "lat": info["lat"],
                "lon": info["lon"],
                "flight_level": info["flight_level"],
                "segment_length_km": segment_lengths_km[index],
                "pressure_pa": info["pressure_pa"],
                "temperature_c": info["temperature_k"] - 273.15,
                "rhi": info["rhi"],
                "tlm_c": info["tlm_c"],
                "tlc_c": info["tlc_c"],
                "sac_forms": info["sac_forms"],
                "ambient_ice_supersaturated": info["ambient_ice_supersaturated"],
                "p_persist": probability,
                "decision_status": decision["status"] if decision else None,
                "recommended_fl": decision["recommended_fl"] if decision else None,
                "delta_fuel_kg": decision["delta_fuel_kg"] if decision else None,
                "delta_co2_kg": decision["delta_co2_kg"] if decision else None,
                "avoided_contrail_ef_j": (
                    decision["avoided_contrail_ef_j"] if decision else None
                ),
            }
        )

    # Group consecutive flagged segments into advisory events.
    groups: list[list[int]] = []
    for index in flagged_indices:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])

    reroutes: list[dict[str, Any]] = []
    for group in groups:
        representative = max(group, key=lambda item: probabilities[item])
        decision = decisions[representative]
        if decision["status"] != "eligible":
            continue
        reroutes.append(
            {
                "event_index": len(reroutes),
                "segment_index": representative,
                "time_utc": metadata[representative]["time_utc"],
                "segments_in_event": len(group),
                "p_persist": float(probabilities[representative]),
                "baseline_fl": float(decision["baseline_fl"]),
                "segment_length_km": float(decision["segment_length_km"]),
                "recommended_fl": float(decision["recommended_fl"]),
                "direction": (
                    "climb"
                    if float(decision["recommended_fl"]) > float(decision["baseline_fl"])
                    else "descent"
                ),
                "delta_fuel_kg": float(decision["delta_fuel_kg"]),
                "delta_co2_kg": float(decision["delta_co2_kg"]),
                "avoided_contrail_ef_j": float(decision["avoided_contrail_ef_j"]),
                "benefit_to_cost_ratio": float(decision["benefit_to_cost_ratio"]),
            }
        )

    total_avoidable_warming_j = float(
        sum(entry["avoided_contrail_ef_j"] for entry in reroutes)
    )
    total_extra_fuel_kg = float(sum(entry["delta_fuel_kg"] for entry in reroutes))
    total_extra_co2_kg = float(sum(entry["delta_co2_kg"] for entry in reroutes))

    return {
        "total_segments": int(len(metadata)),
        "contrail_forming_segments": int(forming_segments),
        "contrail_persistent_segments": int(persistent_segments),
        "flagged_segments": int(len(flagged_indices)),
        "out_of_domain_segments": int(out_of_domain_segments),
        "total_avoidable_warming_j": total_avoidable_warming_j,
        "total_extra_fuel_kg": total_extra_fuel_kg,
        "total_extra_co2_kg": total_extra_co2_kg,
        "recommended_reroutes": reroutes,
        "segments": segments,
    }
