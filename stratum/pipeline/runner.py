"""Corridor-level pipeline orchestration (Step 8).

Ties the whole STRATUM chain together for a single flight: feature
assembly along the cleaned ADS-B trajectory, calibrated persistence
probability, per-segment counterfactual evaluation, and aggregation into
an actionable reroute advisory. Consecutive flagged segments are grouped
into one *event* so the same avoided contrail is not counted twice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from stratum.decide.counterfactual import (
    ABSTAIN_THRESHOLD,
    AltitudeDecision,
    evaluate_altitude_candidates,
)
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
        - ``total_avoidable_warming_j`` : avoided contrail effective
          forcing summed over reroute events [J];
        - ``recommended_reroutes`` : one structured advisory per
          contiguous flagged run (level change, fuel/CO2 delta, avoided
          warming, benefit-to-cost ratio);
        - ``segments`` : per-segment atmospheric and decision records.

    Raises
    ------
    ValueError
        Propagated from feature extraction for malformed trajectories.
    """
    features, metadata = extract_features_for_trajectory(
        df_trajectory, ds_met, aircraft_type
    )
    probabilities = np.asarray(model.predict_proba(features), dtype=float)[:, 1]

    forming_segments = 0
    persistent_segments = 0
    flagged_indices: list[int] = []
    decisions: dict[int, AltitudeDecision] = {}
    segments: list[dict[str, Any]] = []

    for index, info in enumerate(metadata):
        probability = float(probabilities[index])
        if info["sac_forms"]:
            forming_segments += 1
        if probability >= PERSISTENT_PROBABILITY_THRESHOLD:
            persistent_segments += 1

        decision: AltitudeDecision | None = None
        if probability >= ABSTAIN_THRESHOLD:
            decision = evaluate_altitude_candidates(
                probability, float(info["flight_level"]), aircraft_type
            )
            decisions[index] = decision
            flagged_indices.append(index)

        segments.append(
            {
                "segment_index": index,
                "time_utc": info["time_utc"],
                "lat": info["lat"],
                "lon": info["lon"],
                "flight_level": info["flight_level"],
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
        "total_avoidable_warming_j": total_avoidable_warming_j,
        "total_extra_fuel_kg": total_extra_fuel_kg,
        "total_extra_co2_kg": total_extra_co2_kg,
        "recommended_reroutes": reroutes,
        "segments": segments,
    }
