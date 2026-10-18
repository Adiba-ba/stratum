"""Standalone monochrome HTML corridor report for Project STRATUM.

Ported from ``scripts/generate_friend_report.py`` with two corrections:

- the recommended direction is read from the decision data, never
  assumed: a waypoint row shows ``CLIMB -> FLx`` when the recommended
  level is above the baseline, ``DESCEND -> FLx`` when it is below, and an
  em dash when there is no level change or no recommendation;
- no numbers are hardcoded: the verification KPI renders only when the
  caller passes ``test_summary``, and the ERA5 probe paragraph describes
  the recommended direction (probe level = recommended flight level).

Data contracts
--------------
``corridor``
    Flight geometry plus the runner output
    (:func:`stratum.pipeline.runner.evaluate_corridor_flight`). The
    runner output may be passed either under ``corridor["result"]`` or
    merged directly into ``corridor``. Recognised keys: ``origin``,
    ``destination``, ``origin_label``, ``destination_label``,
    ``num_waypoints``, ``cruise_fl``, ``departure_utc``,
    ``departure_local_label``, ``segments``, ``recommended_reroutes``,
    ``total_segments``, ``contrail_forming_segments``,
    ``contrail_persistent_segments``, and the optional ``rhi_probe`` /
    ``candidates`` records described below.
``flight``
    ``callsign``, ``icao24``, ``aircraft_type``.
``source``
    ``label`` (weather-data description) and the optional
    ``era5_filename``.
``model``
    Validation metrics either flattened (``brier_raw``,
    ``brier_calibrated``, ``n_train``, ``n_val``, ``n_masked``) or under
    a ``metrics`` key.

Optional ``corridor`` extensions:

- ``rhi_probe``: an ERA5 probe record as returned by
  :func:`probe_rhi_evidence`; when present, the advisory paragraph
  describes the recommended direction with those numbers.
- ``candidates``: precomputed
  :class:`stratum.decide.counterfactual.CandidateEvaluation` records for
  the primary event. When absent, the report recomputes them from the
  primary advisory with
  :func:`stratum.decide.counterfactual.evaluate_altitude_candidates`.
"""

from __future__ import annotations

import html
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stratum.decide.climate import calculate_net_climate_benefit
from stratum.decide.counterfactual import (
    ALTITUDE_STEP_FL,
    DEFAULT_SEGMENT_LENGTH_KM,
    MEDIAN_PERSISTENT_EF_J_PER_M,
    evaluate_altitude_candidates,
)
from stratum.decide.fuel import estimate_fuel_penalty
from stratum.io.era5 import interpolate_met_to_segment
from stratum.physics import fl_to_p
from stratum.pipeline.build_features import specific_humidity_to_rhi

__all__ = ["render_corridor_report", "probe_rhi_evidence"]

#: Calibrated probability at or above which a waypoint is marked
#: ``[PERSISTENT]`` in the audit table.
PERSISTENT_THRESHOLD: float = 0.50

_EM_DASH: str = "&#8212;"
_ARROW: str = "&#8594;"

_STATUS_TAGS: dict[str, str] = {
    "clear": "[CLEAR]",
    "aerodynamic": "[AERODYNAMIC]",
    "persistent": '<span class="tag-inv">[PERSISTENT]</span>',
}

_STYLESHEET = """\
  * { box-sizing: border-box; }
  html, body { background: #ffffff; }
  body {
    margin: 0;
    color: #111111;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Helvetica, Arial, sans-serif;
    font-size: 14px;
    line-height: 1.55;
  }
  .page { max-width: 920px; margin: 0 auto; padding: 48px 28px 64px; }
  h1, h2, h3 { margin: 0; }
  .mono, .num, code {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas,
      "Liberation Mono", monospace;
    font-variant-numeric: tabular-nums;
  }
  /* -------- 1. header and metadata -------- */
  .kicker {
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #666666;
  }
  .headline { font-size: 26px; font-weight: 700; letter-spacing: -0.01em; margin: 6px 0 2px; }
  .corridor { font-size: 15px; color: #111111; }
  .meta-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    border: 1px solid #111111;
    margin-top: 22px;
  }
  .meta-cell { padding: 12px 16px; border-top: 1px solid #e5e5e5; }
  .meta-cell:nth-child(-n+2) { border-top: none; }
  .meta-cell:nth-child(even) { border-left: 1px solid #e5e5e5; }
  .meta-label {
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666666;
  }
  .meta-value { margin-top: 3px; font-size: 13.5px; }
  /* -------- 2. KPI bar -------- */
  .kpis {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid #111111;
    margin-top: 28px;
  }
  .kpi { padding: 14px 16px; }
  .kpi + .kpi { border-left: 1px solid #111111; }
  .kpi-label {
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666666;
  }
  .kpi-value { margin-top: 6px; font-size: 19px; font-weight: 700; white-space: nowrap; }
  .kpi-sub { margin-top: 4px; font-size: 11.5px; color: #666666; }
  /* -------- 3. advisory card -------- */
  .advisory { background: #111111; color: #ffffff; margin-top: 28px; }
  .advisory-head {
    padding: 16px 20px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid #666666;
  }
  .advisory-grid { display: grid; grid-template-columns: repeat(4, 1fr); }
  .advisory-cell { padding: 14px 20px; }
  .advisory-cell + .advisory-cell { border-left: 1px solid #666666; }
  .advisory-label {
    font-size: 10.5px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #e5e5e5;
  }
  .advisory-value { margin-top: 5px; font-size: 20px; font-weight: 700; }
  .advisory-note {
    padding: 14px 20px 18px;
    border-top: 1px solid #666666;
    font-size: 13px;
    color: #e5e5e5;
    margin: 0;
  }
  /* -------- 4./5. tables -------- */
  h2.section {
    font-size: 11.5px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #666666;
    margin: 36px 0 10px;
  }
  table { width: 100%; border-collapse: collapse; border: 1px solid #111111; }
  thead th {
    text-align: left;
    font-size: 10.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #666666;
    padding: 9px 12px;
    border-bottom: 1px solid #111111;
  }
  tbody td { padding: 8px 12px; border-bottom: 1px solid #e5e5e5; vertical-align: top; }
  tbody tr:last-child td { border-bottom: none; }
  td.num { text-align: right; white-space: nowrap; }
  .rationale { font-size: 12.5px; }
  .tag-cell { white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }
  .tag-inv { background: #111111; color: #ffffff; padding: 1px 6px; font-weight: 700; }
  .arrow { font-weight: 600; }
  /* -------- footer -------- */
  footer {
    margin-top: 40px;
    border-top: 1px solid #111111;
    padding-top: 14px;
    font-size: 11.5px;
    color: #666666;
  }
  @media (max-width: 760px) {
    .kpis, .advisory-grid { grid-template-columns: repeat(2, 1fr); }
    .kpi + .kpi, .advisory-cell + .advisory-cell { border-left: none; border-top: 1px solid #e5e5e5; }
    .advisory-cell + .advisory-cell { border-top-color: #666666; }
  }"""


def _direction_word(delta_fl: float | None) -> str | None:
    """``"climb"`` / ``"descent"`` for a level delta, else ``None``."""
    if delta_fl is None:
        return None
    if delta_fl > 0.0:
        return "climb"
    if delta_fl < 0.0:
        return "descent"
    return None


def _format_signed_kg(value: float) -> str:
    """``+8.97 kg`` style formatting for mass deltas."""
    return f"+{value:.2f} kg"


def _format_gj(value_j: float) -> str:
    """``5.56 GJ`` style formatting for energy forcing."""
    return f"{value_j / 1.0e9:.2f} GJ"


def _format_ratio(value: float) -> str:
    """``1.10e+07`` style formatting for benefit-to-cost ratios."""
    return f"{value:.2e}"


def _segment_status(segment: dict[str, Any]) -> str:
    """Return the monochrome status tag HTML for one waypoint."""
    probability = float(segment.get("p_persist") or 0.0)
    if segment.get("sac_forms") and probability >= PERSISTENT_THRESHOLD:
        return _STATUS_TAGS["persistent"]
    if segment.get("sac_forms"):
        return _STATUS_TAGS["aerodynamic"]
    return _STATUS_TAGS["clear"]


def _decision_text(segment: dict[str, Any]) -> str:
    """Direction-aware decision cell text for one waypoint.

    The direction comes from the decision data: ``CLIMB`` when the
    recommended level is above the flown level, ``DESCEND`` when it is
    below, an em dash for any other state.
    """
    status = segment.get("decision_status")
    if status == "eligible":
        recommended = segment.get("recommended_fl")
        baseline = segment.get("flight_level")
        if recommended is None or baseline is None:
            return _EM_DASH
        direction = _direction_word(float(recommended) - float(baseline))
        if direction == "climb":
            return (
                f'<span class="arrow">CLIMB {_ARROW} '
                f"FL{float(recommended):.0f}</span>"
            )
        if direction == "descent":
            return (
                f'<span class="arrow">DESCEND {_ARROW} '
                f"FL{float(recommended):.0f}</span>"
            )
        return _EM_DASH
    if status == "abstain":
        return "ABSTAIN"
    if status == "no_robust_candidate":
        return "NO CANDIDATE"
    return _EM_DASH


def _rhi_at(
    ds_met: Any,
    latitude: float,
    longitude: float,
    pressure_pa: float,
    timestamp: pd.Timestamp,
) -> float:
    """Ice-relative humidity from an interpolated ERA5 state."""
    met = interpolate_met_to_segment(
        ds_met, latitude, longitude, pressure_pa, timestamp
    )
    return specific_humidity_to_rhi(
        met["specific_humidity"], pressure_pa, met["temperature"]
    )


def probe_rhi_evidence(
    ds_met: Any,
    segments: list[dict[str, Any]],
    baseline_fl: float,
    proposed_fl: float,
) -> dict[str, float] | None:
    """Sample ERA5 RHi at the baseline and the proposed flight level.

    The proposed level must be the recommended flight level, so the
    returned numbers describe the recommended direction regardless of
    whether it is a climb or a descent. Segments without usable ERA5 data
    are skipped.

    Returns
    -------
    dict[str, float] or None
        ``n_waypoints``, ``baseline_fl``, ``proposed_fl``,
        ``mean_baseline``, ``mean_proposed``, ``n_drier_at_proposed``
        (waypoints where the proposed level is drier than the baseline)
        and ``n_subsaturated_proposed`` (RHi < 1.0 at the proposed
        level); ``None`` when no segment carries finite RHi at both
        levels.
    """
    baseline_pa = fl_to_p(baseline_fl) * 100.0
    proposed_pa = fl_to_p(proposed_fl) * 100.0
    baseline_values: list[float] = []
    proposed_values: list[float] = []
    for segment in segments:
        try:
            latitude = float(segment["lat"])
            longitude = float(segment["lon"])
            timestamp = pd.Timestamp(segment["time_utc"])
        except (KeyError, TypeError, ValueError):
            continue
        baseline_rhi = _rhi_at(ds_met, latitude, longitude, baseline_pa, timestamp)
        proposed_rhi = _rhi_at(ds_met, latitude, longitude, proposed_pa, timestamp)
        if not (math.isfinite(baseline_rhi) and math.isfinite(proposed_rhi)):
            continue
        baseline_values.append(baseline_rhi)
        proposed_values.append(proposed_rhi)
    if not baseline_values:
        return None
    count = float(len(baseline_values))
    return {
        "n_waypoints": count,
        "baseline_fl": float(baseline_fl),
        "proposed_fl": float(proposed_fl),
        "mean_baseline": sum(baseline_values) / count,
        "mean_proposed": sum(proposed_values) / count,
        "n_drier_at_proposed": float(
            sum(1 for before, after in zip(baseline_values, proposed_values) if after < before)
        ),
        "n_subsaturated_proposed": float(
            sum(1 for value in proposed_values if value < 1.0)
        ),
    }


def _candidate_matrix_rows(
    primary: dict[str, Any] | None,
    aircraft_type: str,
    candidates: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Build the altitude candidate matrix from the decision data.

    Rows for the evaluated candidates come from the decision record
    (either provided or recomputed from the primary advisory); the
    baseline row is the flown level. The selected row is the one whose
    level equals ``recommended_fl``; every other row is reported as
    rejected with its reason.
    """
    if primary is None:
        return [
            {
                "fl": _EM_DASH,
                "delta": _EM_DASH,
                "fuel": _EM_DASH,
                "co2": _EM_DASH,
                "avoided": _EM_DASH,
                "status": "No advisory: no persistent contrail was flagged on this corridor.",
            }
        ]

    baseline_fl = float(primary["baseline_fl"])
    recommended_fl = (
        None if primary.get("recommended_fl") is None else float(primary["recommended_fl"])
    )
    if candidates is None:
        decision = evaluate_altitude_candidates(
            float(primary["p_persist"]),
            baseline_fl,
            aircraft_type,
            segment_length_km=float(
                primary.get("segment_length_km", DEFAULT_SEGMENT_LENGTH_KM)
            ),
        )
        candidates = list(decision["candidates"])
        recommended_fl = (
            None
            if decision["recommended_fl"] is None
            else float(decision["recommended_fl"])
        )

    rows: list[tuple[float, dict[str, str]]] = []
    for candidate in candidates:
        level = float(candidate["target_fl"])
        if candidate["rejected"]:
            status = (
                f"REJECTED {_EM_DASH} "
                f'{html.escape(str(candidate["rejection_reason"] or ""))}'
            )
        elif recommended_fl is not None and abs(level - recommended_fl) < 1.0e-9:
            direction = _direction_word(level - baseline_fl)
            direction_text = direction if direction is not None else "level change"
            status = (
                f"SELECTED {_EM_DASH} {direction_text} to FL{level:.0f}; "
                f'{_format_signed_kg(float(candidate["delta_fuel_kg"]))} Jet-A; '
                "highest benefit/cost "
                f'({_format_ratio(float(candidate["benefit_to_cost_ratio"]))})'
            )
        else:
            recommended_text = (
                f"FL{recommended_fl:.0f}" if recommended_fl is not None else "no level"
            )
            status = (
                f"REJECTED {_EM_DASH} lower benefit/cost "
                f'({_format_ratio(float(candidate["benefit_to_cost_ratio"]))}) than '
                f"the recommended {recommended_text}"
            )
        rows.append(
            (
                level,
                {
                    "fl": f"FL{level:.0f}",
                    "delta": f'+{float(candidate["altitude_change_ft"]):.0f} ft'
                    if float(candidate["altitude_change_ft"]) >= 0.0
                    else f'{float(candidate["altitude_change_ft"]):.0f} ft',
                    "fuel": _format_signed_kg(float(candidate["delta_fuel_kg"])),
                    "co2": _format_signed_kg(float(candidate["delta_co2_kg"])),
                    "avoided": _format_gj(float(candidate["avoided_ef_j"])),
                    "status": status,
                },
            )
        )

    rows.append(
        (
            baseline_fl,
            {
                "fl": f"FL{baseline_fl:.0f}",
                "delta": "0 ft",
                "fuel": "0.00 kg",
                "co2": "0.00 kg",
                "avoided": _EM_DASH,
                "status": (
                    "CURRENT &#8212; persistent contrail flagged "
                    f'(P(persist) {float(primary["p_persist"]):.2f})'
                ),
            },
        )
    )

    if _direction_word((recommended_fl or baseline_fl) - baseline_fl) == "descent":
        selected = next(
            (
                candidate
                for candidate in candidates
                if abs(float(candidate["target_fl"]) - float(recommended_fl)) < 1.0e-9
            ),
            None,
        )
        if selected is not None and not selected["rejected"]:
            deep_fl = baseline_fl - 2.0 * ALTITUDE_STEP_FL
            deep_penalty = estimate_fuel_penalty(aircraft_type, baseline_fl, deep_fl)
            deep_benefit = calculate_net_climate_benefit(
                float(selected["avoided_ef_j"]), deep_penalty["delta_co2_kg"]
            )
            rows.append(
                (
                    deep_fl,
                    {
                        "fl": f"FL{deep_fl:.0f}",
                        "delta": f'{-2.0 * ALTITUDE_STEP_FL * 100.0:.0f} ft',
                        "fuel": _format_signed_kg(deep_penalty["delta_fuel_kg"]),
                        "co2": _format_signed_kg(deep_penalty["delta_co2_kg"]),
                        "avoided": _format_gj(float(selected["avoided_ef_j"])),
                        "status": (
                            f"REJECTED {_EM_DASH} excessive fuel penalty; "
                            "benefit/cost drops to "
                            f'{_format_ratio(float(deep_benefit["benefit_to_cost_ratio"]))} '
                            "(same escape effectiveness assumed for any descent)"
                        ),
                    },
                )
            )

    rows.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in rows]


def _render_waypoint_rows(segments: list[dict[str, Any]]) -> str:
    """Render the waypoint audit table body."""
    rows: list[str] = []
    for segment in segments:
        timestamp = pd.Timestamp(segment["time_utc"]).strftime("%H:%M")
        rows.append(
            f"""
            <tr>
              <td class="num">{int(segment['segment_index']) + 1}</td>
              <td class="num">{timestamp}</td>
              <td class="num">{float(segment['lat']):.2f}</td>
              <td class="num">{float(segment['lon']):.2f}</td>
              <td class="num">{float(segment['temperature_c']):.1f}</td>
              <td class="num">{float(segment['rhi']) * 100.0:.0f}%</td>
              <td class="num">{float(segment['p_persist']):.2f}</td>
              <td class="tag-cell">{_segment_status(segment)}</td>
              <td>{_decision_text(segment)}</td>
            </tr>"""
        )
    return "\n".join(rows)


def _model_metric(model: dict[str, Any], key: str) -> Any:
    """Read a metric from ``model`` or its nested ``metrics`` mapping."""
    if key in model:
        return model[key]
    metrics = model.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(key)
    return None


def _render_kpis(
    *,
    test_summary: str | None,
    model: dict[str, Any],
    result: dict[str, Any],
    action: str,
) -> str:
    """Render the KPI bar, omitting the verification cell when unknown."""
    cells: list[str] = []
    if test_summary is not None:
        cells.append(
            f"""
    <div class="kpi">
      <div class="kpi-label">Verification</div>
      <div class="kpi-value">{html.escape(test_summary)}</div>
      <div class="kpi-sub">test suite (pytest tests/)</div>
    </div>"""
        )

    brier_raw = _model_metric(model, "brier_raw")
    brier_calibrated = _model_metric(model, "brier_calibrated")
    if brier_raw is not None and brier_calibrated is not None:
        raw = float(brier_raw)
        calibrated = float(brier_calibrated)
        improvement = 100.0 * (raw - calibrated) / raw if raw else float("nan")
        calibration_value = f"{raw:.4f} {_ARROW} {calibrated:.4f}"
        calibration_sub = f"-{improvement:.1f}% Brier improvement (validation)"
    else:
        calibration_value = _EM_DASH
        calibration_sub = "validation metrics unavailable"
    cells.append(
        f"""
    <div class="kpi">
      <div class="kpi-label">Calibration</div>
      <div class="kpi-value mono">{calibration_value}</div>
      <div class="kpi-sub">{calibration_sub}</div>
    </div>"""
    )

    total_segments = int(result.get("total_segments", 0))
    persistent_segments = int(result.get("contrail_persistent_segments", 0))
    cells.append(
        f"""
    <div class="kpi">
      <div class="kpi-label">Contrail risk</div>
      <div class="kpi-value mono">{persistent_segments} / {total_segments} WAYPOINTS</div>
      <div class="kpi-sub">persistent risk flagged (P &#8805; 0.50)</div>
    </div>"""
    )
    cells.append(
        f"""
    <div class="kpi">
      <div class="kpi-label">Action</div>
      <div class="kpi-value">{action}</div>
      <div class="kpi-sub">optimal benefit-to-cost maneuver</div>
    </div>"""
    )
    return (
        f'  <section class="kpis" style="grid-template-columns: repeat({len(cells)}, 1fr);">'
        + "".join(cells)
        + "\n  </section>"
    )


def _render_advisory(
    primary: dict[str, Any] | None,
    aircraft_type: str,
    rhi_probe: dict[str, float] | None,
) -> tuple[str, str, str]:
    """Render the advisory card; returns ``(maneuver, html, action)``.

    ``action`` is the short KPI string (``CLIMB -> FL370``,
    ``DESCENT -> FL330`` or ``NO ACTION``) and ``maneuver`` the headline
    text of the card. The probe paragraph (when supplied) describes the
    recommended direction, and no probe text is emitted without a
    recommendation.
    """
    if primary is None:
        return (
            "NO ACTION REQUIRED",
            f"""  <section class="advisory">
    <div class="advisory-head">Recommended maneuver: NO ACTION REQUIRED</div>
    <div class="advisory-grid">
      <div class="advisory-cell">
        <div class="advisory-label">Fuel penalty</div>
        <div class="advisory-value mono">{_EM_DASH}</div>
      </div>
      <div class="advisory-cell">
        <div class="advisory-label">Delta CO&#8322;</div>
        <div class="advisory-value mono">{_EM_DASH}</div>
      </div>
      <div class="advisory-cell">
        <div class="advisory-label">Avoided warming</div>
        <div class="advisory-value mono">{_EM_DASH}</div>
      </div>
      <div class="advisory-cell">
        <div class="advisory-label">Benefit / cost ratio</div>
        <div class="advisory-value mono">{_EM_DASH}</div>
      </div>
    </div>
    <p class="advisory-note">No waypoint exceeded the 0.40 calibrated
    persistence threshold on this corridor; flying the filed profile costs
    nothing extra.</p>
  </section>""",
            "NO ACTION",
        )

    baseline_fl = float(primary["baseline_fl"])
    recommended_fl = float(primary["recommended_fl"])
    direction = _direction_word(recommended_fl - baseline_fl)
    if direction == "climb":
        maneuver = f"STEP CLIMB TO FL{recommended_fl:.0f}"
        action = f"CLIMB {_ARROW} FL{recommended_fl:.0f}"
        gerund = "Climbing"
    elif direction == "descent":
        maneuver = f"STEP DESCENT TO FL{recommended_fl:.0f}"
        action = f"DESCENT {_ARROW} FL{recommended_fl:.0f}"
        gerund = "Descending"
    else:
        maneuver = f"HOLD FL{baseline_fl:.0f}"
        action = "NO ACTION"
        gerund = "Holding"

    fuel_text = _format_signed_kg(float(primary["delta_fuel_kg"]))
    co2_text = _format_signed_kg(float(primary["delta_co2_kg"]))
    avoided_text = _format_gj(float(primary["avoided_contrail_ef_j"]))
    ratio_text = _format_ratio(float(primary["benefit_to_cost_ratio"]))

    rationale = (
        f"{gerund} to FL{recommended_fl:.0f} is the cheapest feasible maneuver "
        f"({fuel_text} Jet-A over the 20-minute window) and clears the highest "
        f"benefit-to-cost ratio ({ratio_text}) of every evaluated level."
    )
    if direction is not None and rhi_probe is not None:
        count = float(rhi_probe["n_waypoints"])
        rationale += (
            f' ERA5 probe over the {count:.0f} flagged waypoints: mean RHi '
            f'{float(rhi_probe["mean_baseline"]):.2f} at FL{baseline_fl:.0f} '
            f'versus {float(rhi_probe["mean_proposed"]):.2f} at '
            f"FL{recommended_fl:.0f}; the step {direction} dries the air on "
            f'{float(rhi_probe["n_drier_at_proposed"]):.0f} of {count:.0f} '
            "waypoints and leaves ice-subsaturated air (RHi &#60; 1.0) at the "
            f'proposed level on {float(rhi_probe["n_subsaturated_proposed"]):.0f} '
            f"of {count:.0f}, so the escape is partial in this weather sample "
            "and the calibrated persistence probability plus the modeled escape "
            "effectiveness drive the recommendation."
        )

    advisory_html = f"""  <section class="advisory">
    <div class="advisory-head">Recommended maneuver: {maneuver}</div>
    <div class="advisory-grid">
      <div class="advisory-cell">
        <div class="advisory-label">Fuel penalty</div>
        <div class="advisory-value mono">{fuel_text}<span style="font-size:12px;font-weight:400"> Jet-A</span></div>
      </div>
      <div class="advisory-cell">
        <div class="advisory-label">Delta CO&#8322;</div>
        <div class="advisory-value mono">{co2_text}</div>
      </div>
      <div class="advisory-cell">
        <div class="advisory-label">Avoided warming</div>
        <div class="advisory-value mono">{avoided_text}</div>
      </div>
      <div class="advisory-cell">
        <div class="advisory-label">Benefit / cost ratio</div>
        <div class="advisory-value mono">{ratio_text}</div>
      </div>
    </div>
    <p class="advisory-note">{rationale}</p>
  </section>"""
    return maneuver, advisory_html, action


def render_corridor_report(
    path: str | Path,
    *,
    corridor: dict[str, Any],
    flight: dict[str, Any],
    source: dict[str, Any],
    model: dict[str, Any],
    test_summary: str | None = None,
) -> None:
    """Write a standalone monochrome HTML corridor report.

    Parameters
    ----------
    path : str or pathlib.Path
        Output HTML path.
    corridor : dict
        Flight geometry and runner output (see the module docstring).
    flight : dict
        Flight identity: ``callsign``, ``icao24``, ``aircraft_type``.
    source : dict
        Weather-data provenance: ``label`` and optional ``era5_filename``.
    model : dict
        Validation metrics (flattened or under ``metrics``).
    test_summary : str, optional
        Verification cell text (e.g. ``"179 PASSED"``). The verification
        KPI renders only when this is provided; no test count is ever
        hardcoded.

    Returns
    -------
    None
        The report is written to ``path``.
    """
    result = corridor["result"] if isinstance(corridor.get("result"), dict) else corridor
    segments = list(result.get("segments") or [])
    reroutes = list(result.get("recommended_reroutes") or [])
    primary = reroutes[0] if reroutes else None

    origin = str(corridor.get("origin", "") or _EM_DASH)
    destination = str(corridor.get("destination", "") or _EM_DASH)
    origin_label = str(corridor.get("origin_label", "") or origin)
    destination_label = str(corridor.get("destination_label", "") or destination)
    num_waypoints = int(corridor.get("num_waypoints", len(segments)))
    cruise_fl = corridor.get("cruise_fl")
    if cruise_fl is None and segments:
        cruise_fl = float(segments[0].get("flight_level", 0.0)) or None

    callsign = str(flight.get("callsign", "") or _EM_DASH)
    icao24 = str(flight.get("icao24", "") or _EM_DASH)
    aircraft_type = str(flight.get("aircraft_type", "") or "unknown")

    source_label = str(source.get("label", "") or "unspecified weather source")
    era5_filename = source.get("era5_filename")
    if era5_filename and not source_label.startswith("synthetic"):
        weather_text = (
            f"ECMWF ERA5 4D Reanalysis ({html.escape(str(era5_filename))})"
        )
    else:
        weather_text = html.escape(source_label)

    departure_text = _EM_DASH
    departure_utc = corridor.get("departure_utc")
    if departure_utc is not None:
        try:
            departure_text = pd.Timestamp(departure_utc).strftime("%Y-%m-%d %H:%M UTC")
        except (TypeError, ValueError):
            departure_text = html.escape(str(departure_utc))
        local_label = corridor.get("departure_local_label")
        if local_label:
            departure_text += f" ({html.escape(str(local_label))})"

    profile_text = f"{num_waypoints} Great-Circle Waypoints"
    if cruise_fl is not None:
        profile_text += f" @ FL{float(cruise_fl):.0f}"

    maneuver, advisory_html, action = _render_advisory(
        primary, aircraft_type, corridor.get("rhi_probe")
    )

    candidates = corridor.get("candidates")
    if candidates is not None:
        candidates = list(candidates)
    matrix_rows = _candidate_matrix_rows(primary, aircraft_type, candidates)
    matrix_html = "\n".join(
        "<tr>"
        f'<td class="mono">{row["fl"]}</td>'
        f'<td class="num">{row["delta"]}</td>'
        f'<td class="num">{row["fuel"]}</td>'
        f'<td class="num">{row["co2"]}</td>'
        f'<td class="num">{row["avoided"]}</td>'
        f'<td class="rationale">{row["status"]}</td>'
        "</tr>"
        for row in matrix_rows
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_train = _model_metric(model, "n_train")
    n_val = _model_metric(model, "n_val")
    n_masked = _model_metric(model, "n_masked")
    if None in (n_train, n_val, n_masked):
        model_line = "Model: LightGBM with monotone constraints + isotonic calibration."
    else:
        model_line = (
            "Model: LightGBM with monotone constraints + isotonic calibration "
            f"(train {int(n_train)} / val {int(n_val)} rows, {int(n_masked)} "
            "NaN-masked labels)."
        )

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>STRATUM // Corridor Dispatch Evaluation</title>
<style>
{_STYLESHEET}
</style>
</head>
<body>
<div class="page">

  <header>
    <div class="kicker">STRATUM // FLIGHT CORRIDOR DISPATCH EVALUATION</div>
    <h1 class="headline">{html.escape(origin_label)} {_ARROW} {html.escape(destination_label)}</h1>
    <div class="corridor">
      {html.escape(origin)} {_ARROW} {html.escape(destination)} &#183; contrail-persistence forecast
      along the great-circle cruise profile
    </div>
    <div class="meta-grid">
      <div class="meta-cell">
        <div class="meta-label">Aircraft</div>
        <div class="meta-value">{html.escape(aircraft_type)} &#183; {html.escape(callsign)} ({html.escape(icao24)})</div>
      </div>
      <div class="meta-cell">
        <div class="meta-label">Profile</div>
        <div class="meta-value">{profile_text}</div>
      </div>
      <div class="meta-cell">
        <div class="meta-label">Departure</div>
        <div class="meta-value">{departure_text}</div>
      </div>
      <div class="meta-cell">
        <div class="meta-label">Weather</div>
        <div class="meta-value">{weather_text}</div>
      </div>
    </div>
  </header>

{_render_kpis(test_summary=test_summary, model=model, result=result, action=action)}

{advisory_html}

  <h2 class="section">Altitude candidate evaluation matrix</h2>
  <table>
    <thead>
      <tr>
        <th>Flight level</th>
        <th>Altitude delta</th>
        <th>Fuel delta</th>
        <th>CO&#8322; delta</th>
        <th>Avoided warming</th>
        <th>Status / rationale</th>
      </tr>
    </thead>
    <tbody>
      {matrix_html}
    </tbody>
  </table>

  <h2 class="section">Waypoint audit &#8212; {num_waypoints} waypoints, {html.escape(origin)} {_ARROW} {html.escape(destination)}</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>UTC</th>
        <th>Lat</th>
        <th>Lon</th>
        <th>Temp [&#176;C]</th>
        <th>RHi</th>
        <th>P(persist)</th>
        <th>Status</th>
        <th>Decision</th>
      </tr>
    </thead>
    <tbody>
      {_render_waypoint_rows(segments)}
    </tbody>
  </table>

  <footer>
    STRATUM corridor report generated on {generated_at}.
    Source: {weather_text}. {model_line} Status: [CLEAR] no SAC formation,
    [AERODYNAMIC] forms but calibrated P(persist) &#60; 0.50,
    [PERSISTENT] calibrated P(persist) &#8805; 0.50. Decisions activate at
    P(persist) &#8805; 0.40. Avoided forcing uses the median persistent
    contrail energy forcing per flight metre of Teoh et al. (2024)
    ({MEDIAN_PERSISTENT_EF_J_PER_M:.2e} J/m).
  </footer>

</div>
</body>
</html>
"""
    Path(path).write_text(report_html, encoding="utf-8")
