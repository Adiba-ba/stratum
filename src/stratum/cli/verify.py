"""Verification and screening subcommands for the STRATUM command line.

Ports ``scripts/verify_baseline.py`` and ``scripts/screen_indian_routes.py``
without changing their computation logic. Each public function parses its
own arguments (the subcommand name has already been removed by
:mod:`stratum.cli.main`) and returns a process exit code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from stratum.physics import sac, thermo
from stratum.routes.corridor import (
    AIRPORTS,
    RouteProfileResult,
    evaluate_route_profile,
    great_circle_waypoints,
    piecewise_profile,
)

__all__ = ["screen_routes", "verify_baseline"]

P250_PA = 25000.0
DEFAULT_ETA = 0.30

#: Flight level, waypoint count, and engine efficiency of the screening.
SCREEN_FL: float = 340.0
SCREEN_NUM_POINTS: int = 25

SCREEN_CITIES: dict[str, str] = {
    "VIDP": "Delhi",
    "VABB": "Mumbai",
    "VOBL": "Bengaluru",
    "VECC": "Kolkata",
    "VAAH": "Ahmedabad",
}

WINTER: str = "Winter jet (Jan-Feb)"
SUMMER: str = "Summer/monsoon (Jul-Aug)"


@dataclasses.dataclass(frozen=True)
class _CheckRow:
    """One row of the baseline verification audit table."""

    name: str
    expected: str
    actual: str
    delta: str
    passed: bool


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One route screened under one atmospheric regime."""

    origin: str
    destination: str
    season: str
    temp_knots: tuple[tuple[float, float], ...]
    rhi_knots: tuple[tuple[float, float], ...]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "VIDP", "VABB", WINTER,
        temp_knots=((0.0, -52.0), (1.0, -44.0)),
        rhi_knots=((0.0, 0.45), (1.0, 1.15)),
    ),
    Scenario(
        "VIDP", "VECC", WINTER,
        temp_knots=((0.0, -54.0), (1.0, -47.0)),
        rhi_knots=((0.0, 0.65), (0.55, 1.00), (1.0, 1.25)),
    ),
    Scenario(
        "VABB", "VOBL", WINTER,
        temp_knots=((0.0, -38.0), (1.0, -34.0)),
        rhi_knots=((0.0, 0.30), (1.0, 0.75)),
    ),
    Scenario(
        "VIDP", "VAAH", WINTER,
        temp_knots=((0.0, -50.0), (1.0, -43.0)),
        rhi_knots=((0.0, 0.40), (1.0, 1.05)),
    ),
    Scenario(
        "VIDP", "VABB", SUMMER,
        temp_knots=((0.0, -35.0), (1.0, -38.0)),
        rhi_knots=((0.0, 0.55), (1.0, 0.95)),
    ),
    Scenario(
        "VIDP", "VECC", SUMMER,
        temp_knots=((0.0, -36.0), (1.0, -39.0)),
        rhi_knots=((0.0, 0.70), (1.0, 1.15)),
    ),
    Scenario(
        "VABB", "VOBL", SUMMER,
        temp_knots=((0.0, -35.0), (1.0, -37.0)),
        rhi_knots=((0.0, 0.45), (1.0, 0.85)),
    ),
    Scenario(
        "VIDP", "VAAH", SUMMER,
        temp_knots=((0.0, -35.0), (1.0, -38.0)),
        rhi_knots=((0.0, 0.40), (1.0, 0.80)),
    ),
)


def _baseline_rows(
    eta: float,
) -> tuple[list[_CheckRow], list[tuple[str, float, float]]]:
    """Recompute the baseline quantities and return the audit rows.

    Returns
    -------
    rows : list of _CheckRow
        One row per check, in the order the original script printed them.
    context : list of (label, TLC_C, TLM_C)
        Derived threshold context for the three SAC points.
    """
    rows: list[_CheckRow] = []

    def add_float(
        name: str,
        expected: float,
        actual: float,
        tolerance: float,
        fmt: str = "{:.10f}",
    ) -> None:
        ok = abs(actual - expected) <= tolerance
        rows.append(
            _CheckRow(
                name,
                fmt.format(expected),
                fmt.format(actual),
                f"{actual - expected:+.3e}",
                ok,
            )
        )

    def add_bool(name: str, expected: bool, actual: bool) -> None:
        rows.append(
            _CheckRow(name, str(expected), str(actual), "-", expected == actual)
        )

    g = sac.G(P250_PA, eta)
    add_float("G @ 250 hPa, eta=0.30 [Pa/K]", 1.6413706426, g, 1e-9)

    tlm_c = sac.solve_TLM(P250_PA, eta) - 273.15
    add_float("T_LM @ 250 hPa, eta=0.30 [degC]", -41.9128742, tlm_c, 0.01, fmt="{:.7f}")

    add_float("ISA pressure at FL250 [hPa]", 376.015, thermo.fl_to_p(250.0), 0.05)
    add_float("ISA pressure at FL340 [hPa]", 249.996, thermo.fl_to_p(340.0), 0.05)

    t_k = 231.0
    h = 1e-5
    fd = (thermo.e_w(t_k + h) - thermo.e_w(t_k - h)) / (2.0 * h)
    rel_err = abs(thermo.d_e_w_dT(t_k) - fd) / fd
    add_float(
        "d(e_w)/dT analytic rel. error @ 231 K [-]", 0.0, rel_err, 1e-8, fmt="{:.3e}"
    )

    points = [
        ("A", -52.0, 0.60, True, False),
        ("B", -50.0, 1.12, True, True),
        ("C", -38.0, 0.58, False, False),
    ]
    context: list[tuple[str, float, float]] = []
    for name, t_c, rhi, forms, persists in points:
        result = sac.evaluate_sac(P250_PA, t_c + 273.15, rhi, eta=eta)
        label = f"Point {name} (T={t_c:g} C, RHi={rhi:.0%})"
        add_bool(f"{label} forms", forms, result["forms"])
        add_bool(f"{label} persists", persists, result["persists"])
        context.append((label, float(result["TLC_C"]), float(result["TLM_C"])))

    return rows, context


def verify_baseline(argv: list[str]) -> int:
    """Run the baseline checks and print the audit table.

    Parameters
    ----------
    argv : list of str
        Arguments after ``verify-baseline``; supports ``--json``.

    Returns
    -------
    int
        0 when every check passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="stratum verify-baseline",
        description=(
            "Recompute the STRATUM baseline quantities and compare them with "
            "the audited expected values."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write machine-readable results to stdout instead of a table",
    )
    args = parser.parse_args(argv)

    rows, context = _baseline_rows(DEFAULT_ETA)
    failures = [row for row in rows if not row.passed]
    passed = not failures

    if args.json:
        payload = {
            "passed": passed,
            "checks_passed": len(rows) - len(failures),
            "checks_total": len(rows),
            "checks": [
                {
                    "name": row.name,
                    "expected": row.expected,
                    "actual": row.actual,
                    "delta": row.delta,
                    "passed": row.passed,
                }
                for row in rows
            ],
            "context": [
                {"label": label, "tlc_c": tlc_c, "tlm_c": tlm_c}
                for label, tlc_c, tlm_c in context
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1

    check_w = max(len(row.name) for row in rows)
    expected_w = max(len(row.expected) for row in rows)
    actual_w = max(len(row.actual) for row in rows)
    delta_w = max(len(row.delta) for row in rows)

    header = (
        f"{'CHECK':<{check_w}}  {'EXPECTED':>{expected_w}}  "
        f"{'ACTUAL':>{actual_w}}  {'DELTA':>{delta_w}}  STATUS"
    )
    print("STRATUM baseline verification")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.name:<{check_w}}  {row.expected:>{expected_w}}  "
            f"{row.actual:>{actual_w}}  {row.delta:>{delta_w}}  "
            f"{'PASS' if row.passed else 'FAIL'}"
        )

    print()
    print("Context (derived thresholds, 250 hPa, eta=0.30):")
    for label, tlc_c, tlm_c_ctx in context:
        print(f"  {label}: T_LC = {tlc_c:.3f} C (T_LM = {tlm_c_ctx:.3f} C)")

    print()
    print(f"Checks passed: {len(rows) - len(failures)} / {len(rows)}")
    if failures:
        print("Baseline verification FAILED")
        return 1
    print("Baseline verification PASSED")
    return 0


def _explanation(result: RouteProfileResult) -> str:
    """Return a one-sentence physical explanation for a route result."""
    total = result["total_waypoints"]
    formation = result["formation_count"]
    persistence = result["persistence_count"]
    details = result["details"]
    # Persistent contrails require SAC and ice supersaturation at the same
    # waypoint; the two sets can be disjoint on a warming, moistening route.
    both = sum(1 for detail in details if detail["forms"] and detail["persists"])
    coldest_tlc = min(float(detail["TLC_C"]) for detail in details)

    if formation == 0 and persistence == 0:
        return (
            "No formation, no persistence: ambient T stays warmer than T_LC "
            f"(coldest threshold {coldest_tlc:.1f} C) and RHi stays <= 100%."
        )
    if formation == 0:
        return (
            "No formation: T > T_LC everywhere "
            f"(coldest threshold {coldest_tlc:.1f} C); ice-supersaturated air "
            f"at {persistence}/{total} waypoints is too warm for the SAC."
        )
    if persistence == 0:
        return (
            f"SAC holds on {formation}/{total} cold waypoints, but RHi stays "
            "<= 100% there: no ice supersaturation, contrails do not persist."
        )
    if both == formation == persistence:
        return (
            f"Persistent on {both}/{total} waypoints: T < T_LC coincides with "
            "ice-supersaturated (RHi > 100%) air."
        )
    if both == persistence:
        return (
            f"Persistent on {persistence}/{total} ice-supersaturated waypoints; "
            f"SAC additionally holds on {formation - both} cold, dry waypoints."
        )
    if both == formation:
        return (
            f"Persistent on all {formation} formation waypoints; a further "
            f"{persistence - both} ice-supersaturated waypoints stay above T_LC."
        )
    if both == 0:
        return (
            f"Formation ({formation}/{total}, cold-dry segment) and ice "
            f"supersaturation ({persistence}/{total}, warm-moist segment) do "
            "not overlap: no waypoint satisfies both, so no persistent "
            "contrail is supported."
        )
    return (
        f"SAC holds on {formation}/{total} and ice supersaturation on "
        f"{persistence}/{total} waypoints; {both} satisfy both, where "
        "persistent contrails occur."
    )


def screen_routes(argv: list[str]) -> int:
    """Screen major Indian corridors and print a Markdown summary table.

    Parameters
    ----------
    argv : list of str
        Arguments after ``screen-routes``; supports ``--eta`` and ``--out``.

    Returns
    -------
    int
        Always 0. The screening is descriptive, so it has no pass/fail gate.
    """
    parser = argparse.ArgumentParser(
        prog="stratum screen-routes",
        description=(
            "Screen major Indian flight corridors for contrail formation and "
            "persistence with the Schmidt-Appleman criterion."
        ),
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=DEFAULT_ETA,
        help=f"engine efficiency used by the SAC (default {DEFAULT_ETA:.2f})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write the Markdown table to this file",
    )
    args = parser.parse_args(argv)

    lines: list[str] = []
    p_340 = thermo.fl_to_p(SCREEN_FL)
    lines.append(
        "Synthetic scenario screening - not real data; every row uses "
        "fixed, invented temperature and RHi knots."
    )
    lines.append("Project STRATUM - Indian corridor contrail screening")
    lines.append(
        f"FL{SCREEN_FL:.0f} ({p_340:.1f} hPa ISA), {SCREEN_NUM_POINTS} "
        f"great-circle waypoints per route, engine efficiency eta = {args.eta:.2f}"
    )
    lines.append("")
    lines.append(
        "| Route | Season/Regime | Formation % | Persistence % | "
        "Physical Explanation |"
    )
    lines.append("| --- | --- | --- | --- | --- |")

    for scenario in SCENARIOS:
        waypoints = great_circle_waypoints(
            AIRPORTS[scenario.origin], AIRPORTS[scenario.destination],
            num_points=SCREEN_NUM_POINTS, fl=SCREEN_FL,
        )
        temps = piecewise_profile(scenario.temp_knots, SCREEN_NUM_POINTS)
        rhi_values = piecewise_profile(scenario.rhi_knots, SCREEN_NUM_POINTS)
        result = evaluate_route_profile(waypoints, temps, rhi_values, eta=args.eta)

        route = (
            f"{scenario.origin}-{scenario.destination} "
            f"({SCREEN_CITIES[scenario.origin]}-{SCREEN_CITIES[scenario.destination]})"
        )
        lines.append(
            f"| {route} | {scenario.season} | {result['formation_pct']:.1f} | "
            f"{result['persistence_pct']:.1f} | {_explanation(result)} |"
        )

    text = "\n".join(lines)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0
