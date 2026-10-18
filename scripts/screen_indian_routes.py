#!/usr/bin/env python3
"""Screen major Indian flight corridors for contrail formation and persistence.

Runs the Schmidt-Appleman criterion (SAC) along FL340 great-circle tracks
for four trunk routes under two atmospheric regimes:

A. Winter subtropical-jet regime (Jan-Feb).
B. Summer/monsoon regime (Jul-Aug).

Ambient profiles are prescribed as piecewise-linear (fraction, value)
knots per route. Output is a Markdown summary table on stdout.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratum.physics import thermo  # noqa: E402
from stratum.routes.corridor import (  # noqa: E402
    AIRPORTS,
    RouteProfileResult,
    evaluate_route_profile,
    great_circle_waypoints,
    piecewise_profile,
)

FL: float = 340.0
NUM_POINTS: int = 25
ETA: float = 0.30

CITIES: dict[str, str] = {
    "VIDP": "Delhi",
    "VABB": "Mumbai",
    "VOBL": "Bengaluru",
    "VECC": "Kolkata",
    "VAAH": "Ahmedabad",
}

WINTER: str = "Winter jet (Jan-Feb)"
SUMMER: str = "Summer/monsoon (Jul-Aug)"


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


def main() -> int:
    """Execute the screening and print the Markdown summary table."""
    p_340 = thermo.fl_to_p(FL)
    print("Project STRATUM - Indian corridor contrail screening")
    print(
        f"FL{FL:.0f} ({p_340:.1f} hPa ISA), {NUM_POINTS} great-circle waypoints "
        f"per route, engine efficiency eta = {ETA:.2f}"
    )
    print()
    print("| Route | Season/Regime | Formation % | Persistence % | Physical Explanation |")
    print("| --- | --- | --- | --- | --- |")

    for scenario in SCENARIOS:
        waypoints = great_circle_waypoints(
            AIRPORTS[scenario.origin], AIRPORTS[scenario.destination],
            num_points=NUM_POINTS, fl=FL,
        )
        temps = piecewise_profile(scenario.temp_knots, NUM_POINTS)
        rhi_values = piecewise_profile(scenario.rhi_knots, NUM_POINTS)
        result = evaluate_route_profile(waypoints, temps, rhi_values, eta=ETA)

        route = (
            f"{scenario.origin}-{scenario.destination} "
            f"({CITIES[scenario.origin]}-{CITIES[scenario.destination]})"
        )
        print(
            f"| {route} | {scenario.season} | {result['formation_pct']:.1f} | "
            f"{result['persistence_pct']:.1f} | {_explanation(result)} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
