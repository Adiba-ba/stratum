#!/usr/bin/env python3
"""Auditable baseline verification for Project STRATUM.

Recomputes the STRATUM verification quantities (mixing slope G, T_LM,
SAC points A/B/C, and ISA pressure conversion) and prints a comparison
table to stdout. Exit code is 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stratum.physics import sac, thermo  # noqa: E402

P250_PA = 25000.0
ETA = 0.30

Row = tuple[str, str, str, str, bool]


def main() -> int:
    """Run the baseline checks and print the audit table."""
    rows: list[Row] = []

    def add_float(
        name: str,
        expected: float,
        actual: float,
        tolerance: float,
        fmt: str = "{:.10f}",
    ) -> None:
        ok = abs(actual - expected) <= tolerance
        rows.append(
            (name, fmt.format(expected), fmt.format(actual), f"{actual - expected:+.3e}", ok)
        )

    def add_bool(name: str, expected: bool, actual: bool) -> None:
        rows.append((name, str(expected), str(actual), "-", expected == actual))

    g = sac.G(P250_PA, ETA)
    add_float("G @ 250 hPa, eta=0.30 [Pa/K]", 1.6413706426, g, 1e-9)

    tlm_c = sac.solve_TLM(P250_PA, ETA) - 273.15
    add_float("T_LM @ 250 hPa, eta=0.30 [degC]", -41.9128742, tlm_c, 0.01, fmt="{:.7f}")

    add_float("ISA pressure at FL250 [hPa]", 376.015, thermo.fl_to_p(250.0), 0.05)
    add_float("ISA pressure at FL340 [hPa]", 249.996, thermo.fl_to_p(340.0), 0.05)

    t_k = 231.0
    h = 1e-5
    fd = (thermo.e_w(t_k + h) - thermo.e_w(t_k - h)) / (2.0 * h)
    rel_err = abs(thermo.d_e_w_dT(t_k) - fd) / fd
    add_float("d(e_w)/dT analytic rel. error @ 231 K [-]", 0.0, rel_err, 1e-8, fmt="{:.3e}")

    points = [
        ("A", -52.0, 0.60, True, False),
        ("B", -50.0, 1.12, True, True),
        ("C", -38.0, 0.58, False, False),
    ]
    context: list[tuple[str, float, float]] = []
    for name, t_c, rhi, forms, persists in points:
        result = sac.evaluate_sac(P250_PA, t_c + 273.15, rhi, eta=ETA)
        label = f"Point {name} (T={t_c:g} C, RHi={rhi:.0%})"
        add_bool(f"{label} forms", forms, result["forms"])
        add_bool(f"{label} persists", persists, result["persists"])
        context.append((label, float(result["TLC_C"]), float(result["TLM_C"])))

    check_w = max(len(r[0]) for r in rows)
    expected_w = max(len(r[1]) for r in rows)
    actual_w = max(len(r[2]) for r in rows)
    delta_w = max(len(r[3]) for r in rows)

    header = (
        f"{'CHECK':<{check_w}}  {'EXPECTED':>{expected_w}}  "
        f"{'ACTUAL':>{actual_w}}  {'DELTA':>{delta_w}}  STATUS"
    )
    print("STRATUM baseline verification")
    print(header)
    print("-" * len(header))
    for name, expected, actual, delta, ok in rows:
        print(
            f"{name:<{check_w}}  {expected:>{expected_w}}  "
            f"{actual:>{actual_w}}  {delta:>{delta_w}}  {'PASS' if ok else 'FAIL'}"
        )

    print()
    print("Context (derived thresholds, 250 hPa, eta=0.30):")
    for label, tlc_c, tlm_c_ctx in context:
        print(f"  {label}: T_LC = {tlc_c:.3f} C (T_LM = {tlm_c_ctx:.3f} C)")

    failures = [r for r in rows if not r[4]]
    print()
    print(f"Checks passed: {len(rows) - len(failures)} / {len(rows)}")
    if failures:
        print("Baseline verification FAILED")
        return 1
    print("Baseline verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
