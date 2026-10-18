"""Fuel penalty estimates for counterfactual altitude changes.

Section 14.4 of the STRATUM paper: a 2,000 ft step climb of a typical
narrow-body transport (A320/B737, nominal cruise burn ~0.65 kg/s) costs
roughly +1.5-2.5% of the fuel burned during the maneuver window, while a
2,000 ft descent costs +0.8-1.5% because of the higher air density at the
lower level. Both alternatives burn *extra* fuel; the decision layer must
weigh that cost against the avoided contrail energy forcing.
"""

from __future__ import annotations

__all__ = [
    "CO2_PER_KG_FUEL",
    "AIRCRAFT_CRUISE_BURN_KG_S",
    "estimate_fuel_penalty",
]

#: CO2 released per kilogram of burned jet fuel [kg(CO2)/kg(fuel)]
#: (Section 14.4).
CO2_PER_KG_FUEL: float = 3.16

#: Nominal cruise fuel burn [kg/s] per supported aircraft type.
AIRCRAFT_CRUISE_BURN_KG_S: dict[str, float] = {
    "A320": 0.65,
    "B737": 0.65,
}

#: Midpoints [%] of the Section 14.4 penalty ranges, per 2,000 ft change.
_CLIMB_PENALTY_PCT_PER_2000FT: float = 2.0
_DESCENT_PENALTY_PCT_PER_2000FT: float = 1.15

_FT_PER_FL: float = 100.0
_REFERENCE_STEP_FT: float = 2000.0
_SECONDS_PER_MINUTE: float = 60.0


def estimate_fuel_penalty(
    aircraft_type: str,
    baseline_fl: float,
    target_fl: float,
    duration_min: float = 20.0,
) -> dict[str, float]:
    """Estimate the extra fuel burned by a 2,000 ft altitude change.

    Flight levels are given in hundreds of feet (FL350 -> ``350.0``).
    The penalty applies to the fuel burned during ``duration_min`` at the
    nominal cruise burn of ``aircraft_type``. Both a step climb and a
    descent burn extra fuel relative to level cruise at the baseline
    flight level.

    Parameters
    ----------
    aircraft_type : str
        Supported types: ``"A320"``, ``"B737"`` (case-insensitive).
    baseline_fl : float
        Current flight level, in hundreds of feet.
    target_fl : float
        Counterfactual flight level, in hundreds of feet.
    duration_min : float, optional
        Analysis window after the level change [min], by default 20.

    Returns
    -------
    dict[str, float]
        - ``delta_fuel_kg``: extra fuel [kg], always >= 0.
        - ``delta_co2_kg``: ``delta_fuel_kg * CO2_PER_KG_FUEL`` [kg].
        - ``penalty_pct``: penalty relative to window fuel [%].

    Raises
    ------
    ValueError
        For an unsupported aircraft type or non-positive inputs.
    """
    key = aircraft_type.strip().upper()
    if key not in AIRCRAFT_CRUISE_BURN_KG_S:
        supported = ", ".join(sorted(AIRCRAFT_CRUISE_BURN_KG_S))
        raise ValueError(
            f"unsupported aircraft_type {aircraft_type!r}; expected one of {supported}"
        )
    if baseline_fl <= 0.0 or target_fl <= 0.0:
        raise ValueError(
            f"flight levels must be positive, got {baseline_fl!r} and {target_fl!r}"
        )
    if duration_min <= 0.0:
        raise ValueError(f"duration_min must be positive, got {duration_min!r}")

    change_ft = (target_fl - baseline_fl) * _FT_PER_FL
    if change_ft == 0.0:
        return {"delta_fuel_kg": 0.0, "delta_co2_kg": 0.0, "penalty_pct": 0.0}

    window_fuel_kg = (
        AIRCRAFT_CRUISE_BURN_KG_S[key] * duration_min * _SECONDS_PER_MINUTE
    )
    magnitude = abs(change_ft) / _REFERENCE_STEP_FT
    if change_ft > 0.0:
        base_pct = _CLIMB_PENALTY_PCT_PER_2000FT
    else:
        base_pct = _DESCENT_PENALTY_PCT_PER_2000FT
    penalty_pct = base_pct * magnitude
    delta_fuel_kg = window_fuel_kg * penalty_pct / 100.0
    return {
        "delta_fuel_kg": delta_fuel_kg,
        "delta_co2_kg": delta_fuel_kg * CO2_PER_KG_FUEL,
        "penalty_pct": penalty_pct,
    }
