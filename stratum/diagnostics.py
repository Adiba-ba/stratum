"""Backward-compatibility shim for the diagnostics module.

The atmospheric diagnostics and aircraft registry moved to
:mod:`stratum.physics.diagnostics` in the engine-microphysics upgrade.
This module re-exports the public names so existing imports keep working.
"""

from __future__ import annotations

from stratum.physics.diagnostics import (
    AIRCRAFT_ENGINE_DATABASE,
    C_P,
    G,
    KAPPA,
    R_D,
    compute_richardson_number,
    compute_solar_zenith_angle,
    get_aircraft_params,
)

__all__ = [
    "AIRCRAFT_ENGINE_DATABASE",
    "C_P",
    "G",
    "KAPPA",
    "R_D",
    "compute_richardson_number",
    "compute_solar_zenith_angle",
    "get_aircraft_params",
]
