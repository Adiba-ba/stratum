"""Physics subpackage for Project STRATUM.

Re-exports the thermodynamic primitives and Schmidt-Appleman criterion
routines used across the project.
"""

from stratum.physics.sac import (
    EI_H2O,
    Q,
    c_p,
    compute_sac_slope,
    critical_T,
    eps,
    evaluate_sac,
    G,
    solve_TLM,
)
from stratum.physics.diagnostics import (
    AIRCRAFT_ENGINE_DATABASE,
    compute_richardson_number,
    compute_solar_zenith_angle,
    get_aircraft_params,
)
from stratum.physics.thermo import d_e_w_dT, e_i, e_w, fl_to_p

__all__ = [
    "AIRCRAFT_ENGINE_DATABASE",
    "EI_H2O",
    "Q",
    "c_p",
    "eps",
    "compute_sac_slope",
    "G",
    "solve_TLM",
    "critical_T",
    "evaluate_sac",
    "compute_richardson_number",
    "compute_solar_zenith_angle",
    "get_aircraft_params",
    "e_w",
    "e_i",
    "d_e_w_dT",
    "fl_to_p",
]
