"""Schmidt-Appleman criterion (SAC) for contrail formation.

Implements the mixing-line slope ``G``, the liquid saturation threshold
temperature ``T_LM``, the critical ambient temperature ``T_LC``, and a
single-point evaluation of the Schmidt-Appleman criterion for Project
STRATUM.

Engine microphysics enters through three parameters: the propulsion
efficiency ``eta``, the fuel lower heating value ``q_fuel`` [J/kg], and
the water emission index ``ei_h2o`` [kg(H2O)/kg(fuel)]. The generic entry
point :func:`compute_sac_slope` carries the engine-registry defaults
(``eta=0.32``, ``q_fuel=43.1e6``, ``ei_h2o=1.25``). The module constants
:data:`EI_H2O` (1.23) and :data:`Q` (43.2e6) are the Step-1 verification
set; :func:`G`, :func:`solve_TLM` and :func:`evaluate_sac` keep those as
their defaults (with ``eta=0.30``) so the pinned Step-1 quantities are
unchanged. Callers that know the aircraft pass the registry values from
:func:`stratum.physics.diagnostics.get_aircraft_params` explicitly.

References
----------
Schumann, U. (1996). On conditions for contrail formation from aircraft
exhausts. Meteorologische Zeitschrift, 5(1), 4-23.

Ponater, M., Marquart, S., Sausen, R., & Schumann, U. (2002). Contrails
in a comprehensive global climate model. Journal of Geophysical
Research, 107(D13).
"""

from __future__ import annotations

from stratum.physics import thermo

__all__ = [
    "EI_H2O",
    "c_p",
    "eps",
    "Q",
    "compute_sac_slope",
    "G",
    "solve_TLM",
    "critical_T",
    "evaluate_sac",
]

#: Emission index of water vapor [kg(H2O)/kg(fuel)].
EI_H2O: float = 1.23
#: Specific heat capacity of ambient air [J/(kg*K)].
c_p: float = 1004.0
#: Ratio of molar masses of water vapor and dry air [-].
eps: float = 0.622
#: Specific combustion heat of the fuel [J/kg].
Q: float = 43.2e6

#: Below this water relative humidity the solution of eq. (11) is used;
#: at or above it the mixing line is tangent at T_LM and T_LC = T_LM.
_RHW_TANGENT_LEVEL: float = 0.999

#: Temperature bracket [K] containing the unique solution of
#: ``d(e_w)/dT = G`` for all physically relevant slopes.
_TLM_BRACKET_K: tuple[float, float] = (180.0, 273.15)
#: Cold bound [K] of the bracket for the T_LC root search.
_TLC_COLD_BOUND_K: float = 160.0
#: Number of bisection halvings (enough to reach double precision).
_BISECTION_ITERATIONS: int = 200

_KELVIN_OFFSET: float = 273.15


def compute_sac_slope(
    pressure_pa: float,
    eta: float = 0.32,
    q_fuel: float = 43.1e6,
    ei_h2o: float = 1.25,
) -> float:
    """Mixing-line slope G [Pa/K] for the given engine parameters.

    ``G = (c_p * p * ei_h2o) / (eps * q_fuel * (1 - eta))`` after
    Schumann (1996), eq. (4). ``eta`` is the propulsion efficiency
    (0-1), ``q_fuel`` the fuel lower heating value [J/kg], and
    ``ei_h2o`` the water emission index [kg(H2O)/kg(fuel)].
    """
    return (c_p * pressure_pa * ei_h2o) / (eps * q_fuel * (1.0 - eta))


def G(
    p_Pa: float,
    eta: float = 0.30,
    q_fuel: float = Q,
    ei_h2o: float = EI_H2O,
) -> float:
    """Mixing-line slope G [Pa/K] at ambient pressure ``p_Pa`` [Pa].

    Legacy entry point kept numerically identical to the Step-1
    verification set: default ``eta=0.30``, ``q_fuel=Q``, and
    ``ei_h2o=EI_H2O``. Engine-registry callers use
    :func:`compute_sac_slope` or pass the aircraft values explicitly.
    Schumann (1996), eq. (4).
    """
    return compute_sac_slope(p_Pa, eta=eta, q_fuel=q_fuel, ei_h2o=ei_h2o)


def solve_TLM(
    p_Pa: float,
    eta: float = 0.30,
    q_fuel: float = Q,
    ei_h2o: float = EI_H2O,
) -> float:
    """Temperature T_LM [K] where the saturation curve has slope G.

    Solves ``d(e_w)/dT(T) = G(p_Pa, eta, q_fuel, ei_h2o)`` by bisection.
    Schumann (1996), eq. (10). T_LM is the warmest temperature at which
    the plume mixing line is saturated (100% RH over liquid). Defaults
    match the Step-1 verification set.
    """
    g = G(p_Pa, eta, q_fuel=q_fuel, ei_h2o=ei_h2o)
    lo, hi = _TLM_BRACKET_K
    if not (thermo.d_e_w_dT(lo) < g < thermo.d_e_w_dT(hi)):
        raise ValueError(
            "G={g:.6f} Pa/K outside the solved bracket for p={p:.1f} Pa".format(
                g=g, p=p_Pa
            )
        )
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if thermo.d_e_w_dT(mid) < g:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def critical_T(rhw: float, TLM: float, G: float) -> float:
    """Critical ambient temperature T_LC [K] for water RH ``rhw`` [-].

    Solves Schumann (1996), eq. (11)::

        rhw * e_w(T_LC) = e_w(TLM) + G * (T_LC - TLM)

    The Schmidt-Appleman criterion holds for ambient temperatures below
    T_LC. For ``rhw >= 0.999`` the mixing line is tangent to the
    saturation curve at T_LM itself, so T_LM is returned (Schumann 1996,
    p. 10).
    """
    if rhw <= 0.0:
        raise ValueError(f"rhw must be positive, got {rhw!r}")
    if rhw >= _RHW_TANGENT_LEVEL:
        return TLM

    e_w_tlm = thermo.e_w(TLM)

    def residual(t: float) -> float:
        return rhw * thermo.e_w(t) - e_w_tlm - G * (t - TLM)

    # residual > 0 at the cold bound and < 0 at T_LM; the root between the
    # two is the relevant (cold) crossing.
    lo, hi = _TLC_COLD_BOUND_K, TLM
    if residual(lo) <= 0.0 or residual(hi) >= 0.0:
        return TLM
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate_sac(
    p_Pa: float,
    T_K: float,
    rhi: float,
    eta: float = 0.30,
    q_fuel: float = Q,
    ei_h2o: float = EI_H2O,
) -> dict[str, float | bool]:
    """Evaluate the Schmidt-Appleman criterion at a single point.

    Parameters
    ----------
    p_Pa : float
        Ambient pressure [Pa].
    T_K : float
        Ambient temperature [K].
    rhi : float
        Ambient relative humidity over ice [-]. It is converted to
        relative humidity over water before evaluating the SAC, because
        the criterion is defined against the liquid saturation curve.
    eta : float, optional
        Propulsion efficiency (0-1), by default 0.30 (Step-1 set).
    q_fuel : float, optional
        Fuel lower heating value [J/kg], by default :data:`Q`.
    ei_h2o : float, optional
        Water emission index [kg/kg], by default :data:`EI_H2O`.

    Returns
    -------
    dict
        Mapping with keys:

        - ``forms``: bool, True if the SAC holds (contrail forms).
        - ``persists``: bool, True if the ambient air is
          ice-supersaturated (``rhi`` > 1).
        - ``TLM_C``: float, liquid-hat threshold temperature [deg C].
        - ``TLC_C``: float, critical ambient temperature [deg C].
        - ``G``: float, mixing-line slope [Pa/K].
    """
    g = G(p_Pa, eta, q_fuel=q_fuel, ei_h2o=ei_h2o)
    tlm_k = solve_TLM(p_Pa, eta, q_fuel=q_fuel, ei_h2o=ei_h2o)
    rhw = rhi * thermo.e_i(T_K) / thermo.e_w(T_K)
    tlc_k = critical_T(rhw, tlm_k, g)
    return {
        "forms": T_K < tlc_k,
        "persists": rhi > 1.0,
        "TLM_C": tlm_k - _KELVIN_OFFSET,
        "TLC_C": tlc_k - _KELVIN_OFFSET,
        "G": g,
    }
