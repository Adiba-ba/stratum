# Physics

This document records the formulas and constants as implemented. Every value below is read from `src/stratum/`.

## Schmidt-Appleman criterion

`stratum/physics/sac.py` implements the mixing-line slope, the two threshold temperatures, and one single-point evaluation.

Mixing-line slope, Schumann (1996) eq. (4), `compute_sac_slope`:

```
G = (c_p * p * ei_h2o) / (eps * q_fuel * (1 - eta))    [Pa/K]
```

Default arguments of `compute_sac_slope`: `eta=0.32`, `q_fuel=43.1e6` J/kg, `ei_h2o=1.25` kg/kg. The module-level constants `EI_H2O = 1.23`, `c_p = 1004.0`, `eps = 0.622`, `Q = 43.2e6` are the Step-1 verification set; `G`, `solve_TLM` and `evaluate_sac` keep them with `eta=0.30`.

`T_LM` solves the slope condition by bisection, Schumann (1996) eq. (10):

```
d(e_w)/dT (T_LM) = G
```

Bracket `[180.0, 273.15]` K, 200 bisection halvings. A `G` outside the bracket raises `ValueError`.

`T_LC` solves Schumann (1996) eq. (11) by bisection with the residual

```
residual(T) = rhw * e_w(T) - e_w(T_LM) - G * (T - T_LM)
```

over `[160.0, T_LM]` K. The code returns `T_LM` directly when `rhw >= 0.999`, where the mixing line is tangent at `T_LM`. `rhw` itself is converted from the ambient ice relative humidity:

```
rhw = rhi * e_i(T) / e_w(T)
```

`evaluate_sac` returns `forms = T_K < T_LC` and `persists = rhi > 1.0`, plus `TLM_C`, `TLC_C`, and `G`.

Pipeline use (`stratum/pipeline/build_features.py`): the thresholds are solved with the aircraft propulsion efficiency from the engine registry, for example `eta = 0.32` for A320. But the legacy `q_fuel = 43.2e6` J/kg and `ei_h2o = 1.23` kg/kg. The feature builder floors `rhw` at `1.0e-9` before `T_LC` is solved; `critical_T` itself also accepts exactly zero, which is what the route-profile evaluator passes when `RHi = 0`.

## Saturation curves and ISA

`stratum/physics/thermo.py` implements Murphy & Koop (2005).

Liquid water, eq. (10), computed in log space:

```
tanh_term = tanh(0.0415 * (T - 218.8))
inner     = 53.878 - 1331.22 / T - 9.44523 * ln(T) + 0.014025 * T
ln(e_w)   = 54.842763 - 6763.22 / T - 4.210 * ln(T) + 0.000367 * T
            + tanh_term * inner
```

The derivative `d_e_w_dT` is analytic, not a finite difference:

```
d(ln e_w)/dT = 6763.22 / T^2 - 4.210 / T + 0.000367
               + d(tanh_term)/dT * inner + tanh_term * d(inner)/dT
d(tanh_term)/dT = 0.0415 * (1 - tanh_term^2)
```

Ice, eq. (7):

```
ln(e_i) = 9.550426 - 5723.265 / T + 3.53068 * ln(T) - 0.00728332 * T
```

`fl_to_p` returns ISA pressure in hPa. Constants: `P0_PA = 101325.0`, `T0_K = 288.15`, `LAPSE_K_PER_M = 0.0065`, `H_TROPOPAUSE_M = 11000.0`, `T_TROPOPAUSE_K = 216.65`, `FT_TO_M = 0.3048`, `G0_M_PER_S2 = 9.80665`, `R_AIR_J_PER_KG_K = 8.31446 / 0.0289644`. Below 11 km the pressure follows the lapse law; above it the isothermal layer. The inverse conversion `pressure_to_altitude_m` reuses the same constants.

## Humidity, shear, and derived features

- `RHi`: `e = q * p / (eps + (1 - eps) * q)`, then `RHi = e / e_i(T)`, with `eps = 0.622`. Non-finite inputs return `nan`.
- `rhi_above` and `rhi_below` sample the same formula at `p - 25 hPa` and `p + 25 hPa` (`HUMIDITY_OFFSET_HPA = 25.0`). `rhi_grad = rhi_above - rhi_below`.
- `vertical_wind_shear = hypot(du, dv) / |dz|` between the two offset layers, with `dz` from the ISA inversion.
- `dist_to_tropopause_m = altitude_m - 16000.0`; the reference tropopause is the fixed constant `TROPOPAUSE_ALTITUDE_M = 16000.0`.
- `cloud_ice = 1.0` when `fraction_of_cloud_cover > 0.5`, else `0.0` (`CLOUD_ICE_COVER_THRESHOLD = 0.5`).
- `local_solar_time_hours = (hour + minute/60 + lon/15) mod 24`.

## Atmospheric diagnostics

`stratum/physics/diagnostics.py`:

```
Ri = (g / theta) * (d(theta)/dz) / ((du/dz)^2 + (dv/dz)^2)
theta = T * (100000 / p)^kappa,  kappa = R_D / C_P = 287.05 / 1004.0
```

The potential temperatures use `p - 2500` Pa above and `p + 2500` Pa below. Fallbacks: `Ri = 1.0` when `|dz| < 1.0` m, `Ri = 10.0` when the squared shear is below `1e-8`, and the result is clipped to `[-5.0, 10.0]`.

Solar zenith angle follows the standard NOAA approximation: day of year and fractional hour drive a sine series for declination and the equation of time; `time_offset = eq_time_min + 4 * lon`, the hour angle follows from solar time, and `cos(zenith) = sin(lat) sin(decl) + cos(lat) cos(decl) cos(hour_angle)`. The result is clipped before `acos`.

Engine registry (`AIRCRAFT_ENGINE_DATABASE`), used for `engine_efficiency_eta` and `log_soot_ei_n = log10(ei_n)`:

| Type | eta | ei_n [1/kg fuel] | q_fuel [J/kg] |
| --- | --- | --- | --- |
| A320 | 0.32 | 1.5e15 | 43.1e6 |
| A20N | 0.38 | 4.0e14 | 43.2e6 |
| B738 | 0.31 | 1.8e15 | 43.1e6 |
| B38M | 0.37 | 5.0e14 | 43.2e6 |
| B77W | 0.35 | 8.0e14 | 43.0e6 |
| DEFAULT | 0.33 | 1.0e15 | 43.1e6 |

## Detection physics

`stratum/detect/btd.py` computes the split-window brightness temperature difference `BTD = BT10.8 - BT12.0` [K]. `normalize_btd` removes a `15 x 15` median background and divides the residual by `1.4826 * MAD` of the residual, floored at `1e-3`.

`stratum/detect/filters.py` convolves the normalized field with eight oriented line kernels, keeps the maximum response and best angle per pixel, applies hysteresis thresholding with weak level `0.7 * threshold` (default `threshold = 2.0` robust sigma, `min_length_px = 8`), and labels connected components.

## Matching geometry

- `stratum/match/advect.py`: first-order tangent-plane drift on a sphere of radius `6371000.0` m: `d_lat = v * t / R`, `d_lon = u * t / (R * cos(lat))`, longitude wrapped to `[-180, 180)`.
- `stratum/match/parallax.py`: Meteosat-9 at `35786000.0` m over `45.5` deg E; the satellite-to-target ray is intersected with the spherical Earth and the near-side entry point gives the apparent surface position. The displacement is roughly `altitude * tan(viewing zenith)`.
- `stratum/match/observe.py`: a scene is observable only when `high_cloud_fraction < 0.5`, `viewing_zenith_angle_deg < 60.0`, and `opaque_cloud` is False.
- `stratum/match/label.py`: `haversine_km` uses the mean Earth radius. `match_segment_to_detections` matches when a detection centroid lies within `distance_tolerance_km = 10.0` km and its orientation agrees with the segment heading within `angle_tolerance_deg = 20.0` (mod 180 degrees). `assign_tristate_label` returns `1.0`, `0.0`, or `nan` (unobservable) from the match state and the number of observable slots.

## References used by this layer

1. Schumann, U. (1996). On conditions for contrail formation from aircraft exhausts. Meteorologische Zeitschrift, 5(1), 4-23.
2. Murphy, D. M., & Koop, T. (2005). Review of the vapour pressures of ice and supercooled water for atmospheric applications. Quarterly Journal of the Royal Meteorological Society, 131(608), 1539-1565.
3. Ponater, M., Marquart, S., Sausen, R., & Schumann, U. (2002). Contrails in a broad global climate model. Journal of Geophysical Research, 107(D13).
4. ICAO (1993). Manual of the ICAO Standard Atmosphere, 3rd ed., Doc 7488.
