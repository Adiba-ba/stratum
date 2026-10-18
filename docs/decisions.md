# Decisions

The decision layer lives in `src/stratum/decide/`. It answers one question per flagged corridor segment: is a 2,000 ft level change worth its fuel and CO2 cost, given the calibrated persistence probability?

## Candidate logic

`evaluate_altitude_candidates(baseline_p_persist, baseline_fl, aircraft_type, segment_length_km)`:

- Candidates are the two levels `baseline_fl +/- ALTITUDE_STEP_FL` where `ALTITUDE_STEP_FL = 20.0` (2,000 ft). No other offsets are tested.
- The allowed band is `MIN_ALLOWED_FL = 270.0` to `MAX_ALLOWED_FL = 390.0`. Candidates outside it are rejected with a reason and carry zero benefit.
- The decision status is `abstain` when `p_persist < ABSTAIN_THRESHOLD (0.40)`, `eligible` when at least one candidate has a positive net climate benefit, and `no_robust_candidate` when nothing passes.
- Between eligible candidates the code recommends the best benefit-to-cost ratio.

Avoided forcing of one candidate:

```
avoided_ef_j = p_persist * effectiveness * MEDIAN_PERSISTENT_EF_J_PER_M
               * segment_length_km * 1000.0
```

`MEDIAN_PERSISTENT_EF_J_PER_M = 1.74e7` J/m, the median energy forcing per flight metre of persistent-contrail waypoints in the Teoh et al. (2024) CoCiP inventory for 2019. Escape effectiveness is `0.85` for a step climb and `0.80` for a descent; both are engineering assumptions, not measured removal rates.

The decision layer reads the real segment length from the corridor runner (`segment_length_km` from the haversine distance between neighbouring waypoints). `DEFAULT_SEGMENT_LENGTH_KM = 48.0` applies only when no length is supplied, matching the nominal 25-waypoint spacing.

## Fuel and CO2 penalty

`estimate_fuel_penalty(aircraft_type, baseline_fl, target_fl, duration_min=20.0)` prices the extra fuel burned in a 20-minute window:

```
window_fuel_kg = cruise_burn_kg_s * duration_min * 60
penalty_pct    = base_pct * (|delta_fl| * 100 / 2000)
delta_fuel_kg  = window_fuel_kg * penalty_pct / 100
delta_co2_kg   = delta_fuel_kg * 3.16
```

- `base_pct` is `2.0` per 2,000 ft for a climb and `1.15` per 2,000 ft for a descent, the midpoints of the 1.5-2.5% and 0.8-1.5% ranges recorded in the module comments. These are engineering defaults, not operator data.
- Cruise burn defaults [kg/s]: A320 0.65, A20N 0.58, B737 0.65, B738 0.65, B38M 0.60, B77W 2.44, DEFAULT 0.65.
- `CO2_PER_KG_FUEL = 3.16` kg CO2 per kg fuel.
- A descent isn't free: it burns extra fuel relative to level cruise in this model.

## Climate accounting

`calculate_net_climate_benefit(avoided_contrail_ef_j, delta_co2_kg, metric_horizon_years=20)` converts both sides to Earth-mean integrated impacts [J/m^2]:

```
avoided_flux = avoided_contrail_ef_j / EARTH_SURFACE_AREA_M2
co2_flux     = delta_co2_kg * AGWP_co2(horizon) * SECONDS_PER_YEAR
net_benefit_positive = avoided_flux > co2_flux
benefit_to_cost_ratio = avoided_flux / co2_flux
```

Constants: `AGWP_CO2_20YR_W_YR_PER_M2_PER_KG = 2.39e-14`, `AGWP_CO2_100YR_W_YR_PER_M2_PER_KG = 8.80e-14` (Teoh et al. 2024, ACP 24, 6071-6093; the same values appear in IPCC AR6), `SECONDS_PER_YEAR = 3.15576e7`, `EARTH_SURFACE_AREA_M2 = 5.10072e14`. Intermediate horizons add linearly between 20 and 100 years. That interpolation is a documented approximation of the impulse-response curve; horizons outside `[20, 100]` raise `ValueError`.

## Worked example

Source: `reports/vidp_vabb_2025-01-01.json`, produced by:

```bash
stratum corridor \
  --era5 data/era5_del_bom_20250101.nc \
  --route VIDP-VABB --flight-level 350 \
  --departure 2025-01-01T20:00:00 \
  --model models/era5-physics-2025-01-01.joblib \
  --out reports/vidp_vabb_2025-01-01.html
```

Flight state: 25 waypoints, 0 out-of-domain segments, 11 SAC-forming segments, 6 segments with calibrated `P(persist) >= 0.50`, 6 flagged segments with `P(persist) >= 0.40`. The 6 flagged segments form one contiguous reroute event.

Event record: `segment_index` 3, time 2025-01-01T20:18:00+00:00, `p_persist` 1.0, `segments_in_event` 6, recommendation `descent` from FL350 to FL330, `segment_length_km` 47.42603553240151 (47426.0355 m).

Avoided forcing:

```
1.0 * 0.80 * 1.74e7 J/m * 47426.0355 m = 660170414611.029 J (660.17 GJ)
```

Cost, A320 at 0.65 kg/s over a 20-minute window:

```
window fuel = 0.65 * 20 * 60 = 780 kg
penalty     = 1.15% per 2000 ft * 1 step * 780 kg = 8.97 kg fuel
CO2         = 8.97 * 3.16 = 28.3452 kg
```

Benefit-to-cost comparison, 20-year horizon:

```
avoided flux = 660170414611.029 / 5.10072e14 = 1.2943e-3 J/m^2
CO2 flux     = 28.3452 * 2.39e-14 * 3.15576e7 = 2.1378e-5 J/m^2
ratio        = 60.540106412958814
```

Reported totals: `total_extra_fuel_kg` 8.97, `total_extra_co2_kg` 28.3452, `total_avoidable_warming_j` 660170414611.029. Grouping the six flagged segments into one event prevents counting the same contrail six times.

## Vertical profile decision

`stratum profile` repeats the risk view per waypoint across `--levels`. For each waypoint and level it computes `combined_risk`: the calibrated `p_persist` when the SAC holds at that level, and `0.0` otherwise, so a level where no contrail can form carries no risk and a level without weather data is never scored. Every level record carries a `risk_band`: low below 0.10, moderate from 0.10 to 0.40, and high at or above 0.40 (`ABSTAIN_THRESHOLD`). The recommendation is the level with the lowest `combined_risk` among levels that carry ERA5 data and lie inside the model's training envelope (FL330..FL370 for the shipped artifact); levels outside it are flagged `training_envelope: false` and are excluded from the recommendation. When the artifact records no envelope, or no such level exists, the choice falls back to all levels with data and `envelope_fallback` is set. Ties break toward the baseline (default FL350), then toward the lower level. The fuel delta uses `estimate_fuel_penalty` against the baseline over a 20-minute window.

```bash
stratum profile \
  --era5 data/era5_del_bom_20250101.nc \
  --model models/era5-physics-2025-01-01.joblib \
  --route VIDP-VABB --levels 290,310,330,350,370,390
```

For that run: 25 waypoints, recommended FL330 at 6 waypoints and FL350 at 19 waypoints, `+53.82 kg` fuel and `+170.07 kg` CO2 against the FL350 baseline over a 20-minute window per waypoint. FL290 and FL310 sit outside the training envelope at every waypoint and are never recommended. The six FL330 waypoints are the waypoints where FL350 carried a high band, matching the six flagged segments of the corridor run.

## Engineering defaults and their limits

- Fuel penalties come from engineering ranges, not flight data. Replace them with operator performance data before any operational use.
- Efficiencies `0.85` and `0.80` assume the contrail is left behind when the level changes. Real escape depends on the vertical extent of the supersaturated layer.
- The median energy forcing is a fleet-median proxy: it applies one number to every persistent contrail.
- The CO2 side uses AGWP endpoints with linear interpolation; the contrail side uses a median EF rather than per-flight radiative transfer.
- The 0.40 abstain threshold and the 0.50 reporting threshold are policy choices in code, not calibrated operational limits.
