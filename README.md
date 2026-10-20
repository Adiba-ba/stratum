# STRATUM

STRATUM predicts where contrails form and persist along a flight corridor, and turns that prediction into an altitude recommendation for the dispatcher. It runs as a Python package with a single command-line entry point, `stratum`.

## What this is

The physics baseline is the Schmidt-Appleman criterion (SAC). For each candidate point along a route, the code computes the mixing-line slope `G`, solves the liquid saturation threshold `T_LM` and the critical temperature `T_LC`, and checks the ambient state against them. A contrail forms when the ambient temperature is below `T_LC`; it persists when the ambient air stays ice-supersaturated. Inputs come from real ERA5 reanalysis fields, and the saturation curves use the Murphy & Koop (2005) formulations. The baseline is deterministic and auditable, but it treats every forming contrail as equally harmful.

The calibrated ML surrogate removes that limitation. A monotone LightGBM classifier learns the persistence outcome from 21 atmospheric, diagnostic, and engine features, and isotonic regression maps its scores onto calibrated probabilities. The monotone constraints encode physical direction: ice supersaturation and warmer `T_LM` raise the risk, while stronger shear lowers it. Rows the physics can't resolve, such as points where no contrail forms, are censored instead of labeled, so the model learns persistence only.

The altitude decision layer answers the operational question. For each waypoint the calibrated persistence probability is combined with the SAC result into a per-altitude risk, and for each flagged segment the counterfactual levels at plus and minus 2,000 ft are priced: extra fuel and CO2 against the avoided contrail energy forcing. A recommendation is issued only when the benefit-to-cost comparison passes and the calibrated probability clears 0.40; otherwise the decision layer abstains.

## How it fits together

```
ERA5 reanalysis      ADS-B trajectories     satellite BTD pairs
(NetCDF, 6 vars)     (canonical schema)     (GOES-16 ABI / SEVIRI)
      |                     |                       |
      v                     v                       v
 stratum.io.era5     stratum.io.adsb        stratum.detect.filters
      |                     |                (BTD + oriented kernels)
      +----------+----------+                       |
                 v                                  v
    stratum.pipeline.build_features       stratum.labels.observed
    (6 ERA5 vars -> RHI, SAC, Ri, ...)    (observability + matching)
                 |                                  |
                 v                                  v
          21-feature rows ----> labels CSV <--- tri-state labels
                 |
                 v
  stratum.model.train: monotone LightGBM (rhi +, tlm_k +, Ri -)
                 |
                 v
  stratum.model.calibrate: isotonic map -> calibrated P(persist)
                 |
                 v
  per-waypoint risk profile over FL290..FL390
                 |
                 v
  stratum.decide.counterfactual: fuel/CO2 cost vs avoided energy
                 |
                 v
  altitude recommendation, reroute event, HTML/JSON report
```

## Install

```bash
pip install -e ".[dev]"
```

Extras: `data` adds `cdsapi` and `eumdac`, `figures` adds `matplotlib`, `geo` adds `pyproj`. Core dependencies are numpy, pandas, scipy, xarray, netCDF4, scikit-learn, lightgbm, and joblib.

## Commands

| Command | Purpose |
| --- | --- |
| `stratum verify-baseline` | Run the physics baseline audit table (11 checks) and print PASS or FAIL per check. |
| `stratum screen-routes` | Screen built-in Indian airport pairs for SAC forming share at a given flight level. |
| `stratum build-physics-labels` | Sample ERA5 along route profiles and write a labeled training CSV. |
| `stratum build-observed-labels` | Match satellite detections to flight segments and write observed labels. |
| `stratum train` | Fit the monotone LightGBM model with isotonic calibration and write a joblib artifact. |
| `stratum corridor` | Evaluate one corridor: per-segment probabilities, reroute events, HTML/JSON report. |
| `stratum profile` | Vertical risk profile per waypoint and flight level with a recommended level. |
| `stratum detect-scene` | Detect contrail lines in a split-window satellite scene and georeference them. |
| `stratum fetch goes` | Download GOES-16 ABI L2 CMI granules from the public bucket. |
| `stratum fetch adsb-live` | Snapshot OpenSky `/api/states/all` to CSV, anonymous or authenticated. |
| `stratum fetch adsb-history` | Pull OpenSky flight tracks over an interval into the canonical ADS-B CSV. |
| `stratum fetch seviri` | Download a MSG SEVIRI Level 1.5 granule from the EUMETSAT Data Store. |
| `stratum fetch era5` | Request ERA5 pressure-level fields from the Climate Data Store. |
| `stratum check-live` | Sample the Open-Meteo pressure-level forecast along the built-in RJAA to RJBB track. |
| `stratum demo` | Run one synthetic fixture demo: `detector`, `matching`, `decision`, `model-fixture`. |

Every command accepts `--help` for its full flag list.

The three authenticated fetch commands read `OPENSKY_CLIENT_ID`, `OPENSKY_CLIENT_SECRET`, `EUMETSAT_CONSUMER_KEY`, `EUMETSAT_CONSUMER_SECRET`, `EUMETSAT_API_TOKEN`, `CDSAPI_URL` and `CDSAPI_KEY` from the environment, and fall back to a `.env` file in the working directory. `.env.example` lists the names and `.env` stays out of git.

## Quickstart

Run the real pipeline on the bundled ERA5 day. The label build loops over 7,200 samples; allow a few minutes.

```bash
# 1. Label ERA5 profiles with the SAC + supersaturation rule (7,200 rows)
stratum build-physics-labels \
  --era5 data/era5_del_bom_20250101.nc \
  --routes VIDP-VABB,VIDP-VOBL,VIDP-VECC,VABB-VAAH \
  --levels 330,350,370 \
  --out data/labels_physics_2025-01-01.csv

# 2. Train and calibrate the correction model
stratum train \
  --dataset data/labels_physics_2025-01-01.csv \
  --out models/era5-physics-2025-01-01.joblib

# 3. Evaluate the Delhi to Mumbai corridor and write an HTML report
stratum corridor \
  --era5 data/era5_del_bom_20250101.nc \
  --route VIDP-VABB --flight-level 350 \
  --departure 2025-01-01T20:00:00 \
  --model models/era5-physics-2025-01-01.joblib \
  --out reports/vidp_vabb_2025-01-01.html
```

Observed result of step 1: 7,200 rows, 256 labeled persistent, 1,596 labeled subsaturated, 5,348 censored (4,124 `sac_not_formed`, 1,224 `data_unavailable`); 5,976 rows carry ERA5 data. Result of step 2: validation Brier score 0.00345 raw and 0.00173 calibrated, measured on the 579 labeled rows of the 2,160-row validation split. Result of step 3: 25 segments, 11 SAC-forming, 6 predicted persistent, 6 flagged, one advisory event (descent FL350 to FL330 from segment 4) with +8.97 kg fuel, +28.35 kg CO2, 660.17 GJ avoided contrail energy forcing, benefit-to-cost ratio 60.54.

Per-level dispatch view for the same departure. `stratum profile` samples every requested level at every waypoint, flags each level against the model's training envelope, and recommends the lowest-risk level inside it:

```bash
stratum profile \
  --era5 data/era5_del_bom_20250101.nc \
  --model models/era5-physics-2025-01-01.joblib \
  --route VIDP-VABB --levels 290,310,330,350,370,390 \
  --out reports/vidp_vabb_profile_2025-01-01.json
```

Observed result: 25 waypoints, recommended FL330 at 6 waypoints and FL350 at 19, +53.82 kg fuel and +170.07 kg CO2 against the FL350 baseline, summed over a 20-minute window per waypoint. Risk bands: low `< 0.10`, moderate `0.10 <= risk < 0.40`, high `>= 0.40` (the decision-layer `ABSTAIN_THRESHOLD`). Levels outside the training envelope (FL330..FL370 for the shipped model) are flagged `training_envelope: false` and excluded from the recommendation, with a fallback warning when no level with data lies inside it.

## Data inputs

- ERA5 reanalysis: `temperature`, `specific_humidity`, `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity`, `fraction_of_cloud_cover` on pressure levels. The bundled file `data/era5_del_bom_20250101.nc` covers 15-32 N, 68-82 E, 200-400 hPa, 24 hourly steps on 2025-01-01 (6.6 MB). `stratum fetch era5` pulls a wider extract: 5-40 N, 60-95 E, hourly, one file per day under `data/cache/era5/`, and the operating domain follows that area.
- ADS-B trajectories: canonical OpenSky/ADS-B Exchange schema. The CLI takes trajectories from CSV, or generates great-circle route profiles when no ADS-B file is supplied. `stratum fetch adsb-history` writes that CSV from OpenSky flight tracks; the track endpoint serves the last 30 days only, and ground speed and vertical rate are derived from consecutive fixes because the endpoint reports neither.
- Satellite scenes: GOES-16 ABI L2 CMI bands 14 and 15 in `data/cache/goes16/`, plus SEVIRI IR_108/IR_120/IR_087 support for the Indian domain through EUMETSAT. `stratum fetch seviri` downloads the MSG Level 1.5 granule; the account has to accept the Data Store General Licence once, and the native HRIT files still need conversion to NetCDF for `detect-scene`.
- Open-Meteo forecast: used only by `stratum check-live` for the built-in RJAA to RJBB track.

## Status and limitations

- The bundled ERA5 file is one day over one Indian subdomain. The shipped model is trained on 7,200 rows from that day: 1,852 carry labels (256 persistent, 1,596 subsaturated) and 5,348 are censored (4,124 `sac_not_formed`, 1,224 `data_unavailable`). Metrics on this set measure plumbing, not global skill. The physics labels are a reanalysis surrogate: the SAC and ice-supersaturation rule applied to ERA5 fields, not observed contrails.
- Real-satellite evidence: `stratum detect-scene` on the bundled GOES-16 ABI L2 CMI bands C14/C15 for 2025-01-01 20:00 UTC over the western North Atlantic finds 94 raw line objects, 68 of which georeference inside the 55N..40N / 70W..30W analysis box; the run is `reports/scene_goes16_20250101_2000/summary.json` with a preview in `reports/scene_goes16_20250101_2000/scene.png`. The scene proves the detection and georeferencing path only; no labels were derived from it.
- Fuel penalties are engineering defaults: +2.0% per 2,000 ft climb and +1.15% per 2,000 ft descent on a 20-minute window, with nominal cruise burn per aircraft type. They aren't operator data.
- The decision layer compares an Earth-mean flux: avoided contrail energy forcing divided by Earth surface area against CO2 cost through the 20-year CO2 AGWP. It's an audit-grade cost-benefit proxy, not a certified climate metric.
- The recommendation band is FL270 to FL390 in 2,000 ft steps. Below calibrated P(persist) 0.40 the decision layer abstains.
- `stratum check-live` ships geometry only for RJAA to RJBB; other pairs need a code change.

## References

1. Schumann, U. (2012). A contrail cirrus prediction model. Geoscientific Model Development, 5, 543-580.
2. Teoh, R., Engberg, Z., Shapiro, M., Dray, L., & Stettler, M. E. J. (2024). Global aviation contrail climate effects from 2019 to 2021. Atmospheric Chemistry and Physics, 24, 6071-6093.
3. Kärcher, B. (2018). Formation and radiative forcing of contrail cirrus. Nature Communications, 9, 1824.
4. Lee, D. S., Fahey, D. W., Skowron, A., et al. (2021). The contribution of global aviation to anthropogenic climate forcing for 2000 to 2018. Atmospheric Environment, 244, 117834.
5. Voigt, C., Schumann, U., Minikin, A., et al. (2017). ML-CIRRUS: The airborne experiment on natural cirrus and contrail cirrus with the high-altitude long-range research aircraft HALO. Bulletin of the American Meteorological Society, 98(2), 271-288.
6. Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999-2049.
7. EUMETSAT. MSG/SEVIRI Level 1.5 Image Data Format Description. EUMETSAT, Darmstadt.
8. NOAA/NESDIS. GOES-R Series ABI Level 2 Cloud and Moisture Imagery Product (CMIP). NOAA, College Park.
9. Schäfer, M., Strohmeier, M., Lenders, V., Martinovic, I., & Wilhelm, M. (2014). Bringing up OpenSky: A large-scale ADS-B sensor network for research. Proceedings of the 13th International Symposium on Information Processing in Sensor Networks (IPSN).
10. Murphy, D. M., & Koop, T. (2005). Review of the vapour pressures of ice and supercooled water for atmospheric applications. Quarterly Journal of the Royal Meteorological Society, 131(608), 1539-1565.
11. Teoh, R., Stettler, M. E. J., Majumdar, A., Schumann, U., Sehra, A., & Ko, S. (2020). Mitigating the climate forcing of aircraft contrails by small-scale diversions and technology adoption. Environmental Science & Technology, 54(5), 2941-2950.
