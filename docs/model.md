# Model

STRATUM trains a monotone LightGBM classifier on atmospheric features, then maps its scores to calibrated probabilities with isotonic regression. The training code lives in `src/stratum/model/` and `src/stratum/cli/train.py`.

## Feature schema

`FEATURE_NAMES` in `src/stratum/model/features.py` is frozen. The dataset columns, the training matrix, and the saved artifact all use this order.

| # | Feature | Meaning |
| --- | --- | --- |
| 1 | `rhi` | Ice relative humidity at the point, from ERA5 specific humidity. |
| 2 | `rhi_above` | RHI at `p - 25 hPa`. |
| 3 | `rhi_below` | RHI at `p + 25 hPa`. |
| 4 | `rhi_grad` | `rhi_above - rhi_below`. |
| 5 | `temp_k` | Ambient temperature [K]. |
| 6 | `tlm_k` | SAC liquid-hat threshold temperature [K]. |
| 7 | `tlc_k` | SAC critical ambient temperature [K]. |
| 8 | `pressure_pa` | Ambient pressure [Pa]. |
| 9 | `cloud_ice` | `1.0` when ERA5 cloud fraction exceeds `0.5`, else `0.0`. |
| 10 | `high_cloud_fraction` | ERA5 total cloud cover fraction. |
| 11 | `vertical_velocity_omega` | ERA5 vertical velocity [Pa/s]. |
| 12 | `vertical_wind_shear` | Wind speed difference across the +/-25 hPa layers divided by `dz`. |
| 13 | `dist_to_tropopause_m` | `altitude_m - 16000.0`. |
| 14 | `lat` | Latitude [deg N]. |
| 15 | `lon` | Longitude [deg E]. |
| 16 | `month` | Month 1-12. |
| 17 | `local_solar_time_hours` | `(hour + minute/60 + lon/15) mod 24`. |
| 18 | `richardson_number` | Gradient Richardson number, clipped to `[-5, 10]`. |
| 19 | `solar_zenith_deg` | Solar zenith angle [deg]; `100.0` is the neutral default. |
| 20 | `engine_efficiency_eta` | Propulsion efficiency from the aircraft registry (A320: `0.32`). |
| 21 | `log_soot_ei_n` | `log10` of the nvPM number emission index (A320: `15.176091259055681`). |

Rows where ERA5 data is unavailable carry `nan` in every feature column. LightGBM handles missing values natively; no imputation happens.

## Monotone constraints

`MONOTONE_CONSTRAINTS` fixes the physical direction of five features; every other feature is unconstrained (`0`).

| Feature | Constraint | Reason |
| --- | --- | --- |
| `rhi` | +1 | Higher ice supersaturation sustains contrails. |
| `tlm_k` | +1 | A warmer liquid-hat threshold raises the critical temperature. |
| `richardson_number` | -1 | Large Ri means weak shear; the model treats turbulent shear as the persistence-favouring state. |
| `engine_efficiency_eta` | +1 | Higher efficiency lowers the mixing-line slope and the critical temperature, so formation becomes easier. |
| `log_soot_ei_n` | +1 | More soot nuclei favor ice formation. |

`monotone_constraints_for` orders the constraints to match the feature list, so a renamed or reordered schema is rejected rather than misapplied.

## Label protocols

Two builders produce the same CSV layout. Physics labels (`src/stratum/labels/physics.py`):

- `1.0` when the SAC holds (`T < T_LC`) and `rhi > 1.0`, so the contrail forms and persists.
- `0.0` when the SAC holds but `rhi <= 1.0`, so it forms and sublimates.
- `NaN` when ERA5 data is unavailable (`censored_reason = "data_unavailable"`), or when the SAC doesn't hold (`censored_reason = "sac_not_formed"`). A non-forming point says nothing about persistence, so the code censors it instead of teaching the model about formation.

Counts on the shipped seven-day dataset `data/labels_physics_2025-01-01_07d.csv`: 75,600 rows over 168 hourly steps from 2025-01-01 to 2025-01-07, six routes at FL330, FL350 and FL370. 2,140 rows are labeled persistent, 5,944 labeled subsaturated, and 67,516 censored, all of them `sac_not_formed`. No row is censored for missing data: the widened ERA5 domain covers every sampled waypoint. The bundled single-day dataset `data/labels_physics_2025-01-01.csv` holds 7,200 rows, with 256 persistent, 1,596 subsaturated and 5,348 censored (4,124 `sac_not_formed`, 1,224 `data_unavailable`).

Observed labels (`src/stratum/labels/observed.py`) return `1.0` on a detection match, `0.0` when the scene was observable but no detection matched, and `NaN` when the scene is unobservable. Both protocols feed the same tri-state contract: `0.0`, `1.0`, or `NaN`.

## Training protocol

`stratum train` reads the CSV, orders rows by the `time` column, and takes the validation block as the last contiguous stretch of time steps that carries at least 50 labeled rows and at least 20 persistent ones. Training uses the rows strictly before that block, so no later row enters training and the split stays free of time leakage. Rows with `NaN` labels are masked from training and from scoring; they are never coerced to `0.0`. `--min-train-rows` (default `50`) guards against training on a nearly empty label set, `--min-val-positive` (default `20`) sets the persistent-row floor of the validation block, and `--val-fraction` (default `0.3`) is recorded in the artifact metadata.

The block rule replaces a blind time-ordered tail. Contrail formation is synoptic: a cold spell leaves whole days without a single persistent row, and the last 106 time steps of this week carry none at all. A tail split over the final 30% of the rows would therefore have produced a validation set with no positive example to score, and the Brier number would have described an empty class. Walking backwards from the last time step until the block holds enough labeled and persistent rows keeps the split scorable. On the shipped dataset the block starts at 2025-01-03T11:00Z and covers 109 of the 168 time steps; the remaining 59 steps are the training rows. When no trailing block qualifies, the command exits 1 and names the two counts it needs.

LightGBM parameters are fixed in `_MODEL_PARAMS`:

```
objective = "binary"
n_estimators = 300
learning_rate = 0.05
num_leaves = 31
min_child_samples = 20
monotone_constraints_method = "intermediate"
random_state = 42
verbosity = -1
```

`fit_calibrator` fits `sklearn.isotonic.IsotonicRegression` on the validation probabilities. Isotonic regression is monotonically non-decreasing, so the base model's monotone constraints survive calibration. The CLI fits the calibrator on the validation split and then scores the calibrated model on that same split, so the calibrated Brier number below is in-sample for the calibration step. `save_model` writes one joblib payload with `format_version = 1`, the model, and metadata: dataset path, SHA-256 and size, `feature_names`, metrics, and the training summary. `load_model` returns `(CalibratedModel, metadata)`.

## Shipped artifact

`models/era5-physics-2025-01-01_07d.metrics.json` records `data/labels_physics_2025-01-01_07d.csv` (SHA-256 `8b557eee4f7b965da482419418a279f77b4d4eceffae312f0838626580c59963`, 25,927,708 bytes) as the training dataset. The model directory is gitignored, so the committed copy of that payload is `reports/model_metrics_2025-01-01_07d.json`.

| Quantity | Value |
| --- | --- |
| Samples | 75600 |
| Training rows | 26550 |
| Labeled training rows | 6881 |
| Validation rows | 49050 |
| Labeled validation rows | 1203 |
| Persistent validation rows | 24 |
| Subsaturated validation rows | 1179 |
| Masked (NaN) rows | 67516 |
| Brier score, raw | 2.4100414095124528e-14 |
| Brier score, calibrated | 0.0 |
| Mean calibrated probability, persistent rows | 1.0 |
| Mean calibrated probability, subsaturated rows | 0.0 |
| Monotone check `rhi` | true |
| Monotone check `tlm_k` | true |
| Time-ordered split | true |
| Training envelope | FL330..FL370, 21663.29..26201.36 Pa |

The class-wise means say what the Brier score leaves implicit: the calibrated probability is saturated at 1.0 on the persistent rows and 0.0 on the subsaturated ones. A Brier near zero on a rare event needs that context, because a model that simply never fires scores 0.0 against a block that holds no positive row at all.

These numbers describe a physics surrogate, not contrail skill. Every label comes from the Schmidt-Appleman rule applied to ERA5, and the feature vector contains that rule's own inputs, so a monotone model can reproduce the decision boundary on held-out time. Nothing in the dataset comes from an observed contrail. The older single-day artifact `models/era5-physics-2025-01-01.metrics.json` is still on disk next to it, trained on `data/labels_physics_2025-01-01.csv`: 7,200 samples, 5,040 training rows (1,273 labeled), 2,160 validation rows (579 labeled), Brier 0.0034539009366374744 raw and 0.0017271157167530224 calibrated.

## Input envelope

`stratum train` records a `training_envelope` in the artifact metadata and in the metrics JSON: the `pressure_pa` range covered by the data-backed training rows, plus the ISA flight levels those pressures round to. A row is data-backed when it carries a finite label or, when the label is censored, a fully finite feature vector. Validation rows and rows without ERA5 data never widen the envelope. The shipped artifact records 21663.29..26201.36 Pa, i.e. FL330, FL350 and FL370.

`stratum profile` classifies every sampled level against this envelope (`training_envelope: true` inside, `false` outside) and recommends only from levels inside it. When the artifact records no envelope, or when no level with data lies inside, the command falls back to all levels with data and sets `envelope_fallback`. The reason is extrapolation: on pressures the model never saw it has no behaviour guarantee, and the previous artifact extrapolated to `p_persist = 1.0` at FL290, a level absent from the training day.

## Confidence semantics

`CalibratedModel.predict_proba(X)[:, 1]` is the calibrated persistence probability `P(persist)` in `[0, 1]`. `predict_raw_proba` exposes the uncalibrated LightGBM score for audits. The code reuses the probability at three thresholds:

| Consumer | Threshold | Effect |
| --- | --- | --- |
| `stratum profile` risk bands | `low < 0.10`, `moderate 0.10 <= risk < 0.40`, `high >= 0.40` | Band label per waypoint and level. |
| Decision layer (`ABSTAIN_THRESHOLD`) | `0.40` | Below it the decision layer abstains; no level change is recommended. |
| Corridor statistics (`PERSISTENT_PROBABILITY_THRESHOLD`) | `0.50` | A segment counts as persistent in the flight report. |

The 0.40 gate is the activation threshold; the 0.50 count is a reporting statistic. Both live in code and are cited in every report.

## Limitations

- Seven ERA5 days over the Indian domain produced the shipped dataset. The model hasn't seen another season, another region, or another year.
- Only 8,084 of 75,600 rows carry labels. The other 67,516 are censored, every one of them because the Schmidt-Appleman criterion doesn't hold there.
- The near-zero Brier score reflects a physics surrogate that reproduces the Schmidt-Appleman rule on held-out time. It isn't skill against observed contrails.
- The calibrated Brier score is in-sample for the isotonic map. A held-out calibration split would give a fairer number.
- The validation block holds 24 persistent rows. The positive side of every class-wise number rests on those rows alone.
- Isotonic calibration is piecewise constant, so probabilities saturate in ranges with few labeled points.
- The monotone checks in the metrics file cover `rhi` and `tlm_k` only; the other three constrained features are enforced by LightGBM but not re-verified per training run.
- Feature defaults (`richardson_number = 1.0`, `solar_zenith_deg = 100.0`) apply when a diagnostic can't be computed; those rows are still trainable, which softens the affected constraint.
