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

Observed labels (`src/stratum/labels/observed.py`) return `1.0` on a detection match, `0.0` when the scene was observable but no detection matched, and `NaN` when the scene is unobservable. Both protocols feed the same tri-state contract: `0.0`, `1.0`, or `NaN`.

## Training protocol

`stratum train` reads the CSV, orders rows by the `time` column, and splits the last `--val-fraction` (default `0.3`) into validation. Rows with `NaN` labels are masked from training and from scoring; they are never coerced to `0.0`. `--min-train-rows` (default `50`) guards against training on a nearly empty label set.

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

`models/era5-physics-2025-01-01.metrics.json` records `data/labels_physics_2025-01-01.csv` (SHA-256 `2f1e20bf4fb8de7e6401d7c1613ce0804dc561cff0704839b81f6b033c92bb49`) as the training dataset.

| Quantity | Value |
| --- | --- |
| Samples | 7200 |
| Training rows | 5040 |
| Labeled training rows | 1273 |
| Validation rows | 2160 |
| Labeled validation rows | 579 |
| Masked (NaN) rows | 5348 |
| Brier score, raw | 0.0034539009366374744 |
| Brier score, calibrated | 0.0017271157167530224 |
| Monotone check `rhi` | true |
| Monotone check `tlm_k` | true |
| Time-ordered split | true |
| Training envelope | FL330..FL370, 21663.29..26201.36 Pa |

Calibration roughly halves the raw Brier score on this split. Both numbers are measured on 579 labeled validation rows from one day, so they show the model separates the two labeled classes on this sample; they aren't evidence of global skill.

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

- One ERA5 day over one Indian subdomain produced the shipped dataset. The model hasn't seen another season, another region, or another year.
- Only 1,852 of 7,200 rows carry labels. The other 5,348 are censored: 1,224 rows with no ERA5 state behind the sample, and 4,124 rows where the SAC doesn't hold.
- The calibrated Brier score is in-sample for the isotonic map. A held-out calibration split would give a fairer number.
- Isotonic calibration is piecewise constant, so probabilities saturate in ranges with few labeled points.
- The monotone checks in the metrics file cover `rhi` and `tlm_k` only; the other three constrained features are enforced by LightGBM but not re-verified per training run.
- Feature defaults (`richardson_number = 1.0`, `solar_zenith_deg = 100.0`) apply when a diagnostic can't be computed; those rows are still trainable, which softens the affected constraint.
