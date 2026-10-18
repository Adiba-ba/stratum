# Architecture

STRATUM is a Python package under `src/stratum/`. One command-line entry point, `stratum`, calls every public behavior; the package modules can also be imported directly.

## Module map

| Module | Role |
| --- | --- |
| `cli/main.py` | Registers the subcommands and dispatches to lazy imports. `stratum <command>` is the only entry point. |
| `cli/verify.py` | `verify-baseline` audit table and `screen-routes` SAC screening. |
| `cli/labels.py` | `build-physics-labels` and `build-observed-labels`. |
| `cli/train.py` | `train`: loads a labeled CSV, splits, fits, calibrates, writes metrics. |
| `cli/corridor.py` | `corridor`: one flight from ERA5 plus trajectory to an HTML/JSON report. |
| `cli/profile.py` | `profile`: per-waypoint risk across flight levels, with `risk_band`. |
| `cli/scene.py` | `detect-scene`: split-window detection and georeferencing. |
| `cli/fetch.py` | `fetch goes` and `fetch adsb-live` downloaders. |
| `cli/live.py` | `check-live`: Open-Meteo forecast along the built-in RJAA to RJBB track. |
| `cli/demos.py` | `demo` fixtures: `detector`, `matching`, `decision`, `model-fixture`. |
| `physics/thermo.py` | Murphy & Koop (2005) saturation curves, their analytic derivative, ISA flight level to pressure. |
| `physics/sac.py` | SAC mixing-line slope, `T_LM` and `T_LC` solves, single-point `evaluate_sac`. |
| `physics/diagnostics.py` | Gradient Richardson number, solar zenith angle, aircraft registry. |
| `io/era5.py` | ERA5 NetCDF validation, domain check, point interpolation. |
| `io/adsb.py` | OpenSky-style CSV to the canonical trajectory schema. |
| `io/adsb_schema.py` | Canonical ADS-B columns, cleaning filters, de-duplication. |
| `io/meteosat.py` | SEVIRI channel names and geographic cropping. |
| `detect/btd.py` | Split-window BTD and local-noise normalization. |
| `detect/filters.py` | Oriented matched-filter line detection with hysteresis. |
| `geo/navigation.py` | Geostationary imager navigation on an oblate ellipsoid. |
| `geo/angles.py` | Detection headings and image coordinate lookups. |
| `routes/corridor.py` | Airport table, great-circle waypoints, route-level SAC reduction. |
| `pipeline/build_features.py` | ERA5 point or trajectory to the 21-feature row; RHI derivation. |
| `pipeline/runner.py` | `evaluate_corridor_flight`: probabilities, event grouping, aggregates. |
| `labels/physics.py` | Physics labels and censoring from ERA5 samples. |
| `labels/observed.py` | Observed labels: satellite detections against ADS-B. |
| `match/advect.py` | Wind advection of segment coordinates. |
| `match/parallax.py` | Meteosat-9 parallax displacement. |
| `match/observe.py` | Scene observability gate. |
| `match/label.py` | Detection-to-segment matching and tri-state labels. |
| `model/features.py` | Frozen 21-name schema and row assembly. |
| `model/train.py` | Monotone LightGBM training and Brier score. |
| `model/calibrate.py` | Isotonic calibration wrapper `CalibratedModel`. |
| `model/artifacts.py` | Joblib save and load with self-describing metadata. |
| `decide/fuel.py` | Fuel and CO2 penalty of a 2,000 ft level change. |
| `decide/climate.py` | CO2 AGWP interpolation and net climate benefit. |
| `decide/counterfactual.py` | Candidate levels, gates, abstain policy. |
| `reporting/html.py` | Standalone monochrome HTML corridor report. |

## Data path 1: physics labels, training, corridor scoring

```text
stratum build-physics-labels
  stratum.cli.labels.build_physics_labels
    stratum.io.era5.load_era5_dataset
    stratum.routes.corridor.great_circle_waypoints
    stratum.labels.physics.sample_route_profiles
    stratum.labels.physics.build_physics_labels
      stratum.pipeline.build_features.extract_features_for_point
        stratum.io.era5.is_in_domain
        stratum.io.era5.interpolate_met_to_segment
        stratum.physics.thermo.fl_to_p, e_w, e_i
        stratum.physics.sac.compute_sac_slope, solve_TLM, critical_T
        stratum.physics.diagnostics.compute_richardson_number
        stratum.physics.diagnostics.compute_solar_zenith_angle
    stratum.labels.physics.write_dataset

stratum train
  stratum.cli.train.train
    stratum.labels.physics.read_dataset
    stratum.model.train.train_contrail_model
    stratum.model.calibrate.fit_calibrator
    stratum.model.artifacts.save_model

stratum corridor
  stratum.cli.corridor.corridor
    stratum.io.era5.load_era5_dataset
    stratum.io.adsb.load_adsb_csv -> stratum.io.adsb_schema.clean_trajectory_points
      -> stratum.io.adsb.select_trajectory       (when a trajectory is given)
    stratum.routes.corridor.great_circle_waypoints (when no trajectory is given)
    stratum.model.artifacts.load_model
    stratum.pipeline.runner.evaluate_corridor_flight
      stratum.pipeline.build_features.extract_features_for_trajectory
      stratum.model.calibrate.CalibratedModel.predict_proba
      stratum.decide.counterfactual.evaluate_altitude_candidates
        stratum.decide.fuel.estimate_fuel_penalty
        stratum.decide.climate.calculate_net_climate_benefit
    stratum.reporting.html.probe_rhi_evidence
    stratum.reporting.html.render_corridor_report
```

`stratum profile` reuses the same pieces on a vertical grid: `great_circle_waypoints`, then `extract_features_for_point` per level, `CalibratedModel.predict_proba`, and `stratum.cli.profile.risk_band` for the per-level band. The recommendation prefers levels inside the `training_envelope` recorded in the model artifact and falls back to all levels with ERA5 data when none qualifies.

## Data path 2: observed labels from satellite scenes

```text
stratum detect-scene
  stratum.cli.scene.detect_scene
    stratum.labels.observed.load_bt_pair
    stratum.detect.btd.compute_btd, normalize_btd
    stratum.detect.filters.build_oriented_line_kernels, detect_contrail_lines
    stratum.geo.navigation.GeostationaryNavigation.from_goes_attrs
    stratum.geo.angles.initial_bearing_deg, pixel_angle_to_heading_deg

stratum build-observed-labels
  stratum.cli.labels.build_observed_labels
    stratum.labels.observed.load_bt_pair
    stratum.detect.filters.detect_contrail_lines
    stratum.labels.observed.viewing_zenith_angle_deg
    stratum.match.observe.is_scene_observable
    stratum.match.parallax.calculate_parallax_displacement
    stratum.match.advect.advect_segment
    stratum.match.label.match_segment_to_detections
    stratum.match.label.assign_tristate_label
```

Both label paths write one row per sample in the same frozen feature order, so one training command consumes either dataset.

## Entry point and tests

The CLI is the only supported entry point. `python -m stratum` and direct module execution aren't documented paths. Inside the package, modules never read `sys.argv`.

Tests live in `tests/`, one file per subsystem: `test_physics`, `test_diagnostics`, `test_io`, `test_adsb`, `test_labels`, `test_features`, `test_model`, `test_pipeline`, `test_decide`, `test_detect`, `test_match`, `test_geo`, `test_corridors`, `test_reporting`, `test_cli`, `test_cli_core`, and `test_cli_profile`. They run with `pytest` from the repository root; fixtures write to temporary directories, so no network access is required.
