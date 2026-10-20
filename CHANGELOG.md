# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning.

## [0.2.0] - Unreleased

### Added

- `src/` package layout with `pyproject.toml` and the `stratum` console script.
- Geostationary navigation on the GOES-R fixed grid, both sweep axes, with array
  georeferencing and pixel-angle to compass-heading conversion.
- OpenSky-compatible ADS-B CSV ingestion with unit conversion, cleaning, and
  callsign or `icao24` selection.
- Physics label builder: SAC formation plus ice supersaturation on real ERA5,
  with non-forming points censored.
- Observed label builder: split-window detections matched to advected,
  parallax-corrected ADS-B segments.
- CLI commands `verify-baseline`, `screen-routes`, `corridor`, `profile`,
  `build-physics-labels`, `build-observed-labels`, `train`, `detect-scene`,
  `fetch goes`, `fetch adsb-live`, `fetch adsb-history`, `fetch seviri`,
  `fetch era5`, `check-live`, and `demo`.
- Authenticated fetch commands: OpenSky OAuth2 client credentials for flight
  tracks, EUMETSAT Data Store access for MSG SEVIRI granules, and CDS access
  for ERA5 requests.
- Credentials read from the environment with a gitignored `.env` fallback, plus
  an `.env.example` that lists the variable names.
- Direction-aware HTML corridor report that reads the decision data instead of
  hardcoding a descent narrative.
- Joblib model artifacts with self-describing metadata sidecars and a recorded
  training envelope (pressure range and flight levels).
- ERA5 in-memory load with out-of-domain and missing-met handling.
- Seven-day ERA5 extract: `stratum fetch era5` requests `--days` days in one
  file (default 7), and the cache keeps one file per day plus the merged
  seven-day NetCDF under `data/cache/era5/`.
- Seven-day training set `data/labels_physics_2025-01-01_07d.csv`: 75,600
  rows over 168 hourly steps from 2025-01-01 to 2025-01-07, built from six
  Indian corridors at FL330, FL350 and FL370. 2,140 rows are labeled
  persistent, 5,944 labeled subsaturated, and 67,516 censored, all of them
  `sac_not_formed`. Every row carries ERA5 data, because the widened domain
  removed the out-of-domain samples.
- Class-wise mean probability in the metrics JSON: `mean_probability_positive`
  and `mean_probability_negative` next to the Brier score, so a near-zero Brier
  on a rare event is read next to the two class means.
- `--min-val-positive` on `stratum train` to set the persistent-row floor of
  the validation block.
- Documentation set: README and `docs/architecture.md`, `docs/physics.md`,
  `docs/model.md`, `docs/data.md`, `docs/decisions.md`, `docs/backlog.md`, and
  this changelog.
- Real evidence: ERA5 physics labels, corridor HTML/JSON, vertical profile JSON,
  the seven-day training metrics JSON, and the GOES-16 scene detection with
  preview image.
- Tests expanded from 120 to 271 collected cases.

### Fixed

- `tlm_k` monotone constraint sign flipped to +1. Risk rises with the
  mixing-line temperature.
- CO2 side of the benefit-to-cost ratio moved to J/m^2 with cited AR6 AGWP
  values (`2.39e-14` at 20 yr, `8.80e-14` at 100 yr, linear interpolation).
- Per-segment contrail energy forcing replaced with the cited median persistent
  EF of `1.74e7` J/m, scaled by the real waypoint spacing.
- Cruise burn registry expanded to every engine key with documented defaults and
  a DEFAULT fallback.
- `+-25 hPa` humidity probes made optional: probe-derived features become NaN
  instead of discarding the row.
- Fake neutral Richardson fallback dropped.
- Failed SAC bracket and antipodal great-circle input now raise instead of
  returning silent wrong values.
- `viewing_zenith_angle_deg` uses exact geostationary geometry `(r+h)*sin(s)`;
  the old nadir-angle form capped the angle near 9.7 deg and could never trip
  the 60 deg observability gate.
- `screen-routes` prepends a synthetic-scenario banner so invented
  temperature/RHi knots can't be mistaken for real screening.
- Ground speed and vertical rate are derived from consecutive OpenSky track
  fixes, because the track endpoint reports neither; the loader's kinematic
  gate dropped every such row before.
- Ground speed is written in knots, matching the canonical `gs_kt` column.
- The OpenSky 30-day track window is checked before a request goes out, and the
  sentinel 429 is reported as the window limit rather than a retry time.
- The CDS fetch probes the host before queueing, so an unreachable Climate Data
  Store fails fast instead of retrying for hours.
- ERA5 interpolation selects the two bracketing hours before interpolating the
  space dimensions, so a query costs the same on a one-week file as on a
  one-day one: 4 ms against 1.2 s per call on the seven-day extract, with the
  result unchanged to 1e-14 and the same NaN pattern outside the time range.

### Changed

- Standalone `scripts/` replaced by the `stratum` CLI.
- Package moved from top-level `stratum/` to `src/stratum/`.
- `build_features.py` rewritten under `src/stratum/pipeline/` for the 21-feature
  schema.
- The ERA5 operating domain widened to 5-40 N and 60-95 E, matching the
  seven-day extract that now covers every Indian airport in the route set.
- The validation split is now the last contiguous block of time steps carrying
  enough labeled and persistent rows, instead of a blind time-ordered tail.
  Contrail formation is synoptic, and the last 106 time steps of the seven-day
  week hold no persistent row at all, so a tail split produced a validation set
  with no positive example to score. The command exits 1 and names the two
  counts it needs when no trailing block qualifies.
- The model artifact of record is now `models/era5-physics-2025-01-01_07d.joblib`,
  trained on `data/labels_physics_2025-01-01_07d.csv`. The corridor and vertical
  profile reports were regenerated with it.

### Removed

- `scripts/`.
- `COMPLETE_CODEBASE.md`.
- `stratum_project.zip`.
- Legacy `stratum/diagnostics.py` shim.

## [0.1.0] - 2026-10-18

### Added

- Initial prototype committed as `3b9efe4`: Schmidt-Appleman criterion with
  Murphy-Koop (2005) saturation curves, ISA pressure, Richardson number, and
  solar zenith diagnostics under `stratum/`.
- Split-window contrail detection: SEVIRI BTD, robust normalization, oriented
  matched filters with hysteresis and PCA line gating.
- Matching: tangent-plane advection, Meteosat-9 parallax, observability gate,
  and tri-state (1.0 / 0.0 / NaN) segment labels.
- I/O: ADS-B cleaning schema, ERA5 loader and 4-D interpolation, SEVIRI channel
  constants and cropping.
- Model: frozen 21-feature schema, monotone LightGBM classifier, isotonic
  calibration wrapper, Brier scoring.
- Decision layer: fuel/CO2 penalty model, global climate benefit accounting, and
  altitude counterfactual evaluation with abstain threshold.
- Pipeline and routes: trajectory feature builder, corridor runner, great-circle
  route profiles, and SAC screening.
- Synthetic-fixture demo scripts and a physics-only decision path.
- Bundled ERA5 sample, generated `stratum_model_progress.html` report, and
  `COMPLETE_CODEBASE.md` snapshot.
