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
  `fetch goes`, `fetch adsb-live`, `check-live`, and `demo`.
- Direction-aware HTML corridor report that reads the decision data instead of
  hardcoding a descent narrative.
- Joblib model artifacts with self-describing metadata sidecars and a recorded
  training envelope (pressure range and flight levels).
- ERA5 in-memory load with out-of-domain and missing-met handling.
- Documentation set: README and `docs/architecture.md`, `docs/physics.md`,
  `docs/model.md`, `docs/data.md`, `docs/decisions.md`.
- Real evidence: ERA5 physics labels, corridor HTML/JSON, vertical profile JSON,
  and the GOES-16 scene detection with preview image.
- Tests expanded from 120 to 243 collected cases.

### Fixed

- `tlm_k` monotone constraint sign flipped to +1; risk rises with the
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
  temperature/RHi knots cannot be mistaken for real screening.

### Changed

- Standalone `scripts/` replaced by the `stratum` CLI.
- Package moved from top-level `stratum/` to `src/stratum/`.
- `build_features.py` rewritten under `src/stratum/pipeline/` for the 21-feature
  schema.

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
