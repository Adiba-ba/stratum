# Backlog

Ordered by urgency. Each item states what blocks it and what done means.

## 1. Observed-label end-to-end

Fetch a SEVIRI scene and OpenSky tracks over India aligned on 2025-01-01
20:00 UTC, run `stratum build-observed-labels`, then retrain on the observed
labels. Credentials for OpenSky OAuth2 and the EUMETSAT Data Store now sit in
`.env`, so the download step is unblocked. Done when an observed-label CSV and a
retrained artifact exist, with the matching rate recorded.

## 2. ERA5 expansion

Pull 0.25 deg hourly fields for 2025-01-01..2025-01-07: `fraction_of_cloud_cover`,
`relative_humidity`, `specific_cloud_ice_water_content`, `specific_humidity`,
`temperature`, `u_component_of_wind`, `v_component_of_wind`,
`vertical_velocity`, on levels 200/225/250/300/350/400 hPa over 40N/60E/5N/95E,
roughly 700 MB raw NetCDF. A detached retry loop already polls CDS because
`cds.climate.copernicus.eu` is unreachable from this network. Done when the
multi-day file lands and the label builder covers the wider domain.

## 3. OpenContrails adapter

Wire the OpenContrails tfrecord shards. Each shard is 8-9 GB, so the adapter
needs a subset strategy (region, time slice, or sharded streaming) before it can
read anything. Done when the adapter loads one subset and emits rows in the frozen
21-feature schema.

## 4. Trino/MinIO historical ADS-B

Pull historical ADS-B older than 30 days from the Trino/MinIO store, since the
live OpenSky snapshot cannot be matched to historical ERA5. Done when a historical
CSV reaches the canonical schema and a corridor run uses a real trajectory.

## 5. CI workflow and 0.2.0 patch release

Add a CI workflow running `pytest` (243 collected cases) plus a CLI smoke test,
then cut the 0.2.0 patch release. Done when CI passes on a clean checkout and the
tag exists.

## 6. Foundation-model mix

Train a foundation model on multi-year ERA5 physics labels combined with observed
labels, so the surrogate and the satellite evidence are pooled. Done when the mix
trains, calibrates, and reports a Brier score against a held-out split.
