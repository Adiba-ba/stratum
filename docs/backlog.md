# Backlog

Ordered by urgency. Each item states what blocks it and what done means.

## 1. Observed-label end-to-end

Build an observed-label set from a SEVIRI scene and OpenSky tracks over India, run
`stratum build-observed-labels`, then retrain on the observed labels. Two
constraints shape the work, both found by probing the live services:

- OpenSky serves flight tracks for the last 30 days only, so the satellite scene
  and the trajectories have to share a date inside that window. ERA5 lags by about
  five days, so the ERA5 preliminary dataset (`reanalysis-era5-pressure-levels-preliminary`)
  is the match for the most recent scene.
- The EUMETSAT download returns HTTP 403 until the account accepts the Data Store
  General Licence. Accepting it in the account portal is a one-time step.

OpenSky coverage over India is thin: sampled tracks topped out between 9,000 and
25,000 ft and carried no cruise-level fixes, so the observable-segment rate over
Indian routes will be low until another ADS-B source is added. Done when an
observed-label CSV and a retrained artifact exist, with the matching rate recorded.

## 2. ERA5 expansion

Done for the first week: 0.25 deg hourly fields for 2025-01-01..2025-01-07 with
`fraction_of_cloud_cover`, `relative_humidity`, `specific_cloud_ice_water_content`,
`specific_humidity`, `temperature`, `u_component_of_wind`, `v_component_of_wind` and
`vertical_velocity` on levels 200/225/250/300/350/400 hPa over 40N/60E/5N/95E,
about 34 MB per day, cached under `data/cache/era5/`. The operating domain now
matches that area.

Still open: widen the training window to several months, which needs a per-day
request loop and roughly 34 MB per day of download. Done when the label set spans
at least one season.

## 3. OpenContrails adapter

Wire the OpenContrails tfrecord shards. Each shard is 8-9 GB, so the adapter
needs a subset strategy (region, time slice, or sharded streaming) before it can
read anything. The dataset also carries its own contrail labels, which makes it
the strongest source for observation-trained labels over the Americas. Done when
the adapter loads one subset and emits rows in the frozen 21-feature schema.

## 4. Trino/MinIO historical ADS-B

Pull historical ADS-B older than 30 days from the Trino/MinIO store, since the
track endpoint refuses anything older. This is the only public OpenSky route to
the trajectories behind the 2025-01-01 satellite scenes. Done when a historical
CSV reaches the canonical schema and a corridor run uses a real trajectory.

## 5. Native SEVIRI reader

The Data Store delivers SEVIRI as zipped HRIT `.nat` members, while
`stratum detect-scene` reads NetCDF with `IR_108` and `IR_120` arrays. A reader
(or a satpy-based conversion step) closes the gap so an Indian scene can be
detected without manual conversion. Done when `detect-scene` accepts a fetched
SEVIRI granule directly.

## 6. CI workflow and 0.2.0 patch release

Add a CI workflow running `pytest` plus a CLI smoke test, then cut the 0.2.0 patch
release. Done when CI passes on a clean checkout and the tag exists.

## 7. Foundation-model mix

Train a foundation model on multi-year ERA5 physics labels combined with observed
labels. So the surrogate and the satellite evidence are pooled. Done when the mix
trains, calibrates, and reports a Brier score against a held-out split.
