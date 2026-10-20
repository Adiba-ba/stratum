# Data

STRATUM consumes four real data sources. Nothing in the repository is synthetic. This document records provenance, bundled coverage, validation, and what happens when an input is missing.

## ERA5 reanalysis

Fields used (`ERA5_VARIABLES` in `src/stratum/io/era5.py`): `temperature`, `specific_humidity`, `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity`, `fraction_of_cloud_cover`. `load_era5_dataset` accepts CDS short names (`t`, `q`, `u`, `v`, `w`, `cc`) and resolves coordinate aliases (`lat`/`latitude`, `lon`/`longitude`, `level`/`pressure_level`, `time`/`valid_time`).

Ingestion domain (`ERA5_BBOX`, north/west/south/east): `(40.0, 60.0, 5.0, 95.0)`, the Indian subcontinent with the Arabian Sea and the Bay of Bengal. Pressure levels: 200, 225, 250, 300, 350, 400 hPa. `is_in_domain` uses inclusive bounds with tolerances `1e-6` deg and `1e-3` Pa; non-finite inputs are outside.

Provenance: ERA5 is produced by ECMWF and distributed through the Copernicus Climate Data Store. `stratum fetch era5` submits the request; the pressure levels are imported from `ERA5_PRESSURE_LEVELS_HPA` in `src/stratum/io/era5.py` so the request stays in sync with the loader, while the variables are the eight-variable fetch set (`fraction_of_cloud_cover`, `relative_humidity`, `specific_cloud_ice_water_content`, `specific_humidity`, `temperature`, `u_component_of_wind`, `v_component_of_wind`, `vertical_velocity`). The request domain is north 40 / west 60 / south 5 / east 95 [deg], format NetCDF, all 24 hours, `--days` days from `--date`.

```bash
stratum fetch era5 --date 2025-01-01 --days 1 --out data/cache/era5/era5_2025-01-01.nc
```

Authentication: `cdsapi.Client()` reads `~/.cdsapirc`. When `CDSAPI_URL` and `CDSAPI_KEY` are set they are passed to `cdsapi.Client(url=..., key=...)`. Missing credentials exit 2 with a pointer to `~/.cdsapirc`. The command probes the host before queueing, because the client otherwise retries a failing host for hours; an unreachable CDS host exits 2 with the probe error.

The shipped seven-day extract lives in `data/cache/era5/`: one file per day (about 34 MB) plus the merged `era5_india_2025-01-01_2025-01-07.nc` (168 hourly steps, 141 x 141 points at 0.25 degree, 5-40 N and 60-95 E). The cache is gitignored because of the size; the repository ships the 6.6 MB single-day corridor extract instead.

Bundled file `data/era5_del_bom_20250101.nc` (6.6 MB): 15-32 N, 68-82 E on a 0.25 degree grid (69 x 57 points), pressure levels 200-400 hPa, 24 hourly steps from 2025-01-01T00:00 to 23:00 UTC, for the Delhi to Mumbai corridor day.

Interpolation: `interpolate_met_to_segment` performs multi-dimensional linear interpolation over latitude, longitude, pressure, and time. Query points outside the dataset return `nan` values.

## ADS-B trajectories

Canonical schema (`ADSB_COLUMNS`): `icao24`, `time`, `lat`, `lon`, `baro_alt_ft`, nullable `geo_alt_ft`, `gs_kt`, `track_deg`, `vrate_fpm`, `callsign`, `on_ground`, `source`. `load_adsb_csv` maps recognized OpenSky and ADS-B Exchange header names through `OPENSKY_COLUMN_MAP`; `source` defaults to `"opensky"`.

Cleaning (`clean_trajectory_points`): airborne rows only, barometric altitude at or above FL250 (`25000 ft`), latitude in `[6.0, 37.0]` deg and longitude in `[68.0, 98.0]` deg, `|vrate_fpm| <= 6000`, ground speed jumps `<= 100 kt`, then de-duplication. `select_trajectory` picks one flight by callsign or `icao24`. The CLI falls back to great-circle route profiles when no ADS-B file is supplied, and records `trajectory_source = "great_circle_profile"` in the report.

Provenance: OpenSky Network publishes a live state vector API and historical flight exports. `stratum fetch adsb-live` snapshots `/api/states/all`; anonymous when no credentials are set (it prints a note that authenticated history needs the account), or authenticated with `--time EPOCH` when `OPENSKY_CLIENT_ID` and `OPENSKY_CLIENT_SECRET` are set. `stratum fetch adsb-history` pulls flight tracks for an interval and writes the same canonical CSV headers as `load_adsb_csv`.

```bash
stratum fetch adsb-live --out data/cache/opensky_snapshot.csv --bbox 32,68,15,82

stratum fetch adsb-history \
  --begin 2025-01-01T00:00:00Z --end 2025-01-01T01:00:00Z \
  --icao24 4ca2d1 --max-flights 25 \
  --out data/cache/adsb_history.csv
```

OpenSky authenticated endpoints: token POST to `https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token` with `grant_type=client_credentials`; tokens last ~1800 s. Interval limits: 2 hours for `/flights/all` (no `--icao24`), 2 days for `/flights/aircraft` (with `--icao24`); authenticated `/states/all` `--time` is at most 1 hour back. A 429 reports `X-Rate-Limit-Retry-After-Seconds`; retry then. OpenSky applies per-account credits/rate limits, so `--max-flights` caps the number of track requests. Missing `OPENSKY_CLIENT_ID`/`OPENSKY_CLIENT_SECRET` exits 2.

Two service limits shape what this command can deliver. Flight tracks exist for the last 30 days only; an older interval is refused before the request goes out, and the older ADS-B record has to come from the Trino/MinIO historical interface. The track endpoint reports neither ground speed nor vertical rate, so both are derived from consecutive fixes: ground speed from the great-circle distance over the time step, vertical rate from the barometric altitude change. The first fix borrows the difference to the second fix.

Credentials come from the environment. The fetch commands read `OPENSKY_CLIENT_ID`, `OPENSKY_CLIENT_SECRET`, `EUMETSAT_CONSUMER_KEY`, `EUMETSAT_CONSUMER_SECRET`, `EUMETSAT_API_TOKEN` and `CDSAPI_URL`/`CDSAPI_KEY` from the process environment, and fall back to a `.env` file in the working directory. Variables already exported win over the file, empty values in the file are ignored, and `.env` is gitignored. `.env.example` lists the names.

## Satellite scenes

Split-window detection needs brightness temperatures near 10.8 and 12.0 micrometres.

GOES-16: `stratum fetch goes` downloads real ABI L2 CMIPF granules from the public `gcp-public-data-goes-16` bucket without authentication. Defaults: `--band 14,15`, `--out data/cache/goes16`. Cached in the repository: `OR_ABI-L2-CMIPF-M6C14_G16_s20250012000206_e20250012009514_c20250012009597.nc` and the matching C15 granule.

```bash
stratum fetch goes --date 2025-01-01 --hour 20 --band 14,15 --out data/cache/goes16

stratum detect-scene \
  --satellite data/cache/goes16/OR_ABI-L2-CMIPF-M6C14_G16_s20250012000206_e20250012009514_c20250012009597.nc \
  --satellite-b data/cache/goes16/OR_ABI-L2-CMIPF-M6C15_G16_s20250012000206_e20250012009520_c20250012009577.nc \
  --out-dir reports/scene_goes16_20250101_2000
```

The shipped run wrote `reports/scene_goes16_20250101_2000/summary.json`: 68 detections inside the analysis box (`num_detected`) out of 94 raw detections before the box filter (`num_detected_raw`), `threshold = 2.0`, `min_length_px = 8`, `georeferencing = "goes_imager_projection"`, observation time 2025-01-01T20:00:20.6 UTC. Both granules came from the fetch command above (`--date 2025-01-01 --hour 20 --band 14,15`). The row/column crop of the curved fixed grid is approximate at the box edges, so georeferenced centroids are filtered against the nominal 55N..40N / 70W..30W box a second time; `num_detected_raw` keeps the pre-filter count (example objects from the JSON: object 27 at 52.74 N, 64.43 W, heading 249.5 deg; object 53 at 48.25 N, 38.53 W, heading 249.5 deg). The run also wrote `reports/scene_goes16_20250101_2000/scene.png`, the BTD image with the 68 detection markers. This scene is evidence for the detection and georeferencing path only: no training labels were derived from it.

SEVIRI: `src/stratum/io/meteosat.py` defines the channels (`IR_108`, `IR_120`, `IR_087`) and `crop_to_domain`. `stratum fetch seviri` downloads the EUMETSAT MSG SEVIRI Level 1.5 granule nearest a UTC time; the Indian Ocean collection `EO:EUM:DAT:MSG:HRSEVIRI-IODC` is the default (`--collection` overrides, e.g. `EO:EUM:DAT:MSG:HRSEVIRI` for 0 degree). Authentication uses `EUMETSAT_CONSUMER_KEY` and `EUMETSAT_CONSUMER_SECRET` (OAuth2 client credentials); `EUMETSAT_API_TOKEN` is a fallback Bearer that the Data Store does not accept for downloads. Granules are `application/zip` archives of native HRIT `.nat` files; the command unzips and keeps the member files. Missing credentials, a network failure, or a 403 exits 2 with guidance: a 403 on download means the account has not accepted the EUMETSAT Data Store General Licence, which is a one-time action in the account portal. Note that `stratum detect-scene` reads NetCDF with `IR_108`/`IR_120` arrays, so the native HRIT files need conversion (e.g. satpy) before detection.

```bash
stratum fetch seviri --date 2025-01-01 --hour 20 --channels 108,120 --out data/cache/seviri
```

## Open-Meteo forecast

`stratum check-live` samples the Open-Meteo hourly pressure-level forecast along the built-in RJAA (Narita) to RJBB (Kansai) track. It needs no credentials, covers the default hour 03:00 UTC, and rejects other airport pairs until geometry is added.

## Cache layout and failures

```
data/
  era5_del_bom_20250101.nc              bundled ERA5 day
  labels_physics_2025-01-01.csv         bundled labels (7200 rows; 1852 labeled, 5348 censored)
  cache/goes16/*.nc                     fetched GOES-16 granules
  cache/opensky_snapshot.csv            fetched live ADS-B snapshot
  cache/adsb_history.csv                fetched OpenSky flight tracks
  cache/seviri/*.nat                    fetched MSG SEVIRI HRIT members
  cache/era5/era5_YYYY-MM-DD.nc         fetched ERA5 NetCDF
```

The physics label CSV is a reanalysis surrogate: the SAC and ice-supersaturation rule applied to ERA5 fields, not observed contrails. No label in the repository comes from the GOES-16 scene above.

Missing inputs degrade per row, not per run:

- Outside the ERA5 domain: `unavailable_reason = "out_of_domain"`, all features `nan`, no decision, and the segment is counted in `out_of_domain_segments`.
- Inside the domain but a field interpolates to `nan`: `unavailable_reason = "missing_met"`, same masking.
- No satellite scene for a segment: the observed-label builder marks the scene unobservable and the label stays `nan`, so training skips it.
- Rows without met data never enter an event and never contribute avoided-forcing or fuel numbers.
