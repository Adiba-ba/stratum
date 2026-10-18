"""Real-data label builders for Project STRATUM (Step 5).

Two label paths share the frozen feature schema:

- :mod:`stratum.labels.physics` derives labels from ERA5 reanalysis alone
  (SAC formation plus ice supersaturation);
- :mod:`stratum.labels.observed` derives labels from satellite contrail
  detections matched against advected, parallax-corrected ADS-B segments.

Neither path synthesizes data: every feature comes from a real ERA5
lookup, and observed labels come from real detector output.
"""

from stratum.labels.observed import (
    EARTH_RADIUS_KM,
    GEOSTATIONARY_ALTITUDE_KM,
    SatelliteScene,
    build_observed_labels,
    load_bt_pair,
    viewing_zenith_angle_deg,
)
from stratum.labels.physics import (
    CENSORED_DATA_UNAVAILABLE,
    CENSORED_SAC_NOT_FORMED,
    LABEL_FORMING_PERSISTENT,
    LABEL_FORMING_SUBSATURATED,
    PROVENANCE_COLUMNS,
    RHI_PERSISTENCE_THRESHOLD,
    PhysicsSample,
    build_physics_labels,
    read_dataset,
    sample_route_profiles,
    write_dataset,
)

__all__ = [
    "PhysicsSample",
    "PROVENANCE_COLUMNS",
    "RHI_PERSISTENCE_THRESHOLD",
    "LABEL_FORMING_PERSISTENT",
    "LABEL_FORMING_SUBSATURATED",
    "CENSORED_SAC_NOT_FORMED",
    "CENSORED_DATA_UNAVAILABLE",
    "build_physics_labels",
    "sample_route_profiles",
    "write_dataset",
    "read_dataset",
    "SatelliteScene",
    "EARTH_RADIUS_KM",
    "GEOSTATIONARY_ALTITUDE_KM",
    "load_bt_pair",
    "viewing_zenith_angle_deg",
    "build_observed_labels",
]
