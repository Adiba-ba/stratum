"""Real-data ingestion and I/O subpackage for Project STRATUM.

Re-exports the ADS-B schema and cleaning, ERA5 ingestion and point
interpolation, and the Meteosat SEVIRI cropping helper.
"""

from stratum.io.adsb_schema import (
    ADSB_COLUMNS,
    clean_trajectory_points,
)
from stratum.io.era5 import (
    ERA5_BBOX,
    ERA5_PRESSURE_LEVELS_HPA,
    ERA5_VARIABLES,
    interpolate_met_to_segment,
    load_era5_dataset,
)
from stratum.io.meteosat import SEVIRI_CHANNELS, crop_to_domain

__all__ = [
    "ADSB_COLUMNS",
    "clean_trajectory_points",
    "ERA5_BBOX",
    "ERA5_PRESSURE_LEVELS_HPA",
    "ERA5_VARIABLES",
    "load_era5_dataset",
    "interpolate_met_to_segment",
    "SEVIRI_CHANNELS",
    "crop_to_domain",
]
