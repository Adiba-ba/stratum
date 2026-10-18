"""Geostationary geometry subpackage for Project STRATUM.

Re-exports the fixed-grid navigation (GOES ABI and Meteosat SEVIRI
conventions), the image-space heading helpers and the shared Earth radius.
"""

from stratum.geo.angles import (
    ArrayGeoreferencer,
    initial_bearing_deg,
    pixel_angle_to_heading_deg,
)
from stratum.geo.constants import EARTH_RADIUS_M
from stratum.geo.navigation import GeostationaryNavigation, from_goes_attrs

__all__ = [
    "ArrayGeoreferencer",
    "EARTH_RADIUS_M",
    "GeostationaryNavigation",
    "from_goes_attrs",
    "initial_bearing_deg",
    "pixel_angle_to_heading_deg",
]
