"""Shared geodetic constants for the ``stratum.geo`` subpackage.

The project's spherical-Earth models (parallax displacement, great-circle
distance) all use the same mean-radius convention, kept here as the single
source for new code.
"""

from __future__ import annotations

__all__ = ["EARTH_RADIUS_M"]

#: Mean Earth radius [m] used by the project's spherical-Earth models.
EARTH_RADIUS_M: float = 6_371_000.0
