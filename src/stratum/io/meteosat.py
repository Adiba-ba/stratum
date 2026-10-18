"""Meteosat SEVIRI ingestion helpers for Project STRATUM.

Defines the SEVIRI infrared channels used by the contrail detector
(``IR_108``, ``IR_120``, ``IR_087``) and a domain cropping helper that
selects pixels inside a geographic bounding box while preserving the
native coordinate ordering (ascending or descending).
"""

from __future__ import annotations

import numpy as np

__all__ = ["SEVIRI_CHANNELS", "crop_to_domain"]

#: SEVIRI channels ingested for the contrail detector.
SEVIRI_CHANNELS: tuple[str, ...] = ("IR_108", "IR_120", "IR_087")


def crop_to_domain(
    image_array: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a geostationary image to a geographic bounding box.

    Parameters
    ----------
    image_array : numpy.ndarray
        Two-dimensional image of shape ``(len(lats), len(lons))``.
    lats : numpy.ndarray
        One-dimensional latitude coordinates [deg N] for the rows,
        ascending or descending.
    lons : numpy.ndarray
        One-dimensional longitude coordinates [deg E] for the columns,
        ascending or descending.
    bbox : tuple of float
        Bounding box as ``(north, west, south, east)`` [deg], inclusive.

    Returns
    -------
    tuple of numpy.ndarray
        ``(cropped_image, cropped_lats, cropped_lons)`` with the native
        coordinate ordering preserved.

    Raises
    ------
    ValueError
        If the bounding box is malformed, the image shape does not match
        the coordinate lengths, or the box contains no pixel centers.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (north, west, south, east), got {bbox!r}")
    north, west, south, east = (float(value) for value in bbox)
    if north <= south:
        raise ValueError(f"bbox north ({north}) must exceed south ({south})")
    if east <= west:
        raise ValueError(f"bbox east ({east}) must exceed west ({west})")

    latitude_values = np.asarray(lats, dtype=float)
    longitude_values = np.asarray(lons, dtype=float)
    image = np.asarray(image_array)
    expected_shape = (latitude_values.size, longitude_values.size)
    if image.ndim != 2 or image.shape != expected_shape:
        raise ValueError(
            f"image shape {image.shape!r} does not match coordinate lengths "
            f"{expected_shape!r}"
        )

    latitude_mask = (latitude_values >= south) & (latitude_values <= north)
    longitude_mask = (longitude_values >= west) & (longitude_values <= east)
    if not latitude_mask.any() or not longitude_mask.any():
        raise ValueError(f"bounding box {bbox!r} contains no pixel centers")

    cropped = image[np.ix_(latitude_mask, longitude_mask)]
    return cropped, latitude_values[latitude_mask], longitude_values[longitude_mask]
