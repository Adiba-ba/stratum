"""Brightness-temperature-difference (BTD) computation and normalization.

The classical satellite contrail detector uses the split-window
brightness temperature difference BTD = BT10.8 - BT12.0, which is
slightly positive for thin ice clouds (contrails, cirrus) and near zero
for clear sky. A local median background is removed and the residual is
scaled by a local robust standard deviation so that matched-filter
thresholds are expressed in units of local noise.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["compute_btd", "normalize_btd"]

#: Gaussian-consistency factor converting MAD to an equivalent sigma.
MAD_TO_SIGMA: float = 1.4826
#: Floor for the local robust sigma, preventing division by zero.
MIN_ROBUST_SIGMA: float = 1e-3


def _validate_2d(array: np.ndarray, name: str) -> np.ndarray:
    """Return ``array`` as a float ndarray, enforcing 2D shape."""
    result = np.asarray(array, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={result.ndim}")
    return result


def compute_btd(bt108: np.ndarray, bt120: np.ndarray) -> np.ndarray:
    """Compute the split-window brightness temperature difference.

    Parameters
    ----------
    bt108 : np.ndarray
        Brightness temperature at 10.8 um [K].
    bt120 : np.ndarray
        Brightness temperature at 12.0 um [K], same shape as ``bt108``.

    Returns
    -------
    np.ndarray
        ``BTD = bt108 - bt120`` [K].
    """
    bt108 = _validate_2d(bt108, "bt108")
    bt120 = _validate_2d(bt120, "bt120")
    if bt108.shape != bt120.shape:
        raise ValueError(
            f"bt108 and bt120 must have the same shape, got {bt108.shape} and {bt120.shape}"
        )
    return bt108 - bt120


def normalize_btd(btd: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Normalize BTD by a local median background and local robust sigma.

    The background is a ``kernel_size x kernel_size`` median filter; the
    residual (``btd - background``) is divided by ``1.4826 * MAD`` of the
    local residual, so the output is in units of local robust standard
    deviations. The sigma is clamped to ``MIN_ROBUST_SIGMA`` to prevent
    division by zero on constant fields.

    Parameters
    ----------
    btd : np.ndarray
        Brightness temperature difference [K].
    kernel_size : int, optional
        Window length in pixels for both the background and the local
        MAD filters, by default 15.

    Returns
    -------
    np.ndarray
        Standardized BTD field [robust sigma units], same shape as ``btd``.
    """
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    btd = _validate_2d(btd, "btd")

    background = ndimage.median_filter(btd, size=kernel_size, mode="reflect")
    residual = btd - background
    local_mad = ndimage.median_filter(np.abs(residual), size=kernel_size, mode="reflect")
    sigma = np.maximum(MAD_TO_SIGMA * local_mad, MIN_ROBUST_SIGMA)
    return residual / sigma
