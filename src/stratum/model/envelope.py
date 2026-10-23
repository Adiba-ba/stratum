"""Shared helpers for the model training envelope.

The model artifact metadata written by ``stratum train`` records the
``training_envelope`` mapping: the ambient pressure range (and flight
levels) covered by the data-backed training rows. Consumers such as
``stratum profile`` and ``stratum corridor`` read it through
:func:`pressure_envelope_bounds` so a hand-edited or incomplete artifact
degrades to "range unknown" instead of breaking the command.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

__all__ = ["pressure_envelope_bounds"]


def pressure_envelope_bounds(envelope: object) -> tuple[float, float] | None:
    """Return the ``(pressure_pa_min, pressure_pa_max)`` pair of an envelope.

    The envelope is the ``training_envelope`` mapping written by
    ``stratum train``. ``None`` is returned when it is missing, malformed,
    non-finite or inverted, so a hand-edited artifact degrades to "range
    unknown" instead of breaking the caller.
    """
    if not isinstance(envelope, Mapping):
        return None
    try:
        low = float(envelope["pressure_pa_min"])
        high = float(envelope["pressure_pa_max"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        return None
    return low, high
