"""Reporting subpackage for Project STRATUM.

Renders the standalone monochrome corridor report and the ERA5 probe
evidence used to annotate it.
"""

from stratum.reporting.html import (
    PERSISTENT_THRESHOLD,
    probe_rhi_evidence,
    render_corridor_report,
)

__all__ = [
    "PERSISTENT_THRESHOLD",
    "probe_rhi_evidence",
    "render_corridor_report",
]
