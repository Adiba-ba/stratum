"""Scene observability gate for satellite contrail screening."""

from __future__ import annotations

__all__ = ["is_scene_observable"]

#: Maximum high-cloud fraction allowed for a usable scene.
MAX_HIGH_CLOUD_FRACTION: float = 0.5
#: Maximum viewing zenith angle allowed for a usable scene [deg].
MAX_VIEWING_ZENITH_ANGLE_DEG: float = 60.0


def is_scene_observable(
    high_cloud_fraction: float,
    viewing_zenith_angle_deg: float,
    opaque_cloud: bool = False,
) -> bool:
    """Return True only when a satellite scene is usable for screening.

    A scene is observable if and only if all three conditions hold:

    - ``high_cloud_fraction < 0.5`` (contrails are not masked by high
      cloud in the field of view),
    - ``viewing_zenith_angle_deg < 60.0`` (avoid strong geometric
      distortion and parallax at the swath edge),
    - ``opaque_cloud`` is False (no fully opaque cloud screens the scene).

    Parameters
    ----------
    high_cloud_fraction : float
        Fraction of the scene covered by high cloud [0-1].
    viewing_zenith_angle_deg : float
        Satellite viewing zenith angle [deg].
    opaque_cloud : bool, optional
        True when the scene contains opaque cloud, by default False.

    Returns
    -------
    bool
        True if the scene is observable, False otherwise.
    """
    return (
        high_cloud_fraction < MAX_HIGH_CLOUD_FRACTION
        and viewing_zenith_angle_deg < MAX_VIEWING_ZENITH_ANGLE_DEG
        and not opaque_cloud
    )
