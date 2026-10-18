"""Satellite scene detection subcommand for the STRATUM command line.

Loads a real split-window brightness-temperature pair, georeferences the
detections with whichever navigation the NetCDF file actually carries
(GOES fixed-grid scan angles when ``goes_imager_projection`` is present,
otherwise 2-D or 1-D latitude/longitude arrays), and prints a JSON
summary. Nothing is synthesized: when a file lacks the two channels the
command exits with code 2 instead of substituting fake brightness
temperatures.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from stratum.detect.btd import compute_btd
from stratum.detect.filters import detect_contrail_lines
from stratum.geo import (
    ArrayGeoreferencer,
    from_goes_attrs,
    pixel_angle_to_heading_deg,
)
from stratum.io.meteosat import crop_to_domain
from stratum.labels.observed import load_bt_pair

__all__ = ["SceneImage", "detect_scene", "load_scene"]

#: Default channel variable names of the split-window pair.
DEFAULT_CHANNEL_108: str = "IR_108"
DEFAULT_CHANNEL_120: str = "IR_120"

#: Candidate names of the 2-D and 1-D coordinate variables.
_LATITUDE_NAMES: tuple[str, ...] = ("latitude", "lat")
_LONGITUDE_NAMES: tuple[str, ...] = ("longitude", "lon")
_SCAN_X_NAMES: tuple[str, ...] = ("x", "x_image", "x_rad")
_SCAN_Y_NAMES: tuple[str, ...] = ("y", "y_image", "y_rad")
_TIME_NAMES: tuple[str, ...] = ("time", "t", "valid_time")

#: Attribute names holding the scene observation time.
_TIME_ATTRS: tuple[str, ...] = ("time_coverage_start", "date_created")


class _ScanAngleGeoreferencer:
    """Map image indices to lat/lon through GOES fixed-grid scan angles.

    Wraps a :class:`stratum.geo.navigation.GeostationaryNavigation` and the
    file's 1-D ``x``/``y`` scan-angle axes. Fractional image indices are
    linearly interpolated on the axes before the navigation equations are
    applied, matching the detector's sub-pixel centroids.
    """

    def __init__(
        self,
        navigation: Any,
        x_rad: np.ndarray,
        y_rad: np.ndarray,
    ) -> None:
        self.navigation = navigation
        self.subsatellite_lon_deg = float(navigation.longitude_of_projection_deg)
        self._x = np.asarray(x_rad, dtype=float)
        self._y = np.asarray(y_rad, dtype=float)
        self._col_index = np.arange(self._x.size, dtype=float)
        self._row_index = np.arange(self._y.size, dtype=float)

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape ``(n_rows, n_cols)`` of the wrapped scan angles."""
        return (self._y.size, self._x.size)

    def to_latlon(self, row: float, col: float) -> tuple[float, float]:
        """Interpolate lat/lon at fractional image index ``(row, col)``."""
        if not (math.isfinite(row) and math.isfinite(col)):
            return (math.nan, math.nan)
        n_rows, n_cols = self.shape
        if row < 0.0 or row > n_rows - 1 or col < 0.0 or col > n_cols - 1:
            return (math.nan, math.nan)
        x_rad = float(np.interp(col, self._col_index, self._x))
        y_rad = float(np.interp(row, self._row_index, self._y))
        lat, lon = self.navigation.scan_angles_to_latlon(x_rad, y_rad)
        return (float(lat), float(lon))


@dataclass(frozen=True)
class SceneImage:
    """A loaded scene ready for detection and reporting.

    Attributes
    ----------
    bt_108, bt_120 : numpy.ndarray
        Split-window brightness temperatures [K] on the image grid.
    georef : object or None
        Georeferencer exposing ``to_latlon(row, col)``, or ``None`` when
        the file carries no usable navigation.
    georeferencing : str or None
        How the georeferencer was built: ``"goes_imager_projection"``,
        ``"latlon_arrays"`` or ``None``.
    observation_time : pandas.Timestamp or None
        Scene time read from the file, when present.
    cropped : bool
        Whether a ``--bbox`` crop was applied.
    """

    bt_108: np.ndarray
    bt_120: np.ndarray
    georef: Any | None
    georeferencing: str | None
    observation_time: Any | None
    cropped: bool


def _first_1d(dataset: xr.Dataset, names: tuple[str, ...]) -> np.ndarray | None:
    """Return the first 1-D variable in ``names``, else ``None``."""
    for name in names:
        if name in dataset.variables:
            values = np.asarray(dataset[name].values, dtype=float)
            if values.ndim == 1:
                return values
    return None


def _first_2d(dataset: xr.Dataset, names: tuple[str, ...]) -> np.ndarray | None:
    """Return the first 2-D variable in ``names``, else ``None``."""
    for name in names:
        if name in dataset.variables:
            values = np.asarray(dataset[name].values, dtype=float)
            if values.ndim == 2:
                return values
    return None


def _projection_attrs(dataset: xr.Dataset) -> dict[str, Any] | None:
    """Return the geostationary projection attributes of the file.

    Prefers the ``goes_imager_projection`` variable attributes and falls
    back to global attributes that carry the same keys. Returns ``None``
    when the file has no geostationary projection.
    """
    if "goes_imager_projection" in dataset.variables:
        attrs = dict(dataset["goes_imager_projection"].attrs)
        if attrs:
            return attrs
    attrs = dict(dataset.attrs)
    has_height = "perspective_point_height" in attrs or "satellite_height" in attrs
    if has_height and "longitude_of_projection_origin" in attrs:
        return attrs
    return None


def _scene_time(dataset: xr.Dataset) -> Any | None:
    """Observation time from the file's attributes or time coordinate."""
    for name in _TIME_ATTRS:
        value = dataset.attrs.get(name)
        if value is None:
            continue
        try:
            return _to_utc(value)
        except (TypeError, ValueError):
            continue
    for name in _TIME_NAMES:
        if name in dataset.variables:
            values = np.asarray(dataset[name].values).ravel()
            if values.size:
                try:
                    return _to_utc(values[0])
                except (TypeError, ValueError):
                    continue
    return None


def _to_utc(value: Any) -> Any:
    """Return ``value`` as a UTC pandas timestamp."""
    import pandas as pd

    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{value!r} is not a valid timestamp")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _crop(
    scene: SceneImage,
    row_lats: np.ndarray,
    col_lons: np.ndarray,
    bbox: tuple[float, float, float, float],
    x_rad: np.ndarray | None,
    y_rad: np.ndarray | None,
    lat_grid: np.ndarray | None,
    lon_grid: np.ndarray | None,
    navigation: Any | None,
) -> tuple[SceneImage, np.ndarray | None, np.ndarray | None]:
    """Crop the scene rows/columns to a geographic bounding box.

    The image crop is performed by
    :func:`stratum.io.meteosat.crop_to_domain` with the row latitudes and
    column longitudes; the same row/column masks then select the
    coordinate arrays that rebuild the georeferencer. Row and column
    coordinates are the per-row mean latitude and per-column mean
    longitude, so the gate is approximate on a rotated or strongly
    curved grid - it selects a region, it is not navigation.
    """
    north, west, south, east = bbox
    cropped_bt_108, cropped_row_lats, cropped_col_lons = crop_to_domain(
        scene.bt_108, row_lats, col_lons, bbox
    )
    cropped_bt_120, _, _ = crop_to_domain(scene.bt_120, row_lats, col_lons, bbox)
    row_mask = (row_lats >= south) & (row_lats <= north)
    col_mask = (col_lons >= west) & (col_lons <= east)
    del cropped_row_lats, cropped_col_lons

    georef: Any | None = None
    georeferencing: str | None = None
    if scene.georeferencing == "goes_imager_projection" and x_rad is not None:
        if y_rad is None or navigation is None:  # pragma: no cover - guarded upstream
            raise ValueError("scan-angle grid is incomplete; cannot crop")
        georef = _ScanAngleGeoreferencer(
            navigation, x_rad[col_mask], y_rad[row_mask]
        )
        georeferencing = "goes_imager_projection"
    elif lat_grid is not None and lon_grid is not None:
        georef = ArrayGeoreferencer(
            lat_grid[np.ix_(row_mask, col_mask)],
            lon_grid[np.ix_(row_mask, col_mask)],
        )
        georeferencing = "latlon_arrays"

    return (
        SceneImage(
            bt_108=cropped_bt_108,
            bt_120=cropped_bt_120,
            georef=georef,
            georeferencing=georeferencing,
            observation_time=scene.observation_time,
            cropped=True,
        ),
        x_rad,
        y_rad,
    )


def _read_channel(path: str | Path, name: str) -> np.ndarray:
    """Read one 2-D brightness-temperature variable from a NetCDF file."""
    with xr.open_dataset(path) as dataset:
        if name not in dataset.data_vars:
            raise ValueError(
                f"scene {path} is missing channel {name!r}; available "
                f"variables: {sorted(dataset.data_vars)}"
            )
        values = np.asarray(dataset[name].values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"channel {name!r} must be 2-D, got shape {values.shape!r}")
    return values


def load_scene(
    path: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    channel_108: str = DEFAULT_CHANNEL_108,
    channel_120: str = DEFAULT_CHANNEL_120,
    path_120: str | Path | None = None,
) -> SceneImage:
    """Load a satellite scene with its georeferencing.

    Parameters
    ----------
    path : str or pathlib.Path
        NetCDF scene file; see
        :func:`stratum.labels.observed.load_bt_pair` for the channel
        layout. When ``path_120`` is given, this file supplies the
        10.8 um channel and all navigation metadata.
    bbox : tuple of float, optional
        ``(north, west, south, east)`` crop box in degrees. Requires the
        file to carry navigation coordinates.
    channel_108, channel_120 : str, optional
        Variable names of the 10.8 um and 12.0 um channels.
    path_120 : str or pathlib.Path, optional
        Separate NetCDF file holding the 12.0 um channel, for products
        that store one band per file (e.g. GOES-16 ABI L2 CMI granules,
        which use the variable name ``CMI``).

    Returns
    -------
    SceneImage
        Brightness temperatures, georeferencer and scene time.

    Raises
    ------
    ValueError
        When a channel is missing, the crop box cannot be applied, the
        two files disagree on shape, or the scan-angle axes do not match
        the channel shape.
    """
    if path_120 is None:
        bt_108, bt_120 = load_bt_pair(path, channel_108, channel_120)
    else:
        bt_108 = _read_channel(path, channel_108)
        bt_120 = _read_channel(path_120, channel_120)
        if bt_108.shape != bt_120.shape:
            raise ValueError(
                "the two channel files disagree on shape: "
                f"{bt_108.shape!r} vs {bt_120.shape!r}"
            )
    shape = bt_108.shape

    with xr.open_dataset(path) as dataset:
        observation_time = _scene_time(dataset)
        projection = _projection_attrs(dataset)

        navigation = None
        x_rad: np.ndarray | None = None
        y_rad: np.ndarray | None = None
        row_lats: np.ndarray | None = None
        col_lons: np.ndarray | None = None
        lat_grid: np.ndarray | None = None
        lon_grid: np.ndarray | None = None
        georef: Any | None = None
        georeferencing: str | None = None

        if projection is not None:
            x_rad = _first_1d(dataset, _SCAN_X_NAMES)
            y_rad = _first_1d(dataset, _SCAN_Y_NAMES)
            if x_rad is not None and y_rad is not None:
                if (y_rad.size, x_rad.size) != shape:
                    raise ValueError(
                        "scan-angle axes do not match the channel shape: "
                        f"{(y_rad.size, x_rad.size)!r} vs {shape!r}"
                    )
                navigation = from_goes_attrs(projection)
                georef = _ScanAngleGeoreferencer(navigation, x_rad, y_rad)
                georeferencing = "goes_imager_projection"
                x_center = 0.5 * (x_rad.min() + x_rad.max())
                y_center = 0.5 * (y_rad.min() + y_rad.max())
                row_lats = np.asarray(
                    navigation.scan_angles_to_latlon(
                        np.full(y_rad.shape, x_center), y_rad
                    )[0],
                    dtype=float,
                )
                col_lons = np.asarray(
                    navigation.scan_angles_to_latlon(
                        x_rad, np.full(x_rad.shape, y_center)
                    )[1],
                    dtype=float,
                )

        if georef is None:
            lat_grid = _first_2d(dataset, _LATITUDE_NAMES)
            lon_grid = _first_2d(dataset, _LONGITUDE_NAMES)
            if (
                lat_grid is not None
                and lon_grid is not None
                and lat_grid.shape == shape
                and lon_grid.shape == shape
            ):
                georef = ArrayGeoreferencer(lat_grid, lon_grid)
                georeferencing = "latlon_arrays"
                row_lats = lat_grid.mean(axis=1)
                col_lons = lon_grid.mean(axis=0)
            else:
                lat_1d = _first_1d(dataset, _LATITUDE_NAMES)
                lon_1d = _first_1d(dataset, _LONGITUDE_NAMES)
                lat_grid = None
                lon_grid = None
                if (
                    lat_1d is not None
                    and lon_1d is not None
                    and (lat_1d.size, lon_1d.size) == shape
                ):
                    lat_grid, lon_grid = np.meshgrid(
                        lat_1d, lon_1d, indexing="ij"
                    )
                    georef = ArrayGeoreferencer(lat_grid, lon_grid)
                    georeferencing = "latlon_arrays"
                    row_lats = lat_1d
                    col_lons = lon_1d

    scene = SceneImage(
        bt_108=bt_108,
        bt_120=bt_120,
        georef=georef,
        georeferencing=georeferencing,
        observation_time=observation_time,
        cropped=False,
    )

    if bbox is None:
        return scene
    if row_lats is None or col_lons is None:
        raise ValueError(
            "the scene file carries no georeferencing coordinates, so --bbox "
            "cannot be applied"
        )
    cropped_scene, _, _ = _crop(
        scene, row_lats, col_lons, bbox, x_rad, y_rad, lat_grid, lon_grid, navigation
    )
    return cropped_scene


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Parse ``N,W,S,E`` into a ``(north, west, south, east)`` tuple."""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox must be N,W,S,E, got {value!r}")
    north, west, south, east = (float(part) for part in parts)
    if north <= south:
        raise ValueError(f"--bbox north ({north}) must exceed south ({south})")
    if east <= west:
        raise ValueError(f"--bbox east ({east}) must exceed west ({west})")
    return (north, west, south, east)


def _write_scene_png(
    path: Path,
    btd: np.ndarray,
    objects: list[dict[str, Any]],
) -> bool:
    """Write the BTD scene with detection markers; ``False`` without matplotlib.

    Matplotlib is imported lazily so the command works without it.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    figure, axis = plt.subplots(figsize=(6.4, 4.8), dpi=110)
    axis.imshow(btd, cmap="gray", origin="upper")
    if objects:
        rows = [float(obj["centroid_px"][0]) for obj in objects]
        cols = [float(obj["centroid_px"][1]) for obj in objects]
        axis.scatter(
            cols,
            rows,
            s=32,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.2,
            label=f"{len(objects)} detection(s)",
        )
        axis.legend(loc="upper right", fontsize=8)
    axis.set_title("STRATUM detect-scene: split-window BTD [K]")
    axis.set_xlabel("column [px]")
    axis.set_ylabel("row [px]")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return True


def detect_scene(argv: list[str]) -> int:
    """Detect contrail lines in one satellite scene and report them.

    Parameters
    ----------
    argv : list of str
        Arguments after ``detect-scene``: ``--satellite`` (required),
        ``--satellite-b``, ``--var-108``, ``--var-120``, ``--out-dir``,
        ``--threshold``, ``--min-length-px`` and ``--bbox N,W,S,E``.

    Returns
    -------
    int
        0 on success, 2 for a missing file, missing channels, a malformed
        crop box, or invalid detector parameters.
    """
    parser = argparse.ArgumentParser(
        prog="stratum detect-scene",
        description=(
            "Detect contrail-like lines in a real split-window satellite "
            "scene and georeference the detections when the file carries "
            "navigation metadata."
        ),
    )
    parser.add_argument(
        "--satellite",
        required=True,
        type=Path,
        metavar="NC",
        help="NetCDF scene with the IR_108/IR_120 channel pair",
    )
    parser.add_argument(
        "--satellite-b",
        type=Path,
        default=None,
        metavar="NC",
        help=(
            "separate NetCDF file holding the 12.0 um channel, for one-band-"
            "per-file products such as GOES-16 ABI L2 CMI"
        ),
    )
    parser.add_argument(
        "--var-108",
        default=DEFAULT_CHANNEL_108,
        metavar="NAME",
        help="variable name of the 10.8 um channel (default IR_108)",
    )
    parser.add_argument(
        "--var-120",
        default=DEFAULT_CHANNEL_120,
        metavar="NAME",
        help="variable name of the 12.0 um channel (default IR_120)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "write summary.json (and scene.png when matplotlib is available) "
            "to this directory; without it only the JSON summary is printed"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="matched-filter response threshold [robust sigma] (default 2.0)",
    )
    parser.add_argument(
        "--min-length-px",
        type=int,
        default=8,
        help="minimum accepted detection length [px] (default 8)",
    )
    parser.add_argument(
        "--bbox",
        default=None,
        metavar="N,W,S,E",
        help="optional crop box in degrees before detection",
    )
    args = parser.parse_args(argv)

    if not args.satellite.is_file():
        print(
            f"ERROR: satellite scene not found: {args.satellite}\n"
            "Download a real granule first (see `stratum fetch goes`) or pass "
            "an existing NetCDF file; no synthetic scene is substituted.",
            file=sys.stderr,
        )
        return 2
    if args.satellite_b is not None and not args.satellite_b.is_file():
        print(
            f"ERROR: second satellite scene not found: {args.satellite_b}",
            file=sys.stderr,
        )
        return 2

    bbox: tuple[float, float, float, float] | None = None
    if args.bbox is not None:
        try:
            bbox = _parse_bbox(args.bbox)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    try:
        scene = load_scene(
            args.satellite,
            bbox=bbox,
            channel_108=args.var_108,
            channel_120=args.var_120,
            path_120=args.satellite_b,
        )
        result = detect_contrail_lines(
            scene.bt_108,
            scene.bt_120,
            threshold=args.threshold,
            min_length_px=args.min_length_px,
        )
    except FileNotFoundError:
        print(f"ERROR: satellite scene not found: {args.satellite}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    objects: list[dict[str, Any]] = []
    for index, obj in enumerate(result["objects"], start=1):
        row, col = (float(value) for value in obj["centroid"])
        record: dict[str, Any] = {
            "index": index,
            "centroid_px": [row, col],
            "length_px": float(obj["length_px"]),
            "angle_px_deg": float(obj["angle_deg"]),
            "mean_btd": float(obj["mean_btd"]),
        }
        if scene.georef is not None:
            lat, lon = scene.georef.to_latlon(row, col)
            record["lat"] = float(lat)
            record["lon"] = float(lon)
            heading = pixel_angle_to_heading_deg(
                float(obj["angle_deg"]), row, col, scene.georef.to_latlon
            )
            record["heading_deg"] = None if not math.isfinite(heading) else float(heading)
        objects.append(record)

    # The row/column crop of a curved geostationary grid is approximate at
    # the box edges, so georeferenced centroids are filtered against the
    # nominal box as well. Objects without coordinates are kept.
    num_detected_raw = int(result["num_detected"])
    if bbox is not None:
        north, west, south, east = bbox

        def _in_box(record: dict[str, Any]) -> bool:
            if "lat" not in record:
                return True
            return (
                math.isfinite(record["lat"])
                and math.isfinite(record["lon"])
                and south <= record["lat"] <= north
                and west <= record["lon"] <= east
            )

        objects = [record for record in objects if _in_box(record)]

    summary: dict[str, Any] = {
        "satellite_file": str(args.satellite),
        "observation_time": (
            None
            if scene.observation_time is None
            else scene.observation_time.isoformat()
        ),
        "georeferencing": scene.georeferencing,
        "cropped": scene.cropped,
        "bbox": None if bbox is None else list(bbox),
        "threshold": float(args.threshold),
        "min_length_px": int(args.min_length_px),
        "image_shape": [int(scene.bt_108.shape[0]), int(scene.bt_108.shape[1])],
        "num_detected": len(objects),
        "num_detected_raw": num_detected_raw,
        "objects": objects,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if scene.georef is None:
        print(
            "NOTE: no georeferencing available (neither goes_imager_projection "
            "nor latitude/longitude coordinates); centroid geo coordinates and "
            "compass headings are omitted."
        )

    if args.out_dir is None:
        print("No --out-dir given: summary printed only, no files written.")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Summary written: {summary_path}")

    btd = compute_btd(scene.bt_108, scene.bt_120)
    scene_path = out_dir / "scene.png"
    if _write_scene_png(scene_path, btd, objects):
        print(f"Scene image written: {scene_path}")
    else:
        print(
            "matplotlib is not importable; scene.png was not written "
            "(the detector itself does not need it)."
        )
    return 0
