"""Target-grid construction and bbox subsetting.

Vendored from downscaling-temperature@1d2c3bb (common/grid.py). The fine grid MUST be
bit-identical to the temperature pipeline's (same CRS, res, bbox snapping) so
the two variables can be compared cell by cell.

The target grid is a regular grid in the projected target CRS (default
EPSG:5070 CONUS Albers) at ``target_res_m`` spacing. Building it once and
reusing it everywhere guarantees every source lands on identical cell centers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer


@dataclass
class TargetGrid:
    crs: str
    res_m: float
    x: np.ndarray          # projected cell-center x coords (1-D, meters)
    y: np.ndarray          # projected cell-center y coords (1-D, meters)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.y.size, self.x.size)

    def meshgrid(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.x, self.y)

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) 2-D arrays of cell centers for regridding."""
        xx, yy = self.meshgrid()
        tf = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(xx, yy)
        return lat, lon


def build_target_grid(domain: dict, grid: dict) -> TargetGrid:
    """Construct the target grid covering ``domain`` (lat/lon bbox).

    The bbox corners are projected into the target CRS and snapped outward to
    the resolution so the grid fully contains the requested domain.
    """
    crs = grid["target_crs"]
    res = float(grid["target_res_m"])
    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    # Project all four corners; bbox in projected space is their extent.
    lons = [domain["lon_min"], domain["lon_max"], domain["lon_min"], domain["lon_max"]]
    lats = [domain["lat_min"], domain["lat_min"], domain["lat_max"], domain["lat_max"]]
    xs, ys = tf.transform(lons, lats)

    x_min = np.floor(min(xs) / res) * res
    x_max = np.ceil(max(xs) / res) * res
    y_min = np.floor(min(ys) / res) * res
    y_max = np.ceil(max(ys) / res) * res

    # Cell centers.
    x = np.arange(x_min + res / 2, x_max, res)
    y = np.arange(y_max - res / 2, y_min, -res)   # north-up
    return TargetGrid(crs=crs, res_m=res, x=x, y=y)


def subset_bbox(ds, domain: dict, lat_name: str = "lat", lon_name: str = "lon"):
    """Subset an xarray Dataset/DataArray to the domain bbox.

    Handles both ascending and descending latitude and 0..360 longitudes.
    """
    lon = ds[lon_name]
    lon_min, lon_max = domain["lon_min"], domain["lon_max"]
    if float(lon.max()) > 180:  # convert requested bounds to 0..360
        lon_min = lon_min % 360
        lon_max = lon_max % 360

    lat = ds[lat_name]
    lat_slice = (
        slice(domain["lat_max"], domain["lat_min"])
        if float(lat[0]) > float(lat[-1])
        else slice(domain["lat_min"], domain["lat_max"])
    )
    return ds.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)})
