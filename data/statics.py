"""Static covariates on the fine grid.

Two sources, both reused from the temperature project rather than rebuilt:

1. The temperature super-store (``sources.statics.temperature_store``) already
   holds dem, coastal_dist, land_mask, NLCD fractions and terrain shape on the
   identical 10 km EPSG:5070 grid. Channels are read from day 0 of its input
   tensor, which holds PHYSICAL values (the temperature trainer normalises on
   the fly from norm_stats.json; checked 2026-10: dem 0-2173 m, coastal_dist
   0-962 km). The grid coordinates are checked, not assumed.

2. The raw 3 arcsec DEM (``dem_tif``) for the radiation-specific geometry the
   temperature work never needed: slope, aspect, horizon angles, sky-view
   factor. Reprojected (average) to ``terrain_res_m`` (1 km), derived there,
   and block-averaged to 10 km (data/terrain.py explains why not at 10 km).

Aspect is returned as slope-weighted sin/cos (eastness/northness at 1 km,
then averaged): on flat cells aspect is numerical noise, and weighting by
sin(slope) sends those cells to 0 instead of a random bearing.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from common.grid import TargetGrid
from data import terrain


def import_temperature_statics(store: str, channels: list[str],
                               grid: TargetGrid) -> dict[str, np.ndarray]:
    import xarray as xr

    ds = xr.open_zarr(store)
    if ds.sizes["y"] != grid.y.size or ds.sizes["x"] != grid.x.size or not (
            np.allclose(ds.x.values, grid.x) and np.allclose(ds.y.values, grid.y)):
        raise ValueError(f"{store} is not on this grid "
                         f"({ds.sizes['y']}x{ds.sizes['x']} vs {grid.shape})")
    have = [str(c) for c in ds.channel_in.values]
    out = {}
    for ch in channels:
        if ch not in have:
            warnings.warn(f"[statics] {ch} not in temperature store; skipped")
            continue
        out[ch] = ds["input"].isel(time=0).sel(channel_in=ch).values.astype("float32")
    return out


# NLCD Level-1 groups, as in the temperature project's data/covariates.py, so
# natively built fractions mean the same thing as the imported 10 km ones.
NLCD_GROUPS = {
    "lc_water": (11, 12),
    "lc_developed": (21, 22, 23, 24),
    "lc_forest": (41, 42, 43),
    "lc_shrub": (51, 52),
    "lc_herbaceous": (71, 72, 73, 74),
    "lc_cultivated": (81, 82),
    "lc_wetland": (90, 95),
}


def _aggregate(src: np.ndarray, transform, crs, grid: TargetGrid) -> np.ndarray:
    """Area-average a native raster onto the grid; NaN source pixels are skipped."""
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    out = np.full(grid.shape, np.nan, dtype="float32")
    reproject(np.ascontiguousarray(src, dtype="float32"), out, src_transform=transform,
              src_crs=crs, src_nodata=np.nan,
              dst_transform=from_origin(grid.x[0] - grid.res_m / 2, grid.y[0] + grid.res_m / 2,
                                        grid.res_m, grid.res_m),
              dst_crs=grid.crs, dst_nodata=np.nan, resampling=Resampling.average)
    return out


def landcover_fractions(path: str, grid: TargetGrid) -> dict[str, np.ndarray]:
    """Fraction of each cell in each NLCD group, from the native (~200 m) raster.
    Code 0 / nodata (outside NLCD: Mexico, open Gulf) is left out of the
    denominator, so a coastal cell reports the composition of its covered part."""
    import rasterio

    with rasterio.open(path) as src:
        codes, transform, crs, nodata = src.read(1), src.transform, src.crs, src.nodata
    valid = (codes != 0) if nodata is None else (codes != nodata) & (codes != 0)
    out = {}
    for name, cls in NLCD_GROUPS.items():
        m = np.isin(codes, cls).astype("float32")
        m[~valid] = np.nan
        out[name] = _aggregate(m, transform, crs, grid)
    return out


def coastal_distance(path: str, grid: TargetGrid, pad_deg: float = 8.0) -> np.ndarray:
    """Distance (km) from each cell centre to the nearest coastline segment."""
    import geopandas as gpd
    import shapely

    lat, lon = grid.latlon()
    bbox = (float(lon.min()) - pad_deg, float(lat.min()) - pad_deg,
            float(lon.max()) + pad_deg, float(lat.max()) + pad_deg)
    line = gpd.read_file(path, bbox=bbox).to_crs(grid.crs).geometry.union_all()
    xx, yy = np.meshgrid(grid.x, grid.y)
    d = shapely.distance(shapely.points(xx.ravel(), yy.ravel()), line)
    return (d.reshape(grid.shape) / 1000.0).astype("float32")


def dem_moments(path: str, grid: TargetGrid) -> tuple[np.ndarray, np.ndarray]:
    """Cell mean elevation and the std of the native (~90 m) DEM within the cell:
    sub-grid relief, the temperature project's ``elev_std``."""
    import rasterio

    with rasterio.open(path) as src:
        z, transform, crs = src.read(1).astype("float32"), src.transform, src.crs
        if src.nodata is not None:
            z[z == src.nodata] = np.nan
    m1 = _aggregate(z, transform, crs, grid)
    m2 = _aggregate(z * z, transform, crs, grid)
    return m1, np.sqrt(np.clip(m2 - m1 ** 2, 0, None)).astype("float32")


def mosaic_dem(tiles: str, out: str, res_deg: float = 1 / 1200) -> None:
    """Average-resample DEM tiles (e.g. Copernicus 1 arcsec) to one ``res_deg``
    GeoTIFF (default 3 arcsec, ~90 m): the input every DEM-derived static uses."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in sorted(Path(tiles).glob("*.tif"))]
    arr, transform = merge(srcs, res=res_deg, resampling=Resampling.average, nodata=-9999)
    prof = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=1,
                dtype="float32", crs=srcs[0].crs, transform=transform, nodata=-9999,
                compress="deflate", tiled=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(arr[0].astype("float32"), 1)
    print(f"[statics] DEM mosaic {arr.shape[1:]} from {len(srcs)} tiles -> {out}")


def land_mask_from(spec: dict, grid: TargetGrid) -> np.ndarray:
    """``spec = {npz, res_m}``: another grid's land mask over the same domain,
    looked up at this grid's cell centres (both grids share the domain's
    north-west corner, so a 5 km cell lies inside exactly one 10 km cell)."""
    lm = np.load(spec["npz"])["land_mask"]
    r = float(spec["res_m"])
    x0, y0 = grid.x[0] - grid.res_m / 2, grid.y[0] + grid.res_m / 2
    col = np.clip(((grid.x - x0) // r).astype(int), 0, lm.shape[1] - 1)
    row = np.clip(((y0 - grid.y) // r).astype(int), 0, lm.shape[0] - 1)
    return lm[np.ix_(row, col)].astype("float32")


def native_statics(sc: dict, grid: TargetGrid) -> dict[str, np.ndarray]:
    """Land cover, coastal distance, land mask, dem and elev_std built from the
    raw rasters on ANY grid (the temperature store exists only at 10 km)."""
    out: dict[str, np.ndarray] = {}
    if sc.get("landcover_tif") and Path(sc["landcover_tif"]).exists():
        out.update(landcover_fractions(sc["landcover_tif"], grid))
        out["land_mask"] = (np.isfinite(out["lc_water"]) & (out["lc_water"] < 0.5)).astype("float32")
    if sc.get("land_mask_from"):
        # Same land area as another grid's statics (e.g. 10 km), so results on
        # the two grids are scored over the same cells. NLCD stops at the
        # border; that grid's mask includes northern Mexico.
        out["land_mask"] = land_mask_from(sc["land_mask_from"], grid)
    # Cells outside NLCD (Mexico) take the composition of the nearest covered
    # cell, as the temperature store's gap fill does.
    gaps = np.isnan(out.get("lc_water", np.zeros(1)))
    if gaps.any() and (~gaps).any():
        from scipy import ndimage

        idx = ndimage.distance_transform_edt(gaps, return_distances=False, return_indices=True)
        for k in NLCD_GROUPS:
            out[k] = out[k][tuple(idx)]
    if sc.get("coastline_shp") and Path(sc["coastline_shp"]).exists():
        out["coastal_dist"] = coastal_distance(sc["coastline_shp"], grid)
    if sc.get("dem_tif") and Path(sc["dem_tif"]).exists():
        out["dem"], out["elev_std"] = dem_moments(sc["dem_tif"], grid)
    return out


def _subgrid(grid: TargetGrid, res_m: float):
    f = int(round(grid.res_m / res_m))
    if f < 1 or not np.isclose(grid.res_m / f, res_m):
        raise ValueError(f"terrain_res_m {res_m} must divide {grid.res_m}")
    r = grid.res_m / f
    x = grid.x[0] - grid.res_m / 2 + r / 2 + r * np.arange(grid.x.size * f)
    y = grid.y[0] + grid.res_m / 2 - r / 2 - r * np.arange(grid.y.size * f)
    return f, TargetGrid(crs=grid.crs, res_m=r, x=x, y=y)


def reproject_to(path: str, grid: TargetGrid, resampling: str = "average") -> np.ndarray:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    ny, nx = grid.shape
    dst = np.full((ny, nx), np.nan, dtype="float32")
    transform = from_origin(grid.x[0] - grid.res_m / 2, grid.y[0] + grid.res_m / 2,
                            grid.res_m, grid.res_m)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=transform,
                  dst_crs=grid.crs, dst_nodata=np.nan,
                  resampling=getattr(Resampling, resampling))
    return dst


def terrain_statics(dem_fine: np.ndarray, dx: float, f: int, n_az: int,
                    max_dist: float) -> dict[str, np.ndarray]:
    """Derive geometry on a fine DEM and average to the target grid (factor f)."""
    if np.isnan(dem_fine).any():
        from scipy import ndimage

        idx = ndimage.distance_transform_edt(np.isnan(dem_fine), return_distances=False,
                                             return_indices=True)
        dem_fine = dem_fine[tuple(idx)]
    slope, aspect = terrain.slope_aspect(dem_fine, dx)
    hz = terrain.horizon_angles(dem_fine, dx, n_az=n_az, max_dist=max_dist)
    svf = terrain.sky_view_factor(hz)
    def bm(a):
        return terrain.block_mean(a, f).astype("float32")

    mean = terrain.block_mean(dem_fine, f)
    var = np.clip(terrain.block_mean(dem_fine ** 2, f) - mean ** 2, 0, None)
    return {
        "dem": mean.astype("float32"),
        "elev_std_1km": np.sqrt(var).astype("float32"),
        "slope": bm(slope),
        "aspect_sin": bm(np.sin(aspect) * np.sin(slope)),
        "aspect_cos": bm(np.cos(aspect) * np.sin(slope)),
        "svf": bm(svf),
        "horizon": bm(hz),
    }


def build_statics(cfg, grid: TargetGrid) -> dict[str, np.ndarray]:
    """All statics as {name: (ny, nx)} plus 'horizon' (n_az, ny, nx).

    ``sources.statics.precomputed`` (an .npz written by ``save_statics``)
    short-circuits everything: the cluster has neither the temperature store
    nor the 630 MB DEM, and the statics are identical wherever they are built.
    """
    sc = cfg["sources"]["statics"]
    pre = sc.get("precomputed")
    if pre and Path(pre).exists():
        z = np.load(pre)
        out = {k: z[k] for k in z.files if k != "shade_beam"}
        if out["dem"].shape != grid.shape:
            raise ValueError(f"{pre} is {out['dem'].shape}, grid is {grid.shape}")
        return out
    out: dict[str, np.ndarray] = {}
    ts = sc.get("temperature_store")
    if ts and Path(ts).exists():
        out.update(import_temperature_statics(ts, list(sc.get("import_channels", [])),
                                              grid))
    elif sc.get("landcover_tif") or sc.get("coastline_shp"):
        out.update(native_statics(sc, grid))
    else:
        warnings.warn(f"[statics] temperature store {ts!r} not found; land cover "
                      "and coastal distance will be missing")

    dem_tif = sc.get("dem_tif")
    if dem_tif and Path(dem_tif).exists():
        res = float(sc.get("terrain_res_m", 1000))
        f, sub = _subgrid(grid, res)
        print(f"[statics] DEM -> {sub.shape} at {res:.0f} m for terrain geometry")
        dem_fine = reproject_to(dem_tif, sub, "average")
        geo = terrain_statics(dem_fine, res, f, int(sc.get("horizon_azimuths", 16)),
                              float(sc.get("horizon_max_km", 30)) * 1000)
        out.setdefault("dem", geo["dem"])
        if "elev_std" not in out:
            out["elev_std"] = geo["elev_std_1km"]
        for k in ("slope", "aspect_sin", "aspect_cos", "svf", "horizon"):
            out[k] = geo[k]
    else:
        warnings.warn(f"[statics] dem_tif {dem_tif!r} not found; no terrain geometry")

    if sc.get("albedo_tif") and Path(sc["albedo_tif"]).exists():
        out["albedo"] = reproject_to(sc["albedo_tif"], grid, "average")
    if "land_mask" not in out:
        out["land_mask"] = np.ones(grid.shape, dtype="float32")
    return out


def save_statics(cfg, path: str) -> None:
    """Build statics + beam shading here and write them to one .npz."""
    from common import build_target_grid
    from data.terrain import beam_shading

    grid = build_target_grid(cfg["domain"], cfg["grid"])
    st = build_statics(cfg, grid)
    if "horizon" in st:
        lat, lon = grid.latlon()
        st["shade_beam"] = beam_shading(st["horizon"], lat, lon)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **st)
    print(f"[statics] wrote {path}: {sorted(st)}")


if __name__ == "__main__":
    import argparse

    from common import load_config

    ap = argparse.ArgumentParser(description="precompute statics to .npz")
    ap.add_argument("out")
    ap.add_argument("--mosaic-dem", metavar="TILES_DIR", default=None,
                    help="instead: mosaic DEM tiles to OUT (3 arcsec GeoTIFF)")
    a = ap.parse_args()
    if a.mosaic_dem:
        mosaic_dem(a.mosaic_dem, a.out)
    else:
        save_statics(load_config("data"), a.out)
