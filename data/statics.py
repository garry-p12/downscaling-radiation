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
    save_statics(load_config("data"), ap.parse_args().out)
