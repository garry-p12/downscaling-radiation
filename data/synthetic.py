"""Synthetic store: physically shaped fake POWER/NSRDB for offline smoke runs.

Not a benchmark. It exists so every stage (features, B0-B3, XGBoost, Swin,
evaluation) can be run end to end without 100 GB of downloads, and so the
tests have something with the right structure to chew on. The truth is built
to contain the things the real problem contains:

  * clear-sky kt that varies with elevation and distance to a "coast";
  * moving, persistent (AR(1)) cloud fields at ~20-60 km scales — finer than
    the 1 deg coarse grid, so there is real sub-grid signal to recover;
  * POWER = coarse block mean of the truth, with a multiplicative bias and
    noise (POWER and NSRDB disagree systematically in reality too);
  * a Track B "MODIS cloud fraction" = the true cloud field plus noise.

The truth lives on a ~4 km lattice of NSRDB-like points and goes through the
same binning remap (data/build_store.nsrdb_to_grid) as the real data.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_synthetic.yaml python -m data.synthetic
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy import ndimage

from common import build_target_grid, load_config
from common import regrid as rg
from common.solar import toa_daily_kwh
from data.build_store import assemble, nsrdb_to_grid, write
from data.terrain import beam_shading, horizon_angles, sky_view_factor


def _smooth_noise(rng, shape, sigma):
    return ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")


def make(cfg, seed: int = 0, pt_res: float = 0.04):
    rng = np.random.default_rng(seed)
    dom = cfg["domain"]
    grid = build_target_grid(dom, cfg["grid"])
    lat2d, lon2d = grid.latlon()
    times = pd.date_range(cfg["time"]["train"][0], cfg["time"]["test"][1], freq="D")
    T = len(times)

    # --- NSRDB-like point lattice ------------------------------------------
    pad = 0.3
    plat = np.arange(dom["lat_min"] - pad, dom["lat_max"] + pad, pt_res)
    plon = np.arange(dom["lon_min"] - pad, dom["lon_max"] + pad, pt_res)
    PLAT, PLON = np.meshgrid(plat, plon, indexing="ij")
    ny, nx = PLAT.shape

    # Terrain: an escarpment running NE-SW plus hills; coast to the SE.
    s = (PLON - dom["lon_min"]) / (dom["lon_max"] - dom["lon_min"])
    t = (PLAT - dom["lat_min"]) / (dom["lat_max"] - dom["lat_min"])
    dem_pts = (600 / (1 + np.exp((s - 0.45 + 0.2 * t) * 25)) + 150
               + 80 * _smooth_noise(rng, PLAT.shape, 6))
    coast_d = np.clip((1.0 - s) + t - 0.35, 0, None) * 300        # km-ish
    kt_clear = (0.70 + 0.00008 * dem_pts - 0.06 * np.exp(-coast_d / 60)).clip(0.5, 0.82)

    # Clouds: AR(1) in time, advected eastward, two spatial scales.
    cloud = np.zeros((T, ny, nx), dtype="float32")
    state = _smooth_noise(rng, PLAT.shape, 8)
    season = 0.25 + 0.15 * np.cos(2 * np.pi * (times.dayofyear.values - 15) / 365)
    for i in range(T):
        state = np.roll(state, 2, axis=1)
        state = 0.7 * state + 0.5 * (_smooth_noise(rng, PLAT.shape, 8)
                                     + 0.6 * _smooth_noise(rng, PLAT.shape, 3))
        cf = 1 / (1 + np.exp(-(state * 3.0 + (season[i] - 0.4) * 8)))
        cloud[i] = cf + 0.25 * np.exp(-coast_d / 40) * (times.month[i] in (11, 12, 1, 2))
    cloud = cloud.clip(0, 1)
    kt_pts = kt_clear[None] * (1 - 0.72 * cloud)
    doy = times.dayofyear.values
    toa_pts = np.stack([toa_daily_kwh(plat, d)[:, None].repeat(nx, 1) for d in doy])
    ghi_pts = (kt_pts * toa_pts).astype("float32")

    nsrdb = xr.DataArray(ghi_pts.reshape(T, -1), dims=("time", "point"),
                         coords=dict(time=times.values, lat=("point", PLAT.ravel()),
                                     lon=("point", PLON.ravel())))
    target, counts = nsrdb_to_grid(nsrdb, grid, None,
                                   int(cfg["dataset"].get("min_nsrdb_points", 1)))

    # --- POWER: 1 deg block means of truth, biased ------------------------
    cres = float(cfg["grid"].get("coarse_res_deg", 1.0))
    cp = 1.5
    clat = np.arange(np.floor(dom["lat_min"] - cp) + cres / 2, dom["lat_max"] + cp, cres)
    clon = np.arange(np.floor(dom["lon_min"] - cp) + cres / 2, dom["lon_max"] + cp, cres)
    enc = rg.enclosing_index(clat, clon, PLAT.ravel(), PLON.ravel())
    nc = clat.size * clon.size

    def coarse(a):                                   # (T, npts) -> (T, nc)
        out = rg.coarse_mean(a, enc, nc)
        # Cells the lattice does not reach: fill from the domain mean.
        m = np.nanmean(out, axis=1, keepdims=True)
        return np.where(np.isfinite(out), out, m).reshape(-1, clat.size, clon.size)

    CLAT, _ = np.meshgrid(clat, clon, indexing="ij")
    toa_c = np.stack([toa_daily_kwh(CLAT, d) for d in doy])
    kt_c = coarse(kt_pts.reshape(T, -1)) * 1.04 + 0.02 * rng.standard_normal((T, clat.size, clon.size))
    kt_c = kt_c.clip(0.02, 0.85)
    clr_c = coarse(np.broadcast_to(kt_clear.ravel(), (1, ny * nx)))[0] * 1.03
    cloud_c = coarse(cloud.reshape(T, -1))
    ghi_c = kt_c * toa_c
    dims = ("time", "clat", "clon")
    power = xr.Dataset({
        "power_ghi": (dims, ghi_c), "power_kt": (dims, kt_c),
        "power_toa": (dims, toa_c),
        "power_ghi_clr": (dims, np.broadcast_to(clr_c, kt_c.shape) * toa_c),
        "power_dhi": (dims, ghi_c * (0.25 + 0.6 * cloud_c)),
        "power_albedo": (dims, np.full(kt_c.shape, 0.18)),
        "power_cloud": (dims, cloud_c),
        "power_pw": (dims, 2.5 + 0.5 * np.cos(2 * np.pi * doy / 365)[:, None, None]
                     + 0.1 * rng.standard_normal(kt_c.shape)),
        "power_aod": (dims, 0.15 + 0.05 * rng.random(kt_c.shape)),
    }, coords=dict(time=times.values, clat=clat, clon=clon)).astype("float32")

    # --- statics on the fine grid -----------------------------------------
    tf = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    px, py = tf.transform(PLON.ravel(), PLAT.ravel())
    W, _ = rg.binning_matrix(px, py, grid)
    to_grid = lambda a: rg.apply(W, a.ravel()[None], out_shape=grid.shape)[0]  # noqa: E731
    dem = to_grid(dem_pts)
    hz = horizon_angles(np.nan_to_num(dem, nan=np.nanmean(dem)), grid.res_m, 8, 30_000)
    lc = rng.dirichlet(np.ones(4), size=grid.shape).astype("float32")
    statics = {
        "dem": dem,
        "elev_std": ndimage.generic_filter(np.nan_to_num(dem), np.std, size=3),
        "slope": np.hypot(*np.gradient(np.nan_to_num(dem), grid.res_m)),
        "aspect_sin": np.zeros(grid.shape), "aspect_cos": np.zeros(grid.shape),
        "svf": sky_view_factor(hz), "horizon": hz,
        "coastal_dist": to_grid(coast_d),
        "land_mask": (to_grid(coast_d) > 5).astype("float32"),
        "lc_forest": lc[..., 0], "lc_herbaceous": lc[..., 1],
        "lc_cultivated": lc[..., 2], "lc_developed": lc[..., 3],
    }
    shade = beam_shading(hz, lat2d, lon2d, step_min=60)
    cf_fine = rg.apply(W, cloud.reshape(T, -1), out_shape=grid.shape)
    dynamic = {"modis_cf": (cf_fine + 0.08 * rng.standard_normal(cf_fine.shape)).clip(0, 1)}
    return assemble(cfg, grid, power, target, times.values, counts, statics,
                    dynamic, shade)


def main(seed: int = 0) -> None:
    cfg = load_config("data")
    ds = make(cfg, seed)
    write(ds, cfg["dataset"]["out_zarr"])


if __name__ == "__main__":
    main()
