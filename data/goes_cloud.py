"""GOES-16 ABI Clear Sky Mask -> daily, sunlight-weighted 10 km cloud fraction.

TRACK C, and why it is flagged. NSRDB derives its clouds from GOES. Feeding
GOES cloud information to a model scored against NSRDB lets the model partly
re-derive NSRDB instead of downscaling (plan 2.4). It is done here only
because MODIS cannot place sub-60 km daily texture (two snapshots a day;
correlation with NSRDB 0.03-0.07 below 60 km), and every Track C result is
checked against independent ground stations (USCRN) for exactly that failure:
evaluation/evaluate.station_eval reports how far each model's station errors
track NSRDB's own station errors.

Differences from NSRDB's own cloud inputs, for the record: NSRDB uses
UW-GOES / PATMOS-x cloud TYPE and optical properties; this uses NOAA's
operational binary cloud mask (BCM) only — no optical depth, no type.

Source: s3://noaa-goes16/ABI-L2-ACMC (CONUS sector, 2 km, every 5 min),
anonymous. Two scans per hour (nearest to :00 and :30) are read straight from
S3 in memory; nothing raw is stored. Available from 2018-01-01.

Daily aggregation per 10 km cell: each scan's 2 km mask is binned to the cell
(area mean; cell NaN if < 50% observed), then the day's scans are averaged
with weight max(cos z, 0) at the cell — the sunlit, energy-relevant part of
the day. The day is the local SOLAR day; in this domain solar midnight is at
06:16-06:56 UTC, so the UTC window [D 06:30, D+1 06:30) is used and the night
scans at its ends carry zero weight. Outputs:

    goes_cf      sunlight-weighted daily cloud fraction
    goes_cf_am   same, scans before local solar noon
    goes_cf_pm   same, after

Second product, ``--product cod`` (s3://noaa-goes16/ABI-L2-CODC, 4 km, daytime
cloud optical depth tau). Clear pixels (DQF 6) count as tau = 0. Each pixel's
tau becomes a two-stream cloud transmittance T = 1 / (1 + 0.75 (1 - g) tau),
g = 0.85, which is roughly linear in the surface flux it removes; T is binned and
sunlight-weighted exactly like the mask, giving goes_tcld / _am / _pm. Unlike
the binary mask this says how much a cloud dims the surface, not only whether
it is there.

Usage (vista2):
    python -m data.goes_cloud days --start 2018-01-01 --end 2021-12-31 --workers 4
    python -m data.goes_cloud assemble        # daily npz -> one nc per year
    python -m data.goes_cloud add-to-store
    python -m data.goes_cloud --product cod days --start 2018-01-01 --end 2021-12-31
"""
from __future__ import annotations

import argparse
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common import build_target_grid, load_config
from common import regrid as rg
from common.solar import cos_zenith

G_ASYM = 0.85


def _bcm(ds) -> np.ndarray:
    return ds["BCM"].values.astype("float64")


def _transmittance(ds) -> np.ndarray:
    tau = ds["COD"].values.astype("float64")
    tau = np.where(ds["DQF"].values == 6, 0.0, tau)          # 6 = clear
    return 1.0 / (1.0 + 0.75 * (1.0 - G_ASYM) * tau)


# product -> (bucket, per-pixel field, raw subdir, output variable prefix)
PRODUCTS = {"acm": ("noaa-goes16/ABI-L2-ACMC", _bcm, "goes", "goes_cf"),
            "cod": ("noaa-goes16/ABI-L2-CODC", _transmittance, "goes_cod", "goes_tcld")}
PRODUCT = "acm"
MIN_COVERAGE = 0.5
MIN_DAY_WEIGHT = 0.7          # fraction of the day's sunlight weight observed


def _root(cfg) -> Path:
    d = Path(cfg["store_root"]) / "raw" / PRODUCTS[PRODUCT][2]
    (d / "daily").mkdir(parents=True, exist_ok=True)
    return d


def _scan_start(key: str) -> pd.Timestamp:
    s = key.rsplit("_s", 1)[1][:13]                    # YYYYJJJHHMMSS
    return pd.to_datetime(s[:7], format="%Y%j") + pd.Timedelta(
        hours=int(s[7:9]), minutes=int(s[9:11]), seconds=int(s[11:13]))


def day_keys(fs, day: pd.Timestamp) -> list[str]:
    """Scan keys nearest :00 and :30 of every hour in [day 06:30, day+1 06:30)."""
    t0 = day + pd.Timedelta(hours=6, minutes=30)
    keys = []
    for h in range(25):
        t = t0 + pd.Timedelta(hours=h)
        pref = f"{PRODUCTS[PRODUCT][0]}/{t:%Y}/{t.dayofyear:03d}/{t:%H}/"
        try:
            ks = fs.ls(pref)
        except FileNotFoundError:
            continue
        for k in ks:
            st = _scan_start(k)
            if t0 <= st < t0 + pd.Timedelta(days=1) and st.minute % 30 < 5:
                keys.append(k)
    return sorted(set(keys))


class Mapper:
    """Fixed-grid (x, y) -> 10 km binning, cached per fixed-grid definition."""

    def __init__(self, grid, mode: str = "bin"):
        self.grid, self.mode = grid, mode       # mode: config grid.point_remap
        self.cache: dict = {}

    def get(self, ds: xr.Dataset):
        p = ds.goes_imager_projection.attrs
        key = (float(ds.x[0]), float(ds.y[0]), ds.sizes["x"], ds.sizes["y"],
               float(p["longitude_of_projection_origin"]))
        if key not in self.cache:
            from pyproj import Proj, Transformer

            h = float(p["perspective_point_height"])
            geos = Proj(proj="geos", h=h, lon_0=key[4], sweep=p["sweep_angle_axis"],
                        a=float(p["semi_major_axis"]), b=float(p["semi_minor_axis"]))
            X, Y = np.meshgrid(ds.x.values * h, ds.y.values * h)
            lon, lat = geos(X, Y, inverse=True)
            ok = np.isfinite(lon) & (np.abs(lon) < 1e20)
            g = self.grid
            tf = Transformer.from_crs("EPSG:4326", g.crs, always_xy=True)
            px, py = tf.transform(np.where(ok, lon, 0), np.where(ok, lat, 0))
            px = np.where(ok, px, -1e12)                 # off-disk: outside the grid
            W, _ = rg.point_matrix(px.ravel(), py.ravel(), g, self.mode)
            rows = np.nonzero(np.asarray(W.getnnz(axis=0)) > 0)[0]   # used source px
            self.cache[key] = (W[:, rows], rows)
        return self.cache[key]


_MAPPER = None          # one per worker process


def process_day(args) -> str:
    cfg, day, product = args
    global PRODUCT
    PRODUCT = product
    out = _root(cfg) / "daily" / f"{day:%Y%m%d}.npz"
    if out.exists():
        return "skip"
    import s3fs

    fs = s3fs.S3FileSystem(anon=True)
    global _MAPPER
    grid = build_target_grid(cfg["domain"], cfg["grid"])
    lat, lon = grid.latlon()
    if _MAPPER is None:
        _MAPPER = Mapper(grid, cfg["grid"].get("point_remap", "bin"))
    mapper = _MAPPER
    shape = grid.shape
    acc = {k: np.zeros(shape) for k in ("all", "am", "pm")}
    wts = {k: np.zeros(shape) for k in ("all", "am", "pm")}
    wtot = {k: np.zeros(shape) for k in ("all", "am", "pm")}
    keys = day_keys(fs, day)
    for k in keys:
        try:
            ds = xr.open_dataset(io.BytesIO(fs.cat(k)), engine="h5netcdf")
        except Exception:  # noqa: BLE001  (a corrupt scan is a missing scan)
            continue
        W, rows = mapper.get(ds)
        bcm = PRODUCTS[PRODUCT][1](ds).ravel()[rows]
        cf = rg.apply(W, bcm[None])[0].reshape(shape)
        cov = rg.apply(W, np.isfinite(bcm)[None].astype("float64"))[0].reshape(shape)
        cf = np.where(cov >= MIN_COVERAGE, cf, np.nan)
        t = np.datetime64(_scan_start(k).to_datetime64(), "ns")
        w = np.clip(cos_zenith(np.array([t]), lat.ravel(), lon.ravel())[0], 0, None).reshape(shape)
        solar_h = (t.astype("datetime64[m]").astype("int64") / 60.0 + lon / 15.0) % 24
        for part, sel in (("all", np.ones(shape, bool)), ("am", solar_h < 12), ("pm", solar_h >= 12)):
            ws = np.where(sel, w, 0.0)
            wtot[part] += ws
            good = np.isfinite(cf)
            acc[part] += np.where(good, ws * cf, 0.0)
            wts[part] += np.where(good, ws, 0.0)
    res = {}
    for part in acc:
        frac = np.where(wtot[part] > 0, wts[part] / np.maximum(wtot[part], 1e-12), 0)
        val = np.where(wts[part] > 0, acc[part] / np.maximum(wts[part], 1e-12), np.nan)
        res[part] = np.where(frac >= MIN_DAY_WEIGHT, val, np.nan).astype("float32")
    np.savez_compressed(out, cf=res["all"], am=res["am"], pm=res["pm"], n_scans=len(keys))
    return f"{day:%Y-%m-%d} {len(keys)} scans"


def run_days(cfg, start: str, end: str, workers: int) -> None:
    days = pd.date_range(start, end, freq="D")
    cfg = dict(cfg)
    with ProcessPoolExecutor(workers) as ex:
        for i, msg in enumerate(ex.map(process_day, [(cfg, d, PRODUCT) for d in days],
                                       chunksize=4)):
            if msg != "skip" and i % 20 == 0:
                print(f"[goes] {i + 1}/{len(days)} {msg}", flush=True)
    print("[goes] days done", flush=True)


def assemble(cfg) -> None:
    root = _root(cfg)
    files = sorted((root / "daily").glob("*.npz"))
    grid = build_target_grid(cfg["domain"], cfg["grid"])
    by_year: dict[str, list] = {}
    for f in files:
        by_year.setdefault(f.stem[:4], []).append(f)
    for y, fs_ in sorted(by_year.items()):
        t = pd.to_datetime([f.stem for f in fs_], format="%Y%m%d")
        arrs = {k: np.stack([np.load(f)[k] for f in fs_]) for k in ("cf", "am", "pm")}
        pre = PRODUCTS[PRODUCT][3]
        ds = xr.Dataset({pre if k == "cf" else f"{pre}_{k}": (("time", "y", "x"), v)
                         for k, v in arrs.items()},
                        coords=dict(time=t, y=grid.y, x=grid.x))
        ds.to_netcdf(root / f"{pre}_{y}.nc")
        print(f"[goes] {y}: {len(t)} days, NaN {100 * float(np.isnan(arrs['cf']).mean()):.1f}%")


def add_to_store(cfg) -> None:
    store = cfg["dataset"]["out_zarr"]
    st = xr.open_zarr(store)
    files = sorted(_root(cfg).glob(f"{PRODUCTS[PRODUCT][3]}_*.nc"))
    g = xr.concat([xr.open_dataset(f) for f in files], "time").sortby("time")
    g = g.reindex(time=st.time.values)
    from data.build_store import append_to_store

    # One variable at a time: three full-period 5 km stacks do not fit a login node.
    for k in g.data_vars:
        v = g[k].values.astype("float32")
        append_to_store(xr.Dataset({f"dyn_{k}": (("time", "y", "x"), v)}).chunk(
            {"time": st.chunks["time"][0]}), store)
        print(f"[goes] dyn_{k}: NaN {100 * float(np.isnan(v).mean()):.1f}%", flush=True)
        del v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", choices=sorted(PRODUCTS), default="acm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("days")
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--workers", type=int, default=4)
    sub.add_parser("assemble")
    sub.add_parser("add-to-store")
    a = ap.parse_args()
    PRODUCT = a.product
    c = load_config("data")
    if a.cmd == "days":
        run_days(c, a.start, a.end, a.workers)
    elif a.cmd == "assemble":
        assemble(c)
    else:
        add_to_store(c)
