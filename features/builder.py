"""Per-pixel, per-day features (plan 4) and the leakage-safe statistics behind them.

One ``FeatureBuilder`` per store. On construction it loads the store into
memory (10 km x 21 years is ~0.3 GB of float32), precomputes the coarse->fine
bilinear matrix and enclosing-cell index (cached on disk), and FITS every
statistic that touches the target on TRAINING YEARS x TRAINING PIXELS ONLY:

  * kt_clim_fine  NSRDB kt day-of-year climatology (plan 4.7). Inside the
                  holdout + buffer it is filled by Gaussian normalised
                  convolution from training pixels — the holdout's own NSRDB
                  values never enter;
  * kt_bil_clim   bilinear-POWER kt climatology on the same days, so
                  B3's ratio map compares like with like;
  * POWER kt climatology per coarse cell (for kt_anom) — input-only, but
                  still training days only (plan 6).

Target. The model target is the clearness-index residual
    r = kt_nsrdb - kt_bilinear,   kt_nsrdb = GHI_nsrdb / TOA_fine,
with TOA_fine analytic per pixel per day (common/solar.toa_daily_kwh). The
same TOA reconstructs GHI from a predicted kt, so the reframing is exact.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from common import regrid as rg
from common.solar import toa_daily_kwh

STATIC_NAMES = ("dem", "slope", "aspect_sin", "aspect_cos", "svf", "elev_std",
                "coastal_dist", "albedo", "lc_water", "lc_developed", "lc_barren",
                "lc_forest", "lc_shrub", "lc_herbaceous", "lc_cultivated",
                "lc_wetland")
REFERENCES = {"NSRDB": "nsrdb_ghi", "ERA5L": "era5l_ghi"}   # label -> store variable
POWER_DYNAMIC = {"power_pw": "power_pw", "power_aod": "power_aod",
                 "power_cloud": "power_cloud"}


def doy_climatology(field: np.ndarray, doy: np.ndarray, use: np.ndarray,
                    smooth_days: int) -> np.ndarray:
    """(T, ...) -> (366, ...) mean per day-of-year over ``use`` days, smoothed
    with a WRAPPED centred window (Dec 31 adjacent to Jan 1). DOYs with no
    sample take the nearest DOY that has one."""
    shape = field.shape[1:]
    s = np.zeros((366,) + shape)
    c = np.zeros((366,) + shape)
    for d in range(1, 367):
        sel = use & (doy == d)
        if sel.any():
            f = field[sel]
            s[d - 1] = np.nansum(f, axis=0)
            c[d - 1] = np.isfinite(f).sum(0)
    if smooth_days and smooth_days > 1:
        k = int(smooth_days)
        kern = np.ones(k)
        s = ndimage.convolve1d(s, kern, axis=0, mode="wrap")
        c = ndimage.convolve1d(c, kern, axis=0, mode="wrap")
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(c > 0, s / c, np.nan)
    flat = clim.reshape(366, -1)
    missing = np.isnan(flat).all(1)
    if missing.any() and (~missing).any():
        good = np.nonzero(~missing)[0]
        for d in np.nonzero(missing)[0]:
            flat[d] = flat[good[np.abs(good - d).argmin()]]
    return clim.astype("float32")


def fill_from(values: np.ndarray, known: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Replace values outside ``known`` by Gaussian normalised convolution of
    the known ones (over the trailing two axes). Known cells are untouched."""
    v = np.where(known, np.nan_to_num(values), 0.0)
    w = known.astype("float64")
    axes_sigma = (0,) * (values.ndim - 2) + (sigma_cells, sigma_cells)
    num = ndimage.gaussian_filter(v, axes_sigma, mode="nearest")
    den = ndimage.gaussian_filter(np.broadcast_to(w, v.shape).astype("float64"),
                                  axes_sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        filled = np.where(den > 1e-6, num / den, np.nan)
    # Far from any training pixel the kernel vanishes: fall back to the mean.
    filled = np.where(np.isfinite(filled), filled,
                      np.nanmean(np.where(known, values, np.nan),
                                 axis=(-2, -1), keepdims=True))
    return np.where(known, values, filled).astype("float32")


def physical_clip(kt: np.ndarray, kt_clr: np.ndarray | None, kt_max: float,
                  clr_margin: float) -> np.ndarray:
    """Clip kt to [0, min(kt_max, kt_clr * clr_margin)] (plan 2.2)."""
    hi = np.full_like(kt, kt_max) if kt_clr is None else np.minimum(
        kt_max, np.nan_to_num(kt_clr, nan=kt_max) * clr_margin)
    return np.clip(kt, 0.0, hi)


def _rolling(a: np.ndarray, k: int, center: bool) -> np.ndarray:
    """Rolling mean along time of a (T, N) array, NaN-aware, edges partial."""
    df = pd.DataFrame(a)
    return df.rolling(k, center=center, min_periods=1).mean().to_numpy("float32")


class FeatureBuilder:
    def __init__(self, ds: xr.Dataset, train_cfg: dict, cache_dir: str | None = None,
                 target_var: str = "nsrdb_ghi"):
        self.cfg = train_cfg
        fcfg = train_cfg["features"]
        self.times = pd.DatetimeIndex(ds.time.values)
        self.doy = self.times.dayofyear.values
        self.split = np.asarray(ds.split.values).astype(str)
        self.ny, self.nx = ds.sizes["y"], ds.sizes["x"]
        self.lat = ds.lat.values.astype("float64")
        self.lon = ds.lon.values.astype("float64")
        self.res_km = float(ds.attrs.get("res_m", 10000)) / 1000

        self.valid = ds.valid.values.astype(bool)
        self.holdout = ds.holdout.values.astype(bool)
        self.exclusion = ds.exclusion.values.astype(bool)
        self.train_pix = self.valid & ~self.exclusion
        self.train_days = self.split == "train"

        # Coarse grid and mappings.
        clat, clon = ds.clat.values, ds.clon.values
        self.nc = clat.size * clon.size
        key = rg.grid_key(clat, clon, self.lat, self.lon)
        cdir = Path(cache_dir) if cache_dir else None
        self.W = (rg.cached(cdir / f"bilinear_{key}.npz",
                            lambda: rg.bilinear_matrix(clat, clon, self.lat, self.lon))
                  if cdir else rg.bilinear_matrix(clat, clon, self.lat, self.lon))
        self.enc = rg.enclosing_index(clat, clon, self.lat, self.lon)

        def coarse(name):
            return ds[name].values.reshape(len(self.times), -1).astype("float32") \
                if name in ds else None

        self.c = {k: coarse(k) for k in ("power_ghi", "power_kt", "power_toa",
                                         "power_ghi_clr", "power_dhi",
                                         *POWER_DYNAMIC.values())}
        if self.c["power_kt"] is None:
            self.c["power_kt"] = self.c["power_ghi"] / self.c["power_toa"]
        self.c["kt_clr"] = (self.c["power_ghi_clr"] / self.c["power_toa"]
                            if self.c["power_ghi_clr"] is not None else None)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.c["clr_ratio"] = (self.c["power_ghi"] / self.c["power_ghi_clr"]
                                   if self.c["power_ghi_clr"] is not None else None)
            self.c["dhi_frac"] = (np.clip(self.c["power_dhi"] / self.c["power_ghi"], 0, 1)
                                  if self.c["power_dhi"] is not None else None)
        center = bool(fcfg.get("rolling_center", True))
        for k in fcfg.get("rolling", [3, 7]):
            self.c[f"roll{k}"] = _rolling(self.c["power_kt"], int(k), center)
        sm = int(fcfg.get("clim_smooth_days", 31))
        self.c_clim = doy_climatology(self.c["power_kt"], self.doy, self.train_days, sm)

        # Statics.
        self.static = {k: ds[k].values.astype("float32") for k in STATIC_NAMES if k in ds}
        if "dem" in self.static:
            dem = np.nan_to_num(self.static["dem"], nan=np.nanmean(self.static["dem"]))
            dem_c = rg.coarse_mean(dem.ravel()[None], self.enc, self.nc)[0]
            dem_c = np.where(np.isfinite(dem_c), dem_c, np.nanmean(dem))
            self.static["dem_anom"] = (dem - (self.W @ dem_c).reshape(self.ny, self.nx)).astype("float32")
        self.shade = ds["shade_beam"].values if "shade_beam" in ds else None
        self.dyn = {k[4:]: ds[k].values.astype("float32")
                    for k in ds.data_vars if k.startswith("dyn_")}

        # Target in kt space. TOA is analytic; cache per DOY.
        self._toa = {}
        # The training target (NSRDB by default; ERA5-Land for the target-swap
        # experiment) and every gridded reference product available, which the
        # station check scores side by side.
        self.target_var = target_var
        self.ghi = ds[target_var].values.astype("float32")
        self.refs = {lab: (self.ghi if var == target_var else ds[var].values.astype("float32"))
                     for lab, var in REFERENCES.items() if var in ds}
        self.kt = np.empty_like(self.ghi)
        for i, d in enumerate(self.doy):
            self.kt[i] = self.ghi[i] / np.maximum(self.toa(d), 1e-6)
        self.kt[:, ~self.valid] = np.nan

        # Training-only fine climatologies, filled into the excluded zone.
        sigma = float(fcfg.get("holdout_fill_sigma_km", 40)) / self.res_km
        kt_train = np.where(self.train_pix[None], self.kt, np.nan)
        clim_fine = doy_climatology(kt_train, self.doy, self.train_days, sm)
        self.kt_clim_fine = fill_from(clim_fine, self.train_pix[None].repeat(366, 0), sigma)
        bil_all = self.bilinear(self.c["power_kt"])
        self.kt_bil_clim = doy_climatology(bil_all, self.doy, self.train_days, sm)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.b3_ratio = np.clip(self.kt_clim_fine / self.kt_bil_clim, 0.5, 2.0)

    # ------------------------------------------------------------------ #
    def toa(self, doy: int) -> np.ndarray:
        if doy not in self._toa:
            self._toa[doy] = toa_daily_kwh(self.lat, doy).astype("float32")
        return self._toa[doy]

    def bilinear(self, coarse_flat: np.ndarray) -> np.ndarray:
        """(..., nc) coarse -> (..., ny, nx) via the cached bilinear matrix."""
        out = rg.apply(self.W, coarse_flat)
        return out.reshape(*coarse_flat.shape[:-1], self.ny, self.nx).astype("float32")

    def kt_ceiling(self, t: int) -> np.ndarray | None:
        return None if self.c["kt_clr"] is None else self.bilinear(self.c["kt_clr"][t])

    def clip(self, kt: np.ndarray, t: int) -> np.ndarray:
        k = self.cfg["kt"]
        return physical_clip(kt, self.kt_ceiling(t), float(k["max"]),
                             float(k["clr_margin"]))

    def available(self, names: list[str]) -> list[str]:
        """Drop requested features this store cannot provide, loudly."""
        ok, missing = [], []
        for n in names:
            (ok if self._has(n) else missing).append(n)
        if missing:
            warnings.warn(f"[features] not available in this store, dropped: {missing}")
        return ok

    def _has(self, n: str) -> bool:
        if n.startswith("kt_roll"):
            return f"roll{n[7:]}" in self.c
        if n in ("kt_bilinear", "kt_coarse_cell", "kt_anom", "kt_clim_fine",
                 "doy_sin", "doy_cos", "lat", "lon"):
            return True
        if n == "kt_clr_ratio":
            return self.c["clr_ratio"] is not None
        if n in POWER_DYNAMIC:
            return self.c[POWER_DYNAMIC[n]] is not None
        return n in self.static or n in self.dyn

    def track(self, name: str) -> list[str]:
        """A: POWER + statics. B: A + MODIS/POWER dynamics. C: B + GOES clouds."""
        f = self.cfg["features"]
        names = list(f["track_A"])
        if name.upper() in ("B", "C"):
            names += list(f["track_B_extra"])
        if name.upper() == "C":
            names += list(f.get("track_C_extra", []))
        return self.available(names)

    # ------------------------------------------------------------------ #
    def day_fields(self, t: int, names: list[str]) -> dict[str, np.ndarray]:
        """All requested features for day index ``t`` as (ny, nx) float32."""
        d = self.doy[t]
        kt = self.c["power_kt"][t]
        out = {}
        for n in names:
            if n == "kt_bilinear":
                v = self.bilinear(kt)
            elif n == "kt_coarse_cell":
                v = kt[self.enc].reshape(self.ny, self.nx)
            elif n == "kt_clr_ratio":
                v = self.bilinear(self.c["clr_ratio"][t])
            elif n == "kt_anom":
                v = self.bilinear(kt - self.c_clim[d - 1])
            elif n.startswith("kt_roll"):
                v = self.bilinear(self.c[f"roll{n[7:]}"][t])
            elif n == "kt_clim_fine":
                v = self.kt_clim_fine[d - 1]
            elif n == "doy_sin":
                v = np.full((self.ny, self.nx), np.sin(2 * np.pi * d / 365.25))
            elif n == "doy_cos":
                v = np.full((self.ny, self.nx), np.cos(2 * np.pi * d / 365.25))
            elif n == "lat":
                v = self.lat
            elif n == "lon":
                v = self.lon
            elif n in POWER_DYNAMIC:
                v = self.bilinear(self.c[POWER_DYNAMIC[n]][t])
            elif n in self.static:
                v = self.static[n]
            elif n in self.dyn:
                v = self.dyn[n][t]
            else:
                raise KeyError(n)
            out[n] = np.asarray(v, dtype="float32")
        return out

    def stack(self, t: int, names: list[str]) -> np.ndarray:
        f = self.day_fields(t, names)
        return np.stack([f[n] for n in names])

    def kt_bilinear(self, t: int) -> np.ndarray:
        return self.bilinear(self.c["power_kt"][t])

    def residual_target(self, t: int) -> np.ndarray:
        return self.kt[t] - self.kt_bilinear(t)

    def _by_day(self, rows_t: np.ndarray):
        order = np.argsort(rows_t, kind="stable")
        ts, starts = np.unique(rows_t[order], return_index=True)
        bounds = np.r_[starts, order.size]
        for k, t in enumerate(ts):
            yield int(t), order[bounds[k]:bounds[k + 1]]

    def gather_kt_bilinear(self, rows_t: np.ndarray, rows_pix: np.ndarray):
        out = np.empty(rows_t.size, dtype="float32")
        for t, idx in self._by_day(rows_t):
            out[idx] = self.kt_bilinear(t).ravel()[rows_pix[idx]]
        return out

    def table(self, rows_t: np.ndarray, rows_pix: np.ndarray, names: list[str]):
        """Gather features for (day, flat pixel) pairs, grouped by day so each
        day's fields are built once.

        Returns X (n, F), y (n,) residual kt target, and aux: toa, kt_bil,
        kt_ceil (POWER clear-sky ceiling or NaN), ghi (target kWh/m2/day).
        """
        n = rows_t.size
        X = np.empty((n, len(names)), dtype="float32")
        y = np.empty(n, dtype="float32")
        aux = {k: np.empty(n, dtype="float32") for k in ("toa", "kt_bil", "kt_ceil", "ghi")}
        for t, idx in self._by_day(rows_t):
            pix = rows_pix[idx]
            S = self.stack(t, names).reshape(len(names), -1)
            X[idx] = S[:, pix].T
            kb = self.kt_bilinear(t).ravel()[pix]
            y[idx] = self.kt[t].ravel()[pix] - kb
            aux["kt_bil"][idx] = kb
            aux["toa"][idx] = self.toa(self.doy[t]).ravel()[pix]
            ceil = self.kt_ceiling(t)
            aux["kt_ceil"][idx] = np.nan if ceil is None else ceil.ravel()[pix]
            aux["ghi"][idx] = self.ghi[t].ravel()[pix]
        return X, y, aux

    def reconstruct(self, kt_bil, resid, toa, kt_ceil) -> np.ndarray:
        """Residual kt -> clipped kt -> GHI (kWh/m2/day), elementwise."""
        k = self.cfg["kt"]
        ceil = None if np.all(np.isnan(kt_ceil)) else kt_ceil
        kt = physical_clip(kt_bil + resid, ceil, float(k["max"]), float(k["clr_margin"]))
        return kt * toa
