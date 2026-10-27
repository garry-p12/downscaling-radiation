"""Feature builder: climatology, fill, clipping and — most important — that
nothing about the holdout's TARGET reaches any feature or baseline."""
from __future__ import annotations

import numpy as np

from features.builder import FeatureBuilder, doy_climatology, fill_from, physical_clip


def test_physical_clip():
    kt = np.array([-0.1, 0.5, 0.8, 0.95])
    assert np.allclose(physical_clip(kt, None, 0.85, 1.1), [0, 0.5, 0.8, 0.85])
    ceil = np.array([0.7, 0.7, 0.7, np.nan])
    assert np.allclose(physical_clip(kt, ceil, 0.85, 1.1), [0, 0.5, 0.77, 0.85])


def test_climatology_wraps_new_year():
    doy = np.r_[np.arange(1, 366), np.arange(1, 366)]
    field = np.where((doy <= 3) | (doy >= 363), 1.0, 0.0)[:, None]
    use = np.ones(doy.size, bool)
    clim = doy_climatology(field, doy, use, smooth_days=5)
    assert clim[0, 0] > 0.5 and clim[364, 0] > 0.5 and clim[180, 0] == 0


def test_climatology_uses_only_selected_days():
    doy = np.r_[np.arange(1, 366), np.arange(1, 366)]
    field = np.r_[np.zeros(365), np.ones(365)][:, None]
    use = np.r_[np.ones(365, bool), np.zeros(365, bool)]
    assert np.allclose(doy_climatology(field, doy, use, 1)[:365], 0)


def test_fill_keeps_known_cells(rng):
    v = rng.normal(size=(2, 10, 10))
    known = np.ones((10, 10), bool); known[3:6, 3:6] = False
    out = fill_from(v, np.broadcast_to(known, v.shape), 2.0)
    assert np.allclose(out[:, known], v[:, known]) and np.isfinite(out).all()


def test_holdout_target_never_leaks(synth_store, train_cfg):
    """Scramble NSRDB inside holdout+buffer and in val/test years: every
    feature on every day, and B3's ratio map, must be bit-identical."""
    ds = synth_store
    fb0 = FeatureBuilder(ds, train_cfg)
    excl = ds.exclusion.values.astype(bool)
    split = ds.split.values.astype(str)
    g = ds.nsrdb_ghi.values.copy()
    g[:, excl] = np.random.default_rng(5).uniform(0, 9, g[:, excl].shape)
    g[split != "train"] = g[split != "train"][::-1]
    fb1 = FeatureBuilder(ds.assign(nsrdb_ghi=(ds.nsrdb_ghi.dims, g)), train_cfg)
    names = fb0.track("B")
    for t in (0, len(fb0.times) // 2, len(fb0.times) - 1):
        a, b = fb0.day_fields(t, names), fb1.day_fields(t, names)
        for n in names:
            assert np.array_equal(a[n], b[n], equal_nan=True), n
    assert np.array_equal(fb0.b3_ratio, fb1.b3_ratio, equal_nan=True)


def test_kt_target_roundtrips_to_ghi(synth_store, train_cfg):
    fb = FeatureBuilder(synth_store, train_cfg)
    t = 40
    ghi = fb.kt[t] * fb.toa(fb.doy[t])
    m = fb.valid
    assert np.allclose(ghi[m], fb.ghi[t][m], rtol=1e-5, equal_nan=True)


def test_clearsky_normaliser_round_trips(synth_store, train_cfg):
    import copy

    cfg = copy.deepcopy(train_cfg)
    cfg["kt"]["normalise"] = "clearsky"
    fb = FeatureBuilder(synth_store, cfg)
    t = int(np.nonzero(fb.split == "test")[0][0])
    # index x normaliser gives back the target, and the clear-sky normaliser
    # sits below TOA (clear-sky transmittance < 1)
    ok = fb.valid & np.isfinite(fb.ghi[t])
    assert np.allclose((fb.kt[t] * fb.norm(t))[ok], fb.ghi[t][ok], rtol=1e-5)
    assert np.all(fb.norm(t)[ok] <= fb.toa(fb.doy[t])[ok] + 1e-6)
    # a zero residual reproduces POWER's own all-sky/clear-sky ratio
    assert np.allclose(fb.kt_bilinear(t), fb.bilinear(np.clip(fb.c["clr_ratio"][t], 0, 2)))


def test_temporal_features_are_shifted_days(synth_store, train_cfg):
    fb = FeatureBuilder(synth_store, train_cfg)
    names = fb.track("A+T")
    assert "kt_bilinear@-1" in names and "kt_bilinear@+1" in names
    f = fb.day_fields(10, ["kt_bilinear", "kt_bilinear@-1", "kt_bilinear@+1"])
    assert np.allclose(f["kt_bilinear@-1"], fb.kt_bilinear(9))
    assert np.allclose(f["kt_bilinear@+1"], fb.kt_bilinear(11))
    # clamped at the record's edges
    assert np.allclose(fb.day_fields(0, ["kt_bilinear@-1"])["kt_bilinear@-1"], fb.kt_bilinear(0))
