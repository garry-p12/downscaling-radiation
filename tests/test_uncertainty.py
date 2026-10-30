import numpy as np
import pandas as pd

from evaluation import uncertainty as U


def _frame(n=4000, seed=0):
    """Two years of station-days whose error sd is proportional to s_mm6."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.05, 0.5, n)
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 731, n), "D")
    centre = rng.uniform(1, 7, n)
    return pd.DataFrame({"date": dates, "obs": centre + rng.normal(0, 1, n) * s, U.CENTRE: centre,
                         "s_mm6": s, "s_static": 1.0})


def test_conformal_covers_nominal_level():
    df = _frame()
    for m in ("mm6", "static"):
        w, qs = U.calibrate(df, m)
        sc = U.score(df, w)
        assert abs(sc["coverage"] - U.LEVEL) < 0.03
        assert set(qs) == {2020, 2021}


def test_informative_spread_is_sharper_and_ranks_errors():
    df = _frame()
    s_mm, s_st = U.score(df, U.calibrate(df, "mm6")[0]), U.score(df, U.calibrate(df, "static")[0])
    assert s_mm["interval_score"] < s_st["interval_score"]
    assert s_mm["rank_corr"] > 0.2


def test_index_terciles():
    w = np.arange(9.0)
    assert list(U.index_classes(w)) == ["green"] * 3 + ["amber"] * 3 + ["red"] * 3


def test_kl_zero_for_identical_and_positive_otherwise():
    x = np.random.default_rng(1).uniform(0, 8, 5000)
    assert U.kl(x, x) == 0
    assert U.kl(x, np.clip(x + 1, 0, 9)) > 0.05
