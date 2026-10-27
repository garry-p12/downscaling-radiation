"""The NSRDB-mimicry check: a model that copies NSRDB must be caught."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluation.evaluate import station_eval


def test_copying_nsrdb_is_flagged_and_truth_is_not(tmp_path, rng):
    T, ny, nx = 60, 4, 4
    times = pd.date_range("2020-01-01", periods=T)
    truth = rng.uniform(2, 8, (T, ny, nx))
    nsrdb = truth + 0.5 + rng.normal(0, 0.6, truth.shape)        # biased, noisy target
    b1 = truth + rng.normal(0, 1.0, truth.shape)
    fb = SimpleNamespace(times=times.values, valid=np.ones((ny, nx), bool), ghi=nsrdb)
    stations = pd.DataFrame([dict(station_id=f"s{i}", lat=0, lon=0, row=i, col=i,
                                  in_grid=True, region="r", network="x") for i in range(3)])
    daily = pd.concat([pd.DataFrame({"station_id": f"s{i}", "date": times,
                                     "ghi_kwh": truth[:, i, i]}) for i in range(3)])
    (tmp_path / "st").mkdir()
    stations.to_csv(tmp_path / "st/stations.csv", index=False)
    daily.to_csv(tmp_path / "st/station_daily.csv", index=False)
    dcfg = {"sources": {"stations": {"out_dir": str(tmp_path / "st")}}}
    preds = {"B1": b1, "copy": nsrdb.copy(),
             "oracle": truth + rng.normal(0, 0.1, truth.shape)}   # independent errors
    _, summ = station_eval(fb, dcfg, preds, np.arange(T))
    s = summ[summ.region == "all"].set_index("model")
    assert s.loc["copy", "nsrdb_mimicry"] > 0.99 and s.loc["copy", "rmse_nsrdb"] < 1e-9
    assert s.loc["oracle", "rmse_stn"] < 0.15 and abs(s.loc["oracle", "nsrdb_mimicry"]) < 0.3
    # the copy looks perfect against NSRDB but not against the stations
    assert s.loc["copy", "skill_nsrdb_vs_B1"] > 0.99 > s.loc["copy", "skill_stn_vs_B1"]


def test_centred_skill_ignores_bias_and_has_ci(tmp_path, rng):
    T, ny, nx = 120, 3, 3
    times = pd.date_range("2020-01-01", periods=T)
    truth = rng.uniform(2, 8, (T, ny, nx))
    b1 = truth + rng.normal(0, 1.0, truth.shape)
    fb = SimpleNamespace(times=times.values, valid=np.ones((ny, nx), bool),
                         ghi=truth + rng.normal(0, 0.5, truth.shape))
    stations = pd.DataFrame([dict(station_id=f"s{i}", lat=0, lon=0, row=i, col=i,
                                  in_grid=True, region="r", network="x") for i in range(3)])
    daily = pd.concat([pd.DataFrame({"station_id": f"s{i}", "date": times,
                                     "ghi_kwh": truth[:, i, i]}) for i in range(3)])
    (tmp_path / "st").mkdir()
    stations.to_csv(tmp_path / "st/stations.csv", index=False)
    daily.to_csv(tmp_path / "st/station_daily.csv", index=False)
    dcfg = {"sources": {"stations": {"out_dir": str(tmp_path / "st")}}}
    preds = {"B1": b1, "biased_b1": b1 + 2.0}            # same day-to-day error, +2 bias
    per, summ = station_eval(fb, dcfg, preds, np.arange(T))
    s = summ[summ.region == "all"].set_index("model")
    assert abs(s.loc["biased_b1", "skill_crmse_vs_B1"]) < 1e-9      # bias is not day-to-day error
    assert s.loc["biased_b1", "skill_stn_vs_B1"] < -1                # but it is total error
    assert s.loc["NSRDB", "skill_crmse_lo90"] <= s.loc["NSRDB", "skill_crmse_vs_B1"] <= s.loc["NSRDB", "skill_crmse_hi90"]
    assert "pairs" in per.attrs and len(per.attrs["pairs"]) == 3 * T


def test_mimicry_is_per_reference(tmp_path, rng):
    T, ny, nx = 90, 3, 3
    times = pd.date_range("2020-01-01", periods=T)
    truth = rng.uniform(2, 8, (T, ny, nx))
    x = truth + rng.normal(0, 0.6, truth.shape)
    y = truth + rng.normal(0, 0.6, truth.shape)            # independent errors
    fb = SimpleNamespace(times=times.values, valid=np.ones((ny, nx), bool), ghi=x,
                         refs={"NSRDB": x, "ERA5L": y})
    stations = pd.DataFrame([dict(station_id=f"s{i}", lat=0, lon=0, row=i, col=i,
                                  in_grid=True, region="r", network="x") for i in range(3)])
    daily = pd.concat([pd.DataFrame({"station_id": f"s{i}", "date": times,
                                     "ghi_kwh": truth[:, i, i]}) for i in range(3)])
    (tmp_path / "st").mkdir()
    stations.to_csv(tmp_path / "st/stations.csv", index=False)
    daily.to_csv(tmp_path / "st/station_daily.csv", index=False)
    dcfg = {"sources": {"stations": {"out_dir": str(tmp_path / "st")}}}
    _, summ = station_eval(fb, dcfg, {"B1": truth + rng.normal(0, 1, truth.shape),
                                      "copy_y": y + 1e-3}, np.arange(T))
    s = summ[summ.region == "all"].set_index("model")
    assert s.loc["copy_y", "era5l_mimicry"] > 0.99
    assert abs(s.loc["copy_y", "nsrdb_mimicry"]) < 0.3
