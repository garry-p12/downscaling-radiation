"""Station parsing and BSRN QC, offline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.stations import bsrn_qc, parse_mts, station_daily

MTS = """  101 ! (c) 2020 Oklahoma Climatological Survey and the Oklahoma Mesonet - all rights reserved
  21 2020 06 01 00 00 00
 STID  STNM  TIME   RELH   TAIR   SRAD
 ARD2   126     0     45   27.0    200
 ARD2   126     5     45   27.2   -996
 ARD2   126    10     45   27.2    310
"""


def test_parse_mts():
    df = parse_mts(MTS)
    assert df.time.iloc[2] == pd.Timestamp("2020-06-01 00:10")
    assert np.isnan(df.ghi.iloc[1]) and df.ghi.iloc[2] == 310


def test_qc_removes_impossible_values():
    t = pd.date_range("2020-06-01 06:00", periods=4, freq="h")    # solar midnight-ish at 97 W
    noon = pd.Timestamp("2020-06-01 18:30")
    df = pd.DataFrame({"time": [*t, noon, noon], "ghi": [0, 5, 400, -50, 900, 3000]})
    out, rep = bsrn_qc(df, 34.2, -97.1)
    assert np.isnan(out.ghi.iloc[2]) and np.isnan(out.ghi.iloc[3]) and np.isnan(out.ghi.iloc[5])
    assert out.ghi.iloc[4] == 900


def test_closure_test_flags_inconsistent_components():
    noon = pd.Timestamp("2020-06-01 18:30")
    df = pd.DataFrame({"time": [noon, noon], "ghi": [900.0, 500.0],
                       "dni": [850.0, 850.0], "dhi": [100.0, 100.0]})
    out, _ = bsrn_qc(df, 34.2, -97.1)
    assert np.isfinite(out.ghi.iloc[0]) and np.isnan(out.ghi.iloc[1])


def test_station_daily_uses_solar_day():
    from common.solar import cos_zenith

    t = pd.date_range("2020-06-01", "2020-06-04", freq="5min", inclusive="left")
    v = np.clip(cos_zenith(t.values, [34.2], [-97.1])[:, 0], 0, None) * 900
    d = station_daily(pd.DataFrame({"time": t, "ghi": v}), 34.2, -97.1, "5min", "end", 0.1)
    full = d.set_index("date").ghi_kwh
    assert np.isfinite(full[pd.Timestamp("2020-06-02")])
    assert 7 < full[pd.Timestamp("2020-06-02")] < 10


def test_parse_mesonet_daily_units_and_missing():
    from data.stations import parse_mesonet_daily

    text = "STID,YEAR,MONTH,DAY,ATOT\nACME,2020,6,1,28.8\nacme ,2020,6,2,-996\n"
    d = parse_mesonet_daily(text)
    assert list(d.station_id) == ["ACME", "ACME"]
    assert abs(d.ghi_kwh.iloc[0] - 8.0) < 1e-9 and np.isnan(d.ghi_kwh.iloc[1])
    assert str(d.date.iloc[1].date()) == "2020-06-02"
