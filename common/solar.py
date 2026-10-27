"""Solar geometry: declination, Earth-Sun distance, zenith, azimuth, daily TOA.

Everything downstream that needs "how much sunlight would arrive with no
atmosphere" goes through this module, so the TOA used to build the clearness
index target (kt = GHI / TOA) and the TOA used to reconstruct GHI from a
predicted kt are the same function by construction. That identity is what
makes the kt reframing (plan 2.2) exact: no learning error enters through TOA.

Formulas
--------
``spencer`` (default): Spencer (1971) Fourier series for declination,
eccentricity correction and equation of time. Accurate to ~0.0006 rad in
declination, well inside anything a daily total can resolve.

``fao56``: the simpler FAO-56 (Allen et al. 1998, eqs. 21-25) closed forms.
Kept because FAO-56 publishes a worked example (Example 8: 20 S, 3 Sept,
Ra = 32.2 MJ/m2/day) that the test suite pins the daily integral against.

The solar constant defaults to 1361 W/m2 (Kopp & Lean 2011, the CERES-era
value). FAO-56 uses 1367 (0.0820 MJ/m2/min); pass ``gsc=1367`` to reproduce it.
"""
from __future__ import annotations

import numpy as np

GSC = 1361.0                      # W/m2, total solar irradiance at 1 AU


def _day_angle(doy) -> np.ndarray:
    """Spencer day angle (rad); doy 1 = Jan 1."""
    return 2.0 * np.pi * (np.asarray(doy, dtype="float64") - 1.0) / 365.0


def declination(doy, method: str = "spencer") -> np.ndarray:
    """Solar declination (rad)."""
    if method == "fao56":
        return 0.409 * np.sin(2 * np.pi * np.asarray(doy, "float64") / 365 - 1.39)
    g = _day_angle(doy)
    return (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
            - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))


def eccentricity(doy, method: str = "spencer") -> np.ndarray:
    """(r0 / r)^2, the inverse-square Earth-Sun distance correction."""
    if method == "fao56":
        return 1 + 0.033 * np.cos(2 * np.pi * np.asarray(doy, "float64") / 365)
    g = _day_angle(doy)
    return (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g)
            + 0.000719 * np.cos(2 * g) + 0.000077 * np.sin(2 * g))


def equation_of_time_min(doy) -> np.ndarray:
    """Equation of time (minutes): apparent minus mean solar time."""
    g = _day_angle(doy)
    return 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                     - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))


def toa_daily_kwh(lat_deg, doy, gsc: float = GSC,
                  method: str = "spencer") -> np.ndarray:
    """Daily extraterrestrial irradiation on a horizontal plane, kWh/m2/day.

    Analytic integral of G_sc * E0 * cos(z) over the sunlit part of the day
    (Duffie & Beckman eq. 1.10.3). Broadcasts ``lat_deg`` against ``doy``, so
    a (H, W) latitude field and a scalar day gives the TOA map for that day.
    Polar day/night are handled by clipping the sunset hour angle.
    """
    phi = np.deg2rad(np.asarray(lat_deg, dtype="float64"))
    dec = declination(doy, method)
    e0 = eccentricity(doy, method)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0)
    ws = np.arccos(x)                                     # sunset hour angle
    h0 = (24.0 / np.pi) * gsc * e0 * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return np.maximum(h0, 0.0) / 1000.0                   # Wh -> kWh


def _hour_angle(times_utc, lon_deg) -> tuple[np.ndarray, np.ndarray]:
    """(hour angle rad, doy) for UTC instants broadcast against longitudes.

    ``times_utc`` is a datetime64 array of shape (T,) and ``lon_deg`` any shape
    broadcastable against (T, 1) — typically (N,), giving (T, N).
    """
    t = np.asarray(times_utc, dtype="datetime64[ns]")
    day = t.astype("datetime64[D]")
    doy = (day - day.astype("datetime64[Y]")).astype("int64") + 1
    hours = (t - day).astype("timedelta64[ns]").astype("float64") / 3.6e12
    lon = np.asarray(lon_deg, dtype="float64")
    tst = (hours[:, None] + lon[None, ...] / 15.0
           + equation_of_time_min(doy)[:, None] / 60.0)  # true solar time, h
    return np.deg2rad(15.0 * (tst - 12.0)), doy


def cos_zenith(times_utc, lat_deg, lon_deg) -> np.ndarray:
    """cos(solar zenith) at UTC instants (T,) for sites (N,) -> (T, N).

    Not clipped: negative values mean the sun is below the horizon, which the
    daily aggregation uses to tell night from missing data.
    """
    ha, doy = _hour_angle(times_utc, lon_deg)
    dec = declination(doy)[:, None]
    phi = np.deg2rad(np.asarray(lat_deg, dtype="float64"))[None, ...]
    return np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)


def sun_position(times_utc, lat_deg, lon_deg) -> tuple[np.ndarray, np.ndarray]:
    """(elevation rad, azimuth rad clockwise from north) -> each (T, N)."""
    ha, doy = _hour_angle(times_utc, lon_deg)
    dec = declination(doy)[:, None]
    phi = np.deg2rad(np.asarray(lat_deg, dtype="float64"))[None, ...]
    cz = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    elev = np.arcsin(np.clip(cz, -1, 1))
    az = np.arctan2(np.sin(ha),
                    np.cos(ha) * np.sin(phi) - np.tan(dec) * np.cos(phi))
    return elev, np.mod(az + np.pi, 2 * np.pi)


def extraterrestrial_normal(times_utc) -> np.ndarray:
    """Extraterrestrial normal irradiance (W/m2) at each instant, shape (T,)."""
    t = np.asarray(times_utc, dtype="datetime64[D]")
    doy = (t - t.astype("datetime64[Y]")).astype("int64") + 1
    return GSC * eccentricity(doy)
