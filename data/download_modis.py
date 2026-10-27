"""Download daily MODIS cloud state (MOD09GA / MYD09GA state_1km) via AppEEARS.

Why this product. Track B needs fine-scale cloud information that is NOT
derived from GOES (NSRDB is built from GOES clouds, plan 2.4). The MODIS daily
surface-reflectance product carries a 1 km cloud-state QA layer (bits 0-1:
0 clear, 1 cloudy, 2 mixed, 3 not set/assumed clear) from the MOD35 cloud mask,
on the gridded sinusoidal tiles, for Terra (~10:30 local, 2000-) and Aqua
(~13:30, 2002-07-). AppEEARS subsets it server-side to the domain and
reprojects to geographic, so only the one uint16 layer is transferred
(~1 GB/product/year instead of terabytes of full granules or swaths).

Auth: Earthdata login from ~/.netrc (machine urs.earthdata.nasa.gov) -> an
AppEEARS bearer token. Never printed or written.

One area task per (product, year), submitted together; AppEEARS queues them.
Task ids are recorded in <out_dir>/tasks.json so a re-run polls and downloads
instead of resubmitting. Completed downloads are skipped.

Usage (on vista2, plan: downloads run there):
    python -m data.download_modis submit   --years 2001 ... 2021
    python -m data.download_modis download             # poll + fetch, resumable
"""
from __future__ import annotations

import argparse
import json
import netrc
import time
import urllib.error
import urllib.request
from base64 import b64encode
from pathlib import Path

from common import load_config

API = "https://appeears.earthdatacloud.nasa.gov/api"
PRODUCTS = {"terra": "MOD09GA.061", "aqua": "MYD09GA.061"}
LAYER = "state_1km_1"
AQUA_START = 2002


def _req(url: str, token: str | None = None, data: dict | None = None,
         method: str | None = None, auth: str | None = None, timeout=300):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if auth:
        headers["Authorization"] = f"Basic {auth}"
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    for attempt in range(8):
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            # AppEEARS rate-limits per user; back off instead of dying.
            if e.code not in (429, 502, 503, 504) or attempt == 7:
                raise
            time.sleep(min(600, 30 * 2 ** attempt))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            # DNS / network blips (one killed the loop during the 2026-10-22
            # Vista /work outage): wait and retry rather than die.
            if attempt == 7:
                raise
            time.sleep(min(600, 30 * 2 ** attempt))
    raise RuntimeError("unreachable")


def login() -> str:
    creds = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    if not creds:
        raise SystemExit("no urs.earthdata.nasa.gov entry in ~/.netrc")
    user, _, pw = creds
    auth = b64encode(f"{user}:{pw}".encode()).decode()
    tok = json.loads(_req(f"{API}/login", auth=auth, method="POST", data={}))
    return tok["token"]


def _out(cfg) -> Path:
    d = Path(cfg["sources"]["dynamic"]["modis_cf"].get(
        "raw_dir", Path(cfg["store_root"]) / "raw" / "modis"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _task(cfg, sat: str, year: int) -> dict:
    d, pad = cfg["domain"], 0.2
    lo, hi = d["lon_min"] - pad, d["lon_max"] + pad
    la, ha = d["lat_min"] - pad, d["lat_max"] + pad
    start = f"07-04-{year}" if (sat == "aqua" and year == AQUA_START) else f"01-01-{year}"
    return {
        "task_type": "area",
        "task_name": f"rad_{sat}_{year}",
        "params": {
            "dates": [{"startDate": start, "endDate": f"12-31-{year}"}],
            "layers": [{"product": PRODUCTS[sat], "layer": LAYER}],
            "output": {"format": {"type": "netcdf4"}, "projection": "geographic"},
            "geo": {"type": "FeatureCollection", "features": [{
                "type": "Feature", "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [lo, la], [hi, la], [hi, ha], [lo, ha], [lo, la]]]}}]},
        },
    }


def submit(cfg, years: list[int]) -> None:
    out = _out(cfg)
    reg_path = out / "tasks.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    token = login()
    for year in years:
        for sat in PRODUCTS:
            key = f"{sat}_{year}"
            if key in reg or (sat == "aqua" and year < AQUA_START):
                continue
            r = json.loads(_req(f"{API}/task", token=token, data=_task(cfg, sat, year),
                                method="POST"))
            reg[key] = r["task_id"]
            reg_path.write_text(json.dumps(reg, indent=2))
            print(f"[modis] submitted {key} -> {r['task_id']}", flush=True)
            time.sleep(1)


def download(cfg, poll_s: int = 300) -> None:
    out = _out(cfg)
    reg = json.loads((out / "tasks.json").read_text())
    token = login()
    todo = {k: v for k, v in reg.items() if not (out / k / "DONE").exists()}
    while todo:
        for key, tid in list(todo.items()):
            time.sleep(3)                     # stay well under the rate limit
            try:
                st = json.loads(_req(f"{API}/task/{tid}", token=token))
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    token = login()
                    continue
                raise
            status = st.get("status")
            if status in ("error", "expired"):
                print(f"[modis] {key} {status}: resubmit with `submit` after "
                      f"removing it from tasks.json", flush=True)
                todo.pop(key)
                continue
            if status != "done":
                continue
            files = json.loads(_req(f"{API}/bundle/{tid}", token=token))["files"]
            d = out / key
            d.mkdir(exist_ok=True)
            for f in files:
                if not f["file_name"].endswith(".nc"):
                    continue
                dest = d / Path(f["file_name"]).name
                if dest.exists() and dest.stat().st_size == f["file_size"]:
                    continue
                raw = _req(f"{API}/bundle/{tid}/{f['file_id']}", token=token, timeout=3600)
                dest.write_bytes(raw)
                print(f"[modis] {key}: {dest.name} {len(raw) / 1e6:.0f} MB", flush=True)
            (d / "DONE").touch()
            todo.pop(key)
        if todo:
            print(f"[modis] {len(todo)} task(s) pending; next poll in {poll_s}s", flush=True)
            time.sleep(poll_s)
    print("[modis] all downloaded", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--years", type=int, nargs="+", required=True)
    sub.add_parser("download")
    a = ap.parse_args()
    c = load_config("data")
    if a.cmd == "submit":
        submit(c, a.years)
    else:
        download(c)
