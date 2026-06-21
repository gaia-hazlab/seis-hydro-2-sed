#!/usr/bin/env python3
"""Fetch auxiliary multidisciplinary data for the bedload animation:
   - NRCS SNOTEL hourly precipitation near Mt. Rainier (drives the ARs)
   - USGS discharge at several gages along/near the corridor
Saves config/aux_timeseries.json (time-indexed series, SI units).

Usage: pixi run python workflows/07_fetch_aux_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import requests
from obspy import UTCDateTime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from utils import fetch_usgs_gage_timeseries  # noqa: E402

DATA = ROOT / "notebooks" / "data"
OUT = ROOT / "config" / "aux_timeseries.json"
START, END = UTCDateTime("2025-12-01"), UTCDateTime("2025-12-16")

SNOTEL = {  # triplet -> (name, lon, lat)
    "679:WA:SNTL": ("Paradise", -121.748, 46.783),
    "941:WA:SNTL": ("Mowich", -121.952, 46.928),
    "928:WA:SNTL": ("Huckleberry Creek", -121.588, 47.066),
}
GAGES = {  # id -> (name, lon, lat)
    "12092000": ("Puyallup nr Electron", -122.0351, 46.9037),
    "12093500": ("Puyallup nr Orting", -122.2079, 47.0392),
    "12094000": ("Carbon nr Fairfax", -122.0326, 47.0279),
    "12101500": ("Puyallup at Puyallup", -122.3271, 47.2084),
    "12082500": ("Nisqually nr National", -122.0828, 46.7558),
}
IN_TO_MM = 25.4


def fetch_snotel() -> dict:
    out = {}
    base = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
    for trip, (name, lon, lat) in SNOTEL.items():
        try:
            r = requests.get(base, params={
                "stationTriplets": trip, "elements": "PREC", "duration": "HOURLY",
                "beginDate": "2025-12-01", "endDate": "2025-12-16",
                "centralTendencyType": "NONE", "returnFlags": "false"}, timeout=60)
            r.raise_for_status()
            js = r.json()
            vals = js[0]["data"][0]["values"]
            s = pd.Series({v["date"]: v.get("value") for v in vals}, dtype="float64").dropna()
            s.index = pd.to_datetime(s.index, utc=True)
            incr = (s.diff().clip(lower=0) * IN_TO_MM)            # hourly precip increment (mm)
            out[name] = dict(lon=lon, lat=lat,
                             time=[t.isoformat() for t in incr.index],
                             precip_mm=[round(float(x), 2) for x in incr.values])
            print(f"  SNOTEL {name}: {len(incr)} hourly values, total {incr.sum():.0f} mm")
        except Exception as e:
            print(f"  SNOTEL {name} failed: {e}")
    return out


def fetch_discharge() -> dict:
    out = {}
    for gid, (name, lon, lat) in GAGES.items():
        try:
            df = fetch_usgs_gage_timeseries(gid, START, END, data_dir=DATA, use_cache=True)
            col = "discharge_cms" if "discharge_cms" in df.columns else None
            if col is None:
                print(f"  gage {gid}: no discharge_cms"); continue
            s = pd.to_numeric(df[col], errors="coerce").dropna().resample("1h").median().dropna()
            out[gid] = dict(name=name, lon=lon, lat=lat,
                            time=[t.isoformat() for t in s.index],
                            q_cms=[round(float(x), 2) for x in s.values])
            print(f"  gage {gid} {name}: {len(s)} hourly, peak {s.max():.0f} m³/s")
        except Exception as e:
            print(f"  gage {gid} failed: {e}")
    return out


def main() -> int:
    print("Fetching SNOTEL precip …")
    precip = fetch_snotel()
    print("Fetching USGS discharge …")
    disch = fetch_discharge()
    OUT.write_text(json.dumps({"precip": precip, "discharge": disch}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
