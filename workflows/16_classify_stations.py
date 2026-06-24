#!/usr/bin/env python3
"""Classify each processed station by whether river seismology yielded a signal,
so all figures can integrate every station and honestly mark where it did NOT work.

Status from the best flow-band P–Q correlation r (|log10 P| vs |log10 Q|):
  observed  : r >= 0.70   (usable virtual gage / scaling)
  marginal  : 0.50–0.70
  none      : r < 0.50    (attempted, no usable river signal)
  control   : out-of-drainage / traffic reference (UW.BHW, UW.TEHA)
Writes config/station_status.json (with coords, sample rate, basin).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "data" / "results"

import sys; sys.path.insert(0, str(ROOT / "src"))
from riverseis.analysis import clip_event, fit_scaling, load_timeseries  # noqa: E402
CONTROL = {"UW.BHW", "UW.TEHA"}
FLOW_PREF = ["5.0-15.0", "2.0-8.0", "1.0-5.0"]   # best -> fallback flow band
FN = re.compile(r"^(?P<sid>[A-Z0-9]+\.[A-Z0-9]+)_(?P<band>[\d.]+-[\d.]+)Hz_timeseries\.csv$")


def best_r(sid):
    bands = {}
    for f in RESULTS.glob(f"{sid}_*Hz_timeseries.csv"):
        m = FN.match(f.name)
        if m:
            bands[m["band"]] = f
    for b in FLOW_PREF:
        if b in bands:
            # Use the SAME robust, flood-windowed fit as the scaling table (02) so
            # the classification r matches the reported b/r exactly (a plain
            # corrcoef here diverged once NWIS data introduced a few outliers).
            j = clip_event(load_timeseries(bands[b]))
            if len(j) > 80:
                lo, hi = b.split("-")
                fit = fit_scaling(j, sid, (float(lo), float(hi)))
                return b, float(fit.r), float(fit.b_ols), int(fit.n)
    return None, np.nan, np.nan, 0


def main() -> int:
    coords = {f'CC.{r["sta"]}': dict(lon=r["lon"], lat=r["lat"], sr=r["sr"], basin=r.get("basin", ""))
              for r in json.loads((ROOT / "config" / "cc_stations.json").read_text())}
    coords.update({f'UW.{r["sta"]}': dict(lon=r["lon"], lat=r["lat"], sr=r["sr"], basin="")
                   for r in json.loads((ROOT / "config" / "uw_stations.json").read_text())})
    sids = sorted({FN.match(f.name)["sid"] for f in RESULTS.glob("*_*Hz_timeseries.csv") if FN.match(f.name)})
    out = []
    for sid in sids:
        band, r, b, n = best_r(sid)
        if sid in CONTROL:
            status = "control"
        elif not np.isfinite(r):
            status = "none"
        elif r >= 0.70:
            status = "observed"
        elif r >= 0.50:
            status = "marginal"
        else:
            status = "none"
        c = coords.get(sid, {})
        out.append(dict(station=sid, status=status, flow_band=band, r=None if not np.isfinite(r) else round(r, 2),
                        b=None if not np.isfinite(b) else round(b, 2), n=n,
                        lon=c.get("lon"), lat=c.get("lat"), sr=c.get("sr"), basin=c.get("basin")))
    out.sort(key=lambda d: ({"observed": 0, "marginal": 1, "none": 2, "control": 3}[d["status"]], d["station"]))
    (ROOT / "config" / "station_status.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(out)[["station", "status", "flow_band", "r", "b", "sr", "basin"]].to_string(index=False))
    print(f"\nwrote config/station_status.json ({len(out)} stations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
