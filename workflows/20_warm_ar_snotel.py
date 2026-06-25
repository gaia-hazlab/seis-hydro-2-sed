#!/usr/bin/env python3
"""Warm-AR vs cold-late-December contrast from high-elevation SNOTEL.

The December-2025 floods were driven by *warm* atmospheric rivers with a high
freezing level (rain-on-snow), whereas the later-December storms were colder and
fell largely as snow — so they did not produce the same runoff/seismic response.
This script makes that mechanism explicit using NRCS SNOTEL stations near the
*tops* of the study drainages:

  - Paradise (679:WA:SNTL, ~1563 m) — head of the Nisqually
  - Mowich   (941:WA:SNTL, ~?    m) — NW Puyallup/Carbon headwaters

It pulls hourly **air temperature** (TOBS), **snow-water-equivalent** (WTEQ) and
**snow depth** (SNWD), and plots:
  (a) air temperature with the 0 °C freezing line and the AR windows — warm ARs
      sit near/above freezing at altitude (rain-on-snow), late December drops well
      below freezing (snow);
  (b) SWE / snow depth — SWE is drawn down during the warm ARs (melt + rain pass-
      through) and *accumulates* during the cold late-December storms.

Outputs paper/figures/fig21_warm_ar_snow.png and config/warm_ar_snotel.json.

Usage: pixi run python workflows/20_warm_ar_snotel.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"
CFG = ROOT / "config"
import sys; sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402
paper_style()

START, END = "2025-12-01", "2026-01-01"
AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
# High stations near the tops of the study drainages (lon, lat, elev_m)
STATIONS = {
    "679:WA:SNTL": ("Paradise", 1563),
    "941:WA:SNTL": ("Mowich", 1295),
}
F_TO_C = lambda f: (f - 32.0) * 5.0 / 9.0
IN_TO_CM = 2.54


def fetch_element(trip: str, element: str, duration: str = "HOURLY") -> pd.Series:
    r = requests.get(AWDB, params={
        "stationTriplets": trip, "elements": element, "duration": duration,
        "beginDate": START, "endDate": END,
        "centralTendencyType": "NONE", "returnFlags": "false"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    vals = js[0]["data"][0]["values"]
    s = pd.Series({v["date"]: v.get("value") for v in vals}, dtype="float64").dropna()
    s.index = pd.to_datetime(s.index, utc=True)
    return s.sort_index()


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    ar = []
    arf = CFG / "ar_windows.json"
    if arf.exists():
        ar = json.loads(arf.read_text())

    data = {}
    for trip, (name, elev) in STATIONS.items():
        rec = {"elev_m": elev}
        for el, key in (("TOBS", "tempC"), ("WTEQ", "swe_cm"), ("SNWD", "snowdepth_cm")):
            try:
                s = fetch_element(trip, el)
                if el == "TOBS":
                    s = F_TO_C(s)
                else:
                    s = s * IN_TO_CM
                rec[key] = s
                print(f"  {name} {el}: {len(s)} values "
                      f"[{np.nanmin(s.values):.1f}, {np.nanmax(s.values):.1f}]")
            except Exception as e:  # noqa: BLE001
                print(f"  {name} {el} failed: {e}")
        data[name] = rec

    # ---- figure ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 1], hspace=0.08))
    colors = {"Paradise": "#d73027", "Mowich": "#4575b4"}
    for (s0, pk, s1, lab) in [(w["start"], w["peak"], w["end"], w["label"]) for w in ar]:
        for a in (a1, a2):
            a.axvspan(pd.Timestamp(s0), pd.Timestamp(s1), color="#cfe3f2", alpha=0.45, zorder=0)
        a1.text(pd.Timestamp(pk), 0.96, lab, transform=a1.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, fontweight="bold", color="#2166ac")

    # (a) air temperature with 0 °C freezing line
    a1.axhline(0.0, color="0.4", lw=1.2, ls="--")
    a1.text(pd.Timestamp(START), 0.3, " freezing level (0 °C)", color="0.4", fontsize=9)
    for name, rec in data.items():
        if "tempC" in rec:
            a1.plot(rec["tempC"].index, rec["tempC"].values, lw=1.6,
                    color=colors.get(name), label=f"{name} ({rec['elev_m']} m)")
    a1.set_ylabel("air temperature (°C)")
    a1.legend(loc="upper right", fontsize=10)

    # (b) SWE (solid) — accumulation vs ablation
    for name, rec in data.items():
        if "swe_cm" in rec:
            a2.plot(rec["swe_cm"].index, rec["swe_cm"].values, lw=2.0,
                    color=colors.get(name), label=f"{name} SWE")
        if "snowdepth_cm" in rec:
            a2.plot(rec["snowdepth_cm"].index, rec["snowdepth_cm"].values, lw=1.0, ls=":",
                    color=colors.get(name), alpha=0.7, label=f"{name} snow depth")
    a2.set_ylabel("snow water equiv. /\ndepth (cm)")
    a2.set_xlabel("December 2025 (UTC)")
    a2.legend(loc="upper left", fontsize=9, ncol=2)

    fig.suptitle("Warm rain-on-snow ARs drive the floods; the cold late-December "
                 "rain→snow transition shuts off\nrunoff and sediment supply — the "
                 "supply shutoff that quiets the post-event matched-Q baseline (fig22)",
                 fontsize=11)
    fig.autofmt_xdate()
    out = FIGDIR / "fig21_warm_ar_snow.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # ---- persist (JSON-serialisable) ----
    ser = {n: {k: ({"time": [t.isoformat() for t in v.index],
                    "value": [round(float(x), 2) for x in v.values]}
                   if isinstance(v, pd.Series) else v)
               for k, v in rec.items()} for n, rec in data.items()}
    (CFG / "warm_ar_snotel.json").write_text(json.dumps(ser, indent=2))
    print(f"wrote {CFG/'warm_ar_snotel.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
