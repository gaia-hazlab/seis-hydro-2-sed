#!/usr/bin/env python3
"""Stage–discharge rating curves: how discharge Q relates to gage height (stage) h.

USGS gages measure stage continuously and convert to discharge via a site-specific
rating Q = C (h - h0)^β (h0 = stage of zero flow), calibrated by direct
measurements. This matters here because flood/inundation hazard is set by *stage*,
while the seismic proxy estimates *discharge*: Q_seis → h via the rating gives a
seismic stage nowcast. We fit the rating from December-2025 instantaneous stage +
discharge at the corridor gages.

Outputs fig15_rating.png and config/rating_fits.json.
Usage: pixi run python workflows/17_rating.py
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
CFS, FT = 0.0283168466, 0.3048
GAGES = [("12092000", "Puyallup nr Electron", "#0072B2"),
         ("12093500", "Puyallup nr Orting", "#009E73"),
         ("12101500", "Puyallup at Puyallup", "#D55E00"),
         ("12082500", "Nisqually nr National", "#CC79A7")]


def fetch(gid):
    r = requests.get("https://waterservices.usgs.gov/nwis/iv/", params={
        "format": "json", "sites": gid, "startDT": "2025-12-01", "endDT": "2025-12-24",
        "parameterCd": "00060,00065"}, timeout=60)
    d = {}
    for t in r.json()["value"]["timeSeries"]:
        code = t["variable"]["variableCode"][0]["value"]
        v = t["values"][0]["value"]
        s = pd.Series({x["dateTime"]: float(x["value"]) for x in v if x["value"] not in ("", "-999999")})
        s.index = pd.to_datetime(s.index); d[code] = s
    if "00060" not in d or "00065" not in d:
        return None
    j = pd.concat([d["00060"].rename("Q"), d["00065"].rename("h")], axis=1).dropna()
    return j["Q"] * CFS, j["h"] * FT


def fit_rating(Q, h):
    best = None
    for h0 in np.linspace(h.min() - 3, h.min() - 0.05, 80):
        x = h - h0
        if x.min() <= 0:
            continue
        b, a = np.polyfit(np.log10(x), np.log10(Q), 1)
        res = float(np.sum((np.log10(Q) - (a + b * np.log10(x))) ** 2))
        if best is None or res < best[0]:
            best = (res, float(h0), float(b), float(10 ** a))
    return best[1], best[2], best[3]   # h0, beta, C


def main() -> int:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fits = []
    for gid, nm, c in GAGES:
        r = fetch(gid)
        if r is None:
            print(f"{gid}: no dual stage+Q"); continue
        Q, h = r
        h0, beta, C = fit_rating(Q, h)
        fits.append(dict(gage=gid, name=nm, h0=round(h0, 2), beta=round(beta, 2), C=round(C, 2),
                         h_min=round(float(h.min()), 2), h_max=round(float(h.max()), 2),
                         Q_max=round(float(Q.max()), 0)))
        ax.scatter(h, Q, s=5, alpha=0.25, color=c)
        hs = np.linspace(h.min(), h.max(), 100)
        ax.plot(hs, C * np.maximum(hs - h0, 1e-6) ** beta, color=c, lw=2,
                label=f"{nm}: Q={C:.2g}(h−{h0:.1f})$^{{{beta:.1f}}}$")
    ax.set_yscale("log")
    ax.set_xlabel("gage height / stage  h  (m)")
    ax.set_ylabel("discharge  Q  (m³ s⁻¹)")
    ax.set_title("Stage–discharge rating  Q = C (h − h₀)$^β$  (Dec 2025)", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig15_rating.png", dpi=200)
    plt.close(fig)
    (ROOT / "config" / "rating_fits.json").write_text(json.dumps(fits, indent=2))
    print(pd.DataFrame(fits).to_string(index=False))
    print(f"\nwrote {FIGDIR}/fig15_rating.png + config/rating_fits.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
