#!/usr/bin/env python3
"""Early-warning framing: upstream signals lead the downstream flood peak.

(a) Mainstem discharge at Electron (RM~41) -> Orting (RM~30) -> Puyallup (RM~10):
    the peak arrives progressively later downstream (flood routing + downstream
    basin contribution), giving tens of hours of lead.
(b) Upstream seismic power (CC.PR03, 5-15 Hz) vs downstream stage (Puyallup at
    Puyallup): the upstream seismic peak leads the downstream discharge peak.

Outputs fig10_early_warning.png. Usage: pixi run python workflows/10_early_warning.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight"})


def gser(aux, gid):
    d = aux["discharge"][gid]
    return pd.Series(d["q_cms"], index=pd.to_datetime(d["time"], utc=True)).sort_index()


def main() -> int:
    aux = json.loads((ROOT / "config" / "aux_timeseries.json").read_text())
    mainstem = [("12092000", "Electron (RM~41)"), ("12093500", "Orting (RM~30)"),
                ("12101500", "Puyallup (RM~10)")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2))

    # Panel A: downstream propagation of the peak
    peaks = {}
    for i, (gid, nm) in enumerate(mainstem):
        s = gser(aux, gid)
        a1.plot(s.index, s.values, color=OKABE[i], lw=1.5, label=nm)
        tpk = s.rolling(3, center=True, min_periods=1).mean().idxmax()
        peaks[nm] = (tpk, s.max())
        a1.scatter([tpk], [s.max()], color=OKABE[i], s=40, zorder=5)
    t0 = peaks["Electron (RM~41)"][0]
    lead = (peaks["Puyallup (RM~10)"][0] - t0).total_seconds() / 3600
    a1.set_ylabel("discharge (m³ s⁻¹)")
    a1.set_title(f"Flood peak propagates downstream — Puyallup peak lags Electron by ~{lead:.0f} h", loc="left")
    a1.legend(fontsize=8, loc="upper left")

    # Panel B: upstream seismic leads downstream stage
    pr = pd.read_csv(ROOT / "notebooks/data/results/CC.PR03_5.0-15.0Hz_timeseries.csv",
                     parse_dates=["time_utc"]).set_index("time_utc")
    P = pd.to_numeric(pr["proxy"], errors="coerce").rolling(12, center=True, min_periods=1).median()
    Pn = (P / P.median())
    qd = gser(aux, "12101500")
    a2b = a2.twinx(); a2b.grid(False)
    h1, = a2.semilogy(Pn.index, Pn.values, color="#D55E00", lw=1.0, label="CC.PR03 seismic 5–15 Hz (upstream)")
    h2, = a2b.plot(qd.index, qd.values, color="k", lw=1.6, ls="--", label="Puyallup discharge (downstream)")
    sp = P["2025-12-09":"2025-12-12"].idxmax()
    qp = qd["2025-12-10":"2025-12-13"].idxmax()
    a2.axvline(sp, color="#D55E00", lw=1, ls=":"); a2b.axvline(qp, color="k", lw=1, ls=":")
    slead = (qp - sp).total_seconds() / 3600
    a2.set_ylabel("seismic power / median"); a2b.set_ylabel("discharge (m³ s⁻¹)")
    a2.set_xlabel("December 2025 (UTC)")
    a2.set_title(f"Upstream seismic peak leads downstream stage by ~{slead:.0f} h", loc="left")
    a2.legend(handles=[h1, h2], fontsize=8, loc="upper left")
    a2.set_xlim(pd.Timestamp("2025-12-05", tz="UTC"), pd.Timestamp("2025-12-15", tz="UTC"))
    a1.set_xlim(*a2.get_xlim())
    fig.autofmt_xdate()
    fig.savefig(FIGDIR / "fig10_early_warning.png")
    plt.close(fig)
    print(f"downstream peak lead ~{lead:.0f} h; seismic lead ~{slead:.0f} h")
    print(f"wrote {FIGDIR}/fig10_early_warning.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
