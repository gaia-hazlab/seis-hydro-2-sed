#!/usr/bin/env python3
"""Time-resolved seismic-discharge scaling exponent b(t) per station.

Closer to the physical model than a raw power ratio: in a sliding window we fit
P ∝ Q^b (log-log) for the bedload band (5-15 Hz). When b rises above the
turbulent-flow baseline (~0.9-1.4; Gimbert 2014) the band is bedload-dominated.
b is only defined where the window spans enough discharge range. Colorblind-safe
(Okabe-Ito) palette. Outputs fig8_b_of_time.png.

Usage: pixi run python workflows/08_b_of_time.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"

import sys; sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402
paper_style()

EXCLUDE = {"UW.BHW", "UW.TEHA"}
BASELINE = (0.9, 1.4)
WIN = pd.Timedelta("24h")
STEP = pd.Timedelta("3h")
MIN_N, MIN_QSPAN, MIN_R = 24, 0.35, 0.5
HF_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)_5\.0-15\.0Hz_timeseries\.csv$")
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
AR_COLORS = {"pre-AR": "#999999", "AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight"})


def b_of_time(j: pd.DataFrame) -> pd.Series:
    lp = np.log10(j["P"].clip(lower=1e-30))
    lq = np.log10(j["Q"].clip(lower=1e-6))
    t0, t1 = j.index.min(), j.index.max()
    centers = pd.date_range(t0 + WIN / 2, t1 - WIN / 2, freq=STEP)
    out = {}
    for c in centers:
        m = (j.index >= c - WIN / 2) & (j.index < c + WIN / 2)
        x, y = lq[m].values, lp[m].values
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < MIN_N or (x.max() - x.min()) < MIN_QSPAN:
            continue
        if abs(np.corrcoef(x, y)[0, 1]) < MIN_R:
            continue
        out[c] = float(np.polyfit(x, y, 1)[0])
    return pd.Series(out)


def main() -> int:
    series, qref = {}, None
    for f in sorted(RESULTS.glob("*_5.0-15.0Hz_timeseries.csv")):
        m = HF_RE.match(f.name)
        sid = f'{m["net"]}.{m["sta"]}' if m else None
        if not sid or sid in EXCLUDE:
            continue
        df = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc")
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()]
        P = pd.to_numeric(df["proxy"], errors="coerce")
        Q = pd.to_numeric(df["gauge"], errors="coerce")
        j = pd.concat([P.rename("P"), Q.rename("Q")], axis=1).sort_index()
        j["Q"] = j["Q"].interpolate("linear", limit=12)
        j = j.dropna()
        series[sid] = b_of_time(j)
        if qref is None:
            qref = j["Q"].resample("1h").median().dropna()
    series = {s: v for s, v in series.items() if len(v)}
    if not series:
        print("no b(t) computed"); return 1

    ars = json.loads((ROOT / "config" / "ar_windows.json").read_text())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 6), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 2]))
    a1.plot(qref.index, qref.values, color="k", lw=1.1)
    a1.set_ylabel("discharge\n(m³ s⁻¹)")
    for w in ars:
        s0, s1 = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        for a in (a1, a2):
            a.axvspan(s0, s1, color=AR_COLORS.get(w["label"], "#999"), alpha=0.16, zorder=0)
        a1.text(pd.Timestamp(w["peak"]), a1.get_ylim()[1] * 0.9, w["label"], ha="center",
                fontsize=8.5, fontweight="bold")

    a2.axhspan(*BASELINE, color="0.6", alpha=0.18, zorder=1,
               label=f"turbulence baseline ({BASELINE[0]}–{BASELINE[1]})")
    a2.axhline(1.0, color="0.5", ls=":", lw=1)
    for i, sid in enumerate(sorted(series)):
        v = series[sid]
        a2.plot(v.index, v.values, marker="o", ms=2.5, lw=1.3, color=OKABE[i % len(OKABE)], label=sid)
    a2.set_ylabel(r"scaling exponent $b(t)$  ($P_{5\text{–}15\,\mathrm{Hz}} \propto Q^{\,b}$)")
    a2.set_xlabel("December 2025 (UTC)")
    a2.set_ylim(-0.5, 3.0)
    a2.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    # (no figure title — described by the manuscript caption)
    fig.autofmt_xdate()
    fig.savefig(FIGDIR / "fig8_b_of_time.png", bbox_inches="tight")
    plt.close(fig)
    print("b(t) points per station:", {s: len(v) for s, v in series.items()})
    print(f"wrote {FIGDIR}/fig8_b_of_time.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
