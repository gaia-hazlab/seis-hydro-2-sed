#!/usr/bin/env python3
"""Seismic 'virtual discharge': calibrate P = C Q^b per river-proximal station,
invert to Q_seis, validate against the co-located gage, and test whether the
turbulence exponent b is stable (good rating) or varies in time/space (science).

Outputs config/virtual_q.json (per-station a,b,r, b(t) stability, NSE) and
fig12_virtual_q.png (virtual vs real discharge).

Usage: pixi run python workflows/12_virtual_q.py
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
EXCLUDE = {"UW.BHW", "UW.TEHA"}
BAND_RE = re.compile(r"^(?P<sid>[A-Z0-9]+\.[A-Z0-9]+)_5\.0-15\.0Hz_timeseries\.csv$")
WIN, STEP = pd.Timedelta("24h"), pd.Timedelta("6h")
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000", "#999999"]


def load(path):
    df = pd.read_csv(path, parse_dates=["time_utc"]).set_index("time_utc")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    P = pd.to_numeric(df["proxy"], errors="coerce")
    Q = pd.to_numeric(df["gauge"], errors="coerce")
    j = pd.concat([P.rename("P"), Q.rename("Q")], axis=1).sort_index()
    j["Q"] = j["Q"].interpolate("linear", limit=12)
    return j.dropna()


def b_of_time(lq, lp):
    centers = pd.date_range(lq.index.min() + WIN / 2, lq.index.max() - WIN / 2, freq=STEP)
    out = {}
    for c in centers:
        m = (lq.index >= c - WIN / 2) & (lq.index < c + WIN / 2)
        x, y = lq[m].values, lp[m].values
        if len(x) < 24 or (x.max() - x.min()) < 0.35 or abs(np.corrcoef(x, y)[0, 1]) < 0.5:
            continue
        out[c] = np.polyfit(x, y, 1)[0]
    return pd.Series(out)


def main() -> int:
    items = {}
    for f in sorted(RESULTS.glob("*_5.0-15.0Hz_timeseries.csv")):
        m = BAND_RE.match(f.name)
        if m and m["sid"] not in EXCLUDE:
            items[m["sid"]] = f
    rows, virt = [], {}
    for sid, f in items.items():
        j = load(f)
        if len(j) < 100:
            continue
        lq = np.log10(j["Q"].clip(lower=1e-6)); lp = np.log10(j["P"].clip(lower=1e-30))
        b, a = np.polyfit(lq.values, lp.values, 1)          # log10 P = a + b log10 Q
        r = float(np.corrcoef(lq, lp)[0, 1])
        q_seis = 10 ** ((lp - a) / b)                       # invert -> virtual discharge
        # Nash–Sutcliffe of virtual vs real (log space, robust)
        resid = (np.log10(q_seis.clip(lower=1e-6)) - lq)
        nse = 1 - float(np.nansum(resid ** 2) / np.nansum((lq - lq.mean()) ** 2))
        bt = b_of_time(lq, lp)
        rows.append(dict(station=sid, a=round(float(a), 3), b=round(float(b), 3), r=round(r, 3),
                         nse_logQ=round(nse, 3), b_t_mean=round(float(bt.mean()), 3) if len(bt) else None,
                         b_t_std=round(float(bt.std()), 3) if len(bt) else None, n=len(j)))
        virt[sid] = dict(time=[t.isoformat() for t in q_seis.index],
                         q_seis=[round(float(x), 2) for x in q_seis.values], a=float(a), b=float(b))
    (ROOT / "config" / "virtual_q.json").write_text(json.dumps(virt))
    tbl = pd.DataFrame(rows).sort_values("b", ascending=False)
    print(tbl.to_string(index=False))
    (ROOT / "config" / "virtual_q_fit.json").write_text(json.dumps(rows, indent=2))

    # validation figure: virtual vs real discharge per station
    n = len(items); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow), squeeze=False, sharex=True)
    for ax, (sid, f) in zip(axes.ravel(), items.items()):
        j = load(f); v = virt[sid]
        qs = pd.Series(v["q_seis"], index=pd.to_datetime(v["time"], utc=True))
        ax.plot(j.index, j["Q"], color="k", lw=1.3, label="gage Q")
        ax.plot(qs.index, qs.values, color="#D55E00", lw=1.0, alpha=0.8, label="seismic virtual Q")
        rr = next(x for x in rows if x["station"] == sid)
        ax.set_title(f"{sid}  b={rr['b']:.2f} r={rr['r']:.2f}", fontsize=8)
        ax.set_yscale("log")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes.ravel()[0].legend(fontsize=6.5, loc="upper left")
    fig.supylabel("discharge (m³ s⁻¹)"); fig.supxlabel("December 2025 (UTC)")
    fig.suptitle("Seismic virtual discharge vs co-located gage (turbulence-band rating P∝Q$^b$)")
    fig.tight_layout(); fig.autofmt_xdate()
    fig.savefig(FIGDIR / "fig12_virtual_q.png", dpi=200)
    plt.close(fig)
    print(f"\nwrote {FIGDIR}/fig12_virtual_q.png + config/virtual_q.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
