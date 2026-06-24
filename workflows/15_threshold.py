#!/usr/bin/env python3
"""Estimate a threshold discharge Qc for transport onset from seismic power.

Bedload is threshold-gated: below Qc the seismic power tracks turbulent flow
(slope b1 ≈ 1), above Qc an added, steeper contribution can switch on (slope b2 >
b1). We fit a continuous broken-stick (segmented) regression of log10 P vs
log10 Q per station and take the breakpoint as a candidate Qc, following the
seismic transport-onset literature (Burtin 2008; Roth 2016/2017). Physical anchor:
the slope-dependent critical Shields stress τ*c = 0.15 S^0.25 (Lamb et al. 2008)
implies high Qc on steep, coarse Rainier reaches.

Outputs config/threshold_qc.json and fig14_threshold.png.
Usage: pixi run python workflows/15_threshold.py
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

import sys; sys.path.insert(0, str(ROOT / "src"))
from riverseis.analysis import clip_event  # noqa: E402
FIGDIR = ROOT / "paper" / "figures"
EXCLUDE = {"UW.BHW", "UW.TEHA"}
RE = re.compile(r"^(?P<sid>[A-Z0-9]+\.[A-Z0-9]+)_5\.0-15\.0Hz_timeseries\.csv$")


def load(path):
    df = pd.read_csv(path, parse_dates=["time_utc"]).set_index("time_utc")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    P = pd.to_numeric(df["proxy"], errors="coerce"); Q = pd.to_numeric(df["gauge"], errors="coerce")
    j = pd.concat([P.rename("P"), Q.rename("Q")], axis=1).sort_index()
    j["Q"] = j["Q"].interpolate("linear", limit=12)
    j = clip_event(j.dropna())   # breakpoint fit on the flood window
    lp = np.log10(j["P"].clip(lower=1e-30)); lq = np.log10(j["Q"].clip(lower=1e-6))
    med = lp.median(); mad = 1.4826 * (lp - med).abs().median()
    k = (lp - med).abs() < 6 * max(mad, 1e-9)
    return lq[k].values, lp[k].values


def broken_stick(x, y):
    """Continuous two-segment fit; return (x0, b1, b2, a, sse, sse_lin)."""
    order = np.argsort(x); x, y = x[order], y[order]
    lo, hi = np.quantile(x, 0.15), np.quantile(x, 0.85)
    cands = np.linspace(lo, hi, 40)
    best = None
    for x0 in cands:
        X = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - x0)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((y - X @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, x0, coef)
    sse, x0, coef = best
    a, b1, dslope = coef
    b1lin, _ = np.polyfit(x, y, 1), None
    sse_lin = float(np.sum((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2))
    return x0, float(b1), float(b1 + dslope), float(a), sse, sse_lin


def main() -> int:
    items = {}
    for f in sorted(RESULTS.glob("*_5.0-15.0Hz_timeseries.csv")):
        m = RE.match(f.name)
        if m and m["sid"] not in EXCLUDE:
            items[m["sid"]] = f
    rows = []
    n = len(items); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.7 * nrow), squeeze=False, sharex=True, sharey=True)
    for ax, (sid, f) in zip(axes.ravel(), items.items()):
        x, y = load(f)
        if len(x) < 80:
            ax.set_visible(False); continue
        x0, b1, b2, a, sse, sse_lin = broken_stick(x, y)
        Qc = 10 ** x0
        improve = 1 - sse / sse_lin                          # variance reduction vs single line
        # Model comparison: single line (k=2: a,b) vs continuous broken-stick
        # (k=4: a,b1,Δslope,x0). Consecutive 10-min flood samples are strongly
        # autocorrelated, so the raw n hugely overstates independent information
        # and would make every break "significant". Use an effective sample size
        # from the lag-1 autocorrelation of the (time-ordered) broken-stick
        # residuals: n_eff = n·(1−ρ)/(1+ρ). ΔBIC>10 then = very strong evidence
        # for two segments (Kass & Raftery 1995).
        nx = len(x)
        yhat = a + b1 * x + (b2 - b1) * np.maximum(0.0, x - x0)   # x,y are time-ordered
        resid = y - yhat
        rho = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if nx > 3 else 0.0
        rho = min(max(rho, 0.0), 0.999)
        n_eff = max(5.0, nx * (1.0 - rho) / (1.0 + rho))
        bic_lin = n_eff * np.log(sse_lin / nx) + 2 * np.log(n_eff)
        bic_bs = n_eff * np.log(sse / nx) + 4 * np.log(n_eff)
        aic_lin = n_eff * np.log(sse_lin / nx) + 2 * 2
        aic_bs = n_eff * np.log(sse / nx) + 2 * 4
        dBIC = bic_lin - bic_bs
        dAIC = aic_lin - aic_bs
        significant = bool(dBIC > 10 and abs(b2 - b1) > 0.3)
        direction = "steepening" if b2 > b1 + 0.3 else ("flattening" if b2 < b1 - 0.3 else "none")
        steepens = significant and direction == "steepening"
        rows.append(dict(station=sid, Qc_cms=round(float(Qc), 1), b_below=round(b1, 2),
                         b_above=round(b2, 2), dslope=round(b2 - b1, 2),
                         var_reduction=round(improve, 3), dBIC=round(float(dBIC), 1),
                         dAIC=round(float(dAIC), 1), n_eff=round(float(n_eff), 0),
                         rho=round(float(rho), 3), significant_break=significant,
                         direction=direction, onset=bool(steepens)))
        ax.scatter(x, y, s=3, alpha=0.25, color="0.5")
        xs = np.linspace(x.min(), x.max(), 100)
        yhat = a + b1 * xs + (b2 - b1) * np.maximum(0, xs - x0)
        ax.plot(xs, yhat, "r-", lw=1.8)
        if significant:
            ax.axvline(x0, color="#0072B2", ls="--", lw=1.2)
            ax.text(x0, ax.get_ylim()[1], f" Qc≈{Qc:.0f}", color="#0072B2", fontsize=7, va="top")
        tag = f"  ΔBIC={dBIC:.0f}" if significant else "  (no break)"
        ax.set_title(f"{sid}  b:{b1:.2f}→{b2:.2f}{tag}", fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.supxlabel(r"$\log_{10} Q$ (m³ s⁻¹)"); fig.supylabel(r"$\log_{10} P$ (5–15 Hz)")
    # (no figure suptitle — described by the manuscript caption; per-panel b:b1→b2 kept)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig14_threshold.png", dpi=200)
    plt.close(fig)
    (ROOT / "config" / "threshold_qc.json").write_text(json.dumps(rows, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {FIGDIR}/fig14_threshold.png + config/threshold_qc.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
