#!/usr/bin/env python3
r"""Linear-axis stage–discharge ratings with time coloring — the hysteresis companion.

Companion to workflows/25_rating_geometry.py, which shows the ratings on log–log
axes (where the slope is the local exponent $\beta$). Co-authors more used to seeing
a *linear* stage–discharge plot asked what that looks like, and the linear view also
makes **hysteresis** legible: within a flood pulse the rising limb and the falling
limb trace different stage–discharge paths, so the time-ordered trace opens into a
loop rather than collapsing onto a single curve.

Two physically distinct effects produce that departure, and only one is a bed-change
diagnostic:

  1. *Unsteady-flow (looped-rating / Jones) effect* — purely hydraulic: the
     flood-wave water-surface slope is steeper on the rising limb, so a given stage
     conveys more discharge while the wave passes (equivalently, lower stage at a
     given $Q$ on the rising limb). It is dominantly hydraulic, is expected on any
     flashy river, and produces a loop of a *fixed sense* — the same sense a sediment
     scour-and-fill cycle would, so the within-event loop alone is not a bed signal.
  2. *Net rating shift between/through events* — the stage–discharge control changes.
     A net **rise** in stage at a fixed discharge means the channel conveys the same
     flow at a higher water level, *consistent with* net **aggradation** (deposition).
     A net **fall** is consistent with net **degradation** (bed lowering). Stage–
     discharge data alone cannot prove a bed cause: increased roughness, changing
     backwater (bars, wood/debris jams), control-section reorganization, or USGS
     rating-shift artifacts produce the same drift. The aggradation reading rests on
     convergence with independent evidence (satellite channel change; the basin's
     supply-rich aggradational regime), not on this figure alone.

We therefore (a) draw the time-coloured trajectory (dark = early Dec → light = late
Dec) with direction arrows, and (b) quantify, per gage at elevated flow ($Q>$ median),
the within-event loop amplitude (rising − falling stage residual at matched $Q$) and
the net cross-event drift (Spearman $\rho$ of the stage residual with time, and the
late−early median shift in dex).

We find: the confined source gages (Electron, National) are essentially single-valued
(stable hydraulic control); the **alluvial mid Puyallup-nr-Orting and lowland
Puyallup-at-Puyallup gages drift to higher stage at fixed $Q$ through the AR sequence
($\rho\approx0.7$) — consistent with net aggradation** in the unconfined, depositional
reaches. This is the time-domain counterpart to the reach-geometry-dependent,
event-specific ratings documented for the braided reaches (main text §sec-braided,
§sec-reorg).

Outputs paper/figures/fig29_rating_hysteresis_linear.png + config/rating_hysteresis.json.

Usage: pixi run python workflows/29_rating_hysteresis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

DATA = ROOT / "notebooks" / "data"
CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"
# gage, label, colour (kept consistent with workflows/25_rating_geometry.py)
GAGES = [
    ("12092000", "Puyallup nr Electron (confined source)", "#e31a1c"),
    ("12082500", "Nisqually nr National (source)", "#1f78b4"),
    ("12093500", "Puyallup nr Orting (mid)", "#33a02c"),
    ("12101500", "Puyallup at Puyallup (lowland, overbank)", "#6a3d9a"),
]


def load_rating_timed(site: str):
    """Return time-ordered (t, Q, h-h0) — keep chronological order for hysteresis."""
    f = DATA / f"usgs_iv_{site}_2025-12-01_2026-01-01.csv"
    d = pd.read_csv(f)
    t = pd.to_datetime(d["time_utc"], utc=True, errors="coerce")
    Q = pd.to_numeric(d["discharge_cfs"], errors="coerce") * 0.0283168
    H = pd.to_numeric(d["gage_height_ft"], errors="coerce") * 0.3048
    m = t.notna() & Q.notna() & H.notna() & (Q > 0)
    t, Q, H = t[m], Q[m].values, H[m].values
    order = np.argsort(t.values)
    t = t.values[order]
    Q, H = Q[order], H[order]
    h0 = H.min() - 0.01
    return t, Q, H - h0


def hysteresis_diag(t, Q, dh):
    """Loop amplitude (rising−falling stage at matched Q) and net cross-event drift.

    Residuals are taken about a single-power-law reference rating; because rising vs
    falling and early vs late are compared at the *same* discharge range, the choice
    of reference does not bias the loop/drift signs. Restricted to elevated flow
    (Q > median), where the loops live.
    """
    b = np.polyfit(np.log10(Q), np.log10(dh), 1)
    resid = np.log10(dh) - np.polyval(b, np.log10(Q))   # + = stage above the mean rating
    tsec = (t - t[0]).astype("timedelta64[s]").astype(float)
    Qs = pd.Series(Q).rolling(9, center=True, min_periods=1).mean().values
    dQdt = np.gradient(Qs, tsec)
    hi = Q > np.median(Q)
    rise = (dQdt > 0) & hi
    fall = (dQdt < 0) & hi
    loop = float(np.median(resid[rise]) - np.median(resid[fall]))   # within-event loop
    rho = float(spearmanr(tsec[hi], resid[hi]).statistic)           # net drift vs time
    half = t[len(t) // 2]
    drift = float(np.median(resid[(t >= half) & hi]) - np.median(resid[(t < half) & hi]))
    if rho > 0.4 and drift > 0.03:
        verdict = "drift up (~aggradation)"
    elif rho < -0.4 and drift < -0.03:
        verdict = "drift down (~degradation)"
    else:
        verdict = "stable (single-valued)"
    return dict(loop_dex=round(loop, 3), drift_dex=round(drift, 3),
                rho_t=round(rho, 2), verdict=verdict)


def add_direction_arrows(ax, x, y, color, n=6):
    """Drop a few arrowheads along the time-ordered path to show loop direction."""
    idx = (np.linspace(0.04, 0.96, n) * (len(x) - 1)).astype(int)
    for i in idx:
        ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0,
                                    mutation_scale=16, alpha=0.9), zorder=5)


def main() -> int:
    paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.8), constrained_layout=True)
    axes = axes.ravel()

    # one shared time axis (numeric days) so a single colorbar spans every panel
    t_all = [load_rating_timed(site)[0] for site, *_ in GAGES]
    tnum0 = mdates.date2num(min(tt.min() for tt in t_all))
    tnum1 = mdates.date2num(max(tt.max() for tt in t_all))
    cmap = plt.cm.viridis

    out, sm = {}, None
    for ax, (site, label, col) in zip(axes, GAGES):
        t, Q, dh = load_rating_timed(site)
        tn = mdates.date2num(t)
        pts = np.column_stack([Q, dh]).reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, alpha=0.85, linewidths=1.4)
        lc.set_array(tn[:-1]); lc.set_clim(tnum0, tnum1)
        ax.add_collection(lc)
        ax.scatter(Q, dh, c=tn, cmap=cmap, vmin=tnum0, vmax=tnum1,
                   s=4, alpha=0.5, zorder=3, rasterized=True)
        add_direction_arrows(ax, Q, dh, col)
        sm = lc

        diag = hysteresis_diag(t, Q, dh)
        out[site] = dict(name=label, **diag)
        ax.text(0.035, 0.96,
                f"loop {diag['loop_dex']:+.2f} dex\n"
                f"drift {diag['drift_dex']:+.2f} dex ($\\rho_t$={diag['rho_t']:+.2f})\n"
                f"→ {diag['verdict']}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=col, alpha=0.92))

        ax.set_xlim(0, Q.max() * 1.04)
        ax.set_ylim(0, dh.max() * 1.08)
        ax.set_title(label, fontsize=11, color=col)
        ax.set_xlabel("discharge $Q$  (m³ s⁻¹)")
        ax.set_ylabel("stage above zero-flow $h-h_0$  (m)")

    cbar = fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("measurement time (UTC)")
    cbar.ax.yaxis.set_major_locator(mdates.DayLocator(interval=5))
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.suptitle("Stage–discharge ratings (linear axes) coloured by time: confined source gages stay "
                 "single-valued;\nalluvial mid/lowland gages drift to higher stage at fixed $Q$ — consistent with aggradation",
                 fontsize=12.5)
    out_png = FIGDIR / "fig29_rating_hysteresis_linear.png"
    fig.savefig(out_png)
    plt.close(fig)
    (CONFIG / "rating_hysteresis.json").write_text(json.dumps(out, indent=2))
    for site, d in out.items():
        print(f"{d['name']:42s} loop={d['loop_dex']:+.3f} drift={d['drift_dex']:+.3f} "
              f"rho_t={d['rho_t']:+.2f}  -> {d['verdict']}")
    print(f"wrote {out_png} + config/rating_hysteresis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
