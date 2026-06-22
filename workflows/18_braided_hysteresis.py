#!/usr/bin/env python3
"""Geometric vs. transport onset in the braided Puyallup reach (PR01/PR02/PR03).

In a braided reach the seismic source is a *non-stationary, spatially distributed*
set of active anabranches rather than a single line channel (Coppin & Burtin 2022;
Burtin et al. 2011). Two onset mechanisms can both produce a steepening P–Q break:

  (a) TRANSPORT onset  -> clockwise hysteresis (more power on the rising limb at a
      given Q; supply exhaustion / armoring on the falling limb), and a peak power
      that DECLINES across successive ARs as the supply depletes.
  (b) GEOMETRIC/wetted-front onset -> the P–Q loop is single-valued & REVERSIBLE
      (HI ~ 0) and REPEATS across ARs (set by stage geometry, not supply); an
      avulsion shows up instead as an irreversible between-AR baseline OFFSET.

We compute, per station per AR: the Lawler clockwise hysteresis index (HI), the
in-window peak log-power, and the cross-AR baseline offset at matched discharge
(median residual about the pooled P–Q fit). Outputs fig16_braided_hysteresis.png
and config/braided_hysteresis.json.

Usage: pixi run python workflows/18_braided_hysteresis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.analysis import load_timeseries, lawler_hysteresis_index  # noqa: E402

RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
STATIONS = ["CC.PR01", "CC.PR02", "CC.PR03"]
AR_COLORS = {"pre-AR": "#999999", "AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}


def main() -> int:
    ars = json.loads((ROOT / "config" / "ar_windows.json").read_text())
    fig, axes = plt.subplots(1, len(STATIONS), figsize=(4.2 * len(STATIONS), 4.4),
                             sharey=False)
    out = {}
    for ax, sid in zip(axes, STATIONS):
        j = load_timeseries(RESULTS / f"{sid}_5.0-15.0Hz_timeseries.csv")
        lp_all = np.log10(j["P"].clip(lower=1e-30))
        lq_all = np.log10(j["Q"].clip(lower=1e-6))
        # pooled P–Q reference fit (used to measure per-AR baseline offset)
        b, a = np.polyfit(lq_all.values, lp_all.values, 1)
        recs = []
        for w in ars:
            if w["label"] == "pre-AR":
                continue
            sl = j[(j.index >= pd.Timestamp(w["start"])) & (j.index <= pd.Timestamp(w["end"]))]
            if len(sl) < 8:
                continue
            q = sl["Q"].values
            lp = np.log10(sl["P"].clip(lower=1e-30)).values
            lq = np.log10(sl["Q"].clip(lower=1e-6)).values
            hi = lawler_hysteresis_index(q, lp)
            offset = float(np.median(lp - (a + b * lq)))      # baseline shift vs pooled fit
            recs.append(dict(AR=w["label"], HI=None if not np.isfinite(hi) else round(hi, 3),
                             peak_logP=round(float(lp.max()), 3),
                             baseline_offset=round(offset, 3), n=len(sl)))
            # plot the loop: rising limb solid, falling limb dashed
            pk = int(np.argmax(q))
            c = AR_COLORS.get(w["label"], "#333")
            ax.plot(lq[:pk + 1], lp[:pk + 1], "-", color=c, lw=1.6, alpha=0.9,
                    label=f"{w['label']} (HI={hi:+.2f})")
            ax.plot(lq[pk:], lp[pk:], "--", color=c, lw=1.3, alpha=0.7)
        xs = np.array([lq_all.min(), lq_all.max()])
        ax.plot(xs, a + b * xs, color="0.3", lw=1.0, ls=":", zorder=0)
        ax.set_title(sid, fontsize=10)
        ax.set_xlabel(r"$\log_{10} Q$ (m³ s⁻¹)")
        ax.legend(fontsize=7, loc="lower right")
        out[sid] = dict(b_pooled=round(float(b), 2), per_AR=recs)
    axes[0].set_ylabel(r"$\log_{10} P$ (5–15 Hz)")
    fig.suptitle("Braided-reach onset diagnostic — solid = rising limb, dashed = falling limb\n"
                 "HI>0 clockwise (transport); HI≈0 reversible (geometric); cross-AR offset = avulsion",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig16_braided_hysteresis.png", dpi=200)
    plt.close(fig)
    (ROOT / "config" / "braided_hysteresis.json").write_text(json.dumps(out, indent=2))
    for sid, d in out.items():
        print(f"\n{sid}  b_pooled={d['b_pooled']}")
        print(pd.DataFrame(d["per_AR"]).to_string(index=False))
    print(f"\nwrote {FIGDIR}/fig16_braided_hysteresis.png + config/braided_hysteresis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
