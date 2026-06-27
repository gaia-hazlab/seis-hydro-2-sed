#!/usr/bin/env python3
r"""At-a-station width-stage hysteresis on the rising vs falling limb (M4).

Co-author (D. Montgomery) question: "What is the change in width with respect to
the change in height in the rising limb and the falling limbs?"

We have continuous *stage* (Electron gage 12092000) and a *wetted-channel-width
proxy* at five SAR epochs (the December series in braid_cache: PR01/PR02 active
area normalized to PR03 and to the pre-flood epoch). Plotting the proxy against
the epoch-mean stage traces an at-a-station width-stage loop and separates the
rising limb (Dec 1-8 -> Dec 9-12 peak) from the falling limb (peak -> late Dec).

The finding: width responds far more steeply on the rise than the fall (a
counter-clockwise loop), and the PR01 thread does **not** re-widen on the
recession - it ends *below* its pre-flood width (the abandoned thread of the
avulsion), while PR02 returns to ~baseline. That asymmetric dW/dH is the
geometric signature of braid reorganization, not a reversible stage response.

RESOLUTION CAVEAT (stated in the caption): five coarse epochs, an area-ratio
proxy (not metric width), and epoch-averaged stage. A >=3 m width time series
would sharpen dW/dH; the asymmetry and the PR01 net-narrowing are already clear.

Fully offline: notebooks/data/braid_cache/puyallup_december_series.csv +
notebooks/data/usgs_iv_12092000_*.csv. Outputs fig28_width_stage.png.

Usage: pixi run python workflows/27_width_stage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "braid_cache"
DATA = ROOT / "notebooks" / "data"
FIGDIR = ROOT / "paper" / "figures"

# SAR epoch -> (start, end) UTC date windows (2025); the gage backs the Dec windows.
EPOCH_WIN = {
    "Dec 1–8": ("2025-12-01", "2025-12-08 23:59"),
    "Dec 9–12 (AR peak)": ("2025-12-09", "2025-12-12 23:59"),
    "Dec 13–20": ("2025-12-13", "2025-12-20 23:59"),
    "Dec 21–31": ("2025-12-21", "2025-12-31 23:59"),
}
ORDER = ["Nov 16–30", "Dec 1–8", "Dec 9–12 (AR peak)", "Dec 13–20", "Dec 21–31"]


def epoch_stage() -> dict[str, dict]:
    """Mean / min / max gage stage (ft) per Dec epoch window."""
    df = pd.read_csv(DATA / "usgs_iv_12092000_2025-12-01_2026-01-01.csv",
                     parse_dates=["time_utc"]).set_index("time_utc")
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    h = df["gage_height_ft"]
    out = {}
    for lab, (a, b) in EPOCH_WIN.items():
        s = h.loc[a:b]
        out[lab] = dict(mean=float(s.mean()), lo=float(s.min()), hi=float(s.max()))
    return out


def main() -> int:
    paper_style()
    series = pd.read_csv(CACHE / "puyallup_december_series.csv").set_index("epoch_label")
    st = epoch_stage()
    # Anchor the Nov pre-flood reference at the Dec 1-8 low-flow stage (open marker):
    # the imagery normalizes width to Nov, but the CSV stage starts 1 Dec.
    st_full = {"Nov 16–30": dict(mean=st["Dec 1–8"]["mean"], lo=np.nan, hi=np.nan), **st}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.28))

    # ---- (a) width-stage loop ----
    cols = {"PR01_rel_to_PR03": ("#e31a1c", "CC.PR01"),
            "PR02_rel_to_PR03": ("#ff7f00", "CC.PR02")}
    H = np.array([st_full[e]["mean"] for e in ORDER])
    for key, (c, lab) in cols.items():
        W = np.array([series.loc[e, key] for e in ORDER])
        lo = np.array([st_full[e]["lo"] for e in ORDER])
        hi = np.array([st_full[e]["hi"] for e in ORDER])
        axL.errorbar(H[1:], W[1:], xerr=[H[1:] - lo[1:], hi[1:] - H[1:]],
                     fmt="none", ecolor=c, alpha=0.35, lw=1, capsize=2, zorder=2)
        # directed path through time
        for i in range(len(ORDER) - 1):
            axL.annotate("", xy=(H[i + 1], W[i + 1]), xytext=(H[i], W[i]),
                         arrowprops=dict(arrowstyle="->", color=c, lw=1.6, alpha=0.8),
                         zorder=3)
        axL.plot(H[0], W[0], "o", mfc="white", mec=c, mew=1.6, ms=8, zorder=4)  # Nov ref
        axL.plot(H[1:], W[1:], "o", color=c, ms=7, zorder=4, label=lab)
    axL.axhline(1.0, color="0.6", ls=":", lw=1)
    axL.annotate("pre-flood width (Nov, ref)", (H.min(), 1.0), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color="0.4")
    # epoch labels on PR01 track
    Wp = np.array([series.loc[e, "PR01_rel_to_PR03"] for e in ORDER])
    for e, x, y in zip(ORDER, H, Wp):
        dx, dy = (6, -2)
        if e == "Dec 9–12 (AR peak)":
            dx, dy = (-8, -12)
        axL.annotate(e.replace(" (AR peak)", " (peak)"), (x, y),
                     xytext=(dx, dy), ha="right" if dx < 0 else "left",
                     textcoords="offset points", fontsize=7.5, color="0.3")
    axL.set_xlabel("gage stage at Electron (ft, epoch mean; whiskers = epoch min–max)")
    axL.set_ylabel("wetted active-channel width proxy\n(SAR area, ÷PR03, ÷Nov baseline)")
    axL.set_title("(a) Width–stage hysteresis: rise widens, fall does not recover",
                  fontsize=10.5, loc="left")
    axL.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # ---- (b) secant dW/dH on rising vs falling limb ----
    def dwdh(key):
        W = {e: series.loc[e, key] for e in ORDER}
        rise = (W["Dec 9–12 (AR peak)"] - W["Dec 1–8"]) / (
            st["Dec 9–12 (AR peak)"]["mean"] - st["Dec 1–8"]["mean"])
        fall = (W["Dec 21–31"] - W["Dec 9–12 (AR peak)"]) / (
            st["Dec 21–31"]["mean"] - st["Dec 9–12 (AR peak)"]["mean"])
        return rise, fall

    labs = ["CC.PR01", "CC.PR02"]
    rises = [dwdh("PR01_rel_to_PR03")[0], dwdh("PR02_rel_to_PR03")[0]]
    falls = [dwdh("PR01_rel_to_PR03")[1], dwdh("PR02_rel_to_PR03")[1]]
    x = np.arange(2)
    axR.bar(x - 0.18, rises, 0.34, color="#2c7fb8", label="rising limb")
    axR.bar(x + 0.18, falls, 0.34, color="#cb6a3a", label="falling limb")
    axR.axhline(0, color="k", lw=0.8)
    for xi, r, f in zip(x, rises, falls):
        axR.annotate(f"{r:+.2f}", (xi - 0.18, r), ha="center",
                     va="bottom" if r >= 0 else "top", fontsize=8)
        axR.annotate(f"{f:+.2f}", (xi + 0.18, f), ha="center",
                     va="bottom" if f >= 0 else "top", fontsize=8)
    axR.set_xticks(x); axR.set_xticklabels(labs)
    axR.set_ylabel("secant dW/dH  (Δ width-proxy / Δ stage, ft⁻¹)")
    axR.set_ylim(0, max(rises) * 1.28)
    axR.set_title("(b) dW/dH steeper on the rise than the fall",
                  fontsize=10.5, loc="left")
    axR.legend(loc="upper right", fontsize=8.5, framealpha=0.95)

    fig.suptitle("At-a-station width–stage relation, rising vs falling limb "
                 "(Puyallup source, M4)", fontsize=12.5, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGDIR / "fig28_width_stage.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    # console summary for the manuscript text
    for key, lab in [("PR01_rel_to_PR03", "PR01"), ("PR02_rel_to_PR03", "PR02")]:
        r, f = dwdh(key)
        peak = series.loc["Dec 9–12 (AR peak)", key]
        late = series.loc["Dec 21–31", key]
        print(f"{lab}: peak={peak:.2f}x  late={late:.2f}x  dW/dH rise={r:+.2f} fall={f:+.2f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
