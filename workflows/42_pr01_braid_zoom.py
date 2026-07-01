#!/usr/bin/env python3
r"""PR01 braidplain zoom — traced active-channel boundaries, pre vs post AR (issue #10).

Montgomery's review ask (M3): "trace the boundary of the river channels/banks to
demonstrate the multi-braided system … zoom WAY in around PR01 … really zoom in." The
hero geomorphic figure for the main manuscript is a high-resolution zoom of the CC.PR01
braided source reach with the active-channel/bank boundaries traced before and after the
December-2025 floods, showing the multi-thread system and its reorganization.

**What this figure is — and is not.** It is a *placeholder* built offline from the
committed 10-m Sentinel-2 braid cache. At 10 m the pre-flood active channel at PR01 is
only ~1 pixel (<10 m) wide — i.e. the individual braided threads and vegetated bank
lines are **sub-resolution** and cannot be traced. So this panel honestly shows what
10 m resolves: the pre→post change in wetted active-channel extent and the newly-wet /
newly-dry reorganization pattern. Tracing the *threads and banks* Montgomery asked for
needs ≈3-m imagery (PlanetScope / Google Earth) or, better, the repeat-lidar
DEMs-of-Difference of @anderson2025 — flagged for the final figure.

**Anderson (2026) framing.** CC.PR01 sits in the *upper* Puyallup, which the 2002–2022
repeat-lidar sediment budget shows to be a **net-erosional sediment source** (bank/bluff
erosion, persistent channel reorganization) that feeds the aggrading lowland downstream —
*not* a locally aggrading reach. The reorganization imaged here (threads captured and
abandoned) is the surface expression of that source-reach behaviour: lateral
thread-switching and bank erosion, not vertical build-up. A repeat-lidar DoD panel over
this reach would independently document the chronic (pre-2025) reorganization regime and
locate erosion vs deposition within the braidplain — the natural geomorphic complement to
this event-scale optical trace.

Outputs paper/figures/fig33_pr01_braid_zoom.png. Offline (reads notebooks/data/braid_cache).

Usage: pixi run python workflows/42_pr01_braid_zoom.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "braid_cache"
FIGDIR = ROOT / "paper" / "figures"
STATION = "CC.PR01"
HALF = 550.0                      # half-window (m) around PR01 -> ~1.1 km zoom
SCALEBAR_M = 200.0


def crop(x, y, win):
    """Index masks for a HALF-m box around (win = (cx, cy))."""
    cx, cy = win
    mx = (x >= cx - HALF) & (x <= cx + HALF)
    my = (y >= cy - HALF) & (y <= cy + HALF)
    return mx, my


def main() -> int:
    paper_style()
    base = np.load(CACHE / "puyallup_basemap.npz")
    ras = np.load(CACHE / "puyallup_rasters.npz")
    spx = json.loads((CACHE / "puyallup_spx.json").read_text())
    x, y = base["x"], base["y"]
    rgb = base["rgb"]
    cpre, cpost = ras["channel_pre"], ras["channel_post"]
    cx, cy = spx[STATION][2], spx[STATION][3]

    mx, my = crop(x, y, (cx, cy))
    xs, ys = x[mx], y[my]
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    sub_rgb = rgb[np.ix_(my, mx)]
    pre = cpre[np.ix_(my, mx)].astype(float)
    post = cpost[np.ix_(my, mx)].astype(float)
    XX, YY = np.meshgrid(xs, ys)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.4, 6.3))

    # --- (a) traced active-channel outlines, pre vs post ---
    axa.imshow(sub_rgb, extent=ext, origin="upper", interpolation="nearest")
    # contour the binary wetted masks at 0.5 -> active-channel boundary
    if pre.any():
        axa.contour(XX, YY, pre, levels=[0.5], colors="#00e5ff", linewidths=1.6)
    if post.any():
        axa.contour(XX, YY, post, levels=[0.5], colors="#ff8c1a", linewidths=1.6)
    axa.plot(cx, cy, marker="^", ms=13, mfc="yellow", mec="k", mew=1.2, zorder=6)
    axa.annotate(" PR01", (cx, cy), color="yellow", fontweight="bold", fontsize=10,
                 va="center", ha="left")
    axa.set_title("Active-channel outline: pre (Nov) vs post (Dec–Jan)", fontsize=10.5, loc="left")
    _scalebar(axa, ext)
    axa.legend(handles=[Line2D([0], [0], color="#00e5ff", lw=1.8, label="pre-flood (Nov 16–30)"),
                        Line2D([0], [0], color="#ff8c1a", lw=1.8, label="post/peak (Dec–Jan)")],
               loc="lower right", fontsize=8, framealpha=0.9)
    axa.set_xticks([]); axa.set_yticks([])

    # --- (b) reorganization classes ---
    faded = (0.55 * sub_rgb.astype(float) + 0.45 * 255).astype("uint8")
    axb.imshow(faded, extent=ext, origin="upper", interpolation="nearest")
    persistent = (pre > 0) & (post > 0)
    newly_wet = (post > 0) & (pre == 0)
    newly_dry = (pre > 0) & (post == 0)
    overlay = np.zeros((*pre.shape, 4))
    overlay[persistent] = (0.10, 0.45, 0.90, 0.95)     # blue
    overlay[newly_wet] = (0.90, 0.10, 0.15, 0.95)      # red
    overlay[newly_dry] = (1.00, 0.60, 0.00, 0.95)      # orange
    axb.imshow(overlay, extent=ext, origin="upper", interpolation="nearest")
    axb.plot(cx, cy, marker="^", ms=13, mfc="yellow", mec="k", mew=1.2, zorder=6)
    axb.set_title("Reorganization: thread capture & abandonment", fontsize=10.5, loc="left")
    _scalebar(axb, ext)
    axb.legend(handles=[Line2D([0], [0], marker="s", ls="", mfc="#1a73e6", mec="none", label="persistent"),
                        Line2D([0], [0], marker="s", ls="", mfc="#e61a26", mec="none", label="newly wet"),
                        Line2D([0], [0], marker="s", ls="", mfc="#ff9900", mec="none", label="newly dry (abandoned)")],
               loc="lower right", fontsize=8, framealpha=0.9)
    axb.set_xticks([]); axb.set_yticks([])

    fig.suptitle("CC.PR01 braided source reach (upper Puyallup) — thread-switching & bank erosion in a "
                 "net-erosional\nsource reach [Anderson 2026].  PLACEHOLDER: 10 m Sentinel-2 — threads sub-pixel; "
                 "≈3 m / lidar-DoD needed to trace threads & banks", fontsize=10.5, x=0.5, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.06)
    out = FIGDIR / "fig33_pr01_braid_zoom.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  window ±{HALF:.0f} m at PR01; pre wet px={int((pre>0).sum())} post wet px={int((post>0).sum())} "
          f"(newly-wet {int(((post>0)&(pre==0)).sum())}, newly-dry {int(((pre>0)&(post==0)).sum())})")
    return 0


def _scalebar(ax, ext):
    x0 = ext[0] + 0.06 * (ext[1] - ext[0])
    y0 = ext[2] + 0.08 * (ext[3] - ext[2])
    ax.add_patch(Rectangle((x0, y0), SCALEBAR_M, 0.012 * (ext[3] - ext[2]),
                           fc="white", ec="k", lw=0.8, zorder=7))
    ax.text(x0 + SCALEBAR_M / 2, y0 + 0.03 * (ext[3] - ext[2]), f"{SCALEBAR_M:.0f} m",
            color="white", fontsize=8, ha="center", va="bottom", fontweight="bold", zorder=7)


if __name__ == "__main__":
    raise SystemExit(main())
