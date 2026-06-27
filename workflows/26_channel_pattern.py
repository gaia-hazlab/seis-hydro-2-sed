#!/usr/bin/env python3
"""Channel-pattern map: braided source reaches vs single-thread downstream (M1).

Co-author (D. Montgomery) question: "single-thread meandering channel vs single
channel with braids — show a map that compares down the flow past CC.TRON and
upstream at the PR stations and UW.LON stations."

Left: a corridor locator (OPERA DSWx-S1 flood water, 10 Dec 2025, cached; NHD
mainstems; seismic stations + gages) with the three zoom reaches boxed. Right, top→
bottom along the flow: the **braided** Puyallup source at the PR cluster and the
**braided** Nisqually at UW.LON (tight crops of the cached Sentinel active-channel
rasters — multi-thread), versus the **single-thread, meandering** lower Puyallup past
CC.TRON (OPERA flood water). The contrast in planform IS the point.

Fully offline: notebooks/data/braid_cache/{opera_corridor,puyallup_rasters,
nisqually_rasters}.npz + config/nhd_rivers.json. Outputs fig27_channel_pattern.png.

Usage: pixi run python workflows/26_channel_pattern.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "braid_cache"
CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"
VIEW = (-122.36, -121.78, 46.73, 47.24)
STATIONS = {
    "PR01": (-122.0376, 46.9101, "#e31a1c"), "PR02": (-122.0487, 46.9183, "#e31a1c"),
    "PR03": (-122.0327, 46.9034, "#e31a1c"), "LON": (-121.8096, 46.7506, "#1f78b4"),
    "TRON": (-122.1753, 46.9977, "#ff7f00"),
}
GAGES = {"Electron": (-122.0351, 46.9037), "Orting": (-122.208, 47.039),
         "Puyallup": (-122.230, 47.185), "National": (-122.0837, 46.7526)}
# downstream single-thread reach to crop from OPERA (lon0, lon1, lat0, lat1)
DOWN = (-122.245, -122.165, 46.99, 47.10)


def opera_crop(op, box):
    w = op["water"]; L, R, B, T = (float(op[k]) for k in ("left", "right", "bottom", "top"))
    ny, nx = w.shape
    x0 = int((box[0] - L) / (R - L) * nx); x1 = int((box[1] - L) / (R - L) * nx)
    y0 = int((T - box[3]) / (T - B) * ny); y1 = int((T - box[2]) / (T - B) * ny)
    return w[y0:y1, x0:x1]


def braid_crop(npz, pad=20):
    """channel_post cropped tight to the active channel (so a thin braid fills the panel)."""
    z = np.load(CACHE / npz)
    ch = z["channel_post"].astype(float)
    rows, cols = np.where(ch > 0)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, ch.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, ch.shape[1])
    return ch[r0:r1, c0:c1]


def main() -> int:
    paper_style()
    op = np.load(CACHE / "opera_corridor.npz")
    water = op["water"]
    ext = [float(op["left"]), float(op["right"]), float(op["bottom"]), float(op["top"])]
    rivers = json.loads((CONFIG / "nhd_rivers.json").read_text())

    fig = plt.figure(figsize=(12.5, 8.5))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.7, 1], wspace=0.12, hspace=0.18)
    axm = fig.add_subplot(gs[:, 0])

    # --- locator map ---
    axm.imshow(np.ma.masked_where(water == 0, water), extent=ext, origin="upper",
               cmap=ListedColormap(["#2c7fb8"]), alpha=0.85, zorder=1)
    for name, col in [("Puyallup", "#08519c"), ("Carbon", "#6baed6"),
                      ("Nisqually", "#08519c"), ("White", "#9ecae1")]:
        for seg in rivers.get(name, []):
            a = np.asarray(seg, float)
            if a.ndim == 2 and a.shape[1] == 2:
                axm.plot(a[:, 0], a[:, 1], "-", color=col, lw=0.8, alpha=0.7, zorder=2)
    for sid, (lon, lat, col) in STATIONS.items():
        axm.plot(lon, lat, "^", ms=10, mfc=col, mec="k", mew=0.8, zorder=6)
        axm.annotate(sid, (lon, lat), color="k", fontsize=8, fontweight="bold",
                     xytext=(4, 4), textcoords="offset points", zorder=7)
    for lon, lat in GAGES.values():
        axm.plot(lon, lat, "D", ms=6, mfc="cyan", mec="k", mew=0.6, zorder=5)
    # boxes marking the zoom reaches
    for box, col in [((-122.07, -122.01, 46.885, 46.93), "#e31a1c"),
                     ((-121.83, -121.79, 46.74, 46.765), "#1f78b4"),
                     (DOWN, "#238b45")]:
        axm.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                                fill=False, ec=col, lw=1.6, zorder=8))
    axm.annotate("BRAIDED source\n(Puyallup, PR cluster)", (-122.04, 46.93),
                 xytext=(-122.33, 46.96), fontsize=9.5, fontweight="bold", color="#a50026",
                 arrowprops=dict(arrowstyle="->", color="#a50026"))
    axm.annotate("BRAIDED\n(Nisqually, UW.LON)", (-121.81, 46.74),
                 xytext=(-122.02, 46.745), fontsize=9.5, fontweight="bold", color="#08519c",
                 arrowprops=dict(arrowstyle="->", color="#08519c"))
    axm.annotate("SINGLE-THREAD,\nmeandering\n(past TRON)", (-122.205, 47.05),
                 xytext=(-122.34, 47.13), fontsize=9.5, fontweight="bold", color="#238b45",
                 arrowprops=dict(arrowstyle="->", color="#238b45"))
    axm.plot([], [], "^", mfc="0.6", mec="k", label="seismic station")
    axm.plot([], [], "D", mfc="cyan", mec="k", label="USGS gage")
    axm.plot([], [], "s", mfc="#2c7fb8", mec="none", label="flood water (OPERA, 10 Dec)")
    axm.set_xlim(VIEW[0], VIEW[1]); axm.set_ylim(VIEW[2], VIEW[3])
    axm.set_aspect(1 / np.cos(np.radians(47.0)))
    axm.set_xlabel("longitude"); axm.set_ylabel("latitude")
    axm.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axm.set_title("(a) Corridor — pattern transition along the flow", fontsize=11, loc="left")

    # --- zoom panels along the flow ---
    chmap = ListedColormap(["#f7f7f7", "#3690c0"])
    panels = [
        ("braid", "puyallup_rasters.npz", "#e31a1c",
         "(b) Puyallup source — BRAIDED, compact (PR, tens of m)"),
        ("braid", "nisqually_rasters.npz", "#1f78b4",
         "(c) Nisqually — BRAIDED, wide (UW.LON, hundreds of m)"),
        ("opera", DOWN, "#238b45",
         "(d) lower Puyallup past TRON — SINGLE-THREAD, meandering"),
    ]
    for i, (kind, src, col, title) in enumerate(panels):
        axz = fig.add_subplot(gs[i, 1])
        if kind == "opera":
            img = opera_crop(op, src)
            axz.imshow(np.ma.masked_where(img == 0, img), origin="upper", extent=src,
                       cmap=ListedColormap(["#2c7fb8"]), aspect="auto", zorder=1)
            axz.set_facecolor("#f7f7f7")
            for seg in rivers.get("Puyallup", []):         # overlay the single NHD thread
                a = np.asarray(seg, float)
                if a.ndim == 2 and a.shape[1] == 2:
                    axz.plot(a[:, 0], a[:, 1], "-", color="#08519c", lw=1.4, zorder=3)
            axz.set_xlim(src[0], src[1]); axz.set_ylim(src[2], src[3])
        else:
            axz.imshow(braid_crop(src), origin="upper", cmap=chmap, vmin=0, vmax=1, aspect="auto")
        axz.set_xticks([]); axz.set_yticks([])
        for sp in axz.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2.2)
        axz.set_title(title, fontsize=9, color=col, pad=2, loc="left")

    fig.suptitle("Channel pattern along the flow path — braided source reaches vs "
                 "single-thread lowland (M1)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGDIR / "fig27_channel_pattern.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
