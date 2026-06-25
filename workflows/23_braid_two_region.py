#!/usr/bin/env python3
"""Two-region braid-change comparison (fig24) — Puyallup vs Nisqually, from cache.

Side-by-side active-channel change maps (Nov-2025 → Jan-2026) for the two source
reaches, built OFFLINE from the committed satellite-artefact cache
(notebooks/data/braid_cache/*.npz) — no Planetary Computer. The point is the
geomorphic contrast that underlies the domain-of-applicability result (REVIEW §9 /
fig23): the Puyallup PR cluster sits on a COMPACT (tens-of-m), incised channel where
the seismic source is near-stationary and the gage is co-located; the Nisqually
UW.LON sits on a WIDE (hundreds-of-m), unconfined braidplain gauged 13–21 km
downstream, where the source is distributed and non-stationary.

Reads only the committed npz/json cache + the optical configs. Outputs
paper/figures/fig24_braid_two_region.png.

Usage: pixi run python workflows/23_braid_two_region.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "braid_cache"
CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"
REGIONS = [
    ("puyallup", "Puyallup — PR cluster", "braid_optical_change.json"),
    ("nisqually", "Nisqually — UW.LON", "braid_optical_change_nisqually.json"),
]
CHG_CMAP = ListedColormap(["#f7f7f7", "#3690c0", "#e31a1c", "#fdb863"])  # bg/persist/wet/dry


def _change_map(channel_pre: np.ndarray, channel_post: np.ndarray) -> np.ndarray:
    pre, post = channel_pre.astype(int), channel_post.astype(int)
    chg = np.zeros_like(pre)
    chg[(pre == 1) & (post == 1)] = 1   # persistent
    chg[(pre == 0) & (post == 1)] = 2   # newly wet
    chg[(pre == 1) & (post == 0)] = 3   # newly dry
    return chg


def main() -> int:
    paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    for ax, (region, title, cfg_name) in zip(axes, REGIONS):
        npz = CACHE / f"{region}_rasters.npz"
        if not npz.exists():
            ax.text(0.5, 0.5, f"no cache for {region}\n(run workflow 19 live once)",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        z = np.load(npz)
        ext = [float(z["x"].min()), float(z["x"].max()),
               float(z["y"].min()), float(z["y"].max())]
        chg = _change_map(z["channel_pre"], z["channel_post"])
        ax.imshow(chg, extent=ext, origin="upper", cmap=CHG_CMAP, vmin=-0.5, vmax=3.5)
        spx = json.loads((CACHE / f"{region}_spx.json").read_text())
        for name, (_, _, ux, uy) in spx.items():
            ax.plot(ux, uy, "^", ms=12, mfc="yellow", mec="k", mew=1.3, zorder=5)
            ax.annotate(name.split(".")[1], (ux, uy), color="k", fontsize=9,
                        xytext=(6, 5), textcoords="offset points", fontweight="bold")
        # channel-width annotation from the optical config (the geomorphic contrast)
        st = json.loads((CONFIG / cfg_name).read_text()).get("stations", {})
        widths = [v.get("W_pre") for v in st.values() if v.get("W_pre")]
        wtxt = (f"active-channel width ≈ {min(widths):.0f}–{max(widths):.0f} m"
                if len(widths) > 1 else
                (f"active-channel width ≈ {widths[0]:.0f} m" if widths else ""))
        ax.set_title(f"{title}\n{wtxt}", fontsize=11.5)
        ax.set_xlabel("UTM E (m)"); ax.set_ylabel("UTM N (m)")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=8)

    handles = [plt.Line2D([], [], marker="s", ls="", mfc=c, mec="0.6", ms=11, label=lab)
               for c, lab in zip(CHG_CMAP.colors[1:],
                                 ["persistent channel", "newly wet", "newly dry"])]
    handles.append(plt.Line2D([], [], marker="^", ls="", mfc="yellow", mec="k",
                              ms=11, label="seismic station"))
    fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Active-channel change Nov-2025 → Jan-2026 (S2∪S1) — compact incised "
                 "Puyallup vs wide Nisqually braidplain", fontsize=12.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = FIGDIR / "fig24_braid_two_region.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
