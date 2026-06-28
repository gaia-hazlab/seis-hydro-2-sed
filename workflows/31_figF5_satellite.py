#!/usr/bin/env python3
"""Composite figure F5 — "Satellite corroboration of thread migration (two basins)."

MERGES the old fig19 (Puyallup, 3 panels), fig19_nisqually (Nisqually, 3 panels)
and fig24 (two-region) into one 2-row figure for the manuscript. Top row =
Puyallup (CC.PR cluster); bottom row = Nisqually (UW.LON). Per basin:

  (col1) Sentinel-2 true-color BASEMAP (illustrative landscape context) with the
         POST active-channel outline overlaid (bright red) + station triangles.
  (col2) Active-channel CHANGE map (quantitative S2∪S1 mask): persistent / newly-
         wet / newly-dry, faint over the basemap for landscape context.
  (col3) Per-station predicted geometric baseline drift ΔlogP = pred_dlog10P with
         MNDWI-threshold-ensemble min/max whiskers; observed +0.2 cross-AR drift
         as a dashed green reference. The two basins SHARE one drift panel that
         compares PR01/PR02/PR03 and UW.LON (less clutter than two bar panels).

Basemap is illustrative; the channel masks/outlines are the quantitative layer —
they share the EXACT same UTM 10N (EPSG:32610) grid + extent as the basemap, so
overlays co-register pixel-for-pixel. Fully offline from the committed cache:
notebooks/data/braid_cache/{region}_rasters.npz, {region}_basemap.npz,
{region}_spx.json + config/braid_optical_change{,_nisqually}.json.

Outputs paper/figures/figF5_satellite.png.

Usage: pixi run python workflows/31_figF5_satellite.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style          # noqa: E402
from riverseis.basemap import imshow_basemap        # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "braid_cache"
CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"

# (region, pretty basin name, config file, drift-highlight station)
REGIONS = [
    ("puyallup", "Puyallup", "braid_optical_change.json", "CC.PR01"),
    ("nisqually", "Nisqually", "braid_optical_change_nisqually.json", "UW.LON"),
]
# bg(transparent placeholder) / persistent / newly-wet / newly-dry
CHG_COLORS = ["#3690c0", "#e31a1c", "#fdb863"]


def _change_map(channel_pre: np.ndarray, channel_post: np.ndarray) -> np.ndarray:
    """0=unchanged-dry, 1=persistent, 2=newly-wet, 3=newly-dry."""
    pre, post = channel_pre.astype(int), channel_post.astype(int)
    chg = np.zeros_like(pre)
    chg[(pre == 1) & (post == 1)] = 1
    chg[(pre == 0) & (post == 1)] = 2
    chg[(pre == 1) & (post == 0)] = 3
    return chg


def _stations(ax, spx: dict, fs: int = 12):
    # alternate the label offset above/below so neighbouring triangles whose
    # labels would otherwise stack do not collide on the map.
    for i, (name, (_r, _c, ux, uy)) in enumerate(sorted(spx.items())):
        ax.plot(ux, uy, "^", ms=11, mfc="yellow", mec="k", mew=1.3, zorder=6)
        dy = 7 if i % 2 == 0 else -16
        va = "bottom" if dy > 0 else "top"
        ax.annotate(name.split(".")[1], (ux, uy), color="white", fontsize=fs,
                    xytext=(9, dy), textcoords="offset points", va=va,
                    zorder=7,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])


def main() -> int:
    paper_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panel = iter("abcdef")
    drift = []  # collected for the shared col-3 panel

    for ri, (region, basin, cfg_name, hilite) in enumerate(REGIONS):
        z = np.load(CACHE / f"{region}_rasters.npz")
        ext = [float(z["x"].min()), float(z["x"].max()),
               float(z["y"].min()), float(z["y"].max())]
        spx = {k: tuple(v) for k, v in
               json.loads((CACHE / f"{region}_spx.json").read_text()).items()}
        cfg = json.loads((CONFIG / cfg_name).read_text())
        st = cfg["stations"]
        for name, v in st.items():
            drift.append((name, v["pred_dlog10P_median"],
                          v["pred_dlog10P_min"], v["pred_dlog10P_max"],
                          name == hilite))

        # ---- col1: S2 true-color basemap + post active-channel outline ----
        ax = axes[ri, 0]
        imshow_basemap(ax, region)
        ax.contour(z["channel_post"].astype(float), levels=[0.5], extent=ext,
                   origin="upper", colors="#ff2d2d", linewidths=1.1, zorder=4)
        _stations(ax, spx)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"({next(panel)}) {basin} — S2 true-color + post channel",
                     fontsize=13.5)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=11)

        # ---- col2: active-channel change map (quantitative), faint over basemap ----
        ax = axes[ri, 1]
        imshow_basemap(ax, region, alpha=0.45)
        chg = _change_map(z["channel_pre"], z["channel_post"])
        rgba = np.zeros((*chg.shape, 4), float)
        for cls, hexc in enumerate(CHG_COLORS, start=1):
            r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
            sel = chg == cls
            rgba[sel] = (r, g, b, 1.0)
        ax.imshow(rgba, extent=ext, origin="upper", zorder=3, interpolation="nearest")
        _stations(ax, spx)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"({next(panel)}) Active-channel change (S2$\\cup$S1)",
                     fontsize=13.5)
        # colour key as a compact in-panel legend (was an over-long title)
        from matplotlib.patches import Patch as _Patch
        ax.legend(handles=[_Patch(fc=CHG_COLORS[0], label="persistent"),
                           _Patch(fc=CHG_COLORS[1], label="newly-wet"),
                           _Patch(fc=CHG_COLORS[2], label="newly-dry")],
                  fontsize=11, loc="lower left", frameon=True, framealpha=0.9,
                  handlelength=1.1, handleheight=1.0, borderpad=0.4,
                  labelspacing=0.3).set_zorder(8)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=11)

    # ---- col3 (shared, spanning both rows): predicted geometric drift ΔlogP ----
    gs = axes[0, 2].get_gridspec()
    for ax in (axes[0, 2], axes[1, 2]):
        ax.remove()
    axd = fig.add_subplot(gs[:, 2])
    names = [d[0] for d in drift]
    x = np.arange(len(names))
    med = np.array([d[1] for d in drift])
    lo = med - np.array([d[2] for d in drift])
    hi = np.array([d[3] for d in drift]) - med
    colors = ["#e31a1c" if d[4] else "#08519c" for d in drift]
    axd.bar(x, med, 0.6, yerr=[lo, hi], capsize=5, color=colors,
            error_kw=dict(ecolor="0.3", lw=1.3))
    r_e = float(json.loads((CONFIG / REGIONS[0][2]).read_text())["r_e_m"])
    obs = float(json.loads((CONFIG / REGIONS[0][2]).read_text())
                .get("observed_baseline_drift_log10", 0.2))
    axd.axhline(obs, ls="--", color="green", lw=1.6)
    axd.annotate(f"observed cross-AR drift +{obs:g}", (0.0, obs),
                 xytext=(4, 5), textcoords="offset points",
                 color="green", fontsize=12, ha="left", va="bottom")
    axd.axhline(0, color="k", lw=0.8)
    axd.set_xticks(x)
    axd.set_xticklabels([n.split(".")[1] for n in names], rotation=15, ha="right")
    axd.tick_params(labelsize=12)
    axd.set_ylabel(r"predicted $\Delta\log_{10}P=\log_{10}(W_{post}/W_{pre})$",
                   fontsize=13)
    axd.set_title("(c/f) Predicted geometric drift $\\Delta\\log P$", fontsize=13.5)
    axd.text(0.5, -0.13, r"$W=\sum A\,r^{-1}e^{-r/r_e}$, "
             + f"$r_e$={r_e:.0f} m; whiskers = MNDWI-threshold ensemble (min, max)",
             transform=axd.transAxes, ha="center", fontsize=11, color="0.3")
    # legend distinguishing the two basins' highlight scheme; placed at lower-left
    # so it never collides with the panel title or the tall bars/whiskers on top.
    from matplotlib.patches import Patch
    axd.legend(handles=[Patch(fc="#e31a1c", label="braidplain-central / focus station"),
                        Patch(fc="#08519c", label="other co-located station")],
               fontsize=12, loc="upper left", frameon=True, framealpha=0.9)

    out = FIGDIR / "figF5_satellite.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"r_e = {r_e:.0f} m; observed cross-AR drift = +{obs:g}")
    for n, m, mn, mx, _ in drift:
        print(f"  {n:9s} pred ΔlogP {m:+.3f} [{mn:+.3f},{mx:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
