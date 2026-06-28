#!/usr/bin/env python3
"""Composite figure F1 — "Study area, channel pattern, and the warm-AR flood driver."

Merges three former figures into one paper-grade composite:
  fig1  (transect locator)         -> panel (a)
  fig27 (channel-pattern zoom)     -> panels (b),(c),(d)
  fig21 (warm-AR SNOTEL driver)    -> panel (e)

Layout (gridspec, ~13x9 in):
  (a) BIG locator on the cached corridor Sentinel-2 basemap (lon/lat): OPERA
      flood water, NHD mainstems, seismic stations coloured by sample rate,
      USGS gages + SNOTEL, and the three zoom-reach boxes.
  (b) Puyallup PR braidplain zoom (UTM) — basemap + active-channel outline.
  (c) Nisqually UW.LON zoom (UTM)       — basemap + active-channel outline.
  (d) lower Puyallup past TRON — single-thread, meandering (corridor basemap crop
      + NHD centreline).
  (e) Warm-AR SNOTEL flood-driver inset — air temperature vs the 0 C freezing
      line, with SWE on a twin axis.

Fully offline: notebooks/data/braid_cache/{opera_corridor,puyallup_rasters,
nisqually_rasters,*_basemap}.npz + config/{nhd_rivers,cc_stations,uw_stations,
map_layers,warm_ar_snotel,ar_windows}.json.

Outputs paper/figures/figF1_studyarea.png.

Usage: pixi run python workflows/30_figF1_studyarea.py
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
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style          # noqa: E402
from riverseis.basemap import imshow_basemap, load_basemap  # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "braid_cache"
CFG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"

# corridor locator view = the cached corridor basemap extent (lon/lat)
VIEW = (-122.36, -121.78, 46.73, 47.24)

# zoom-reach boxes (lon0, lon1, lat0, lat1) drawn on panel (a)
BOX_PUY = (-122.07, -122.025, 46.895, 46.935)     # Puyallup braidplain (b)
BOX_NIS = (-121.828, -121.792, 46.738, 46.764)    # Nisqually UW.LON (c)
BOX_DOWN = (-122.245, -122.165, 46.99, 47.10)     # lower Puyallup single-thread (d)

# stations to label on (a) (key clusters / transect anchors)
LABEL = {"PR01", "PR02", "PR03", "LON", "TRON", "SIFT", "STYX"}

# PR cluster + UW.LON coords for the zoom panels (lon, lat)
PR = {"PR01": (-122.0376, 46.9101), "PR02": (-122.0487, 46.9183),
      "PR03": (-122.0327, 46.9034)}
LON = (-121.8096, 46.7506)

UTM = "EPSG:32610"
RAINIER = (-121.7603, 46.8523)


def lighten(rgb: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Blend an RGB uint8 image toward white by ``frac`` so overlays read on the
    dark forest basemap. Returns float [0,1]."""
    a = rgb.astype("float32") / 255.0
    return np.clip(a + (1.0 - a) * frac, 0, 1)


def to_utm(lon: float, lat: float):
    import rasterio.warp
    x, y = rasterio.warp.transform("EPSG:4326", UTM, [lon], [lat])
    return x[0], y[0]


def opera_mask(op):
    water = op["water"]
    ext = [float(op["left"]), float(op["right"]), float(op["bottom"]), float(op["top"])]
    return water, ext


# ---------------------------------------------------------------------------
def panel_a(ax, op, rivers, cc, uw, gages, snotel):
    """BIG corridor locator on the lightened Sentinel-2 basemap (lon/lat)."""
    bm = load_basemap("corridor")
    ax.imshow(lighten(bm["rgb"], 0.18), extent=bm["extent"], origin="upper",
              zorder=0, interpolation="nearest")
    ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3])

    # OPERA flood water (semi-transparent blue), mask out dry
    water, wext = opera_mask(op)
    ax.imshow(np.ma.masked_where(water == 0, water), extent=wext, origin="upper",
              cmap=ListedColormap(["#2b8cbe"]), alpha=0.55, zorder=1,
              interpolation="nearest")

    # NHD mainstem centrelines
    for name, col in [("Puyallup", "#08519c"), ("Carbon", "#3182bd"),
                      ("Nisqually", "#08519c"), ("White", "#6baed6")]:
        for seg in rivers.get(name, []):
            a = np.asarray(seg, float)
            if a.ndim == 2 and a.shape[1] == 2:
                ax.plot(a[:, 0], a[:, 1], "-", color=col, lw=1.0, alpha=0.85, zorder=2)

    # Seismic stations, coloured by sample rate (the 2026 sample-rate "what-if")
    seis = [("CC." + s["sta"], s["lon"], s["lat"], float(s["sr"])) for s in cc]
    seis += [("UW." + s["sta"], s["lon"], s["lat"], float(s["sr"])) for s in uw]
    seen = set()
    for sid, lon, lat, sr in seis:
        sta = sid.split(".")[1]
        if (lon, lat) in seen:          # LON/LO2 are co-located; draw once
            continue
        seen.add((lon, lat))
        if not (VIEW[0] <= lon <= VIEW[1] and VIEW[2] <= lat <= VIEW[3]):
            continue
        hi = sr >= 500
        ax.plot(lon, lat, "^", ms=10 if hi else 8,
                mfc="#b30000" if hi else "#fdae61", mec="k",
                mew=1.1 if hi else 0.7, zorder=6)
        if sta in LABEL:
            # per-station label offset to avoid colliding with the zoom boxes,
            # river lines, and neighbouring station labels
            dx, dy, ha = 6, 5, "left"
            if sta == "PR01":
                dx, dy, ha = 8, -4, "left"
            elif sta == "PR02":
                dx, dy, ha = -9, 9, "right"
            elif sta == "PR03":
                dx, dy, ha = 8, -12, "left"
            elif sta == "TRON":
                dx, dy, ha = 8, 6, "left"
            elif sta == "STYX":
                dx, dy, ha = -8, 6, "right"
            elif sta == "SIFT":
                dx, dy, ha = 8, -3, "left"
            elif sta == "LON":
                dx, dy, ha = 8, 4, "left"
            ax.annotate(sta, (lon, lat), color="k", fontsize=12, fontweight="semibold",
                        ha=ha, va="center", xytext=(dx, dy),
                        textcoords="offset points", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.72))

    # USGS gages (diamonds) + SNOTEL (squares)
    for g in gages:
        ax.plot(g["lon"], g["lat"], "D", ms=6, mfc="cyan", mec="k", mew=0.6, zorder=5)
    for s in snotel:
        ax.plot(s["lon"], s["lat"], "s", ms=7, mfc="#fee08b", mec="k", mew=0.6, zorder=5)

    # zoom-reach boxes
    for box, col, lab in [(BOX_PUY, "#e31a1c", "(b)"), (BOX_NIS, "#ffff33", "(c)"),
                          (BOX_DOWN, "#00e5ff", "(d)")]:
        ax.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2],
                               fill=False, ec=col, lw=2.0, zorder=8))
        ax.text(box[1], box[3], lab, color=col, fontsize=13, fontweight="semibold",
                ha="left", va="bottom", zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc="0.15", ec="none", alpha=0.7))

    # reach annotations
    ax.annotate("braided source\n(Puyallup, PR cluster)", (-122.05, 46.915),
                xytext=(-122.350, 46.960), fontsize=12, fontweight="semibold",
                color="#a50026", zorder=9, va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78),
                arrowprops=dict(arrowstyle="->", color="#a50026", lw=1.4))
    ax.annotate("braided\n(Nisqually, UW.LON)", (-121.81, 46.751),
                xytext=(-122.07, 46.738), fontsize=12, fontweight="semibold",
                color="#08519c", zorder=9, va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78),
                arrowprops=dict(arrowstyle="->", color="#08519c", lw=1.4))
    ax.annotate("single-thread,\nmeandering (past TRON)", (-122.205, 47.045),
                xytext=(-122.355, 47.155), fontsize=12, fontweight="semibold",
                color="#006d2c", zorder=9, va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78),
                arrowprops=dict(arrowstyle="->", color="#006d2c", lw=1.4))

    # Mt. Rainier toward the SE corner (summit is just off-extent)
    ax.annotate("Mt. Rainier ▲", (-121.795, 46.847), fontsize=12,
                fontweight="semibold", color="0.1", ha="right", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.78))

    ax.set_aspect(1.0 / np.cos(np.radians(47.0)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("(a) Study corridor — channel pattern along the flow",
                 fontsize=14, loc="left")

    handles = [
        Line2D([], [], ls="none", marker="^", ms=10, mfc="#b30000", mec="k",
               label="seismic station, ≥500 sps"),
        Line2D([], [], ls="none", marker="^", ms=8, mfc="#fdae61", mec="k",
               label="seismic station, <500 sps"),
        Line2D([], [], ls="none", marker="D", ms=6, mfc="cyan", mec="k",
               label="USGS gage"),
        Line2D([], [], ls="none", marker="s", ms=7, mfc="#fee08b", mec="k",
               label="SNOTEL"),
        Line2D([], [], color="#2b8cbe", lw=6, alpha=0.55,
               label="flood water (OPERA, 10 Dec 2025)"),
        Line2D([], [], color="#08519c", lw=1.4, label="NHD river"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=12, framealpha=1.0,
              borderpad=0.5, labelspacing=0.4, handletextpad=0.5).set_zorder(20)


def panel_braid(ax, region, ststns, title, col):
    """Zoom panel (UTM): lightened basemap + active-channel outline + stations."""
    bm = load_basemap(region)
    ext = bm["extent"]
    ax.imshow(lighten(bm["rgb"], 0.15), extent=ext, origin="upper", zorder=0,
              interpolation="nearest")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])

    z = np.load(CACHE / f"{region}_rasters.npz")
    ch = z["channel_post"].astype(float)
    # contour the active-channel mask at 0.5 on the basemap extent (origin upper)
    ax.contour(ch, levels=[0.5], extent=ext, origin="upper",
               colors="#ff2a2a", linewidths=1.1, zorder=3)
    # faint fill so multi-thread braids read even where the outline is thin
    ax.imshow(np.ma.masked_where(ch < 0.5, ch), extent=ext, origin="upper",
              cmap=ListedColormap(["#00e5ff"]), alpha=0.30, zorder=2,
              interpolation="nearest")

    for sta, (lon, lat) in ststns.items():
        x, y = to_utm(lon, lat)
        ax.plot(x, y, "^", ms=12, mfc="yellow", mec="k", mew=1.3, zorder=5)
        ax.annotate(sta, (x, y), color="k", fontsize=12, fontweight="semibold",
                    xytext=(8, 5), textcoords="offset points", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.8))

    ax.set_aspect("equal")          # UTM metres -> braids keep true planform
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2.4)
    ax.set_title(title, fontsize=14, color=col, loc="left", pad=4)


def panel_down(ax, rivers, col):
    """Lower-Puyallup single-thread reach: crop the corridor basemap to BOX_DOWN
    and overlay the NHD Puyallup centreline."""
    bm = load_basemap("corridor")
    rgb = lighten(bm["rgb"], 0.15)
    l, r, b, t = bm["extent"]
    H, W = rgb.shape[:2]
    lon0, lon1, lat0, lat1 = BOX_DOWN
    c0 = int(round((lon0 - l) / (r - l) * W)); c1 = int(round((lon1 - l) / (r - l) * W))
    # rows: extent top->bottom maps to row 0->H (origin upper)
    rr0 = int(round((t - lat1) / (t - b) * H)); rr1 = int(round((t - lat0) / (t - b) * H))
    c0, c1 = max(c0, 0), min(c1, W); rr0, rr1 = max(rr0, 0), min(rr1, H)
    crop = rgb[rr0:rr1, c0:c1]
    ax.imshow(crop, extent=[lon0, lon1, lat0, lat1], origin="upper", zorder=0,
              interpolation="nearest")
    ax.set_xlim(lon0, lon1); ax.set_ylim(lat0, lat1)

    for seg in rivers.get("Puyallup", []):
        a = np.asarray(seg, float)
        if a.ndim == 2 and a.shape[1] == 2:
            ax.plot(a[:, 0], a[:, 1], "-", color="#00e5ff", lw=2.0, zorder=3)
    # mark TRON if it falls in the crop
    tron = (-122.1753, 46.9977)
    if lon0 <= tron[0] <= lon1 and lat0 <= tron[1] <= lat1:
        ax.plot(*tron, "^", ms=12, mfc="#b30000", mec="k", mew=1.2, zorder=5)
        ax.annotate("TRON", tron, color="k", fontsize=12, fontweight="semibold",
                    ha="right", va="bottom", xytext=(-8, 5),
                    textcoords="offset points", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.8))

    ax.set_aspect(1.0 / np.cos(np.radians(47.0)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#006d2c"); sp.set_linewidth(2.4)
    ax.set_title("(d) lower Puyallup\n    single-thread", fontsize=14,
                 color="#006d2c", loc="left", pad=4)


def panel_snotel(ax, war, ar):
    """Warm-AR flood-driver inset: air temp vs 0 C, SWE on a twin axis."""
    colors = {"Paradise": "#d73027", "Mowich": "#4575b4"}

    def series(rec, key):
        t = pd.to_datetime(rec[key]["time"], utc=True)
        v = np.asarray(rec[key]["value"], float)
        return t, v

    # AR windows shading
    for w in ar:
        ax.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                   color="#cfe3f2", alpha=0.45, zorder=0)

    # 0 C freezing line (warm AR = above => rain-on-snow)
    ax.axhline(0.0, color="0.35", lw=1.3, ls="--", zorder=2)

    for name, rec in war.items():
        t, v = series(rec, "tempC")
        ax.plot(t, v, lw=1.5, color=colors.get(name),
                label=f"{name} T ({rec['elev_m']} m)", zorder=4)
    ax.set_ylabel("air temp (°C)", fontsize=13)

    # date ticks: weekly, formatted compactly, gently rotated so they don't crowd
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", labelsize=12)
    for lab in ax.get_xticklabels():
        lab.set_rotation(35); lab.set_ha("right")
    ax.tick_params(axis="y", labelsize=12)
    ax.margins(x=0.02)
    ax.set_title("(e) Warm-AR flood driver (SNOTEL)", fontsize=14, loc="left", pad=4)

    # SWE on a twin axis (dashed)
    ax2 = ax.twinx()
    for name, rec in war.items():
        if "swe_cm" in rec:
            t, v = series(rec, "swe_cm")
            ax2.plot(t, v, lw=1.4, ls=":", color=colors.get(name), alpha=0.85,
                     label=f"{name} SWE", zorder=3)
    ax2.set_ylabel("SWE (cm)", fontsize=13)
    ax2.tick_params(axis="y", labelsize=12)

    # generous headroom above the temperature traces so the 2-row legend sits
    # fully clear of the data peaks (the warm-AR spikes are tall)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.72 * (yhi - ylo))
    # keep the SWE twin axis in step so its dotted traces also clear the legend
    s2lo, s2hi = ax2.get_ylim()
    ax2.set_ylim(s2lo, s2hi + 0.72 * (s2hi - s2lo))
    # 0 C label in the clear lower-left, away from the warm AR traces
    ax.text(0.015, 0.0, "0 °C (rain/snow)", transform=ax.get_yaxis_transform(),
            color="0.25", fontsize=12, va="bottom", ha="left", zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=12, ncol=2,
                    framealpha=0.96, handlelength=1.5, columnspacing=1.0,
                    labelspacing=0.35, borderpad=0.4, borderaxespad=0.3)
    leg.set_zorder(20)


def main() -> int:
    paper_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    op = np.load(CACHE / "opera_corridor.npz")
    rivers = json.loads((CFG / "nhd_rivers.json").read_text())
    cc = json.loads((CFG / "cc_stations.json").read_text())
    uw = json.loads((CFG / "uw_stations.json").read_text())
    layers = json.loads((CFG / "map_layers.json").read_text())
    war = json.loads((CFG / "warm_ar_snotel.json").read_text())
    arf = CFG / "ar_windows.json"
    ar = json.loads(arf.read_text()) if arf.exists() else []

    fig = plt.figure(figsize=(13, 9))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.55, 1.0],
                           height_ratios=[1, 1, 1], wspace=0.13, hspace=0.30)
    axa = fig.add_subplot(gs[:, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 1])
    # split the bottom-right cell into (d) and (e) side by side
    gd = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2, 1],
                                          width_ratios=[0.62, 1.0], wspace=0.40)
    axd = fig.add_subplot(gd[0, 0])
    axe = fig.add_subplot(gd[0, 1])

    panel_a(axa, op, rivers, cc, uw, layers["gages"], layers["snotel"])
    panel_braid(axb, "puyallup", PR, "(b) Puyallup source — braided, compact",
                "#e31a1c")
    panel_braid(axc, "nisqually", {"LON": LON},
                "(c) Nisqually at UW.LON — braided, wide", "#1f78b4")
    panel_down(axd, rivers, "#006d2c")
    panel_snotel(axe, war, ar)

    out = FIGDIR / "figF1_studyarea.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
