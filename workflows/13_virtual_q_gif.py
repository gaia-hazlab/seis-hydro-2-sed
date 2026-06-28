#!/usr/bin/env python3
"""Public-facing flood-monitoring animation: the live discharge field across the
western Mt. Rainier drainages during the December-2025 atmospheric rivers,
combining USGS river gauges (diamonds) with SEISMIC virtual gauges (triangles,
Q inverted from the calibrated P∝Q^b rating). The story for a general audience:
*seismometers fill the gaps between sparse river gauges*, a step toward earlier
flood awareness in this lahar-hazard corridor.

Design: a dark "monitoring dashboard" over a real Sentinel-2 true-color basemap
(cached, offline), with rivers that swell and glow as discharge rises, markers
that brighten/grow with discharge, the downstream at-risk communities labelled,
a live clock + atmospheric-river event chip, and a discharge ticker strip.

Outputs paper/figures/virtual_q_animation.gif. Rebuilds OFFLINE from the cached
basemap + config JSON (no network).

Usage: pixi run python workflows/13_virtual_q_gif.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style          # noqa: E402
from riverseis.basemap import load_basemap, has_basemap  # noqa: E402

GIF = ROOT / "paper" / "figures" / "virtual_q_animation.gif"
CONFIG = ROOT / "config"
SUMMIT = (-121.7603, 46.8523)
WIN = (pd.Timestamp("2025-12-05", tz="UTC"), pd.Timestamp("2025-12-14", tz="UTC"))
STEP_H = 3
QMIN, QMAX = 5.0, 1400.0

# dark dashboard palette
BG, PANEL, FG, MUTED, ACCENT = "#0c1016", "#161c26", "#e8eef6", "#8b97a8", "#ff5a5f"
AR_CHIP = {"pre-AR": "#5b6675", "AR1": "#3aa0ff", "AR2": "#7fd4ff", "AR3": "#ffb13a"}
# downstream at-risk communities in the Puyallup lahar/flood corridor
TOWNS = {"Orting": (-122.204, 47.098), "Sumner": (-122.241, 47.203),
         "Puyallup": (-122.293, 47.185)}
# flood-intensity colormap: calm blue -> cyan -> yellow -> orange -> deep red
FLOOD = LinearSegmentedColormap.from_list(
    "flood", ["#2c7fff", "#27d3e6", "#7CFC8A", "#ffe14d", "#ff8c1a", "#e01e1e"])


def on_frames(frames, idx, val):
    s = pd.Series(val, index=pd.to_datetime(idx, utc=True)).sort_index()
    return s.reindex(s.index.union(frames)).interpolate("time").reindex(frames)


def main() -> int:
    paper_style()
    frames = pd.date_range(WIN[0], WIN[1], freq=f"{STEP_H}h")
    aux = json.loads((CONFIG / "aux_timeseries.json").read_text())
    virt = json.loads((CONFIG / "virtual_q.json").read_text())
    ars = json.loads((CONFIG / "ar_windows.json").read_text())
    coords = {f'CC.{r["sta"]}': r for r in json.loads((CONFIG / "cc_stations.json").read_text())}
    coords.update({f'UW.{r["sta"]}': r for r in json.loads((CONFIG / "uw_stations.json").read_text())})

    if not has_basemap("anim"):
        print("no anim basemap — run: pixi run python workflows/28_fetch_basemaps.py --region anim")
        return 1
    bm = load_basemap("anim")
    ext = bm["extent"]                                   # [lon0, lon1, lat0, lat1]

    # gauges (measured Q) within the frame
    gages = []
    for gid, d in aux["discharge"].items():
        if ext[0] <= d["lon"] <= ext[1] and ext[2] <= d["lat"] <= ext[3]:
            gages.append(dict(lon=d["lon"], lat=d["lat"], s=on_frames(frames, d["time"], d["q_cms"])))
    # seismic virtual gauges — split usable vs no-signal
    sp = CONFIG / "station_status.json"
    status = {s["station"]: s["status"] for s in json.loads(sp.read_text())} if sp.exists() else {}
    vg, vg_nosig = [], []
    for sid, v in virt.items():
        if sid not in coords:
            continue
        rec = dict(sid=sid, lon=coords[sid]["lon"], lat=coords[sid]["lat"],
                   s=on_frames(frames, v["time"], v["q_seis"]))
        (vg if status.get(sid, "observed") in ("observed", "marginal") else vg_nosig).append(rec)

    rivers = json.loads((CONFIG / "nhd_rivers.json").read_text())
    river_segs = [np.array(seg)[:, :2] for segs in rivers.values() for seg in segs]
    norm = LogNorm(QMIN, QMAX)

    # ---------------- figure scaffold (dark dashboard) ----------------
    fig = plt.figure(figsize=(8.6, 10.0), facecolor=BG)
    axm = fig.add_axes([0.035, 0.275, 0.83, 0.595]); axm.set_facecolor(BG)
    cax = fig.add_axes([0.875, 0.30, 0.022, 0.52])
    axq = fig.add_axes([0.085, 0.075, 0.83, 0.135]); axq.set_facecolor(PANEL)

    # header
    fig.text(0.035, 0.955, "WESTERN MT. RAINIER  ·  LIVE DISCHARGE FIELD", color=FG,
             fontsize=20, fontweight="semibold", va="top")
    fig.text(0.035, 0.917, "Seismometers fill the gaps between river gauges — "
             "a step toward earlier flood awareness", color=MUTED, fontsize=12.5, va="top")
    fig.text(0.035, 0.892, "December 2025 atmospheric-river floods", color=AR_CHIP["AR2"],
             fontsize=12, va="top", style="italic")

    # basemap: brighten the dark forest a touch, then a light dark-wash so markers pop
    aspect = 1.0 / np.cos(np.radians(0.5 * (ext[2] + ext[3])))   # geographic aspect
    rgb_b = np.clip(bm["rgb"].astype(float) * 1.35 + 12, 0, 255).astype("uint8")
    axm.imshow(rgb_b, extent=ext, origin="upper", zorder=0, aspect=aspect)
    axm.imshow(np.zeros((2, 2, 4)) + np.array([0.02, 0.04, 0.07, 0.18]), extent=ext,
               origin="upper", zorder=0.5, aspect="auto")
    axm.set_aspect(aspect)
    # river glow (under) + crisp line (over); widths animate with discharge
    glow = LineCollection(river_segs, colors="#9fe6ff", linewidths=3.0, alpha=0.20, zorder=1)
    river_lc = LineCollection(river_segs, colors="#cdeeff", linewidths=1.3, alpha=0.9, zorder=1.4)
    axm.add_collection(glow); axm.add_collection(river_lc)

    # iconic summit + at-risk communities
    axm.plot(*SUMMIT, marker="^", ms=15, color="#ffffff", mec="k", mew=0.8, zorder=6)
    axm.text(SUMMIT[0], SUMMIT[1] - 0.028, "Mt. Rainier", ha="center", color=FG,
             fontsize=11, fontweight="semibold", zorder=6)
    for name, (lon, lat) in TOWNS.items():
        axm.scatter(lon, lat, s=70, marker="s", facecolor="#ffffff", edgecolor="k",
                    lw=0.7, zorder=6)
        axm.annotate(name, (lon, lat), xytext=(7, 4), textcoords="offset points",
                     color="#ffffff", fontsize=10.5, fontweight="semibold", zorder=7,
                     path_effects=_halo())
    axm.text(0.5, -0.015, "▼ downstream communities", transform=axm.transAxes,
             ha="center", va="top", color=MUTED, fontsize=10)

    # no-signal stations: hollow markers (sensor present, no usable estimate)
    if vg_nosig:
        axm.scatter([v["lon"] for v in vg_nosig], [v["lat"] for v in vg_nosig], s=110,
                    marker="^", facecolor="none", edgecolor=MUTED, lw=1.4, zorder=3)
    axm.set_xlim(ext[0], ext[1]); axm.set_ylim(ext[2], ext[3])
    axm.set_xticks([]); axm.set_yticks([])
    for sp_ in axm.spines.values():
        sp_.set_edgecolor("#2a3340")

    # colorbar
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=FLOOD), cax=cax)
    cb.set_label("river discharge  (m³ s⁻¹)", color=FG, fontsize=12)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=FG, labelsize=11)
    cb.outline.set_edgecolor("#2a3340")

    # animated marker layers: a soft glow behind + crisp marker on top
    glon = np.array([g["lon"] for g in gages]); glat = np.array([g["lat"] for g in gages])
    vlon = np.array([v["lon"] for v in vg]); vlat = np.array([v["lat"] for v in vg])
    g_glow = axm.scatter(glon, glat, s=320, marker="D", c=np.full(len(gages), QMIN),
                         cmap=FLOOD, norm=norm, alpha=0.28, zorder=3.4, edgecolor="none")
    v_glow = axm.scatter(vlon, vlat, s=420, marker="^", c=np.full(len(vg), QMIN),
                         cmap=FLOOD, norm=norm, alpha=0.28, zorder=3.4, edgecolor="none")
    sc_g = axm.scatter(glon, glat, s=95, marker="D", c=np.full(len(gages), QMIN),
                       cmap=FLOOD, norm=norm, edgecolor="white", lw=0.8, zorder=4)
    sc_v = axm.scatter(vlon, vlat, s=150, marker="^", c=np.full(len(vg), QMIN),
                       cmap=FLOOD, norm=norm, edgecolor="white", lw=0.9, zorder=4)

    # in-map legend (dark)
    leg = [mpl.lines.Line2D([], [], marker="D", color="none", mfc="#cccccc", mec="white",
                            ms=10, label="USGS gauge (measured)"),
           mpl.lines.Line2D([], [], marker="^", color="none", mfc="#cccccc", mec="white",
                            ms=12, label="seismic gauge (estimated)"),
           mpl.lines.Line2D([], [], marker="^", color="none", mfc="none", mec=MUTED,
                            ms=12, label="sensor — no usable signal"),
           mpl.lines.Line2D([], [], marker="s", color="none", mfc="white", mec="k",
                            ms=9, label="at-risk community")]
    lg = axm.legend(handles=leg, loc="lower left", fontsize=10.5, framealpha=0.92,
                    facecolor=PANEL, edgecolor="#2a3340", labelcolor=FG)
    lg.set_zorder(8)

    # live clock + AR event chip (top-right of the map)
    clock = axm.text(0.985, 0.975, "", transform=axm.transAxes, ha="right", va="top",
                     color=FG, fontsize=14, fontweight="semibold", zorder=9, fontfamily="monospace")
    chip = axm.text(0.985, 0.915, "", transform=axm.transAxes, ha="right", va="top",
                    color="#0c1016", fontsize=12, fontweight="semibold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc=AR_CHIP["pre-AR"], ec="none"))

    # ---------------- discharge ticker strip ----------------
    qref = next((g["s"] for g in gages), vg[0]["s"])
    for g in gages:
        axq.semilogy(frames, g["s"].values, color="#3d4b5c", lw=1.0)
    # highlight the downstream mainstem gauge (largest) in accent
    big = max(gages, key=lambda g: np.nanmax(g["s"].values)) if gages else None
    if big is not None:
        axq.semilogy(frames, big["s"].values, color=AR_CHIP["AR2"], lw=1.8)
    for w in ars:
        axq.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                    color=AR_CHIP.get(w["label"], "#5b6675"), alpha=0.16, zorder=0)
    axq.set_xlim(frames[0], frames[-1])
    axq.set_ylabel("gauge Q  (m³ s⁻¹)", color=FG, fontsize=11)
    axq.tick_params(colors=MUTED, labelsize=10)
    for s_ in axq.spines.values():
        s_.set_edgecolor("#2a3340")
    axq.set_title("downstream river gauges (log scale)", color=MUTED, fontsize=10.5,
                  loc="left", pad=3)
    cur = axq.axvline(frames[0], color=ACCENT, lw=1.8)

    qn = ((qref - qref.min()) / (qref.max() - qref.min() + 1e-9)).fillna(0).values

    def update(i):
        t = frames[i]
        gv = np.array([np.clip(g["s"].iloc[i], QMIN, QMAX) for g in gages])
        vv = np.array([np.clip(v["s"].iloc[i], QMIN, QMAX) for v in vg])
        for sc in (sc_g, g_glow):
            sc.set_array(gv)
        for sc in (sc_v, v_glow):
            sc.set_array(vv)
        # markers grow with discharge (normalized 0..1 on the log scale)
        fg = (np.log10(gv) - np.log10(QMIN)) / (np.log10(QMAX) - np.log10(QMIN))
        fv = (np.log10(vv) - np.log10(QMIN)) / (np.log10(QMAX) - np.log10(QMIN))
        sc_g.set_sizes(70 + 240 * fg); g_glow.set_sizes(220 + 900 * fg)
        sc_v.set_sizes(110 + 320 * fv); v_glow.set_sizes(300 + 1200 * fv)
        river_lc.set_linewidths(1.1 + 4.5 * qn[i])
        glow.set_linewidths(2.6 + 11 * qn[i])
        cur.set_xdata([t, t])
        clock.set_text(f"{t:%b %d  %H:%M} UTC")
        lab = next((w["label"] for w in ars
                    if pd.Timestamp(w["start"]) <= t < pd.Timestamp(w["end"])), "quiet")
        chip.set_text(f"  {lab}  ")
        chip.get_bbox_patch().set_facecolor(AR_CHIP.get(lab, "#5b6675"))
        return sc_g, sc_v, g_glow, v_glow, river_lc, glow, cur, clock, chip

    FuncAnimation(fig, update, frames=len(frames), interval=180, blit=False).save(
        GIF, writer=PillowWriter(fps=6), dpi=72)
    _optimize(GIF)
    plt.close(fig)
    print(f"wrote {GIF} ({len(frames)} frames; {len(gages)} gauges + {len(vg)} seismic gauges)")
    return 0


def _halo():
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=2.4, foreground="#0c1016")]


def _optimize(gif: Path) -> None:
    try:
        from PIL import Image, ImageSequence
        im = Image.open(gif)
        fr = [f.copy().convert("P", palette=Image.ADAPTIVE, colors=80)
              for f in ImageSequence.Iterator(im)]
        fr[0].save(gif, save_all=True, append_images=fr[1:], loop=0,
                   duration=im.info.get("duration", 170), optimize=True)
    except Exception as e:                       # noqa: BLE001
        print("optimize skipped:", e)


if __name__ == "__main__":
    raise SystemExit(main())
