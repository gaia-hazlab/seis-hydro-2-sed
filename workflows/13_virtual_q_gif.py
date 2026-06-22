#!/usr/bin/env python3
"""Space-time virtual-discharge animation: the distributed discharge field across
the western Rainier drainages through the Dec-2025 ARs, combining USGS gages
(diamonds) and SEISMIC virtual gages (triangles, Q inverted from P∝Q^b) on one
color scale. Demonstrates a seismic network as distributed, gap-tolerant
streamflow monitoring. Outputs paper/figures/virtual_q_animation.gif.

Usage: pixi run python workflows/13_virtual_q_gif.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource, LogNorm
import pygmt

ROOT = Path(__file__).resolve().parents[1]
GIF = ROOT / "paper" / "figures" / "virtual_q_animation.gif"
REGION = [-122.55, -121.55, 46.70, 47.34]
SUMMIT = (-121.7603, 46.8523)
WIN = (pd.Timestamp("2025-12-05", tz="UTC"), pd.Timestamp("2025-12-14", tz="UTC"))
STEP_H = 3
QMIN, QMAX = 5.0, 1400.0
AR_COLORS = {"pre-AR": "#999999", "AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}


def on_frames(frames, idx, val):
    s = pd.Series(val, index=pd.to_datetime(idx, utc=True)).sort_index()
    return s.reindex(s.index.union(frames)).interpolate("time").reindex(frames)


def main() -> int:
    frames = pd.date_range(WIN[0], WIN[1], freq=f"{STEP_H}h")
    aux = json.loads((ROOT / "config" / "aux_timeseries.json").read_text())
    virt = json.loads((ROOT / "config" / "virtual_q.json").read_text())
    ars = json.loads((ROOT / "config" / "ar_windows.json").read_text())
    coords = {f'CC.{r["sta"]}': r for r in json.loads((ROOT / "config" / "cc_stations.json").read_text())}
    coords.update({f'UW.{r["sta"]}': r for r in json.loads((ROOT / "config" / "uw_stations.json").read_text())})

    # gages (real Q) within the map
    gages = []
    for gid, d in aux["discharge"].items():
        if REGION[0] <= d["lon"] <= REGION[1] and REGION[2] <= d["lat"] <= REGION[3]:
            gages.append(dict(lon=d["lon"], lat=d["lat"], s=on_frames(frames, d["time"], d["q_cms"])))
    # seismic virtual gages — split valid (observed/marginal) vs no-signal
    sp = ROOT / "config" / "station_status.json"
    status = {s["station"]: s["status"] for s in json.loads(sp.read_text())} if sp.exists() else {}
    vg, vg_nosig = [], []
    for sid, v in virt.items():
        if sid not in coords:
            continue
        rec = dict(sid=sid, lon=coords[sid]["lon"], lat=coords[sid]["lat"],
                   s=on_frames(frames, v["time"], v["q_seis"]))
        (vg if status.get(sid, "observed") in ("observed", "marginal") else vg_nosig).append(rec)

    grid = pygmt.datasets.load_earth_relief(resolution="15s", region=REGION)
    hs = LightSource(azdeg=300, altdeg=35).hillshade(np.asarray(grid), vert_exag=2, dx=400, dy=400)
    rivers = json.loads((ROOT / "config" / "nhd_rivers.json").read_text())
    river_segs = [np.array(seg)[:, :2] for segs in rivers.values() for seg in segs]
    cmap = mpl.cm.turbo
    norm = LogNorm(QMIN, QMAX)

    fig = plt.figure(figsize=(7.8, 8.2))
    axm = fig.add_axes([0.06, 0.30, 0.88, 0.66])
    axq = fig.add_axes([0.10, 0.07, 0.80, 0.16])
    axm.imshow(hs, cmap="gray", extent=REGION, origin="lower", vmin=0, vmax=1, aspect=1.45, zorder=0)
    river_lc = LineCollection(river_segs, colors="#1f6fb2", linewidths=1.6, alpha=0.85, zorder=1)
    axm.add_collection(river_lc)
    axm.plot(*SUMMIT, marker="^", ms=11, color="firebrick", mec="k", zorder=5)
    axm.text(SUMMIT[0], SUMMIT[1] - 0.03, "Mt. Rainier", ha="center", fontsize=8, fontweight="bold")
    for v in vg + vg_nosig:
        axm.text(v["lon"], v["lat"] + 0.012, v["sid"].split(".")[1], ha="center", fontsize=6, zorder=6)
    # no-signal stations: hollow grey triangles (sensor present, no usable estimate)
    if vg_nosig:
        axm.scatter([v["lon"] for v in vg_nosig], [v["lat"] for v in vg_nosig], s=140,
                    marker="^", facecolor="none", edgecolor="0.4", lw=1.6, zorder=4)
    axm.set_xlim(*REGION[:2]); axm.set_ylim(*REGION[2:]); axm.set_xticks([]); axm.set_yticks([])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axm, fraction=0.035, pad=0.01)
    cb.set_label("discharge (m³ s⁻¹)")
    axm.scatter([], [], marker="D", c="gray", edgecolor="k", label="USGS gage (measured Q)")
    axm.scatter([], [], marker="^", c="gray", edgecolor="k", label="seismic virtual gage (Q ∝ P$^{1/b}$)")
    axm.scatter([], [], marker="^", facecolor="none", edgecolor="0.4", lw=1.6, label="no usable signal")
    axm.legend(loc="lower left", fontsize=7, framealpha=0.9)

    # discharge context panel
    qref = next((g["s"] for g in gages), vg[0]["s"])
    for g in gages:
        axq.semilogy(frames, g["s"].values, color="0.6", lw=0.8)
    axq.set_ylabel("gage Q (m³/s)", fontsize=8); axq.set_xlim(frames[0], frames[-1])
    axq.set_xlabel("December 2025 (UTC)", fontsize=8)
    for w in ars:
        axq.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                    color=AR_COLORS.get(w["label"], "#999"), alpha=0.15, zorder=0)
    cur = axq.axvline(frames[0], color="red", lw=1.3)

    glon = np.array([g["lon"] for g in gages]); glat = np.array([g["lat"] for g in gages])
    vlon = np.array([v["lon"] for v in vg]); vlat = np.array([v["lat"] for v in vg])
    sc_g = axm.scatter(glon, glat, s=90, marker="D", c=np.full(len(gages), QMIN),
                       cmap=cmap, norm=norm, edgecolor="k", lw=0.7, zorder=4)
    sc_v = axm.scatter(vlon, vlat, s=150, marker="^", c=np.full(len(vg), QMIN),
                       cmap=cmap, norm=norm, edgecolor="k", lw=0.9, zorder=4)
    qn = ((qref - qref.min()) / (qref.max() - qref.min() + 1e-9)).fillna(0).values
    title = axm.set_title("", loc="left", fontsize=12, fontweight="bold")

    def update(i):
        t = frames[i]
        sc_g.set_array(np.array([np.clip(g["s"].iloc[i], QMIN, QMAX) for g in gages]))
        sc_v.set_array(np.array([np.clip(v["s"].iloc[i], QMIN, QMAX) for v in vg]))
        river_lc.set_linewidths(1.4 + 5.0 * qn[i])
        cur.set_xdata([t, t])
        lab = next((w["label"] for w in ars
                    if pd.Timestamp(w["start"]) <= t < pd.Timestamp(w["end"])), "")
        title.set_text(f"Virtual + measured discharge — {t:%Y-%m-%d %H:%M} UTC   {lab}")
        return sc_g, sc_v, river_lc, cur, title

    FuncAnimation(fig, update, frames=len(frames), interval=200, blit=False).save(
        GIF, writer=PillowWriter(fps=6), dpi=80)
    try:
        from PIL import Image, ImageSequence
        im = Image.open(GIF)
        fr = [f.copy().convert("P", palette=Image.ADAPTIVE, colors=128) for f in ImageSequence.Iterator(im)]
        fr[0].save(GIF, save_all=True, append_images=fr[1:], loop=0,
                   duration=im.info.get("duration", 170), optimize=True)
    except Exception as e:
        print("optimize skipped:", e)
    plt.close(fig)
    print(f"wrote {GIF} ({len(frames)} frames; {len(gages)} gages + {len(vg)} virtual gages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
