#!/usr/bin/env python3
"""Multidisciplinary animation of bedload transport through the Dec-2025 ARs.

Three stacked panels animate together:
  (top)    map: per-station bedload strength (5-15 Hz power vs pre-flood median),
           color+size coded; the traffic-contaminated UW.TEHA is shown as a grey
           control that does NOT light up (localization to the river network).
  (mid)    SNOTEL precipitation near Mt. Rainier (the meteorological driver).
  (bottom) USGS discharge at 5 gages down the corridor (the hydrologic response).
A moving cursor + pre-AR/AR1/AR2/AR3 shading tie the panels in time.

Outputs paper/figures/bedload_animation.gif.
Usage: pixi run python workflows/06_bedload_gif.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LightSource

import pygmt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "data" / "results"
GIF = ROOT / "paper" / "figures" / "bedload_animation.gif"
RIVER_CLEAN = {"UW.BHW"}          # excluded entirely
CONTROL = "UW.TEHA"               # shown as traffic control
REGION = [-122.55, -121.55, 46.72, 47.34]
SUMMIT = (-121.7603, 46.8523)
PREFLOOD_END = pd.Timestamp("2025-12-08", tz="UTC")
WIN = (pd.Timestamp("2025-12-05T00", tz="UTC"), pd.Timestamp("2025-12-14T00", tz="UTC"))
STEP_H = 3
CLIP = (0.0, 2.5)
HF_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)_5\.0-15\.0Hz_timeseries\.csv$")
AR_COLORS = {"pre-AR": "#999999", "AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}


def _series_on(frames, idx, val):
    s = pd.Series(val, index=pd.to_datetime(idx, utc=True)).sort_index()
    return s.reindex(s.index.union(frames)).interpolate("time").reindex(frames)


def main() -> int:
    coords = {f'{v["net"]}.{v["sta"]}': v for v in
              json.loads((ROOT / "config" / "_transect_discovery.json").read_text())["stations"]}
    aux = json.loads((ROOT / "config" / "aux_timeseries.json").read_text())
    ars = json.loads((ROOT / "config" / "ar_windows.json").read_text())
    frames = pd.date_range(WIN[0], WIN[1], freq=f"{STEP_H}h")

    # bedload strength per station (clean river + the traffic control)
    sta = {}
    for f in sorted(RESULTS.glob("*_5.0-15.0Hz_timeseries.csv")):
        m = HF_RE.match(f.name)
        sid = f'{m["net"]}.{m["sta"]}' if m else None
        if not sid or sid in RIVER_CLEAN or sid not in coords:
            continue
        df = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc")
        P = pd.to_numeric(df["proxy"], errors="coerce").dropna()
        base = P[P.index < PREFLOOD_END].median() or P.median()
        s = _series_on(frames, P.index, np.log10((P / base).clip(lower=0.1)).values)
        sta[sid] = dict(lon=coords[sid]["lon"], lat=coords[sid]["lat"],
                        s=s, control=(sid == CONTROL))

    precip = {k: _series_on(frames, v["time"], v["precip_mm"]) for k, v in aux["precip"].items()}
    disch = {k: dict(name=v["name"], s=_series_on(frames, v["time"], v["q_cms"]))
             for k, v in aux["discharge"].items()}

    # hillshade background
    grid = pygmt.datasets.load_earth_relief(resolution="15s", region=REGION)
    hs = LightSource(azdeg=300, altdeg=35).hillshade(np.asarray(grid), vert_exag=2, dx=400, dy=400)
    rivers = json.loads((ROOT / "config" / "nhd_rivers.json").read_text())
    cmap = mpl.cm.turbo
    norm = mpl.colors.Normalize(*CLIP)

    fig = plt.figure(figsize=(7.6, 9.4))
    axm = fig.add_axes([0.08, 0.40, 0.84, 0.56])
    axp = fig.add_axes([0.08, 0.27, 0.84, 0.10])
    axq = fig.add_axes([0.08, 0.07, 0.84, 0.15])

    axm.imshow(hs, cmap="gray", extent=REGION, origin="lower", vmin=0, vmax=1, aspect=1.45, zorder=0)
    # rivers as a LineCollection so width can swell with discharge through time
    river_segs = [np.array(seg)[:, :2] for segs in rivers.values() for seg in segs]
    river_lc = LineCollection(river_segs, colors="#1f6fb2", linewidths=1.8, alpha=0.9, zorder=1)
    axm.add_collection(river_lc)
    # USGS gage markers
    gvals = list(aux["discharge"].values())
    axm.scatter([d["lon"] for d in gvals], [d["lat"] for d in gvals],
                marker="D", s=48, c="cyan", edgecolor="k", lw=0.6, zorder=2.5, label="USGS gage")
    for d in gvals:
        axm.text(d["lon"], d["lat"], "  " + d["name"].split(" nr")[0].split(" at")[0],
                 fontsize=5.5, color="navy", va="center", ha="left", zorder=4)
    axm.plot(*SUMMIT, marker="^", ms=11, color="firebrick", mec="k", zorder=2)
    axm.text(SUMMIT[0], SUMMIT[1] - 0.03, "Mt. Rainier", ha="center", fontsize=8, fontweight="bold")
    for sid, d in sta.items():
        axm.text(d["lon"], d["lat"] + 0.012, sid.split(".")[1], ha="center",
                 fontsize=6.5, color=("0.35" if d["control"] else "k"), zorder=4)
    axm.set_xlim(*REGION[:2]); axm.set_ylim(*REGION[2:]); axm.set_xticks([]); axm.set_yticks([])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axm, fraction=0.035, pad=0.01)
    cb.set_label(r"bedload strength $\log_{10}$(P/median)")
    axm.scatter([], [], marker="s", c="0.6", edgecolor="k", label=f"{CONTROL} (traffic control)")
    axm.legend(loc="lower left", fontsize=6.5, framealpha=0.85)

    # precip panel
    for k, s in precip.items():
        axp.plot(frames, s.values, lw=1.0, label=k)
    axp.set_ylabel("precip\n(mm/h)", fontsize=8); axp.legend(fontsize=6, ncol=3, loc="upper left")
    axp.set_xlim(frames[0], frames[-1]); axp.set_xticklabels([])
    # discharge panel
    for gid, d in disch.items():
        axq.semilogy(frames, d["s"].values, lw=1.1, label=d["name"].replace("Puyallup", "Puy"))
    axq.set_ylabel("Q (m³/s)", fontsize=8); axq.legend(fontsize=5.5, ncol=2, loc="upper left")
    axq.set_xlim(frames[0], frames[-1]); axq.set_xlabel("December 2025 (UTC)", fontsize=8)

    for w in ars:
        s0, s1 = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
        for a in (axp, axq):
            a.axvspan(s0, s1, color=AR_COLORS.get(w["label"], "#999"), alpha=0.15, zorder=0)
    cur_p = axp.axvline(frames[0], color="red", lw=1.3)
    cur_q = axq.axvline(frames[0], color="red", lw=1.3)

    river_ids = [s for s in sta if not sta[s]["control"]]
    ctrl_ids = [s for s in sta if sta[s]["control"]]
    scat = axm.scatter([], [], s=[], c=[], cmap=cmap, norm=norm, edgecolor="k", lw=0.6, zorder=3)
    ctrl = axm.scatter([d["lon"] for s, d in sta.items() if d["control"]],
                       [d["lat"] for s, d in sta.items() if d["control"]],
                       s=70, marker="s", c="0.6", edgecolor="k", lw=0.6, zorder=3)
    lon = np.array([sta[s]["lon"] for s in river_ids]); lat = np.array([sta[s]["lat"] for s in river_ids])
    title = axm.set_title("", loc="left", fontsize=11, fontweight="bold")
    # normalized basin discharge drives the river "swell" (line width) through time
    qsw = disch["12101500"]["s"] if "12101500" in disch else next(iter(disch.values()))["s"]
    qn = ((qsw - qsw.min()) / (qsw.max() - qsw.min() + 1e-9)).fillna(0).values

    def update(i):
        t = frames[i]
        v = np.array([np.clip(sta[s]["s"].iloc[i], *CLIP) for s in river_ids])
        scat.set_offsets(np.column_stack([lon, lat])); scat.set_array(v)
        scat.set_sizes(50 + 230 * (v - CLIP[0]) / (CLIP[1] - CLIP[0]))
        river_lc.set_linewidths(1.6 + 5.5 * qn[i])      # rivers swell with discharge
        river_lc.set_alpha(0.6 + 0.4 * qn[i])
        cur_p.set_xdata([t, t]); cur_q.set_xdata([t, t])
        lab = next((w["label"] for w in ars
                    if pd.Timestamp(w["start"]) <= t < pd.Timestamp(w["end"])), "")
        title.set_text(f"Bedload transport — {t:%Y-%m-%d %H:%M} UTC   {lab}")
        return scat, river_lc, cur_p, cur_q, title

    FuncAnimation(fig, update, frames=len(frames), interval=200, blit=False).save(
        GIF, writer=PillowWriter(fps=6), dpi=80)
    plt.close(fig)
    # optimize: quantize frames to a shared 128-color palette to shrink the file
    try:
        from PIL import Image, ImageSequence
        im = Image.open(GIF)
        fr = [f.copy().convert("P", palette=Image.ADAPTIVE, colors=128)
              for f in ImageSequence.Iterator(im)]
        fr[0].save(GIF, save_all=True, append_images=fr[1:], loop=0,
                   duration=im.info.get("duration", 160), optimize=True)
    except Exception as e:
        print(f"  (gif optimize skipped: {e})")
    print(f"wrote {GIF} ({len(frames)} frames; {len(river_ids)} river + {len(ctrl_ids)} control)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
