#!/usr/bin/env python3
"""Animated map of bedload strength at each station through the three Dec-2025 ARs.

Each station marker is color- and size-coded by its instantaneous bedload-band
(5–15 Hz) power relative to its pre-flood median, animated through the flood.
A discharge strip below shows the three AR pulses with a moving time cursor.
Outputs paper/figures/bedload_animation.gif.

Usage: pixi run python workflows/06_bedload_gif.py
"""
from __future__ import annotations

import json
import re
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LightSource

import pygmt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "data" / "results"
GIF = ROOT / "paper" / "figures" / "bedload_animation.gif"
EXCLUDE = {"UW.BHW", "UW.TEHA"}
REGION = [-122.55, -121.55, 46.72, 47.34]
SUMMIT = (-121.7603, 46.8523)
PREFLOOD_END = pd.Timestamp("2025-12-08", tz="UTC")
WIN = (pd.Timestamp("2025-12-07T12", tz="UTC"), pd.Timestamp("2025-12-13T12", tz="UTC"))
STEP_H = 2
ARS = [  # (start, end, label)
    ("2025-12-07T21", "2025-12-10T03", "AR1"),
    ("2025-12-10T03", "2025-12-11T10", "AR2"),
    ("2025-12-11T10", "2025-12-13T04", "AR3"),
]
CLIP = (0.0, 2.5)  # log10(power/median): 1x .. ~300x
HF_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)_5\.0-15\.0Hz_timeseries\.csv$")


def hav(lon1, lat1, lon2, lat2):
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    return 2 * 6371.0 * asin(sqrt(sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2))


def main() -> int:
    coords = {}
    disc = ROOT / "config" / "_transect_discovery.json"
    for v in json.loads(disc.read_text()).get("stations", []):
        coords[f'{v["net"]}.{v["sta"]}'] = (v["lon"], v["lat"])

    frames = pd.date_range(WIN[0], WIN[1], freq=f"{STEP_H}h")
    sta = {}
    q = None
    for f in sorted(RESULTS.glob("*_5.0-15.0Hz_timeseries.csv")):
        m = HF_RE.match(f.name)
        sid = f'{m["net"]}.{m["sta"]}' if m else None
        if not sid or sid in EXCLUDE or sid not in coords:
            continue
        df = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc")
        P = pd.to_numeric(df["proxy"], errors="coerce").dropna()
        base = P[P.index < PREFLOOD_END].median() or P.median()
        norm = np.log10((P / base).clip(lower=0.1))
        sta[sid] = dict(lon=coords[sid][0], lat=coords[sid][1],
                        s=norm.reindex(norm.index.union(frames)).interpolate("time").reindex(frames))
        if q is None:
            q = pd.to_numeric(df["gauge"], errors="coerce").dropna().reindex(
                norm.index.union(frames)).interpolate("time").reindex(frames)

    # hillshade background
    grid = pygmt.datasets.load_earth_relief(resolution="15s", region=REGION)
    z = np.asarray(grid)
    ls = LightSource(azdeg=300, altdeg=35)
    hs = ls.hillshade(z, vert_exag=2, dx=400, dy=400)

    rivers = json.loads((ROOT / "config" / "nhd_rivers.json").read_text())
    cmap = mpl.cm.turbo
    norm_c = mpl.colors.Normalize(*CLIP)

    fig = plt.figure(figsize=(7.5, 8.2))
    axm = fig.add_axes([0.08, 0.30, 0.86, 0.66])
    axq = fig.add_axes([0.08, 0.08, 0.86, 0.15])

    axm.imshow(hs, cmap="gray", extent=REGION, origin="lower", vmin=0, vmax=1, aspect=1.45, zorder=0)
    for segs in rivers.values():
        for seg in segs:
            a = np.array(seg)
            axm.plot(a[:, 0], a[:, 1], color="#3b6fb0", lw=0.7, alpha=0.8, zorder=1)
    axm.plot(*SUMMIT, marker="^", ms=11, color="firebrick", mec="k", zorder=2)
    axm.text(SUMMIT[0], SUMMIT[1] - 0.03, "Mt. Rainier", ha="center", fontsize=8, fontweight="bold")
    for sid, d in sta.items():
        axm.text(d["lon"], d["lat"] + 0.012, sid.split(".")[1], ha="center", fontsize=6.5, zorder=4)
    axm.set_xlim(REGION[0], REGION[1]); axm.set_ylim(REGION[2], REGION[3])
    axm.set_xlabel("longitude"); axm.set_ylabel("latitude")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_c, cmap=cmap), ax=axm,
                      fraction=0.04, pad=0.02)
    cb.set_label(r"bedload strength  $\log_{10}$(power / pre-flood median)")

    # discharge strip
    axq.plot(frames, q.values, color="k", lw=1.2)
    for s0, s1, lab in ARS:
        axq.axvspan(pd.Timestamp(s0, tz="UTC"), pd.Timestamp(s1, tz="UTC"), color="tab:blue", alpha=0.13)
        axq.text(pd.Timestamp(s0, tz="UTC") + (pd.Timestamp(s1, tz="UTC") - pd.Timestamp(s0, tz="UTC")) / 2,
                 q.max() * 0.9, lab, ha="center", fontsize=9, fontweight="bold")
    axq.set_ylabel("Q (m³ s⁻¹)"); axq.set_xlabel("December 2025 (UTC)")
    axq.set_xlim(frames[0], frames[-1])
    cursor = axq.axvline(frames[0], color="red", lw=1.5)

    scat = axm.scatter([], [], s=[], c=[], cmap=cmap, norm=norm_c,
                       edgecolor="k", linewidth=0.6, zorder=3)
    lons = np.array([d["lon"] for d in sta.values()])
    lats = np.array([d["lat"] for d in sta.values()])
    title = axm.set_title("", loc="left", fontsize=11, fontweight="bold")

    def update(i):
        t = frames[i]
        vals = np.array([np.clip(d["s"].iloc[i], *CLIP) if np.isfinite(d["s"].iloc[i]) else CLIP[0]
                         for d in sta.values()])
        scat.set_offsets(np.column_stack([lons, lats]))
        scat.set_array(vals)
        scat.set_sizes(60 + 240 * (vals - CLIP[0]) / (CLIP[1] - CLIP[0]))
        cursor.set_xdata([t, t])
        lab = next((l for s0, s1, l in ARS if pd.Timestamp(s0, tz="UTC") <= t < pd.Timestamp(s1, tz="UTC")), "")
        title.set_text(f"Bedload strength  {t:%Y-%m-%d %H:%M} UTC   {lab}")
        return scat, cursor, title

    anim = FuncAnimation(fig, update, frames=len(frames), interval=180, blit=False)
    anim.save(GIF, writer=PillowWriter(fps=7))
    plt.close(fig)
    print(f"wrote {GIF}  ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
