#!/usr/bin/env python3
"""Composite F9 — "Bedload as a bounded hypothesis (the frequency evidence)."

Merges old fig11 (flood-vs-quiet ground-velocity spectra) and fig6 (5-15 Hz
bedload-band time series per atmospheric river) into one two-panel figure that
keeps the bedload claim HONEST: a bounded hypothesis, not a measurement.

  (a) Flood-vs-quiet PSD: a 50-sps source station (Nyquist 25 Hz) vs a 200-sps
      lowland station (Nyquist 100 Hz). The canonical 30-80 Hz bedload band is
      UNSAMPLED where the river is actually sensed (50 sps -> Nyquist 25 Hz).
      Built from cached spectra if available; otherwise the committed
      fig11_spectra.png is imshow-ed in (offline fallback).
  (b) 5-15 Hz bedload-proxy time series across the ARs (AR2 is the maximum;
      proxy decays downstream). Built from the results CSVs + AR windows.

Outputs paper/figures/figF9_bedload.png. Rebuilds OFFLINE.

Usage: pixi run python workflows/38_figF9_bedload.py
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

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
CONFIG = ROOT / "config"

import sys; sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402
paper_style()

EXCLUDE = {"UW.BHW", "UW.TEHA"}
SUMMIT = (-121.7603, 46.8523)
PREFLOOD_END = pd.Timestamp("2025-12-08", tz="UTC")
HF_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)_5\.0-15\.0Hz_timeseries\.csv$")
AR_COLORS = {"pre-AR": "#999999", "AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}


def hav(lon1, lat1, lon2, lat2):
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    return 2 * 6371.0 * asin(sqrt(sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2))


def load_hf() -> dict:
    coords = {}
    disc = CONFIG / "_transect_discovery.json"
    if disc.exists():
        for v in json.loads(disc.read_text()).get("stations", []):
            coords[f'{v["net"]}.{v["sta"]}'] = (v["lon"], v["lat"])
    out = {}
    for f in sorted(RESULTS.glob("*_5.0-15.0Hz_timeseries.csv")):
        m = HF_RE.match(f.name)
        if not m:
            continue
        sid = f'{m["net"]}.{m["sta"]}'
        if sid in EXCLUDE or sid not in coords:
            continue
        df = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc")
        P = pd.to_numeric(df["proxy"], errors="coerce").dropna()
        Q = pd.to_numeric(df["gauge"], errors="coerce")
        out[sid] = dict(P=P, Q=Q, dist=hav(*SUMMIT, *coords[sid]))
    return out


def load_ar_windows() -> list[tuple]:
    """Read the persisted AR windows (written by workflow 05) so the composite
    stays consistent with fig6 without re-detecting peaks here."""
    p = CONFIG / "ar_windows.json"
    if not p.exists():
        return []
    out = []
    for w in json.loads(p.read_text()):
        out.append((pd.Timestamp(w["start"]), pd.Timestamp(w["peak"]),
                    pd.Timestamp(w["end"]), w["label"]))
    return out


def panel_a(ax) -> None:
    """Flood-vs-quiet PSD. No cached spectra arrays exist offline, so we imshow
    the committed fig11_spectra.png. (If cached spectra were ever added, this is
    where they'd be re-plotted instead.)"""
    img = FIGDIR / "fig11_spectra.png"
    ax.imshow(plt.imread(str(img)))
    ax.axis("off")
    ax.set_title("(a) Flood vs quiet spectra — bedload band (30–80 Hz) unsampled "
                 "at 50 sps (Nyquist 25 Hz)", loc="left", fontsize=11)


def panel_b(ax_d, ax_b) -> dict:
    data = load_hf()
    stations = sorted(data, key=lambda s: data[s]["dist"])
    dmin = min(d["dist"] for d in data.values())
    dmax = max(d["dist"] for d in data.values())
    norm = mpl.colors.Normalize(dmin, dmax)
    cmap = mpl.cm.viridis
    col = {s: cmap(norm(data[s]["dist"])) for s in stations}

    # source-station gage for the discharge curve
    q = data[stations[0]]["Q"].dropna()
    for s in stations:
        if data[s]["dist"] == dmin:
            q = data[s]["Q"].dropna(); break

    ars = load_ar_windows()

    # normalize each station's proxy to its pre-flood median
    for s in stations:
        P = data[s]["P"]
        base = P[P.index < PREFLOOD_END].median()
        data[s]["nrm"] = P / (base if base and np.isfinite(base) else P.median())

    ax_d.plot(q.index, q.values, color="k", lw=1.4)
    ax_d.set_ylabel("discharge\n(m³ s⁻¹)", fontsize=11)
    ax_d.tick_params(labelsize=10)
    ax_d.set_title("(b) 5–15 Hz bedload proxy across the December-2025 atmospheric rivers",
                   loc="left", fontsize=11)
    ytop = ax_d.get_ylim()[1]
    # stagger label heights so the closely-spaced pre-AR/AR1/AR2/AR3 don't collide
    yfrac = {"pre-AR": 1.05, "AR1": 0.78, "AR2": 1.05, "AR3": 0.78}
    for (s0, pk, s1, lab) in ars:
        for a in (ax_d, ax_b):
            a.axvspan(s0, s1, color=AR_COLORS.get(lab, "#999999"), alpha=0.22, zorder=0)
        ax_d.text(pk, ytop * yfrac.get(lab, 0.9), lab, ha="center", va="bottom",
                  fontsize=9.5, fontweight="bold", clip_on=False)
    ax_d.set_ylim(top=ytop * 1.28)  # headroom for the raised labels

    for s in stations:
        ax_b.semilogy(data[s]["nrm"].index, data[s]["nrm"].values, lw=1.3, color=col[s],
                      label=f"{s} ({data[s]['dist']:.0f} km)")
    ax_b.axhline(1.0, color="0.5", ls=":", lw=1)
    ymax = max(np.nanmax(data[s]["nrm"].values) for s in stations)
    ax_b.set_ylim(0.5, ymax * 1.4)
    ax_b.set_ylabel("5–15 Hz power /\npre-flood median", fontsize=11)
    ax_b.set_xlabel("December 2025 (UTC)", fontsize=11)
    ax_b.tick_params(labelsize=10)
    ax_b.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9, borderpad=0.4)

    # per-AR means for reporting
    per_ar = {}
    for s in stations:
        rec = {}
        for (s0, pk, s1, lab) in ars:
            seg = data[s]["nrm"][(data[s]["nrm"].index >= s0) & (data[s]["nrm"].index < s1)]
            rec[lab] = round(float(seg.mean()), 2) if len(seg) else None
        per_ar[s] = (data[s]["dist"], rec)
    return per_ar


def main() -> int:
    fig = plt.figure(figsize=(13, 6))
    # (a) spans full height on the left; (b) is a stacked discharge+proxy pair on the right
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1, 2.1],
                          hspace=0.08, wspace=0.14)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 1], sharex=ax_d)
    plt.setp(ax_d.get_xticklabels(), visible=False)

    panel_a(ax_a)
    per_ar = panel_b(ax_d, ax_b)

    for lab in ax_b.get_xticklabels():
        lab.set_rotation(25); lab.set_ha("right")

    fig.savefig(FIGDIR / "figF9_bedload.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- report ----
    print(f"wrote {FIGDIR / 'figF9_bedload.png'}")
    print("Nyquist: 50 sps -> 25 Hz (bedload band 30-80 Hz UNSAMPLED); 200 sps -> 100 Hz")
    print("\nPer-AR mean 5-15 Hz proxy (x pre-flood median), source -> downstream:")
    for s, (dist, rec) in sorted(per_ar.items(), key=lambda kv: kv[1][0]):
        cells = "  ".join(f"{k}={v}" for k, v in rec.items())
        print(f"  {s:9s} {dist:5.1f} km  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
