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


SPEC = RESULTS / "spectra_psd.npz"
# (station id, colour, sample rate) for the two contrasting stations
SPEC_STA = [("CC.PR03", "#0072B2", 50), ("UW.UPS", "#D55E00", 200)]


def panel_a(ax) -> None:
    """Flood-vs-quiet ground-velocity PSD, re-plotted from the cached spectra
    (notebooks/data/results/spectra_psd.npz, written by workflows/11_spectra.py).
    Falls back to the committed fig11_spectra.png if the cache is absent."""
    if not SPEC.exists():
        ax.imshow(plt.imread(str(FIGDIR / "fig11_spectra.png")))
        ax.axis("off")
        ax.set_title("(a) Flood vs quiet spectra", loc="left", fontsize=14)
        return
    z = np.load(SPEC)
    ax.axvspan(1, 20, color="#0072B2", alpha=0.08, zorder=0)
    ax.axvspan(30, 80, color="#E69F00", alpha=0.14, zorder=0)
    for sid, c, sps in SPEC_STA:
        for day, ls, lab, al, lw in [("20251203", ":", "quiet", 0.75, 1.4),
                                     ("20251210", "-", "flood", 0.95, 2.0)]:
            fk, pk = f"{sid}_{day}_f", f"{sid}_{day}_p"
            if fk not in z.files:
                continue
            ax.semilogx(z[fk], z[pk], ls, color=c, lw=lw, alpha=al,
                        label=f"{sid} ({sps} sps) — {lab}")
        ax.axvline(sps / 2, color=c, ls="--", lw=0.9, alpha=0.55)   # Nyquist
    ax.set_xlim(0.5, 100)
    ax.set_ylim(-175, -100)                          # focus on the signal band
    ax.set_xlabel("frequency (Hz)", fontsize=13)
    ax.set_ylabel("velocity PSD (dB re m² s⁻² Hz⁻¹)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.text(6, -102, "turbulence\n1–20 Hz", ha="center", va="top",
            fontsize=11.5, color="#0072B2")
    ax.text(49, -102, "bedload\n30–80 Hz", ha="center", va="top",
            fontsize=11.5, color="#a8780a")
    ax.set_title("(a) Flood vs quiet spectra", loc="left", fontsize=14)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.92,
              borderpad=0.4, labelspacing=0.3)
    ax.grid(alpha=0.25, which="both")


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
    ax_d.set_ylabel("discharge\n(m³ s⁻¹)", fontsize=13)
    ax_d.tick_params(labelsize=12)
    ax_d.set_title("(b) 5–15 Hz bedload proxy", loc="left", fontsize=14)
    # Give the AR labels a dedicated band of headroom ABOVE all discharge data so
    # they never sit on the peaks; stagger two heights so adjacent labels (which
    # can be close in time) never touch each other.
    qtop = float(np.nanmax(q.values))
    ax_d.set_ylim(top=qtop * 1.55)  # reserve the upper third for labels
    ylo = qtop * 1.18
    yhi = qtop * 1.40
    yfrac = {"pre-AR": ylo, "AR1": yhi, "AR2": ylo, "AR3": yhi}
    for (s0, pk, s1, lab) in ars:
        for a in (ax_d, ax_b):
            a.axvspan(s0, s1, color=AR_COLORS.get(lab, "#999999"), alpha=0.22, zorder=0)
        ax_d.text(pk, yfrac.get(lab, ylo), lab, ha="center", va="bottom",
                  fontsize=12, fontweight="semibold", clip_on=False)

    for s in stations:
        ax_b.semilogy(data[s]["nrm"].index, data[s]["nrm"].values, lw=1.3, color=col[s],
                      label=f"{s} ({data[s]['dist']:.0f} km)")
    ax_b.axhline(1.0, color="0.5", ls=":", lw=1)
    ymax = max(np.nanmax(data[s]["nrm"].values) for s in stations)
    ax_b.set_ylim(0.5, ymax * 1.5)            # legend now lives below panel (a)
    ax_b.set_ylabel("5–15 Hz power /\npre-flood median", fontsize=13)
    ax_b.set_xlabel("December 2025 (UTC)", fontsize=13)
    ax_b.tick_params(labelsize=12)
    handles, labels = ax_b.get_legend_handles_labels()   # drawn below panel (a) in main()

    # per-AR means for reporting
    per_ar = {}
    for s in stations:
        rec = {}
        for (s0, pk, s1, lab) in ars:
            seg = data[s]["nrm"][(data[s]["nrm"].index >= s0) & (data[s]["nrm"].index < s1)]
            rec[lab] = round(float(seg.mean()), 2) if len(seg) else None
        per_ar[s] = (data[s]["dist"], rec)
    return per_ar, handles, labels


def main() -> int:
    fig = plt.figure(figsize=(13, 6.2))
    # right column = (b) discharge strip + proxy; left column = (a) spectra on top
    # with panel (b)'s station legend parked in a strip BELOW it (declutters (b)).
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], height_ratios=[1, 2.1],
                          hspace=0.08, wspace=0.16)
    left = gs[:, 0].subgridspec(6, 1, hspace=0.0)
    ax_a = fig.add_subplot(left[0:5, 0])
    ax_leg = fig.add_subplot(left[5, 0]); ax_leg.axis("off")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1, 1], sharex=ax_d)
    plt.setp(ax_d.get_xticklabels(), visible=False)

    panel_a(ax_a)
    per_ar, handles, labels = panel_b(ax_d, ax_b)
    ax_leg.legend(handles, labels, ncol=2, loc="center", fontsize=11,
                  framealpha=0, columnspacing=1.3, handlelength=1.7, borderpad=0.2,
                  title="5–15 Hz proxy — station (km from summit)", title_fontsize=11)

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
