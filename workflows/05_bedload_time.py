#!/usr/bin/env python3
"""Time-dependent bedload analysis across the three December-2025 atmospheric rivers.

Uses the high-frequency (5–15 Hz) bedload-band power as a bedload proxy
(P ∝ q_b·D³; Tsai 2012). For each clean station we normalize the proxy to its
pre-flood (Dec 1–7) median ("× above background"), detect the three AR discharge
pulses, and compute the per-AR average bedload. Outputs:

  fig6_bedload_time.png   discharge (with the 3 ARs shaded) + bedload time series
  fig7_bedload_per_AR.png per-AR mean bedload per station (source→downstream)
  config/bedload_per_AR.json

Usage: pixi run python workflows/05_bedload_time.py
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
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"

import sys; sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402
paper_style()
EXCLUDE = {"UW.BHW", "UW.TEHA"}
SUMMIT = (-121.7603, 46.8523)
PREFLOOD_END = pd.Timestamp("2025-12-08", tz="UTC")
HF_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)_5\.0-15\.0Hz_timeseries\.csv$")

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight"})


def hav(lon1, lat1, lon2, lat2):
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    return 2 * 6371.0 * asin(sqrt(sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2))


def load() -> dict:
    coords = {}
    disc = ROOT / "config" / "_transect_discovery.json"
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


def detect_ars(q: pd.Series) -> list[tuple]:
    """Return (start, peak, end, label) AR windows from the discharge series.

    The three largest hydrograph peaks are the main ARs (Dec 9/10/11); a weaker
    earlier peak (~Dec 6), below the main height filter but above baseline, is
    included as the 'pre-AR'. Inner boundaries are troughs between peaks; outer
    boundaries are the flood onset/recession (not the quiet baseline).
    """
    qs = q.resample("1h").median().interpolate(limit=12).dropna()
    v = qs.values
    mx = float(np.nanmax(v))
    pk, _ = find_peaks(v, distance=18, height=0.08 * mx)
    if len(pk) == 0:
        return []
    hr = lambda h: int(round(h))                                  # noqa: E731
    main = sorted(pk[np.argsort(v[pk])[::-1][:3]].tolist())       # 3 main ARs
    # pre-AR: largest peak before the main flood, below the main-AR height band
    pre = [p for p in pk if p < main[0] and v[p] < 0.4 * mx]
    pre_pk = max(pre, key=lambda p: v[p]) if pre else None

    wins = []
    if pre_pk is not None:
        end = min(pre_pk + hr(30), main[0] - hr(30))
        wins.append((qs.index[max(0, pre_pk - hr(28))], qs.index[pre_pk], qs.index[max(end, pre_pk + 1)], "pre-AR"))
    bounds = [max(0, main[0] - hr(30))]
    for a, b in zip(main[:-1], main[1:]):
        bounds.append(a + int(np.argmin(v[a:b])))
    bounds.append(min(len(v) - 1, main[-1] + hr(36)))
    for i, p in enumerate(main):
        wins.append((qs.index[bounds[i]], qs.index[p], qs.index[bounds[i + 1]], f"AR{i+1}"))
    return wins


# Colorblind-safe (Okabe–Ito) palette for AR shading: pre-AR grey, then 3 ARs.
AR_COLORS = {"pre-AR": "#999999", "AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}


def main() -> int:
    data = load()
    if not data:
        print("no clean HF series found")
        return 1
    stations = sorted(data, key=lambda s: data[s]["dist"])
    dmin, dmax = min(d["dist"] for d in data.values()), max(d["dist"] for d in data.values())
    norm = mpl.colors.Normalize(dmin, dmax)
    cmap = mpl.cm.viridis
    col = {s: cmap(norm(data[s]["dist"])) for s in stations}

    q = data[stations[0]]["Q"].dropna()
    for s in stations:  # prefer a source station's gage for AR detection
        if data[s]["dist"] == dmin:
            q = data[s]["Q"].dropna(); break
    ars = detect_ars(q)
    # persist AR windows so other workflows (b(t), GIF) stay consistent
    (ROOT / "config" / "ar_windows.json").write_text(json.dumps(
        [{"start": s0.isoformat(), "peak": pk.isoformat(), "end": s1.isoformat(), "label": lab}
         for (s0, pk, s1, lab) in ars], indent=2))

    # normalize each station's bedload proxy to its pre-flood median
    for s in stations:
        P = data[s]["P"]
        base = P[P.index < PREFLOOD_END].median()
        data[s]["norm"] = P / (base if base and np.isfinite(base) else P.median())

    # ---- fig6: discharge + bedload time series ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.6, 5.2), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 2.1], hspace=0.06))
    a1.plot(q.index, q.values, color="k", lw=1.5)
    a1.set_ylabel("discharge\n(m³ s⁻¹)", fontsize=12)
    a1.tick_params(labelsize=11)
    labels = [w[3] for w in ars]
    for (s0, pk, s1, lab) in ars:
        for a in (a1, a2):
            a.axvspan(s0, s1, color=AR_COLORS.get(lab, "#999999"), alpha=0.22, zorder=0)
        a1.text(pk, a1.get_ylim()[1] * 0.86, lab, ha="center", fontsize=11, fontweight="bold")
    for s in stations:
        a2.semilogy(data[s]["norm"].index, data[s]["norm"].values, lw=1.4, color=col[s],
                    label=f"{s} ({data[s]['dist']:.0f} km)")
    a2.axhline(1.0, color="0.5", ls=":", lw=1)
    # emphasize the AR-driven excursions: clip the empty low tail, pad just above peak
    ymax = max(np.nanmax(data[s]["norm"].values) for s in stations)
    a2.set_ylim(0.5, ymax * 1.4)
    a2.set_ylabel("5–15 Hz power / pre-flood median\n(turbulence + low-freq bedload)", fontsize=11)
    a2.set_xlabel("December 2025 (UTC)", fontsize=12)
    a2.tick_params(labelsize=11)
    # legend inside, over the quiet pre-flood (upper-left), away from the AR spikes
    a2.legend(fontsize=8.5, ncol=2, loc="upper left", framealpha=0.9, borderpad=0.4)
    fig.autofmt_xdate()
    fig.savefig(FIGDIR / "fig6_bedload_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- fig7: per-AR mean bedload per station ----
    rows = []
    for s in stations:
        rec = {"station": s, "dist_km": round(data[s]["dist"], 1)}
        for (s0, pk, s1, lab) in ars:
            seg = data[s]["norm"][(data[s]["norm"].index >= s0) & (data[s]["norm"].index < s1)]
            rec[lab] = round(float(seg.mean()), 2) if len(seg) else None
        rows.append(rec)
    (ROOT / "config" / "bedload_per_AR.json").write_text(json.dumps(rows, indent=2))

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(stations)); n = len(labels); w = 0.8 / n
    for i, lab in enumerate(labels):
        vals = [r.get(lab) or np.nan for r in rows]
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, color=AR_COLORS.get(lab, "#999999"),
               label=lab, edgecolor="0.3", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n{data[s]['dist']:.0f} km" for s in stations], fontsize=8)
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    ax.set_ylabel("mean bedload-band power / pre-flood median")
    ax.legend(title="AR pulse")
    fig.savefig(FIGDIR / "fig7_bedload_per_AR.png")
    plt.close(fig)

    print("AR windows:")
    for (s0, pk, s1, lab) in ars:
        print(f"  {lab:7s}: {s0:%m-%d %H:%M} … peak {pk:%m-%d %H:%M} … {s1:%m-%d %H:%M}")
    print("\nPer-AR mean bedload (×pre-flood median):")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {FIGDIR}/fig6_bedload_time.png, fig7_bedload_per_AR.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
