#!/usr/bin/env python3
r"""Composite figure F3 — "Transport-onset threshold and rating geometry."

Merges three previously separate diagnostics into one paper figure:

  (a) Two-regime broken-stick P–Q fits with a transport-onset threshold Qc per
      station (old fig14 / workflows/15_threshold.py). Below Qc the 5–15 Hz
      seismic power tracks turbulent flow (slope b_below); above Qc a steeper
      contribution switches on (slope b_above). Cached fit parameters live in
      config/threshold_qc.json; the log-log scatter is overlaid from the cached
      per-station timeseries CSVs when present (the panel still renders fit-only
      if those are absent, so it rebuilds fully offline from JSON).

  (b) Stage–discharge rating geometry (old fig26 / workflows/25_rating_geometry).
      Local rating exponent beta(Q)=dlogQ/dlog(h-h0) vs discharge for four USGS
      gages. The confined source gages (Electron, National) steepen with Q and the
      seismic Qc (dotted) lands in the beta-steepening band; the lowland
      Puyallup-at-Puyallup gage stays low (overbank). From config/rating_geometry.json.

  (c) Station-skill summary: P–Q correlation r per station ordered source→downstream,
      coloured by usability (usable / marginal / no-signal). The "where river
      seismology works" panel. From config/station_status.json.

Rebuilds OFFLINE from cached JSON (+ optional cached CSVs).

Usage: pixi run python workflows/34_figF3_threshold.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

CONFIG = ROOT / "config"
RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
OUT = FIGDIR / "figF3_threshold.png"

# Okabe-Ito-ish palette reused for the panel-(a) station set
PANEL_A_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# Status -> colour / marker for the skill panel (c)
STATUS_STYLE = {
    "observed": ("#009E73", "usable"),
    "marginal": ("#E69F00", "marginal"),
    "none":     ("#D55E00", "no signal"),
    "control":  ("0.6", "control"),
}


def load_scatter(sid: str):
    """Cached log-log (logQ, logP) points for the flood window, mirroring fig14.

    Returns (None, None) when the cached timeseries CSV is absent so the panel
    can still render fit-only offline.
    """
    f = RESULTS / f"{sid}_5.0-15.0Hz_timeseries.csv"
    if not f.exists():
        return None, None
    try:
        from riverseis.analysis import clip_event
        df = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc")
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[df.index.notna()]
        P = pd.to_numeric(df["proxy"], errors="coerce")
        Q = pd.to_numeric(df["gauge"], errors="coerce")
        j = pd.concat([P.rename("P"), Q.rename("Q")], axis=1).sort_index()
        j["Q"] = j["Q"].interpolate("linear", limit=12)
        j = clip_event(j.dropna())
        lp = np.log10(j["P"].clip(lower=1e-30))
        lq = np.log10(j["Q"].clip(lower=1e-6))
        med = lp.median()
        mad = 1.4826 * (lp - med).abs().median()
        k = (lp - med).abs() < 6 * max(mad, 1e-9)
        return lq[k].values, lp[k].values
    except Exception:
        return None, None


def panel_a(ax, thr):
    """Broken-stick two-regime P–Q fits + Qc for the key onset/source stations."""
    # the source/usable stations that best show the transport-onset break;
    # kept small so the log-log clouds + fits stay legible.
    want = ["UW.LON", "CC.PR01", "CC.GTWY", "CC.TRON"]
    sel = [s for s in want if s in thr]
    for i, sid in enumerate(sel):
        r = thr[sid]
        col = PANEL_A_COLORS[i % len(PANEL_A_COLORS)]
        x0 = np.log10(r["Qc_cms"])
        b1, b2 = r["b_below"], r["b_above"]
        lq, lp = load_scatter(sid)
        if lq is not None and len(lq) > 10:
            # anchor the continuous fit to the below-Qc cloud, then vertically
            # offset each station so the four breaks read without overlapping.
            xs_lo = lq[lq <= x0]
            yref = float(np.median(lp[lq <= x0])) if xs_lo.size > 5 else float(np.median(lp))
            xref = float(np.median(xs_lo)) if xs_lo.size > 5 else float(np.median(lq))
            a = yref - b1 * xref
            off = 2.0 * i
            ax.scatter(lq, lp - a + off, s=2.5, alpha=0.10, color=col, rasterized=True)
            xspan = (lq.min(), lq.max())
        else:
            a, off = 0.0, 2.0 * i
            xspan = (x0 - 0.7, x0 + 0.7)
        xs = np.linspace(xspan[0], xspan[1], 120)
        yhat = b1 * xs + (b2 - b1) * np.maximum(0.0, xs - x0) + off  # intercept removed
        lbl = f"{sid}  b:{b1:+.1f}→{b2:+.1f}  ($Q_c\\!\\approx${r['Qc_cms']:.0f})"
        ax.plot(xs, yhat, "-", color=col, lw=2.2, label=lbl, zorder=5)
        ax.axvline(x0, color=col, ls=":", lw=1.2, alpha=0.8)
        ax.plot([x0], [b1 * x0 + off], "o", color=col, ms=6, mec="white",
                mew=0.8, zorder=6)

    ax.set_xlabel(r"$\log_{10} Q$  (m$^3$ s$^{-1}$)")
    ax.set_ylabel(r"$\log_{10} P$ (5–15 Hz), intercept-removed + offset")
    ax.set_title("(a) Two-regime P–Q fits", fontsize=14)
    ax.legend(fontsize=12, loc="upper left", handlelength=1.6, labelspacing=0.4,
              framealpha=0.93, borderpad=0.5)
    ax.margins(x=0.02)
    # extra headroom on top so the upper-left legend clears the highest-offset
    # scatter cloud instead of sitting on it
    ax.margins(y=0.10)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.30 * (y1 - y0))


def panel_b(ax, rg, thr):
    """Local rating exponent beta(Q) for the four gages + seismic Qc overlays."""
    order = ["12092000", "12082500", "12093500", "12101500"]
    colors = {"12092000": "#D55E00", "12082500": "#0072B2",
              "12093500": "#009E73", "12101500": "#6a3d9a"}
    # confinement / overbank reference band
    ax.axhspan(0, 0.5, color="0.85", alpha=0.6, zorder=0)

    for site in order:
        if site not in rg:
            continue
        d = rg[site]
        col = colors.get(site, "0.4")
        qs = np.array([float(k) for k in d["beta_by_Q"].keys()])
        bs = np.array([float(v) for v in d["beta_by_Q"].values()])
        o = np.argsort(qs)
        name = d["name"]
        ax.plot(qs[o], bs[o], "-o", color=col, lw=1.9, ms=5, label=name)
        if "seismic_Qc_cms" in d:
            qcv = d["seismic_Qc_cms"]
            ax.axvline(qcv, color=col, ls=":", lw=1.5)
            # stagger the two source labels so they don't collide with each
            # other, the fit lines, or the upper-left legend
            ytxt = 0.72 if site == "12092000" else 1.62
            ax.annotate(f"seismic $Q_c$\n{d.get('seismic_break','')}",
                        xy=(qcv, ytxt), xytext=(qcv * 0.80, ytxt),
                        color=col, fontsize=12, va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec=col, alpha=0.85, lw=0.7))

    ax.set_xscale("log")
    ax.set_ylim(0, 2.3)
    ax.text(ax.get_xlim()[0] * 1.15, 0.25, "overbank / flat section",
            fontsize=12, color="0.4", va="center")
    ax.set_xlabel(r"discharge $Q$  (m$^3$ s$^{-1}$)")
    ax.set_ylabel(r"local rating exponent $\beta=\mathrm{d}\log Q/\mathrm{d}\log(h-h_0)$")
    ax.set_title("(b) Rating geometry $\\beta(Q)$",
                 fontsize=14)
    ax.legend(fontsize=12, loc="upper left", handlelength=1.4, framealpha=0.92)


def panel_c(ax, status):
    """P–Q correlation r per station, ordered source→downstream, by usability."""
    # source -> downstream ordering (upstream Rainier sources first), drop pure
    # controls / NaN-r stations so the bars read as the usability summary.
    order = ["UW.LON", "CC.PR01", "CC.GTWY", "CC.PR02", "CC.PR03", "CC.STYX",
             "CC.TRON", "CC.SIFT", "CC.CARB", "UW.STOR", "UW.RER", "UW.UPS"]
    by = {s["station"]: s for s in status}
    rows = [by[s] for s in order if s in by and by[s].get("r") is not None]

    labels = [r["station"] for r in rows]
    rvals = [r["r"] for r in rows]
    cols = [STATUS_STYLE.get(r["status"], ("0.6", ""))[0] for r in rows]
    y = np.arange(len(rows))[::-1]  # source at top

    ax.barh(y, rvals, color=cols, edgecolor="0.3", lw=0.4, height=0.7)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel(r"P–Q correlation  $r$")
    ax.set_xlim(-0.5, 1.05)
    ax.set_title("(c) Station skill", fontsize=14)

    # usability legend
    seen, handles = [], []
    for st in ["observed", "marginal", "none"]:
        c, lab = STATUS_STYLE[st]
        if any(r["status"] == st for r in rows) and lab not in seen:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=c))
            seen.append(lab)
    ax.legend(handles, seen, fontsize=12, loc="lower right", title=None,
              framealpha=0.92)


def main() -> int:
    paper_style()
    thr = {r["station"]: r for r in json.loads((CONFIG / "threshold_qc.json").read_text())}
    rg = json.loads((CONFIG / "rating_geometry.json").read_text())
    status = json.loads((CONFIG / "station_status.json").read_text())

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.05, 0.75]})
    panel_a(axA, thr)
    panel_b(axB, rg, thr)
    panel_c(axC, status)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    plt.close(fig)

    # ---- console report --------------------------------------------------
    print(f"wrote {OUT}")
    src_qc = {"UW.LON (Nisqually nr National)": thr.get("UW.LON", {}).get("Qc_cms"),
              "CC.PR01 (Puyallup nr Electron)": thr.get("CC.PR01", {}).get("Qc_cms")}
    print("source-station Qc (cms):", src_qc)
    for site, d in rg.items():
        bs = [float(v) for v in d["beta_by_Q"].values()]
        tag = f" seismic Qc={d['seismic_Qc_cms']}" if "seismic_Qc_cms" in d else ""
        print(f"  {site} {d['name']:24s} beta {min(bs):.2f}->{max(bs):.2f}{tag}")
    usable = [s["station"] for s in status if s["status"] == "observed"]
    marginal = [s["station"] for s in status if s["status"] == "marginal"]
    print(f"usable (observed): {len(usable)} {usable}")
    print(f"marginal: {len(marginal)} {marginal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
