#!/usr/bin/env python3
"""Domain of applicability for source-resolved river seismology (fig23).

Renders the §9 (REVIEW_2026) two-basin contrast as a figure: *why* the matched-Q
braided-reorganization diagnostic is clean at the Puyallup PR cluster and confounded
at the Nisqually (UW.LON/GTWY). Two panels:

  (A) the two dominant, fully-measured controls — station→gage distance (x) and
      active-channel width (y), both log — with the "clean domain" (co-located gage
      AND compact channel) shaded. A reach is readable only in the lower-left box;
      UW.LON fails on both axes. Marker size = elevation, colour = verdict, label = b.
  (B) a normalised factor heat-strip for ALL five stations across the four controls
      (gage distance, channel width, elevation→snow fraction, scaling exponent b);
      darker = more confounding. GTWY's channel width is not optically measured
      (its AOI was not processed) and is shown hatched, not invented.

Reads only committed products (config/braided_reorg_timing_*.json, the optical
configs); no network. Outputs paper/figures/fig23_domain.png.

Usage: pixi run python workflows/22_domain_panel.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"

# elevation (m) from notebooks/data/stations_by_basin_with_gages.csv
ELEV = {"CC.PR01": 648, "CC.PR02": 461, "CC.PR03": 523, "CC.GTWY": 617, "UW.LON": 853}
PEAK_Q = {"puyallup": 323, "nisqually": 425}        # m³/s, this event
CLEAN_MAX_KM, CLEAN_MAX_W = 3.0, 100.0              # the "readable" domain box


def optical_width(sid: str) -> float | None:
    """Pre-flood active-channel wet width (m) from the braid-optical configs."""
    for fn in ("braid_optical_change.json", "braid_optical_change_nisqually.json"):
        d = json.loads((CONFIG / fn).read_text())
        if sid in d.get("stations", {}):
            return float(d["stations"][sid].get("W_pre", float("nan")))
    return None


def load_station(sid: str, basin: str) -> dict:
    d = json.loads((CONFIG / f"braided_reorg_timing_{basin}.json").read_text())
    s = d["stations"][sid]
    if s["reversible"]:
        verdict = "reversible / unresolved"
    elif s.get("supply_confounded"):
        verdict = "supply-confounded"
    elif s["direction"] == "positive":
        verdict = "clean geometric step"
    else:
        verdict = "persistent (other)"
    return dict(sid=sid, basin=basin, b=s["b_pooled"],
                km=s["lag_correction"]["station_gage_km"],
                width=optical_width(sid), elev=ELEV[sid], verdict=verdict)


VCOLOR = {"clean geometric step": "#2ca02c", "supply-confounded": "#d62728",
          "reversible / unresolved": "#ff7f0e", "persistent (other)": "#9467bd"}


def main() -> int:
    paper_style()
    stations = ([load_station(s, "puyallup") for s in ("CC.PR01", "CC.PR02", "CC.PR03")]
                + [load_station(s, "nisqually") for s in ("UW.LON", "CC.GTWY")])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- Panel A: gage distance × channel width, clean domain shaded ----------
    axA.add_patch(Rectangle((0.1, 5), CLEAN_MAX_KM - 0.1, CLEAN_MAX_W - 5,
                            facecolor="#2ca02c", alpha=0.10, zorder=0))
    axA.text(0.16, 6.2, "readable domain\n(co-located gage +\ncompact channel)",
             fontsize=9, color="#1a701a", va="bottom")
    for st in stations:
        if st["width"] is None:
            continue
        axA.scatter(st["km"], st["width"], s=30 + (st["elev"] - 440) * 1.1,
                    c=VCOLOR[st["verdict"]], edgecolor="k", lw=0.8, zorder=5)
        axA.annotate(f"{st['sid'].split('.')[1]}\nb={st['b']}",
                     (st["km"], st["width"]), textcoords="offset points",
                     xytext=(8, -4), fontsize=9)
    # GTWY: width not optically measured -> show as a distance marker w/ up-arrow
    g = next(s for s in stations if s["sid"] == "CC.GTWY")
    axA.annotate("GTWY\n(width n/a)", (g["km"], 150), textcoords="offset points",
                 xytext=(6, 0), fontsize=9, color="#d62728")
    axA.scatter(g["km"], 150, marker="^", s=70, facecolor="none",
                edgecolor="#d62728", lw=1.4, zorder=5)
    axA.annotate("", xy=(g["km"], 230), xytext=(g["km"], 150),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2))
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("station → gage distance (km)")
    axA.set_ylabel("active-channel width (m, optical)")
    axA.set_xlim(0.15, 35); axA.set_ylim(10, 800)
    axA.set_title("(A) two dominant controls — only the lower-left is readable", fontsize=11)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="k",
                          label=v) for v, c in VCOLOR.items() if v != "persistent (other)"]
    axA.legend(handles=handles, fontsize=8.5, loc="lower right", title="reorg-timing verdict")

    # ---- Panel B: normalised confounding-factor heat-strip --------------------
    order = ["CC.PR03", "CC.PR02", "CC.PR01", "CC.GTWY", "UW.LON"]
    rows = {s["sid"]: s for s in stations}
    factors = ["gage dist", "width", "elevation", "b (steep)"]

    def norm(vals):
        v = np.array([np.nan if x is None else x for x in vals], float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        return (v - lo) / (hi - lo + 1e-9)

    M = np.vstack([
        norm([rows[s]["km"] for s in order]),
        norm([rows[s]["width"] for s in order]),
        norm([rows[s]["elev"] for s in order]),
        norm([rows[s]["b"] for s in order]),
    ]).T                                              # rows=stations, cols=factors

    axB.imshow(M, cmap="OrRd", vmin=0, vmax=1, aspect="auto")
    for i, s in enumerate(order):
        for j in range(len(factors)):
            if np.isnan(M[i, j]):
                axB.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, hatch="////",
                                        fill=False, edgecolor="0.5", lw=0))
                axB.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="0.4")
        v = rows[s]["verdict"]
        mark = "✓" if v == "clean geometric step" else "✗"
        axB.text(len(factors) - 0.5 + 0.7, i, mark, ha="center", va="center",
                 fontsize=15, color=VCOLOR[v])
    axB.set_xticks(range(len(factors))); axB.set_xticklabels(factors, fontsize=9.5)
    axB.set_yticks(range(len(order)))
    axB.set_yticklabels([s.split(".")[1] for s in order], fontsize=10)
    axB.set_xlim(-0.5, len(factors) + 0.4)
    axB.set_title("(B) confounding factors (darker = worse) → verdict", fontsize=11)
    for sp in axB.spines.values():
        sp.set_visible(False)
    axB.tick_params(length=0)

    fig.suptitle("Domain of applicability — Puyallup (clean) vs Nisqually (confounded)",
                 fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGDIR / "fig23_domain.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    for s in stations:
        w = "n/a" if s["width"] is None else f"{s['width']:.0f} m"
        print(f"  {s['sid']:8s} {s['basin']:9s} gage {s['km']:5.1f} km  width {w:>6}  "
              f"elev {s['elev']} m  b={s['b']}  -> {s['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
