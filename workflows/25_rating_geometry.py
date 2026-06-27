#!/usr/bin/env python3
r"""Stage–discharge rating geometry: confinement vs overbank, and the seismic break (M2).

Co-author (D. Montgomery) question: "Find out the stage vs discharge wherever we can
to identify overflow and change in geometry." Independent of the seismic data, the
USGS stage–discharge rating at each gage records how the *channel cross-section*
conveys flow: the local rating exponent $\beta(Q)=\mathrm{d}\log Q/\mathrm{d}\log(h-h_0)$
(from $Q=C\,(h-h_0)^{\beta}$) rises where the section is confined and deepening, and
*flattens* where flow spreads overbank.

We find: at the confined upstream source gages $\beta$ climbs steeply with discharge
(Puyallup nr Electron 0.1→2.0; Nisqually nr National 0.2→1.6), and the
**seismic $P$–$Q$ break $Q_c$ falls within the steepening band** — independent
hydraulic corroboration that a *geometry* transition occurs near that discharge,
consistent with the geometric (not bed-mechanical) reading of the break. The
downstream Puyallup-at-Puyallup gage instead stays *flat* ($\beta\lesssim0.8$) — the
signature of overbank spreading on the lowland floodplain.

Outputs paper/figures/fig26_rating_geometry.png + config/rating_geometry.json.

Usage: pixi run python workflows/25_rating_geometry.py
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

DATA = ROOT / "notebooks" / "data"
CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"
# gage, label, file-site, seismic station co-located (for Qc overlay), colour
GAGES = [
    ("12092000", "Puyallup nr Electron\n(confined source)", "CC.PR01", "#e31a1c"),
    ("12082500", "Nisqually nr National\n(source)", "UW.LON", "#1f78b4"),
    ("12093500", "Puyallup nr Orting\n(mid)", None, "#33a02c"),
    ("12101500", "Puyallup at Puyallup\n(lowland, overbank)", None, "#6a3d9a"),
]


def load_rating(site: str):
    f = DATA / f"usgs_iv_{site}_2025-12-01_2026-01-01.csv"
    d = pd.read_csv(f)
    Q = pd.to_numeric(d["discharge_cfs"], errors="coerce") * 0.0283168
    H = pd.to_numeric(d["gage_height_ft"], errors="coerce") * 0.3048
    m = Q.notna() & H.notna() & (Q > 0)
    Q, H = Q[m].values, H[m].values
    h0 = H.min() - 0.01
    return Q, H, h0


def local_beta(Q, H, h0, nbin=8):
    """Local rating exponent in log-spaced discharge bins: (Qmid, beta)."""
    edges = np.geomspace(max(Q.min(), 1), Q.max(), nbin + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (Q >= lo) & (Q < hi)
        if sel.sum() > 25:
            b = np.polyfit(np.log10(H[sel] - h0), np.log10(Q[sel]), 1)[0]
            out.append((np.sqrt(lo * hi), b))
    return np.array(out).T if out else (np.array([]), np.array([]))


def main() -> int:
    paper_style()
    qc = {r["station"]: r for r in json.loads((CONFIG / "threshold_qc.json").read_text())}
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    out = {}

    for site, label, sta, col in GAGES:
        Q, H, h0 = load_rating(site)
        name = label.split("\n")[0]
        # Panel A — the rating curves (stage above zero-flow vs discharge)
        o = np.argsort(Q)
        axA.plot(Q[o], (H - h0)[o], ".", ms=1.5, color=col, alpha=0.5, rasterized=True)
        axA.plot([], [], "-", color=col, lw=2, label=name)
        # Panel B — local exponent beta(Q)
        qm, bm = local_beta(Q, H, h0)
        if len(qm):
            axB.plot(qm, bm, "-o", color=col, lw=1.8, ms=5, label=name)
        out[site] = dict(name=name, h0_m=round(h0, 2),
                         beta_by_Q={round(float(q)): round(float(b), 2) for q, b in zip(qm, bm)})
        # seismic Qc overlay (source gages)
        if sta in qc:
            qcv = qc[sta]["Qc_cms"]
            axB.axvline(qcv, color=col, ls=":", lw=1.4)
            axB.text(qcv, 2.05, f" seismic $Q_c$\n {sta}", color=col, fontsize=7.5, va="top")
            out[site]["seismic_Qc_cms"] = qcv
            out[site]["seismic_break"] = f"{qc[sta]['b_below']:.1f}->{qc[sta]['b_above']:.1f}"

    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("discharge $Q$  (m³ s⁻¹)")
    axA.set_ylabel("stage above zero-flow $h-h_0$  (m)")
    axA.set_title("(A) Stage–discharge ratings", fontsize=11)
    axA.legend(fontsize=8, loc="lower right")

    axB.set_xscale("log")
    axB.axhspan(0, 0.5, color="0.85", alpha=0.5, zorder=0)
    axB.text(axB.get_xlim()[0] * 1.1 if False else 12, 0.25, "overbank / flat section",
             fontsize=8, color="0.4", va="center")
    axB.set_xlabel("discharge $Q$  (m³ s⁻¹)")
    axB.set_ylabel(r"local rating exponent $\beta = \mathrm{d}\log Q/\mathrm{d}\log(h-h_0)$")
    axB.set_title("(B) Channel geometry vs discharge — confinement (β↑) vs overbank (β flat)",
                  fontsize=10.5)
    axB.set_ylim(0, 2.3)
    axB.legend(fontsize=8, loc="upper left")

    fig.suptitle("Stage–discharge geometry: the seismic break sits in the rating-steepening "
                 "(geometry-change) band; downstream overbank flattens β", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = FIGDIR / "fig26_rating_geometry.png"
    fig.savefig(out_png)
    plt.close(fig)
    (CONFIG / "rating_geometry.json").write_text(json.dumps(out, indent=2))

    for site, d in out.items():
        line = f"{d['name']:28s} beta(Q): " + " ".join(
            f"{q}:{b}" for q, b in d["beta_by_Q"].items())
        if "seismic_Qc_cms" in d:
            line += f"  | seismic Qc={d['seismic_Qc_cms']:.0f} ({d['seismic_break']})"
        print(line)
    print(f"wrote {out_png} + config/rating_geometry.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
