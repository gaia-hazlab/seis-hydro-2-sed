#!/usr/bin/env python3
"""Composite figure F4 — "The braided source: a distributed, non-stationary seismic source."

Merges the two braided-reach diagnostics into one paper figure:

  (a) Per-atmospheric-river (AR) P–Q hysteresis loops for the three Puyallup
      braided-reach stations (CC.PR01/PR02/PR03).  Rising limb solid, falling
      limb dashed, with the Lawler clockwise hysteresis index (HI) annotated.
      The loops are near-reversible (HI ≈ 0 -> geometric / wetted-front onset,
      NOT supply-limited transport), and the cross-AR baseline offset drifts
      coherently positive AR1 -> AR3 — the active anabranch migrating toward the
      cluster.  (Mirrors old fig16 / workflows/18_braided_hysteresis.py.)

  (b) Seismic reach r_e(f) = v_c Q(f) / (2πf) for the PNW (Rainier-edifice
      Q=25 f^0.5) and Tsai (Q=20) parameterizations, plus the observed weak
      near-source decay establishing an e-folding of ~780 m — so the PR cluster
      integrates a *moving, spatially distributed* source rather than a single
      line channel.  (Mirrors old fig9 / workflows/09_attenuation.py.)

Rebuilds OFFLINE from cached JSON only:
  config/braided_hysteresis.json, config/attenuation_fit.json.

Usage: pixi run python workflows/35_figF4_braided.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402
from riverseis.analysis import load_timeseries  # noqa: E402

CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"
RESULTS = ROOT / "notebooks" / "data" / "results"

STATIONS = ["CC.PR01", "CC.PR02", "CC.PR03"]
AR_COLORS = {"AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}


def real_loops(sid):
    """Return the real per-AR P–Q loops for station ``sid``, mirroring
    workflows/18_braided_hysteresis.py exactly.

    Reads the cached raw timeseries CSV (time_utc, proxy->P, gauge->Q),
    clips to each AR window from config/ar_windows.json, splits each AR at
    the peak discharge into a rising limb (solid) and a falling limb
    (dashed), and yields log10(Q) vs log10(P) for each.
    """
    ars = json.loads((CONFIG / "ar_windows.json").read_text())
    j = load_timeseries(RESULTS / f"{sid}_5.0-15.0Hz_timeseries.csv")
    loops = []
    for w in ars:
        if w["label"] == "pre-AR":
            continue
        sl = j[(j.index >= pd.Timestamp(w["start"])) &
               (j.index <= pd.Timestamp(w["end"]))]
        if len(sl) < 8:
            continue
        q = sl["Q"].values
        lp = np.log10(sl["P"].clip(lower=1e-30)).values
        lq = np.log10(sl["Q"].clip(lower=1e-6)).values
        pk = int(np.argmax(q))                       # split at peak discharge
        loops.append(dict(AR=w["label"],
                          lq_up=lq[:pk + 1], lp_up=lp[:pk + 1],
                          lq_dn=lq[pk:], lp_dn=lp[pk:]))
    return loops


def panel_hysteresis(ax, hyst):
    """Panel (a): REAL per-AR P–Q hysteresis loops for the 3 PR stations.

    Loops are drawn straight from the raw cached timeseries (not synthesized
    from summary stats), so the data curvature/aperture is shown faithfully.
    Each station's family is vertically shifted to its own band so the three
    stack cleanly; HI annotations and the AR1->AR3 baseline-drift arrow keep
    the cached values from config/braided_hysteresis.json.
    """
    # vertically offset each station's family of loops so they stack cleanly.
    # Real log-power baselines differ per station, so shift each family to a
    # target center band while preserving the true within-station shape.
    station_band = {"CC.PR01": 0.0, "CC.PR02": 2.6, "CC.PR03": 5.2}
    hi_text = []
    all_lq = []
    label_y = {}
    for sid in STATIONS:
        rec = hyst[sid]
        loops = real_loops(sid)
        # center this station's loops on its target band
        lp_vals = np.concatenate([np.r_[L["lp_up"], L["lp_dn"]] for L in loops])
        sh = station_band[sid] - float(np.median(lp_vals))
        for L in loops:
            c = AR_COLORS.get(L["AR"], "#333")
            ax.plot(L["lq_up"], L["lp_up"] + sh, "-", color=c, lw=1.7, alpha=0.95)
            ax.plot(L["lq_dn"], L["lp_dn"] + sh, "--", color=c, lw=1.4, alpha=0.75)
            all_lq.append(L["lq_up"]); all_lq.append(L["lq_dn"])
        # station label at the high-Q edge of its band
        lq_max = max(float(L["lq_up"].max()) for L in loops)
        y_band = station_band[sid]
        label_y[sid] = (lq_max, y_band)
        # arrow tracing the AR1->AR3 coherent positive baseline drift (cached
        # baseline_offset), drawn at the low-Q side of the band.
        offs = [a["baseline_offset"] for a in rec["per_AR"]]
        x_arr = float(min(L["lq_up"].min() for L in loops)) - 0.04
        ax.annotate("", xy=(x_arr, y_band + 1.4 * offs[-1]),
                    xytext=(x_arr, y_band + 1.4 * offs[0]),
                    arrowprops=dict(arrowstyle="-|>", color="0.35", lw=2.0))
        hi_text.append("{}  HI {:+.3f}/{:+.3f}/{:+.3f}".format(
            sid.split(".")[1], *[a["HI"] for a in rec["per_AR"]]))

    lq_hi = max(a.max() for a in all_lq)
    lq_lo = min(a.min() for a in all_lq)
    # station labels in the right gutter (outside the loops, which end at lq_hi)
    for sid, (lqm, yb) in label_y.items():
        ax.text(lq_hi + 0.07, yb, sid.split(".")[1], fontsize=13,
                color="0.2", va="center", ha="left")

    # legend: AR colors (rising solid) once. Placed in the right gutter below the
    # data so it never overlaps the loops.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=AR_COLORS[a], lw=2.4, label=a)
               for a in ("AR1", "AR2", "AR3")]
    handles += [
        Line2D([0], [0], color="0.3", lw=1.9, ls="-", label="rising limb"),
        Line2D([0], [0], color="0.3", lw=1.5, ls="--", label="falling limb"),
        Line2D([0], [0], color="0.35", lw=2.0, label="AR1→AR3 drift"),
    ]
    ax.legend(handles=handles, fontsize=12, loc="lower right", ncol=1,
              framealpha=0.92, borderpad=0.6, labelspacing=0.35,
              handlelength=1.6)

    # Reserve clear headroom above the loops and a strip below them so the two
    # text boxes sit in genuinely empty regions, not on the data.
    y_top = 6.6        # PR03 band tops out ~6.33
    y_bot = -1.89      # PR01 band floor
    ax.set_ylim(y_bot - 2.6, y_top + 1.9)

    # interpretation note: in the clear headroom strip above all three bands.
    ax.text(0.5, 0.985,
            "loops near-reversible (HI ≈ 0) → geometric onset;\n"
            "coherent baseline drift AR1→AR3 → channel migration",
            transform=ax.transAxes, fontsize=12, color="0.25",
            va="top", ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.92))
    # per-station HI readout: in the clear strip below the PR01 band, lower-left.
    ax.text(0.015, 0.015, "\n".join(hi_text), transform=ax.transAxes,
            fontsize=12, color="0.25", va="bottom", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.92))

    ax.set_xlim(lq_lo - 0.12, lq_hi + 0.30)
    ax.set_xlabel(r"$\log_{10} Q$  (m$^3$ s$^{-1}$)")
    ax.set_ylabel(r"$\log_{10} P$  (5–15 Hz, stacked per station)")
    ax.set_title("(a) Per-AR P–Q loops", loc="left", fontsize=14)


def panel_reach(ax, att):
    """Panel (b): seismic reach r_e(f) for PNW vs Tsai + observed e-folding."""
    F0, XI, VC0 = 1.0, 0.374, 1295.0          # Tsai 2012 velocity dispersion
    fc = att["fc_hz"]
    Q0, eta, Q_tsai = att["Q0_pnw"], att["eta"], att["Q_tsai"]
    re_pnw = att["r_e_pnw_m"]                  # ~781 m
    re_tsai = att["r_e_tsai_m"]               # ~212 m

    ff = np.logspace(0, 2, 300)
    vcf = VC0 * (ff / F0) ** (-XI)            # phase velocity dispersion
    re = lambda Q: vcf * Q / (2 * np.pi * ff) / 1000.0   # km

    # PNW plausible Q range envelope
    ax.fill_between(ff, re(20 * ff ** 0.3), re(60 * ff ** 0.8),
                    color="0.82", alpha=0.6, label="PNW range")
    ax.plot(ff, re(Q0 * ff ** eta), "k-", lw=2.0,
            label=r"PNW  Q=25 f$^{0.5}$")
    ax.plot(ff, re(np.full_like(ff, Q_tsai)), color="tab:red", ls=":", lw=1.6,
            label="Tsai  Q=20")

    # band shading + labels (band labels sit low, in the clear lower band area)
    ax.axvspan(1, 20, color="#0072B2", alpha=0.10)
    ax.axvspan(30, 80, color="#E69F00", alpha=0.13)
    ax.text(4.5, 0.018, "turbulence\n1–20 Hz", fontsize=12, ha="center",
            va="bottom", color="0.3")
    ax.text(49, 0.018, "bedload\n30–80 Hz", fontsize=12, ha="center",
            va="bottom", color="0.35")
    ax.axvline(25, color="0.45", ls="--", lw=0.9)
    ax.text(24, 14, "50-sps Nyquist", fontsize=11, color="0.35",
            rotation=90, va="top", ha="right")

    # band center marker + the two cached r_e values
    ax.axvline(fc, color="0.4", ls="-", lw=0.7, alpha=0.6)
    ax.scatter([fc, fc], [re_pnw / 1000.0, re_tsai / 1000.0],
               color=["k", "tab:red"], zorder=6, s=48, edgecolor="w", lw=0.8)
    ax.annotate(rf"$r_e\approx${re_pnw:.0f} m (PNW)", (fc, re_pnw / 1000.0),
                xytext=(9, 9), textcoords="offset points", fontsize=12)
    ax.annotate(rf"$r_e\approx${re_tsai:.0f} m (Tsai)", (fc, re_tsai / 1000.0),
                xytext=(9, -16), textcoords="offset points", fontsize=12,
                color="tab:red")

    # observed PR-cluster decay band: ~flat over 0.2–2 km -> weak decay,
    # consistent with (even exceeding) the ~780 m PNW e-folding. The PR standoff
    # lines sit at the left edge so labels never collide with the decay box.
    dpk = att["observed_decay_per_km"]
    ax.axhspan(0.20, 1.90, xmin=0.0, xmax=1.0, color="#0072B2", alpha=0.06)
    for r_km, lab in [(0.19, "PR01"), (0.71, "PR02"), (1.90, "PR03")]:
        ax.axhline(r_km, color="0.55", ls=":", lw=0.8)
        ax.text(1.06, r_km * 1.05, lab, fontsize=11, color="0.4",
                va="bottom", ha="left")
    ax.text(0.985, 0.035,
            f"observed PR-cluster decay ≈ {dpk:.3f} km$^{{-1}}$\n"
            f"(near-flat 0.2–2 km) ⇒ $r_e \\geq$ {re_pnw:.0f} m:\n"
            "cluster integrates a moving distributed source",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
            color="0.25",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.92))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 100); ax.set_ylim(0.01, 30)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(r"e-folding distance $r_e$  (km)")
    ax.set_title(r"(b) Seismic reach $r_e(f)$",
                 loc="left", fontsize=14)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.92,
              borderpad=0.5, labelspacing=0.35)


def main() -> int:
    paper_style()
    hyst = json.loads((CONFIG / "braided_hysteresis.json").read_text())
    att = json.loads((CONFIG / "attenuation_fit.json").read_text())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6.2))
    panel_hysteresis(axA, hyst)
    panel_reach(axB, att)
    fig.tight_layout(pad=1.2, w_pad=2.0)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "figF4_braided.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print(f"wrote {out}")
    print("\nPanel (a) HI per station (AR1/AR2/AR3):")
    for sid in STATIONS:
        his = [a["HI"] for a in hyst[sid]["per_AR"]]
        offs = [a["baseline_offset"] for a in hyst[sid]["per_AR"]]
        print(f"  {sid}: HI={his}  baseline_offset={offs}  b_pooled={hyst[sid]['b_pooled']}")
    print("\nPanel (b):")
    print(f"  fc={att['fc_hz']} Hz, vc={att['vc_ms']} m/s, Q_pnw(fc)={att['Q_pnw_at_fc']}")
    print(f"  r_e PNW={att['r_e_pnw_m']} m, r_e Tsai={att['r_e_tsai_m']} m")
    print(f"  observed decay={att['observed_decay_per_km']} per km")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
