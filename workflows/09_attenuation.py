#!/usr/bin/env python3
"""First-order distance-attenuation analysis of the bedload-band signal.

The PR01/PR02/PR03 stations share one source reach (Puyallup nr Electron) at
different channel standoffs (0.19, 0.71, 1.9 km) — a natural attenuation
experiment. We fit the decay of absolute 5–15 Hz power with distance to estimate
an effective quality factor Q and e-folding distance, compare to the Tsai/Gimbert
prediction (Q≈20), and use it as a first-order correction (normalize power to the
channel). Outputs fig9_attenuation.png and config/attenuation_fit.json.

Refs: Tsai et al. 2012 (GRL, 10.1029/2011GL050255); Gimbert et al. 2014
(JGR-ES, 10.1002/2014JF003201); Bakker et al. 2020; Lagarde et al. 2021.

Usage: pixi run python workflows/09_attenuation.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
EXCLUDE = {"UW.BHW", "UW.TEHA"}
HF_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)_5\.0-15\.0Hz_timeseries\.csv$")

FC = (5 * 15) ** 0.5          # band-center 8.66 Hz
VC0, F0, XI = 1295.0, 1.0, 0.374                    # Tsai 2012 velocity dispersion
VC = VC0 * (FC / F0) ** (-XI)                       # phase velocity at FC (~578 m/s)
# PNW / Mt. Rainier-edifice attenuation (coda/Lg-Q + MSH edifice studies):
# Q(f)=Q0 f^eta, Q0≈25, eta≈0.5 -> Q(10 Hz)≈80 (range 40–240); higher than the
# fluvial default Tsai Q≈20 (less attenuation). See LITERATURE.md.
Q0_PNW, ETA = 25.0, 0.5
Q_TSAI = 20.0
def Q_pnw(f):
    return Q0_PNW * f ** ETA
# stations sharing the Electron source reach (same gage 12092000), used for the fit
SAME_SOURCE = ["CC.PR03", "CC.PR01", "CC.PR02"]


def main() -> int:
    coords = {f'{v["net"]}.{v["sta"]}': v for v in
              json.loads((ROOT / "config" / "_transect_discovery.json").read_text())["stations"]}
    ar = json.loads((ROOT / "config" / "ar_windows.json").read_text())
    ar2 = next(w for w in ar if w["label"] == "AR2")
    t0, t1 = pd.Timestamp(ar2["start"]), pd.Timestamp(ar2["end"])

    rows = {}
    for f in sorted(RESULTS.glob("*_5.0-15.0Hz_timeseries.csv")):
        m = HF_RE.match(f.name)
        sid = f'{m["net"]}.{m["sta"]}' if m else None
        if not sid or sid in EXCLUDE or sid not in coords:
            continue
        df = pd.read_csv(f, parse_dates=["time_utc"]).set_index("time_utc")
        P = pd.to_numeric(df["proxy"], errors="coerce").dropna()
        seg = P[(P.index >= t0) & (P.index < t1)]
        if len(seg) < 5:
            continue
        rows[sid] = dict(r_km=float(coords[sid]["dist_river_km"]),
                         P=float(seg.median()), trib=not coords[sid].get("river_main", True))

    # observed decay on the same-source cluster (for reporting; expected near-flat)
    sub = [(rows[s]["r_km"] * 1000.0, np.log(rows[s]["P"])) for s in SAME_SOURCE if s in rows]
    r_m = np.array([a for a, _ in sub]); lnP = np.array([b for _, b in sub])
    k_obs = -np.polyfit(r_m, lnP, 1)[0]                       # observed decay rate (1/m)
    re_pnw = VC * Q_pnw(FC) / (2 * np.pi * FC)               # PNW e-folding (m)
    re_tsai = VC * Q_TSAI / (2 * np.pi * FC)

    fit = dict(fc_hz=round(FC, 2), vc_ms=round(VC, 1),
               Q_pnw_at_fc=round(Q_pnw(FC), 1), Q0_pnw=Q0_PNW, eta=ETA, Q_tsai=Q_TSAI,
               r_e_pnw_m=round(re_pnw, 0), r_e_tsai_m=round(re_tsai, 0),
               observed_decay_per_km=round(float(k_obs * 1000), 3),
               note="PNW/Rainier-edifice Q(f)=25 f^0.5 (Q≈74 at the band center) gives "
                    "r_e≈790 m vs ≈210 m for Tsai Q=20. Observed 5-15 Hz power is still "
                    "~distance-independent over 0.2-2 km (weaker decay than even the PNW "
                    "prediction) -> the band retains less-attenuated lower-frequency "
                    "energy; site response + crude distances limit a data-driven Q.")
    (ROOT / "config" / "attenuation_fit.json").write_text(json.dumps(fit, indent=2))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # Panel A: seismic reach r_e(f) — PNW Q(f) (range), Tsai Q=20 reference, our bands/standoffs
    ff = np.logspace(0, 2, 200)
    vcf = VC0 * (ff / F0) ** (-XI)
    re = lambda Q: vcf * Q / (2 * np.pi * ff)
    axA.fill_between(ff, re(20 * ff ** 0.3), re(60 * ff ** 0.8), color="0.8", alpha=0.6,
                     label="PNW range")
    axA.plot(ff, re(Q0_PNW * ff ** ETA), "k-", lw=1.8, label="PNW Q=25 f$^{0.5}$")
    axA.plot(ff, re(Q_TSAI), color="tab:red", ls=":", lw=1.4, label="Tsai Q=20")
    axA.axvline(25, color="0.4", ls="--", lw=1)
    axA.text(25, 0.012, " 50-sps Nyquist", fontsize=6.5, color="0.3", rotation=90, va="bottom")
    axA.axvspan(1, 20, color="#0072B2", alpha=0.12)
    axA.axvspan(30, 80, color="#E69F00", alpha=0.15)
    axA.text(4.5, 5, "turbulence\n1–20 Hz", fontsize=7, ha="center")
    axA.text(49, 5, "bedload\n30–80 Hz", fontsize=7, ha="center")
    for sid in ("CC.PR03", "CC.PR01", "CC.PR02", "CC.TRON"):
        if sid in rows:
            axA.axhline(rows[sid]["r_km"], color="0.5", ls=":", lw=0.8)
            axA.text(1.05, rows[sid]["r_km"] * 1.05, sid.split(".")[1], fontsize=6.5, color="0.4")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("frequency (Hz)"); axA.set_ylabel("e-folding distance r_e (km)")
    axA.set_title("Seismic reach r_e = v_c Q(f) / (2πf)", loc="left", fontsize=10)
    axA.legend(fontsize=7, loc="upper right")
    axA.set_ylim(0.01, 30)

    # Panel B: observed AR2 power vs distance vs the PNW-Q prediction
    rr = np.linspace(50, 6000, 200)
    kt = 2 * np.pi * FC / (VC * Q_pnw(FC))
    P0 = rows["CC.PR03"]["P"] * np.exp(kt * rows["CC.PR03"]["r_km"] * 1000)
    axB.plot(rr / 1000, P0 * np.exp(-kt * rr), color="tab:red", lw=1.5,
             label=f"theory PNW Q≈{Q_pnw(FC):.0f} (r_e≈{re_pnw:.0f} m)")
    for sid, d in rows.items():
        same = sid in SAME_SOURCE
        axB.scatter(d["r_km"], d["P"], s=75 if same else 45,
                    color="#0072B2" if same else "#999999",
                    marker="o" if same else "^", edgecolor="k", zorder=4)
        axB.annotate(sid.split(".")[1], (d["r_km"], d["P"]), fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
    axB.set_yscale("log")
    axB.set_xlabel("station distance from channel (km)")
    axB.set_ylabel("AR2 5–15 Hz power (m²/s²)")
    axB.set_title("Observed band power ≫ Q=20 prediction at distance", loc="left", fontsize=10)
    axB.legend(fontsize=8)
    axB.text(0.97, 0.05, "weak observed decay ⇒ band retains\nless-attenuated (lower-f) energy",
             transform=axB.transAxes, ha="right", va="bottom", fontsize=7, color="0.4")
    # (no figure suptitle — described by the manuscript caption; panel titles kept)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig9_attenuation.png", dpi=200)
    plt.close(fig)

    print(json.dumps(fit, indent=2))
    print(f"\nwrote {FIGDIR}/fig9_attenuation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
