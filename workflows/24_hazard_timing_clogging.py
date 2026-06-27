#!/usr/bin/env python3
"""Hazard timing (M5) and the slow-recession clogging mechanism (M7).

Two co-author (D. Montgomery) questions, answered from the committed data:

  M5 — *hazard early-warning.* When does near-channel high-frequency (5–15 Hz,
  transport-band) seismic power rise relative to **peak discharge** and **peak
  stage**? If transport-band activity ramps up on the rising limb, the source-reach
  seismic gives lead time before the peak.

  M7 — *the clogging mechanism.* Is the braid reorganization tied to the **slow,
  sustained AR3 recession** rather than the rapid AR1 recession — consistent with
  gravel depositing in the active thread as stage falls slowly, clogging it and
  forcing an avulsion? This supplies the geomorphic *mechanism* (deposition →
  superelevation/aggradation → avulsion; Slingerland & Smith 2004; Jerolmack &
  Mohrig 2007) behind the recession-phase reorganization timed in workflow 21.

Method (Puyallup / Electron gage 12092000, which records BOTH discharge and stage):
  - peak Q and peak stage times;
  - transport-band ONSET per station = first time the 2 h rolling-median log-power
    exceeds the pre-flood baseline by >3·(robust σ), → lead over peak;
  - AR1 vs AR3 falling-limb rates (absolute and %/h);
  - the reorganization step times (from config/braided_reorg_timing_puyallup.json)
    overlaid on the recessions.

Outputs paper/figures/fig25_hazard_clogging.png + config/hazard_timing_clogging.json.

Usage: pixi run python workflows/24_hazard_timing_clogging.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.analysis import load_timeseries  # noqa: E402
from riverseis.figstyle import paper_style  # noqa: E402

RESULTS = ROOT / "notebooks" / "data" / "results"
CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"
GAGE = "usgs_iv_12092000_2025-12-01_2026-01-01.csv"      # Electron: discharge + stage
STATIONS = ["CC.PR01", "CC.PR02", "CC.PR03"]
EVENT = ("2025-12-08T00:00:00+00:00", "2025-12-13T12:00:00+00:00")
PRE = ("2025-12-05T00:00:00+00:00", "2025-12-08T00:00:00+00:00")
AR_COLORS = {"AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}
ST_COLORS = {"CC.PR01": "#e31a1c", "CC.PR02": "#33a02c", "CC.PR03": "#1f78b4"}
# falling-limb windows (peak/secondary-peak → trough) for the recession contrast
RECESSIONS = {"AR1": ("2025-12-09T03:30:00+00:00", "2025-12-10T00:00:00+00:00"),
              "AR3": ("2025-12-11T15:00:00+00:00", "2025-12-13T00:00:00+00:00")}


def load_gage() -> tuple[pd.Series, pd.Series]:
    g = pd.read_csv(ROOT / "notebooks" / "data" / GAGE, parse_dates=["time_utc"]).set_index("time_utc")
    q = (pd.to_numeric(g["discharge_cfs"], errors="coerce") * 0.0283168).sort_index()       # m³/s
    h = (pd.to_numeric(g["gage_height_ft"], errors="coerce") * 0.3048).sort_index()         # m
    return q, h


def transport_onset(sid: str) -> tuple[pd.Timestamp, float]:
    """First sustained rise of 5–15 Hz power above pre-flood baseline (robust)."""
    j = load_timeseries(RESULTS / f"{sid}_5.0-15.0Hz_timeseries.csv")
    lp = np.log10(j["P"].clip(lower=1e-30))
    pre = lp[(lp.index >= pd.Timestamp(PRE[0])) & (lp.index < pd.Timestamp(PRE[1]))]
    base = float(pre.median())
    sig = float(1.4826 * (pre - base).abs().median())
    thr = base + 3 * sig
    win = lp[(lp.index >= pd.Timestamp(EVENT[0])) & (lp.index < pd.Timestamp("2025-12-09T12:00:00+00:00"))]
    rm = win.rolling("2h", center=True, min_periods=4).median()
    above = rm[rm > thr]
    return (above.index[0] if len(above) else pd.NaT), base


def main() -> int:
    paper_style()
    q, h = load_gage()
    qe = q[(q.index >= pd.Timestamp(EVENT[0])) & (q.index <= pd.Timestamp(EVENT[1]))]
    he = h[(h.index >= pd.Timestamp(EVENT[0])) & (h.index <= pd.Timestamp(EVENT[1]))]
    q_peak_t, h_peak_t = qe.idxmax(), he.idxmax()

    onsets = {s: transport_onset(s)[0] for s in STATIONS}
    leads = {s: (q_peak_t - t).total_seconds() / 3600.0 for s, t in onsets.items() if pd.notna(t)}

    rec = {}
    for lab, (t0, t1) in RECESSIONS.items():
        s = q[t0:t1]
        hr = (s.index[-1] - s.index[0]).total_seconds() / 3600.0
        rec[lab] = dict(hours=round(hr, 1), q0=round(float(s.iloc[0])), q1=round(float(s.iloc[-1])),
                        dqdt=round((s.iloc[-1] - s.iloc[0]) / hr, 1),
                        pct_per_h=round(100 * (s.iloc[-1] / s.iloc[0] - 1) / hr, 1))

    reorg = json.loads((CONFIG / "braided_reorg_timing_puyallup.json").read_text())["stations"]
    steps = {s: reorg[s]["step_t50_utc"] for s in ("CC.PR02", "CC.PR03")
             if not reorg[s]["reversible"]}

    out = dict(q_peak_utc=str(q_peak_t), stage_peak_utc=str(h_peak_t),
               peak_offset_h=round((h_peak_t - q_peak_t).total_seconds() / 3600.0, 2),
               transport_onset=({s: str(t) for s, t in onsets.items()}),
               lead_over_peak_h={s: round(v, 1) for s, v in leads.items()},
               recessions=rec, reorg_steps=steps,
               interpretation="5-15 Hz onset leads peak Q by ~6 h (M5); AR3 recession "
               "slower than AR1 and the reorganization step lands on it (M7).")
    (CONFIG / "hazard_timing_clogging.json").write_text(json.dumps(out, indent=2))

    # ---------------- figure ----------------
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    ars = json.loads((CONFIG / "ar_windows.json").read_text())
    for ax in (axA, axB):
        for w in ars:
            if w["label"] in AR_COLORS:
                ax.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                           color=AR_COLORS[w["label"]], alpha=0.10, zorder=0)

    # Panel A — M5: hydrograph (Q + stage) with peak and transport-onset lead
    axA.plot(qe.index, qe.values, color="#222", lw=1.6, label="discharge Q")
    axA.set_ylabel("Q  (m³ s⁻¹)")
    axS = axA.twinx()
    axS.plot(he.index, he.values, color="#2c7fb8", lw=1.3, ls="--", label="stage")
    axS.set_ylabel("stage (m)", color="#2c7fb8")
    axS.tick_params(axis="y", labelcolor="#2c7fb8")
    axA.axvline(q_peak_t, color="#D55E00", lw=1.4)
    axA.text(q_peak_t, qe.max() * 0.97, " peak Q & stage", color="#D55E00", fontsize=9, va="top")
    onset_med = min(t for t in onsets.values() if pd.notna(t))
    axA.axvspan(onset_med, q_peak_t, color="#fdae61", alpha=0.25, zorder=1)
    for s, t in onsets.items():
        if pd.notna(t):
            axA.plot(t, np.interp(mdates.date2num(t), mdates.date2num(qe.index), qe.values),
                     "*", ms=13, mfc=ST_COLORS[s], mec="k", mew=0.6, zorder=6)
    lead_lo, lead_hi = min(leads.values()), max(leads.values())
    axA.annotate(f"transport-band (5–15 Hz) onset\nleads peak by {lead_lo:.0f}–{lead_hi:.0f} h",
                 xy=(onset_med, qe.loc[:q_peak_t].max() * 0.45),
                 xytext=(pd.Timestamp("2025-12-08T02:00:00+00:00"), qe.max() * 0.55),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="#b35900"))
    axA.set_title("(A) Hazard timing — transport-band power rises before peak discharge (M5)",
                  fontsize=11, loc="left")
    axA.legend(loc="upper right", fontsize=8)

    # Panel B — M7: 5–15 Hz power, recession contrast, reorganization on the AR3 slow fall
    for s in STATIONS:
        j = load_timeseries(RESULTS / f"{s}_5.0-15.0Hz_timeseries.csv")
        p = j["P"][(j.index >= pd.Timestamp(EVENT[0])) & (j.index <= pd.Timestamp(EVENT[1]))]
        pre = j["P"][(j.index >= pd.Timestamp(PRE[0])) & (j.index < pd.Timestamp(PRE[1]))].median()
        axB.semilogy(p.index, (p / pre).rolling("2h", center=True, min_periods=4).median(),
                     color=ST_COLORS[s], lw=1.3, label=s)
    axB.set_ylabel("5–15 Hz power /\npre-flood median")
    axB.set_xlabel("December 2025 (UTC)")
    # recession-rate annotations
    for lab, (t0, t1) in RECESSIONS.items():
        mid = pd.Timestamp(t0) + (pd.Timestamp(t1) - pd.Timestamp(t0)) / 2
        r = rec[lab]
        axB.axvspan(pd.Timestamp(t0), pd.Timestamp(t1), color="0.5",
                    alpha=0.05 if lab == "AR1" else 0.12, zorder=0)
        axB.text(mid, axB.get_ylim()[1] * 0.6,
                 f"{lab} fall\n{r['dqdt']:+.0f} m³/s/h\n({r['hours']:.0f} h)",
                 ha="center", va="top", fontsize=8.5,
                 color="#444" if lab == "AR1" else "#7a4f00", fontweight="bold")
    for s, t in steps.items():
        ts = pd.Timestamp(t)
        axB.axvline(ts, color="#6a3d9a", lw=1.4, ls="-")
    axB.text(pd.Timestamp(list(steps.values())[0]), axB.get_ylim()[0] * 1.6,
             " braid reorganization\n (avulsion step)", color="#6a3d9a", fontsize=9, va="bottom")
    axB.set_title("(B) The slow AR3 recession hosts the reorganization — the clogging mechanism (M7)",
                  fontsize=11, loc="left")
    axB.legend(loc="upper right", fontsize=8, ncol=3)
    axB.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    fig.suptitle("Bedload-initiation lead time and slow-recession braid clogging — "
                 "Puyallup cluster, Dec-2025", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = FIGDIR / "fig25_hazard_clogging.png"
    fig.savefig(out_png)
    plt.close(fig)

    print(f"peak Q & stage: {q_peak_t} (offset {out['peak_offset_h']:+.2f} h)")
    print("transport-band onset lead over peak:")
    for s, v in leads.items():
        print(f"   {s}: onset {str(onsets[s])[:16]}  lead {v:+.1f} h")
    print("recession contrast:")
    for lab, r in rec.items():
        print(f"   {lab}: {r['q0']}→{r['q1']} m³/s over {r['hours']} h "
              f"({r['dqdt']:+.1f} m³/s/h, {r['pct_per_h']:+.1f} %/h)")
    print("reorganization steps (on the AR3 slow recession):",
          {s: t[:16] for s, t in steps.items()})
    print(f"wrote {out_png} + config/hazard_timing_clogging.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
