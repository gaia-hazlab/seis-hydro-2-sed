#!/usr/bin/env python3
"""Composite figure F8 — Distributed discharge sensing, virtual gage, and
downstream early warning.

Merges three previously separate figures into one publication panel:

  (a) Seismic *virtual discharge* vs the co-located gage for the best-validated
      river-proximal stations (old fig12 / workflow 12_virtual_q.py).  The 5–15 Hz
      seismic power is inverted through the per-station P–Q rating (a, b stored in
      config/virtual_q.json) to a virtual discharge time series and overlaid on the
      gage; per-panel b, r and log-Q Nash–Sutcliffe (NSE) quantify the skill.

  (b) Transport-onset warning window (old fig25 panel a / workflow 24): the
      Electron hydrograph (discharge + stage) through the Dec-2025 flood with the
      5–15 Hz transport-band onset (stars) leading peak discharge by ~5–7 h.
      Numbers from config/hazard_timing_clogging.json.

  (c) Downstream early warning (old fig10 / workflow 10): the upstream CC.PR03
      seismic 5–15 Hz peak leads the downstream Puyallup discharge peak, and the
      flood peak itself propagates Electron -> Orting -> Puyallup, giving a
      multi-tens-of-hours corridor lead.

All panels rebuild offline from cached JSON/CSV.  Outputs
paper/figures/figF8_virtualq.png.

Usage: pixi run python workflows/37_figF8_virtualq.py
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
from riverseis.figstyle import paper_style  # noqa: E402

RESULTS = ROOT / "notebooks" / "data" / "results"
CONFIG = ROOT / "config"
DATA = ROOT / "notebooks" / "data"
FIGDIR = ROOT / "paper" / "figures"

# Panel (a): best-validated virtual gages (high log-Q NSE in virtual_q_fit.json)
VALID_STATIONS = ["CC.PR03", "CC.PR02", "CC.STYX"]
ST_COLORS = {"CC.PR01": "#e31a1c", "CC.PR02": "#33a02c", "CC.PR03": "#1f78b4"}

# Panel (b): hazard / transport-onset (Electron gage records Q and stage)
GAGE = DATA / "usgs_iv_12092000_2025-12-01_2026-01-01.csv"
HAZ_STATIONS = ["CC.PR01", "CC.PR02", "CC.PR03"]
EVENT = ("2025-12-08T00:00:00+00:00", "2025-12-13T12:00:00+00:00")

# Panel (c): downstream corridor (mainstem gages) + upstream seismic
MAINSTEM = [("12092000", "Electron (RM~41)", "#0072B2"),
            ("12093500", "Orting (RM~30)", "#E69F00"),
            ("12101500", "Puyallup (RM~10)", "#009E73")]


# ---------------------------------------------------------------------------
def load_proxy_gage(sid: str) -> pd.DataFrame:
    """5–15 Hz proxy + gage discharge for a station, on the flood window."""
    df = pd.read_csv(RESULTS / f"{sid}_5.0-15.0Hz_timeseries.csv",
                     parse_dates=["time_utc"]).set_index("time_utc")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    P = pd.to_numeric(df["proxy"], errors="coerce")
    Q = pd.to_numeric(df["gauge"], errors="coerce")
    j = pd.concat([P.rename("P"), Q.rename("Q")], axis=1).sort_index()
    j["Q"] = j["Q"].interpolate("linear", limit=12)
    return j.dropna()


def gser(aux: dict, gid: str) -> pd.Series:
    d = aux["discharge"][gid]
    return pd.Series(d["q_cms"], index=pd.to_datetime(d["time"], utc=True)).sort_index()


# ---------------------------------------------------------------------------
def panel_a(ax_list, virt: dict, fit: dict) -> dict:
    """Seismic virtual discharge vs co-located gage (validation)."""
    fitmap = {r["station"]: r for r in fit}
    out = {}
    for ax, sid in zip(ax_list, VALID_STATIONS):
        j = load_proxy_gage(sid)
        v = virt[sid]
        qs = pd.Series(v["q_seis"], index=pd.to_datetime(v["time"], utc=True))
        # restrict virtual series to the gage-observed window for a fair overlay
        qs = qs[(qs.index >= j.index.min()) & (qs.index <= j.index.max())]
        col = ST_COLORS.get(sid, "#1f78b4")
        ax.plot(j.index, j["Q"], color="k", lw=1.4, label="gage Q", zorder=3)
        ax.plot(qs.index, qs.values, color="#D55E00", lw=1.0, alpha=0.85,
                label="seismic virtual Q", zorder=2)
        rr = fitmap[sid]
        ax.set_title(f"{sid}   b={rr['b']:.2f}  r={rr['r']:.2f}  NSE={rr['nse_logQ']:.2f}",
                     fontsize=10, loc="left")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelsize=9)
        out[sid] = rr
    ax_list[0].legend(fontsize=8.5, loc="upper right")
    ax_list[0].set_ylabel("discharge (m³ s⁻¹)")
    return out


def panel_b(ax) -> dict:
    """Transport-onset warning window: Q + stage with onset stars before peak."""
    haz = json.loads((CONFIG / "hazard_timing_clogging.json").read_text())
    g = pd.read_csv(GAGE, parse_dates=["time_utc"]).set_index("time_utc")
    q = (pd.to_numeric(g["discharge_cfs"], errors="coerce") * 0.0283168).sort_index()  # m³/s
    h = (pd.to_numeric(g["gage_height_ft"], errors="coerce") * 0.3048).sort_index()    # m
    qe = q[(q.index >= pd.Timestamp(EVENT[0])) & (q.index <= pd.Timestamp(EVENT[1]))]
    he = h[(h.index >= pd.Timestamp(EVENT[0])) & (h.index <= pd.Timestamp(EVENT[1]))]
    q_peak_t = pd.Timestamp(haz["q_peak_utc"])

    onsets = {s: pd.Timestamp(t) for s, t in haz["transport_onset"].items()}
    leads = haz["lead_over_peak_h"]

    ax.plot(qe.index, qe.values, color="#222", lw=1.7, label="discharge Q", zorder=3)
    ax.set_ylabel("Q  (m³ s⁻¹)")
    axS = ax.twinx()
    axS.grid(False)
    axS.plot(he.index, he.values, color="#2c7fb8", lw=1.3, ls="--", label="stage", zorder=2)
    axS.set_ylabel("stage (m)", color="#2c7fb8")
    axS.tick_params(axis="y", labelcolor="#2c7fb8")

    ax.axvline(q_peak_t, color="#D55E00", lw=1.5, zorder=4)
    ax.text(q_peak_t, qe.max() * 0.97, " peak Q & stage", color="#D55E00",
            fontsize=9, va="top")

    onset_first = min(onsets.values())
    ax.axvspan(onset_first, q_peak_t, color="#fdae61", alpha=0.28, zorder=1)
    for s, t in onsets.items():
        yv = np.interp(mdates.date2num(t), mdates.date2num(qe.index), qe.values)
        ax.plot(t, yv, "*", ms=15, mfc=ST_COLORS[s], mec="k", mew=0.6, zorder=6,
                label=f"{s} onset")
    lead_lo, lead_hi = min(leads.values()), max(leads.values())
    ax.annotate(f"5–15 Hz transport-band onset\nleads peak by {lead_lo:.0f}–{lead_hi:.0f} h",
                xy=(onset_first, qe.loc[:q_peak_t].max() * 0.45),
                xytext=(pd.Timestamp("2025-12-08T02:00:00+00:00"), qe.max() * 0.62),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#b35900"))
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("December 2025 (UTC)")
    return dict(leads=leads, q_peak_utc=str(q_peak_t))


def panel_c(ax) -> dict:
    """Downstream early warning: upstream seismic + flood-peak corridor lead."""
    aux = json.loads((CONFIG / "aux_timeseries.json").read_text())
    peaks = {}
    for gid, nm, col in MAINSTEM:
        s = gser(aux, gid)
        ax.plot(s.index, s.values, color=col, lw=1.6, label=nm, zorder=3)
        tpk = s.rolling(3, center=True, min_periods=1).mean().idxmax()
        peaks[nm] = (tpk, s.max())
        ax.scatter([tpk], [s.max()], color=col, s=45, edgecolor="k", lw=0.5, zorder=5)
    t_up = peaks["Electron (RM~41)"][0]
    t_dn = peaks["Puyallup (RM~10)"][0]
    corridor_lead = (t_dn - t_up).total_seconds() / 3600.0
    ax.set_ylabel("discharge (m³ s⁻¹)")

    # upstream seismic (CC.PR03 5–15 Hz) on a twin axis
    axb = ax.twinx()
    axb.grid(False)
    pr = pd.read_csv(RESULTS / "CC.PR03_5.0-15.0Hz_timeseries.csv",
                     parse_dates=["time_utc"]).set_index("time_utc")
    pr.index = pd.to_datetime(pr.index, utc=True, errors="coerce")
    P = pd.to_numeric(pr["proxy"], errors="coerce").rolling(12, center=True, min_periods=1).median()
    Pn = (P / P.median())
    h1, = axb.semilogy(Pn.index, Pn.values, color="#D55E00", lw=1.0, alpha=0.85,
                       label="CC.PR03 seismic 5–15 Hz (upstream)")
    axb.set_ylabel("seismic power / median", color="#D55E00")
    axb.tick_params(axis="y", labelcolor="#D55E00")

    sp = P["2025-12-09":"2025-12-12"].idxmax()
    qd = gser(aux, "12101500")
    qp = qd["2025-12-10":"2025-12-13"].idxmax()
    seis_lead = (qp - sp).total_seconds() / 3600.0
    axb.axvline(sp, color="#D55E00", lw=1.0, ls=":")
    ax.axvline(qp, color="#009E73", lw=1.0, ls=":")
    # lead arrow placed mid-height (in seismic-axis log coords) so it clears titles
    ylo, yhi = axb.get_ylim()
    yarr = ylo * (yhi / ylo) ** 0.55
    axb.annotate("", xy=(qp, yarr), xytext=(sp, yarr),
                 arrowprops=dict(arrowstyle="<->", color="#7a4f00", lw=1.4))
    axb.text(sp + (qp - sp) / 2, yarr * 1.12,
             f"upstream seismic leads\ndownstream peak ~{seis_lead:.0f} h",
             ha="center", va="bottom", fontsize=8.5, color="#7a4f00")

    ax.set_xlim(pd.Timestamp("2025-12-05", tz="UTC"), pd.Timestamp("2025-12-15", tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("December 2025 (UTC)")
    # combined legend, compact, below the upper crowding
    handles_a, labels_a = ax.get_legend_handles_labels()
    short = {"Electron (RM~41)": "Electron", "Orting (RM~30)": "Orting",
             "Puyallup (RM~10)": "Puyallup"}
    labels_a = [short.get(l, l) for l in labels_a]
    ax.legend(handles_a + [h1], labels_a + ["CC.PR03 seismic"],
              fontsize=7.5, loc="upper left", framealpha=0.85)
    return dict(corridor_lead_h=corridor_lead, seismic_lead_h=seis_lead,
                electron_peak=str(t_up), puyallup_peak=str(t_dn))


# ---------------------------------------------------------------------------
def main() -> int:
    paper_style()
    virt = json.loads((CONFIG / "virtual_q.json").read_text())
    fit = json.loads((CONFIG / "virtual_q_fit.json").read_text())

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05],
                          hspace=0.42, wspace=0.30)
    ax_a = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_c = fig.add_subplot(gs[1, 2])

    # Panel letter tags
    ax_a[0].text(-0.18, 1.12, "(a)", transform=ax_a[0].transAxes,
                 fontsize=14, fontweight="bold", va="bottom")
    ax_b.text(-0.10, 1.04, "(b)", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="bottom")
    ax_c.text(-0.28, 1.10, "(c)", transform=ax_c.transAxes,
              fontsize=14, fontweight="bold", va="bottom")

    skill = panel_a(ax_a, virt, fit)
    fig.text(0.5, 0.965, "Seismic virtual discharge vs co-located gage",
             ha="center", fontsize=11.5, fontweight="bold")

    haz = panel_b(ax_b)
    ax_b.set_title("Transport-onset warning window (Electron)", fontsize=11, loc="left")

    cor = panel_c(ax_c)
    ax_c.set_title("Downstream early warning", fontsize=11, loc="left", pad=8)

    out_png = FIGDIR / "figF8_virtualq.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"wrote {out_png}")
    print("(a) virtual-Q skill:")
    for sid, rr in skill.items():
        print(f"    {sid}: b={rr['b']:.2f} r={rr['r']:.2f} NSE_logQ={rr['nse_logQ']:.2f}")
    print("(b) transport-onset lead over peak Q:")
    for s, v in haz["leads"].items():
        print(f"    {s}: {v:+.1f} h")
    print(f"(c) downstream corridor lead (Electron->Puyallup peak): {cor['corridor_lead_h']:.0f} h")
    print(f"    upstream seismic leads downstream peak: {cor['seismic_lead_h']:.0f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
