#!/usr/bin/env python3
"""F6 — Event-scale timing of braided-channel reorganization (manuscript SPINE figure).

Composite, full-page figure that merges three previously-separate analyses into the
paper's central timing figure:

  (a) [top, wide]  Matched-discharge reorganization timing at the Puyallup PR cluster
                   (old fig22, workflow 21). Discharge strip with the AR pulses shaded
                   and the main peak marked, then the first-rising-limb-referenced
                   matched-discharge baseline c(t) (snapshots) with the cached logistic
                   step fit per station, onset (dashed), and step midpoint t50 with
                   the bootstrap CI band.
  (b) [bottom-left]  Width-stage hysteresis loop (old fig28, workflow 27): SAR
                   wetted-width proxy (PR01/PR02 ÷PR03 ÷Nov) vs epoch-mean gage stage,
                   arrows in time order. PR01 ends below pre-flood baseline (abandoned
                   thread); PR02 returns to ~baseline.
  (c) [bottom-right]  Recession-rate clogging contrast (old fig25 panel B,
                   workflow 24): 5–15 Hz power per PR station (÷ pre-flood median) over
                   the event, with the fast AR1 recession (no step) vs the slow AR3
                   recession (where the reorganization step lands) shaded/annotated.

Rebuilds entirely OFFLINE from cached JSON/CSV. No suptitle, no in-figure caption
(the manuscript supplies it); only short panel letter-titles.

Usage: pixi run python workflows/32_figF6_timing.py
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
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.analysis import estimate_pq_lag, load_timeseries  # noqa: E402
from riverseis.figstyle import paper_style  # noqa: E402

RESULTS = ROOT / "notebooks" / "data" / "results"
CACHE = ROOT / "notebooks" / "data" / "braid_cache"
DATA = ROOT / "notebooks" / "data"
CONFIG = ROOT / "config"
FIGDIR = ROOT / "paper" / "figures"
BAND = "5.0-15.0Hz"
GAGE_IV = "usgs_iv_12092000_2025-12-01_2026-01-01.csv"

STATIONS = ["CC.PR01", "CC.PR02", "CC.PR03"]
ST_COLORS = {"CC.PR01": "#e31a1c", "CC.PR02": "#33a02c", "CC.PR03": "#1f78b4"}
AR_COLORS = {"AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}

WIN = ("2025-12-05T00:00:00+00:00", "2025-12-14T00:00:00+00:00")
REF_END = "2025-12-09T03:30:00+00:00"          # first-pulse (AR1) peak = pre-reorg reference
LEVELS = tuple(range(60, 165, 10))             # matched-discharge crossings
GAGE_COORD = (46.9037, -122.0351)
COORDS = {"CC.PR01": (46.9101, -122.0376), "CC.PR02": (46.9183, -122.0487),
          "CC.PR03": (46.9034, -122.0327)}

# fig25 (panel c) windows
EVENT = ("2025-12-08T00:00:00+00:00", "2025-12-13T12:00:00+00:00")
PRE = ("2025-12-05T00:00:00+00:00", "2025-12-08T00:00:00+00:00")
RECESSIONS = {"AR1": ("2025-12-09T03:30:00+00:00", "2025-12-10T00:00:00+00:00"),
              "AR3": ("2025-12-11T15:00:00+00:00", "2025-12-13T00:00:00+00:00")}

# fig28 (panel b) epoch windows
EPOCH_WIN = {
    "Dec 1–8": ("2025-12-01", "2025-12-08 23:59"),
    "Dec 9–12 (AR peak)": ("2025-12-09", "2025-12-12 23:59"),
    "Dec 13–20": ("2025-12-13", "2025-12-20 23:59"),
    "Dec 21–31": ("2025-12-21", "2025-12-31 23:59"),
}
ORDER = ["Nov 16–30", "Dec 1–8", "Dec 9–12 (AR peak)", "Dec 13–20", "Dec 21–31"]


# ---------------------------------------------------------------------------
# (a) matched-discharge baseline c(t) — reconstructed exactly as in workflow 21
# ---------------------------------------------------------------------------
def _haversine_km(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h))) / 1000.0


def matched_q_baseline(sid: str):
    """First-rising-limb-referenced matched-discharge baseline c(t) for one station.

    Mirrors workflow 21: residual r(t)=log10 P-(a+b log10 Q) sampled at ref-level
    crossings, each level referenced to its first rising-limb value. For the Puyallup
    gage (co-located, <3 km) no flood-wave lag correction is applied (matches cache).
    Returns (df[t,Qstar,c], Q(t) series).
    """
    t0, t1 = pd.Timestamp(WIN[0]), pd.Timestamp(WIN[1])
    ref_end = pd.Timestamp(REF_END)
    j = load_timeseries(RESULTS / f"{sid}_{BAND}_timeseries.csv")
    j = j[(j.index >= t0) & (j.index <= t1)].copy()
    dist_km = _haversine_km(COORDS[sid], GAGE_COORD)
    lag_min, _ = estimate_pq_lag(np.log10(j["P"].clip(lower=1e-30)),
                                 np.log10(j["Q"].clip(lower=1e-6)))
    if lag_min != 0 and dist_km >= 3.0:          # co-located gage -> no correction
        qs = j["Q"].copy()
        qs.index = qs.index + pd.Timedelta(minutes=lag_min)
        j["Q"] = qs.reindex(j.index, method="nearest", tolerance=pd.Timedelta("10min"))
        j = j.dropna(subset=["Q"])
    lp = np.log10(j["P"].clip(lower=1e-30))
    lq = np.log10(j["Q"].clip(lower=1e-6))
    b, a = np.polyfit(lq.values, lp.values, 1)
    r = lp - (a + b * lq)
    q, t = j["Q"].values, j.index
    recs = []
    for lvl in LEVELS:
        for ci in np.where(np.diff(np.sign(q - lvl)) != 0)[0]:
            tc = t[ci]
            w = (t >= tc - pd.Timedelta("1h")) & (t <= tc + pd.Timedelta("1h"))
            if w.sum() >= 3:
                recs.append((tc, lvl, float(np.median(r.values[w]))))
    df = pd.DataFrame(recs, columns=["t", "Qstar", "resid"]).sort_values("t")
    rise = df[df.t <= ref_end]
    ref = rise.groupby("Qstar")["resid"].first()
    df = df[df.Qstar.isin(ref.index)].copy()
    df["c"] = df.resid - df.Qstar.map(ref)
    return df, j["Q"]


def _sigmoid(t, c0, d, tm, tau):
    return c0 + d / (1.0 + np.exp(-(t - tm) / tau))


def fit_step(th, y, lo_h):
    sign = np.sign(np.mean(y[th > th.max() - 24])) if len(y) else 1.0
    p0 = [0.0, 0.2 * (sign or 1.0), float(np.median(th)), 10.0]
    popt, _ = curve_fit(_sigmoid, th, y, p0=p0, maxfev=30000,
                        bounds=([-1, -1, lo_h, 1.5], [1, 1, th.max(), 60]))
    return popt


# ---------------------------------------------------------------------------
# (b) width-stage helpers (workflow 27)
# ---------------------------------------------------------------------------
def epoch_stage():
    df = pd.read_csv(DATA / GAGE_IV, parse_dates=["time_utc"]).set_index("time_utc")
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    h = df["gage_height_ft"]
    out = {}
    for lab, (a, b) in EPOCH_WIN.items():
        s = h.loc[a:b]
        out[lab] = dict(mean=float(s.mean()), lo=float(s.min()), hi=float(s.max()))
    return out


# ---------------------------------------------------------------------------
# (c) gage loader (workflow 24)
# ---------------------------------------------------------------------------
def load_gage_q():
    g = pd.read_csv(DATA / GAGE_IV, parse_dates=["time_utc"]).set_index("time_utc")
    return (pd.to_numeric(g["discharge_cfs"], errors="coerce") * 0.0283168).sort_index()


def main() -> int:
    paper_style()
    t0 = pd.Timestamp(WIN[0])
    q_peak = pd.Timestamp(REF_END)
    lo_h = (q_peak - t0).total_seconds() / 3600.0
    ars = json.loads((CONFIG / "ar_windows.json").read_text())
    reorg = json.loads((CONFIG / "braided_reorg_timing_puyallup.json").read_text())
    haz = json.loads((CONFIG / "hazard_timing_clogging.json").read_text())

    # ----- layout: full page, 2 conceptual rows ------------------------------
    # Top "row" = panel (a) split into a discharge strip + 3 station baseline rows.
    # Bottom row = (b) width-stage loop | (c) recession clogging.
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.32, 1.0], hspace=0.30, wspace=0.22,
        left=0.07, right=0.965, top=0.965, bottom=0.075)

    # (a) nested gridspec: strip + 3 stations
    gsa = gs[0, :].subgridspec(4, 1, height_ratios=[0.9, 1, 1, 1], hspace=0.0)
    ax_q = fig.add_subplot(gsa[0])
    ax_st = [fig.add_subplot(gsa[i + 1], sharex=ax_q) for i in range(3)]

    # ---------------- (a) discharge strip ----------------
    _, q0 = matched_q_baseline(STATIONS[0])
    ax_q.plot(q0.index, q0.values, color="#222", lw=1.5)
    ax_q.set_ylabel("Q\n(m³ s⁻¹)", fontsize=10)
    ax_q.axvline(q_peak, color="#D55E00", lw=1.4, zorder=5)
    for w in ars:
        if w["label"] in AR_COLORS:
            ax_q.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                         color=AR_COLORS[w["label"]], alpha=0.13, zorder=0)
            ax_q.text(pd.Timestamp(w["peak"]), q0.max() * 0.98, w["label"],
                      ha="center", va="top", fontsize=9, color=AR_COLORS[w["label"]],
                      fontweight="bold")
    ax_q.text(q_peak, q0.max() * 0.40, " peak Q", color="#D55E00", fontsize=8.5, va="center")
    ax_q.set_title("(a)  Matched-discharge reorganization timing — Puyallup PR cluster",
                   fontsize=12, loc="left", fontweight="bold")
    ax_q.tick_params(labelbottom=False)
    ax_q.margins(x=0.005)

    # ---------------- (a) per-station matched-Q baselines ----------------
    for ax, sid in zip(ax_st, STATIONS):
        d = reorg["stations"][sid]
        df, _ = matched_q_baseline(sid)
        th = (df.t - t0).dt.total_seconds().values / 3600.0
        y = df.c.values
        # reuse cached fit parameters where available; refit for the smooth curve
        popt = fit_step(th, y, lo_h)
        order = np.argsort(th)
        yhat = _sigmoid(th, *popt)

        ax.axhline(0, color="0.5", lw=0.6, ls=":")
        for w in ars:                       # faint AR shading for vertical registration
            if w["label"] in AR_COLORS:
                ax.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                           color=AR_COLORS[w["label"]], alpha=0.06, zorder=0)
        ax.axvline(q_peak, color="#D55E00", lw=1.0, alpha=0.7)
        ax.scatter(df.t, y, s=12, c="#9a9a9a", alpha=0.6, zorder=2,
                   label="matched-Q snapshots")
        ax.plot(df.t.values[order], yhat[order], color=ST_COLORS[sid], lw=2.2,
                zorder=3, label="logistic step fit")

        onset = d.get("onset_utc")
        if onset:
            ax.axvline(pd.Timestamp(onset), color="#CC79A7", lw=1.2, ls="--", zorder=4)
        reversible = d["reversible"]
        if not reversible:
            t50 = pd.Timestamp(d["step_t50_utc"])
            ax.axvline(t50, color="#009E73", lw=1.8, zorder=4)
            ci = d.get("step_t50_lag_CI_h")
            if ci:                          # CI band, lag (h vs q_peak) -> absolute time
                lo = q_peak + pd.Timedelta(hours=ci[0])
                hi = q_peak + pd.Timedelta(hours=ci[1])
                ax.axvspan(lo, hi, color="#009E73", alpha=0.18, zorder=1)
            lag = d["step_t50_lag_vs_Qpeak_h"]
            head = (f"{sid}: persistent step  t50 {t50:%m-%d %H:%M}Z "
                    f"(+{lag:.0f} h vs peak)  Δ={d['magnitude_log10']:+.2f} log₁₀  R²={d['r2']:.2f}")
        else:
            head = (f"{sid}: reversible / transient — returns to baseline "
                    f"(end-state {d['persistent_offset_log10']:+.2f} log₁₀)")
        ax.text(0.012, 0.90, head, transform=ax.transAxes, fontsize=9.0,
                va="top", ha="left", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))
        ax.set_ylabel("c = Δlog₁₀P", fontsize=9.5)
        ax.set_ylim(min(-0.55, y.min() - 0.08), max(0.85, y.max() + 0.08))
        ax.margins(x=0.005)
        if sid != STATIONS[-1]:
            ax.tick_params(labelbottom=False)

    # legend + onset key on the first station panel only
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="#9a9a9a", mec="none", ms=6,
                   label="matched-Q snapshots"),
        plt.Line2D([], [], color="#444", lw=2.2, label="logistic step fit"),
        plt.Line2D([], [], color="#CC79A7", lw=1.2, ls="--", label="onset"),
        plt.Line2D([], [], color="#009E73", lw=1.8, label="step t50 (±95% CI)"),
        plt.Line2D([], [], color="#D55E00", lw=1.0, label="peak Q"),
    ]
    ax_st[0].legend(handles=handles, loc="lower right", fontsize=7.8, ncol=3,
                    framealpha=0.92, columnspacing=1.0, handlelength=1.4)
    ax_st[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_st[-1].set_xlabel("December 2025 (UTC)", fontsize=10)

    # ---------------- (b) width-stage hysteresis ----------------
    axB = fig.add_subplot(gs[1, 0])
    series = pd.read_csv(CACHE / "puyallup_december_series.csv").set_index("epoch_label")
    st = epoch_stage()
    st_full = {"Nov 16–30": dict(mean=st["Dec 1–8"]["mean"], lo=np.nan, hi=np.nan), **st}
    cols = {"PR01_rel_to_PR03": ("#e31a1c", "CC.PR01"),
            "PR02_rel_to_PR03": ("#33a02c", "CC.PR02")}
    H = np.array([st_full[e]["mean"] for e in ORDER])
    for key, (c, lab) in cols.items():
        W = np.array([series.loc[e, key] for e in ORDER])
        lo = np.array([st_full[e]["lo"] for e in ORDER])
        hi = np.array([st_full[e]["hi"] for e in ORDER])
        axB.errorbar(H[1:], W[1:], xerr=[H[1:] - lo[1:], hi[1:] - H[1:]],
                     fmt="none", ecolor=c, alpha=0.30, lw=1, capsize=2, zorder=2)
        for i in range(len(ORDER) - 1):
            axB.annotate("", xy=(H[i + 1], W[i + 1]), xytext=(H[i], W[i]),
                         arrowprops=dict(arrowstyle="->", color=c, lw=1.6, alpha=0.85),
                         zorder=3)
        axB.plot(H[0], W[0], "o", mfc="white", mec=c, mew=1.6, ms=8, zorder=4)
        axB.plot(H[1:], W[1:], "o", color=c, ms=7, zorder=4, label=lab)
    axB.axhline(1.0, color="0.6", ls=":", lw=1)
    axB.annotate("pre-flood width (Nov, ref)", (H.min(), 1.0), xytext=(2, 4),
                 textcoords="offset points", fontsize=8, color="0.4")
    Wp = np.array([series.loc[e, "PR01_rel_to_PR03"] for e in ORDER])
    for e, x, y in zip(ORDER, H, Wp):
        dx, dy = (6, 2)
        if e == "Dec 9–12 (AR peak)":
            dx, dy = (-8, -12)
        axB.annotate(e.replace(" (AR peak)", " (peak)"), (x, y), xytext=(dx, dy),
                     ha="right" if dx < 0 else "left", textcoords="offset points",
                     fontsize=7.5, color="0.3")
    axB.set_xlabel("gage stage at Electron (ft; epoch mean, whiskers = min–max)", fontsize=10)
    axB.set_ylabel("wetted active-channel width proxy\n(SAR area ÷PR03 ÷Nov baseline)", fontsize=10)
    axB.set_title("(b)  Width–stage hysteresis: rise widens, fall does not recover",
                  fontsize=11.5, loc="left", fontweight="bold")
    axB.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # ---------------- (c) recession-rate clogging ----------------
    axC = fig.add_subplot(gs[1, 1])
    for w in ars:
        if w["label"] in AR_COLORS:
            axC.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                        color=AR_COLORS[w["label"]], alpha=0.08, zorder=0)
    for s in STATIONS:
        j = load_timeseries(RESULTS / f"{s}_{BAND}_timeseries.csv")
        p = j["P"][(j.index >= pd.Timestamp(EVENT[0])) & (j.index <= pd.Timestamp(EVENT[1]))]
        pre = j["P"][(j.index >= pd.Timestamp(PRE[0])) & (j.index < pd.Timestamp(PRE[1]))].median()
        axC.semilogy(p.index, (p / pre).rolling("2h", center=True, min_periods=4).median(),
                     color=ST_COLORS[s], lw=1.4, label=s)
    axC.set_ylabel("5–15 Hz power /\npre-flood median", fontsize=10)
    axC.set_xlabel("December 2025 (UTC)", fontsize=10)
    rec = haz["recessions"]
    for lab, (rt0, rt1) in RECESSIONS.items():
        mid = pd.Timestamp(rt0) + (pd.Timestamp(rt1) - pd.Timestamp(rt0)) / 2
        r = rec[lab]
        axC.axvspan(pd.Timestamp(rt0), pd.Timestamp(rt1), color="0.5",
                    alpha=0.05 if lab == "AR1" else 0.14, zorder=0)
        axC.text(mid, axC.get_ylim()[1] * 0.55,
                 f"{lab} fall\n{r['dqdt']:+.0f} m³/s/h\n({r['hours']:.0f} h)",
                 ha="center", va="top", fontsize=8.5,
                 color="#444" if lab == "AR1" else "#7a4f00", fontweight="bold")
    steps = haz["reorg_steps"]
    for s, t in steps.items():
        axC.axvline(pd.Timestamp(t), color="#6a3d9a", lw=1.5, ls="-", zorder=5)
    axC.text(pd.Timestamp(list(steps.values())[0]), axC.get_ylim()[0] * 1.7,
             " braid reorganization\n (avulsion step)", color="#6a3d9a",
             fontsize=8.8, va="bottom", ha="left")
    axC.set_title("(c)  Slow AR3 recession hosts the reorganization (clogging)",
                  fontsize=11.5, loc="left", fontweight="bold")
    axC.legend(loc="upper right", fontsize=8.5, ncol=3, framealpha=0.9,
               columnspacing=0.9, handlelength=1.2)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    out = FIGDIR / "figF6_timing.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")

    # console summary for the manuscript caption numbers
    print("\n=== F6 caption numbers ===")
    for sid in STATIONS:
        d = reorg["stations"][sid]
        if d["reversible"]:
            print(f"(a) {sid}: reversible (end-state {d['persistent_offset_log10']:+.2f} log10), "
                  f"onset +{d['onset_lag_h']} h")
        else:
            print(f"(a) {sid}: onset +{d['onset_lag_h']} h; step t50 +"
                  f"{d['step_t50_lag_vs_Qpeak_h']} h (CI {d['step_t50_lag_CI_h']}); "
                  f"Δ={d['magnitude_log10']:+.2f} log10; R²={d['r2']}")
    for key, lab in [("PR01_rel_to_PR03", "PR01"), ("PR02_rel_to_PR03", "PR02")]:
        peak = series.loc["Dec 9–12 (AR peak)", key]
        late = series.loc["Dec 21–31", key]
        print(f"(b) {lab}: peak={peak:.2f}x  late={late:.2f}x")
    for lab in ("AR1", "AR3"):
        r = rec[lab]
        print(f"(c) {lab} recession: {r['q0']}→{r['q1']} m³/s over {r['hours']} h "
              f"({r['dqdt']:+.1f} m³/s/h, {r['pct_per_h']:+.1f} %/h)")
    print("(c) reorg steps:", {s: t[:16] for s, t in steps.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
