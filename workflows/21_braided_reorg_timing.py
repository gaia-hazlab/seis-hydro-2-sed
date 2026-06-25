#!/usr/bin/env python3
"""Event-scale TIMING of braided-channel reorganization, resolved seismically.

Motivation (fluvial-geomorphology angle, June-2026 referee point)
-----------------------------------------------------------------
Workflow 18 establishes *that* the braided Puyallup source reorganizes across the
Dec-2025 compound flood: the per-AR P–Q loops are near-reversible (HI~0) but the
cross-AR baseline OFFSET steps up irreversibly (an avulsion / lateral-migration
signature, not transport hysteresis). That reports the *existence* of migration.

The seismic network's unique affordance, though, is CONTINUOUS, sub-daily,
spatially-distributed illumination of the bed — something neither gages nor
before/after satellite snapshots provide. So we can ask the question a
geomorphologist currently cannot answer from gages or imagery:

    WHEN, relative to the hydrograph, does the active thread reorganize — on the
    rising limb, at peak, or on recession? Does it LEAD or LAG peak discharge? Is
    the drift STEPWISE (a discrete avulsion) or GRADUAL (continuous reworking),
    and is it PERSISTENT (irreversible) or REVERSIBLE? Does it propagate?

Method — matched-discharge baseline tracking
--------------------------------------------
The geometry signal is the part of seismic power NOT explained by discharge:

    r(t) = log10 P(t) - (a + b * log10 Q(t))            [pooled P–Q rating a,b]

r(t) still mixes the geometry baseline with within-event hysteresis and rating
curvature (a function of Q only). To cancel the Q-dependence we track the residual
at MATCHED DISCHARGE: for reference levels Q* that the compound hydrograph crosses
repeatedly DURING the event (the inter-pulse troughs and limbs), we sample r at
every crossing and reference each level to its FIRST (first-rising-limb) value —
the pre-reorganization channel. The pooled series c(t) is then a sequence of
same-stage "snapshots": any change is source/channel geometry, not discharge.

On c(t) we measure ONSET (first sustained >2σ departure), the DOMINANT step via a
logistic fit  c0 + Δ/(1+exp(-(t-t50)/τ))  (midpoint t50 -> lead/lag vs peak Q;
4·τ -> stepwise vs gradual; Δ, R² and a REVERSIBLE flag), with a bootstrap CI, and
cross-station PROPAGATION.

Runs on any braided cluster via --basin (puyallup | nisqually). NB the basins have
different compound structure: Puyallup's main peak is the FIRST pulse (12-09);
Nisqually's main peak is the SECOND pulse (12-10), so the rising-limb reference uses
the first pulse while lead/lag is measured against the larger, later peak.

Outputs config/braided_reorg_timing_<basin>.json and
paper/figures/fig22_braided_reorg_timing[_<basin>].png.

Usage: pixi run python workflows/21_braided_reorg_timing.py [--basin puyallup|nisqually]
"""
from __future__ import annotations

import argparse
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
FIGDIR = ROOT / "paper" / "figures"
CONFIG = ROOT / "config"
BAND = "5.0-15.0Hz"
AR_COLORS = {"AR1": "#0072B2", "AR2": "#56B4E9", "AR3": "#E69F00"}

# Per-basin configuration. q_peak = main hydrograph peak (for lead/lag); ref_end =
# end of the FIRST rising limb / first-pulse peak (defines the pre-reorganization
# reference). levels = matched discharge crossings spanning trough..below-peak.
BASINS = {
    "puyallup": dict(
        stations=["CC.PR01", "CC.PR02", "CC.PR03"],
        gage="Puyallup nr Electron 12092000",
        q_peak="2025-12-09T03:30:00+00:00",     # main peak = AR1 (first pulse)
        ref_end="2025-12-09T03:30:00+00:00",
        win=("2025-12-05T00:00:00+00:00", "2025-12-14T00:00:00+00:00"),
        levels=tuple(range(60, 165, 10)),
        fig="fig22_braided_reorg_timing.png",
    ),
    "nisqually": dict(
        stations=["UW.LON", "CC.GTWY"],
        gage="Nisqually nr National 12082500",
        q_peak="2025-12-10T14:15:00+00:00",     # main peak = AR2 (second pulse)
        ref_end="2025-12-09T09:00:00+00:00",    # first-pulse (AR1) peak
        win=("2025-12-05T00:00:00+00:00", "2025-12-14T00:00:00+00:00"),
        levels=tuple(range(70, 320, 20)),       # bigger river; low levels (70–140)
        # anchor the AR1-rise reference AND the post-event recession end-state
        fig="fig22_braided_reorg_timing_nisqually.png",
    ),
}

# station coordinates (lat, lon) for cross-station separation
COORDS = {
    "CC.PR01": (46.9101, -122.0376), "CC.PR02": (46.9183, -122.0487),
    "CC.PR03": (46.9034, -122.0327), "UW.LON": (46.7506, -121.8096),
    "CC.GTWY": (46.7402, -121.917),
}
# the discharge gage each station is paired to (lat, lon). The Puyallup gage is
# essentially CO-LOCATED with the PR cluster (no flood-wave travel-time lag); the
# Nisqually gage (National) is 10–20 km DOWNSTREAM of UW.LON/GTWY, so gage Q lags
# the discharge actually passing the station by the wave travel time — a P–Q
# misalignment that inflates matched-Q hysteresis unless corrected.
GAGE_COORDS = {"puyallup": (46.9037, -122.0351), "nisqually": (46.7526, -122.0837)}


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    R = 6371000.0
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def matched_q_baseline(sid: str, cfg: dict, t0: pd.Timestamp,
                       q_peak: pd.Timestamp, ref_end: pd.Timestamp,
                       apply_lag: bool = True):
    """First-rising-limb-referenced, matched-discharge baseline c(t) for one station.

    When apply_lag, the gage discharge is shifted to the station by the estimated
    flood-wave travel-time lag before forming the residual (essential where the gage
    is far downstream, as on the Nisqually). Returns (crossings DataFrame
    [t, Qstar, resid, c], pooled slope b, r(t), Q(t), lag_info dict).
    """
    j = load_timeseries(RESULTS / f"{sid}_{BAND}_timeseries.csv")
    j = j[(j.index >= t0) & (j.index <= pd.Timestamp(cfg["win"][1]))].copy()
    gage = GAGE_COORDS[cfg["basin"]]
    dist_km = _haversine_m(COORDS[sid], gage) / 1000.0
    lag_min, lag_r = estimate_pq_lag(np.log10(j["P"].clip(lower=1e-30)),
                                     np.log10(j["Q"].clip(lower=1e-6)))
    # Only correct where a real flood-wave travel time exists: at a co-located gage
    # (<3 km, as on the Puyallup) the estimated lag is sub-resolution noise and
    # shifting by it only injects error, so leave Q unshifted there.
    applied = bool(apply_lag and lag_min != 0 and dist_km >= 3.0)
    if applied:
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
    for qs_lvl in cfg["levels"]:
        for ci in np.where(np.diff(np.sign(q - qs_lvl)) != 0)[0]:
            tc = t[ci]
            w = (t >= tc - pd.Timedelta("1h")) & (t <= tc + pd.Timedelta("1h"))
            if w.sum() >= 3:
                recs.append((tc, qs_lvl, float(np.median(r.values[w]))))
    df = pd.DataFrame(recs, columns=["t", "Qstar", "resid"]).sort_values("t")
    # reference each level to its first-rising-limb crossing (pre-reorganization)
    rise = df[df.t <= ref_end]
    ref = rise.groupby("Qstar")["resid"].first()
    df = df[df.Qstar.isin(ref.index)].copy()
    df["c"] = df.resid - df.Qstar.map(ref)
    # flood-wave celerity implied by the lag and the station→gage distance
    celerity = (dist_km * 1000.0 / abs(lag_min * 60.0)) if (applied and lag_min) else None
    lag_info = dict(lag_min=int(lag_min), lag_xcorr_r=round(lag_r, 2),
                    station_gage_km=round(dist_km, 1), applied=applied,
                    implied_celerity_m_s=round(celerity, 2) if celerity else None)
    return df, float(b), r, j["Q"], lag_info


def supply_shutoff_decline(sid: str, q_peak: pd.Timestamp) -> float:
    """Raw log10-P change from ~2 days to ~7 days after peak — the rain→snow supply cut.

    After the warm AR the freezing level drops and precipitation falls (and stays) as
    snow, so drainage-wide runoff — and the sediment it mobilizes — collapses. Matched-Q
    already controls for discharge, but it does NOT control for *what sources that Q*:
    storm flow is sediment-loaded (loud), the post-event snowmelt-free recession is
    sediment-starved (quiet), so the bed goes quiet at equal Q. This rough metric flags
    that supply shutoff so a NEGATIVE matched-Q "step" can be checked against it (it
    confounds negative steps; positive geometric steps survive it and are conservative).
    The higher, colder Nisqually converts more of its area to snow -> larger effect.
    Returns NaN if the record does not extend far enough past the peak.
    """
    j = load_timeseries(RESULTS / f"{sid}_{BAND}_timeseries.csv")
    lp = np.log10(j["P"].clip(lower=1e-30))
    early = lp[(lp.index >= q_peak + pd.Timedelta("2d")) & (lp.index < q_peak + pd.Timedelta("3d"))]
    late = lp[lp.index >= q_peak + pd.Timedelta("7d")]
    if len(early) < 6 or len(late) < 6:
        return float("nan")
    return float(late.median() - early.median())


def _sigmoid(t, c0, d, tm, tau):
    return c0 + d / (1.0 + np.exp(-(t - tm) / tau))


def fit_step(th: np.ndarray, y: np.ndarray, lo_h: float) -> np.ndarray:
    sign = np.sign(np.mean(y[th > th.max() - 24])) if len(y) else 1.0
    p0 = [0.0, 0.2 * (sign or 1.0), float(np.median(th)), 10.0]
    popt, _ = curve_fit(_sigmoid, th, y, p0=p0, maxfev=30000,
                        bounds=([-1, -1, lo_h, 1.5], [1, 1, th.max(), 60]))
    return popt


def detect_onset(df: pd.DataFrame, q_peak: pd.Timestamp, ref_end: pd.Timestamp):
    """First post-peak crossing where |c| rises >2σ above the reference band and stays."""
    ref = df[df.t <= ref_end]["c"]
    sd = float(ref.std(ddof=1)) if len(ref) > 2 else 0.05
    thr = 2.0 * max(sd, 0.02)
    post = df[df.t > q_peak]
    for i, (_, row) in enumerate(post.iterrows()):
        later = post.iloc[i:]
        if abs(row.c) > thr and abs(float(later["c"].median())) > thr:
            return row.t
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basin", choices=list(BASINS), default="puyallup")
    args = ap.parse_args()
    cfg = BASINS[args.basin]
    cfg["basin"] = args.basin
    paper_style()

    stations = cfg["stations"]
    t0 = pd.Timestamp(cfg["win"][0])
    q_peak = pd.Timestamp(cfg["q_peak"])
    ref_end = pd.Timestamp(cfg["ref_end"])
    lo_h = (q_peak - t0).total_seconds() / 3600.0      # steps can't precede the peak's window
    ars = json.loads((CONFIG / "ar_windows.json").read_text())
    rng = np.random.default_rng(0)

    out: dict = {
        "basin": args.basin, "gage": cfg["gage"], "band": BAND,
        "q_peak_utc": str(q_peak), "ref_levels_m3s": list(cfg["levels"]),
        "method": ("matched-discharge crossings, first-rising-limb-referenced; "
                   "logistic step fit with bootstrap CI; reversibility from sign(Δ)/R²"),
        "stations": {},
    }
    fig, axes = plt.subplots(len(stations) + 1, 1, figsize=(9.5, 3 + 2.6 * len(stations)),
                             sharex=True,
                             gridspec_kw={"height_ratios": [1.05] + [1] * len(stations)})

    # top: hydrograph + AR windows
    _, _, _, q0, _ = matched_q_baseline(stations[0], cfg, t0, q_peak, ref_end)
    ax0 = axes[0]
    ax0.plot(q0.index, q0.values, color="#222", lw=1.4)
    ax0.set_ylabel("Q  (m³ s⁻¹)")
    ax0.axvline(q_peak, color="#D55E00", lw=1.3)
    for w in ars:
        if w["label"] in AR_COLORS:
            ax0.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]),
                        color=AR_COLORS[w["label"]], alpha=0.12)
            ax0.text(pd.Timestamp(w["peak"]), q0.max() * 0.97, w["label"],
                     ha="center", va="top", fontsize=9, color=AR_COLORS[w["label"]])
    ax0.set_title(f"Event-scale timing of braided-channel reorganization — "
                  f"{args.basin.capitalize()} cluster, Dec-2025", fontsize=12)

    step_times = {}
    for ax, sid in zip(axes[1:], stations):
        df, b, r, q, lag_info = matched_q_baseline(sid, cfg, t0, q_peak, ref_end)
        th = (df.t - t0).dt.total_seconds().values / 3600.0
        y = df.c.values
        popt = fit_step(th, y, lo_h)
        yhat = _sigmoid(th, *popt)
        r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        t50 = t0 + pd.Timedelta(hours=float(popt[2]))
        tau, delta = float(popt[3]), float(popt[1])
        lead_lag = (t50 - q_peak).total_seconds() / 3600.0
        onset = detect_onset(df, q_peak, ref_end)

        t50s, taus, deltas = [], [], []
        for _ in range(400):
            idx = rng.integers(0, len(y), len(y))
            try:
                p = fit_step(th[idx], y[idx], lo_h)
                t50s.append(p[2]); taus.append(p[3]); deltas.append(p[1])
            except Exception:
                pass
        lo, hi = np.percentile(t50s, [2.5, 97.5]) if t50s else (np.nan, np.nan)
        # Persistence is decided by the END-STATE offset (median of the last day of
        # matched-Q snapshots vs the rising-limb reference), NOT the sigmoid Δ: a
        # station whose baseline returns to ~0 had a transient/reversible excursion
        # (e.g. supply/timing hysteresis), whereas a sustained offset of either sign
        # is an irreversible reorganization (thread toward = +, thread away = −).
        tail = df[df.t >= df.t.max() - pd.Timedelta("24h")]["c"]
        persistent_offset = float(tail.median()) if len(tail) else float("nan")
        # reversible/unresolved if the baseline returns to ~0, OR the logistic step
        # does not fit (R²<0.25), OR the fitted step direction disagrees with the
        # end-state sign (sign(Δ)≠sign(offset) → a non-monotone, oscillating excursion
        # whose tail median is not a trustworthy persistent offset, e.g. a steep reach
        # whose limb swings survive lag correction).
        reversible = bool(abs(persistent_offset) < 0.10 or r2 < 0.25
                          or (delta * persistent_offset < 0))
        direction = "positive" if persistent_offset >= 0 else "negative"

        supply_drop = supply_shutoff_decline(sid, q_peak)

        out["stations"][sid] = dict(
            b_pooled=round(b, 2), n_crossings=int(len(y)),
            lag_correction=lag_info,
            persistent_offset_log10=round(persistent_offset, 3),
            direction=None if reversible else direction,
            onset_utc=str(onset) if onset is not None else None,
            onset_lag_h=round((onset - q_peak).total_seconds() / 3600.0, 1)
            if onset is not None else None,
            step_t50_utc=str(t50), step_t50_lag_vs_Qpeak_h=round(lead_lag, 1),
            step_t50_lag_CI_h=[round((t0 + pd.Timedelta(hours=lo) - q_peak).total_seconds() / 3600.0, 1),
                               round((t0 + pd.Timedelta(hours=hi) - q_peak).total_seconds() / 3600.0, 1)]
            if t50s else None,
            transition_width_h=round(4 * tau, 1),
            tau_CI_h=[round(float(np.percentile(taus, 2.5)), 1),
                      round(float(np.percentile(taus, 97.5)), 1)] if taus else None,
            magnitude_log10=round(delta, 3),
            magnitude_CI=[round(float(np.percentile(deltas, 2.5)), 3),
                          round(float(np.percentile(deltas, 97.5)), 3)] if deltas else None,
            r2=round(float(r2), 2),
            character="stepwise" if 4 * tau < 24 else "gradual",
            reversible=reversible,
            supply_shutoff_decline_log10=round(supply_drop, 2) if np.isfinite(supply_drop) else None,
            supply_confounded=bool(
                (not reversible) and persistent_offset < 0 and np.isfinite(supply_drop)
                and supply_drop <= -0.3),       # negative step + same-sign supply shutoff
        )
        step_times[sid] = t50

        ax.scatter(df.t, y, s=14, c="#888", alpha=0.6, zorder=2, label="matched-Q snapshots")
        order = np.argsort(th)
        ax.plot(df.t.values[order], yhat[order], color="#0072B2", lw=2.0, zorder=3,
                label="logistic step fit")
        ax.axhline(0, color="0.5", lw=0.6, ls=":")
        ax.axvline(q_peak, color="#D55E00", lw=1.0, alpha=0.7)
        if not reversible:
            ax.axvline(t50, color="#009E73", lw=1.6, zorder=4)
            if t50s:
                ax.axvspan(t0 + pd.Timedelta(hours=lo), t0 + pd.Timedelta(hours=hi),
                           color="#009E73", alpha=0.15, zorder=1)
        if onset is not None:
            ax.axvline(onset, color="#CC79A7", lw=1.2, ls="--", zorder=4)
        if reversible:
            head = (f"{sid}: REVERSIBLE / transient — returns to baseline "
                    f"(end-state {persistent_offset:+.2f} log₁₀; |excursion|≤{max(abs(y.min()),abs(y.max())):.1f})")
        else:
            sgn = "lag" if lead_lag > 0 else "lead"
            head = (f"{sid}: persistent {direction} step {t50:%m-%d %H:%M}Z "
                    f"({abs(lead_lag):.0f} h {sgn} of Q-peak) · end-state {persistent_offset:+.2f} log₁₀ · "
                    f"{'stepwise' if 4*tau<24 else 'gradual'} (width≈{4*tau:.0f} h) · R²={r2:.2f}")
        ax.set_title(head, fontsize=9.5, loc="left")
        ax.set_ylabel("matched-Q baseline\nc = Δlog₁₀P")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(min(-0.6, y.min() - 0.1), max(0.85, y.max() + 0.1))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].set_xlabel("2025  (UTC)")

    # propagation across the cluster (persistent stations only)
    persist = [s for s in stations if not out["stations"][s]["reversible"]]
    order = sorted(persist, key=lambda s: step_times[s])
    prop = {"persistent_stations": persist, "order_earliest_first": order, "pairs": []}
    for i in range(len(order)):
        for j2 in range(i + 1, len(order)):
            s1, s2 = order[i], order[j2]
            dt_h = (step_times[s2] - step_times[s1]).total_seconds() / 3600.0
            dist_m = _haversine_m(COORDS[s1], COORDS[s2])
            cel = dist_m / (dt_h * 3600.0) if abs(dt_h) > 1e-6 else None
            prop["pairs"].append(dict(
                from_=s1, to=s2, dt_h=round(dt_h, 1), separation_m=round(dist_m),
                apparent_celerity_m_per_s=round(cel, 3) if cel else None))
    out["propagation"] = prop

    fig.tight_layout()
    fig.savefig(FIGDIR / cfg["fig"])
    plt.close(fig)
    out_json = CONFIG / f"braided_reorg_timing_{args.basin}.json"
    out_json.write_text(json.dumps(out, indent=2))

    print(f"\n=== Braided reorganization TIMING — {args.basin} ({cfg['gage']}) ===")
    for sid in stations:
        d = out["stations"][sid]
        li = d["lag_correction"]
        if li["applied"]:
            print(f"\n[{sid}] flood-wave lag {li['lag_min']:+d} min (r={li['lag_xcorr_r']}); "
                  f"gage {li['station_gage_km']} km downstream → celerity "
                  f"{li['implied_celerity_m_s']} m/s — Q shifted to station before matching")
        else:
            print(f"\n[{sid}] gage {li['station_gage_km']} km away (co-located) — "
                  f"no lag correction (est. {li['lag_min']:+d} min is sub-resolution noise)")
        if d["reversible"]:
            print(f"{sid}: REVERSIBLE/transient — end-state {d['persistent_offset_log10']:+.2f} log10 "
                  f"(returns to baseline; R²={d['r2']}) — no persistent step")
        else:
            flag = "  ⚠ SUPPLY-CONFOUNDED (rain→snow)" if d["supply_confounded"] else ""
            print(f"{sid}: persistent {d['direction']} step — onset {str(d['onset_utc'])[:16]}Z "
                  f"(+{d['onset_lag_h']:.0f} h); t50 {d['step_t50_utc'][:16]}Z "
                  f"(+{d['step_t50_lag_vs_Qpeak_h']:.0f} h vs Q-peak, CI {d['step_t50_lag_CI_h']} h){flag}")
            print(f"   end-state {d['persistent_offset_log10']:+.2f} log10; sigmoid Δ={d['magnitude_log10']:+.2f}; "
                  f"{d['character']} (width≈{d['transition_width_h']:.0f} h); R²={d['r2']}; "
                  f"supply-shutoff decline {d['supply_shutoff_decline_log10']} log10")
    if len(order) >= 2:
        print("\nPropagation (persistent stations, earliest first):", " -> ".join(order))
        for p in prop["pairs"]:
            print(f"   {p['from_']} -> {p['to']}: Δt={p['dt_h']:+.1f} h, sep={p['separation_m']} m, "
                  f"v_app={p['apparent_celerity_m_per_s']} m/s")
    print(f"\nwrote {FIGDIR / cfg['fig']} + {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
