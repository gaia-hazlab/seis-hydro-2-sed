#!/usr/bin/env python3
"""Composite figure F8 — Distributed sensing, virtual gage, and a routing
proof-of-concept.

  (a) Seismic *virtual discharge* vs the co-located gage for the three
      best-validated river-proximal stations, all on ONE log-y discharge axis.
      The 5-15 Hz seismic power is inverted through the per-station P-Q rating
      (a, b in config/virtual_q.json) to a virtual discharge series; per-station
      b / r / log-Q Nash-Sutcliffe (NSE) from config/virtual_q_fit.json quantify
      the skill.  Virtual-Q is clipped to the physical [1, 500] m3/s range and a
      known CC.PR03 dropout on 2025-12-17 is masked out.

  (b) Transport-onset warning window: the Electron hydrograph (Q + stage) through
      the Dec-2025 flood with a single shaded lead window from the 5-15 Hz
      bed-transport onset to peak Q (~5-7 h).  Framed as a diagnostic that the
      bed is mobilizing on the rising limb — NOT a forecast.
      Numbers from config/hazard_timing_clogging.json.

  (c) Routing concept: a deterministic single-event demonstration that routes
      upstream discharge (Electron 12092000) to downstream stage (Puyallup
      12101500) with a constant flood-wave lag tau, gain g, and the downstream
      rating curve.  An honest concept — not validated forecast skill.

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

CFS_TO_CMS = 0.0283168
FT_TO_M = 0.3048

# Panel (a): three best-validated virtual gages
VALID_STATIONS = ["CC.PR03", "CC.PR02", "CC.STYX"]
ST_COLORS = {"CC.PR01": "#e31a1c", "CC.PR02": "#33a02c",
             "CC.PR03": "#1f78b4", "CC.STYX": "#6a3d9a"}

# Virtual-Q physical clip and the known CC.PR03 dropout window
Q_CLIP = (1.0, 500.0)
PR03_DROPOUT = ("2025-12-17T08:00:00+00:00", "2025-12-17T18:00:00+00:00")

# Panel (b): hazard / transport-onset (Electron gage records Q and stage)
GAGE = DATA / "usgs_iv_12092000_2025-12-01_2026-01-01.csv"
EVENT = ("2025-12-08T00:00:00+00:00", "2025-12-13T12:00:00+00:00")

# Panel (c): routing concept gages
UP_GAGE = DATA / "usgs_iv_12092000_2025-12-01_2026-01-01.csv"   # Electron (upstream)
DN_GAGE = DATA / "usgs_iv_12101500_2025-12-01_2026-01-01.csv"   # Puyallup (downstream)
# Electron (RM~41) -> Puyallup (RM~10) ~ 31 river miles ~ 50 km
CHANNEL_KM = 50.0
ROUTE_WINDOW = ("2025-12-06T00:00:00+00:00", "2025-12-14T00:00:00+00:00")


# ---------------------------------------------------------------------------
def load_proxy_gage(sid: str) -> pd.DataFrame:
    """5-15 Hz proxy + gage discharge for a station, on the flood window."""
    df = pd.read_csv(RESULTS / f"{sid}_5.0-15.0Hz_timeseries.csv",
                     parse_dates=["time_utc"]).set_index("time_utc")
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    P = pd.to_numeric(df["proxy"], errors="coerce")
    Q = pd.to_numeric(df["gauge"], errors="coerce")
    j = pd.concat([P.rename("P"), Q.rename("Q")], axis=1).sort_index()
    j["Q"] = j["Q"].interpolate("linear", limit=12)
    return j.dropna()


def virtual_series(virt: dict, sid: str) -> pd.Series:
    """Virtual-Q series, clipped to physical range with known dropouts masked."""
    v = virt[sid]
    qs = pd.Series(v["q_seis"], index=pd.to_datetime(v["time"], utc=True)).sort_index()
    qs = pd.to_numeric(qs, errors="coerce")
    # mask the physically implausible sub-1 m3/s dropouts / spikes
    qs = qs.where((qs >= Q_CLIP[0]) & (qs <= Q_CLIP[1]))
    # explicit CC.PR03 dropout window on 2025-12-17
    if sid == "CC.PR03":
        lo, hi = pd.Timestamp(PR03_DROPOUT[0]), pd.Timestamp(PR03_DROPOUT[1])
        qs = qs.where(~((qs.index >= lo) & (qs.index <= hi)))
    return qs


# ---------------------------------------------------------------------------
def panel_a(ax, virt: dict, fit: dict) -> dict:
    """Seismic virtual discharge vs co-located gage, all stations on one axis."""
    fitmap = {r["station"]: r for r in fit}
    out = {}
    for sid in VALID_STATIONS:
        j = load_proxy_gage(sid)
        qs = virtual_series(virt, sid)
        # restrict virtual series to the gage-observed window for a fair overlay
        qs = qs[(qs.index >= j.index.min()) & (qs.index <= j.index.max())]
        col = ST_COLORS.get(sid, "#1f78b4")
        rr = fitmap[sid]
        # gage discharge: thin solid line; virtual-Q: same-colour translucent line
        ax.plot(j.index, j["Q"], color=col, lw=0.9, alpha=0.55, zorder=2)
        ax.plot(qs.index, qs.values, color=col, lw=1.4, alpha=0.95, zorder=3,
                label=f"{sid}   b={rr['b']:.2f}  r={rr['r']:.2f}  NSE={rr['nse_logQ']:.2f}")
        out[sid] = rr
    # one neutral proxy entry to disambiguate gage vs seismic styling
    ax.plot([], [], color="0.35", lw=0.9, alpha=0.6, label="gage Q (thin)")
    ax.plot([], [], color="0.15", lw=1.6, label="seismic virtual Q (bold)")
    ax.set_yscale("log")
    ax.set_ylim(*Q_CLIP)
    ax.set_ylabel("discharge  (m³/s)")
    ax.set_xlabel("December 2025 (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=10)
    ax.legend(fontsize=8.5, loc="lower center", ncol=2, framealpha=0.9)
    ax.set_title("(a) Seismic virtual discharge tracks the gage",
                 fontsize=12, loc="left")
    return out


def panel_b(ax) -> dict:
    """Transport-onset warning window: one shaded lead window onset -> peak."""
    haz = json.loads((CONFIG / "hazard_timing_clogging.json").read_text())
    g = pd.read_csv(GAGE, parse_dates=["time_utc"]).set_index("time_utc")
    q = (pd.to_numeric(g["discharge_cfs"], errors="coerce") * CFS_TO_CMS).sort_index()
    h = (pd.to_numeric(g["gage_height_ft"], errors="coerce") * FT_TO_M).sort_index()
    qe = q[(q.index >= pd.Timestamp(EVENT[0])) & (q.index <= pd.Timestamp(EVENT[1]))]
    he = h[(h.index >= pd.Timestamp(EVENT[0])) & (h.index <= pd.Timestamp(EVENT[1]))]
    q_peak_t = pd.Timestamp(haz["q_peak_utc"])

    onsets = {s: pd.Timestamp(t) for s, t in haz["transport_onset"].items()}
    leads = haz["lead_over_peak_h"]
    onset_first = min(onsets.values())

    # discharge on the left (primary) axis; label outward-left
    ax.plot(qe.index, qe.values, color="#222", lw=1.8, label="discharge Q", zorder=3)
    ax.set_ylabel("Q  (m³/s)")
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis="y", labelleft=True)

    # stage on the right twin axis; label outward-right
    axS = ax.twinx()
    axS.grid(False)
    axS.plot(he.index, he.values, color="#2c7fb8", lw=1.3, ls="--",
             label="stage", zorder=2)
    axS.set_ylabel("stage  (m)", color="#2c7fb8")
    axS.yaxis.set_label_position("right")
    axS.tick_params(axis="y", labelcolor="#2c7fb8", labelright=True)

    # the single clearly-shaded LEAD WINDOW (onset -> peak)
    ax.axvspan(onset_first, q_peak_t, color="#fdae61", alpha=0.32, zorder=1)
    ax.axvline(q_peak_t, color="#D55E00", lw=1.5, zorder=4)
    ax.text(q_peak_t, qe.max() * 0.99, " peak Q & stage", color="#D55E00",
            fontsize=9, va="top", fontweight="medium")

    # bed-transport onset stars
    for s, t in onsets.items():
        yv = np.interp(mdates.date2num(t), mdates.date2num(qe.index), qe.values)
        ax.plot(t, yv, "*", ms=15, mfc=ST_COLORS[s], mec="k", mew=0.6, zorder=6,
                label=f"{s} onset")
    lead_lo, lead_hi = min(leads.values()), max(leads.values())
    ax.annotate("5–15 Hz bed-transport onset leads peak Q by\n"
                f"~{lead_lo:.0f}–{lead_hi:.0f} h — the bed mobilizes on the rising limb",
                xy=(onset_first, qe.loc[:q_peak_t].max() * 0.35),
                xytext=(pd.Timestamp("2025-12-08T01:30:00+00:00"), qe.max() * 0.70),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#b35900"))
    ax.legend(loc="upper right", fontsize=8.5, ncol=2, framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("December 2025 (UTC)")
    ax.set_title("(b) Bed-transport onset leads peak Q (diagnostic)",
                 fontsize=12, loc="left")
    return dict(leads=leads, q_peak_utc=str(q_peak_t))


def _load_gage_cms_stage(path: Path) -> pd.DataFrame:
    g = pd.read_csv(path, parse_dates=["time_utc"]).set_index("time_utc")
    g.index = pd.to_datetime(g.index, utc=True, errors="coerce")
    q = pd.to_numeric(g["discharge_cfs"], errors="coerce") * CFS_TO_CMS
    h = pd.to_numeric(g["gage_height_ft"], errors="coerce") * FT_TO_M
    return pd.DataFrame({"q": q, "h": h}).sort_index()


def panel_c(ax) -> dict:
    """Deterministic routing concept: upstream Q -> downstream stage."""
    up = _load_gage_cms_stage(UP_GAGE)
    dn = _load_gage_cms_stage(DN_GAGE)

    # common 15-min grid over the flood window
    w0, w1 = pd.Timestamp(ROUTE_WINDOW[0]), pd.Timestamp(ROUTE_WINDOW[1])
    idx = pd.date_range(w0, w1, freq="15min", tz="UTC")
    qu = up["q"].reindex(up.index.union(idx)).interpolate("time").reindex(idx)
    qd = dn["q"].reindex(dn.index.union(idx)).interpolate("time").reindex(idx)
    hd = dn["h"].reindex(dn.index.union(idx)).interpolate("time").reindex(idx)

    # ---- flood-wave lag tau via discharge cross-correlation (peak-aligned) ----
    a = (qu - qu.mean()).to_numpy()
    b = (qd - qd.mean()).to_numpy()
    step_h = 0.25
    max_lag = int(round(12.0 / step_h))  # search 0..12 h
    lags = np.arange(0, max_lag + 1)
    cc = []
    for L in lags:
        if L == 0:
            cc.append(np.corrcoef(a, b)[0, 1])
        else:
            cc.append(np.corrcoef(a[:-L], b[L:])[0, 1])
    cc = np.asarray(cc)
    best = int(lags[np.nanargmax(cc)])
    tau_h = best * step_h
    celerity_ms = (CHANNEL_KM * 1000.0) / (tau_h * 3600.0) if tau_h > 0 else float("nan")

    # ---- gain g (downstream/upstream) by least squares on lag-aligned Q ----
    if best > 0:
        x = qu.to_numpy()[:-best]
        y = qd.to_numpy()[best:]
    else:
        x, y = qu.to_numpy(), qd.to_numpy()
    g = float(np.dot(x, y) / np.dot(x, x))

    # ---- predicted downstream discharge: Q_pred(t) = g * Q_up(t - tau) ----
    q_pred = g * qu.shift(best)

    # ---- downstream rating curve h = h0 + (Q/C)^(1/beta) ----
    rfits = {r["gage"]: r for r in json.loads((CONFIG / "rating_fits.json").read_text())}
    rf = rfits["12101500"]
    C, beta = rf["C"], rf["beta"]
    geom = json.loads((CONFIG / "rating_geometry.json").read_text())
    h0 = geom["12101500"]["h0_m"]  # 3.68 m
    h_pred = h0 + (q_pred / C) ** (1.0 / beta)

    # ---- agreement on stage (where both defined) ----
    m = h_pred.notna() & hd.notna()
    hp, ho = h_pred[m].to_numpy(), hd[m].to_numpy()
    r_stage = float(np.corrcoef(hp, ho)[0, 1])
    nse_stage = float(1.0 - np.sum((ho - hp) ** 2) / np.sum((ho - ho.mean()) ** 2))

    # ---- plot observed vs predicted downstream stage ----
    ax.plot(hd.index, hd.values, color="#222", lw=1.8,
            label="observed downstream stage", zorder=3)
    ax.plot(h_pred.index, h_pred.values, color="#D55E00", lw=1.7, ls="--",
            label="predicted (routed from upstream Q)", zorder=4)
    ax.set_ylabel("downstream stage  (m)")
    ax.set_xlim(w0, w1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("December 2025 (UTC)")

    # forecast horizon = tau, annotated
    ax.text(0.025, 0.96,
            f"forecast horizon = {tau_h:.1f} h  (celerity ≈ {celerity_ms:.1f} m/s)\n"
            f"gain g = {g:.2f}   stage r = {r_stage:.2f}   NSE = {nse_stage:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.9))
    ax.annotate("deterministic routing (constant lag + gain + rating); single event —\n"
                "a concept, NOT validated forecast skill (out-of-sample test = March-2026)",
                xy=(0.025, 0.025), xycoords="axes fraction", va="bottom", ha="left",
                fontsize=8, color="#555")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.set_title("(c) Routing concept: upstream Q → downstream stage",
                 fontsize=12, loc="left")
    return dict(tau_h=tau_h, celerity_ms=celerity_ms, gain=g,
                r_stage=r_stage, nse_stage=nse_stage, cc_max=float(np.nanmax(cc)))


# ---------------------------------------------------------------------------
def main() -> int:
    paper_style()
    virt = json.loads((CONFIG / "virtual_q.json").read_text())
    fit = json.loads((CONFIG / "virtual_q_fit.json").read_text())

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.34, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    skill = panel_a(ax_a, virt, fit)
    haz = panel_b(ax_b)
    cor = panel_c(ax_c)

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
    print("(c) routing concept:")
    print(f"    tau = {cor['tau_h']:.2f} h   celerity = {cor['celerity_ms']:.2f} m/s   "
          f"(xcorr peak cc = {cor['cc_max']:.3f})")
    print(f"    gain g = {cor['gain']:.3f}")
    print(f"    predicted-vs-observed downstream stage:  r = {cor['r_stage']:.3f}  "
          f"NSE = {cor['nse_stage']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
