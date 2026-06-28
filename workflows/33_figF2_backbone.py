#!/usr/bin/env python3
"""Composite Figure F2 — "Seismic power tracks discharge (the turbulence backbone)."

Merges three legacy figures into one publication panel, rebuilt OFFLINE from the
cached ``notebooks/data/results/*_timeseries.csv`` products:

  (a) event timeseries  : discharge + 5–15 Hz seismic power over the Dec-2025 event
                          for a few representative source stations (was fig5)
  (b) P–Q log–log scatter: a 3–4 station SUBSET (source → downstream) with robust
                          fit lines and the fitted exponent b in the legend (was fig3)
  (c) scaling exponent b : b vs band-center frequency, per station, with the
                          turbulent-flow baseline shaded (was fig2)

The full P–Q grid and the full per-station/per-band table go to the supplement;
this figure carries only the representative subset that makes the backbone case.

Reuses the proven fit helpers in ``riverseis.analysis`` (``load_timeseries``,
``clean_loglog``, ``fit_scaling``) rather than re-deriving any fit.

Usage: pixi run python workflows/33_figF2_backbone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.analysis import (  # noqa: E402
    WATER_BASELINE, clean_loglog, clip_event, fit_scaling, load_timeseries,
)
from riverseis.figstyle import paper_style  # noqa: E402

RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

paper_style()  # group standard: legible fonts + tight bbox
plt.rcParams.update({
    "axes.grid": True, "grid.alpha": 0.22, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Okabe–Ito colorblind-safe palette; one stable color per station across panels.
_OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
              "#56B4E9", "#F0E442", "#000000"]
_STA_COLORS: dict[str, str] = {}


def color(station: str) -> str:
    if station not in _STA_COLORS:
        _STA_COLORS[station] = _OKABE_ITO[len(_STA_COLORS) % len(_OKABE_ITO)]
    return _STA_COLORS[station]


def ts_path(station: str, band: str) -> Path:
    return RESULTS / f"{station}_{band}_timeseries.csv"


# ---------------------------------------------------------------------------
# Representative subset (source → mid → downstream → 2nd basin). All have a
# 5–15 Hz (bedload-grade) product covering the full Dec-2025 event window.
SUBSET = [
    ("CC.PR03", "source"),       # near the Carbon/Puyallup sediment source
    ("CC.SIFT", "mid"),          # mid-corridor
    ("CC.TRON", "downstream"),   # downstream
    ("UW.LON",  "Nisqually"),    # 2nd basin (geometric/braided generalization)
]
HF_BAND = "5.0-15.0Hz"          # high-frequency (bedload) band for panels (a)/(b)

# For panel (a) keep it readable: 2 representative source stations.
TS_STATIONS = ["CC.PR03", "CC.SIFT"]

# For panel (c): every station whose products we cache, fit b across each band.
# Discover all (station, band) pairs from the cached CSVs.
EXCLUDE = {"UW.BHW", "UW.TEHA"}  # too-far / traffic-polluted (see 02_make_figures.py)


def discover_band_pairs() -> dict[str, list[tuple[float, float, str]]]:
    out: dict[str, list[tuple[float, float, str]]] = {}
    for f in sorted(RESULTS.glob("*Hz_timeseries.csv")):
        name = f.name
        sta, rest = name.split("_", 1)
        if sta in EXCLUDE:
            continue
        band_str = rest.replace("_timeseries.csv", "")  # e.g. "5.0-15.0Hz"
        f1f2 = band_str.replace("Hz", "")
        f1, f2 = (float(x) for x in f1f2.split("-"))
        out.setdefault(sta, []).append((f1, f2, band_str))
    return out


# ---------------------------------------------------------------------------
def panel_a(ax):
    """Discharge (one axis) + 5–15 Hz seismic power (twin axis) over the event."""
    axp = ax.twinx()
    axp.grid(False)
    # draw discharge (on ax) ON TOP of the power traces (on the twin axp)
    ax.set_zorder(axp.get_zorder() + 1)
    ax.patch.set_visible(False)
    q_done = False
    handles, labels = [], []
    for sta in TS_STATIONS:
        j = load_timeseries(ts_path(sta, HF_BAND))
        if j.empty:
            continue
        if not q_done:
            (hq,) = ax.plot(j.index, j["Q"], color="black", lw=1.6, ls="--",
                            zorder=5, label="discharge (gage)")
            handles.append(hq)
            labels.append("discharge (gage)")
            q_done = True
        # normalize each power trace to its own median so multiple stations
        # share the (log) twin axis legibly.
        pn = j["P"] / j["P"].median()
        (hp,) = axp.plot(j.index, pn, color=color(sta), lw=1.0, alpha=0.85)
        handles.append(hp)
        labels.append(f"{sta}  (5–15 Hz)")
    ax.set_ylabel(r"discharge $Q$  (m$^3$ s$^{-1}$)", fontsize=14)
    axp.set_yscale("log")
    # keep the twin-axis title off the panel title: label sits on the outward
    # (right) side and the title is left-aligned, so they never collide.
    axp.set_ylabel("seismic power / median  (5–15 Hz)", fontsize=14, labelpad=8)
    axp.yaxis.set_label_position("right")
    ax.set_xlabel("December 2025 (UTC)", fontsize=14)
    ax.set_title("(a) Power vs discharge", loc="left", fontsize=14)
    ax.tick_params(labelsize=12)
    axp.tick_params(labelsize=12)
    for lab in ax.get_xticklabels():
        lab.set_rotation(25)
        lab.set_ha("right")
    # late-Dec: the discharge tail (black) sits low and the normalized power
    # traces sit high, leaving a clear mid-height band on the right. Anchor the
    # legend there (just right of centre, vertically mid-panel).
    ax.legend(handles, labels, loc="center", bbox_to_anchor=(0.80, 0.52),
              frameon=True, fontsize=12, framealpha=0.92,
              handletextpad=0.4, handlelength=1.8, labelspacing=0.35)


def panel_b(ax):
    """P–Q log–log scatter for the 3–4 station SUBSET with robust fit lines."""
    for sta, role in SUBSET:
        p = ts_path(sta, HF_BAND)
        if not p.exists():
            continue
        j = clip_event(load_timeseries(p))   # fit over the analysis window
        fit = fit_scaling(j, sta, (5.0, 15.0))
        lq, lp, _ = clean_loglog(j)
        c = color(sta)
        ax.scatter(lq, lp, s=4, alpha=0.18, color=c, rasterized=True)
        if np.isfinite(fit.b_ols):
            xs = np.linspace(np.quantile(lq, 0.02), np.quantile(lq, 0.98), 50)
            ax.plot(xs, fit.intercept + fit.b_ols * xs, color=c, lw=2.2, zorder=5,
                    label=f"{sta} ({role})  b={fit.b_ols:.2f} "
                          f"[{fit.b_lo:.2f}, {fit.b_hi:.2f}]")
    ax.set_xlabel(r"$\log_{10}\,Q$  (m$^3$ s$^{-1}$)", fontsize=14)
    ax.set_ylabel(r"$\log_{10}\,P$  (seismic band power)", fontsize=14)
    ax.set_title("(b) P $\\propto$ Q$^{\\,b}$", loc="left",
                 fontsize=14)
    ax.tick_params(labelsize=12)
    # add headroom above the data (data unchanged) so the upper-left legend
    # sits clear of the high CC.PR03 point cloud rather than on top of it.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.32 * (y1 - y0))
    ax.legend(loc="upper left", frameon=True, fontsize=12, framealpha=0.92,
              borderaxespad=0.6, handlelength=1.6, labelspacing=0.4)


def panel_c(ax):
    """Scaling exponent b vs band-center frequency, per station, baseline shaded."""
    ax.axhspan(*WATER_BASELINE, color="0.6", alpha=0.18, zorder=0)
    ax.axhline(1.0, color="0.5", ls=":", lw=1, zorder=0)

    markers = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "*", "p"]
    pairs = discover_band_pairs()
    # highlight the subset (source-ward) stations; others drawn lighter context.
    subset_stas = {s for s, _ in SUBSET}
    reps: dict[str, dict[str, float]] = {}
    for mi, sta in enumerate(sorted(pairs)):
        rows = []
        for f1, f2, band_str in pairs[sta]:
            fc = float(np.sqrt(f1 * f2))
            # skip the oceanic-microseism 0.5–2 Hz band: not river turbulence
            # (can anti-correlate with Q) and distorts the b(f) trend.
            if fc < 1.5:
                continue
            j = clip_event(load_timeseries(ts_path(sta, band_str)))
            fit = fit_scaling(j, sta, (f1, f2))
            if not np.isfinite(fit.b_ols):
                continue
            rows.append((fc, fit.b_ols, fit.b_lo, fit.b_hi, band_str))
            reps.setdefault(sta, {})[band_str] = fit.b_ols
        if not rows:
            continue
        rows.sort()
        fc = np.array([r[0] for r in rows])
        b = np.array([r[1] for r in rows])
        lo = np.array([r[2] for r in rows])
        hi = np.array([r[3] for r in rows])
        is_sub = sta in subset_stas
        ax.errorbar(fc, b, yerr=np.vstack([b - lo, hi - b]),
                    marker=markers[mi % len(markers)],
                    ms=9 if is_sub else 6, capsize=3,
                    lw=2.0 if is_sub else 1.0, alpha=1.0 if is_sub else 0.55,
                    color=color(sta) if is_sub else "0.55",
                    markerfacecolor=color(sta) if is_sub else "0.7",
                    markeredgecolor=color(sta) if is_sub else "0.55",
                    label=sta if is_sub else None, zorder=4 if is_sub else 2)
    ax.set_xscale("log")
    ax.set_xlabel("band center frequency (Hz)", fontsize=14)
    ax.set_ylabel(r"scaling exponent $b$   ($P \propto Q^{\,b}$)", fontsize=14)
    ax.set_title("(c) Exponent vs frequency", loc="left",
                 fontsize=14)
    ax.tick_params(labelsize=12)

    # label the shaded turbulence baseline in-place (no legend entry, so the
    # station legend stays uncluttered) at the left edge inside the band.
    xlo = ax.get_xlim()[0]
    ax.text(xlo * 1.04, np.mean(WATER_BASELINE),
            f"turbulence baseline  b≈{WATER_BASELINE[0]}–{WATER_BASELINE[1]}\n"
            f"(Gimbert et al. 2014)",
            fontsize=11.5, va="center", ha="left", color="0.30",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                      alpha=0.85))
    # near-source stations sit HIGH (b>1.4); context stations sit LOW. The
    # lower-right corner is the empty region for the station legend.
    ax.legend(loc="lower right", frameon=True, fontsize=12, framealpha=0.92,
              borderaxespad=0.6, ncol=2, columnspacing=1.0, handletextpad=0.4)
    return reps


def main() -> int:
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          hspace=0.42, wspace=0.26,
                          left=0.07, right=0.965, top=0.95, bottom=0.09)
    ax_a = fig.add_subplot(gs[0, :])   # top row spans both columns
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    reps = panel_c(ax_c)

    out = FIGDIR / "figF2_backbone.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print(f"Wrote {out}")
    print(f"Turbulence baseline (Gimbert 2014): b = {WATER_BASELINE[0]}–{WATER_BASELINE[1]}")
    print("Representative b (OLS, log–log, clip_event window):")
    for sta in sorted(reps):
        for band, b in sorted(reps[sta].items()):
            print(f"  {sta:10s} {band:11s}  b={b:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
