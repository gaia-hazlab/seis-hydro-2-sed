#!/usr/bin/env python3
"""Generate publication-grade figures from the aligned proxy/discharge products.

Reads every ``notebooks/data/results/*_timeseries.csv`` (proxy + gauge), parses
station/band from the filename, and produces figures into ``paper/figures/``:

  fig1_transect_map.png        station + gage map along the Puyallup corridor
  fig2_scaling_exponent.png    b vs frequency band per station (the core result)
  fig3_pq_scatter.png          log-log P vs Q with robust fits (multi-panel)
  fig4_hysteresis.png          event hysteresis loops colored by time
  fig5_event_timeseries.png    discharge + band power during the Dec-2025 event
  scaling_table.csv            machine-readable fit table (b, CI, r, HI)

Robust to however many stations have been computed; panels fill in as the
batch runner produces more series.

Usage: pixi run python workflows/02_make_figures.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.analysis import (  # noqa: E402
    WATER_BASELINE, event_window, fit_scaling, lawler_hysteresis_index, load_timeseries,
)

RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.22, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,   # auto spacing — no overlapping labels
})

# Stable color per station so a station keeps its color across all figures.
_STA_COLORS: dict[str, str] = {}
def _color(station: str) -> str:
    if station not in _STA_COLORS:
        pal = plt.cm.tab10.colors
        _STA_COLORS[station] = pal[len(_STA_COLORS) % len(pal)]
    return _STA_COLORS[station]

FNAME_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)(?:_(?P<f1>[\d.]+)-(?P<f2>[\d.]+)Hz)?_timeseries\.csv$")

# Stations excluded from the analysis. UW.BHW (lowland Snohomish) is too far from
# any bedload source to be informative; dropped per the 2026 review.
EXCLUDE_STATIONS = {"UW.BHW"}

# Fixed log10(power) y-range for the per-station panels (fig 3, fig 4) so they
# are not vertically squeezed.
P_YLIM = (-15, -11)


def discover() -> list[dict]:
    out = []
    for f in sorted(RESULTS.glob("*_timeseries.csv")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        d = m.groupdict()
        station = f'{d["net"]}.{d["sta"]}'
        if station in EXCLUDE_STATIONS:
            continue
        band = (float(d["f1"]), float(d["f2"])) if d["f1"] else None
        out.append(dict(path=f, station=station, net=d["net"], sta=d["sta"], band=band))
    return out


def fig_scaling_exponent(items: list[dict]) -> pd.DataFrame:
    """b vs frequency band, per station, with the turbulence baseline shaded."""
    rows = []
    for it in items:
        if it["band"] is None:
            continue
        j = load_timeseries(it["path"])
        fit = fit_scaling(j, it["station"], it["band"])
        ev = event_window(j)
        x, y, _ = (np.log10(ev["Q"].clip(lower=1e-6)).values,
                   np.log10(ev["P"].clip(lower=1e-30)).values, None)
        hi = lawler_hysteresis_index(ev["Q"].values, np.log10(ev["P"].clip(lower=1e-30)).values)
        rows.append(dict(station=fit.station, fc=np.sqrt(it["band"][0] * it["band"][1]),
                         band=f'{it["band"][0]:g}-{it["band"][1]:g}', b=fit.b_ols,
                         b_lo=fit.b_lo, b_hi=fit.b_hi, r=fit.r, n=fit.n, HI=hi))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.axhspan(*WATER_BASELINE, color="0.6", alpha=0.18,
               label=f"turbulence baseline (b≈{WATER_BASELINE[0]}–{WATER_BASELINE[1]})")
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    # Exclude the 0.5-2 Hz oceanic-microseism band: it is not river turbulence
    # (it can be anti-correlated with discharge) and distorts the b(f) trend.
    plot_df = df[df["fc"] >= 1.5]
    for sta, g in plot_df.groupby("station"):
        g = g.sort_values("fc")
        yerr = np.vstack([g["b"] - g["b_lo"], g["b_hi"] - g["b"]])
        bvals = g["b"].tolist()
        fit_lbl = (f"b={bvals[0]:.2f}" if len(bvals) == 1
                   else "b=" + "→".join(f"{v:.2f}" for v in bvals))
        ax.errorbar(g["fc"], g["b"], yerr=yerr, marker="o", ms=7, capsize=3, lw=1.8,
                    color=_color(sta), label=f"{sta}   {fit_lbl}")
    ax.set_xscale("log")
    ax.tick_params(labelsize=11)
    ax.set_xlabel("band center frequency (Hz)", fontsize=13)
    ax.set_ylabel(r"scaling exponent $b$   ($P \propto Q^{\,b}$)", fontsize=13)
    ax.set_title("Seismic–discharge scaling vs frequency", loc="left", fontsize=14)
    # legend outside, to the right — shows each station's fitted exponent
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              title="station (fitted exponent)", title_fontsize=12, fontsize=11)
    fig.savefig(FIGDIR / "fig2_scaling_exponent.png")
    plt.close(fig)
    return df


def _grid(n: int) -> tuple[int, int]:
    ncol = min(3, n)
    return int(np.ceil(n / ncol)), ncol


def fig_pq_scatter(items: list[dict]) -> None:
    banded = [it for it in items if it["band"] is not None]
    if not banded:
        return
    nrow, ncol = _grid(len(banded))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.7 * nrow),
                             squeeze=False, sharex=True, sharey=True)
    flat = axes.ravel()
    for ax, it in zip(flat, banded):
        j = load_timeseries(it["path"])
        fit = fit_scaling(j, it["station"], it["band"])
        lq = np.log10(j["Q"].clip(lower=1e-6))
        lp = np.log10(j["P"].clip(lower=1e-30))
        ax.scatter(lq, lp, s=3, alpha=0.25, color="0.55", rasterized=True)
        xs = np.linspace(lq.quantile(0.02), lq.quantile(0.98), 50)
        ax.plot(xs, fit.intercept + fit.b_ols * xs, color=_color(it["station"]), lw=2)
        ax.set_title(f'{it["station"]}  {it["band"][0]:g}–{it["band"][1]:g} Hz')
        # fit stats as an unobtrusive text box (no legend handle to overlap data)
        ax.text(0.04, 0.96, f"b={fit.b_ols:.2f} [{fit.b_lo:.2f}, {fit.b_hi:.2f}]\nr={fit.r:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
        ax.set_ylim(*P_YLIM)   # fixed range so panels are not vertically squeezed
    for ax in flat[len(banded):]:
        ax.set_visible(False)
    fig.supxlabel(r"$\log_{10}\,Q$  (m³ s$^{-1}$)")
    fig.supylabel(r"$\log_{10}\,P$  (seismic band power)")
    fig.suptitle("Band power vs discharge — robust log–log fits (95% CI)")
    fig.savefig(FIGDIR / "fig3_pq_scatter.png")
    plt.close(fig)


def fig_hysteresis(items: list[dict]) -> None:
    banded = [it for it in items if it["band"] is not None]
    if not banded:
        return
    nrow, ncol = _grid(len(banded))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.7 * nrow),
                             squeeze=False, sharey=True)
    flat = axes.ravel()
    sc = None
    for ax, it in zip(flat, banded):
        j = load_timeseries(it["path"])
        ev = event_window(j)
        t = (ev.index - ev.index[0]).total_seconds() / 3600.0
        sc = ax.scatter(ev["Q"], np.log10(ev["P"].clip(lower=1e-30)), c=t, s=9,
                        cmap="viridis", edgecolor="none")
        hi = lawler_hysteresis_index(ev["Q"].values, np.log10(ev["P"].clip(lower=1e-30)).values)
        sense = "clockwise" if hi > 0.02 else ("counter-clockwise" if hi < -0.02 else "≈single-valued")
        ax.set_title(f'{it["station"]} {it["band"][0]:g}–{it["band"][1]:g} Hz\nHI={hi:+.2f} ({sense})')
        ax.set_ylim(*P_YLIM)   # fixed range so panels are not vertically squeezed
    for ax in flat[len(banded):]:
        ax.set_visible(False)
    fig.supxlabel(r"discharge $Q$  (m³ s$^{-1}$)")
    fig.supylabel(r"$\log_{10}\,P$")
    fig.suptitle("Event hysteresis (clockwise = supply/exhaustion; CCW = delayed/distal delivery)")
    if sc is not None:  # single shared colorbar
        fig.colorbar(sc, ax=axes, label="hours into event", shrink=0.85, aspect=30, pad=0.02)
    fig.savefig(FIGDIR / "fig4_hysteresis.png")
    plt.close(fig)


def fig_event_timeseries(items: list[dict]) -> None:
    """Offset (ridgeline) plot of seismic power vs discharge through the event.

    Each station occupies its own vertical lane (offset, so traces never
    overlap), ordered source→downstream by river-km. One color per station;
    within a lane the low-frequency (flow) band is semi-transparent (alpha 0.5)
    and the high-frequency (bedload) band opaque. Each lane shows
    log10(power / its median), so the gridline is the station's median level and
    one lane-unit = one decade. Discharge (dashed) is drawn behind for context.
    """
    from collections import defaultdict
    from matplotlib.lines import Line2D

    banded = [it for it in items if it["band"] is not None]
    if not banded:
        return

    # station -> river-km, for source→sea ordering
    rkm: dict[str, float] = {}
    disc = ROOT / "config" / "_transect_discovery.json"
    if disc.exists():
        for v in json.loads(disc.read_text()).get("stations", []):
            rkm[f'{v["net"]}.{v["sta"]}'] = v.get("river_km", 999)

    by_sta: dict[str, list] = defaultdict(list)
    for it in banded:
        fc = (it["band"][0] * it["band"][1]) ** 0.5
        by_sta[it["station"]].append((fc, it))
    stations = sorted(by_sta, key=lambda s: (rkm.get(s, 999), s))

    STEP = 4.0          # vertical lane spacing, in decades
    FLOOR = 1e-3        # clip for log10
    fig, ax1 = plt.subplots(figsize=(8.8, 1.15 * len(stations) + 1.8))
    ax2 = ax1.twinx()
    ax2.grid(False)
    j0 = load_timeseries(banded[0]["path"])
    hd, = ax2.plot(j0.index, j0["Q"], color="0.2", lw=1.5, ls="--", label="discharge", zorder=1)
    ax2.set_ylabel(r"discharge (m³ s$^{-1}$)")

    yt, yl, base = [], [], 0.0
    for idx, sta in enumerate(stations):
        base = idx * STEP
        c = _color(sta)
        lst = sorted(by_sta[sta])
        ax1.axhline(base, color="0.85", lw=0.6, zorder=0)            # median level
        if len(lst) >= 2:                                            # LF (flow), faint
            jl = load_timeseries(lst[0][1]["path"])
            yv = np.log10((jl["P"] / jl["P"].median()).clip(lower=FLOOR)) + base
            ax1.plot(jl.index, yv, color=c, lw=0.8, alpha=0.5, zorder=3)
        jh = load_timeseries(lst[-1][1]["path"])                     # HF (bedload), opaque
        yv = np.log10((jh["P"] / jh["P"].median()).clip(lower=FLOOR)) + base
        ax1.plot(jh.index, yv, color=c, lw=0.9, alpha=1.0, zorder=4)
        yt.append(base); yl.append(sta)

    ax1.set_yticks(yt)
    ax1.set_yticklabels(yl)
    for t, sta in zip(ax1.get_yticklabels(), yl):
        t.set_color(_color(sta)); t.set_fontweight("bold")
    ax1.set_ylim(-STEP * 0.7, base + STEP * 0.8)
    ax1.set_ylabel("station lane  (offset; gridline = median, 1 lane-unit = ×10)")
    ax1.set_xlabel("December 2025 (UTC)")
    ax1.set_title("Seismic band power vs discharge — offset by station (source → downstream)", loc="left")

    # vertical 1-decade scale bar (top-left)
    x0 = j0.index[int(0.02 * len(j0))]
    ax1.plot([x0, x0], [base + STEP * 0.2, base + STEP * 0.2 + 1.0], color="k", lw=2)
    ax1.text(x0, base + STEP * 0.2 + 0.5, " 1 decade", va="center", fontsize=7)

    style = [
        Line2D([], [], color="0.35", lw=2, alpha=0.5, label="low-freq band (flow)"),
        Line2D([], [], color="0.35", lw=2, alpha=1.0, label="high-freq band (bedload)"),
        hd,
    ]
    fig.legend(handles=style, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, frameon=False)
    fig.autofmt_xdate()
    fig.savefig(FIGDIR / "fig5_event_timeseries.png")
    plt.close(fig)


def fig_transect_map() -> None:
    disc = ROOT / "config" / "_transect_discovery.json"
    if not disc.exists():
        return
    stations = json.loads(disc.read_text())["stations"]
    river = [(46.904, -122.035), (47.039, -122.208), (47.10, -122.22),
             (47.185, -122.230), (47.208, -122.327), (47.270, -122.420)]
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    rl = np.array(river)
    ax.plot(rl[:, 1], rl[:, 0], "-", color="tab:blue", lw=2, alpha=0.6, label="Puyallup mainstem")
    for v in stations:
        bb = v["broadband"]
        ax.scatter(v["lon"], v["lat"], marker="^" if bb else "s",
                   s=55 if bb else 35,
                   color="firebrick" if bb else "0.5",
                   edgecolor="k", lw=0.4, zorder=4)
        if bb or v["dist_river_km"] < 5:
            ax.annotate(f'{v["net"]}.{v["sta"]}', (v["lon"], v["lat"]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.scatter([], [], marker="^", color="firebrick", edgecolor="k", label="broadband (BH/EH) — bedload-grade")
    ax.scatter([], [], marker="s", color="0.5", edgecolor="k", label="urban strong-motion (HN/EN)")
    ax.annotate("Mt. Rainier\n(glacial source)", (-122.0, 46.88), fontsize=8, ha="center", color="darkgreen")
    ax.annotate("Puget Sound\n(Commencement Bay)", (-122.43, 47.29), fontsize=8, ha="center", color="navy")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Mountain-to-sea seismic transect, Puyallup River, WA")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_aspect(1.35)
    fig.savefig(FIGDIR / "fig1_transect_map.png")
    plt.close(fig)


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """Hand-rolled GitHub-flavored markdown table (no tabulate dependency).

    Lets the Quarto book render the scaling table as static markdown, so the
    book builds with Quarto alone — no Python kernel needed on CI.
    """
    cols = ["station", "band", "b", "b_lo", "b_hi", "r", "n", "HI"]
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    for c in d.columns:
        if c not in ("station", "band", "n"):
            d[c] = d[c].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    items = discover()
    print(f"Discovered {len(items)} timeseries products: "
          + ", ".join(sorted({it['station'] for it in items})))
    fig_transect_map()
    table = fig_scaling_exponent(items)
    fig_pq_scatter(items)
    fig_hysteresis(items)
    fig_event_timeseries(items)
    if not table.empty:
        table.to_csv(FIGDIR / "scaling_table.csv", index=False)
        _write_markdown_table(table, FIGDIR / "_scaling_table.md")
        print("\nScaling table:")
        print(table.to_string(index=False))
    print(f"\nFigures written to {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
