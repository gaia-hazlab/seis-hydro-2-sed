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
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

FNAME_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)(?:_(?P<f1>[\d.]+)-(?P<f2>[\d.]+)Hz)?_timeseries\.csv$")


def discover() -> list[dict]:
    out = []
    for f in sorted(RESULTS.glob("*_timeseries.csv")):
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        d = m.groupdict()
        band = (float(d["f1"]), float(d["f2"])) if d["f1"] else None
        out.append(dict(path=f, station=f'{d["net"]}.{d["sta"]}',
                        net=d["net"], sta=d["sta"], band=band))
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

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhspan(*WATER_BASELINE, color="tab:blue", alpha=0.12,
               label=f"turbulence baseline\nb≈{WATER_BASELINE[0]}–{WATER_BASELINE[1]} (Gimbert 2014)")
    ax.axhline(1.0, color="tab:blue", ls=":", lw=1)
    # Exclude the 0.5-2 Hz oceanic-microseism band: it is not river turbulence
    # (it can be anti-correlated with discharge) and distorts the b(f) trend.
    plot_df = df[df["fc"] >= 1.5]
    for sta, g in plot_df.groupby("station"):
        g = g.sort_values("fc")
        yerr = np.vstack([g["b"] - g["b_lo"], g["b_hi"] - g["b"]])
        ax.errorbar(g["fc"], g["b"], yerr=yerr, marker="o", capsize=3, lw=1.5, label=sta)
    ax.set_xscale("log")
    ax.set_xlabel("band center frequency (Hz)")
    ax.set_ylabel(r"scaling exponent  $b$  in  $P \propto Q^{\,b}$")
    ax.set_title("Seismic–discharge scaling steepens with frequency → bedload signature")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.savefig(FIGDIR / "fig2_scaling_exponent.png")
    plt.close(fig)
    return df


def fig_pq_scatter(items: list[dict]) -> None:
    banded = [it for it in items if it["band"] is not None]
    if not banded:
        return
    n = len(banded)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax, it in zip(axes.ravel(), banded):
        j = load_timeseries(it["path"])
        fit = fit_scaling(j, it["station"], it["band"])
        lq = np.log10(j["Q"].clip(lower=1e-6))
        lp = np.log10(j["P"].clip(lower=1e-30))
        ax.scatter(lq, lp, s=4, alpha=0.3, color="0.4")
        xs = np.linspace(lq.quantile(0.02), lq.quantile(0.98), 50)
        ax.plot(xs, fit.intercept + fit.b_ols * xs, "r-", lw=2,
                label=f"b={fit.b_ols:.2f}\n[{fit.b_lo:.2f},{fit.b_hi:.2f}]\nr={fit.r:.2f}")
        ax.set_title(f'{it["station"]}  {it["band"][0]:g}–{it["band"][1]:g} Hz', fontsize=8)
        ax.set_xlabel(r"$\log_{10} Q$ (m³/s)")
        ax.set_ylabel(r"$\log_{10} P$")
        ax.legend(fontsize=6.5, loc="lower right")
    for ax in axes.ravel()[len(banded):]:
        ax.axis("off")
    fig.suptitle("Band power vs discharge (robust log–log fits, bootstrap 95% CI)", y=1.0)
    fig.savefig(FIGDIR / "fig3_pq_scatter.png")
    plt.close(fig)


def fig_hysteresis(items: list[dict]) -> None:
    banded = [it for it in items if it["band"] is not None]
    if not banded:
        return
    n = len(banded); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax, it in zip(axes.ravel(), banded):
        j = load_timeseries(it["path"])
        ev = event_window(j)
        t = (ev.index - ev.index[0]).total_seconds() / 3600.0
        sc = ax.scatter(ev["Q"], np.log10(ev["P"].clip(lower=1e-30)), c=t, s=8, cmap="viridis")
        hi = lawler_hysteresis_index(ev["Q"].values, np.log10(ev["P"].clip(lower=1e-30)).values)
        sense = "CW" if hi > 0.02 else ("CCW" if hi < -0.02 else "~0")
        ax.set_title(f'{it["station"]} {it["band"][0]:g}–{it["band"][1]:g}Hz\nHI={hi:+.2f} ({sense})', fontsize=8)
        ax.set_xlabel("Q (m³/s)"); ax.set_ylabel(r"$\log_{10} P$")
        fig.colorbar(sc, ax=ax, label="hours into event", fraction=0.046)
    for ax in axes.ravel()[len(banded):]:
        ax.axis("off")
    fig.suptitle("Event hysteresis: clockwise=supply/exhaustion, CCW=delayed/distal delivery", y=1.0)
    fig.savefig(FIGDIR / "fig4_hysteresis.png")
    plt.close(fig)


def fig_event_timeseries(items: list[dict]) -> None:
    banded = [it for it in items if it["band"] is not None]
    if not banded:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()
    j0 = load_timeseries(banded[0]["path"])
    ax2.plot(j0.index, j0["Q"], color="k", lw=1.2, ls="--", label="discharge", zorder=5)
    ax2.set_ylabel("discharge (m³/s)")
    for it in banded:
        j = load_timeseries(it["path"])
        pn = j["P"] / j["P"].median()
        ax1.semilogy(j.index, pn, lw=0.9, alpha=0.8,
                     label=f'{it["station"]} {it["band"][0]:g}–{it["band"][1]:g}Hz')
    ax1.set_ylabel("seismic power / median")
    ax1.set_xlabel("UTC")
    ax1.set_title("December 2025 atmospheric-river flood — seismic power vs discharge")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left", ncol=2)
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
