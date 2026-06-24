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
    WATER_BASELINE, clip_event, event_window, fit_scaling, lawler_hysteresis_index,
    load_timeseries,
)

RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

from riverseis.figstyle import paper_style  # noqa: E402
paper_style()                                # legible fonts + tight bbox (group standard)
plt.rcParams.update({
    "savefig.dpi": 300,
    "axes.grid": True, "grid.alpha": 0.22, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.constrained_layout.use": True,   # auto spacing — no overlapping labels
})

# Stable color per station so a station keeps its color across all figures.
# Okabe–Ito colorblind-safe qualitative palette.
_OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
              "#56B4E9", "#F0E442", "#000000"]
_STA_COLORS: dict[str, str] = {}
def _color(station: str) -> str:
    if station not in _STA_COLORS:
        _STA_COLORS[station] = _OKABE_ITO[len(_STA_COLORS) % len(_OKABE_ITO)]
    return _STA_COLORS[station]

FNAME_RE = re.compile(r"^(?P<net>[A-Z0-9]+)\.(?P<sta>[A-Z0-9]+)(?:_(?P<f1>[\d.]+)-(?P<f2>[\d.]+)Hz)?_timeseries\.csv$")

# Stations excluded from the analysis:
#  - UW.BHW: lowland Snohomish, too far from any bedload source.
#  - UW.TEHA: heavily traffic-polluted (traffic_index≈59, weekday/weekend≈21;
#    r≈0 with discharge) — see workflows/04_traffic_noise.py.
EXCLUDE_STATIONS = {"UW.BHW", "UW.TEHA"}

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
        j = clip_event(load_timeseries(it["path"]))   # fit on the flood window
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

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.axhspan(*WATER_BASELINE, color="0.6", alpha=0.18,
               label=f"turbulence baseline (b≈{WATER_BASELINE[0]}–{WATER_BASELINE[1]})")
    ax.axhline(1.0, color="0.5", ls=":", lw=1)
    # Exclude the 0.5-2 Hz oceanic-microseism band: it is not river turbulence
    # (it can be anti-correlated with discharge) and distorts the b(f) trend.
    # none-signal stations drawn hollow/grey (documented, not hidden); a distinct
    # marker SYMBOL per station so they are identifiable without relying on colour.
    sp = ROOT / "config" / "station_status.json"
    status = {s["station"]: s["status"] for s in json.loads(sp.read_text())} if sp.exists() else {}
    # BIC/AIC-significant P–Q breaks from the threshold analysis (fig14): for these
    # stations a two-exponent fit (b below / above Qc) beats a single line, so we
    # report both exponents and star the station here.
    tp = ROOT / "config" / "threshold_qc.json"
    breaks = ({r["station"]: r for r in json.loads(tp.read_text()) if r.get("significant_break")}
              if tp.exists() else {})
    MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "*", "p", "d"]
    plot_df = df[df["fc"] >= 1.5]
    markmap = {s: MARKERS[i % len(MARKERS)]
               for i, s in enumerate(sorted(plot_df["station"].unique()))}
    for sta, g in plot_df.groupby("station"):
        g = g.sort_values("fc")
        yerr = np.vstack([g["b"] - g["b_lo"], g["b_hi"] - g["b"]])
        bvals = g["b"].tolist()
        nosig = status.get(sta) in ("none", "control")
        fit_lbl = (f"b={bvals[0]:.2f}" if len(bvals) == 1
                   else "b=" + "→".join(f"{v:.2f}" for v in bvals))
        brk = breaks.get(sta)
        if brk:
            arr = "↑" if brk["direction"] == "steepening" else "↓"
            fit_lbl += f"  ·  Qc{brk['Qc_cms']:.0f}: {brk['b_below']:.1f}→{brk['b_above']:.1f}{arr}"
        tag = "  (no signal)" if nosig else ""
        ax.errorbar(g["fc"], g["b"], yerr=yerr, marker=markmap[sta], ms=8.5, capsize=3,
                    lw=1.2 if nosig else 1.8, ls=":" if nosig else "-",
                    color="0.6" if nosig else _color(sta),
                    markerfacecolor="none" if nosig else _color(sta),
                    markeredgecolor="k" if brk else ("0.6" if nosig else _color(sta)),
                    markeredgewidth=1.7 if brk else 0.7,
                    label=f"{'★ ' if brk else ''}{sta}   {fit_lbl}{tag}")
    ax.set_xscale("log")
    ax.tick_params(labelsize=11)
    ax.set_xlabel("band center frequency (Hz)", fontsize=13)
    ax.set_ylabel(r"scaling exponent $b$   ($P \propto Q^{\,b}$)", fontsize=13)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
              title=r"station — b vs band;  ★ = BIC-significant Q-break (b$_{<Q_c}\!\to$b$_{>Q_c}$)",
              title_fontsize=10, fontsize=10)
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
        j = clip_event(load_timeseries(it["path"]))   # fit on the flood window
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
    if sc is not None:  # single shared colorbar
        fig.colorbar(sc, ax=axes, label="hours into event", shrink=0.85, aspect=30, pad=0.02)
    fig.savefig(FIGDIR / "fig4_hysteresis.png")
    plt.close(fig)


def fig_event_timeseries(items: list[dict]) -> None:
    """Ridgeline of seismic power through the event, positioned and colored by
    distance downstream from the Mt. Rainier summit (the sediment source).

    The vertical axis is distance-from-summit (km), source at top; each station's
    trace (log10 power / its median, scaled) is centered at its distance and
    colored by that distance. Within a lane the low-frequency (flow) band is
    faint and the high-frequency (bedload) band opaque. Discharge is placed at
    its gage's distance from the summit, scaled the same way. Co-located stations
    are gently de-clustered for legibility (true distance shown by color).
    """
    from collections import defaultdict
    from math import asin, cos, radians, sin, sqrt
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    banded = [it for it in items if it["band"] is not None]
    if not banded:
        return

    SUMMIT = (-121.7603, 46.8523)           # Mt. Rainier (lon, lat)
    GAGE_Q = (-122.0351, 46.9037)           # Puyallup nr Electron 12092000
    AMP = 0.8                               # km per decade of power
    CLIP = (-1.5, 2.0)                      # decades shown

    def hav(lon1, lat1, lon2, lat2):
        p1, p2 = radians(lat1), radians(lat2)
        dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
        return 2 * 6371.0 * asin(sqrt(sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2))

    coords: dict[str, tuple] = {}
    disc = ROOT / "config" / "_transect_discovery.json"
    if disc.exists():
        for v in json.loads(disc.read_text()).get("stations", []):
            coords[f'{v["net"]}.{v["sta"]}'] = (v["lon"], v["lat"])

    by_sta: dict[str, list] = defaultdict(list)
    for it in banded:
        fc = (it["band"][0] * it["band"][1]) ** 0.5
        by_sta[it["station"]].append((fc, it))

    # true distance-from-summit per station; drop very-distant lanes (CC.TRON at
    # 35 km + the urban accelerometers at 44–72 km) that stretch the axis and lie
    # beyond the few-hundred-metre high-frequency seismic reach (§Discussion).
    FAR_KM = 30.0
    dist = {s: hav(*SUMMIT, *coords[s]) for s in by_sta if s in coords}
    dropped = sorted(s for s, d in dist.items() if d > FAR_KM)
    dist = {s: d for s, d in dist.items() if d <= FAR_KM}
    stations = sorted(dist, key=dist.get)
    if not stations:
        return
    if dropped:
        print(f"  fig5: dropped {len(dropped)} station(s) >{FAR_KM:.0f} km from summit: "
              + ", ".join(dropped))
    dgage = hav(*SUMMIT, *GAGE_Q)

    # de-cluster: spread stations that fall within 1.6 km of each other
    lane = {}
    prev = None
    for s in stations:
        y = dist[s]
        if prev is not None and y - prev < 1.6:
            y = prev + 1.6
        lane[s] = y
        prev = y

    dmin, dmax = min(dist.values()), max(dist.values())
    norm = mpl.colors.Normalize(vmin=dmin, vmax=dmax)
    cmap = mpl.cm.viridis

    fig, ax = plt.subplots(figsize=(9.2, 0.95 * len(stations) + 2.2))

    # discharge at the gage's distance from summit (scaled like the power traces)
    j0 = load_timeseries(banded[0]["path"])
    qn = np.log10((j0["Q"] / j0["Q"].median()).clip(lower=10 ** CLIP[0])).clip(*CLIP)
    ax.axhline(dgage, color="0.6", lw=0.6, ls=":", zorder=0)
    hd, = ax.plot(j0.index, dgage - AMP * qn, color="black", lw=1.4, ls="--", zorder=6)

    for s in stations:
        c = cmap(norm(dist[s]))
        y0 = lane[s]
        lst = sorted(by_sta[s])
        ax.axhline(y0, color="0.9", lw=0.5, zorder=0)
        if len(lst) >= 2:                       # LF (flow), faint
            jl = load_timeseries(lst[0][1]["path"])
            v = np.log10((jl["P"] / jl["P"].median()).clip(lower=10 ** CLIP[0])).clip(*CLIP)
            ax.plot(jl.index, y0 - AMP * v, color=c, lw=0.8, alpha=0.45, zorder=3)
        jh = load_timeseries(lst[-1][1]["path"])  # HF (bedload), opaque
        v = np.log10((jh["P"] / jh["P"].median()).clip(lower=10 ** CLIP[0])).clip(*CLIP)
        ax.plot(jh.index, y0 - AMP * v, color=c, lw=1.0, alpha=1.0, zorder=4)
        ax.text(j0.index[0], y0, f" {s}  ({dist[s]:.0f} km)", color=c, fontsize=7.5,
                fontweight="bold", va="center", ha="left", zorder=5)

    ax.set_ylim(dmax + 4, dmin - 4)             # source (small km) at top
    ax.set_ylabel("distance downstream from Mt. Rainier summit (km)")
    ax.set_xlabel("December 2025 (UTC)")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label="distance from summit (km)", shrink=0.8, aspect=30, pad=0.02)

    style = [
        Line2D([], [], color="0.35", lw=2, alpha=0.45, label="low-freq band (flow)"),
        Line2D([], [], color="0.35", lw=2, alpha=1.0, label="high-freq band (bedload)"),
        Line2D([], [], color="black", lw=1.4, ls="--", label="discharge (at gage)"),
    ]
    fig.legend(handles=style, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False)
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
    # fig1 (transect map) is generated separately by workflows/03_make_map.py
    # (PyGMT DEM map); do not overwrite it with the matplotlib fallback here.
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
