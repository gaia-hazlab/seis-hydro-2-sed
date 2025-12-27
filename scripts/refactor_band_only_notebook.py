"""Refactor notebook to band-only workflow.

Edits specific cell IDs in notebooks/uw_river_rumble_dec2025.ipynb to:
- remove broadband `proxies` variable usage
- standardize `band_proxies` to {sta_key: {band: Series}}

Creates a .bak backup next to the notebook on first run.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "uw_river_rumble_dec2025.ipynb"
BACKUP = NOTEBOOK.with_suffix(".ipynb.bak")


def _set_source(nb: dict, cell_id: str, text: str) -> None:
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            cell["source"] = [line if line.endswith("\n") else line + "\n" for line in text.splitlines()]
            return
    raise KeyError(f"Cell id not found: {cell_id}")


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text())
    nb_orig = deepcopy(nb)

    # Markdown: avoid the literal plural word "proxies" (optional)
    _set_source(
        nb,
        "dcc1b13a",
        r"""## Model being tested (what we fit)

### Observations

- Seismic “river rumble” proxy: $P(t)$, computed from band-limited seismic data in sliding windows (either RMS or PSD-integrated bandpower; optionally combined across components by RSS).
- River forcing proxy from USGS gage: $Q(t)$ (typically discharge, else stage).

### 1) Constant-lag alignment

We assume the seismic proxy responds to discharge with a **constant time lag** $\tau$ (hours):
$$
P(t) \;\text{is most directly related to}\; Q(t-\tau).
$$

In code we implement this by shifting the gage timestamps forward by $\tau$ and then correlating at common times:
$$
Q_{\tau}(t) = Q(t+\tau)\quad\Rightarrow\quad \text{compare } P(t)\;\text{with }Q_{\tau}(t).
$$

Interpretation:
- $\tau>0$: seismic proxy **lags** discharge (rumble persists after the flow rise).
- $\tau<0$: seismic proxy **leads** discharge (usually not expected physically; can indicate timing/metadata issues).

We select $\tau$ by scanning a grid (e.g., $|\tau|\le 24\,$h) and maximizing the correlation between transformed series (often in log space).

### 2) Log–log scaling (amplitude relation)

Once aligned, we often test a simple power-law scaling between proxy amplitude and discharge:
$$
P(t) \approx C\,Q(t-\tau)^{\beta}
$$
Equivalently in base-10 log space (what we correlate/fit most often):
$$
\log_{10} P(t) = a + \beta\,\log_{10} Q(t-\tau) + \varepsilon(t),\qquad a=\log_{10} C.
$$

Parameter meaning:
- $\tau$: constant lag between hydraulics and seismic proxy.
- $C$ (or $a$): overall gain (site/path/instrument/geometry).
- $\beta$: sensitivity exponent (how strongly proxy grows with discharge).
- $\varepsilon(t)$: residual variability (e.g., sediment supply changes, bed state, non-fluvial noise).

### 3) Two-band “flow vs bedload” diagnostics (used later)

We compute two proxy time series from different frequency bands:
- $P_{\mathrm{flow}}(t)$: lower-frequency band (more hydraulic control).
- $P_{\mathrm{bed}}(t)$: higher-frequency band (more bedload/impacts).

After applying the chosen $\tau$, we use two simple derived measures (on high-flow times, $Q\ge Q_c$):

**(a) Bedload residual after removing discharge trend**
$$
\log_{10} P_{\mathrm{bed}}(t) = m\,\log_{10} Q(t-\tau) + b + r(t)
$$
where $r(t)$ (the residual) is interpreted as a bed/supply-related modulation beyond hydraulics.

**(b) Band ratio (relative strengthening of bedload band)**
$$
R(t) = \log_{10}\!\left(\frac{P_{\mathrm{bed}}(t)}{P_{\mathrm{flow}}(t)}\right)=\log_{10}P_{\mathrm{bed}}(t)-\log_{10}P_{\mathrm{flow}}(t).
$$

Here $Q_c$ is an effective transport threshold (in this notebook approximated by a discharge percentile).
""",
    )

    # Band proxy computation for all stations
    _set_source(
        nb,
        "d53a27ff",
        """# Section 3: compute band proxy series by downloading waveforms in full UTC days
# (cached as daily MiniSEED). Broadband proxy dict is intentionally removed.

from utils import fetch_usgs_event_times, compute_proxy_from_fdsn

band_proxies: dict[str, dict[tuple[float, float], pd.Series]] = {}
event_cache: dict[str, pd.DatetimeIndex] = {}

bands = FLOW_BANDS + BEDLOAD_BANDS
combine_mode = "rss" if USE_RSS else "z"

# Uses CHANNEL/LOCATION/FDSN_CACHE_DIR from the Configuration section.
for sta_key in SEIS_KEYS:
    net, sta = sta_key.split(".", 1)
    print(f"Processing station {sta_key} (daily FDSN download + cache) ...")

    # Fetch nearby event times once per station (optional)
    ev_times = None
    if EXCLUDE_EARTHQUAKES:
        meta = STATION_META.get(sta_key, {})
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        if lat is not None and lon is not None:
            if sta_key not in event_cache:
                try:
                    event_cache[sta_key] = fetch_usgs_event_times(
                        START,
                        END,
                        lat,
                        lon,
                        min_magnitude=EQ_MIN_MAG,
                        maxradiuskm=EQ_MAXRADIUS_KM,
                    )
                    print(f"{sta_key}: {len(event_cache[sta_key])} events")
                except Exception as e:
                    print(f"{sta_key}: event query failed ({e}); continuing without masking")
                    event_cache[sta_key] = pd.DatetimeIndex([], tz="UTC")
            ev_times = event_cache[sta_key]

    sta_bands: dict[tuple[float, float], pd.Series] = {}
    for (f1, f2) in bands:
        band = (float(f1), float(f2))
        print(f"  - Computing proxy band {band[0]}–{band[1]} Hz")
        sta_bands[band] = compute_proxy_from_fdsn(
            net,
            sta,
            START,
            END,
            fmin=band[0],
            fmax=band[1],
            win_seconds=WIN_SECONDS,
            step_seconds=STEP_SECONDS,
            output=OUTPUT,
            method=PROXY_METHOD,
            remove_response=True,
            combine=combine_mode,
            components=COMPONENTS,
            event_times_utc=ev_times,
            event_buffer_s=EQ_BUFFER_SECONDS if EXCLUDE_EARTHQUAKES else 0,
            clip_impulsive_days=CLIP_IMPULSES,
            sta_seconds=STA_SECONDS,
            lta_seconds=LTA_SECONDS,
            trigger_on=TRIGGER_ON,
            trigger_off=TRIGGER_OFF,
            clip_sigma=CLIP_SIGMA,
            clip_mode=CLIP_MODE,
            despike_proxy=DESPIKE_PROXY,
            despike_window=DESPIKE_WINDOW,
            despike_z=DESPIKE_Z,
            despike_min_periods=DESPIKE_MIN_PERIODS,
            despike_fill=DESPIKE_FILL,
            client_name="IRIS",
            location=LOCATION,
            channel=CHANNEL,
            attach_response=True,
            cache_dir=FDSN_CACHE_DIR,
            use_cache=True,
        )

    band_proxies[sta_key] = sta_bands

# Quick sanity check
list(band_proxies.keys())[:5], {k: {b: len(s) for b, s in list(v.items())[:2]} for k, v in list(band_proxies.items())[:3]}
""",
    )

    # Band exploration + lag selection (focused station, reuses precomputed band_proxies)
    _set_source(
        nb,
        "0078c7fa",
        """# --- CC.PR01: band exploration + constant-lag selection (band-only) ---
from utils import estimate_constant_lag_seconds

sta_key = FOCUS_SEIS_KEY or (SEIS_KEYS[0] if len(SEIS_KEYS) else None)
if sta_key is None:
    raise RuntimeError("No stations in SEIS_KEYS")

if "band_proxies" not in globals() or not band_proxies or sta_key not in band_proxies:
    raise RuntimeError(f"band_proxies not available for {sta_key}. Run the band proxy computation cell first.")

gid = PAIRINGS.get(sta_key)
if gid is None or gid not in gauges:
    raise KeyError(f"No gauge found for {sta_key}")

g = gauges[gid].copy().sort_index()
sta_bands = band_proxies[sta_key]
bands = FLOW_BANDS + BEDLOAD_BANDS


def _corr_at_tau(proxy_s: pd.Series, gauge_df: pd.DataFrame, tau_s: float) -> tuple[float, int]:
    col = "discharge_cfs" if "discharge_cfs" in gauge_df.columns else "gage_height_ft"
    p = proxy_s.dropna().sort_index()
    gg = gauge_df[[col]].dropna().sort_index().copy()
    gg.index = gg.index + pd.Timedelta(seconds=float(tau_s))
    joined = pd.concat([p.rename("p"), gg[col].rename("g")], axis=1).dropna()
    if len(joined) < 30:
        return float("nan"), int(len(joined))
    tiny = np.finfo(float).tiny
    x = np.log10(joined["p"].clip(lower=tiny))
    y = np.log10(joined["g"].clip(lower=tiny))
    return float(x.corr(y)), int(len(joined))


# Pick a single constant lag based on the best FLOW band (by max corr over lag)
lag_s_by_band = {}
lag_tbl_by_band = {}
best_flow_band = None
best_flow_peak = -float("inf")

for band in FLOW_BANDS:
    band = (float(band[0]), float(band[1]))
    if band not in sta_bands:
        continue
    tau_s, tbl = estimate_constant_lag_seconds(
        sta_bands[band],
        g,
        gauge_col="discharge_cfs" if "discharge_cfs" in g.columns else "gage_height_ft",
        max_lag_hours=LAG_MAX_HOURS,
        step_minutes=LAG_STEP_MINUTES,
        min_pairs=30,
    )
    lag_s_by_band[band] = float(tau_s)
    lag_tbl_by_band[band] = tbl
    peak = float(tbl["corr"].max(skipna=True))
    if peak == peak and peak > best_flow_peak:
        best_flow_peak = peak
        best_flow_band = band

if best_flow_band is None:
    raise RuntimeError("Failed to choose a flow band for lag selection")

BEST_TAU_S = float(lag_s_by_band[best_flow_band])
BEST_FLOW_BAND = best_flow_band
print(
    f"Chosen constant lag tau = {BEST_TAU_S/3600:.2f} h "
    f"(selected using FLOW band {BEST_FLOW_BAND[0]}–{BEST_FLOW_BAND[1]} Hz)"
)


# Rank all configured bands at this chosen tau
rows = []
for (f1, f2) in bands:
    band = (float(f1), float(f2))
    if band not in sta_bands:
        continue
    r, n = _corr_at_tau(sta_bands[band], g, BEST_TAU_S)
    rows.append({"band": f"{band[0]}-{band[1]}", "fmin": band[0], "fmax": band[1], "corr_log10": r, "n": n})

band_score = pd.DataFrame(rows).sort_values("corr_log10", ascending=False)
display(band_score)
""",
    )

    # Plot all band proxies against discharge (focused station)
    _set_source(
        nb,
        "c15a07d2",
        """# --- Plot all band proxy series against discharge to compare before selecting best match ---
import matplotlib.pyplot as plt

sta_key = FOCUS_SEIS_KEY or (SEIS_KEYS[0] if len(SEIS_KEYS) else None)
if sta_key is None:
    raise RuntimeError("No stations in SEIS_KEYS")
if "band_proxies" not in globals() or not band_proxies or sta_key not in band_proxies:
    raise RuntimeError(f"band_proxies not available for {sta_key}. Run the band proxy computation cell first.")

gid = PAIRINGS.get(sta_key)
if gid is None or gid not in gauges:
    raise KeyError(f"No gauge found for {sta_key}")

g = gauges[gid].copy().sort_index()
sta_bands = band_proxies[sta_key]
bands = FLOW_BANDS + BEDLOAD_BANDS

fig, axes = plt.subplots(len(bands), 1, figsize=(12, 3 * len(bands)), sharex=True)
if len(bands) == 1:
    axes = [axes]

col = "discharge_cfs" if "discharge_cfs" in g.columns else "gage_height_ft"
gg_raw = g[[col]].dropna().sort_index()

for idx, (f1, f2) in enumerate(bands):
    band = (float(f1), float(f2))
    ax1 = axes[idx]

    if band not in sta_bands:
        ax1.set_title(f"{sta_key}: missing band {band[0]}–{band[1]} Hz")
        continue

    p_band = pd.Series(sta_bands[band]).dropna().sort_index()
    ax1.plot(p_band.index, p_band.values, lw=1.2, color="tab:blue", label=f"Proxy {band[0]}–{band[1]} Hz")
    ax1.set_ylabel(f"Proxy {band[0]}–{band[1]} Hz", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(gg_raw.index, gg_raw[col], lw=1.2, color="tab:orange", label=f"USGS {gid} {col}")
    ax2.set_ylabel(col, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    corr_val = band_score[band_score["band"] == f"{band[0]}-{band[1]}"]["corr_log10"].values
    corr_str = f"(r={corr_val[0]:.3f})" if len(corr_val) > 0 else ""
    ax1.set_title(f"{sta_key}: {band[0]}–{band[1]} Hz vs {gid} {corr_str}")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=9)

axes[-1].set_xlabel("Time (UTC)")
plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
plt.tight_layout()

out_png = FIG_DIR / f"{sta_key}_bands_vs_{gid}.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved {out_png}")
plt.show()

# Optional quick printout if BEST_* bands exist
if "BEST_FLOW_BAND" in globals():
    print(f"\nBest flow band (for lag selection): {BEST_FLOW_BAND[0]}–{BEST_FLOW_BAND[1]} Hz")
if "BEST_BED_BAND" in globals():
    print(f"Best bedload band (at fixed tau): {BEST_BED_BAND[0]}–{BEST_BED_BAND[1]} Hz")
if "BEST_TAU_S" in globals():
    print(f"Chosen constant lag: {BEST_TAU_S/3600:.2f} hours")
""",
    )

    # Bedrock-river oriented diagnostics: residual/ratio (focused station)
    _set_source(
        nb,
        "411858fc",
        """# --- Bedrock-river oriented diagnostics: threshold + residual/ratio (focused station) ---
import matplotlib.pyplot as plt

sta_key = FOCUS_SEIS_KEY or (SEIS_KEYS[0] if len(SEIS_KEYS) else None)
if sta_key is None:
    raise RuntimeError("No stations in SEIS_KEYS")
if "band_proxies" not in globals() or not band_proxies or sta_key not in band_proxies:
    raise RuntimeError(f"band_proxies not available for {sta_key}. Run the band proxy computation cell first.")

gid = PAIRINGS.get(sta_key)
if gid is None or gid not in gauges:
    raise KeyError(f"No gauge found for {sta_key}")

g = gauges[gid].copy().sort_index()
sta_bands = band_proxies[sta_key]

if "band_score" not in globals() or band_score is None or len(band_score) == 0:
    raise RuntimeError("band_score not available. Run the band exploration + lag selection cell first.")
if "BEST_TAU_S" not in globals() or BEST_TAU_S is None:
    raise RuntimeError("BEST_TAU_S not available. Run the band exploration + lag selection cell first.")
if "BEST_FLOW_BAND" not in globals() or BEST_FLOW_BAND is None:
    raise RuntimeError("BEST_FLOW_BAND not available. Run the band exploration + lag selection cell first.")

bed_scores = band_score[band_score["band"].isin([f"{b[0]}-{b[1]}" for b in BEDLOAD_BANDS])].copy()
if bed_scores.empty:
    raise RuntimeError("No BEDLOAD_BANDS found in band_score")
best_bed_row = bed_scores.iloc[0]
BEST_BED_BAND = (float(best_bed_row["fmin"]), float(best_bed_row["fmax"]))
print(
    f"Best bedload band at fixed tau: {BEST_BED_BAND[0]}–{BEST_BED_BAND[1]} Hz "
    f"(corr={best_bed_row['corr_log10']:.3f})"
)

p_flow = pd.Series(sta_bands[BEST_FLOW_BAND]).rename("p_flow")
p_bed = pd.Series(sta_bands[BEST_BED_BAND]).rename("p_bed")
col = "discharge_cfs" if "discharge_cfs" in g.columns else "gage_height_ft"

gg = g[[col]].dropna().sort_index().copy()
gg.index = gg.index + pd.Timedelta(seconds=float(BEST_TAU_S))
gg = gg[col].rename("q")

df = pd.concat([p_flow, p_bed, gg], axis=1).dropna()
if len(df) < 30:
    raise RuntimeError("Not enough overlapping samples after applying tau")

tiny = np.finfo(float).tiny
df["log_flow"] = np.log10(df["p_flow"].clip(lower=tiny))
df["log_bed"] = np.log10(df["p_bed"].clip(lower=tiny))
df["log_q"] = np.log10(df["q"].clip(lower=tiny))

Q_THRESH_PCT = 0.60
q_thr = float(df["q"].quantile(Q_THRESH_PCT))
df_hi = df[df["q"] >= q_thr].copy()
print(f"Thresholded analysis: keeping q >= {q_thr:.2f} ({Q_THRESH_PCT:.0%} percentile), n={len(df_hi)}/{len(df)}")

m, b = np.polyfit(df_hi["log_q"].values, df_hi["log_bed"].values, 1)
df_hi["bed_resid"] = df_hi["log_bed"] - (m * df_hi["log_q"] + b)
df_hi["ratio_bed_flow"] = df_hi["log_bed"] - df_hi["log_flow"]

fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(df_hi.index, df_hi["bed_resid"], lw=1.0)
ax[0].axhline(0, color="k", lw=0.8, alpha=0.5)
ax[0].set_ylabel("Bed residual (log10)")
ax[0].set_title(f"{sta_key} bedload residual (tau={BEST_TAU_S/3600:.2f} h)")

ax[1].plot(df_hi.index, df_hi["ratio_bed_flow"], lw=1.0, color="tab:green")
ax[1].axhline(0, color="k", lw=0.8, alpha=0.5)
ax[1].set_ylabel("log10(P_bed/P_flow)")
ax[1].set_xlabel("Time (UTC)")
plt.tight_layout()

out_png = FIG_DIR / f"{sta_key}_bedrock_diagnostics.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
print(f"Saved {out_png}")
plt.show()
""",
    )

    # Section 4 hysteresis (focused station, band-only)
    _set_source(
        nb,
        "f8501b0c",
        """# --- Hysteresis (proxy vs gage) for the focused station (band-only) ---
sta_key = FOCUS_SEIS_KEY or (SEIS_KEYS[0] if len(SEIS_KEYS) else None)
if sta_key is None:
    raise RuntimeError("No stations in SEIS_KEYS")

gid = PAIRINGS.get(sta_key)
if gid is None or gid not in gauges:
    raise KeyError(f"No gauge found for {sta_key}")

if "band_proxies" not in globals() or not band_proxies or sta_key not in band_proxies:
    raise RuntimeError(f"band_proxies not available for {sta_key}. Run the band proxy computation cell first.")

sta_bands = band_proxies[sta_key]

preferred = []
for nm in ("BEST_BED_BAND", "BEST_FLOW_BAND"):
    b = globals().get(nm)
    if b:
        preferred.append(b)
preferred.extend(list(sta_bands.keys()))

band = next((b for b in preferred if b in sta_bands), None)
if band is None:
    raise RuntimeError(f"No band proxy series available for {sta_key}")

dist_km = STATION_META.get(sta_key, {}).get("gage_distance_km")
dist_str = f" ({dist_km:.1f} km)" if dist_km is not None else ""
rss_str = "RSS" if USE_RSS else "Z"

title = f"Hysteresis: {sta_key} band {band[0]}–{band[1]} Hz vs USGS {gid}{dist_str} ({PROXY_METHOD}, {rss_str})"

hysteresis_plot(
    sta_bands[band],
    gauges[gid],
    title=title,
    save_dir=FIG_DIR,
    filename=f"{sta_key}_{band[0]}-{band[1]}Hz_hysteresis_vs_{gid}_section4.png",
)
""",
    )

    # Combined hysteresis for all bed-transport (BEDLOAD) bands
    _set_source(
        nb,
        "5865bdb3",
        """# --- Combined hysteresis for all bed-transport (BEDLOAD) bands (color-coded by band) ---
# Visualization improvements for very different proxy scales:
# - use a symlog y-scale so low-amplitude and high-amplitude clouds are both visible
# - overlay a binned-median trendline per band to emphasize nonlinearity (“knee”)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sta_key = FOCUS_SEIS_KEY or (SEIS_KEYS[0] if len(SEIS_KEYS) else None)
if sta_key is None:
    raise RuntimeError("No stations in SEIS_KEYS")

gid = PAIRINGS.get(sta_key)
if gid is None or gid not in gauges:
    raise KeyError(f"No gauge found for {sta_key}")

if "band_proxies" not in globals() or not isinstance(band_proxies, dict) or len(band_proxies) == 0:
    raise RuntimeError("band_proxies is not available. Run the band proxy computation cell(s) in Section 3 first.")
if sta_key not in band_proxies:
    raise KeyError(f"No band proxy series found for {sta_key}")

sta_bands = band_proxies[sta_key]
g = gauges[gid].copy().sort_index()

stage_col = (
    "gage_height_ft" if "gage_height_ft" in g.columns else ("discharge_cfs" if "discharge_cfs" in g.columns else None)
)
if stage_col is None:
    raise KeyError("Gauge dataframe missing gage_height_ft and discharge_cfs")

dist_km = STATION_META.get(sta_key, {}).get("gage_distance_km")
dist_str = f" ({dist_km:.1f} km)" if dist_km is not None else ""
rss_str = "RSS" if USE_RSS else "Z"
tolerance = "10min"

USE_SYMLOG_Y = True
N_BINS = 25
SHOW_BINNED_MEDIAN = True

# Choose a symlog threshold based on the smallest positive proxy seen.
min_pos = np.inf
for (f1, f2) in BEDLOAD_BANDS:
    band = (float(f1), float(f2))
    if band not in sta_bands:
        continue
    p = pd.Series(sta_bands[band]).dropna().astype(float)
    ppos = p[p > 0]
    if not ppos.empty:
        min_pos = min(min_pos, float(ppos.min()))

linthresh = (10.0 * min_pos) if np.isfinite(min_pos) else 1e-12

fig, ax = plt.subplots(figsize=(6.6, 5.6))
colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

x_all = g[stage_col].dropna().astype(float)
bin_edges = np.linspace(float(x_all.min()), float(x_all.max()), N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

for i, (f1, f2) in enumerate(BEDLOAD_BANDS):
    band = (float(f1), float(f2))
    if band not in sta_bands:
        continue

    p = pd.Series(sta_bands[band]).dropna().sort_index().astype(float)
    if p.empty:
        continue

    df = pd.DataFrame({"proxy": p})
    df[stage_col] = g[stage_col].reindex(df.index, method="nearest", tolerance=pd.Timedelta(tolerance))
    df = df.dropna()
    if df.empty:
        continue

    c = colors[i % len(colors)] if len(colors) else None
    ax.scatter(
        df[stage_col].values,
        df["proxy"].values,
        s=10,
        alpha=0.35,
        color=c,
        edgecolors="none",
        rasterized=True,
        label=f"{f1:g}–{f2:g} Hz",
    )

    if SHOW_BINNED_MEDIAN:
        x = df[stage_col].values.astype(float)
        y = df["proxy"].values.astype(float)
        bin_idx = np.digitize(x, bin_edges) - 1
        med = np.full(N_BINS, np.nan, dtype=float)
        for b in range(N_BINS):
            yy = y[bin_idx == b]
            if yy.size >= 10:
                med[b] = np.nanmedian(yy)
        ax.plot(bin_centers, med, color=c, lw=2.2, alpha=0.95)

if USE_SYMLOG_Y:
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_ylabel(f"Seismic proxy (symlog, linthresh={linthresh:.2g})")
else:
    ax.set_ylabel("Seismic proxy")

ax.set_xlabel(stage_col)
ax.set_title(f"Hysteresis: {sta_key} bed-transport bands vs USGS {gid}{dist_str} ({PROXY_METHOD}, {rss_str})")
ax.legend(title="Band", fontsize=9)
ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
plt.tight_layout()

out_png = FIG_DIR / f"{sta_key}_hysteresis_bed_bands_vs_{gid}_symlog.png"
fig.savefig(out_png, dpi=250, bbox_inches="tight")
print(f"Saved {out_png}")
plt.show()
""",
    )

    # Export fitted parameters + aligned time series (band-only)
    _set_source(
        nb,
        "f88b754a",
        """# --- Export fitted parameters + aligned time series (band-only) ---
from pathlib import Path

RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _pick_gauge_col(gauge_df: pd.DataFrame) -> str:
    if "discharge_cfs" in gauge_df.columns:
        return "discharge_cfs"
    if "gage_height_ft" in gauge_df.columns:
        return "gage_height_ft"
    raise KeyError("Gauge dataframe missing discharge_cfs and gage_height_ft")


def _fit_loglog_powerlaw(proxy_s: pd.Series, gauge_df: pd.DataFrame, *, tau_s: float, gauge_col: str, min_pairs: int = 30) -> dict:
    # Fit log10(P) = a + beta log10(Q) after shifting gauge by tau_s.
    p = proxy_s.dropna().sort_index()
    g = gauge_df[[gauge_col]].dropna().sort_index().copy()
    g.index = g.index + pd.Timedelta(seconds=float(tau_s))
    joined = pd.concat([p.rename("P"), g[gauge_col].rename("Q")], axis=1).dropna()
    n_pairs = int(len(joined))
    if n_pairs < min_pairs:
        return {"n_pairs": n_pairs, "corr_log10": float("nan"), "beta": float("nan"), "a": float("nan")}
    tiny = np.finfo(float).tiny
    x = np.log10(joined["Q"].clip(lower=tiny).astype(float).values)
    y = np.log10(joined["P"].clip(lower=tiny).astype(float).values)
    beta, a = np.polyfit(x, y, 1)
    corr = float(pd.Series(x).corr(pd.Series(y)))
    return {"n_pairs": n_pairs, "corr_log10": corr, "beta": float(beta), "a": float(a)}


def _build_aligned_timeseries(proxy_s: pd.Series, gauge_df: pd.DataFrame, *, tau_s: float, gauge_col: str) -> pd.DataFrame:
    p = proxy_s.rename("proxy").sort_index()
    g_raw = gauge_df[[gauge_col]].rename(columns={gauge_col: "gauge"}).sort_index()
    g_shift = g_raw.copy()
    g_shift.index = g_shift.index + pd.Timedelta(seconds=float(tau_s))
    g_shift = g_shift.rename(columns={"gauge": "gauge_shifted"})
    return pd.concat([p, g_raw, g_shift], axis=1).sort_index()


def _corr_at_tau(proxy_s: pd.Series, gauge_df: pd.DataFrame, *, tau_s: float, gauge_col: str, min_pairs: int = 30) -> float:
    p = proxy_s.dropna().sort_index()
    g = gauge_df[[gauge_col]].dropna().sort_index().copy()
    g.index = g.index + pd.Timedelta(seconds=float(tau_s))
    joined = pd.concat([p.rename("P"), g[gauge_col].rename("Q")], axis=1).dropna()
    if len(joined) < min_pairs:
        return float("nan")
    tiny = np.finfo(float).tiny
    x = np.log10(joined["Q"].clip(lower=tiny).astype(float))
    y = np.log10(joined["P"].clip(lower=tiny).astype(float))
    return float(x.corr(y))


if "band_proxies" not in globals() or not band_proxies:
    raise RuntimeError("band_proxies is missing/empty. Run the band proxy computation cell(s) first.")

rows = []

for sta_key, sta_bands in band_proxies.items():
    gid = PAIRINGS.get(sta_key)
    if gid is None or gid not in gauges:
        continue

    gauge_df = gauges[gid]
    gauge_col = _pick_gauge_col(gauge_df)

    # Choose tau per station using the best FLOW band
    best_tau_s = None
    best_flow_band = None
    best_peak = -float("inf")
    for band in FLOW_BANDS:
        band = (float(band[0]), float(band[1]))
        if band not in sta_bands:
            continue
        tau_s, tbl = estimate_constant_lag_seconds(
            sta_bands[band],
            gauge_df,
            gauge_col=gauge_col,
            max_lag_hours=LAG_MAX_HOURS,
            step_minutes=LAG_STEP_MINUTES,
            min_pairs=30,
        )
        peak = float(tbl["corr"].max(skipna=True))
        if peak == peak and peak > best_peak:
            best_peak = peak
            best_tau_s = float(tau_s)
            best_flow_band = band

    if best_tau_s is None or best_flow_band is None:
        continue

    # Choose a representative BEDLOAD band at fixed tau
    best_bed_band = None
    best_bed_corr = -float("inf")
    for band in BEDLOAD_BANDS:
        band = (float(band[0]), float(band[1]))
        if band not in sta_bands:
            continue
        r = _corr_at_tau(sta_bands[band], gauge_df, tau_s=best_tau_s, gauge_col=gauge_col)
        if r == r and r > best_bed_corr:
            best_bed_corr = r
            best_bed_band = band

    export_bands = [best_flow_band]
    if best_bed_band is not None and best_bed_band != best_flow_band:
        export_bands.append(best_bed_band)

    for band in export_bands:
        proxy_s = sta_bands[band]
        fit = _fit_loglog_powerlaw(proxy_s, gauge_df, tau_s=best_tau_s, gauge_col=gauge_col, min_pairs=30)
        ts = _build_aligned_timeseries(proxy_s, gauge_df, tau_s=best_tau_s, gauge_col=gauge_col)

        ts_out = RESULTS_DIR / f"{sta_key}_{band[0]}-{band[1]}Hz_timeseries.csv"
        ts.to_csv(ts_out)

        meta = STATION_META.get(sta_key, {})
        rows.append(
            {
                "seis_key": sta_key,
                "network": meta.get("network", sta_key.split(".", 1)[0]),
                "station": meta.get("station", sta_key.split(".", 1)[1] if "." in sta_key else sta_key),
                "gage_id": gid,
                "gage_distance_km": meta.get("gage_distance_km"),
                "proxy_method": PROXY_METHOD,
                "use_rss": bool(USE_RSS),
                "components": ",".join(COMPONENTS),
                "band_fmin": float(band[0]),
                "band_fmax": float(band[1]),
                "tau_s": float(best_tau_s),
                "tau_hours": float(best_tau_s) / 3600.0,
                "gauge_col": gauge_col,
                "n_pairs": fit["n_pairs"],
                "corr_log10": fit["corr_log10"],
                "beta": fit["beta"],
                "a": fit["a"],
                "timeseries_csv": ts_out.name,
            }
        )

fit_parameters = pd.DataFrame(rows).sort_values(["seis_key", "band_fmin", "band_fmax"]).reset_index(drop=True)
display(fit_parameters)

params_csv = RESULTS_DIR / "fit_parameters.csv"
fit_parameters.to_csv(params_csv, index=False)
print(f"Saved {params_csv}")
""",
    )

    # Comparisons loop (band-only)
    _set_source(
        nb,
        "c1d1952a",
        """# Run comparisons for station↔gage pairings derived from metadata (band-only)
rss_str = "RSS" if USE_RSS else "Z"

if "band_proxies" not in globals() or not band_proxies:
    raise RuntimeError("band_proxies is missing/empty. Run the band proxy computation cell(s) first.")

for sta_key, gid in PAIRINGS.items():
    if sta_key not in band_proxies:
        print(f"Skipping {sta_key} ↔ {gid} (missing band series)")
        continue
    if gid not in gauges:
        print(f"Skipping {sta_key} ↔ {gid} (missing gauge data)")
        continue

    dist_km = STATION_META.get(sta_key, {}).get("gage_distance_km")
    dist_str = f" ({dist_km:.1f} km)" if dist_km is not None else ""

    sta_bands = band_proxies[sta_key]
    preferred = []
    for nm in ("BEST_FLOW_BAND", "BEST_BED_BAND"):
        b = globals().get(nm)
        if b:
            preferred.append(b)
    preferred.extend(list(sta_bands.keys()))

    band = next((b for b in preferred if b in sta_bands), None)
    if band is None:
        print(f"Skipping {sta_key} ↔ {gid} (no bands available)")
        continue

    proxy_s = sta_bands[band]
    title = f"{sta_key} band {band[0]}–{band[1]} Hz vs USGS {gid}{dist_str} ({PROXY_METHOD}, {rss_str})"

    plot_proxy_and_gauge(
        proxy_s,
        gauges[gid],
        title=title,
        save_dir=FIG_DIR,
        filename=f"{sta_key}_{band[0]}-{band[1]}Hz_proxy_vs_{gid}_summary.png",
    )
    hysteresis_plot(
        proxy_s,
        gauges[gid],
        title=f"Hysteresis: {title}",
        save_dir=FIG_DIR,
        filename=f"{sta_key}_{band[0]}-{band[1]}Hz_hysteresis_vs_{gid}.png",
    )
""",
    )

    if not BACKUP.exists():
        BACKUP.write_text(json.dumps(nb_orig, ensure_ascii=False, indent=1))

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1))

    # Sanity checks in cell sources: look for standalone `proxies` usage
    # (and avoid matching `band_proxies`).
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    checks: dict[str, re.Pattern[str]] = {
        "proxies_assign": re.compile(r"(?<![A-Za-z0-9_])proxies\s*="),
        "proxies_index": re.compile(r"(?<![A-Za-z0-9_])proxies\s*\["),
        "proxies_items": re.compile(r"(?<![A-Za-z0-9_])proxies\.items\s*\("),
        "list_proxies": re.compile(r"list\(\s*proxies\b"),
    }
    found = {name: bool(rx.search(src)) for name, rx in checks.items()}
    print("Wrote:", NOTEBOOK)
    print("Backup:", BACKUP)
    print("Standalone broadband `proxies` patterns in sources:")
    for k, v in found.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
