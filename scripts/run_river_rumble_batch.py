#!/usr/bin/env python3
"""Batch runner for the uw_river_rumble_dec2025 notebook workflow.

Loops over all seismic stations that have an associated USGS gage in the
GAIA metadata CSV and generates:
- per-station aligned time series CSV(s): proxy + gauge + shifted gauge
- per-station plots: proxy-vs-gauge timeseries and hysteresis scatter
- a combined fit table: fit_parameters.csv

This script intentionally reuses the same helper functions as the notebook
(notebooks/utils.py) so results match the interactive workflow.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime


# Allow importing notebooks/utils.py as `utils`
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))

from utils import (  # noqa: E402
    compute_proxy_from_fdsn,
    estimate_constant_lag_seconds,
    fetch_usgs_event_times,
    fetch_usgs_gage_timeseries,
    load_station_gage_pairs,
    hysteresis_plot,
    plot_proxy_and_gauge,
)


DEFAULT_METADATA_CSV_URL = (
    "https://raw.githubusercontent.com/gaia-hazlab/gaia-data-downloaders/main/"
    "stations_by_basin_with_gages.csv"
)


@dataclass(frozen=True)
class Bands:
    flow: list[tuple[float, float]]
    bedload: list[tuple[float, float]]


def _parse_list_csv(s: str | None) -> list[str] | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_bands_csv(s: str) -> list[tuple[float, float]]:
    """Parse bands like: "0.5-2,1-5,2-8"."""
    out: list[tuple[float, float]] = []
    s = str(s).strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid band spec '{part}' (expected fmin-fmax)")
        a, b = part.split("-", 1)
        out.append((float(a), float(b)))
    return out


def _pick_gauge_col(gauge_df: pd.DataFrame) -> str:
    if "discharge_cfs" in gauge_df.columns:
        return "discharge_cfs"
    if "gage_height_ft" in gauge_df.columns:
        return "gage_height_ft"
    raise KeyError("Gauge dataframe missing discharge_cfs and gage_height_ft")


def _corr_at_tau(
    proxy_s: pd.Series,
    gauge_df: pd.DataFrame,
    *,
    tau_s: float,
    gauge_col: str,
    min_pairs: int = 30,
) -> float:
    p = proxy_s.dropna().sort_index()
    g = gauge_df[[gauge_col]].dropna().sort_index().copy()
    g.index = g.index + pd.Timedelta(seconds=float(tau_s))
    joined = pd.concat([p.rename("P"), g[gauge_col].rename("Q")], axis=1).dropna()
    if len(joined) < int(min_pairs):
        return float("nan")
    tiny = np.finfo(float).tiny
    x = np.log10(joined["Q"].clip(lower=tiny).astype(float))
    y = np.log10(joined["P"].clip(lower=tiny).astype(float))
    return float(x.corr(y))


def _fit_loglog_powerlaw(
    proxy_s: pd.Series,
    gauge_df: pd.DataFrame,
    *,
    tau_s: float,
    gauge_col: str,
    min_pairs: int = 30,
) -> dict[str, float]:
    p = proxy_s.dropna().sort_index()
    g = gauge_df[[gauge_col]].dropna().sort_index().copy()
    g.index = g.index + pd.Timedelta(seconds=float(tau_s))
    joined = pd.concat([p.rename("P"), g[gauge_col].rename("Q")], axis=1).dropna()
    n_pairs = int(len(joined))
    if n_pairs < int(min_pairs):
        return {"n_pairs": float(n_pairs), "corr_log10": float("nan"), "beta": float("nan"), "a": float("nan")}

    tiny = np.finfo(float).tiny
    x = np.log10(joined["Q"].clip(lower=tiny).astype(float).values)
    y = np.log10(joined["P"].clip(lower=tiny).astype(float).values)
    beta, a = np.polyfit(x, y, 1)
    corr = float(pd.Series(x).corr(pd.Series(y)))
    return {"n_pairs": float(n_pairs), "corr_log10": float(corr), "beta": float(beta), "a": float(a)}


def _build_aligned_timeseries(
    proxy_s: pd.Series,
    gauge_df: pd.DataFrame,
    *,
    tau_s: float,
    gauge_col: str,
) -> pd.DataFrame:
    p = proxy_s.rename("proxy").sort_index()
    g_raw = gauge_df[[gauge_col]].rename(columns={gauge_col: "gauge"}).sort_index()
    g_shift = g_raw.copy()
    g_shift.index = g_shift.index + pd.Timedelta(seconds=float(tau_s))
    g_shift = g_shift.rename(columns={"gauge": "gauge_shifted"})
    return pd.concat([p, g_raw, g_shift], axis=1).sort_index()


def _station_meta_from_pairs(pairs_df: pd.DataFrame) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}

    def _f(v):
        return None if pd.isna(v) else v

    for row in pairs_df.itertuples(index=False):
        seis_key = getattr(row, "seis_key")
        out[seis_key] = {
            "network": getattr(row, "network", None),
            "station": getattr(row, "station", None),
            "latitude": float(row.latitude) if hasattr(row, "latitude") and not pd.isna(row.latitude) else None,
            "longitude": float(row.longitude) if hasattr(row, "longitude") and not pd.isna(row.longitude) else None,
            "basin_name": _f(getattr(row, "basin_name", None)),
            "gage_id": getattr(row, "gage_id", None),
            "gage_distance_km": float(row.gage_distance_km)
            if hasattr(row, "gage_distance_km") and not pd.isna(row.gage_distance_km)
            else None,
            "gage_latitude": float(row.gage_latitude)
            if hasattr(row, "gage_latitude") and not pd.isna(row.gage_latitude)
            else None,
            "gage_longitude": float(row.gage_longitude)
            if hasattr(row, "gage_longitude") and not pd.isna(row.gage_longitude)
            else None,
            "gage_name": _f(getattr(row, "gage_name", None)),
        }

    return out


def run_one_station(
    *,
    sta_key: str,
    gid: str,
    gauges: dict[str, pd.DataFrame],
    station_meta: dict[str, dict[str, object]],
    bands: Bands,
    start: UTCDateTime,
    end: UTCDateTime,
    data_dir: Path,
    fig_dir: Path,
    results_dir: Path,
    # proxy controls
    proxy_method: str,
    output: str,
    use_rss: bool,
    components: tuple[str, ...],
    win_seconds: int,
    step_seconds: int,
    # fdsn controls
    client_name: str,
    location: str,
    channel: str,
    # EQ masking
    exclude_earthquakes: bool,
    eq_min_mag: float,
    eq_maxradius_km: float,
    eq_buffer_seconds: float,
    # impulse clipping
    clip_impulses: bool,
    sta_seconds: float,
    lta_seconds: float,
    trigger_on: float,
    trigger_off: float,
    clip_sigma: float,
    clip_mode: str,
    # despike
    despike_proxy: bool,
    despike_window: str,
    despike_z: float,
    despike_min_periods: int,
    despike_fill: str,
    # lag scan
    lag_max_hours: float,
    lag_step_minutes: int,
    min_pairs: int,
) -> tuple[list[dict[str, object]], list[Path]]:
    if gid not in gauges:
        raise KeyError(f"Missing gauge {gid} for station {sta_key}")

    gauge_df = gauges[gid].copy().sort_index()
    gauge_col = _pick_gauge_col(gauge_df)

    meta = station_meta.get(sta_key, {})

    # Optional event masking
    ev_times = None
    if exclude_earthquakes:
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        if lat is not None and lon is not None:
            ev_times = fetch_usgs_event_times(
                start,
                end,
                float(lat),
                float(lon),
                min_magnitude=float(eq_min_mag),
                maxradiuskm=float(eq_maxradius_km),
            )

    net, sta = sta_key.split(".", 1)
    combine_mode = "rss" if bool(use_rss) else "z"

    sta_bands: dict[tuple[float, float], pd.Series] = {}
    all_bands = list(bands.flow) + list(bands.bedload)

    print(f"[{sta_key}] Computing proxies for {len(all_bands)} bands …")
    for (f1, f2) in all_bands:
        band = (float(f1), float(f2))
        sta_bands[band] = compute_proxy_from_fdsn(
            net,
            sta,
            start,
            end,
            fmin=band[0],
            fmax=band[1],
            win_seconds=int(win_seconds),
            step_seconds=int(step_seconds),
            output=str(output),
            method=str(proxy_method),
            remove_response=True,
            combine=combine_mode,
            components=tuple(components),
            event_times_utc=ev_times,
            event_buffer_s=float(eq_buffer_seconds) if exclude_earthquakes else 0.0,
            clip_impulsive_days=bool(clip_impulses),
            sta_seconds=float(sta_seconds),
            lta_seconds=float(lta_seconds),
            trigger_on=float(trigger_on),
            trigger_off=float(trigger_off),
            clip_sigma=float(clip_sigma),
            clip_mode=str(clip_mode),
            despike_proxy=bool(despike_proxy),
            despike_window=str(despike_window),
            despike_z=float(despike_z),
            despike_min_periods=int(despike_min_periods),
            despike_fill=str(despike_fill),
            client_name=str(client_name),
            location=str(location),
            channel=str(channel),
            attach_response=True,
            cache_dir=data_dir / "fdsn_cache",
            use_cache=True,
        )

    # Choose tau per station using the best FLOW band
    best_tau_s = None
    best_flow_band = None
    best_peak = -float("inf")

    for band in bands.flow:
        band = (float(band[0]), float(band[1]))
        proxy_s = sta_bands.get(band)
        if proxy_s is None or proxy_s.empty:
            continue
        try:
            tau_s, tbl = estimate_constant_lag_seconds(
                proxy_s,
                gauge_df,
                gauge_col=gauge_col,
                max_lag_hours=float(lag_max_hours),
                step_minutes=int(lag_step_minutes),
                min_pairs=int(min_pairs),
            )
        except Exception as e:
            print(f"[{sta_key}] Lag scan failed for flow band {band}: {e}")
            continue

        peak = float(tbl["corr"].max(skipna=True))
        if peak == peak and peak > best_peak:
            best_peak = peak
            best_tau_s = float(tau_s)
            best_flow_band = band

    if best_tau_s is None or best_flow_band is None:
        raise RuntimeError(f"[{sta_key}] Could not select a best tau from FLOW bands")

    # Choose best BEDLOAD band at fixed tau
    best_bed_band = None
    best_bed_corr = -float("inf")
    for band in bands.bedload:
        band = (float(band[0]), float(band[1]))
        proxy_s = sta_bands.get(band)
        if proxy_s is None or proxy_s.empty:
            continue
        r = _corr_at_tau(proxy_s, gauge_df, tau_s=best_tau_s, gauge_col=gauge_col, min_pairs=min_pairs)
        if r == r and r > best_bed_corr:
            best_bed_corr = float(r)
            best_bed_band = band

    export_bands = [best_flow_band]
    if best_bed_band is not None and best_bed_band != best_flow_band:
        export_bands.append(best_bed_band)

    rows: list[dict[str, object]] = []
    written: list[Path] = []

    dist_km = meta.get("gage_distance_km")
    dist_str = f" ({dist_km:.1f} km)" if isinstance(dist_km, (float, int)) else ""
    rss_str = "RSS" if use_rss else "Z"

    for band in export_bands:
        proxy_s = sta_bands[band]
        fit = _fit_loglog_powerlaw(proxy_s, gauge_df, tau_s=best_tau_s, gauge_col=gauge_col, min_pairs=min_pairs)
        ts = _build_aligned_timeseries(proxy_s, gauge_df, tau_s=best_tau_s, gauge_col=gauge_col)

        results_dir.mkdir(parents=True, exist_ok=True)
        ts_out = results_dir / f"{sta_key}_{band[0]}-{band[1]}Hz_timeseries.csv"
        ts.to_csv(ts_out)
        written.append(ts_out)

        # Plots
        fig_dir.mkdir(parents=True, exist_ok=True)
        title = f"{sta_key} band {band[0]}–{band[1]} Hz vs USGS {gid}{dist_str} ({proxy_method}, {rss_str})"
        plot_proxy_and_gauge(
            proxy_s,
            gauge_df,
            title=title,
            save_dir=fig_dir,
            filename=f"{sta_key}_{band[0]}-{band[1]}Hz_proxy_vs_{gid}_summary.png",
        )
        hysteresis_plot(
            proxy_s,
            gauge_df,
            title=f"Hysteresis: {title}",
            save_dir=fig_dir,
            filename=f"{sta_key}_{band[0]}-{band[1]}Hz_hysteresis_vs_{gid}.png",
        )

        rows.append(
            {
                "seis_key": sta_key,
                "network": meta.get("network", net),
                "station": meta.get("station", sta),
                "gage_id": gid,
                "gage_distance_km": dist_km,
                "proxy_method": str(proxy_method),
                "use_rss": bool(use_rss),
                "components": ",".join(components),
                "band_fmin": float(band[0]),
                "band_fmax": float(band[1]),
                "tau_s": float(best_tau_s),
                "tau_hours": float(best_tau_s) / 3600.0,
                "gauge_col": gauge_col,
                "n_pairs": int(fit["n_pairs"]),
                "corr_log10": float(fit["corr_log10"]),
                "beta": float(fit["beta"]),
                "a": float(fit["a"]),
                "timeseries_csv": ts_out.name,
            }
        )

    return rows, written


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-generate discharge + seismic-power products for all station↔gage pairs.",
    )

    p.add_argument("--start", default="2025-12-01T00:00:00", help="UTC start (ISO8601)")
    p.add_argument("--end", default="2025-12-24T00:00:00", help="UTC end (ISO8601)")

    p.add_argument("--metadata-csv-url", default=DEFAULT_METADATA_CSV_URL)
    p.add_argument("--network-filter", default="", help="Comma-separated networks (e.g. CC,UW). Empty=all")
    p.add_argument("--basin-filter", default="", help="Comma-separated basin_name filters. Empty=all")
    p.add_argument("--max-pairs", type=int, default=None, help="Limit number of station↔gage pairs")
    p.add_argument("--focus-seis-key", default="", help="If set, run only this station key (e.g. CC.PR03)")

    p.add_argument(
        "--data-dir",
        default=str(ROOT / "notebooks" / "data"),
        help="Data/cache dir (default: notebooks/data)",
    )
    p.add_argument(
        "--fig-dir",
        default=str(ROOT / "notebooks" / "figures"),
        help="Figure output dir (default: notebooks/figures)",
    )

    # Proxy definition
    p.add_argument("--proxy-method", default="bandpower", choices=["bandpower", "rms"])
    p.add_argument("--use-rss", action="store_true", help="Combine Z/N/E as RSS (default: False)")
    p.add_argument("--components", default="Z,N,E", help="Components used when --use-rss is set")
    p.add_argument("--output", default="velocity", choices=["velocity", "acceleration"])
    p.add_argument("--win-seconds", type=int, default=600)
    p.add_argument("--step-seconds", type=int, default=300)

    p.add_argument("--flow-bands", default="0.5-2,1-5,2-8")
    p.add_argument("--bedload-bands", default="5-15,5-20,10-30")

    # Lag scan
    p.add_argument("--lag-max-hours", type=float, default=24.0)
    p.add_argument("--lag-step-minutes", type=int, default=10)
    p.add_argument("--min-pairs", type=int, default=30)

    # FDSN parameters
    p.add_argument("--client", default="IRIS")
    p.add_argument("--location", default="*")
    p.add_argument("--channel", default="BH?")

    # Earthquake exclusion
    p.add_argument("--exclude-earthquakes", action="store_true", help="Mask proxy windows near USGS events")
    p.add_argument("--eq-min-mag", type=float, default=3.5)
    p.add_argument("--eq-maxradius-km", type=float, default=500.0)
    p.add_argument("--eq-buffer-seconds", type=float, default=20 * 60)

    # Impulse clipping (STA/LTA)
    p.add_argument("--clip-impulses", action="store_true")
    p.add_argument("--sta-seconds", type=float, default=1.0)
    p.add_argument("--lta-seconds", type=float, default=20.0)
    p.add_argument("--trigger-on", type=float, default=3.5)
    p.add_argument("--trigger-off", type=float, default=1.0)
    p.add_argument("--clip-sigma", type=float, default=2.0)
    p.add_argument("--clip-mode", default="symmetric", choices=["symmetric", "upper", "abs"])

    # Despike proxy
    p.add_argument("--despike-proxy", action="store_true")
    p.add_argument("--despike-window", default="6H")
    p.add_argument("--despike-z", type=float, default=8.0)
    p.add_argument("--despike-min-periods", type=int, default=10)
    p.add_argument("--despike-fill", default="interpolate", choices=["none", "interpolate"])

    return p


def main() -> int:
    args = build_argparser().parse_args()

    start = UTCDateTime(str(args.start))
    end = UTCDateTime(str(args.end))
    if end <= start:
        raise ValueError("--end must be after --start")

    data_dir = Path(args.data_dir).expanduser().resolve()
    fig_dir = Path(args.fig_dir).expanduser().resolve()
    results_dir = data_dir / "results"

    network_filter = _parse_list_csv(args.network_filter)
    basin_filter = _parse_list_csv(args.basin_filter)
    focus_key = str(args.focus_seis_key).strip() or None

    flow_bands = _parse_bands_csv(args.flow_bands)
    bedload_bands = _parse_bands_csv(args.bedload_bands)
    if not flow_bands:
        raise ValueError("--flow-bands parsed to empty")
    if not bedload_bands:
        raise ValueError("--bedload-bands parsed to empty")

    components = tuple([c.strip() for c in str(args.components).split(",") if c.strip()])

    pairs_df = load_station_gage_pairs(
        str(args.metadata_csv_url),
        data_dir=data_dir,
        network_filter=network_filter,
        basin_filter=basin_filter,
        max_pairs=args.max_pairs,
        use_cache=True,
        choose_closest_gage=True,
    )

    if focus_key is not None:
        pairs_df = pairs_df[pairs_df["seis_key"] == focus_key].reset_index(drop=True)

    if len(pairs_df) == 0:
        print("No station↔gage pairs after filtering; nothing to do.")
        return 0

    station_meta = _station_meta_from_pairs(pairs_df)

    seis_keys = pairs_df["seis_key"].tolist()
    gauge_ids = sorted(pairs_df["gage_id"].unique().tolist())
    pairings = dict(zip(pairs_df["seis_key"].tolist(), pairs_df["gage_id"].tolist()))

    # Fetch gauges
    gauges: dict[str, pd.DataFrame] = {}
    for gid in gauge_ids:
        print(f"Loading USGS site {gid} (cached if available)…")
        try:
            gauges[gid] = fetch_usgs_gage_timeseries(str(gid), start, end, data_dir=data_dir, use_cache=True)
        except Exception as e:
            print(f"  Failed site {gid}: {e}")

    bands = Bands(flow=flow_bands, bedload=bedload_bands)

    all_rows: list[dict[str, object]] = []
    all_written: list[Path] = []

    for sta_key in seis_keys:
        gid = pairings.get(sta_key)
        if gid is None:
            continue
        if gid not in gauges:
            print(f"Skipping {sta_key} ↔ {gid} (missing gauge data)")
            continue

        try:
            rows, written = run_one_station(
                sta_key=sta_key,
                gid=str(gid),
                gauges=gauges,
                station_meta=station_meta,
                bands=bands,
                start=start,
                end=end,
                data_dir=data_dir,
                fig_dir=fig_dir,
                results_dir=results_dir,
                proxy_method=str(args.proxy_method),
                output=str(args.output),
                use_rss=bool(args.use_rss),
                components=components,
                win_seconds=int(args.win_seconds),
                step_seconds=int(args.step_seconds),
                client_name=str(args.client),
                location=str(args.location),
                channel=str(args.channel),
                exclude_earthquakes=bool(args.exclude_earthquakes),
                eq_min_mag=float(args.eq_min_mag),
                eq_maxradius_km=float(args.eq_maxradius_km),
                eq_buffer_seconds=float(args.eq_buffer_seconds),
                clip_impulses=bool(args.clip_impulses),
                sta_seconds=float(args.sta_seconds),
                lta_seconds=float(args.lta_seconds),
                trigger_on=float(args.trigger_on),
                trigger_off=float(args.trigger_off),
                clip_sigma=float(args.clip_sigma),
                clip_mode=str(args.clip_mode),
                despike_proxy=bool(args.despike_proxy),
                despike_window=str(args.despike_window),
                despike_z=float(args.despike_z),
                despike_min_periods=int(args.despike_min_periods),
                despike_fill=str(args.despike_fill),
                lag_max_hours=float(args.lag_max_hours),
                lag_step_minutes=int(args.lag_step_minutes),
                min_pairs=int(args.min_pairs),
            )
            all_rows.extend(rows)
            all_written.extend(written)
        except Exception as e:
            print(f"[{sta_key}] Failed: {e}")
            continue

    if not all_rows:
        print("No station outputs were generated (all stations failed or no overlap).")
        return 2

    results_dir.mkdir(parents=True, exist_ok=True)
    fit_parameters = pd.DataFrame(all_rows).sort_values(["seis_key", "band_fmin", "band_fmax"]).reset_index(drop=True)
    params_csv = results_dir / "fit_parameters.csv"
    fit_parameters.to_csv(params_csv, index=False)

    print(f"\nWrote {len(all_written)} time-series CSV files")
    print(f"Wrote fit table: {params_csv}")
    print(f"Figures in: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
