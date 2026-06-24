"""Fast multi-band seismic proxy: one Welch PSD per window, integrate every band.

The original pipeline (``notebooks.utils.compute_proxy_from_fdsn``) recomputes the
PSD from scratch *per band*, so a 6-band run reads/inverts/Welch-transforms each
waveform day six times. This module computes ONE high-resolution Welch PSD per
window per (component, day) and integrates all requested bands from it — roughly
``len(bands)``× faster, and it returns every band (so the full β(f) curve is
available, not just the two bands the batch exports).

It reuses the cleaning/response helpers from ``notebooks.utils`` so results match
the validated single-band path to within numerical tolerance.

Drop-in usage::

    from riverseis.fast_proxy import compute_multiband_proxy_from_fdsn
    bands = [(0.5,2),(1,5),(2,8),(5,15),(10,30),(30,60)]
    proxies = compute_multiband_proxy_from_fdsn("CC","PR03", t0, t1, bands=bands,
                                                cache_dir=..., use_cache=True)
    # -> {(0.5,2.0): Series, ...}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy import read as obspy_read

# Reuse the validated helpers from the notebook engine.
_NB = Path(__file__).resolve().parents[2] / "notebooks"
sys.path.insert(0, str(_NB))
import utils  # noqa: E402


def multiband_bandpower(
    x: np.ndarray, *, starttime: UTCDateTime, sr: float,
    bands: list[tuple[float, float]], win_s: int, step_s: int,
) -> dict[tuple[float, float], pd.Series]:
    """Welch PSD once per window; integrate each band. Returns {band: Series}."""
    from scipy.signal import welch

    sr = float(sr)
    if sr <= 0:
        raise ValueError("Invalid sampling rate")
    nyq = 0.5 * sr
    win = int(win_s * sr)
    step = int(step_s * sr)
    if win < 8 or step < 1:
        raise ValueError("Window/step too small")

    # Pre-clip band edges to Nyquist; keep the original (fmin,fmax) as dict key.
    eff = [(b, (float(b[0]), min(float(b[1]), 0.99 * nyq))) for b in bands]

    out: dict[tuple[float, float], tuple[list, list]] = {b: ([], []) for b in bands}
    t0 = starttime.timestamp
    nperseg = min(win, int(10 * sr))
    for i in range(0, len(x) - win + 1, step):
        seg = x[i: i + win]
        f, pxx = welch(seg, fs=sr, nperseg=nperseg, detrend="constant", scaling="density")
        ts = pd.to_datetime(t0 + (i + win / 2) / sr, unit="s", utc=True)
        for b, (f1, f2) in eff:
            if f2 <= f1:
                continue
            m = (f >= f1) & (f <= f2)
            val = float(np.trapz(pxx[m], f[m])) if np.any(m) else float("nan")
            out[b][0].append(val)
            out[b][1].append(ts)
    def _utc_index(t):
        idx = pd.DatetimeIndex(t, name="time_utc")
        return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return {
        b: pd.Series(v, index=_utc_index(t)).dropna()
        for b, (v, t) in out.items()
    }


def _component_multiband(tr, *, bands, win_s, step_s, output, remove_response,
                         clip_impulsive_days, clip_kwargs) -> dict:
    tr2 = tr.copy()
    if clip_impulsive_days:
        tr2 = utils.clip_trace_days_on_stalta_impulses(tr2, **clip_kwargs)
    tr2.detrend("demean")
    tr2.taper(0.01)
    if remove_response:
        # strict: raises on missing/failed response (no raw-counts fallback)
        tr2 = utils._remove_instrument_response(
            tr2, fmin=min(b[0] for b in bands), fmax=max(b[1] for b in bands),
            output=output,
        )
    return multiband_bandpower(
        tr2.data.astype("float64"), starttime=tr2.stats.starttime,
        sr=float(tr2.stats.sampling_rate), bands=bands, win_s=win_s, step_s=step_s,
    )


def compute_multiband_proxy_from_fdsn(
    net: str, sta: str, start_utc: UTCDateTime, end_utc: UTCDateTime, *,
    bands: list[tuple[float, float]], win_seconds: int = 600, step_seconds: int = 300,
    output: str = "velocity", combine: str = "rss", components=("Z", "N", "E"),
    remove_response: bool = True, client_name: str = "IRIS", location: str = "*",
    channel: str = "BH?", cache_dir: Path | None = None, use_cache: bool = True,
    clip_impulsive_days: bool = False, sta_seconds: float = 1.0, lta_seconds: float = 20.0,
    trigger_on: float = 3.5, trigger_off: float = 1.0, clip_sigma: float = 2.0,
    clip_mode: str = "symmetric", clip_scope: str = "windows",
    despike_proxy: bool = False, despike_window: str = "6H", despike_z: float = 8.0,
    despike_min_periods: int = 10, despike_fill: str = "none",
    event_times_utc: pd.DatetimeIndex | None = None, event_buffer_s: float = 0.0,
    pad_seconds: float | None = None,
) -> dict[tuple[float, float], pd.Series]:
    """Multi-band proxy via whole-UTC-day blocks; PSD computed once per window.

    Returns {band: proxy Series}. Mirrors ``utils.compute_proxy_from_fdsn`` but
    for all bands at once. RSS combines per-band across components.
    """
    from obspy.clients.fdsn import Client

    net, sta = str(net).strip(), str(sta).strip()
    bands = [(float(a), float(b)) for a, b in bands]
    if pad_seconds is None:
        pad_seconds = float(win_seconds)
    if cache_dir is not None:
        cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)

    client = Client(client_name, timeout=120)
    inventory = None
    if remove_response:
        try:
            inventory = client.get_stations(network=net, station=sta, location=location,
                                            channel=channel, starttime=start_utc,
                                            endtime=end_utc, level="response")
        except Exception:
            inventory = None

    clip_kwargs = dict(sta_seconds=sta_seconds, lta_seconds=lta_seconds,
                       trigger_on=trigger_on, trigger_off=trigger_off,
                       clip_sigma=clip_sigma, clip_mode=clip_mode, clip_scope=clip_scope)

    # accumulate per-band, per-component daily series
    parts: dict[tuple[float, float], dict[str, list[pd.Series]]] = {
        b: {c: [] for c in components} for b in bands
    }
    t0 = start_utc
    while t0 < end_utc:
        next_midnight = UTCDateTime(t0.year, t0.month, t0.day) + 24 * 3600
        t1 = min(next_midnight, end_utc)
        dl0, dl1 = t0 - float(pad_seconds), t1 + float(pad_seconds)

        cache_file = None
        if cache_dir is not None:
            day = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}"
            safe_loc = (location or "").replace("*", "STAR")
            safe_chan = (channel or "").replace("*", "STAR").replace("?", "Q")
            cache_file = cache_dir / f"{net}.{sta}.{safe_loc}.{safe_chan}.{day}.pad{int(round(pad_seconds))}s.mseed"

        st = None
        if use_cache and cache_file is not None and cache_file.exists() and cache_file.stat().st_size > 0:
            try:
                st = obspy_read(str(cache_file))
            except Exception:
                st = None
        if st is None:
            try:
                st = client.get_waveforms(net, sta, location, channel, dl0, dl1, attach_response=True)
                if cache_file is not None:
                    try:
                        st.write(str(cache_file), format="MSEED")
                    except Exception:
                        pass
            except Exception as e:
                print(f"  Failed day {net}.{sta} {dl0}-{dl1}: {e}")
                t0 = t1; continue

        try:
            st.merge(method=1, fill_value="interpolate")
            st.trim(dl0, dl1)
            if inventory is not None:
                try:
                    st.attach_response(inventory)
                except Exception:
                    pass
            for comp in components:
                tr = utils._select_trace_by_component(st, comp)
                if tr is None:
                    continue
                try:
                    band_series = _component_multiband(
                        tr, bands=bands, win_s=win_seconds, step_s=step_seconds,
                        output=output, remove_response=remove_response,
                        clip_impulsive_days=clip_impulsive_days, clip_kwargs=clip_kwargs)
                except Exception as e:
                    print(f"  Skipping {net}.{sta}.{comp} day {t0}: {e}")
                    continue
                t0_ts = pd.to_datetime(t0.datetime, utc=True)
                t1_ts = pd.to_datetime(t1.datetime, utc=True)
                for b, s in band_series.items():
                    s = s.loc[(s.index >= t0_ts) & (s.index < t1_ts)]
                    if not s.empty:
                        parts[b][comp].append(s)
        finally:
            del st
        t0 = t1

    # combine per band
    out: dict[tuple[float, float], pd.Series] = {}
    for b in bands:
        comp_series = {}
        for c in components:
            if parts[b][c]:
                s = pd.concat(parts[b][c]).sort_index()
                comp_series[c] = s[~s.index.duplicated(keep="first")]
        if not comp_series:
            out[b] = pd.Series([], dtype="float64")
            continue
        if combine.lower().strip() == "rss":
            df = pd.concat(comp_series, axis=1).dropna(how="any")
            res = pd.Series(np.sqrt((df.values ** 2).sum(axis=1)), index=df.index, name="proxy_rss")
        else:
            res = comp_series.get("Z", next(iter(comp_series.values())))
        if event_buffer_s and event_times_utc is not None:
            keep = utils.mask_times_near_events(res.index, event_times_utc, event_buffer_s)
            res = res.loc[keep]
        if despike_proxy:
            res = utils.despike_timeseries(res, window=despike_window, z=despike_z,
                                           min_periods=despike_min_periods, fill=despike_fill)
        out[b] = res
    return out
