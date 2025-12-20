from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
import io
import re

import numpy as np
import pandas as pd
import requests
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta, trigger_onset


GAIA_USGS_STREAM_GAGE_BASE_URL = (
    "https://raw.githubusercontent.com/gaia-hazlab/gaia-data-downloaders/main/USGS_Stream_Gage"
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two WGS84 lat/lon points."""
    r_km = 6371.0088
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r_km * np.arcsin(np.sqrt(a)))


def fetch_usgs_site_metadata(
    site: str,
    data_dir: Path,
    *,
    use_cache: bool = True,
) -> dict[str, object]:
    """Fetch basic USGS site metadata (lat/lon/name).

    Uses the NWIS "site" service (RDB) and caches results under
    `<data_dir>/usgs_site_meta/`.
    """
    site = str(site).strip()
    cache_dir = data_dir / "usgs_site_meta"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{site}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        url = "https://waterservices.usgs.gov/nwis/site/"
        params = {"format": "rdb", "sites": site, "siteStatus": "all"}
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()

        lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
        if len(lines) < 3:
            raise RuntimeError(f"USGS site service returned no data for site={site}")

        header = lines[0].split("\t")
        data = lines[2].split("\t")  # first data row
        row = dict(zip(header, data, strict=False))
        df = pd.DataFrame([row])
        df.to_csv(cache_file, index=False)

    # The canonical columns here are `dec_lat_va`, `dec_long_va`, `station_nm`.
    lat = pd.to_numeric(df.loc[0, "dec_lat_va"], errors="coerce") if "dec_lat_va" in df.columns else np.nan
    lon = (
        pd.to_numeric(df.loc[0, "dec_long_va"], errors="coerce") if "dec_long_va" in df.columns else np.nan
    )
    name = df.loc[0, "station_nm"] if "station_nm" in df.columns else None

    return {
        "site": site,
        "latitude": None if pd.isna(lat) else float(lat),
        "longitude": None if pd.isna(lon) else float(lon),
        "name": None if pd.isna(name) else str(name),
    }


def _repair_split_datetime_lines(text: str) -> str:
    """Repair GAIA-export CSVs where the ISO datetime is split across lines.

    Example observed pattern:
      2025-11-01\n06:00:00+00:00,USGS,14233500,...
    """
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2})\s*\n(\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})"
    )
    return pattern.sub(r"\1 \2", text)


def fetch_usgs_gage_timeseries(
    site: str,
    start_utc: UTCDateTime,
    end_utc: UTCDateTime,
    data_dir: Path,
    *,
    source: str = "gaia_github",
    use_cache: bool = True,
    fallback_to_nwis: bool = True,
    gaia_base_url: str = GAIA_USGS_STREAM_GAGE_BASE_URL,
) -> pd.DataFrame:
    """Fetch USGS gage time series.

    Primary mode (default): download pre-generated data from the GAIA
    `USGS_Stream_Gage/<gage_id>/` folders on GitHub.

    Returns a UTC-indexed DataFrame. Typical columns include:
    - discharge_cfs (+ discharge_qual)
    - gage_height_ft (+ gage_height_qual)  (if present in that gage export)

    If `source="nwis"` or GAIA download fails and `fallback_to_nwis=True`,
    falls back to the live USGS NWIS IV service.
    """
    site = str(site).strip()
    data_dir.mkdir(parents=True, exist_ok=True)

    if source.lower() == "nwis":
        return fetch_usgs_nwis_iv(site, start_utc, end_utc, data_dir=data_dir, use_cache=use_cache)

    try:
        cache_dir = data_dir / "gaia_usgs_stream_gage" / site
        cache_dir.mkdir(parents=True, exist_ok=True)
        parsed_cache = cache_dir / f"{site}_parsed.csv"

        if use_cache and parsed_cache.exists():
            df = pd.read_csv(parsed_cache, parse_dates=["time_utc"], index_col="time_utc")
            df.index = pd.to_datetime(df.index, utc=True)
        else:
            data_url = f"{gaia_base_url}/{site}/{site}_data.csv"
            header_url = f"{gaia_base_url}/{site}/{site}_header.txt"

            r = requests.get(data_url, timeout=60)
            r.raise_for_status()
            raw_text = r.text
            raw_text = _repair_split_datetime_lines(raw_text)

            # Best-effort: cache the header for provenance/debugging
            try:
                rh = requests.get(header_url, timeout=30)
                if rh.ok:
                    (cache_dir / f"{site}_header.txt").write_text(rh.text)
            except Exception:
                pass

            df = pd.read_csv(io.StringIO(raw_text))
            if "datetime" not in df.columns:
                raise ValueError(f"GAIA gage CSV for {site} missing 'datetime' column")

            t = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
            df = df.drop(columns=["datetime"]).copy()
            df.insert(0, "time_utc", t)
            df = df.dropna(subset=["time_utc"]).set_index("time_utc").sort_index()

            # Coerce obvious numeric columns
            for col in list(df.columns):
                if col.endswith("_cfs") or col.endswith("_ft"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["site"] = site
            df.to_csv(parsed_cache)

        start_ts = pd.to_datetime(start_utc.datetime, utc=True)
        end_ts = pd.to_datetime(end_utc.datetime, utc=True)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        return df
    except Exception as e:
        if not fallback_to_nwis:
            raise
        print(f"  GAIA gage fetch failed for {site} ({e}); falling back to NWIS IV")
        return fetch_usgs_nwis_iv(site, start_utc, end_utc, data_dir=data_dir, use_cache=use_cache)


def load_station_gage_pairs(
    csv_url: str,
    data_dir: Path,
    network_filter: Iterable[str] | None = None,
    basin_filter: Iterable[str] | None = None,
    max_pairs: int | None = None,
    use_cache: bool = True,
    choose_closest_gage: bool = True,
) -> pd.DataFrame:
    """Load GAIA station↔USGS gage pairs from a CSV URL (cached locally).

    Note: the GAIA table can list multiple gages per seismic station.
    If `choose_closest_gage=True` (default), this function queries USGS site
    metadata to get each gage's lat/lon, computes station↔gage distance, and
    keeps the closest gage for each station.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_file = data_dir / "stations_by_basin_with_gages.csv"

    if use_cache and cache_file.exists():
        pairs = pd.read_csv(cache_file)
    else:
        r = requests.get(csv_url, timeout=30)
        r.raise_for_status()
        cache_file.write_bytes(r.content)
        pairs = pd.read_csv(cache_file)

    pairs = pairs.copy()

    # Normalize common columns
    for col in ("network", "station", "gage_id"):
        if col in pairs.columns:
            pairs[col] = pairs[col].astype(str).str.strip()

    if "basin_name" in pairs.columns:
        pairs["basin_name"] = pairs["basin_name"].astype(str).str.strip()

    # Filter
    if network_filter is not None:
        pairs = pairs[pairs["network"].isin(list(network_filter))]
    if basin_filter is not None and "basin_name" in pairs.columns:
        pairs = pairs[pairs["basin_name"].isin(list(basin_filter))]

    # Drop obvious invalid rows
    pairs = pairs.replace({"": np.nan})
    pairs = pairs.dropna(subset=["network", "station", "gage_id"])

    # Add convenient keys
    pairs["seis_key"] = pairs["network"] + "." + pairs["station"]

    if choose_closest_gage:
        # Best-effort compute distance to each candidate gage.
        pairs["latitude"] = pd.to_numeric(pairs.get("latitude"), errors="coerce")
        pairs["longitude"] = pd.to_numeric(pairs.get("longitude"), errors="coerce")

        gage_ids = sorted(pairs["gage_id"].dropna().astype(str).unique().tolist())
        gage_meta: dict[str, dict[str, object]] = {}
        for gid in gage_ids:
            try:
                gage_meta[gid] = fetch_usgs_site_metadata(gid, data_dir=data_dir, use_cache=True)
            except Exception as e:
                # Keep going; distance will be NaN for this gage.
                gage_meta[gid] = {"site": gid, "latitude": None, "longitude": None, "name": None}
                print(f"  Warning: failed to fetch gage metadata for {gid}: {e}")

        def _dist_row_km(row: pd.Series) -> float:
            try:
                s_lat = float(row["latitude"])
                s_lon = float(row["longitude"])
            except Exception:
                return float("nan")

            gid = str(row["gage_id"])
            gm = gage_meta.get(gid) or {}
            g_lat = gm.get("latitude")
            g_lon = gm.get("longitude")
            if g_lat is None or g_lon is None:
                return float("nan")
            return _haversine_km(s_lat, s_lon, float(g_lat), float(g_lon))

        pairs["gage_distance_km"] = pairs.apply(_dist_row_km, axis=1)
        pairs["gage_latitude"] = pairs["gage_id"].astype(str).map(
            lambda gid: (gage_meta.get(gid) or {}).get("latitude")
        )
        pairs["gage_longitude"] = pairs["gage_id"].astype(str).map(
            lambda gid: (gage_meta.get(gid) or {}).get("longitude")
        )
        pairs["gage_name"] = pairs["gage_id"].astype(str).map(lambda gid: (gage_meta.get(gid) or {}).get("name"))

        # Choose closest gage for each station; if all NaN, keep the first row.
        def _pick_closest(group: pd.DataFrame) -> pd.DataFrame:
            if group["gage_distance_km"].notna().any():
                return group.loc[[group["gage_distance_km"].idxmin()]]
            return group.iloc[[0]]

        pairs = (
            pairs.groupby("seis_key", as_index=False, sort=False)
            .apply(_pick_closest)
            .reset_index(drop=True)
        )
    else:
        # De-duplicate to unique station keys (keep first gage per station)
        pairs = pairs.drop_duplicates(subset=["seis_key"], keep="first")

    if max_pairs is not None:
        pairs = pairs.head(int(max_pairs))

    return pairs.reset_index(drop=True)


def fetch_usgs_nwis_iv(
    site: str,
    start_utc: UTCDateTime,
    end_utc: UTCDateTime,
    data_dir: Path,
    parameter_codes: tuple[str, ...] = ("00060", "00065"),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch USGS NWIS instantaneous values and return a time-indexed DataFrame.

    Columns (when available):
    - discharge_cfs (00060)
    - gage_height_ft (00065)
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    site = str(site).strip()
    cache_file = data_dir / f"usgs_iv_{site}_{start_utc.date}_{end_utc.date}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["time_utc"], index_col="time_utc")
        df.index = pd.to_datetime(df.index, utc=True)
        return df.sort_index()

    base = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": site,
        "startDT": start_utc.datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endDT": end_utc.datetime.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parameterCd": ",".join(parameter_codes),
        "siteStatus": "all",
    }
    url = base + "?" + urlencode(params)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    js = r.json()

    series = js.get("value", {}).get("timeSeries", [])
    if not series:
        raise RuntimeError(f"No NWIS IV timeSeries for site={site}")

    out: pd.DataFrame | None = None
    for ts in series:
        variable = ts.get("variable", {})
        var_code = (variable.get("variableCode") or [{}])[0].get("value")
        values = (ts.get("values") or [{}])[0].get("value", [])
        if not values:
            continue

        times = [v.get("dateTime") for v in values]
        vals = [v.get("value") for v in values]
        t = pd.to_datetime(times, utc=True, errors="coerce")
        x = pd.to_numeric(vals, errors="coerce")
        s = pd.Series(x.values, index=pd.DatetimeIndex(t, name="time_utc")).dropna()
        if s.empty:
            continue

        if var_code == "00060":
            col = "discharge_cfs"
        elif var_code == "00065":
            col = "gage_height_ft"
        else:
            col = f"param_{var_code}"

        df1 = s.to_frame(name=col)
        out = df1 if out is None else out.join(df1, how="outer")

    if out is None or out.empty:
        raise RuntimeError(f"No parseable values for site={site}")

    out = out.sort_index()
    out["site"] = site
    out.to_csv(cache_file)
    return out


def download_streams(
    seis_requests: list[tuple[str, str]],
    start_utc: UTCDateTime,
    end_utc: UTCDateTime,
    client_name: str = "IRIS",
    location: str = "*",
    channel: str = "HH?,HN?,BH?,EH?",
    attach_response: bool = True,
) -> dict[str, Stream]:
    """Download waveforms for a list of (network, station) pairs."""
    from obspy.clients.fdsn import Client

    client = Client(client_name)
    streams: dict[str, Stream] = {}
    for net, sta in seis_requests:
        key = f"{net}.{sta}"
        try:
            st = client.get_waveforms(
                net,
                sta,
                location,
                channel,
                start_utc,
                end_utc,
                attach_response=attach_response,
            )
        except Exception as e:
            print(f"  Failed to download {key}: {e}")
            continue

        st.merge(method=1, fill_value="interpolate")
        streams[key] = st

    return streams


def fetch_usgs_event_times(
    start_utc: UTCDateTime,
    end_utc: UTCDateTime,
    latitude: float,
    longitude: float,
    min_magnitude: float = 3.5,
    maxradiuskm: float = 500,
) -> pd.DatetimeIndex:
    """Fetch USGS earthquake event origin times (UTC) for a region/time window."""
    base = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": start_utc.datetime.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_utc.datetime.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude": latitude,
        "longitude": longitude,
        "maxradiuskm": maxradiuskm,
        "minmagnitude": min_magnitude,
        "orderby": "time-asc",
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    feats = js.get("features", [])
    if not feats:
        return pd.DatetimeIndex([], tz="UTC", name="event_time_utc")

    times_ms = [f.get("properties", {}).get("time") for f in feats]
    times_ms = [t for t in times_ms if t is not None]
    t = pd.to_datetime(times_ms, unit="ms", utc=True)
    return pd.DatetimeIndex(t, name="event_time_utc")


def mask_times_near_events(
    times_utc: pd.DatetimeIndex,
    event_times_utc: pd.DatetimeIndex | None,
    buffer_seconds: float,
) -> np.ndarray:
    """Return boolean mask keeping only times farther than buffer from any event."""
    if event_times_utc is None or len(event_times_utc) == 0:
        return np.ones(len(times_utc), dtype=bool)

    ev = event_times_utc.view("int64")
    tt = times_utc.view("int64")
    buf_ns = int(buffer_seconds * 1e9)

    idx = np.searchsorted(ev, tt, side="left")
    keep = np.ones(len(tt), dtype=bool)

    left = idx - 1
    ok_left = left >= 0
    if np.any(ok_left):
        dt_left = np.abs(tt[ok_left] - ev[left[ok_left]])
        keep[ok_left] &= dt_left > buf_ns

    ok_right = idx < len(ev)
    if np.any(ok_right):
        dt_right = np.abs(tt[ok_right] - ev[idx[ok_right]])
        keep[ok_right] &= dt_right > buf_ns

    return keep


def quick_proxy(
    st: Stream,
    fmin: float = 5,
    fmax: float = 30,
    win_s: int = 600,
    step_s: int = 600,
    event_times_utc: pd.DatetimeIndex | None = None,
    event_buffer_s: float = 0,
    *,
    clip_impulsive_days: bool = False,
    sta_seconds: float = 1.0,
    lta_seconds: float = 20.0,
    trigger_on: float = 3.5,
    trigger_off: float = 1.0,
    clip_sigma: float = 2.0,
    clip_mode: str = "symmetric",
) -> pd.Series:
    """Compute a minimal band-limited RMS proxy from a single trace."""
    tr = None
    for t in st:
        if t.stats.channel.upper().endswith("Z"):
            tr = t
            break
    if tr is None:
        tr = st[0]

    tr = tr.copy()

    if clip_impulsive_days:
        tr = clip_trace_days_on_stalta_impulses(
            tr,
            sta_seconds=sta_seconds,
            lta_seconds=lta_seconds,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            clip_sigma=clip_sigma,
            clip_mode=clip_mode,
        )

    tr.detrend("demean")
    tr.taper(0.01)

    sr = tr.stats.sampling_rate
    x = tr.data.astype("float64")

    x = bandpass(x, fmin, fmax, df=sr, corners=4, zerophase=True)

    win = int(win_s * sr)
    step = int(step_s * sr)

    vals: list[float] = []
    times: list[pd.Timestamp] = []
    t0 = tr.stats.starttime.timestamp

    for i in range(0, len(x) - win + 1, step):
        seg = x[i : i + win]
        vals.append(float(np.sqrt(np.mean(seg**2))))
        times.append(pd.to_datetime(t0 + (i + win / 2) / sr, unit="s", utc=True))

    s = pd.Series(vals, index=pd.DatetimeIndex(times, name="time_utc"))

    if event_buffer_s and event_buffer_s > 0 and event_times_utc is not None:
        keep = mask_times_near_events(s.index, event_times_utc, event_buffer_s)
        s = s.loc[keep]

    return s


def stream_to_proxy_timeseries(
    st: Stream,
    start: UTCDateTime,
    end: UTCDateTime,
    fmin: float,
    fmax: float,
    win_seconds: int,
    step_seconds: int,
    output: str = "velocity",
    event_times_utc: pd.DatetimeIndex | None = None,
    event_buffer_s: float = 0,
    *,
    clip_impulsive_days: bool = False,
    sta_seconds: float = 1.0,
    lta_seconds: float = 20.0,
    trigger_on: float = 3.5,
    trigger_off: float = 1.0,
    clip_sigma: float = 2.0,
    clip_mode: str = "symmetric",
) -> pd.Series:
    """Notebook-friendly proxy wrapper.

    NOTE: this intentionally does not remove instrument response yet.
    """
    _ = (start, end, output)  # reserved for future use
    return quick_proxy(
        st,
        fmin=fmin,
        fmax=fmax,
        win_s=win_seconds,
        step_s=step_seconds,
        event_times_utc=event_times_utc,
        event_buffer_s=event_buffer_s,
        clip_impulsive_days=clip_impulsive_days,
        sta_seconds=sta_seconds,
        lta_seconds=lta_seconds,
        trigger_on=trigger_on,
        trigger_off=trigger_off,
        clip_sigma=clip_sigma,
        clip_mode=clip_mode,
    )


def clip_trace_days_on_stalta_impulses(
    tr,
    *,
    sta_seconds: float = 1.0,
    lta_seconds: float = 20.0,
    trigger_on: float = 3.5,
    trigger_off: float = 1.0,
    clip_sigma: float = 2.0,
    clip_mode: str = "symmetric",
):
    """Detect impulsive behavior with STA/LTA per UTC day, then clip that day's data.

    Rule: if any STA/LTA trigger occurs within a UTC day, clip *all* samples
    in that day to a threshold derived from the day's standard deviation.

    `clip_mode`:
      - "symmetric": clip to ±(clip_sigma * std)
      - "upper": clip only positive excursions to +(clip_sigma * std)
      - "abs": clip based on absolute value to (clip_sigma * std)
    """
    tr2 = tr.copy()
    sr = float(tr2.stats.sampling_rate)
    if sr <= 0:
        return tr2

    x = tr2.data.astype("float64", copy=True)
    n = len(x)
    if n == 0:
        return tr2

    nsta = max(1, int(round(sta_seconds * sr)))
    nlta = max(nsta + 1, int(round(lta_seconds * sr)))

    start = tr2.stats.starttime
    end = tr2.stats.endtime

    # Build UTC-day boundaries covering the trace.
    day0 = UTCDateTime(start.year, start.month, start.day)
    day_end = UTCDateTime(end.year, end.month, end.day) + 24 * 3600

    t = day0
    while t < day_end:
        t0 = t
        t1 = min(t + 24 * 3600, end)

        i0 = int(max(0, np.floor((t0.timestamp - start.timestamp) * sr)))
        i1 = int(min(n, np.ceil((t1.timestamp - start.timestamp) * sr)))
        if i1 - i0 < nlta + 1:
            t += 24 * 3600
            continue

        seg = x[i0:i1]
        # STA/LTA is sensitive to offsets; remove median to stabilize.
        seg0 = seg - np.nanmedian(seg)

        try:
            cft = classic_sta_lta(seg0, nsta, nlta)
            on_off = trigger_onset(cft, trigger_on, trigger_off)
            has_impulse = len(on_off) > 0
        except Exception:
            has_impulse = False

        if has_impulse:
            std = float(np.nanstd(seg0))
            if std > 0 and np.isfinite(std):
                thr = clip_sigma * std
                mode = clip_mode.lower().strip()
                if mode == "upper":
                    seg = np.minimum(seg, thr)
                elif mode == "abs":
                    seg = np.clip(seg, -thr, thr)
                else:  # "symmetric"
                    seg = np.clip(seg, -thr, thr)
                x[i0:i1] = seg

        t += 24 * 3600

    tr2.data = x
    return tr2


def plot_proxy_and_gauge(proxy: pd.Series, gauge_df: pd.DataFrame, title: str = "") -> None:
    import matplotlib.pyplot as plt

    g = gauge_df.copy().sort_index()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(proxy.index, proxy.values, linewidth=1, label="Seismic proxy (band RMS)")
    ax1.set_ylabel("Proxy (RMS)")
    ax1.set_title(title)

    stage_col = (
        "gage_height_ft"
        if "gage_height_ft" in g.columns
        else ("discharge_cfs" if "discharge_cfs" in g.columns else None)
    )
    if stage_col is None:
        raise KeyError("Gauge dataframe missing gage_height_ft and discharge_cfs")

    ax2 = ax1.twinx()
    ax2.plot(g.index, g[stage_col].values, linestyle="--", linewidth=1, label=stage_col)
    ax2.set_ylabel(stage_col)

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def hysteresis_plot(
    proxy: pd.Series,
    gauge_df: pd.DataFrame,
    title: str = "",
    tolerance: str = "10min",
) -> None:
    import matplotlib.pyplot as plt

    g = gauge_df.copy().sort_index()
    stage_col = (
        "gage_height_ft"
        if "gage_height_ft" in g.columns
        else ("discharge_cfs" if "discharge_cfs" in g.columns else None)
    )
    if stage_col is None:
        raise KeyError("Gauge dataframe missing gage_height_ft and discharge_cfs")

    df = pd.DataFrame({"proxy": proxy}).sort_index()
    df[stage_col] = g[stage_col].reindex(
        df.index, method="nearest", tolerance=pd.Timedelta(tolerance)
    )
    df = df.dropna()

    plt.figure(figsize=(5.2, 5.0))
    plt.scatter(df[stage_col], df["proxy"], s=6)
    plt.xlabel(stage_col)
    plt.ylabel("Seismic proxy (RMS)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def band_sweep(
    st: Stream,
    gauge_df: pd.DataFrame,
    bands: Iterable[tuple[float, float]],
    station_name: str = "",
    start: UTCDateTime | None = None,
    end: UTCDateTime | None = None,
    win_seconds: int = 600,
    step_seconds: int = 300,
    output: str = "velocity",
    event_times_utc: pd.DatetimeIndex | None = None,
    event_buffer_s: float = 0,
) -> None:
    if start is None or end is None:
        raise ValueError("band_sweep requires start and end")

    for (fmin, fmax) in bands:
        proxy = stream_to_proxy_timeseries(
            st,
            start,
            end,
            fmin=fmin,
            fmax=fmax,
            win_seconds=win_seconds,
            step_seconds=step_seconds,
            output=output,
            event_times_utc=event_times_utc,
            event_buffer_s=event_buffer_s,
        )
        plot_proxy_and_gauge(proxy, gauge_df, title=f"{station_name}: band {fmin}-{fmax} Hz")
