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
from obspy.core.trace import Trace
from obspy import read as obspy_read
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


def compute_proxy_from_fdsn(
    net: str,
    sta: str,
    start_utc: UTCDateTime,
    end_utc: UTCDateTime,
    *,
    fmin: float,
    fmax: float,
    win_seconds: int,
    step_seconds: int,
    output: str = "velocity",
    method: str = "bandpower",
    remove_response: bool = True,
    combine: str = "rss",
    components: tuple[str, ...] = ("Z", "N", "E"),
    event_times_utc: pd.DatetimeIndex | None = None,
    event_buffer_s: float = 0,
    clip_impulsive_days: bool = False,
    sta_seconds: float = 1.0,
    lta_seconds: float = 20.0,
    trigger_on: float = 3.5,
    trigger_off: float = 1.0,
    clip_sigma: float = 2.0,
    clip_mode: str = "symmetric",
    # Download parameters
    client_name: str = "IRIS",
    location: str = "*",
    channel: str = "HH?,HN?,BH?,EH?",
    attach_response: bool = True,
    pad_seconds: float | None = None,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    # Proxy despiking (post-processing)
    despike_proxy: bool = False,
    despike_window: str | pd.Timedelta = "6H",
    despike_z: float = 8.0,
    despike_min_periods: int = 10,
    despike_fill: str = "none",
) -> pd.Series:
    """Compute a proxy by downloading waveforms in whole UTC-day blocks from FDSN.

    Rationale: multi-week, multi-channel waveform downloads can be huge and
    slow/unreliable. Downloading one UTC day at a time tends to be robust
    while still keeping request counts low.
    """
    from obspy.clients.fdsn import Client

    net = str(net).strip()
    sta = str(sta).strip()

    if pad_seconds is None:
        pad_seconds = float(win_seconds)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Explicit timeout helps avoid hanging reads on large responses.
    client = Client(client_name, timeout=120)

    out_parts: list[pd.Series] = []
    t0 = start_utc

    while t0 < end_utc:
        # Advance to next UTC midnight boundary (whole-day blocks)
        next_midnight = UTCDateTime(t0.year, t0.month, t0.day) + 24 * 3600
        t1 = min(next_midnight, end_utc)
        dl0 = t0 - float(pad_seconds)
        dl1 = t1 + float(pad_seconds)

        cache_file: Path | None = None
        if cache_dir is not None:
            day = f"{t0.year:04d}{t0.month:02d}{t0.day:02d}"
            safe_loc = (location or "").replace("*", "STAR")
            safe_chan = (channel or "").replace("*", "STAR").replace("?", "Q")
            pad_i = int(round(float(pad_seconds)))
            cache_file = (
                cache_dir
                / f"{net}.{sta}.{safe_loc}.{safe_chan}.{day}.pad{pad_i}s.mseed"
            )

        st: Stream | None = None
        if (
            use_cache
            and cache_file is not None
            and cache_file.exists()
            and cache_file.stat().st_size > 0
        ):
            try:
                print(f"  Using cached MSEED {cache_file.name}")
                st = obspy_read(str(cache_file))
            except Exception as e:
                print(f"  Cache read failed ({cache_file}): {e}; re-downloading")
                try:
                    cache_file.unlink(missing_ok=True)
                except Exception:
                    pass
                st = None

        if st is None:
            try:
                print(f"  Downloading {net}.{sta} {dl0}–{dl1} ({channel}, loc={location})")
                st = client.get_waveforms(
                    net,
                    sta,
                    location,
                    channel,
                    dl0,
                    dl1,
                    attach_response=attach_response,
                )
                if cache_file is not None:
                    try:
                        st.write(str(cache_file), format="MSEED")
                    except Exception as e:
                        print(f"  Warning: failed to write cache {cache_file}: {e}")
            except Exception as e:
                print(f"  Failed day {net}.{sta} {dl0}–{dl1}: {e}")
                t0 = t1
                continue

        try:
            st.merge(method=1, fill_value="interpolate")
            st.trim(dl0, dl1)

            s = stream_to_proxy_timeseries(
                st,
                start=t0,
                end=t1,
                fmin=fmin,
                fmax=fmax,
                win_seconds=win_seconds,
                step_seconds=step_seconds,
                output=output,
                event_times_utc=event_times_utc,
                event_buffer_s=event_buffer_s,
                method=method,
                remove_response=remove_response,
                combine=combine,
                components=components,
                clip_impulsive_days=clip_impulsive_days,
                sta_seconds=sta_seconds,
                lta_seconds=lta_seconds,
                trigger_on=trigger_on,
                trigger_off=trigger_off,
                clip_sigma=clip_sigma,
                clip_mode=clip_mode,
                despike_proxy=despike_proxy,
                despike_window=despike_window,
                despike_z=despike_z,
                despike_min_periods=despike_min_periods,
                despike_fill=despike_fill,
            )

            # Keep only proxy timestamps within the chunk (avoid edge duplicates)
            t0_ts = pd.to_datetime(t0.datetime, utc=True)
            t1_ts = pd.to_datetime(t1.datetime, utc=True)
            s = s.loc[(s.index >= t0_ts) & (s.index < t1_ts)]
            if not s.empty:
                out_parts.append(s)
        finally:
            # help GC in notebooks
            del st

        t0 = t1

    if not out_parts:
        return pd.Series([], dtype="float64", name="proxy", index=pd.DatetimeIndex([], tz="UTC"))

    out = pd.concat(out_parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


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
    r = requests.get(base, params=params, timeout=20)
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


def despike_timeseries(
    s: pd.Series,
    *,
    window: str | pd.Timedelta = "6H",
    z: float = 8.0,
    min_periods: int = 10,
    center: bool = True,
    fill: str = "none",
) -> pd.Series:
    """Remove isolated outliers from a time-indexed series.

    Uses a rolling median + MAD (median absolute deviation) to flag points where
    |x - median| > z * (1.4826 * MAD). Flagged points are set to NaN.

    Parameters
    - window: rolling window (e.g., "2H", "6H").
    - z: robust z-score threshold.
    - fill: "none" (default) leaves NaNs; "interpolate" fills with time interpolation.
    """
    if s is None:
        raise ValueError("s is None")
    if len(s) == 0:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("despike_timeseries expects a DatetimeIndex")

    x = s.astype("float64").sort_index()
    w = pd.to_timedelta(str(window).strip().lower()) if isinstance(window, str) else window

    roll = x.rolling(window=w, center=center, min_periods=int(min_periods))
    med = roll.median()

    def _mad(arr: np.ndarray) -> float:
        m = float(np.nanmedian(arr))
        return float(np.nanmedian(np.abs(arr - m)))

    mad = roll.apply(_mad, raw=True)
    robust_sigma = 1.4826 * mad
    with np.errstate(invalid="ignore", divide="ignore"):
        is_outlier = (np.abs(x - med) > float(z) * robust_sigma) & (robust_sigma > 0)

    y = x.mask(is_outlier)

    if str(fill).lower().strip() == "interpolate":
        # Keep it simple; this is mainly for single-sample spikes.
        y = y.interpolate(method="time", limit_direction="both")

    y.name = s.name
    return y


def quick_proxy(
    st: Stream,
    fmin: float = 5,
    fmax: float = 30,
    win_s: int = 600,
    step_s: int = 600,
    event_times_utc: pd.DatetimeIndex | None = None,
    event_buffer_s: float = 0,
    *,
    method: str = "rms",
    remove_response: bool = True,
    output: str = "velocity",
    pre_filt: tuple[float, float, float, float] | None = None,
    water_level: float = 60.0,
    clip_impulsive_days: bool = False,
    sta_seconds: float = 1.0,
    lta_seconds: float = 20.0,
    trigger_on: float = 3.5,
    trigger_off: float = 1.0,
    clip_sigma: float = 2.0,
    clip_mode: str = "symmetric",
    despike_proxy: bool = False,
    despike_window: str | pd.Timedelta = "6H",
    despike_z: float = 8.0,
    despike_min_periods: int = 10,
    despike_fill: str = "none",
) -> pd.Series:
    """Compute a proxy from a single preferred trace.

    method:
      - "rms": bandpass then windowed RMS
      - "bandpower": Welch PSD per window integrated over [fmin, fmax]
    """
    tr = _select_preferred_trace(st)
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

    if remove_response:
        tr = _remove_instrument_response(
            tr,
            fmin=fmin,
            fmax=fmax,
            output=output,
            pre_filt=pre_filt,
            water_level=water_level,
        )

    sr = tr.stats.sampling_rate
    x = tr.data.astype("float64")

    if method.lower().strip() == "bandpower":
        s = _bandpower_timeseries(
            x,
            starttime=tr.stats.starttime,
            sr=float(sr),
            fmin=float(fmin),
            fmax=float(fmax),
            win_s=int(win_s),
            step_s=int(step_s),
        )
    else:
        # RMS proxy
        nyq = 0.5 * float(sr)
        fmax_eff = min(float(fmax), 0.99 * nyq)
        if fmax_eff <= float(fmin):
            raise ValueError(f"Invalid band for sr={sr}: fmin={fmin}, fmax={fmax} (nyq={nyq})")
        x = bandpass(x, float(fmin), fmax_eff, df=float(sr), corners=4, zerophase=True)

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

    if despike_proxy:
        s = despike_timeseries(
            s,
            window=despike_window,
            z=float(despike_z),
            min_periods=int(despike_min_periods),
            fill=despike_fill,
        )

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
    method: str = "rms",
    remove_response: bool = True,
    pre_filt: tuple[float, float, float, float] | None = None,
    water_level: float = 60.0,
    combine: str = "z",
    components: tuple[str, ...] = ("Z", "N", "E"),
    clip_impulsive_days: bool = False,
    sta_seconds: float = 1.0,
    lta_seconds: float = 20.0,
    trigger_on: float = 3.5,
    trigger_off: float = 1.0,
    clip_sigma: float = 2.0,
    clip_mode: str = "symmetric",
    despike_proxy: bool = False,
    despike_window: str | pd.Timedelta = "6H",
    despike_z: float = 8.0,
    despike_min_periods: int = 10,
    despike_fill: str = "none",
) -> pd.Series:
    """Notebook-friendly proxy wrapper.
    """
    _ = (start, end)  # reserved for future use
    if combine.lower().strip() == "rss":
        return stream_to_proxy_timeseries_rss(
            st,
            fmin=fmin,
            fmax=fmax,
            win_seconds=win_seconds,
            step_seconds=step_seconds,
            output=output,
            event_times_utc=event_times_utc,
            event_buffer_s=event_buffer_s,
            method=method,
            remove_response=remove_response,
            pre_filt=pre_filt,
            water_level=water_level,
            components=components,
            clip_impulsive_days=clip_impulsive_days,
            sta_seconds=sta_seconds,
            lta_seconds=lta_seconds,
            trigger_on=trigger_on,
            trigger_off=trigger_off,
            clip_sigma=clip_sigma,
            clip_mode=clip_mode,
            despike_proxy=despike_proxy,
            despike_window=despike_window,
            despike_z=despike_z,
            despike_min_periods=despike_min_periods,
            despike_fill=despike_fill,
        )

    return quick_proxy(
        st,
        fmin=fmin,
        fmax=fmax,
        win_s=win_seconds,
        step_s=step_seconds,
        event_times_utc=event_times_utc,
        event_buffer_s=event_buffer_s,
        method=method,
        remove_response=remove_response,
        output=output,
        pre_filt=pre_filt,
        water_level=water_level,
        clip_impulsive_days=clip_impulsive_days,
        sta_seconds=sta_seconds,
        lta_seconds=lta_seconds,
        trigger_on=trigger_on,
        trigger_off=trigger_off,
        clip_sigma=clip_sigma,
        clip_mode=clip_mode,
        despike_proxy=despike_proxy,
        despike_window=despike_window,
        despike_z=despike_z,
        despike_min_periods=despike_min_periods,
        despike_fill=despike_fill,
    )


def _select_preferred_trace(st: Stream) -> Trace:
    """Pick a single preferred trace (Z if available, else first)."""
    for t in st:
        try:
            if str(t.stats.channel).upper().endswith("Z"):
                return t
        except Exception:
            continue
    return st[0]


def _select_trace_by_component(
    st: Stream,
    component: str,
    *,
    prefer_prefixes: tuple[str, ...] = ("HH", "HN", "EH", "BH", "BN"),
) -> Trace | None:
    """Select a single trace for a component (Z/N/E), preferring higher-rate data."""
    comp = component.upper().strip()
    candidates: list[Trace] = []
    for tr in st:
        ch = str(getattr(tr.stats, "channel", "")).upper()
        if ch.endswith(comp):
            candidates.append(tr)

    if not candidates:
        return None

    def _rank(tr: Trace) -> tuple[int, float]:
        ch = str(getattr(tr.stats, "channel", "")).upper()
        prefix = ch[:2]
        try:
            p = prefer_prefixes.index(prefix)
        except ValueError:
            p = len(prefer_prefixes) + 1
        sr = float(getattr(tr.stats, "sampling_rate", 0.0) or 0.0)
        # smaller tuple sorts first: best prefix, then highest sampling rate
        return (p, -sr)

    return sorted(candidates, key=_rank)[0]


def stream_to_proxy_timeseries_rss(
    st: Stream,
    *,
    fmin: float,
    fmax: float,
    win_seconds: int,
    step_seconds: int,
    output: str = "velocity",
    event_times_utc: pd.DatetimeIndex | None = None,
    event_buffer_s: float = 0,
    method: str = "bandpower",
    remove_response: bool = True,
    pre_filt: tuple[float, float, float, float] | None = None,
    water_level: float = 60.0,
    components: tuple[str, ...] = ("Z", "N", "E"),
    clip_impulsive_days: bool = False,
    sta_seconds: float = 1.0,
    lta_seconds: float = 20.0,
    trigger_on: float = 3.5,
    trigger_off: float = 1.0,
    clip_sigma: float = 2.0,
    clip_mode: str = "symmetric",
    despike_proxy: bool = False,
    despike_window: str | pd.Timedelta = "6H",
    despike_z: float = 8.0,
    despike_min_periods: int = 10,
    despike_fill: str = "none",
) -> pd.Series:
    """Compute proxy per component then combine as RSS: sqrt(sum(proxy_i^2))."""

    def _proxy_from_trace(tr: Trace) -> pd.Series:
        tr2 = tr.copy()

        if clip_impulsive_days:
            tr2 = clip_trace_days_on_stalta_impulses(
                tr2,
                sta_seconds=sta_seconds,
                lta_seconds=lta_seconds,
                trigger_on=trigger_on,
                trigger_off=trigger_off,
                clip_sigma=clip_sigma,
                clip_mode=clip_mode,
            )

        tr2.detrend("demean")
        tr2.taper(0.01)

        if remove_response:
            tr2 = _remove_instrument_response(
                tr2,
                fmin=fmin,
                fmax=fmax,
                output=output,
                pre_filt=pre_filt,
                water_level=water_level,
            )

        sr = float(tr2.stats.sampling_rate)
        x = tr2.data.astype("float64")
        m = method.lower().strip()

        if m == "bandpower":
            return _bandpower_timeseries(
                x,
                starttime=tr2.stats.starttime,
                sr=sr,
                fmin=float(fmin),
                fmax=float(fmax),
                win_s=int(win_seconds),
                step_s=int(step_seconds),
            )

        # RMS proxy
        nyq = 0.5 * float(sr)
        fmax_eff = min(float(fmax), 0.99 * nyq)
        if fmax_eff <= float(fmin):
            raise ValueError(f"Invalid band for sr={sr}: fmin={fmin}, fmax={fmax} (nyq={nyq})")
        y = bandpass(x, float(fmin), fmax_eff, df=float(sr), corners=4, zerophase=True)

        win = int(win_seconds * sr)
        step = int(step_seconds * sr)
        vals: list[float] = []
        times: list[pd.Timestamp] = []
        t0 = tr2.stats.starttime.timestamp
        for i in range(0, len(y) - win + 1, step):
            seg = y[i : i + win]
            vals.append(float(np.sqrt(np.mean(seg**2))))
            times.append(pd.to_datetime(t0 + (i + win / 2) / sr, unit="s", utc=True))
        return pd.Series(vals, index=pd.DatetimeIndex(times, name="time_utc")).dropna()

    series_by_comp: list[pd.Series] = []
    for comp in components:
        tr = _select_trace_by_component(st, comp)
        if tr is None:
            continue
        s = _proxy_from_trace(tr).rename(comp)
        if not s.empty:
            series_by_comp.append(s)

    if not series_by_comp:
        raise RuntimeError("No usable component traces for RSS proxy")

    df = pd.concat(series_by_comp, axis=1).dropna(how="any")
    rss = np.sqrt(np.sum(np.square(df.values), axis=1))
    out = pd.Series(rss, index=df.index, name="proxy_rss")

    if event_buffer_s and event_buffer_s > 0 and event_times_utc is not None:
        keep = mask_times_near_events(out.index, event_times_utc, event_buffer_s)
        out = out.loc[keep]

    if despike_proxy:
        out = despike_timeseries(
            out,
            window=despike_window,
            z=float(despike_z),
            min_periods=int(despike_min_periods),
            fill=despike_fill,
        )

    return out


def estimate_constant_lag_seconds(
    proxy: pd.Series,
    gauge_df: pd.DataFrame,
    *,
    gauge_col: str = "discharge_cfs",
    max_lag_hours: float = 24.0,
    step_minutes: int = 10,
    resample_minutes: int | None = None,
    transform: str = "log10",
    min_pairs: int = 50,
) -> tuple[float, pd.DataFrame]:
    """Scan constant lags and return the best lag (seconds) + a results table.

    Convention: choose tau maximizing corr(proxy(t), gauge(t - tau)).
    Implemented by shifting gauge timestamps forward by +tau before alignment.
    """
    if gauge_col not in gauge_df.columns:
        raise KeyError(f"Gauge dataframe missing {gauge_col}")

    p = proxy.dropna().sort_index()
    g = gauge_df[gauge_col].dropna().sort_index()
    if p.empty or g.empty:
        raise ValueError("Empty proxy or gauge series")

    if resample_minutes is None:
        resample_minutes = step_minutes
    freq = f"{int(resample_minutes)}min"
    p = p.resample(freq).median().dropna()
    g = g.resample(freq).median().dropna()

    if transform.lower().strip() == "log10":
        tiny = np.finfo(float).tiny
        p = np.log10(p.clip(lower=tiny))
        g = np.log10(g.clip(lower=tiny))

    step_s = int(step_minutes * 60)
    max_s = int(max_lag_hours * 3600)
    lags = np.arange(-max_s, max_s + step_s, step_s, dtype=int)

    best_tau = 0
    best_corr = -np.inf
    rows: list[dict[str, float]] = []

    for tau in lags:
        gs = g.copy()
        gs.index = gs.index + pd.Timedelta(seconds=int(tau))
        joined = pd.concat([p.rename("proxy"), gs.rename("gauge")], axis=1).dropna()
        n = int(len(joined))
        if n < int(min_pairs):
            rows.append({"lag_seconds": float(tau), "corr": float("nan"), "n": float(n)})
            continue
        corr = float(joined["proxy"].corr(joined["gauge"]))
        rows.append({"lag_seconds": float(tau), "corr": corr, "n": float(n)})
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_tau = int(tau)

    results = pd.DataFrame(rows)
    return float(best_tau), results


def _remove_instrument_response(
    tr: Trace,
    *,
    fmin: float,
    fmax: float,
    output: str = "velocity",
    pre_filt: tuple[float, float, float, float] | None = None,
    water_level: float = 60.0,
) -> Trace:
    """Remove instrument response to velocity or acceleration.

    If response removal fails (missing metadata), returns the original trace.
    """
    tr2 = tr.copy()
    out = output.lower().strip()
    out_code = "VEL" if out.startswith("vel") else "ACC"

    sr = float(tr2.stats.sampling_rate)
    nyq = 0.5 * sr if sr > 0 else None

    if pre_filt is None:
        # A conservative pre-filter based on the band of interest.
        f1 = max(0.001, 0.5 * float(fmin))
        f2 = max(f1 * 1.5, 0.8 * float(fmin))
        f3 = max(f2 * 1.2, 1.2 * float(fmax))
        f4 = max(f3 * 1.2, 1.5 * float(fmax))
        if nyq is not None:
            f4 = min(f4, 0.99 * nyq)
            f3 = min(f3, 0.95 * f4)
        pre_filt = (f1, f2, f3, f4)

    try:
        tr2.remove_response(output=out_code, pre_filt=pre_filt, water_level=water_level)
        return tr2
    except Exception:
        return tr


def _bandpower_timeseries(
    x: np.ndarray,
    *,
    starttime: UTCDateTime,
    sr: float,
    fmin: float,
    fmax: float,
    win_s: int,
    step_s: int,
) -> pd.Series:
    """Welch PSD per window, integrated band power."""
    try:
        from scipy.signal import welch
    except Exception as e:  # pragma: no cover
        raise ImportError("scipy is required for method='bandpower'") from e

    sr = float(sr)
    if sr <= 0:
        raise ValueError("Invalid sampling rate")

    nyq = 0.5 * sr
    fmax_eff = min(float(fmax), 0.99 * nyq)
    if fmax_eff <= float(fmin):
        raise ValueError(f"Invalid band for sr={sr}: fmin={fmin}, fmax={fmax} (nyq={nyq})")

    win = int(win_s * sr)
    step = int(step_s * sr)
    if win < 8 or step < 1:
        raise ValueError("Window/step too small")

    vals: list[float] = []
    times: list[pd.Timestamp] = []
    t0 = starttime.timestamp

    for i in range(0, len(x) - win + 1, step):
        seg = x[i : i + win]
        # Welch: use a Hann window; nperseg capped for stability
        nperseg = min(win, int(10 * sr))  # ~10 s segments inside Welch
        f, pxx = welch(seg, fs=sr, nperseg=nperseg, detrend="constant", scaling="density")
        mask = (f >= float(fmin)) & (f <= fmax_eff)
        if not np.any(mask):
            vals.append(float("nan"))
        else:
            vals.append(float(np.trapz(pxx[mask], f[mask])))
        times.append(pd.to_datetime(t0 + (i + win / 2) / sr, unit="s", utc=True))

    s = pd.Series(vals, index=pd.DatetimeIndex(times, name="time_utc")).dropna()
    return s


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
    ax1.plot(proxy.index, proxy.values, linewidth=1, label="Seismic proxy")
    ax1.set_ylabel("Proxy")
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
    plt.ylabel("Seismic proxy")
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
    method: str = "rms",
    remove_response: bool = True,
    combine: str = "z",
    components: tuple[str, ...] = ("Z", "N", "E"),
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
            method=method,
            remove_response=remove_response,
            combine=combine,
            components=components,
        )
        plot_proxy_and_gauge(proxy, gauge_df, title=f"{station_name}: band {fmin}-{fmax} Hz")
