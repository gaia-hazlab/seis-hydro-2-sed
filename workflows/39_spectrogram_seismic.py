#!/usr/bin/env python3
r"""Flood seismic spectrogram at the Nisqually source station UW.LON.

A single PSD snapshot (fig11_spectra.py) shows *where* in frequency the river signal
sits; a **spectrogram** shows how that band structure evolves through the flood — the
view Schmandt et al. (2013) used in the Grand Canyon to attribute different parts of
the spectrum to different fluvial processes. We compute a continuous spectrogram of
the vertical ground velocity at the 100-sps Nisqually source station UW.LON across the
December-2025 AR2 pulse (Dec 7–14) and overlay the co-located Nisqually-nr-National
gage discharge (USGS 12082500).

The figure makes two points. (1) Seismic power in the **turbulence band (~1–20 Hz)**
rises and falls with discharge, and a **higher-frequency edge (~30–50 Hz)** brightens
near the peak — the tentative lower bedload edge (cf. fig13_bedload_ch.py); the
*canonical* clean-gravel bedload band (30–80 Hz) is only partly sampled (50 Hz Nyquist
at 100 sps), the same coverage limit discussed in the main text. (2) **Seismic data
alone are spectrally degenerate**: turbulent water flow over the bed and bedload
impacts on the bed radiate into overlapping bands, so a bright band does not by itself
say whether the *source* is at the bed or at the air–water interface.

That degeneracy is what colocated **infrasound** resolves, and — contrary to an
earlier draft of this study — the regional network *does* carry it: the USGS Cascades
Volcano Observatory (network CC) operates microbarometer (BDF) infrasound arrays
colocated with broadband seismometers on the upper-Puyallup corridor, including at the
study's on-channel anchor station CC.PR03 (right on USGS gage 12092000). The infrasound
power spectrogram is built in workflow 40 (fig31) and the colocated seismic–infrasound
coherence discriminant — the actual turbulence-vs-bedload partition — in workflow 41
(fig32). This figure is the *seismic-only* half of that pair: the band structure the
infrasound is then used to attribute.

Outputs paper/figures/fig30_spectrogram_seismic.png (+ cached array for offline
rebuild).

Usage: pixi run python workflows/39_spectrogram_seismic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from obspy import UTCDateTime, read as obspy_read
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "fdsn_cache"
RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"

NET, STA, COMP = "UW", "LON", "Z"          # Nisqually source, 100 sps -> 50 Hz Nyquist
GAGE = "12082500"                          # Nisqually nr National (co-located)
DAYS = ["20251207", "20251208", "20251209", "20251210",
        "20251211", "20251212", "20251213"]
T0, T1 = UTCDateTime("2025-12-07T00:00:00"), UTCDateTime("2025-12-14T00:00:00")
TURB = (1.0, 20.0)        # turbulence-dominated band
BEDL = (30.0, 50.0)       # tentative lower bedload edge (sampled part of 30–80 Hz)
CACHE_NPZ = RESULTS / "spectrogram_UW_LON.npz"


def load_trace():
    """Concatenated vertical-velocity trace over the window (raw counts)."""
    traces = []
    for day in DAYS:
        files = sorted(CACHE.glob(f"{NET}.{STA}.*.{day}.*.mseed"))
        if not files:
            print(f"   missing {NET}.{STA} {day}")
            continue
        st = obspy_read(str(files[-1])).select(component=COMP)
        if not st:
            continue
        ds = UTCDateTime(f"{day[:4]}-{day[4:6]}-{day[6:]}T00:00:00")
        st.trim(ds, ds + 86400, nearest_sample=False)
        traces.append(st[0])
    if not traces:
        return None
    from obspy import Stream
    st = Stream(traces)
    st.merge(method=1, fill_value=0)
    tr = st[0]
    tr.trim(T0, T1, pad=True, fill_value=0)
    tr.detrend("demean")
    return tr


def compute_and_cache():
    """Compute the spectrogram from cached waveforms and write a COMPACT
    (log-frequency, time-decimated) array for offline rebuild. Returns (f, tcol, SdB)."""
    tr = load_trace()
    if tr is None:
        return None
    sr = float(tr.stats.sampling_rate)
    nper = int(sr * 60)                         # 60-s windows
    f, t, Sxx = spectrogram(tr.data.astype(float), fs=sr,
                            nperseg=nper, noverlap=nper // 2, scaling="density")
    SdB = 10 * np.log10(Sxx + 1e-20)
    tcol = np.array(mdates.date2num([(tr.stats.starttime + float(s)).datetime for s in t]))
    # downsample to ~500 log-spaced frequencies (0.4–Nyquist) × ~1800 time columns,
    # float16, so the cached array is a few MB and rebuilds offline (raw mseed is not
    # committed). This resolution is ample for the rendered figure.
    keep_f = np.unique(np.searchsorted(f, np.geomspace(0.4, sr / 2, 500)).clip(1, len(f) - 1))
    f, SdB = f[keep_f], SdB[keep_f]
    step = max(1, SdB.shape[1] // 1800)
    SdB, tcol = SdB[:, ::step], tcol[::step]
    np.savez_compressed(CACHE_NPZ, f=f.astype("float32"), tcol=tcol.astype("float64"),
                        SdB=SdB.astype("float16"), sr=np.array([sr]))
    return f, tcol, SdB.astype("float32"), sr


def main() -> int:
    paper_style()
    res = compute_and_cache()
    if res is None:
        if not CACHE_NPZ.exists():
            print("no waveforms in cache and no cached spectrogram — cannot build")
            return 1
        z = np.load(CACHE_NPZ)                  # offline rebuild from committed cache
        f, tcol = z["f"], z["tcol"]
        SdB, sr = z["SdB"].astype("float32"), float(z["sr"][0])
        print(f"rebuilt from cache {CACHE_NPZ.name}")
    else:
        f, tcol, SdB, sr = res

    # discharge overlay (co-located gage)
    d = pd.read_csv(ROOT / "notebooks" / "data" / f"usgs_iv_{GAGE}_2025-12-01_2026-01-01.csv")
    dt = pd.to_datetime(d["time_utc"], utc=True)
    dq = pd.to_numeric(d["discharge_cfs"], errors="coerce") * 0.0283168
    m = (dt >= T0.datetime.replace(tzinfo=dt.dt.tz)) & (dt <= T1.datetime.replace(tzinfo=dt.dt.tz))

    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    fsel = f >= 0.5
    vmin, vmax = np.percentile(SdB[fsel], [5, 99])
    pcm = ax.pcolormesh(tcol, f[fsel], SdB[fsel], cmap="magma",
                        norm=Normalize(vmin, vmax), shading="auto", rasterized=True)
    ax.set_yscale("log")
    ax.set_ylim(0.5, sr / 2)
    for (lo, hi), lab, col in [(TURB, "turbulence ~1–20 Hz", "#67c7ff"),
                               (BEDL, "bedload edge ~30–50 Hz", "#ffd27f")]:
        ax.axhline(lo, color=col, lw=1.0, ls="--", alpha=0.8)
        ax.axhline(hi, color=col, lw=1.0, ls="--", alpha=0.8)
        ax.text(tcol[2], np.sqrt(lo * hi), " " + lab, color=col, fontsize=9,
                va="center", ha="left", fontweight="bold")
    ax.axhline(sr / 2, color="white", lw=0.8, ls=":", alpha=0.7)
    ax.text(tcol[-2], sr / 2, "Nyquist 50 Hz (canonical bedload 30–80 Hz only partly sampled) ",
            color="white", fontsize=7.5, va="bottom", ha="right", alpha=0.9)

    axq = ax.twinx()
    axq.plot(mdates.date2num(dt[m].dt.tz_convert("UTC").dt.tz_localize(None)), dq[m],
             color="#39ff14", lw=1.8, alpha=0.9)
    axq.set_ylabel("discharge $Q$  (m³ s⁻¹)", color="#1a9e1a")
    axq.tick_params(axis="y", colors="#1a9e1a")
    axq.set_ylim(0, dq[m].max() * 1.05)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(tcol[0], tcol[-1])
    ax.set_xlabel("2025 (UTC)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title(f"{NET}.{STA} (Nisqually source, {int(sr)} sps) vertical-velocity spectrogram "
                 "through AR2 — seismic alone cannot place the source at bed vs air–water interface\n"
                 "(infrasound discriminant: fig31 power, fig32 colocated coherence at CC.PR03)",
                 fontsize=10.5, loc="left")
    fig.subplots_adjust(left=0.055, right=0.85, top=0.93, bottom=0.12)
    cax = fig.add_axes((0.935, 0.12, 0.013, 0.81))
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label("relative power (dB)")
    out_png = FIGDIR / "fig30_spectrogram_seismic.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png} + {CACHE_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
