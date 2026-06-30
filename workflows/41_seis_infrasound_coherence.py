#!/usr/bin/env python3
r"""Colocated seismic–infrasound coherence at the upper-Puyallup arrays — the
turbulence-vs-bedload discriminant the seismic spectrogram (fig30) can only gesture at.

The seismic record alone is spectrally degenerate: turbulent flow over the bed and
bedload impacts on the bed radiate into overlapping bands, so a bright seismic band
cannot say whether the source sits at the **bed/in-water** or at the **air–water
interface** (Schmandt et al. 2013). The physical discriminant is colocated
**infrasound**: free-surface turbulence at the air–water interface (breaking waves,
hydraulic jumps, splashing) radiates into the **air** *and* into air-coupled ground
motion → the two are **coherent**; in-water turbulence over the bed and bedload impacts
radiate into the **ground** only and couple poorly to the air → **incoherent**. The
magnitude-squared coherence γ²(f) between colocated ground velocity and pressure is
normalised by the auto-spectra, so it is **independent of instrument response** and of
absolute level — a clean partition of the seismic band by source type.

**Result.** Across the December-2025 AR2 pulse the colocated seismic–infrasound
coherence stays at the **noise floor** (γ² ≲ 0.01, < 1% shared variance) over 0.5–20 Hz
at *both* CC.PR03 and CC.PR04, even at the discharge peak. This is not a dead sensor: as
a positive control, two elements of the CC.PR04 infrasound **array** are mutually
coherent (γ² ≈ 0.2 in 2–8 Hz), so a real, coherent acoustic field is present — the
seismic river band simply is not phase-locked to it. The flood drives seismic and
infrasound power together (a shared discharge envelope) but they remain statistically
independent at the waveform level. The colocated infrasound therefore **bounds the
air-coupled fraction of the seismic river signal as small**, consistent with the 1–20 Hz
seismic being dominated by ground-radiating sources (in-water turbulence over the bed
and bedload) rather than coherent air–water-interface acoustic radiation. (It does not
by itself exclude spatially-incoherent free-surface turbulence, which would also produce
low point coherence.) The canonical bedload band (30–80 Hz) lies above the 25 Hz
infrasound Nyquist — itself consistent with bedload not radiating into the air.

CC.PR03 is the study's on-channel anchor (USGS gage 12092000, Puyallup nr Electron) and
carries colocated 50-sps broadband seismic (BHZ) and 50-sps microbarometer infrasound
(BDF); CC.PR04 adds a 3-element BDF array a few km up-corridor. Top panel: time-resolved
γ²(f,t) (1-h windows) for PR03 seismic–infrasound through AR2 with discharge. Bottom:
flood-peak-averaged γ²(f) for PR03 seismic–infrasound and PR04 seismic–infrasound
(both at the floor) against the PR04 inter-element infrasound control (coherent), with
the K-segment 95% significance level.

Outputs paper/figures/fig32_seis_infrasound_coherence.png (+ cached arrays).

Usage: pixi run python workflows/41_seis_infrasound_coherence.py
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
from obspy import UTCDateTime
from scipy.signal import coherence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
DATADIR = ROOT / "notebooks" / "data"

NET = "CC"
GAGE = "12092000"                                  # Puyallup nr Electron (colocated w/ PR03)
T0, T1 = UTCDateTime("2025-12-07T00:00:00"), UTCDateTime("2025-12-14T00:00:00")
NPERSEG = 2048                                     # 40.96 s @ 50 sps -> ~0.024 Hz bins
STEP_HOURS = 1                                     # time resolution of the coherence panel
CACHE_NPZ = RESULTS / "coherence_CC_PR03.npz"


def _fetch(sta: str, loc: str, chan: str):
    """Colocated trace, merged/trimmed/padded to [T0, T1] at native 50 sps."""
    from obspy.clients.fdsn import Client
    cl = Client("IRIS", timeout=120)
    st = cl.get_waveforms(NET, sta, loc, chan, T0 - 60, T1 + 60)
    st.merge(method=1, fill_value=0)
    tr = st[0]
    tr.trim(T0, T1, pad=True, fill_value=0)
    tr.detrend("demean")
    if float(tr.stats.sampling_rate) != 50.0:
        tr.resample(50.0)
    return tr.data.astype(float)


def _avg_coh(x, y, t_start, sr):
    """24-h flood/quiet-window averaged coherence γ²(f)."""
    n = min(len(x), len(y))
    i0 = max(0, min(int((t_start - T0) * sr), n - int(86400 * sr)))
    i1 = min(i0 + int(86400 * sr), n)
    f, c = coherence(x[i0:i1], y[i0:i1], fs=sr, nperseg=NPERSEG,
                     noverlap=NPERSEG // 2, detrend="constant")
    K = (i1 - i0 - NPERSEG) // (NPERSEG // 2) + 1
    return f, c, K


def build_cache():
    """Download colocated channels + the array control, compute the hourly coherence
    spectrogram and flood/quiet averaged curves, write a compact npz."""
    sr = 50.0
    try:
        s3 = _fetch("PR03", "*", "BHZ")            # PR03 seismic (anchor)
        i3 = _fetch("PR03", "*", "BDF")            # PR03 infrasound (single element)
        s4 = _fetch("PR04", "", "BHZ")             # PR04 seismic (2nd colocated site)
        i4a = _fetch("PR04", "01", "BDF")          # PR04 infrasound element 01
        i4b = _fetch("PR04", "02", "BDF")          # PR04 infrasound element 02 (control)
    except Exception as e:
        print(f"   fetch failed: {e}")
        return None

    # discharge -> lowest/highest mean-Q full day in window = quiet / flood
    d = pd.read_csv(DATADIR / f"usgs_iv_{GAGE}_2025-12-01_2026-01-01.csv")
    dt = pd.to_datetime(d["time_utc"], utc=True)
    dq = pd.to_numeric(d["discharge_cfs"], errors="coerce") * 0.0283168
    mwin = (dt >= T0.datetime.replace(tzinfo=dt.dt.tz)) & (dt <= T1.datetime.replace(tzinfo=dt.dt.tz))
    dw = pd.DataFrame({"t": dt[mwin], "q": dq[mwin]}).dropna()
    nday = int((T1 - T0) // 86400)
    day_q = {}
    for k in range(nday):
        ds = T0.datetime.replace(tzinfo=dw["t"].dt.tz) + pd.Timedelta(days=k)
        sel = (dw["t"] >= ds) & (dw["t"] < ds + pd.Timedelta(days=1))
        if sel.any():
            day_q[k] = dw["q"][sel].mean()
    quiet_day = T0 + min(day_q, key=day_q.get) * 86400
    flood_day = T0 + max(day_q, key=day_q.get) * 86400

    # flood/quiet averaged curves
    f, coh_si_quiet, _ = _avg_coh(s3, i3, quiet_day, sr)         # PR03 seismic-infra, quiet
    _, coh_si_flood, Kf = _avg_coh(s3, i3, flood_day, sr)        # PR03 seismic-infra, flood
    _, coh_si4_flood, _ = _avg_coh(s4, i4a, flood_day, sr)       # PR04 seismic-infra, flood
    _, coh_ii_flood, _ = _avg_coh(i4a, i4b, flood_day, sr)       # PR04 infra-infra control, flood
    gamma95 = 1.0 - 0.05 ** (1.0 / max(Kf - 1, 1))

    # hourly time-resolved PR03 seismic-infrasound coherence spectrogram
    n = min(len(s3), len(i3))
    x, y = s3[:n], i3[:n]
    cols, tcol = [], []
    step = int(STEP_HOURS * 3600 * sr)
    win = step
    Kcol = max(1, (win - NPERSEG) // (NPERSEG // 2) + 1)
    for k0 in range(0, n - win + 1, step):
        ff, cc = coherence(x[k0:k0 + win], y[k0:k0 + win], fs=sr, nperseg=NPERSEG,
                           noverlap=NPERSEG // 2, detrend="constant")
        cols.append(cc.astype("float16"))
        tcol.append(mdates.date2num((T0 + (k0 + win / 2) / sr).datetime))
    C = np.array(cols).T

    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_NPZ, f=f.astype("float32"), tcol=np.array(tcol, dtype="float64"), C=C,
        coh_si_quiet=coh_si_quiet.astype("float32"), coh_si_flood=coh_si_flood.astype("float32"),
        coh_si4_flood=coh_si4_flood.astype("float32"), coh_ii_flood=coh_ii_flood.astype("float32"),
        sr=np.array([sr]), gamma95=np.array([gamma95]), gamma95_col=np.array([1 - 0.05 ** (1 / max(Kcol - 1, 1))]),
        quiet_day=np.array([quiet_day.timestamp]), flood_day=np.array([flood_day.timestamp]))
    print(f"cached {CACHE_NPZ.name}: C{C.shape}, K_avg={Kf}, γ²₉₅={gamma95:.4f}, "
          f"quiet={quiet_day.date}, flood={flood_day.date}")
    print(f"   PR03 seis-infra flood band-mean γ²(1-20Hz)={np.mean(coh_si_flood[(f>=1)&(f<20)]):.4f}; "
          f"PR04 infra-infra control={np.mean(coh_ii_flood[(f>=2)&(f<8)]):.3f} (2-8 Hz)")
    return dict(np.load(CACHE_NPZ))


def main() -> int:
    paper_style()
    data = build_cache()
    if data is None:
        if not CACHE_NPZ.exists():
            print("no waveforms and no cache — cannot build")
            return 1
        data = dict(np.load(CACHE_NPZ))
        print(f"rebuilt from cache {CACHE_NPZ.name}")

    f = data["f"]; tcol = data["tcol"]; C = data["C"].astype("float32")
    sr = float(data["sr"][0]); gamma95 = float(data["gamma95"][0])
    quiet_day = UTCDateTime(float(data["quiet_day"][0]))
    flood_day = UTCDateTime(float(data["flood_day"][0]))

    d = pd.read_csv(DATADIR / f"usgs_iv_{GAGE}_2025-12-01_2026-01-01.csv")
    dt = pd.to_datetime(d["time_utc"], utc=True)
    dq = pd.to_numeric(d["discharge_cfs"], errors="coerce") * 0.0283168
    m = (dt >= T0.datetime.replace(tzinfo=dt.dt.tz)) & (dt <= T1.datetime.replace(tzinfo=dt.dt.tz))

    fig, (ax, axb) = plt.subplots(2, 1, figsize=(12.6, 8.6),
                                  gridspec_kw=dict(height_ratios=[1.3, 1.0], hspace=0.34))

    # (top) hourly PR03 seismic-infrasound coherence + discharge
    fsel = f >= 0.4
    pcm = ax.pcolormesh(tcol, f[fsel], C[fsel], cmap="viridis",
                        norm=Normalize(0, 1), shading="auto", rasterized=True)
    ax.set_yscale("log"); ax.set_ylim(0.4, sr / 2); ax.set_ylabel("frequency (Hz)")
    ax.axhline(sr / 2, color="white", lw=0.8, ls=":", alpha=0.7)
    ax.text(tcol[-1], sr / 2, "Nyquist 25 Hz ", color="white", fontsize=8,
            va="bottom", ha="right", alpha=0.9)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(tcol[0], tcol[-1]); ax.set_xlabel("2025 (UTC)")
    ax.set_title("CC.PR03 colocated seismic (BHZ) — infrasound (BDF) coherence γ²(f,t): "
                 "stays at the noise floor while discharge spikes — the river band is not air-coupled",
                 fontsize=10.3, loc="left")
    axq = ax.twinx()
    axq.plot(mdates.date2num(dt[m].dt.tz_convert("UTC").dt.tz_localize(None)), dq[m],
             color="#ff2a6d", lw=1.8, alpha=0.95)
    axq.set_ylabel("$Q$ (m³ s⁻¹)", color="#ff2a6d"); axq.tick_params(axis="y", colors="#ff2a6d")
    axq.set_ylim(0, dq[m].max() * 1.05)
    cax = fig.add_axes((0.915, 0.55, 0.013, 0.34))
    cb = fig.colorbar(pcm, cax=cax); cb.set_label("coherence γ²")

    # (bottom) flood-peak averaged γ²(f): two colocated seis-infra (floor) vs array control
    axb.semilogx(f, data["coh_ii_flood"], color="#d48a00", lw=2.2,
                 label="infrasound–infrasound, PR04 array inter-element  (coherent: real acoustic field)")
    axb.semilogx(f, data["coh_si_flood"], color="#1a5fb4", lw=2.0,
                 label="seismic–infrasound, PR03 colocated  (at floor)")
    axb.semilogx(f, data["coh_si4_flood"], color="#5fa8e8", lw=1.6, ls="-",
                 label="seismic–infrasound, PR04 colocated  (at floor)")
    axb.semilogx(f, data["coh_si_quiet"], color="#9aa0a6", lw=1.2, ls=":",
                 label=f"seismic–infrasound, PR03 quiet ({quiet_day.date})")
    axb.axhline(gamma95, color="crimson", lw=1.0, ls="--",
                label=f"95% significance (γ²={gamma95:.3f})")
    axb.axvspan(1.0, 20.0, color="#67c7ff", alpha=0.10)
    axb.text(np.sqrt(1.0 * 20.0), 0.46, "turbulence band ~1–20 Hz", color="#2b8fd6",
             fontsize=8.5, ha="center", va="top")
    axb.set_xlim(0.4, sr / 2); axb.set_ylim(0, 0.5)
    axb.set_xlabel("frequency (Hz)"); axb.set_ylabel("coherence γ²")
    axb.set_title("Flood-peak averaged coherence: a coherent acoustic field exists (array control), "
                  "yet seismic shares < 1% of its variance with it", fontsize=10.3, loc="left")
    axb.legend(loc="upper left", fontsize=8.0, framealpha=0.92)

    fig.subplots_adjust(left=0.07, right=0.895, top=0.94, bottom=0.08)
    out_png = FIGDIR / "fig32_seis_infrasound_coherence.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png} + {CACHE_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
