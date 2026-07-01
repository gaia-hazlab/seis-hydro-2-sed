#!/usr/bin/env python3
r"""Infrasound (microbarometer) power spectrogram through AR2 — the missing half of
the seismic spectrogram (fig30 / workflow 39).

fig30 shows the *seismic* band structure at the Nisqually source UW.LON and notes the
core ambiguity: a bright seismic band cannot say whether the source is at the **bed**
(bedload impacts) or at the **air–water interface** (free-surface turbulence). The
discriminant is colocated **infrasound**: free-surface turbulence — breaking waves,
hydraulic jumps, splashing in rapids — radiates strongly into the air, whereas bedload
grains striking the bed couple poorly to the atmosphere. So an infrasound band that
*tracks discharge* is evidence for an air–water-interface (surface-turbulence) source.

Contrary to the earlier "this network has no infrasound channel" caveat, the regional
network does carry it: the USGS Cascades Volcano Observatory (network **CC**) runs
microbarometer **BDF** arrays colocated with broadband seismometers along the
upper-Puyallup lahar corridor. We use the three with continuous BDF data through the
December-2025 AR2 pulse (Dec 7–14):

  * **CC.PR03** — the study's on-channel anchor, single BDF element, sitting on USGS
    gage 12092000 (Puyallup nr Electron); colocated 50-sps seismic (used in fig32).
  * **CC.PR04** — 3-element BDF array, ~5 km up-corridor.
  * **CC.STYX** — 3-element BDF array, ~7 km up-corridor.

(CC.PR01 / CC.PR02 carried BDF only in 2018–2020 and are unavailable for this event;
the colocated arrays nearest the Nisqually source, e.g. CC.KAUT ~4 km from UW.LON, lack
the AR2 onset.) BDF is 50 sps → **25 Hz Nyquist**, so the canonical bedload band
(30–80 Hz) is *above* the 25 Hz sampling Nyquist and simply not recorded here — an
instrumental limit. Array elements are averaged in power (incoherent stack) to
suppress wind/self-noise. Power is relative (raw counts in dB).

The honest caveat this figure makes visible: infrasound is **heavily wind/weather
contaminated** (the broadband vertical streaks), and band-integrated 1–20 Hz power does
*not* track discharge consistently across sites — Spearman r ranges from −0.36 at PR03
to +0.36 (PR04) and +0.56 (STYX). So raw infrasound power is **not** a clean river-flow
proxy. The robust, response-independent discriminant is the colocated seismic–infrasound
*coherence* (fig32), not infrasound power; this panel documents the field that
coherence is computed against.

Outputs paper/figures/fig31_infrasound_psd.png (+ cached arrays for offline rebuild).

Usage: pixi run python workflows/40_infrasound_psd.py
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
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

RESULTS = ROOT / "notebooks" / "data" / "results"
FIGDIR = ROOT / "paper" / "figures"
DATADIR = ROOT / "notebooks" / "data"

NET = "CC"
# (station, gage, label) — all three share the Puyallup-nr-Electron anchor hydrograph
# (12092000); PR03 sits on that gage, PR04/STYX are a few km up the same corridor.
STATIONS = [
    ("PR03", "anchor, on USGS 12092000 (1 BDF element)"),
    ("PR04", "3-element array, ~5 km up-corridor"),
    ("STYX", "3-element array, ~7 km up-corridor"),
]
GAGE = "12092000"                                  # Puyallup nr Electron (anchor)
T0, T1 = UTCDateTime("2025-12-07T00:00:00"), UTCDateTime("2025-12-14T00:00:00")
TURB = (1.0, 20.0)                                 # turbulence-dominated band
CACHE_NPZ = RESULTS / "infrasound_specgram_CC.npz"


def compute_station(sta: str):
    """Download every BDF element for `sta`, return (f, tcol, SdB) for the
    power-averaged (incoherent array stack) spectrogram. None if no data."""
    from obspy.clients.fdsn import Client
    cl = Client("IRIS", timeout=120)
    try:
        st = cl.get_waveforms(NET, sta, "*", "BDF", T0 - 60, T1 + 60)
    except Exception as e:
        print(f"   {NET}.{sta} BDF fetch failed: {e}")
        return None
    st.merge(method=1, fill_value=0)
    sr = float(st[0].stats.sampling_rate)
    nper = int(sr * 60)                            # 60-s windows
    Sxx_sum, f, t = None, None, None
    n = 0
    for tr in st:
        tr.trim(T0, T1, pad=True, fill_value=0)
        tr.detrend("demean")
        f, t, Sxx = spectrogram(tr.data.astype(float), fs=sr,
                                nperseg=nper, noverlap=nper // 2, scaling="density")
        Sxx_sum = Sxx if Sxx_sum is None else Sxx_sum + Sxx
        n += 1
    Sxx = Sxx_sum / max(n, 1)                       # incoherent power average over elements
    SdB = 10 * np.log10(Sxx + 1e-20)
    tcol = np.array(mdates.date2num([(T0 + float(s)).datetime for s in t]))
    # compact (log-f, time-decimated, float16) cache, as in workflow 39
    keep_f = np.unique(np.searchsorted(f, np.geomspace(0.4, sr / 2, 400)).clip(1, len(f) - 1))
    f, SdB = f[keep_f], SdB[keep_f]
    step = max(1, SdB.shape[1] // 1800)
    SdB, tcol = SdB[:, ::step], tcol[::step]
    print(f"   {NET}.{sta}: {n} BDF element(s), sr={sr:g} Hz, {SdB.shape[1]} cols")
    return f.astype("float32"), tcol.astype("float64"), SdB.astype("float16"), sr


def build_cache():
    """Compute every station and write one compact npz. Returns dict or None."""
    out, ok = {}, False
    for sta, _ in STATIONS:
        res = compute_station(sta)
        if res is None:
            continue
        f, tcol, SdB, sr = res
        out[f"{sta}_f"], out[f"{sta}_t"] = f, tcol
        out[f"{sta}_SdB"], out[f"{sta}_sr"] = SdB, np.array([sr])
        ok = True
    if not ok:
        return None
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_NPZ, **out)
    print(f"cached {CACHE_NPZ.name}")
    return out


def main() -> int:
    paper_style()
    data = build_cache()
    if data is None:
        if not CACHE_NPZ.exists():
            print("no BDF waveforms and no cache — cannot build")
            return 1
        data = dict(np.load(CACHE_NPZ))
        print(f"rebuilt from cache {CACHE_NPZ.name}")

    # discharge overlay (common anchor hydrograph)
    d = pd.read_csv(DATADIR / f"usgs_iv_{GAGE}_2025-12-01_2026-01-01.csv")
    dt = pd.to_datetime(d["time_utc"], utc=True)
    dq = pd.to_numeric(d["discharge_cfs"], errors="coerce") * 0.0283168
    m = (dt >= T0.datetime.replace(tzinfo=dt.dt.tz)) & (dt <= T1.datetime.replace(tzinfo=dt.dt.tz))

    avail = [(sta, lab) for sta, lab in STATIONS if f"{sta}_SdB" in data]
    fig, axes = plt.subplots(len(avail), 1, figsize=(13.0, 2.6 * len(avail) + 1.0),
                             sharex=True)
    if len(avail) == 1:
        axes = [axes]

    pcm = None
    for ax, (sta, lab) in zip(axes, avail):
        f = data[f"{sta}_f"]
        tcol = data[f"{sta}_t"]
        SdB = data[f"{sta}_SdB"].astype("float32")
        sr = float(data[f"{sta}_sr"][0])
        fsel = f >= 0.5
        vmin, vmax = np.percentile(SdB[fsel], [5, 99])
        pcm = ax.pcolormesh(tcol, f[fsel], SdB[fsel], cmap="magma",
                            norm=Normalize(vmin, vmax), shading="auto", rasterized=True)
        ax.set_yscale("log")
        ax.set_ylim(0.5, sr / 2)
        lo, hi = TURB
        ax.axhline(lo, color="#67c7ff", lw=1.0, ls="--", alpha=0.8)
        ax.axhline(hi, color="#67c7ff", lw=1.0, ls="--", alpha=0.8)
        ax.text(tcol[2], np.sqrt(lo * hi), " 1–20 Hz band",
                color="#67c7ff", fontsize=8.5, va="center", ha="left", fontweight="bold")
        ax.axhline(sr / 2, color="white", lw=0.8, ls=":", alpha=0.7)
        ax.text(tcol[-2], sr / 2, "Nyquist 25 Hz — bedload band 30–80 Hz is above the infrasound passband ",
                color="white", fontsize=7.0, va="bottom", ha="right", alpha=0.9)
        ax.set_ylabel("frequency (Hz)")
        ax.set_title(f"{NET}.{sta} BDF infrasound — {lab}", fontsize=10, loc="left")

        axq = ax.twinx()
        axq.plot(mdates.date2num(dt[m].dt.tz_convert("UTC").dt.tz_localize(None)), dq[m],
                 color="#39ff14", lw=1.6, alpha=0.9)
        axq.set_ylabel("$Q$ (m³ s⁻¹)", color="#1a9e1a", fontsize=9)
        axq.tick_params(axis="y", colors="#1a9e1a", labelsize=8)
        axq.set_ylim(0, dq[m].max() * 1.05)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlabel("2025 (UTC)")
    fig.suptitle("CC microbarometer (BDF) infrasound through AR2 — an energetic but wind-contaminated "
                 "field; band power tracks discharge only weakly/inconsistently (cf. coherence, fig32)",
                 fontsize=11.0, x=0.055, ha="left")
    fig.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.10, hspace=0.32)
    cax = fig.add_axes((0.925, 0.10, 0.012, 0.82))
    cb = fig.colorbar(pcm, cax=cax)
    cb.set_label("relative power (dB)")
    out_png = FIGDIR / "fig31_infrasound_psd.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"wrote {out_png} + {CACHE_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
