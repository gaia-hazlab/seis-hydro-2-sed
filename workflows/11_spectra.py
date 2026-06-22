#!/usr/bin/env python3
"""Median ground-velocity PSD, flood vs quiet, to test which frequencies carry
the river signal — and whether a bedload bump exists at 30-80 Hz.

Compares a 50-sps source station (CC.PR03, Nyquist 25 Hz) and a 200-sps lowland
station (UW.UPS, Nyquist 100 Hz) on a flood day (10 Dec, AR2) vs a quiet pre-flood
day (03 Dec). Shades the turbulence (1-20 Hz) and canonical bedload (30-80 Hz)
bands. Outputs fig11_spectra.png.

Usage: pixi run python workflows/11_spectra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime, read as obspy_read
from obspy.clients.fdsn import Client
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks" / "data" / "fdsn_cache"
FIGDIR = ROOT / "paper" / "figures"
STATIONS = [("CC", "PR03", "#0072B2"), ("UW", "UPS", "#D55E00")]
QUIET, FLOOD = "20251203", "20251210"
plt.rcParams.update({"font.size": 9, "savefig.dpi": 200, "savefig.bbox": "tight"})


def day_psd(net, sta, day, inv):
    files = sorted(CACHE.glob(f"{net}.{sta}.*.{day}.*.mseed"))
    if not files:
        return None
    st = obspy_read(str(files[-1]))
    st = st.select(component="Z") or st
    try:
        st.merge(method=1, fill_value="interpolate")
    except Exception:
        pass
    tr = st[0]
    tr.detrend("demean"); tr.taper(0.02)
    if inv is not None:
        try:
            tr.attach_response(inv)
            tr.remove_response(output="VEL", water_level=60,
                               pre_filt=(0.05, 0.1, 0.45 * tr.stats.sampling_rate, 0.49 * tr.stats.sampling_rate))
        except Exception as e:
            print(f"   {net}.{sta} response failed: {e}")
            return None
    sr = float(tr.stats.sampling_rate)
    f, p = welch(tr.data.astype(float), fs=sr, nperseg=int(sr * 100), scaling="density")
    return f[1:], 10 * np.log10(p[1:] + 1e-30)


def main() -> int:
    cl = Client("IRIS", timeout=60)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.axvspan(1, 20, color="#0072B2", alpha=0.08, zorder=0)
    ax.axvspan(30, 80, color="#E69F00", alpha=0.12, zorder=0)
    ax.text(4.5, ax.get_ylim()[1], "", )
    for net, sta, c in STATIONS:
        try:
            inv = cl.get_stations(network=net, station=sta, channel="HH?,BH?,EH?,HN?,EN?",
                                  starttime=UTCDateTime("2025-12-01"), endtime=UTCDateTime("2025-12-24"),
                                  level="response")
        except Exception:
            inv = None
        for day, ls, lab in [(QUIET, ":", "quiet 03 Dec"), (FLOOD, "-", "flood 10 Dec")]:
            r = day_psd(net, sta, day, inv)
            if r is None:
                print(f"  no PSD for {net}.{sta} {day}"); continue
            f, p = r
            ax.semilogx(f, p, ls, color=c, lw=1.6 if ls == "-" else 1.1,
                        alpha=0.95 if ls == "-" else 0.7,
                        label=f"{net}.{sta} ({int(2*f[-1])} sps) — {lab}")
            ax.axvline(f[-1], color=c, ls="--", lw=0.7, alpha=0.5)  # Nyquist
    ax.text(7, ax.get_ylim()[0] + 3, "turbulence\n1–20 Hz", ha="center", fontsize=8, color="#0072B2")
    ax.text(49, ax.get_ylim()[0] + 3, "bedload\n30–80 Hz", ha="center", fontsize=8, color="#a8780a")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("velocity PSD (dB re m² s⁻²/Hz)")
    ax.set_title("Where is the river signal? Flood vs quiet ground-velocity spectra", loc="left")
    ax.set_xlim(0.5, 100)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25, which="both")
    fig.savefig(FIGDIR / "fig11_spectra.png")
    plt.close(fig)
    print(f"wrote {FIGDIR}/fig11_spectra.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
