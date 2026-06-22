#!/usr/bin/env python3
"""Probe the lower bedload edge (30-50 Hz) with the highest-rate data AVAILABLE
for the Dec-2025 event.

The 50-sps CC (BH) channels cap at 25 Hz (turbulence only); the 500-sps CC (CH)
channels came online in 2026 and are not archived for 2025. The best near-channel
high-rate data that exist for the event are the 100-sps (HH) PNSN/UW broadband
stations UW.RER (0.44 km) and UW.LON (0.21 km), Nyquist 50 Hz, reaching a 30-50 Hz
lower bedload edge. We (1) compute band power in a flow band (2-8 Hz) and the
30-50 Hz edge, fit P∝Q^b against the co-located gage, and (2) compare flood-vs-quiet
median PSD to look for elevated high-frequency power during the flood.

Outputs config/bedload_ch_fit.json and fig13_bedload_ch.png.
Usage: pixi run python workflows/14_bedload_ch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime, read as obspy_read
from obspy.clients.fdsn import Client
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from utils import compute_proxy_from_fdsn, fetch_usgs_gage_timeseries  # noqa: E402

DATA = ROOT / "notebooks" / "data"
CACHE = DATA / "fdsn_cache"
FIGDIR = ROOT / "paper" / "figures"
START, END = UTCDateTime("2025-12-04"), UTCDateTime("2025-12-13")
QUIET, FLOOD = "20251204", "20251210"
# Best high-rate data AVAILABLE for Dec-2025: 100-sps UW broadband near-channel
# stations (CC 500-sps CH came online only in 2026; not archived for this event).
# 100 sps -> Nyquist 50 Hz -> reaches the 30-50 Hz lower bedload edge.
STATIONS = [("UW", "RER", "12092000", "#0072B2"), ("UW", "LON", "12082500", "#D55E00")]
BANDS = {"flow_2_8": (2.0, 8.0), "bed_30_50": (30.0, 50.0)}
CHANNEL = "HH?"


def proxy(net, sta, band):
    return compute_proxy_from_fdsn(
        net, sta, START, END, fmin=band[0], fmax=band[1], win_seconds=600, step_seconds=300,
        output="velocity", method="bandpower", combine="rss", components=("Z", "N", "E"),
        channel=CHANNEL, location="*", remove_response=True, cache_dir=CACHE, use_cache=True,
        clip_impulsive_days=True, despike_proxy=True)


def day_psd(net, sta, day, inv):
    files = sorted(CACHE.glob(f"{net}.{sta}.*HHQ*.{day}.*.mseed"))
    if not files:
        return None
    st = obspy_read(str(files[-1])).select(component="Z")
    if not len(st):
        return None
    st.merge(method=1, fill_value="interpolate")
    tr = st[0]; tr.detrend("demean"); tr.taper(0.02)
    try:
        tr.attach_response(inv); tr.remove_response(output="VEL", water_level=60,
            pre_filt=(0.05, 0.1, 0.45 * tr.stats.sampling_rate, 0.49 * tr.stats.sampling_rate))
    except Exception as e:
        print(f"  {sta} {day} resp: {e}"); return None
    sr = float(tr.stats.sampling_rate)
    f, p = welch(tr.data.astype(float), fs=sr, nperseg=int(sr * 50), scaling="density")
    return f[1:], 10 * np.log10(p[1:] + 1e-30)


def main() -> int:
    cl = Client("IRIS", timeout=90)
    fits, fig = [], plt.figure(figsize=(11, 4.6))
    axS = fig.add_subplot(1, 2, 1)   # spectra
    axB = fig.add_subplot(1, 2, 2)   # P-Q scaling in bedload band
    axS.axvspan(1, 20, color="#0072B2", alpha=0.08); axS.axvspan(30, 80, color="#E69F00", alpha=0.12)

    for net, sta, gid, c in STATIONS:
        g = fetch_usgs_gage_timeseries(gid, START, END, data_dir=DATA, use_cache=True)
        Q = pd.to_numeric(g.get("discharge_cms"), errors="coerce").dropna()
        rec = {"station": f"{net}.{sta}"}
        for name, band in BANDS.items():
            try:
                P = proxy(net, sta, band)
            except Exception as e:
                print(f"  {sta} {name}: {e}"); continue
            j = pd.concat([P.rename("P"), Q.rename("Q")], axis=1).sort_index()
            j["Q"] = j["Q"].interpolate("time", limit=12); j = j.dropna()
            if len(j) < 50:
                continue
            lq = np.log10(j["Q"].clip(lower=1e-6)); lp = np.log10(j["P"].clip(lower=1e-30))
            b, a = np.polyfit(lq.values, lp.values, 1); r = float(np.corrcoef(lq, lp)[0, 1])
            rec[name] = dict(b=round(float(b), 2), r=round(r, 2), n=len(j))
            if name == "bed_30_50":
                axB.scatter(lq, lp, s=4, alpha=0.3, color=c)
                xs = np.linspace(lq.min(), lq.max(), 30)
                axB.plot(xs, a + b * xs, color=c, lw=2, label=f"{sta}  b={b:.2f} r={r:.2f}")
        fits.append(rec)
        # spectra flood vs quiet
        try:
            inv = cl.get_stations(network=net, station=sta, channel=CHANNEL,
                                  starttime=START, endtime=END, level="response")
        except Exception:
            inv = None
        for day, ls, lab in [(QUIET, ":", "quiet 04 Dec"), (FLOOD, "-", "flood 10 Dec")]:
            r = day_psd(net, sta, day, inv)
            if r:
                axS.semilogx(r[0], r[1], ls, color=c, lw=1.5 if ls == "-" else 1.0,
                             alpha=0.95 if ls == "-" else 0.6, label=f"{sta} — {lab}")

    axS.set_xlim(0.5, 60); axS.set_xlabel("frequency (Hz)"); axS.set_ylabel("velocity PSD (dB)")
    axS.set_title("100-sps flood-vs-quiet spectra (30–50 Hz lower bedload edge)", loc="left", fontsize=9.5)
    axS.legend(fontsize=7, loc="lower left"); axS.grid(alpha=0.25, which="both")
    axB.set_xlabel(r"$\log_{10} Q$ (m³/s)"); axB.set_ylabel(r"$\log_{10} P$ (30–50 Hz)")
    axB.set_title("30–50 Hz scaling P∝Q$^b$ (100-sps UW)", loc="left", fontsize=9.5)
    axB.legend(fontsize=8); axB.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig13_bedload_ch.png", dpi=200)
    (ROOT / "config" / "bedload_ch_fit.json").write_text(json.dumps(fits, indent=2))
    print(json.dumps(fits, indent=2)); print(f"wrote {FIGDIR}/fig13_bedload_ch.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
