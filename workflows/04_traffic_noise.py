#!/usr/bin/env python3
"""Identify lowland stations polluted by anthropogenic (traffic) noise.

Traffic noise has a strong weekday + daytime diurnal cycle. Over a PRE-FLOOD,
quiet week (2025-12-01 to 2025-12-08; Dec-1 is a Monday, Dec-6/7 the weekend) we
compute hourly band power (4–12 Hz, cultural band, vertical component) for each
candidate station and quantify:

  traffic_index = median(weekday daytime) / median(night)      [diurnal strength]
  weekday_weekend = median(weekday daytime) / median(weekend daytime)

A station is flagged "traffic-polluted" if it shows both a strong day/night
contrast and a clear weekday>weekend excess. Outputs a classification table
(config/traffic_noise.json) and an hour-of-week profile figure.

Usage: pixi run python workflows/04_traffic_noise.py
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
from obspy import UTCDateTime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from utils import compute_proxy_from_fdsn  # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "fdsn_cache"
OUTJSON = ROOT / "config" / "traffic_noise.json"
FIG = ROOT / "paper" / "figures" / "figS_traffic_noise.png"

START = UTCDateTime("2025-12-01T00:00:00")   # pre-flood Mon
END = UTCDateTime("2025-12-08T00:00:00")     # through the weekend, before main flood
PST = -8                                     # local = UTC + PST
BAND = (4.0, 12.0)

# Candidate lowland/urban stations + a few references (source/borehole).
STATIONS = [
    ("UW", "TEHA"), ("UW", "QDNP"), ("UW", "PAYL"), ("UW", "HAWK"),
    ("GS", "TFD"), ("UW", "UPS"), ("UW", "BRPT"), ("UW", "TBPA"),
    ("UW", "QKSO"), ("UW", "QTKM"), ("UW", "GHW"),
    ("PB", "B941"),                 # borehole — expected quiet
    ("CC", "TRON"), ("CC", "PR03"), ("CC", "STYX"),   # river refs
]
CHANNEL = "HH?,HN?,BH?,EH?,EN?,DP?"


def hourly_power(net: str, sta: str) -> pd.Series | None:
    try:
        s = compute_proxy_from_fdsn(
            net, sta, START, END, fmin=BAND[0], fmax=BAND[1],
            win_seconds=3600, step_seconds=3600, output="velocity",
            method="bandpower", combine="z", components=("Z",),
            channel=CHANNEL, location="*", remove_response=True,
            cache_dir=CACHE, use_cache=True,
        )
        return s if s is not None and not s.empty else None
    except Exception as e:
        print(f"  {net}.{sta}: {e}")
        return None


def main() -> int:
    rows, profiles = [], {}
    for net, sta in STATIONS:
        sid = f"{net}.{sta}"
        print(f"[{sid}] hourly band power {BAND[0]}–{BAND[1]} Hz …")
        s = hourly_power(net, sta)
        if s is None or len(s) < 48:
            print(f"  {sid}: insufficient data; skipping")
            continue
        df = pd.DataFrame({"P": s.values}, index=s.index)
        loc = df.index + pd.Timedelta(hours=PST)
        df["hod"] = loc.hour
        df["dow"] = loc.dayofweek            # Mon=0 … Sun=6
        df["hour_of_week"] = df["dow"] * 24 + df["hod"]
        wk_day = df[(df.dow < 5) & df.hod.between(7, 19)]["P"]
        we_day = df[(df.dow >= 5) & df.hod.between(7, 19)]["P"]
        night = df[df.hod.between(0, 5)]["P"]
        ti = float(wk_day.median() / night.median()) if night.median() else np.nan
        ww = float(wk_day.median() / we_day.median()) if len(we_day) and we_day.median() else np.nan
        polluted = bool(np.isfinite(ti) and np.isfinite(ww) and ti > 2.5 and ww > 1.3)
        rows.append(dict(station=sid, n_hours=int(len(df)), traffic_index=round(ti, 2),
                         weekday_weekend=round(ww, 2), polluted=polluted))
        # median hour-of-week profile (normalized to its own median)
        profiles[sid] = (df.groupby("hour_of_week")["P"].median() / df["P"].median())
        print(f"  {sid}: traffic_index={ti:.2f} wkdy/wknd={ww:.2f} -> "
              f"{'TRAFFIC-POLLUTED' if polluted else 'clean'}")

    tbl = pd.DataFrame(rows).sort_values("traffic_index", ascending=False)
    OUTJSON.write_text(json.dumps(rows, indent=2))
    print("\n" + tbl.to_string(index=False))
    print(f"\nwrote {OUTJSON}")

    # figure: hour-of-week profiles, polluted vs clean
    if profiles:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for sid, prof in profiles.items():
            pol = next((r["polluted"] for r in rows if r["station"] == sid), False)
            ax.plot(prof.index, prof.values, lw=1.3 if pol else 0.9,
                    color="firebrick" if pol else "0.55",
                    alpha=0.95 if pol else 0.6,
                    label=sid if pol else None, zorder=3 if pol else 1)
        for d in range(1, 7):  # day boundaries
            ax.axvline(d * 24, color="0.85", lw=0.6, zorder=0)
        for d in range(5, 7):  # shade weekend
            ax.axvspan(d * 24, (d + 1) * 24, color="0.9", alpha=0.5, zorder=0)
        ax.set_yscale("log")
        ax.set_xlim(0, 168)
        ax.set_xticks([12 + 24 * i for i in range(7)])
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlabel("hour of week (local PST)")
        ax.set_ylabel(f"{BAND[0]:g}–{BAND[1]:g} Hz power / median")
        ax.set_title("Anthropogenic (traffic) noise: red = polluted (weekday/daytime cycle)", loc="left")
        ax.legend(fontsize=7, ncol=2, loc="upper right", title="traffic-polluted")
        fig.tight_layout()
        fig.savefig(FIG, dpi=200)
        print(f"wrote {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
