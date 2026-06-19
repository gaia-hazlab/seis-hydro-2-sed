#!/usr/bin/env python3
"""Discover seismic stations + USGS gages along the Puyallup mountain-to-sea corridor.

Queries FDSN (IRIS) for stations in a bounding box from Mt. Rainier's upper
Puyallup (glacial source) to Tacoma / Commencement Bay (Puget Sound), projects
each onto an approximate river polyline to get a downstream river-km, and tags
broadband (BH/EH/HH) vs urban strong-motion (HN/EN). Writes
``config/_transect_discovery.json``.

Usage:
    pixi run python workflows/00_discover_stations.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

ROOT = Path(__file__).resolve().parents[1]

# Approximate Puyallup mainstem polyline, mountain -> sea (lat, lon).
RIVER = [
    (46.904, -122.035),  # nr Electron (glacial source)
    (47.039, -122.208),  # nr Orting
    (47.100, -122.220),
    (47.185, -122.230),  # Alderton
    (47.208, -122.327),  # Puyallup
    (47.270, -122.420),  # Commencement Bay / Puget Sound
]


def _hav(a: float, b: float, c: float, d: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - b)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _cumulative_km() -> list[float]:
    cum = [0.0]
    for i in range(1, len(RIVER)):
        cum.append(cum[-1] + _hav(*RIVER[i - 1], *RIVER[i]))
    return cum


def project(lat: float, lon: float, cum: list[float]) -> tuple[float, float]:
    """Return (distance_to_river_km, river_km_downstream) using nearest vertex.

    Coarse (nearest-vertex) projection; adequate for ordering stations along the
    corridor, not for precise channel distances.
    """
    best = (1e9, 0.0)
    for i, (rla, rlo) in enumerate(RIVER):
        d = _hav(lat, lon, rla, rlo)
        if d < best[0]:
            best = (d, cum[i])
    return best


def main() -> int:
    cl = Client("IRIS", timeout=120)
    inv = cl.get_stations(
        network="*", station="*", channel="HH?,EH?,BH?,HN?,EN?,DP?",
        starttime=UTCDateTime("2025-12-01"), endtime=UTCDateTime("2025-12-24"),
        minlatitude=46.85, maxlatitude=47.32,
        minlongitude=-122.50, maxlongitude=-121.85, level="channel",
    )
    cum = _cumulative_km()
    seen: dict[str, dict] = {}
    for net in inv:
        for sta in net:
            key = f"{net.code}.{sta.code}"
            if key in seen:
                continue
            d, rkm = project(sta.latitude, sta.longitude, cum)
            chans = sorted({c.code[:2] for c in sta.channels})
            seen[key] = dict(
                net=net.code, sta=sta.code,
                lat=round(sta.latitude, 4), lon=round(sta.longitude, 4),
                dist_river_km=round(d, 2), river_km=round(rkm, 1),
                chan=",".join(chans),
                broadband=any(c.startswith(("HH", "BH", "EH")) for c in chans),
                site=(sta.site.name[:40] if sta.site and sta.site.name else ""),
            )

    near = sorted(
        (v for v in seen.values() if v["dist_river_km"] < 8),
        key=lambda v: (v["river_km"], v["dist_river_km"]),
    )
    out = ROOT / "config" / "_transect_discovery.json"
    out.write_text(json.dumps({"stations": near}, indent=2))
    print(f"Wrote {out} ({len(near)} stations within 8 km of the Puyallup mainstem)")
    for v in near:
        print(f'  {v["net"]+"."+v["sta"]:12s} rkm={v["river_km"]:5.1f} '
              f'd={v["dist_river_km"]:4.1f}km bb={"Y" if v["broadband"] else "n"} '
              f'{v["chan"]:10s} {v["site"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
