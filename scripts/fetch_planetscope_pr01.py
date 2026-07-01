#!/usr/bin/env python3
r"""Fetch AOI-clipped PlanetScope scenes over the CC.PR01 braidplain for the hero
zoom figure (issue #10). Uses YOUR Planet Education & Research account.

Auth: reads the Planet API key from $PL_API_KEY (Account -> My Settings -> API key).
      Do NOT hard-code or commit the key.

For each date window (pre-flood, post-flood, and optionally the AR peak) it:
  1. Data-API quick-search for PSScene over the PR01 AOI, cloud_cover <= CLOUD_MAX;
  2. picks the least-cloud scene;
  3. places an Orders-API order for the 4-band surface-reflectance bundle, CLIPPED to
     the AOI (so each download is a few MB, not a full ~24-km scene);
  4. polls the order and downloads the clipped GeoTIFF(s).

Output GeoTIFFs land in notebooks/data/planet_cache/<window>/ (gitignored — PlanetScope
imagery is licensed and must not be committed; the *derived* figure fig33 is publishable
with attribution "(c) Planet Labs PBC"). Then build the upgraded figure:

    pixi run python workflows/42_pr01_braid_zoom.py \
        --pre-raster  notebooks/data/planet_cache/pre/<file>.tif \
        --post-raster notebooks/data/planet_cache/post/<file>.tif

Usage:
    PL_API_KEY=xxxx pixi run python scripts/fetch_planetscope_pr01.py
    # options: --windows pre,post,peak   --cloud-max 0.2   --bundle analytic_sr_udm2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "data" / "planet_cache"
DATA_URL = "https://api.planet.com/data/v1/quick-search"
ORDERS_URL = "https://api.planet.com/compute/ops/orders/v2"

# CC.PR01 (UTM 10N 573289, 5195623) -> lon/lat; AOI = ~2 km box around it.
_LON, _LAT = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True).transform(
    573289.42, 5195623.45)
DLON, DLAT = 0.020, 0.015                        # ~±1.5 km lon, ±1.7 km lat
AOI = {
    "type": "Polygon",
    "coordinates": [[
        [_LON - DLON, _LAT - DLAT], [_LON + DLON, _LAT - DLAT],
        [_LON + DLON, _LAT + DLAT], [_LON - DLON, _LAT + DLAT],
        [_LON - DLON, _LAT - DLAT],
    ]],
}
WINDOWS = {                                       # UTC date ranges
    "pre":  ("2025-11-16T00:00:00Z", "2025-11-30T23:59:59Z"),
    "peak": ("2025-12-09T00:00:00Z", "2025-12-14T23:59:59Z"),
    "post": ("2025-12-20T00:00:00Z", "2026-01-31T23:59:59Z"),
}


def search(sess, t0, t1, cloud_max):
    body = {
        "item_types": ["PSScene"],
        "filter": {"type": "AndFilter", "config": [
            {"type": "GeometryFilter", "field_name": "geometry", "config": AOI},
            {"type": "DateRangeFilter", "field_name": "acquired",
             "config": {"gte": t0, "lte": t1}},
            {"type": "RangeFilter", "field_name": "cloud_cover",
             "config": {"lte": cloud_max}},
        ]},
    }
    r = sess.post(DATA_URL, json=body, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    feats.sort(key=lambda f: f["properties"].get("cloud_cover", 1.0))
    return feats


def order_and_download(sess, item_id, bundle, dest: Path):
    payload = {
        "name": dest.name,
        "products": [{"item_ids": [item_id], "item_type": "PSScene",
                      "product_bundle": bundle}],
        "tools": [{"clip": {"aoi": AOI}}],
    }
    r = sess.post(ORDERS_URL, json=payload, timeout=60)
    if r.status_code >= 400:
        print(f"   order failed ({r.status_code}): {r.text[:300]}")
        return False
    order = r.json()
    self_url = order["_links"]["_self"]
    print(f"   order {order['id']} placed; polling…")
    for _ in range(120):                          # up to ~20 min
        time.sleep(10)
        o = sess.get(self_url, timeout=60).json()
        state = o["state"]
        if state == "success":
            break
        if state in ("failed", "partial"):
            print(f"   order ended in state={state}")
            if state == "failed":
                return False
            break
        print(f"   … {state}")
    dest.mkdir(parents=True, exist_ok=True)
    results = o["_links"].get("results", [])
    tifs = 0
    for res in results:
        name = res["name"].split("/")[-1]
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        loc = res["location"]
        blob = sess.get(loc, timeout=300)
        (dest / name).write_bytes(blob.content)
        print(f"   downloaded {name} ({len(blob.content)//1024} KB)")
        tifs += 1
    return tifs > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="pre,post",
                    help="comma list of: pre, peak, post (default pre,post)")
    ap.add_argument("--cloud-max", type=float, default=0.2)
    ap.add_argument("--bundle", default="analytic_sr_udm2",
                    help="Planet product bundle (analytic_sr_udm2 = 4-band SR incl. NIR)")
    args = ap.parse_args()

    key = os.environ.get("PL_API_KEY")
    if not key:
        print("ERROR: set PL_API_KEY (Planet Account -> My Settings -> API key).")
        return 2

    sess = requests.Session()
    sess.auth = (key, "")
    print(f"AOI center lon={_LON:.5f} lat={_LAT:.5f}  (~{2*DLON*76:.1f}×{2*DLAT*111:.1f} km box)")
    ok = 0
    for w in [s.strip() for s in args.windows.split(",") if s.strip()]:
        if w not in WINDOWS:
            print(f"skip unknown window '{w}'"); continue
        t0, t1 = WINDOWS[w]
        print(f"\n[{w}] {t0[:10]}..{t1[:10]}  cloud<= {args.cloud_max}")
        feats = search(sess, t0, t1, args.cloud_max)
        if not feats:
            print("   no scenes found — widen dates or raise --cloud-max"); continue
        best = feats[0]
        print(f"   {len(feats)} scenes; best {best['id']} "
              f"cloud={best['properties'].get('cloud_cover'):.2f} "
              f"acq={best['properties'].get('acquired','')[:10]}")
        if order_and_download(sess, best["id"], args.bundle, OUT / w):
            ok += 1
    print(f"\nDone: {ok} window(s) downloaded to {OUT}")
    print("Next: pixi run python workflows/42_pr01_braid_zoom.py "
          "--pre-raster notebooks/data/planet_cache/pre/*.tif "
          "--post-raster notebooks/data/planet_cache/post/*.tif")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
