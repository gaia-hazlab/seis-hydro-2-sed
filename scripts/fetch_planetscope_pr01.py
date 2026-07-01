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


ASSET_PREF = ("ortho_analytic_4b_sr", "ortho_analytic_8b_sr",
              "ortho_analytic_4b", "ortho_visual")


def download_asset(sess, item_id, dest: Path):
    """Data-API asset download (no Orders/clip permission needed): activate the best
    available 4-band SR asset, poll until active, stream the full-scene GeoTIFF.
    workflow 42 crops to the PR01 window locally."""
    assets_url = f"https://api.planet.com/data/v1/item-types/PSScene/items/{item_id}/assets"
    assets = sess.get(assets_url, timeout=60).json()
    akey = next((a for a in ASSET_PREF if a in assets), None)
    if akey is None:
        print(f"   no SR/analytic asset; available: {list(assets)}")
        return None
    asset = assets[akey]
    if asset["status"] != "active":
        sess.get(asset["_links"]["activate"], timeout=60)      # trigger activation
        for _ in range(60):                                    # up to ~10 min
            time.sleep(10)
            asset = sess.get(assets_url, timeout=60).json()[akey]
            if asset["status"] == "active":
                break
            print(f"   activating {akey}: {asset['status']}")
    if asset["status"] != "active":
        print(f"   {akey} did not activate in time")
        return None
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"{item_id}_{akey}.tif"
    with sess.get(asset["location"], stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(out, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    print(f"   downloaded {out.name} ({out.stat().st_size // (1 << 20)} MB, asset {akey})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="pre,post",
                    help="comma list of: pre, peak, post (default pre,post)")
    ap.add_argument("--cloud-max", type=float, default=0.2)
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
        if download_asset(sess, best["id"], OUT / w):
            ok += 1
    print(f"\nDone: {ok} window(s) downloaded to {OUT}")
    print("Next: pixi run python workflows/42_pr01_braid_zoom.py "
          "--pre-raster notebooks/data/planet_cache/pre/*.tif "
          "--post-raster notebooks/data/planet_cache/post/*.tif")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
