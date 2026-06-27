#!/usr/bin/env python3
"""Fetch + cache Sentinel-2 true-color basemaps for the geospatial figures (one-time).

The geospatial figures (study-area locator, braidplain zooms, satellite change
maps) underlay a REAL co-located Sentinel-2 true-color image with the
active-channel masks drawn on top. No RGB imagery is cached yet — only MNDWI
indices and binary masks — so this script pulls true-color (B04/B03/B02) once and
commits it, after which every figure rebuilds OFFLINE via riverseis.basemap.

Design
------
* Braidplain regions (`puyallup`, `nisqually`) are cached on the **same UTM 10N
  grid, bbox and resolution** as the active-channel rasters in
  `{region}_rasters.npz` (reusing the AOI/CRS/RES from
  workflows/19_braid_optical_change.py) so masks overlay pixel-for-pixel.
* The `corridor` locator is cached in **lon/lat** at coarse resolution for the
  lon/lat station/river overlays of the study-area map.
* Composite window defaults to a **clear-sky summer 2025** range (PNW Nov is
  cloud-blind); the basemap is illustrative landscape context, not epoch-matched —
  captions say so. Override with --window.
* Robustness: try **Microsoft Planetary Computer** first, fall back to
  **Element84 Earth Search** (no credentials) if PC is flaky (the user flagged PC
  as buggy). Per-band percentile stretch to uint8.

Outputs (committed, force-added like the other braid_cache artefacts):
  notebooks/data/braid_cache/puyallup_basemap.npz   (rgb, x, y, extent, crs)
  notebooks/data/braid_cache/nisqually_basemap.npz
  notebooks/data/braid_cache/corridor_basemap.npz

This is a NETWORK script — deliberately NOT in `make figures-from-cache`.

Usage:
  pixi run python workflows/28_fetch_basemaps.py                 # all regions
  pixi run python workflows/28_fetch_basemaps.py --region puyallup
  pixi run python workflows/28_fetch_basemaps.py --window 2025-06-01/2025-09-30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "notebooks" / "data" / "braid_cache"
sys.path.insert(0, str(ROOT / "src"))

UTM = "EPSG:32610"
# Reuse the exact AOIs/grids of workflows/19 so masks co-register; corridor is new.
REGIONS = {
    "puyallup":  dict(aoi=(-122.070, 46.895, -122.025, 46.935), crs=UTM, res=10),
    "nisqually": dict(aoi=(-121.828, 46.738, -121.792, 46.764), crs=UTM, res=10),
    # wide study-area locator (matches fig27 VIEW); lon/lat, coarse for size
    "corridor":  dict(aoi=(-122.36, 46.73, -121.78, 47.24), crs="EPSG:4326",
                      res=0.0006),
}
DEFAULT_WINDOW = "2025-06-01/2025-09-30"   # clear-sky PNW summer
MAX_CLOUD = 25

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
ES_STAC = "https://earth-search.aws.element84.com/v1"
# RGB band asset keys differ by endpoint
PC_BANDS = ["B04", "B03", "B02"]
ES_BANDS = ["red", "green", "blue"]


def _stretch(rgb: np.ndarray, lo_pct=2.0, hi_pct=98.0) -> np.ndarray:
    """Per-band percentile stretch to uint8; NaN/nodata -> black."""
    out = np.zeros(rgb.shape, dtype="uint8")
    for k in range(rgb.shape[2]):
        b = rgb[:, :, k].astype("float32")
        valid = np.isfinite(b) & (b > 0)
        if valid.sum() < 10:
            continue
        lo, hi = np.percentile(b[valid], [lo_pct, hi_pct])
        if hi <= lo:
            hi = lo + 1
        s = np.clip((b - lo) / (hi - lo), 0, 1)
        s = np.where(valid, s, 0.0)
        out[:, :, k] = (s * 255).astype("uint8")
    return out


def _load_rgb(stac_url, bands, signed, aoi, crs, res, window):
    """Cloud-masked S2 median true-color composite -> (rgb HxWx3 float, x, y)."""
    from pystac_client import Client
    from odc.stac import load as odc_load
    kw = {}
    if signed:
        import planetary_computer as pc
        kw["modifier"] = pc.sign_inplace
    cat = Client.open(stac_url, **kw)
    items = [it for it in cat.search(collections=["sentinel-2-l2a"], bbox=aoi,
                                     datetime=window).items()
             if it.properties.get("eo:cloud_cover", 100) <= MAX_CLOUD]
    if not items:
        raise RuntimeError(f"no S2 scenes < {MAX_CLOUD}% cloud for {window}")
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    items = items[:24]                      # cap scenes pulled into the composite
    scl = "SCL" if signed else "scl"
    ds = odc_load(items, bands=[*bands, scl], bbox=aoi, crs=crs, resolution=res,
                  chunks={}, groupby="solar_day")
    keep = ds[scl].isin([4, 5, 6, 7, 11])   # veg/soil/water/unclass/snow; drop cloud/shadow
    comp = []
    for bnd in bands:
        comp.append(ds[bnd].where(keep).astype("float32").median("time"))
    da = comp[0]
    # odc-stac names spatial dims 'x'/'y' for projected CRS but
    # 'longitude'/'latitude' for geographic — handle both.
    xname = "x" if "x" in da.coords else "longitude"
    yname = "y" if "y" in da.coords else "latitude"
    rgb = np.dstack([c.compute().values for c in comp])
    return (rgb, da[xname].values.astype("float64"),
            da[yname].values.astype("float64"), len(items))


def fetch_region(name: str, window: str) -> int:
    reg = REGIONS[name]
    aoi, crs, res = reg["aoi"], reg["crs"], reg["res"]
    last = None
    for url, bands, signed, tag in [(PC_STAC, PC_BANDS, True, "Planetary Computer"),
                                    (ES_STAC, ES_BANDS, False, "Earth Search")]:
        try:
            print(f"[{name}] {tag}: searching {window} (cloud<{MAX_CLOUD}%, "
                  f"crs={crs}, res={res}) …")
            rgb, x, y, n = _load_rgb(url, bands, signed, aoi, crs, res, window)
            rgb8 = _stretch(rgb)
            extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]
            # imshow origin='upper' expects row 0 = top (max y). odc returns
            # descending y already; guard in case a source returns ascending.
            if y[0] < y[-1]:
                rgb8 = rgb8[::-1]
                y = y[::-1]
            CACHE.mkdir(parents=True, exist_ok=True)
            out = CACHE / f"{name}_basemap.npz"
            np.savez_compressed(out, rgb=rgb8, x=x, y=y, extent=np.array(extent),
                                crs=np.array(crs))
            print(f"[{name}] ✓ {tag}: {n} scenes -> {rgb8.shape} uint8  {out.name} "
                  f"(extent {extent})")
            return 0
        except Exception as e:                      # noqa: BLE001
            print(f"[{name}] {tag} failed: {e}")
            last = e
    print(f"[{name}] ✗ both endpoints failed: {last}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", choices=[*REGIONS, "all"], default="all")
    ap.add_argument("--window", default=DEFAULT_WINDOW,
                    help=f"STAC datetime range (default {DEFAULT_WINDOW})")
    args = ap.parse_args()
    regions = list(REGIONS) if args.region == "all" else [args.region]
    rc = 0
    for r in regions:
        rc |= fetch_region(r, args.window)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
