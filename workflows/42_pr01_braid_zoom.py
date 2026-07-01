#!/usr/bin/env python3
r"""PR01 braidplain zoom — traced active-channel boundaries, pre vs post AR (issue #10).

Montgomery's review ask (M3): "trace the boundary of the river channels/banks to
demonstrate the multi-braided system … zoom WAY in around PR01." The hero geomorphic
figure is a zoom of the CC.PR01 braided source reach with the active-channel/bank
boundaries traced before and after the December-2025 floods.

Two resolution tiers, same figure builder:
  * **Default (offline placeholder):** 10-m Sentinel-2 from the committed braid cache.
    At 10 m the pre-flood threads are ~1 pixel wide — individual braids and bank lines
    are sub-resolution, so this honestly shows only the pre→post change in wetted
    active-channel extent.
  * **Upgrade (``--pre-raster``/``--post-raster``):** ~3-m PlanetScope 4-band
    surface-reflectance GeoTIFFs (fetch with scripts/fetch_planetscope_pr01.py) — or any
    georeferenced 4-band raster. Water is traced from NDWI = (Green−NIR)/(Green+NIR);
    the RGB basemap and outlines are then at native resolution, resolving the threads.
    A repeat-lidar DEM-of-Difference [@anderson2025] would be the ideal further upgrade.

**Anderson (2026) framing (both tiers).** CC.PR01 sits in the *upper* Puyallup, a
**net-erosional sediment source** (bank/bluff erosion, persistent reorganization) that
feeds the aggrading lowland downstream — so the reorganization imaged here is lateral
thread-switching and bank erosion, not local vertical accretion.

PlanetScope imagery is licensed (not committed); the derived figure is publishable with
attribution "© Planet Labs PBC".

Outputs paper/figures/fig33_pr01_braid_zoom.png.

Usage:
  pixi run python workflows/42_pr01_braid_zoom.py            # 10 m cache (offline)
  pixi run python workflows/42_pr01_braid_zoom.py \
      --pre-raster notebooks/data/planet_cache/pre/X.tif \
      --post-raster notebooks/data/planet_cache/post/Y.tif  # ~3 m PlanetScope
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402

CACHE = ROOT / "notebooks" / "data" / "braid_cache"
FIGDIR = ROOT / "paper" / "figures"
STATION = "CC.PR01"
PR01_UTM = (573289.42, 5195623.45)               # EPSG:32610
SCALEBAR_M = 200.0


def _stretch(a, lo=2, hi=98):
    a = a.astype(float)
    p0, p1 = np.nanpercentile(a, [lo, hi])
    return np.clip((a - p0) / max(p1 - p0, 1e-9), 0, 1)


def load_from_cache(half):
    """10-m Sentinel-2: (xs, ys, rgb, pre_mask, post_mask, station_xy)."""
    base = np.load(CACHE / "puyallup_basemap.npz")
    ras = np.load(CACHE / "puyallup_rasters.npz")
    spx = json.loads((CACHE / "puyallup_spx.json").read_text())
    x, y = base["x"], base["y"]
    cx, cy = spx[STATION][2], spx[STATION][3]
    mx = (x >= cx - half) & (x <= cx + half)
    my = (y >= cy - half) & (y <= cy + half)
    return (x[mx], y[my], base["rgb"][np.ix_(my, mx)],
            ras["channel_pre"][np.ix_(my, mx)] > 0,
            ras["channel_post"][np.ix_(my, mx)] > 0, (cx, cy))


def load_from_rasters(pre_path, post_path, half, ndwi_thresh):
    """~3-m PlanetScope 4-band SR (B,G,R,NIR): reproject to UTM 10N, align, crop to the
    PR01 window; water from NDWI. Returns (xs, ys, rgb, pre_mask, post_mask, station_xy)."""
    import rioxarray  # noqa: F401
    import xarray as xr  # noqa: F401
    import rioxarray as rxr

    cx, cy = PR01_UTM
    pre = rxr.open_rasterio(pre_path, masked=True).rio.reproject("EPSG:32610")
    post = rxr.open_rasterio(post_path, masked=True).rio.reproject_match(pre)
    box = (cx - half, cy - half, cx + half, cy + half)
    pre = pre.rio.clip_box(*box)
    post = post.rio.clip_box(*box)
    xs = pre.x.values
    ys = pre.y.values

    def ndwi(da):
        g = da.isel(band=1).values.astype(float)     # Green
        nir = da.isel(band=3).values.astype(float)   # NIR
        with np.errstate(invalid="ignore", divide="ignore"):
            n = (g - nir) / (g + nir)
        valid = np.isfinite(n) & (da.isel(band=0).values > 0)
        return (n > ndwi_thresh) & valid

    pre_mask = ndwi(pre)
    post_mask = ndwi(post)
    # RGB basemap from the post scene (R=band3,G=band2,B=band1 -> 0-indexed 2,1,0)
    b = post.values.astype(float)
    rgb = np.dstack([_stretch(b[2]), _stretch(b[1]), _stretch(b[0])])
    rgb = (np.nan_to_num(rgb) * 255).astype("uint8")
    return xs, ys, rgb, pre_mask, post_mask, (cx, cy)


def fetch_naip(half):
    """Download a NAIP (USGS aerial, ~1 m, public domain) 4-band clip over the PR01
    window from the public ImageServer (no auth). Cached, gitignored."""
    import requests
    dest = CACHE / "naip_pr01.tif"                   # braid_cache (public domain; committed)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    cx, cy = PR01_UTM
    H = half + 50
    px = int(2 * H)                                  # ~1 m pixels
    url = ("https://imagery.nationalmap.gov/arcgis/rest/services/"
           "USGSNAIPImagery/ImageServer/exportImage")
    r = requests.get(url, params=dict(bbox=f"{cx-H},{cy-H},{cx+H},{cy+H}", bboxSR=32610,
                     imageSR=32610, size=f"{px},{px}", format="tiff", pixelType="U8",
                     f="image"), timeout=180)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    print(f"   cached NAIP clip -> {dest} ({dest.stat().st_size//1024} KB)")
    return dest


def _nearest_idx(axis, vals):
    """Nearest-neighbour index of each val into axis (which may be ascending or not)."""
    order = np.argsort(axis)
    a = axis[order]
    j = np.clip(np.searchsorted(a, vals), 1, len(a) - 1)
    j = np.where(np.abs(vals - a[j - 1]) <= np.abs(vals - a[j]), j - 1, j)
    return order[j]


def load_naip(half):
    """NAIP ~1-m RGB basemap (structure) + the 10-m Sentinel-2 pre/post change resampled
    onto the NAIP grid. Best free, no-approval high-res option while PlanetScope E&R
    entitlement is pending."""
    import rioxarray as rxr
    cx, cy = PR01_UTM
    da = rxr.open_rasterio(fetch_naip(half), masked=True).rio.reproject("EPSG:32610")
    da = da.rio.clip_box(cx - half, cy - half, cx + half, cy + half)
    xs, ys = da.x.values, da.y.values
    b = da.values.astype(float)                      # NAIP band order: R,G,B,NIR
    rgb = (np.dstack([_stretch(b[0]), _stretch(b[1]), _stretch(b[2])]) * 255).astype("uint8")
    rgb = np.nan_to_num(rgb).astype("uint8")
    ras = np.load(CACHE / "puyallup_rasters.npz")
    bx = np.load(CACHE / "puyallup_basemap.npz")
    ixm = _nearest_idx(bx["x"], xs)
    iym = _nearest_idx(bx["y"], ys)
    pre = (ras["channel_pre"] > 0)[np.ix_(iym, ixm)]
    post = (ras["channel_post"] > 0)[np.ix_(iym, ixm)]
    return xs, ys, rgb, pre, post, (cx, cy)


def _scalebar(ax, ext):
    x0 = ext[0] + 0.06 * (ext[1] - ext[0])
    y0 = ext[2] + 0.08 * (ext[3] - ext[2])
    ax.add_patch(Rectangle((x0, y0), SCALEBAR_M, 0.012 * (ext[3] - ext[2]),
                           fc="white", ec="k", lw=0.8, zorder=7))
    ax.text(x0 + SCALEBAR_M / 2, y0 + 0.03 * (ext[3] - ext[2]), f"{SCALEBAR_M:.0f} m",
            color="white", fontsize=8, ha="center", va="bottom", fontweight="bold", zorder=7)


def render(xs, ys, rgb, pre, post, station_xy, *, res_m, source, placeholder, tag=None):
    paper_style()
    cx, cy = station_xy
    ext = [xs.min(), xs.max(), ys.min(), ys.max()]
    XX, YY = np.meshgrid(xs, ys)
    pre = pre.astype(float); post = post.astype(float)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.4, 6.3))

    axa.imshow(rgb, extent=ext, origin="upper", interpolation="nearest")
    if pre.any():
        axa.contour(XX, YY, pre, levels=[0.5], colors="#00e5ff", linewidths=1.6)
    if post.any():
        axa.contour(XX, YY, post, levels=[0.5], colors="#ff8c1a", linewidths=1.6)
    axa.plot(cx, cy, marker="^", ms=13, mfc="yellow", mec="k", mew=1.2, zorder=6)
    axa.annotate(" PR01", (cx, cy), color="yellow", fontweight="bold", fontsize=10,
                 va="center", ha="left")
    axa.set_title("Active-channel outline: pre (Nov) vs post (Dec–Jan)", fontsize=10.5, loc="left")
    _scalebar(axa, ext)
    axa.legend(handles=[Line2D([0], [0], color="#00e5ff", lw=1.8, label="pre-flood (Nov)"),
                        Line2D([0], [0], color="#ff8c1a", lw=1.8, label="post/peak (Dec–Jan)")],
               loc="lower right", fontsize=8, framealpha=0.9)
    axa.set_xticks([]); axa.set_yticks([]); axa.set_xlim(ext[0], ext[1]); axa.set_ylim(ext[2], ext[3])

    faded = (0.55 * rgb.astype(float) + 0.45 * 255).astype("uint8")
    axb.imshow(faded, extent=ext, origin="upper", interpolation="nearest")
    persistent = (pre > 0) & (post > 0)
    newly_wet = (post > 0) & (pre == 0)
    newly_dry = (pre > 0) & (post == 0)
    overlay = np.zeros((*pre.shape, 4))
    overlay[persistent] = (0.10, 0.45, 0.90, 0.95)
    overlay[newly_wet] = (0.90, 0.10, 0.15, 0.95)
    overlay[newly_dry] = (1.00, 0.60, 0.00, 0.95)
    axb.imshow(overlay, extent=ext, origin="upper", interpolation="nearest")
    axb.plot(cx, cy, marker="^", ms=13, mfc="yellow", mec="k", mew=1.2, zorder=6)
    axb.set_title("Reorganization: thread capture & abandonment", fontsize=10.5, loc="left")
    _scalebar(axb, ext)
    axb.legend(handles=[Line2D([0], [0], marker="s", ls="", mfc="#1a73e6", mec="none", label="persistent"),
                        Line2D([0], [0], marker="s", ls="", mfc="#e61a26", mec="none", label="newly wet"),
                        Line2D([0], [0], marker="s", ls="", mfc="#ff9900", mec="none", label="newly dry (abandoned)")],
               loc="lower right", fontsize=8, framealpha=0.9)
    axb.set_xticks([]); axb.set_yticks([]); axb.set_xlim(ext[0], ext[1]); axb.set_ylim(ext[2], ext[3])

    if tag is None:
        tag = (f"PLACEHOLDER: {source} — threads sub-pixel; ≈3 m / lidar-DoD needed to trace threads & banks"
               if placeholder else f"{source} (~{res_m:g} m) — © Planet Labs PBC")
    fig.suptitle("CC.PR01 braided source reach (upper Puyallup) — thread-switching & bank erosion in a "
                 f"net-erosional\nsource reach [Anderson 2026].  {tag}", fontsize=10.5, x=0.5, y=0.99)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.06)
    out = FIGDIR / "fig33_pr01_braid_zoom.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (pre wet px={int((pre>0).sum())}, post wet px={int((post>0).sum())}, "
          f"newly-wet {int(newly_wet.sum())}, newly-dry {int(newly_dry.sum())})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--naip", action="store_true",
                    help="NAIP ~1 m basemap (free, no auth) + Sentinel-2 change overlay")
    ap.add_argument("--pre-raster", type=Path, help="pre-flood 4-band SR GeoTIFF (PlanetScope)")
    ap.add_argument("--post-raster", type=Path, help="post-flood 4-band SR GeoTIFF (PlanetScope)")
    ap.add_argument("--half", type=float, default=550.0, help="half-window (m) around PR01")
    ap.add_argument("--ndwi-thresh", type=float, default=-0.02, help="NDWI water threshold (hi-res path)")
    args = ap.parse_args()

    if args.naip:
        xs, ys, rgb, pre, post, st = load_naip(args.half)
        return render(xs, ys, rgb, pre, post, st, res_m=1, source="NAIP", placeholder=False,
                      tag="NAIP 1 m aerial (USGS, public domain) — structure; Sentinel-2 10 m — "
                          "Dec-2025 change. PlanetScope/lidar event-timed upgrade pending Planet E&R approval")

    if args.pre_raster and args.post_raster:
        xs, ys, rgb, pre, post, st = load_from_rasters(
            args.pre_raster, args.post_raster, args.half, args.ndwi_thresh)
        return render(xs, ys, rgb, pre, post, st, res_m=3,
                      source="PlanetScope", placeholder=False)
    if args.pre_raster or args.post_raster:
        print("provide BOTH --pre-raster and --post-raster, or neither (cache mode)")
        return 2
    xs, ys, rgb, pre, post, st = load_from_cache(args.half)
    return render(xs, ys, rgb, pre, post, st, res_m=10,
                  source="10 m Sentinel-2", placeholder=True)


if __name__ == "__main__":
    raise SystemExit(main())
