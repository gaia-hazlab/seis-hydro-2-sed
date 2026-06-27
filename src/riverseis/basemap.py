"""Shared Sentinel-2 true-color basemap loader for the geospatial figures.

The braidplain/zoom panels and the wide corridor locator all underlay the same
cached Sentinel-2 true-color composite so the figures look consistent and stay
*offline-reproducible*. The RGB arrays are built once (network) by
``workflows/28_fetch_basemaps.py`` and committed to ``notebooks/data/braid_cache``;
here we only load and draw them.

Two coordinate conventions, deliberately:
  * the braidplain regions (``puyallup``, ``nisqually``) are cached on the **UTM
    10N (EPSG:32610)** grid — the *same* grid as the active-channel masks in
    ``{region}_rasters.npz`` — so a mask overlays the basemap pixel-for-pixel with
    no reprojection;
  * the ``corridor`` locator is cached in **lon/lat (EPSG:4326)** at coarse
    resolution, so station/gage/river overlays (which are lon/lat) drop straight on.

The basemap is *illustrative landscape context*; the channel masks remain the
quantitative layer (state this in captions).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parents[2] / "notebooks" / "data" / "braid_cache"


def basemap_path(region: str) -> Path:
    return CACHE / f"{region}_basemap.npz"


def has_basemap(region: str) -> bool:
    return basemap_path(region).exists()


def load_basemap(region: str) -> dict:
    """Return {rgb (H,W,3 uint8), extent [l,r,b,t], x, y, crs} for a region.

    ``extent`` is ready for ``imshow(..., extent=extent, origin='upper')``.
    Raises FileNotFoundError with a hint if the cache has not been built yet.
    """
    p = basemap_path(region)
    if not p.exists():
        raise FileNotFoundError(
            f"no basemap cache for '{region}'. Build it once (network):\n"
            f"    pixi run python workflows/28_fetch_basemaps.py --region {region}")
    z = np.load(p)
    return {
        "rgb": z["rgb"], "extent": [float(v) for v in z["extent"]],
        "x": z["x"], "y": z["y"], "crs": str(z["crs"]) if "crs" in z else "EPSG:32610",
    }


def imshow_basemap(ax, region: str, *, alpha: float = 1.0, zorder: int = 0):
    """Draw the cached true-color basemap on ``ax`` and return its extent.

    Sets the axis limits to the basemap extent. Overlay masks/markers afterwards
    with higher zorder.
    """
    bm = load_basemap(region)
    ax.imshow(bm["rgb"], extent=bm["extent"], origin="upper", alpha=alpha,
              zorder=zorder, interpolation="nearest")
    ax.set_xlim(bm["extent"][0], bm["extent"][1])
    ax.set_ylim(bm["extent"][2], bm["extent"][3])
    return bm["extent"]
