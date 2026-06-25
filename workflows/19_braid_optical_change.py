#!/usr/bin/env python3
"""Optical/SAR change detection of the Puyallup braidplain, and its seismic
geometrical-spreading consequence for CC.PR01/PR02/PR03.

Motivation
----------
The manuscript (§sec-braided) argues that the *anomaly* of CC.PR01 — flattest
exponent (b=1.19), lowest correlation (r=0.88), sharpest broken-stick break, and
a coherent +0.2 log-unit baseline drift across AR1->AR3 shared by the whole
co-located cluster — is most parsimoniously explained not by a bedload onset but
by the **active braided channel migrating** across the outwash plain, i.e. a
*geometric* change in the source position relative to each sensor. That is a
falsifiable prediction: the active thread(s) nearest the cluster should have
shifted between the pre-flood (Nov 2025) and post-flood (early Jan 2026) states.

This script tests it directly with free, no-auth satellite imagery from the
**Microsoft Planetary Computer** STAC API (Sentinel-2 L2A optical and
Sentinel-1 RTC SAR), over an AOI covering both the braidplain and the three
stations:

  1. Build a cloud-masked Sentinel-2 median composite per epoch; map wetted
     channel with MNDWI=(green-SWIR)/(green+SWIR) (and NDWI as a cross-check).
  2. Build a Sentinel-1 RTC median composite per epoch; map smooth open water
     with VV backscatter (cloud-penetrating confirmation).
  3. Difference the per-epoch active-water masks -> newly-wet / newly-dry / the
     net lateral shift of the active braid field.
  4. For each station, convert the channel geometry into the manuscript's
     propagation weight  P ∝ r^-1 exp(-r/r_e)  (Tsai 2012; Gimbert 2014; the
     PNW/Rainier r_e≈780 m from workflows/09_attenuation.py):
       - nearest active-water distance r_pre, r_post  ->  Δr;
       - the attenuation-weighted "wetted illumination"
            W = Σ_pixels  A_pix · r^-1 · exp(-r/r_e)
         over the active-water field, which captures a *distributed* braided
         source rather than a single line channel;
       - the predicted log-power baseline shift Δlog10 P = log10(W_post/W_pre),
         to be compared with the observed +0.2 log-unit cross-AR drift.

Outputs
-------
  paper/figures/fig19_braid_change.png   (composites + change map + geometry)
  config/braid_optical_change.json       (per-station r, ΔW, predicted Δlog10 P)

This is the satellite test promised in the §sec-braided callout ("PlanetScope
pre/post imagery, in preparation"); Planetary Computer Sentinel-2/-1 is the
no-credential, reproducible stand-in.

Usage: pixi run python workflows/19_braid_optical_change.py
Refs: Tsai et al. 2012 (10.1029/2011GL050255); Gimbert et al. 2014
(10.1002/2014JF003201); Coppin et al. 2022 (braided-reach array); the Planetary
Computer S2-L2A and S1-RTC collections.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import planetary_computer as pc
import xarray as xr
from pystac_client import Client
from odc.stac import load as odc_load
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"
CFGDIR = ROOT / "config"
# Committed satellite-artefact cache so fig19/fig20 rebuild OFFLINE (no Planetary
# Computer): the figure only needs the derived mndwi + active-channel rasters and
# the station pixels, not the source imagery. A live run saves the cache; a
# --from-cache run replays it. (The per-station geometry + December W series live
# in config/braid_optical_change*.json; a CSV mirror of the latter is written too.)
CACHE = ROOT / "notebooks" / "data" / "braid_cache"

import sys; sys.path.insert(0, str(ROOT / "src"))
from riverseis.figstyle import paper_style  # noqa: E402
paper_style()


def _save_cache(region: str, data: dict, spx: dict) -> None:
    """Persist the fig19 rasters (mndwi+channel per epoch) + station pixels."""
    CACHE.mkdir(parents=True, exist_ok=True)
    pre, post = data["pre"], data["post"]
    arrs = dict(
        x=pre["mndwi"].x.values.astype("float64"),
        y=pre["mndwi"].y.values.astype("float64"),
        mndwi_pre=pre["mndwi"].values.astype("float32"),
        mndwi_post=post["mndwi"].values.astype("float32"),
        channel_pre=pre["channel"].values.astype("uint8"),
        channel_post=post["channel"].values.astype("uint8"),
    )
    np.savez_compressed(CACHE / f"{region}_rasters.npz", **arrs)
    spx_ser = {k: [int(v[0]), int(v[1]), float(v[2]), float(v[3])]
               for k, v in spx.items()}
    (CACHE / f"{region}_spx.json").write_text(json.dumps(spx_ser, indent=2))
    print(f"   cached satellite artefacts -> {CACHE}/{region}_rasters.npz (+_spx.json)")


def _load_cache(region: str):
    """Rebuild (data, geom, spx) for fig19 from the committed cache (no network)."""
    z = np.load(CACHE / f"{region}_rasters.npz")
    coords = {"y": z["y"], "x": z["x"]}
    def _da(a):
        return xr.DataArray(a, coords=coords, dims=("y", "x"))
    data = {"pre": dict(mndwi=_da(z["mndwi_pre"]), channel=_da(z["channel_pre"])),
            "post": dict(mndwi=_da(z["mndwi_post"]), channel=_da(z["channel_post"]))}
    spx = {k: (v[0], v[1], v[2], v[3])
           for k, v in json.loads((CACHE / f"{region}_spx.json").read_text()).items()}
    geom = json.loads((CFGDIR / JSON_NAME).read_text())["stations"]
    return data, geom, spx


def plot_december_from_json() -> None:
    """Redraw fig20 (December series) from the committed config JSON, no network."""
    cfg = json.loads((CFGDIR / JSON_NAME).read_text())
    dec = cfg.get("december_series_W")
    if not dec:
        return
    labels, anchor, rel = dec["labels"], dec["anchor"], dec["rel_to_anchor_norm"]
    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    x = np.arange(len(labels))
    for i, lab in enumerate(labels):
        if "peak" in lab.lower():
            ax.axvspan(i - 0.5, i + 0.5, color="#cfe3f2", alpha=0.7, zorder=0)
    for n, r in rel.items():
        is_anom = n == "CC.PR01"
        ax.plot(x, r, marker="o", ms=8, lw=2.6 if is_anom else 1.6,
                color="#e31a1c" if is_anom else "#888888", zorder=5 if is_anom else 3,
                label=n.split(".")[1] + (" (braidplain-central)" if is_anom else ""))
    ax.axhline(1.0, color="0.6", lw=0.9, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=11)
    ax.set_ylabel(r"illumination relative to anchor " f"{anchor.split('.')[1]}" "\n"
                  r"normalised to Nov", fontsize=11.5)
    ax.legend(title="station", fontsize=10, title_fontsize=10, loc="upper left")
    out = FIGDIR / f"fig20_braid_timeseries{FIGTAG}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (from cache)")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
# AOI: covers the braidplain seed the user flagged in Google Earth AND the three
# co-located stations, with margin. (minlon, minlat, maxlon, maxlat)
AOI = (-122.070, 46.895, -122.025, 46.935)
UTM = "EPSG:32610"          # UTM 10N — metres, for distance/area maths
RES = 10                    # m, target pixel (S2 native green/NIR resolution)

# Region registry. Each: AOI (minlon,minlat,maxlon,maxlat), station cluster
# (lon, lat, b, r), NHD river key, on-channel anchor, and whether the stations
# are co-located closely enough for the relative-to-anchor December series.
REGIONS = {
    "puyallup": dict(
        aoi=(-122.070, 46.895, -122.025, 46.935),
        stations={
            "CC.PR03": (-122.0327, 46.9034, 1.77, 0.97),   # on-channel anchor
            "CC.PR01": (-122.0376, 46.9101, 0.97, 0.78),   # braidplain anomaly
            "CC.PR02": (-122.0487, 46.9183, 1.80, 0.96),
        },
        nhd="Puyallup", anchor="CC.PR03", co_located=True, figtag="",
        json="braid_optical_change.json"),
    # UW.LON reach only (tight AOI; the full LON+GTWY box is too large for one
    # STAC load). UW.LON has the steepest scaling (b=2.24) and the most extreme
    # Q-break (1.2->4.3) — the prime candidate for a braided/anabranching reach.
    "nisqually": dict(
        aoi=(-121.828, 46.738, -121.792, 46.764),
        stations={
            "UW.LON": (-121.8096, 46.7506, 2.24, 0.91),    # steepest b; Qc-break 1.2->4.3
        },
        nhd="Nisqually", anchor="UW.LON", co_located=False, figtag="_nisqually",
        json="braid_optical_change_nisqually.json"),
}
# Module-level state (overwritten by select_region() in main()); defaults = Puyallup.
AOI = REGIONS["puyallup"]["aoi"]
STATIONS = REGIONS["puyallup"]["stations"]
NHD_KEY, ANCHOR, CO_LOCATED = "Puyallup", "CC.PR03", True
FIGTAG, JSON_NAME = "", "braid_optical_change.json"

EPOCHS = {
    "pre":  "2025-11-01/2025-11-30",   # before the Dec 2025 AR floods
    "post": "2025-12-25/2026-01-20",   # early Jan 2026, after the floods
}

S2_MAX_CLOUD = 40           # scene-level eo:cloud_cover ceiling (%) before per-pixel SCL mask
MNDWI_WATER = 0.0           # MNDWI > 0 => water (standard; turbid braids sit near 0)
S1_VV_DB_WATER = -15.0      # VV < -15 dB => smooth open water (RTC, linear->dB)
CORRIDOR_M = 500            # keep active water within this buffer of the NHD Puyallup line
MIN_COMPONENT_PX = 8        # drop connected water blobs smaller than this (speckle/ponds)

# Propagation constants — identical to workflows/09_attenuation.py for consistency
FC = (5 * 15) ** 0.5                       # 5–15 Hz band centre = 8.66 Hz
VC0, F0, XI = 1295.0, 1.0, 0.374           # Tsai 2012 Rayleigh phase-velocity dispersion
VC = VC0 * (FC / F0) ** (-XI)              # ~578 m/s at FC
Q0_PNW, ETA = 25.0, 0.5                    # PNW/Rainier Q(f)=Q0 f^eta
Q_PNW = Q0_PNW * FC ** ETA                 # ~73.5
R_E = VC * Q_PNW / (2 * np.pi * FC)        # e-folding distance ≈ 780 m

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _client():
    return Client.open(STAC, modifier=pc.sign_inplace)


def s2_water(epoch_range: str):
    """Cloud-masked Sentinel-2 median composite -> (mndwi, ndwi, water_mask, ds)."""
    cat = _client()
    items = [it for it in cat.search(collections=["sentinel-2-l2a"], bbox=AOI,
                                     datetime=epoch_range).items()
             if it.properties.get("eo:cloud_cover", 100) <= S2_MAX_CLOUD]
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    if not items:
        raise RuntimeError(f"no S2 scenes < {S2_MAX_CLOUD}% cloud for {epoch_range}")
    ds = odc_load(items, bands=["B03", "B08", "B11", "SCL"], bbox=AOI,
                  crs=UTM, resolution=RES, chunks={}, groupby="solar_day")
    # SCL classes to KEEP: 4 veg, 5 bare soil, 6 water, 7 unclassified, 11 snow/ice.
    # DROP: 0 nodata, 1 saturated, 2 dark, 3 shadow, 8/9/10 cloud, cirrus.
    keep = ds.SCL.isin([4, 5, 6, 7, 11])
    g = ds.B03.where(keep).astype("float32")
    n = ds.B08.where(keep).astype("float32")
    s = ds.B11.where(keep).astype("float32")
    mndwi = ((g - s) / (g + s)).median("time")
    ndwi = ((g - n) / (g + n)).median("time")
    water = (mndwi > MNDWI_WATER).astype("uint8")
    return mndwi.compute(), ndwi.compute(), water.compute(), len(items)


def s1_water(epoch_range: str, composite: str = "median"):
    """Sentinel-1 RTC composite -> (vv_db, water_mask, n).

    composite='median' is robust for the Nov/Jan bracket (many scenes).
    composite='min' (lowest backscatter seen in any pass) is more *sensitive* —
    needed for the short December windows where a median over 2–4 passes misses
    shallow braided water — but its absolute pixel count scales with scene count,
    so December results must use a scene-count-robust *relative* metric (see
    december_series), not absolute wetted area.
    """
    cat = _client()
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=AOI,
                            datetime=epoch_range).items())
    if not items:
        raise RuntimeError(f"no S1-RTC scenes for {epoch_range}")
    ds = odc_load(items, bands=["vv"], bbox=AOI, crs=UTM, resolution=RES,
                  chunks={}, groupby="solar_day")
    vv = ds.vv.where(ds.vv > 0)
    vv = vv.min("time") if composite == "min" else vv.median("time")
    vv_db = (10.0 * np.log10(vv)).compute()
    water = (vv_db < S1_VV_DB_WATER).astype("uint8")
    return vv_db, water, len(items)


def channel_corridor(grid: xr.DataArray, buffer_m: float = CORRIDOR_M):
    """Boolean mask of cells within `buffer_m` of the NHD Puyallup centreline,
    on the grid of `grid`. Isolates the braided mainstem from off-channel ponds."""
    import rasterio.features
    import rasterio.warp
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    paths = json.loads((CFGDIR / "nhd_rivers.json").read_text())[NHD_KEY]
    lines = []
    for path in paths:
        lon = [p[0] for p in path]
        lat = [p[1] for p in path]
        ux, uy = rasterio.warp.transform("EPSG:4326", UTM, lon, lat)
        if len(ux) > 1:
            lines.append(LineString(zip(ux, uy)))
    corridor = unary_union(lines).buffer(buffer_m)
    x = grid.x.values
    y = grid.y.values
    transform = rasterio.transform.from_bounds(
        x.min() - RES / 2, y.min() - RES / 2, x.max() + RES / 2, y.max() + RES / 2,
        len(x), len(y))
    rast = rasterio.features.rasterize(
        [(corridor, 1)], out_shape=(len(y), len(x)), transform=transform,
        fill=0, dtype="uint8")
    # rasterize assumes north-up rows top->bottom; grid.y is descending (origin upper)
    return xr.DataArray(rast, coords={"y": y, "x": x}, dims=("y", "x"))


def clean_channel(active: xr.DataArray, corridor: xr.DataArray):
    """Restrict to the corridor and drop sub-MIN_COMPONENT_PX blobs (speckle)."""
    m = (active.values == 1) & (corridor.values == 1)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = {i + 1 for i, s in enumerate(sizes) if s >= MIN_COMPONENT_PX}
        m = np.isin(lab, list(keep))
    return xr.DataArray(m.astype("uint8"), coords=active.coords, dims=active.dims)


def station_px(mask: xr.DataArray):
    """Return {station: (row, col)} in the mask's UTM grid."""
    import rasterio.warp
    xs = [lon for lon, lat, *_ in STATIONS.values()]
    ys = [lat for lon, lat, *_ in STATIONS.values()]
    ux, uy = rasterio.warp.transform("EPSG:4326", UTM, xs, ys)
    out = {}
    xcoord = mask.x.values
    ycoord = mask.y.values
    for (name, _), x, y in zip(STATIONS.items(), ux, uy):
        col = int(np.abs(xcoord - x).argmin())
        row = int(np.abs(ycoord - y).argmin())
        out[name] = (row, col, x, y)
    return out


def geometry_metrics(water_pre, water_post):
    """Per-station nearest-water distance and attenuation-weighted illumination W."""
    # metres-per-pixel grid coordinates
    X, Y = np.meshgrid(water_pre.x.values, water_pre.y.values)
    apix = RES * RES  # pixel area, m^2
    spx = station_px(water_pre)

    def nearest_dist(mask_np):
        # distance (m) from every cell to nearest water cell, via EDT on the dry field
        if mask_np.sum() == 0:
            return None
        edt = ndimage.distance_transform_edt(mask_np == 0) * RES
        return edt

    out = {}
    masks = {"pre": water_pre.values, "post": water_post.values}
    edts = {k: nearest_dist(v) for k, v in masks.items()}
    for name, (row, col, ux, uy) in spx.items():
        rec = {"utm_x": ux, "utm_y": uy}
        for k in ("pre", "post"):
            m = masks[k]
            r_near = float(edts[k][row, col]) if edts[k] is not None else None
            # attenuation-weighted wetted illumination W = Σ A r^-1 exp(-r/r_e)
            r = np.hypot(X - ux, Y - uy)
            r = np.where(r < RES, RES, r)            # floor at one pixel
            w = (m == 1) * apix / r * np.exp(-r / R_E)
            rec[f"r_near_{k}_m"] = r_near
            rec[f"W_{k}"] = float(w.sum())
            rec[f"wet_area_{k}_m2"] = float((m == 1).sum() * apix)
        rec["dr_near_m"] = (rec["r_near_post_m"] - rec["r_near_pre_m"]
                            if rec["r_near_pre_m"] is not None else None)
        rec["W_ratio_post_pre"] = (rec["W_post"] / rec["W_pre"]
                                   if rec["W_pre"] > 0 else None)
        rec["pred_dlog10P"] = (float(np.log10(rec["W_ratio_post_pre"]))
                               if rec["W_ratio_post_pre"] else None)
        out[name] = rec
    return out, spx


# Weekly-ish windows Nov->Dec 31, SAR-led: PNW optical is cloud-blind through the
# flood (0 clear Sentinel-2 scenes Dec 1–20), so the December channel-migration
# time-series can only be built from cloud-penetrating Sentinel-1.
DEC_EPOCHS = [
    ("Nov 16–30", "2025-11-16/2025-11-30", False),
    ("Dec 1–8",   "2025-12-01/2025-12-08", False),
    ("Dec 9–12 (AR peak)", "2025-12-09/2025-12-12", True),
    ("Dec 13–20", "2025-12-13/2025-12-20", False),
    ("Dec 21–31", "2025-12-21/2025-12-31", False),
]


def illumination_W(mask: xr.DataArray, spx):
    """Attenuation-weighted wetted illumination W per station for one mask."""
    X, Y = np.meshgrid(mask.x.values, mask.y.values)
    apix = RES * RES
    m = mask.values == 1
    out = {}
    for name, (row, col, ux, uy) in spx.items():
        r = np.hypot(X - ux, Y - uy)
        r = np.where(r < RES, RES, r)
        out[name] = float((m * apix / r * np.exp(-r / R_E)).sum())
    return out


def december_series(corridor, spx):
    """SAR-led weekly channel-migration series through 12/31 (the AR1->AR3 test).

    Tracks each station's wetted illumination W(t) over the flood sequence; a
    rising W at the braidplain-central station (CC.PR01) contemporaneous with the
    seismic +0.2-log baseline drift is the time-resolved avulsion signal.
    NOTE: the Dec 9–12 epoch captures peak inundation (transient flood water,
    not just the active braid) — flagged in the figure, not removed.
    """
    other = next(n for n in STATIONS if n != ANCHOR)   # a non-anchor station for the print
    print("\nDecember SAR-led channel-migration series (Sentinel-1 min-composite):")
    labels, flood, Wt = [], [], {n: [] for n in STATIONS}
    for lab, dr, is_peak in DEC_EPOCHS:
        try:
            _, s1w, n1 = s1_water(dr, composite="min")
            ch = clean_channel(s1w, corridor)
            W = illumination_W(ch, spx)
        except Exception as e:                       # noqa: BLE001
            print(f"   {lab}: skipped ({e})")
            continue
        if W[ANCHOR] <= 0:
            print(f"   {lab}: anchor illumination zero — skipped")
            continue
        labels.append(lab); flood.append(is_peak)
        for n in STATIONS:
            Wt[n].append(W[n])
        print(f"   {lab:20s} S1={n1:2d}  channel px={int(ch.values.sum()):4d}  "
              f"W({other.split('.')[1]})/W({ANCHOR.split('.')[1]})={W[other]/W[ANCHOR]:.2f}")
    if len(labels) < 3:
        print("   too few usable SAR epochs — skipping December series figure")
        return None

    # Confound-resistant metric: each station's illumination RELATIVE to the
    # on-channel anchor PR03, normalised to the first epoch. A per-epoch common
    # bias (scene count, overall wetness, flood inundation) cancels in the ratio;
    # what survives is the geometric *redistribution* between stations — a rising
    # PR01/PR03 through the ARs is the active braid migrating toward PR01.
    fig, ax = plt.subplots(figsize=(7.8, 4.4), constrained_layout=True)
    x = np.arange(len(labels))
    Wanchor = np.array(Wt[ANCHOR], float)
    for i, is_peak in enumerate(flood):
        if is_peak:
            ax.axvspan(i - 0.5, i + 0.5, color="#cfe3f2", alpha=0.7, zorder=0)
            ax.text(i, 1.02, "flood peak", ha="center", va="bottom", fontsize=9,
                    color="#2166ac", transform=ax.get_xaxis_transform())
    rel = {}
    for n in STATIONS:
        if n == ANCHOR:
            continue
        r = np.array(Wt[n], float) / Wanchor
        r = r / r[0]
        rel[n] = r.tolist()
        is_anom = (n == "CC.PR01")
        ax.plot(x, r, marker="o", ms=8, lw=2.6 if is_anom else 1.6,
                color="#e31a1c" if is_anom else "#888888",
                zorder=5 if is_anom else 3,
                label=n.split(".")[1] + (" (braidplain-central)" if is_anom else ""))
    ax.axhline(1.0, color="0.6", lw=0.9, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=11)
    ax.set_ylabel(r"illumination relative to anchor PR03" "\n"
                  r"$[\,W_i/W_{\mathrm{PR03}}\,]$, normalised to Nov", fontsize=11.5)
    ax.tick_params(labelsize=11)
    ax.legend(title="station", fontsize=10, title_fontsize=10, loc="upper left")
    out = FIGDIR / f"fig20_braid_timeseries{FIGTAG}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return {"labels": labels, "anchor": ANCHOR, "rel_to_anchor_norm": rel}


def select_region(name: str):
    """Set module-level AOI/STATIONS/NHD_KEY/ANCHOR/etc. for the chosen region."""
    global AOI, STATIONS, NHD_KEY, ANCHOR, CO_LOCATED, FIGTAG, JSON_NAME
    reg = REGIONS[name]
    AOI = reg["aoi"]; STATIONS = reg["stations"]; NHD_KEY = reg["nhd"]
    ANCHOR = reg["anchor"]; CO_LOCATED = reg["co_located"]
    FIGTAG = reg["figtag"]; JSON_NAME = reg["json"]
    print(f"region={name}  AOI={AOI}  stations={list(STATIONS)}  nhd={NHD_KEY}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="puyallup", choices=list(REGIONS))
    ap.add_argument("--from-cache", action="store_true",
                    help="rebuild fig19/fig20 from the committed npz/JSON cache (no network)")
    args = ap.parse_args()
    select_region(args.region)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    if args.from_cache:
        if not (CACHE / f"{args.region}_rasters.npz").exists():
            print(f"no cache for {args.region}; run live once to populate "
                  f"{CACHE}/{args.region}_rasters.npz")
            return 1
        data, geom, spx = _load_cache(args.region)
        make_figure(data, geom, spx)
        plot_december_from_json()
        print(f"rebuilt fig19/fig20 for {args.region} from cache (no network)")
        return 0

    print(f"r_e (5–15 Hz band centre {FC:.2f} Hz, VC={VC:.0f} m/s, Q={Q_PNW:.0f}) = {R_E:.0f} m")

    data = {}
    for ep, dr in EPOCHS.items():
        print(f"[{ep}] {dr}: querying Planetary Computer …")
        mndwi, ndwi, s2w, n2 = s2_water(dr)
        vv_db, s1w, n1 = s1_water(dr)
        # union of optical+SAR water = "active wetted channel" (reconciled, both sensors)
        # align S1 to the S2 grid (same crs/res/bbox already)
        s1w_a = s1w.reindex_like(s2w, method="nearest").fillna(0).astype("uint8")
        active = ((s2w == 1) | (s1w_a == 1)).astype("uint8")
        data[ep] = dict(mndwi=mndwi, s2w=s2w, vv_db=vv_db, s1w=s1w_a,
                        active=active, n2=n2, n1=n1)
        print(f"      S2 scenes={n2}  S1 scenes={n1}  "
              f"S2-water px={int(s2w.values.sum())}  S1-water px={int(s1w_a.values.sum())}  "
              f"active px={int(active.values.sum())}")

    # restrict to the braided mainstem corridor + drop speckle, then measure geometry
    corridor = channel_corridor(data["pre"]["active"])
    for ep in EPOCHS:
        data[ep]["channel"] = clean_channel(data[ep]["active"], corridor)
        print(f"      [{ep}] corridor-cleaned channel px = "
              f"{int(data[ep]['channel'].values.sum())}")
    geom, spx = geometry_metrics(data["pre"]["channel"], data["post"]["channel"])

    # Threshold ENSEMBLE — turbid braided water sits near MNDWI 0, so the wetted
    # area (and W) is threshold-sensitive. Sweep MNDWI to get an honest spread on
    # each station's predicted drift rather than a single brittle point estimate.
    THRS = (-0.10, -0.05, 0.0, 0.05, 0.10)
    ens = {name: {"dr_near_m": [], "pred_dlog10P": []} for name in STATIONS}
    for thr in THRS:
        ch = {}
        for ep in EPOCHS:
            a = (data[ep]["mndwi"] > thr).astype("uint8")
            a = ((a == 1) | (data[ep]["s1w"] == 1)).astype("uint8")
            ch[ep] = clean_channel(a, corridor)
        g, _ = geometry_metrics(ch["pre"], ch["post"])
        for name in STATIONS:
            ens[name]["dr_near_m"].append(g[name]["dr_near_m"])
            ens[name]["pred_dlog10P"].append(g[name]["pred_dlog10P"])
    for name in STATIONS:
        for k in ("dr_near_m", "pred_dlog10P"):
            v = np.array(ens[name][k], float)
            geom[name][f"{k}_median"] = float(np.median(v))
            geom[name][f"{k}_min"] = float(np.min(v))
            geom[name][f"{k}_max"] = float(np.max(v))
    print(f"\nThreshold ensemble (MNDWI {THRS[0]:+.2f}..{THRS[-1]:+.2f}):")
    for name in STATIONS:
        d = geom[name]
        print(f"   {name}: Δr median {d['dr_near_m_median']:+5.0f} m "
              f"[{d['dr_near_m_min']:+.0f},{d['dr_near_m_max']:+.0f}]   "
              f"pred Δlog10P median {d['pred_dlog10P_median']:+.3f} "
              f"[{d['pred_dlog10P_min']:+.3f},{d['pred_dlog10P_max']:+.3f}]")

    # ---- report ----
    print(f"\n{'station':9s} {'r_pre':>7s} {'r_post':>7s} {'Δr':>7s} "
          f"{'W_post/W_pre':>12s} {'pred Δlog10P':>12s}   (obs b, r)")
    for name, rec in geom.items():
        _, _, b, r = STATIONS[name]
        print(f"{name:9s} {rec['r_near_pre_m']:7.0f} {rec['r_near_post_m']:7.0f} "
              f"{rec['dr_near_m']:7.0f} {rec['W_ratio_post_pre']:12.3f} "
              f"{rec['pred_dlog10P']:+12.3f}   (b={b}, r={r})")

    # Relative-to-anchor December series only for a co-located cluster (Puyallup).
    dec = december_series(corridor, spx) if CO_LOCATED else None

    out = {
        "region": NHD_KEY, "aoi": AOI, "epochs": EPOCHS, "crs": UTM, "res_m": RES,
        "december_series_W": dec,
        "r_e_m": R_E, "fc_hz": FC, "vc_ms": VC, "Q_pnw": Q_PNW,
        "thresholds": {"mndwi_water": MNDWI_WATER, "s1_vv_db_water": S1_VV_DB_WATER,
                       "s2_max_cloud_pct": S2_MAX_CLOUD},
        "observed_baseline_drift_log10": 0.2,   # §sec-braided: AR1->AR3, ~1.6x
        "scenes": {ep: {"s2": data[ep]["n2"], "s1": data[ep]["n1"]} for ep in EPOCHS},
        "stations": geom,
    }
    (CFGDIR / JSON_NAME).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {CFGDIR/JSON_NAME}")

    # CSV mirror of the (tabular) December illumination series for offline rebuild
    if dec:
        CACHE.mkdir(parents=True, exist_ok=True)
        import csv
        with (CACHE / f"{args.region}_december_series.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch_label", *[f"{n.split('.')[1]}_rel_to_{dec['anchor'].split('.')[1]}"
                                         for n in dec["rel_to_anchor_norm"]]])
            for i, lab in enumerate(dec["labels"]):
                w.writerow([lab, *[f"{dec['rel_to_anchor_norm'][n][i]:.4f}"
                                   for n in dec["rel_to_anchor_norm"]]])

    make_figure(data, geom, spx)
    _save_cache(args.region, data, spx)        # persist artefacts for offline rebuild
    return 0


def make_figure(data, geom, spx):
    pre, post = data["pre"], data["post"]
    ext = [float(pre["channel"].x.min()), float(pre["channel"].x.max()),
           float(pre["channel"].y.min()), float(pre["channel"].y.max())]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6), constrained_layout=True)

    def plot_stations(a):
        for name, (row, col, ux, uy) in spx.items():
            a.plot(ux, uy, "^", ms=11, mfc="yellow", mec="k", mew=1.3, zorder=5)
            a.annotate(name.split(".")[1], (ux, uy), color="k", fontsize=8,
                       xytext=(6, 4), textcoords="offset points", fontweight="bold")

    # (a) MNDWI pre vs water outlines
    im0 = ax[0].imshow(pre["mndwi"].values, extent=ext, origin="upper",
                       cmap="BrBG", vmin=-0.6, vmax=0.6)
    ax[0].contour(post["channel"].values, levels=[0.5], extent=ext, origin="upper",
                  colors="red", linewidths=0.7)
    ax[0].set_title("(a) Pre (Nov 2025) MNDWI\nred = post active-channel outline")
    plot_stations(ax[0])
    fig.colorbar(im0, ax=ax[0], shrink=0.8, label="MNDWI")

    # (b) change map: newly-wet / newly-dry / persistent (corridor-cleaned channel)
    pre_a = pre["channel"].values.astype(int)
    post_a = post["channel"].values.astype(int)
    chg = np.zeros_like(pre_a)
    chg[(pre_a == 1) & (post_a == 1)] = 1   # persistent
    chg[(pre_a == 0) & (post_a == 1)] = 2   # newly wet
    chg[(pre_a == 1) & (post_a == 0)] = 3   # newly dry
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f7f7f7", "#3690c0", "#e31a1c", "#fdb863"])
    ax[1].imshow(chg, extent=ext, origin="upper", cmap=cmap, vmin=-0.5, vmax=3.5)
    ax[1].set_title("(b) Active-channel change (S2∪S1)\n"
                    "blue=persistent · red=newly wet · orange=newly dry")
    plot_stations(ax[1])

    # (c) predicted geometric baseline drift per station — ensemble median + spread
    names = list(geom)
    x = np.arange(len(names))
    med = [geom[n]["pred_dlog10P_median"] for n in names]
    lo = [geom[n]["pred_dlog10P_median"] - geom[n]["pred_dlog10P_min"] for n in names]
    hi = [geom[n]["pred_dlog10P_max"] - geom[n]["pred_dlog10P_median"] for n in names]
    # highlight the braidplain-anomaly station in red (PR01 on the Puyallup);
    # for a single-station region (Nisqually/UW.LON) just use one colour.
    hilite = "CC.PR01" if "CC.PR01" in names else (names[0] if len(names) == 1 else None)
    colors = ["#e31a1c" if n == hilite else "#08519c" for n in names]
    ax[2].bar(x, med, 0.55, yerr=[lo, hi], capsize=5, color=colors,
              error_kw=dict(ecolor="0.3", lw=1.3))
    ax[2].axhline(0.2, ls="--", color="green", lw=1.2)
    ax[2].annotate("observed cross-AR drift +0.2", (len(names) - 1, 0.2),
                   color="green", fontsize=8, ha="right", va="bottom")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([n.split(".")[1] for n in names])
    ax[2].set_ylabel(r"predicted $\Delta\log_{10}P=\log_{10}(W_{post}/W_{pre})$")
    ax[2].set_title("(c) Predicted geometric baseline drift", fontsize=10)
    ax[2].text(0.5, -0.20, r"$W=\sum A\,r^{-1}e^{-r/r_e}$, "
               + f"$r_e$={R_E:.0f} m; bars = MNDWI-threshold ensemble (median, range)",
               transform=ax[2].transAxes, ha="center", fontsize=8, color="0.3")

    out = FIGDIR / f"fig19_braid_change{FIGTAG}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
