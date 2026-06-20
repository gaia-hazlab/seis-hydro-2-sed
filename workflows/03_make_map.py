#!/usr/bin/env python3
"""Publication map (fig 1) of the Puyallup mountain-to-sea seismic transect.

PyGMT: hillshaded high-resolution DEM (SRTM earth_relief) + GMT rivers/coast +
seismic stations (broadband vs accelerometer) + USGS gages + Mt. Rainier.

Usage: pixi run python workflows/03_make_map.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pygmt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures" / "fig1_transect_map.png"
DISC = ROOT / "config" / "_transect_discovery.json"

REGION = [-122.55, -121.55, 46.72, 47.34]   # W, E, S, N
DEM = "@earth_relief_03s"                    # ~90 m SRTM

# USGS gages along the corridor (id -> name, lon, lat)
GAGES = {
    "12092000": ("Puyallup nr Electron", -122.0351, 46.9037),
    "12094000": ("Carbon nr Fairfax",    -122.0326, 47.0279),
    "12093500": ("Puyallup nr Orting",   -122.2079, 47.0392),
    "12096500": ("Puyallup at Alderton", -122.2296, 47.1851),
    "12101500": ("Puyallup at Puyallup", -122.3271, 47.2084),
    "12082500": ("Nisqually nr National", -122.0828, 46.7558),
}
RAINIER = (-121.7603, 46.8523)
# Stations to label (the analysis transect)
LABEL = {"CC.PR01", "CC.PR02", "CC.PR03", "CC.SIFT", "CC.STYX", "CC.TRON", "UW.TEHA"}


def main() -> int:
    stations = json.loads(DISC.read_text())["stations"] if DISC.exists() else []

    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE="14p,Helvetica-Bold", MAP_FRAME_TYPE="plain")

    # Grayscale relief that emphasizes TOPOGRAPHY (elevation tone) over fine
    # texture: smooth the DEM with a ~1.3 km Gaussian to suppress lowland SRTM
    # speckle, map elevation with a light gray ramp over the actual relief range
    # (more tonal contrast), and add only a gentle hillshade for form.
    grid = pygmt.datasets.load_earth_relief(resolution="03s", region=REGION)
    grid = pygmt.grdfilter(grid=grid, filter="g1.3", distance="4")
    shade = pygmt.grdgradient(grid=grid, azimuth=300, normalize="t0.4")
    elev_cpt = ROOT / "config" / "_elev.cpt"
    pygmt.makecpt(cmap="gray", series=[-300, 2600], reverse=True, output=str(elev_cpt))
    fig.grdimage(grid=grid, shading=shade, projection="M16c", region=REGION,
                 frame=["WSne", "xa0.2", "ya0.2"], cmap=str(elev_cpt))

    # Puget Sound (coast water + shoreline); rivers come from real NHD geometry below
    fig.coast(resolution="f", water="170/200/230", shorelines="0.4p,40/40/60")

    # true river geometry from USGS NLDI / NHD flowlines
    rivers = json.loads((ROOT / "config" / "nhd_rivers.json").read_text())
    for segs in rivers.values():
        xs, ys = [], []
        for seg in segs:
            for pt in seg:
                xs.append(pt[0]); ys.append(pt[1])
            xs.append(np.nan); ys.append(np.nan)
        if xs:
            fig.plot(x=np.array(xs), y=np.array(ys), pen="1.0p,30/90/200")

    # OPERA DSWx-S1 surface-water (flood) extent, 10 Dec 2025 — semi-transparent blue
    water = ROOT / "notebooks" / "data" / "opera" / "water_dec10.grd"
    if water.exists():
        cpt = ROOT / "config" / "_water.cpt"
        cpt.write_text("0\t30/110/230\t2\t30/110/230\nN\t-\n")
        fig.grdimage(grid=str(water), cmap=str(cpt), nan_transparent=True, transparency=35)

    # Mt. Rainier (no text box)
    fig.plot(x=[RAINIER[0]], y=[RAINIER[1]], style="kvolcano/0.55c", fill="firebrick", pen="0.8p,black")
    fig.text(x=RAINIER[0], y=RAINIER[1] - 0.045, text="Mt. Rainier", font="9p,Helvetica-Bold,black",
             justify="TC")

    # USGS gages (no text box)
    gx = [v[1] for v in GAGES.values()]
    gy = [v[2] for v in GAGES.values()]
    fig.plot(x=gx, y=gy, style="d0.34c", fill="cyan", pen="0.7p,black")
    for gid, (nm, lon, lat) in GAGES.items():
        fig.text(x=lon, y=lat, text=nm.split(" nr")[0].split(" at")[0],
                 font="6.5p,Helvetica,30/30/30", offset="0.22c/0.0c", justify="ML")

    # seismic stations: broadband (triangles) vs accelerometer (squares); no text box
    for v in stations:
        sid = f'{v["net"]}.{v["sta"]}'
        bb = v["broadband"]
        emph = sid in LABEL
        fig.plot(x=[v["lon"]], y=[v["lat"]],
                 style=("t0.42c" if bb else "s0.34c"),
                 fill=("red" if bb else "gray60"),
                 pen=("1.0p,black" if emph else "0.4p,black"))
        if emph:
            fig.text(x=v["lon"], y=v["lat"], text=sid, font="7p,Helvetica-Bold,black",
                     offset="0.0c/0.30c", justify="BC")

    # legend (manual via plot + text)
    fig.legend(spec=_legend(), position="jBL+o0.2c", box="+gwhite@20+p0.5p")
    fig.colorbar(cmap=str(elev_cpt), frame=["x+lelevation", "y+lm"], position="JMR+o0.6c/0c+w8c")
    fig.savefig(FIG, dpi=300)
    print(f"wrote {FIG}")
    return 0


def _legend() -> str:
    # GMT legend spec file content
    lines = [
        "G 0.05c",
        "S 0.25c t 0.42c red 1p,black 0.7c broadband station (BH/EH)",
        "S 0.25c s 0.34c gray60 0.4p,black 0.7c strong-motion (HN/EN)",
        "S 0.25c d 0.34c cyan 0.7p,black 0.7c USGS discharge gage",
        "S 0.25c kvolcano 0.5c firebrick 0.8p,black 0.7c Mt. Rainier",
        "S 0.25c r 0.4c 30/110/230@35 - 0.7c flood/water (OPERA DSWx, 10 Dec 2025)",
        "G 0.05c",
    ]
    p = ROOT / "config" / "_map_legend.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


if __name__ == "__main__":
    raise SystemExit(main())
