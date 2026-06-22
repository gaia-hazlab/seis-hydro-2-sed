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

REGION = [-122.75, -121.55, 46.68, 47.40]   # W, E, S, N (western Rainier drainages)
DEM = "@earth_relief_03s"                    # ~90 m SRTM
RAINIER = (-121.7603, 46.8523)
# CC stations to label (the analysis transect + a few notable)
LABEL = {"CC.PR03", "CC.SIFT", "CC.STYX", "CC.TRON", "CC.CARB", "CC.PARA", "CC.GTWY"}
# key gages to label (rest are markers only, to avoid clutter)
GAGE_LABEL = {"12092000", "12093500", "12096500", "12101500", "12094000", "12082500"}
# sample-rate -> color (shows the CC network's increasing sampling rate)
SR_COLOR = {500: "200/30/30", 100: "230/140/0", 50: "120/120/120"}


def main() -> int:
    cc = json.loads((ROOT / "config" / "cc_stations.json").read_text())
    layers = json.loads((ROOT / "config" / "map_layers.json").read_text())
    gages, snotel = layers["gages"], layers["snotel"]
    # station signal classification (where river seismology worked / didn't)
    status = {}
    sp = ROOT / "config" / "station_status.json"
    if sp.exists():
        status = {s["station"]: s for s in json.loads(sp.read_text())}
    NOSIG = {"none", "control"}

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

    # all USGS discharge gages (cyan diamonds); label only the corridor mainstem
    fig.plot(x=[g["lon"] for g in gages], y=[g["lat"] for g in gages],
             style="d0.30c", fill="cyan", pen="0.6p,black")
    for g in gages:
        if g["id"] in GAGE_LABEL:
            fig.text(x=g["lon"], y=g["lat"], text=g["name"].split(" nr")[0].split(" at")[0].title(),
                     font="6p,Helvetica,20/20/20", offset="0.20c/0.0c", justify="ML")

    # SNOTEL / met stations (blue stars)
    fig.plot(x=[s["lon"] for s in snotel], y=[s["lat"] for s in snotel],
             style="a0.34c", fill="dodgerblue4", pen="0.6p,white")
    for s in snotel:
        fig.text(x=s["lon"], y=s["lat"], text=s["name"], font="6p,Helvetica-Bold,dodgerblue4",
                 offset="0.0c/-0.22c", justify="TC")

    # Seismic stations: triangle colored by SAMPLE RATE (network upgrading 50->500 sps).
    # Stations where river seismology was attempted but gave NO signal (or are
    # out-of-drainage/traffic controls) are drawn as HOLLOW triangles with a thick
    # outline, to document where it was not possible/observed.
    seis = [(f'CC.{v["sta"]}', v["lon"], v["lat"], int(v["sr"])) for v in cc]
    uw = json.loads((ROOT / "config" / "uw_stations.json").read_text())
    seen = {s[0] for s in seis}
    for v in uw:                      # add analyzed UW stations not already shown
        sid = f'UW.{v["sta"]}'
        if sid in status and sid not in seen:
            seis.append((sid, v["lon"], v["lat"], int(v["sr"])))
    for sid, lon, lat, sr in seis:
        st = status.get(sid, {}).get("status")
        nosig = st in NOSIG
        fig.plot(x=[lon], y=[lat], style="t0.42c",
                 fill=("white" if nosig else SR_COLOR.get(sr, "120/120/120")),
                 pen=("1.6p,black" if nosig else ("1.1p,black" if sid in LABEL else "0.4p,black")))
        if sid in LABEL or nosig and sid in status:
            fig.text(x=lon, y=lat, text=sid, font="7p,Helvetica-Bold,black",
                     offset="0.0c/0.30c", justify="BC")

    fig.legend(spec=_legend(), position="jBL+o0.2c", box="+gwhite@15+p0.5p")
    fig.colorbar(cmap=str(elev_cpt), frame=["x+lelevation", "y+lm"], position="JMR+o0.6c/0c+w8c")
    fig.savefig(FIG, dpi=300)
    print(f"wrote {FIG}")
    return 0


def _legend() -> str:
    # GMT legend spec file content
    lines = [
        "G 0.05c",
        "S 0.25c t 0.40c 200/30/30 0.4p,black 0.7c CC station, 500 sps",
        "S 0.25c t 0.40c 230/140/0 0.4p,black 0.7c CC station, 100 sps",
        "S 0.25c t 0.40c 120/120/120 0.4p,black 0.7c CC station, 50 sps",
        "S 0.25c t 0.42c white 1.6p,black 0.7c attempted, no river signal",
        "S 0.25c d 0.30c cyan 0.6p,black 0.7c USGS discharge gage",
        "S 0.25c a 0.34c dodgerblue4 0.6p,white 0.7c SNOTEL / met station",
        "S 0.25c kvolcano 0.5c firebrick 0.8p,black 0.7c Mt. Rainier",
        "S 0.25c r 0.4c 30/110/230@35 - 0.7c flood (OPERA DSWx, 10 Dec 2025)",
        "G 0.05c",
    ]
    p = ROOT / "config" / "_map_legend.txt"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


if __name__ == "__main__":
    raise SystemExit(main())
