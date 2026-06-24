#!/usr/bin/env python3
"""Explicit, reproducible river-discharge downloader (USGS NWIS — authoritative).

Why this exists
---------------
The proxy pipeline historically pulled discharge from the GAIA *pre-generated*
mirror (`gaia-hazlab/gaia-data-downloaders/USGS_Stream_Gage/<gage>/`), which is a
static snapshot and was **incomplete for the December-2025 event** — e.g. gage
12092000 (Puyallup nr Electron) stopped at 2025-12-15 while the seismic record
runs to 12/31. This script pulls every corridor gage directly from the live
**USGS NWIS Instantaneous-Values** service (parameter 00060 discharge, 00065 gage
height), so the discharge record is complete and re-fetchable from one explicit
command — no dependency on an external mirror.

The gage list is read from `config/transect_puyallup.yaml` (the `gages:` map plus
the control gage and any gage referenced by a station), so it stays in sync with
the study design. Each gage is cached by `utils.fetch_usgs_nwis_iv` under
`notebooks/data/` in the exact file the batch reads, so a subsequent
`run_river_rumble_batch.py --gage-source nwis` reuses these downloads (no refetch).

Outputs
-------
  notebooks/data/usgs_iv_<gage>_<start>_<end>.csv  (per gage; NWIS cache)
  config/discharge_manifest.json                   (tracked: gage, name, span, n)

Usage:
  pixi run python workflows/01_fetch_discharge.py
  pixi run python workflows/01_fetch_discharge.py --start 2025-12-01 --end 2026-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from obspy import UTCDateTime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from utils import fetch_usgs_nwis_iv  # noqa: E402

DATA_DIR = ROOT / "notebooks" / "data"
CFG = ROOT / "config" / "transect_puyallup.yaml"
MANIFEST = ROOT / "config" / "discharge_manifest.json"


def corridor_gages() -> dict[str, str]:
    """{gage_id: name} from the transect config (gages map + controls + station gages)."""
    cfg = yaml.safe_load(CFG.read_text())
    gages: dict[str, str] = {}
    for gid, meta in (cfg.get("gages") or {}).items():
        gages[str(gid)] = (meta or {}).get("name", "")
    for sect in ("upper_transect", "lower_transect", "controls"):
        for row in cfg.get(sect) or []:
            gid = row.get("gage")
            if gid:
                gages.setdefault(str(gid), row.get("note", "") or row.get("site", ""))
    return gages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-12-01", help="UTC start date (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-01-01", help="UTC end date (YYYY-MM-DD)")
    ap.add_argument("--refresh", action="store_true", help="ignore cache and re-download")
    args = ap.parse_args()

    t0, t1 = UTCDateTime(args.start), UTCDateTime(args.end)
    gages = corridor_gages()
    print(f"Fetching {len(gages)} corridor gages from USGS NWIS, {args.start} → {args.end}")

    manifest = []
    for gid, name in sorted(gages.items()):
        try:
            df = fetch_usgs_nwis_iv(gid, t0, t1, data_dir=DATA_DIR, use_cache=not args.refresh)
        except Exception as e:  # noqa: BLE001
            print(f"  {gid:10s} {name[:34]:34s} FAILED: {e}")
            manifest.append(dict(gage=gid, name=name, ok=False, error=str(e)))
            continue
        qcol = next((c for c in ("discharge_cms", "discharge_cfs") if c in df.columns), None)
        span = (f"{df.index.min()} → {df.index.max()}" if len(df) else "EMPTY")
        cover = ""
        if qcol and len(df):
            q = df[qcol].dropna()
            cover = f" Q[{qcol}] n={len(q)} peak={q.max():.0f}"
        print(f"  {gid:10s} {name[:34]:34s} n={len(df):6d}  {span}{cover}")
        manifest.append(dict(gage=gid, name=name, ok=True, n=len(df),
                             start=str(df.index.min()) if len(df) else None,
                             end=str(df.index.max()) if len(df) else None))

    MANIFEST.write_text(json.dumps(
        {"window": {"start": args.start, "end": args.end},
         "source": "USGS NWIS IV (00060 discharge, 00065 gage height)",
         "gages": manifest}, indent=2))
    print(f"\nwrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
