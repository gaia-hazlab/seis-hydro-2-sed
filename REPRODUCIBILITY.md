# Reproducibility

This project is built to be reproduced from a clean clone. There are **two
reproduction paths** — pick by how much you want to recompute.

| Path | What it needs | Time | Command |
|---|---|---|---|
| **A — figures from archived intermediates** (default) | the committed `results/*.csv` analysis tables | minutes, **offline** | `pixi run make figures-from-cache` |
| **B — full pipeline from raw data** | network (IRIS FDSN, USGS NWIS, Planetary Computer) | hours | `pixi run make repro` |

Path A reproduces every manuscript figure from small, version-controlled
artefacts without touching the network — including the satellite braid-change
figures (fig19/fig20), which replay committed derived rasters
(`notebooks/data/braid_cache/*.npz`) instead of querying the Planetary Computer.
Path B re-derives those tables and refreshes the cache from the raw
seismic/discharge/imagery archives.

Why the satellite artefacts are cached, not re-fetched: the Planetary Computer
STAC API is intermittently unavailable and slow, and a published paper must
rebuild deterministically. Workflow 19 therefore saves the *derived* mndwi /
active-channel rasters once (`--region X` live) and replays them
(`--region X --from-cache`); the source imagery is never needed for the figures.

## Environment (pinned)

The single source of truth is **`pixi.lock`** (exact, multi-platform: osx-arm64,
linux-64, win-64). Never `pip install` into the env — add deps with `pixi add`
so the lock stays authoritative.

```bash
pixi install --locked      # exact environment from pixi.lock
```

A **container** is provided for full isolation (see `Dockerfile`):

```bash
docker build -t seis-hydro .
docker run --rm -it seis-hydro pixi run make figures-from-cache
```

## Determinism

- All stochastic steps are seeded: bootstrap CIs and Theil–Sen subsampling use
  `numpy.random.default_rng(seed)` with `seed: 0` (`config/analysis.yaml`).
- Earthquake masking, STA/LTA impulse clipping, and despiking are deterministic.
- Matplotlib uses the `Agg` backend; no `Date.now()`/RNG in figure code.
- Same input tables + same seed ⇒ byte-identical fits.

## Data provenance (for Path B / the Zenodo archive)

All inputs are public; query parameters are pinned in version control:

| Source | Service | Pinned in |
|---|---|---|
| Seismic waveforms | IRIS FDSN (networks CC, UW, PB) | `config/transect_puyallup.yaml` (stations, channels), `config/analysis.yaml` (window) |
| Discharge / stage | USGS NWIS IV (gage IDs) | `config/transect_puyallup.yaml` |
| Satellite optical/SAR | Microsoft Planetary Computer STAC (sentinel-2-l2a, sentinel-1-rtc) | `workflows/19_braid_optical_change.py` (AOI, dates, collections) — **no credentials**; derived rasters cached in `notebooks/data/braid_cache/` for offline `--from-cache` rebuild |
| Earthquake catalog | USGS (M≥3.5, 500 km) | `config/analysis.yaml` |

Large raw caches (`notebooks/data/fdsn_cache/*.mseed`, NWIS pulls) are
git-ignored for size and will be deposited on **Zenodo** with a DOI (pending);
the manifest above is sufficient to re-fetch them exactly. The small
analysis-ready `results/*.csv` **are** version-controlled so Path A needs no
network.

## What is and isn't in git

- **In git:** code, `config/*` parameters, `results/*.csv` analysis tables,
  `notebooks/data/braid_cache/*` derived satellite artefacts (npz/json/csv),
  generated figures, the rendered-book source. These together make Path A fully
  offline for **every** figure.
- **Zenodo (pending DOI):** raw waveform cache, raw NWIS/SNOTEL pulls, raw
  satellite scenes — too large for git, fully re-fetchable from the manifest above.

## Continuous integration

`.github/workflows/publish.yml` renders the book on push. A pipeline smoke-test
job (regenerate figures from committed `results/*.csv` and assert no error) is
the recommended next CI addition so the analysis path, not just the render, is
tested.
