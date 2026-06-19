# Reproducible workflow — bedload seismology, Puyallup / Mt. Rainier

Run from the repo root inside the pixi env (`pixi shell` or prefix with `pixi run`).

| Step | Script | Output |
|---|---|---|
| 0 | `python workflows/00_discover_stations.py` | `config/_transect_discovery.json` — stations + river-km along the corridor |
| 1 | `python scripts/run_river_rumble_batch.py --focus-seis-key CC.PR03 --use-rss --exclude-earthquakes --clip-impulses --despike-proxy` | `notebooks/data/results/*_timeseries.csv`, `fit_parameters.csv` |
| 2 | `python workflows/02_make_figures.py` | `paper/figures/fig1..5.png`, `scaling_table.csv` |

Batch over the whole upper transect (all CC stations with gages):
```bash
pixi run python scripts/run_river_rumble_batch.py \
  --start 2025-12-01T00:00:00 --end 2025-12-24T00:00:00 \
  --network-filter CC --use-rss --exclude-earthquakes \
  --clip-impulses --despike-proxy \
  --flow-bands 0.5-2,1-5,2-8 --bedload-bands 5-15,10-30,30-60
```

## Repository layout (paper-grade)
```
config/
  transect_puyallup.yaml     mountain-to-sea station↔gage transect (+ feasibility caveats)
  analysis.yaml              proxy/cleaning/lag/fit parameters (single source of truth)
  _transect_discovery.json   generated station list (step 0)
src/riverseis/
  analysis.py                load, robust log-log fit (bootstrap CI), Lawler hysteresis index
scripts/
  run_river_rumble_batch.py  proxy + alignment + per-station fit (uses notebooks/utils.py)
notebooks/
  utils.py                   pipeline engine (Tier-1 fixes applied — see REVIEW_2026.md)
workflows/
  00_discover_stations.py    FDSN + USGS discovery
  02_make_figures.py         publication figures from results/
paper/
  manuscript.md              draft scaffold (findings + reviewer defenses)
  LITERATURE.md              annotated bibliography + novelty matrix
  references.bib             BibTeX
  figures/                   generated figures + scaling_table.csv
REVIEW_2026.md               physics + code review, Tier-1 fixes, recomputed diagnostics
```

## Key parameters (config/analysis.yaml)
- Strict response removal (no raw-counts fallback), RSS Z/N/E, Welch band power.
- Lag scan high-pass detrended (6 h) + sign-constrained.
- Impulse clipping restricted to triggered windows.
- Robust log–log fit with bootstrap 95% CI; turbulence baseline b≈0.9–1.4.

## Pre-commit hook (render check)

A versioned hook in `.githooks/pre-commit` renders the book before a commit, but
**only when book inputs are staged** (`index.qmd`, `_quarto.yml`, `REVIEW_2026.md`,
`paper/**`, `workflows/README.md`) — other commits stay fast. It blocks the commit
if the render fails. Quarto is a pixi dependency, so the hook is self-contained.

One-time install per clone:
```bash
git config core.hooksPath .githooks
```
Bypass for a single commit: `git commit --no-verify`. (`_book/` is git-ignored;
the hook is a render *check* — GitHub Pages publishes from source via CI.)

## Provenance
Seismic: IRIS FDSN (`CC`, `UW`, `GS`, `PB`). Discharge: USGS NWIS IV (cfs). Event: Dec-2025
PNW atmospheric-river floods. All inputs public; large waveform caches are git-ignored.
