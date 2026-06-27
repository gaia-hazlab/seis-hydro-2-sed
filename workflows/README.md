# Reproducible workflow — bedload seismology, Puyallup / Mt. Rainier

Run from the repo root inside the pixi env (`pixi shell` or prefix with `pixi run`).

| Step | Script | Output |
|---|---|---|
| 0 | `python workflows/00_discover_stations.py` | `config/_transect_discovery.json` — stations + river-km along the corridor |
| 0b | `python workflows/01_fetch_discharge.py` | USGS-NWIS discharge for every corridor gage (authoritative; the GAIA mirror was incomplete for Dec-2025) + `config/discharge_manifest.json` |
| 1 | `python scripts/run_river_rumble_batch.py --focus-seis-key CC.PR03 --use-rss --exclude-earthquakes --clip-impulses --despike-proxy` | `notebooks/data/results/*_timeseries.csv`, `fit_parameters.csv` (single-pass multiband proxy; NWIS gages) |
| 2 | `python workflows/02_make_figures.py` | `fig1..5` (scaling, scatter, hysteresis, event ts) + `scaling_table.csv` |
| 3 | `python workflows/03_make_map.py` | `fig1_transect_map.png` (PyGMT DEM + gages + SNOTEL + OPERA + station status) |
| 4 | `python workflows/04_traffic_noise.py` | `figS_traffic_noise.png` (contamination control) |
| 5,7,8 | `python workflows/05_bedload_time.py` / `07_fetch_aux_data.py` / `08_b_of_time.py` | `fig6` bedload time, `fig7` per-AR, `fig8` b(t); `config/aux_timeseries.json` |
| 6,13 | `python workflows/06_bedload_gif.py` / `13_virtual_q_gif.py` | `bedload_animation.gif`, `virtual_q_animation.gif` |
| 9 | `python workflows/09_attenuation.py` | `fig9_attenuation.png` (PNW Q, e-folding distance) |
| 10 | `python workflows/10_early_warning.py` | `fig10_early_warning.png` (~36 h upstream lead) |
| 11,14 | `python workflows/11_spectra.py` / `14_bedload_ch.py` | `fig11` flood-vs-quiet PSD, `fig13` 30–50 Hz edge |
| 12 | `python workflows/12_virtual_q.py` | `fig12_virtual_q.png` + `config/virtual_q*.json` (ratings) |
| 15 | `python workflows/15_threshold.py` | `fig14_threshold.png` + `config/threshold_qc.json` (Qc) |
| 16 | `python workflows/16_classify_stations.py` | `config/station_status.json` (drives hollow no-signal markers) |
| 17 | `python workflows/17_rating.py` | `fig15_rating.png` + `config/rating_fits.json` (stage–discharge) |
| 18 | `python workflows/18_braided_hysteresis.py` | `fig16_braided_hysteresis.png` + `config/braided_hysteresis.json` |
| 19 | `python workflows/19_braid_optical_change.py` | `fig19_braid_change.png`, `fig20_braid_timeseries.png` + `config/braid_optical_change.json` — Sentinel-2/-1 braid change, geometric drift, + SAR December series |
| 20 | `python workflows/20_warm_ar_snotel.py` | `fig21_warm_ar_snow.png` + `config/warm_ar_snotel.json` — high-elevation SNOTEL temperature/SWE: warm rain-on-snow ARs vs cold late-Dec snow accumulation |
| 21 | `python workflows/21_braided_reorg_timing.py --basin puyallup\|nisqually` | `fig22_braided_reorg_timing[_nisqually].png` + `config/braided_reorg_timing_<basin>.json` — event-scale TIMING of braided-channel reorganization (matched-Q baseline, logistic step, flood-wave lag correction, rain→snow supply-shutoff guard) |
| 22 | `python workflows/22_domain_panel.py` | `fig23_domain.png` — domain of applicability (gage distance × channel width × elevation → clean/confounded); reads the two reorg JSONs from step 21 |
| 23 | `python workflows/23_braid_two_region.py` | `fig24_braid_two_region.png` — Puyallup (compact incised) vs Nisqually (wide braidplain) change maps side-by-side, offline from the satellite cache |
| 24 | `python workflows/24_hazard_timing_clogging.py` | `fig25_hazard_clogging.png` + `config/hazard_timing_clogging.json` — bedload-initiation lead over peak Q/stage (M5, hazard) + slow AR3-recession braid clogging (M7, the avulsion mechanism) |

Scripts are standalone and idempotent; most read `notebooks/data/results/` (step 1) plus committed `config/*.json` and write into `paper/figures/`. Steps 3–22 can run in any order after step 1 (22 needs 21's two JSONs). **Step 19 is the only one that queries the network** (Microsoft Planetary Computer STAC, Sentinel-2/-1, no credentials) — but it now also runs fully **offline** with `--from-cache`, replaying the committed derived rasters in `notebooks/data/braid_cache/*.npz` (a live run refreshes that cache). `pixi run make figures-from-cache` rebuilds **every** figure — including fig19/20/22/23 — with no network. See [`paper/ROADMAP.md`](../paper/ROADMAP.md) for which figure supports which research thread.

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
