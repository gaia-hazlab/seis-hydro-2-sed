# Paper-grade workflow. Use inside pixi: `pixi run make <target>`
PY=.pixi/envs/default/bin/python
STATIONS?=CC.PR01,CC.PR02,CC.TRON
START?=2025-12-01T00:00:00
END?=2026-01-01T00:00:00
BATCHFLAGS=--use-rss --exclude-earthquakes --clip-impulses --despike-proxy \
  --flow-bands 0.5-2,1-5,2-8 --bedload-bands 5-15,10-30,30-60

# Figure scripts that read ONLY committed results/*.csv + config/*.json (no network).
# Excludes 03 map (DEM) and 11/14 spectra (waveforms) — those fetch raw data and live
# on the full `repro` path. Workflow 19 (satellite) IS offline-capable via --from-cache
# (replays the committed notebooks/data/braid_cache/*.npz artefacts, no Planetary
# Computer). 21 (reorg timing) and 22 (domain panel) are appended explicitly below
# because they take args / have ordering dependencies.
FIGSCRIPTS=02_make_figures 05_bedload_time 08_b_of_time 09_attenuation \
  10_early_warning 12_virtual_q 15_threshold 16_classify_stations 17_rating \
  18_braided_hysteresis

.PHONY: discover stations figures figures-from-cache repro paper clean
discover:        ## FDSN + USGS station/gage discovery along the corridor
	$(PY) workflows/00_discover_stations.py

stations:        ## process each station in STATIONS (comma list)
	@for s in $$(echo $(STATIONS) | tr ',' ' '); do \
	  echo ">>> $$s"; \
	  $(PY) scripts/run_river_rumble_batch.py --start $(START) --end $(END) \
	    --focus-seis-key $$s $(BATCHFLAGS) || true; \
	done

figures:         ## regenerate the core scaling figures + table (workflow 02)
	$(PY) workflows/02_make_figures.py

figures-from-cache:  ## REPRO PATH A — all offline figures from committed cache (no network)
	@for f in $(FIGSCRIPTS); do echo ">>> $$f"; $(PY) workflows/$$f.py || exit 1; done
	@echo ">>> 21_braided_reorg_timing (both basins)"
	$(PY) workflows/21_braided_reorg_timing.py --basin puyallup
	$(PY) workflows/21_braided_reorg_timing.py --basin nisqually
	@echo ">>> 22_domain_panel (fig23; needs the reorg JSONs above)"
	$(PY) workflows/22_domain_panel.py
	@echo ">>> 19_braid_optical_change --from-cache (both regions; satellite artefacts)"
	$(PY) workflows/19_braid_optical_change.py --region puyallup --from-cache
	$(PY) workflows/19_braid_optical_change.py --region nisqually --from-cache
	@echo ">>> 23_braid_two_region (fig24; from the satellite cache)"
	$(PY) workflows/23_braid_two_region.py
	@echo ">>> 24_hazard_timing_clogging (fig25; gage stage+Q + proxy + reorg JSON)"
	$(PY) workflows/24_hazard_timing_clogging.py

repro:           ## REPRO PATH B — full pipeline from raw data (network; hours)
	$(PY) workflows/00_discover_stations.py
	$(MAKE) stations
	$(PY) workflows/03_make_map.py
	$(PY) workflows/11_spectra.py
	$(PY) workflows/14_bedload_ch.py
	$(PY) workflows/19_braid_optical_change.py --region puyallup   # live; refreshes braid_cache
	$(PY) workflows/19_braid_optical_change.py --region nisqually  # live; refreshes braid_cache
	$(MAKE) figures-from-cache
	$(MAKE) paper

paper:           ## render the full Quarto book to _book/ (quarto is a pixi dep)
	quarto render

clean:
	rm -f paper/figures/fig*.png paper/figures/scaling_table.csv
