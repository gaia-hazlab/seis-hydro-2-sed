# Paper-grade workflow. Use inside pixi: `pixi run make <target>`
PY=.pixi/envs/default/bin/python
STATIONS?=CC.PR01,CC.PR02,CC.TRON
START?=2025-12-01T00:00:00
END?=2025-12-24T00:00:00
BATCHFLAGS=--use-rss --exclude-earthquakes --clip-impulses --despike-proxy \
  --flow-bands 0.5-2,1-5,2-8 --bedload-bands 5-15,10-30,30-60

.PHONY: discover stations figures paper clean
discover:        ## FDSN + USGS station/gage discovery along the corridor
	$(PY) workflows/00_discover_stations.py

stations:        ## process each station in STATIONS (comma list)
	@for s in $$(echo $(STATIONS) | tr ',' ' '); do \
	  echo ">>> $$s"; \
	  $(PY) scripts/run_river_rumble_batch.py --start $(START) --end $(END) \
	    --focus-seis-key $$s $(BATCHFLAGS) || true; \
	done

figures:         ## regenerate publication figures + scaling table
	$(PY) workflows/02_make_figures.py

paper:           ## render the full Quarto book to _book/ (quarto is a pixi dep)
	quarto render

clean:
	rm -f paper/figures/fig*.png paper/figures/scaling_table.csv
