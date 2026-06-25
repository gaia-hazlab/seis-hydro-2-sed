# Resume here (staged 2026-06-24 evening)

Everything below is committed and pushed to `origin/main`. Background tasks stop
when the laptop sleeps, so this is the manual relaunch plan.

## The re-evaluated spine (decided this session)

> **The seismic power–discharge scaling breaks at a critical discharge across a
> glacial-river network, but the break diagnoses a *fluvial regime transition*
> (bed-mechanical entrainment / geometric-braided channel reorganization /
> hydraulic-geometry suppression) — *not* necessarily the "bedload onset" the field
> routinely infers. Seismic hysteresis + satellite channel-change tell them apart;
> the braided Puyallup source (PR01) is the satellite-confirmed geometric
> end-member, the steep Nisqually (UW.LON 1.1→5.0) and the flattening STYX are
> contrasting regimes.**

Full reasoning, the physics re-evaluation, and the updated thread table (T5 upgraded
to the two-regime diagnostic; new T13 warm-AR) are in
[`paper/ROADMAP.md`](paper/ROADMAP.md) → "June-2026 re-evaluation".

## Pending work (in priority order)

### 1. Nisqually SAR (code ready; was blocked on a Planetary Computer outage)
Run when the PC API is back (test first):
```bash
pixi run python -c "import planetary_computer as pc; from pystac_client import Client; \
print(len(list(Client.open('https://planetarycomputer.microsoft.com/api/stac/v1', modifier=pc.sign_inplace)\
.search(collections=['sentinel-2-l2a'], bbox=(-121.828,46.738,-121.792,46.764), datetime='2025-11-01/2025-11-30').items())), 'scenes')"
pixi run python workflows/19_braid_optical_change.py --region nisqually   # -> fig19_braid_change_nisqually.png
```
Then add the figure + a short paragraph to §sec-braided testing whether UW.LON's
steep break is geometric (braided/anabranch) like PR01.

### 2. Reframe the manuscript (Part I) to the regime-transition spine
The threshold/break physics is already written (§"Transport-onset threshold and
two-regime scaling" + fluvial mechanisms). To complete the spine reframe:
- **Abstract:** add one sentence — the P–Q break is a regime transition, not a
  bedload onset; hysteresis + satellite discriminate.
- **Introduction (3 questions):** promote the break/mechanism question to the lead;
  keep bedload-sampling as the bounding caveat.
- **Discussion:** add a synthesis paragraph that gathers PR01 (geometric),
  UW.LON/GTWY (steep Nisqually), STYX (flattening) into the network-diagnostic
  message; cite the survey competitors (Ogiso 2021, Gangemi 2026).
- **Conclusions:** lead contribution (1) becomes the break-diagnostic, not the
  virtual gage.

### 3. Open dependencies (longer horizon)
- March-2026 out-of-sample event (tests break recurrence, rating transfer, lead).
- ≈3 m PlanetScope to calibrate the braided-migration magnitude (10 m Sentinel is
  directional only).
- 2026 CH 500-sps archive → direct 30–80 Hz bedload test.

## Relaunch / verify on any machine
```bash
pixi install --locked
pixi run make figures-from-cache   # all offline figures from committed results/*.csv
pixi run make paper                # render the book to _book/
```
Full-pipeline-from-raw (hours, network): `pixi run make repro`.

## State at handoff
- Pushed: `origin/main` @ the latest commit. Working tree clean.
- All 21 manuscript figures pull from the NWIS / flood-window / robust-fit pipeline.
- The two-regime break analysis (BIC/AIC on autocorrelation-corrected n_eff) +
  fig2 break markers + the text reconciliation are committed.
- The bedload animation now spans all basins (Nisqually + Carbon) to 12/31.
