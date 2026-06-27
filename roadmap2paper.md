# Roadmap → JGR-Earth Surface restructure

Restructuring `paper/paper.qmd` into a clean **IMRaD** fluvial-geomorphology manuscript
supported by seismic geophysics. Spine: **"Braided-channel reorganization during storms,
resolved seismically"** — a geometric/mechanistic reading of the December-2025 Mt. Rainier
atmospheric-river floods, with hazard-assessment potential.

Goals: (1) clean **Intro / Data / Methods / Results / Discussion (incl. Limitations) /
Conclusions**; (2) **~9 main figures** (down from 23) as multi-panel composites; (3) methods
detailed enough that any reader/bot could **re-code each analysis from the text**; (4)
peripheral methods/tests moved to an **extensive Supplement** in chronological (workflow 02→27)
order; (5) geospatial figures underlain by a **real co-located Sentinel-2 true-color image**
with channel/river masks on top.

Staging: drafted in **`paper/paper_restructured.qmd`** alongside the existing `paper.qmd`
(swap once approved). Existing `{#fig-...}` Quarto anchors are **kept** so the ~50 in-text
cross-references keep resolving; merged figures inherit the primary anchor.

---

## Figure consolidation (23 main → 9 composites)

| New | Title | Merges (old labels) | Anchor | Basemap |
|---|---|---|---|---|
| **F1** | Study area, channel pattern, flood driver | fig1 + fig27(b–d) + fig21(inset) | `#fig-map` | Sentinel-2 |
| **F2** | Seismic power tracks discharge (backbone) | fig5 + fig3(subset) + fig2 | `#fig-scaling` | — |
| **F3** | Transport-onset threshold & rating geometry | fig14 + fig26 + skill panel | `#fig-thr` | — |
| **F4** | Braided source: distributed, non-stationary | fig16 + fig9 | `#fig-braid` | — |
| **F5** | Satellite corroboration, two basins | fig19 + fig19-nis + fig24 | `#fig-braidsat` | Sentinel-2 |
| **F6** | Event-scale **timing** of reorganization (spine) | fig22 + fig28 + fig25(b) | `#fig-reorg` | — |
| **F7** | Domain of applicability | fig23 | `#fig-domain` | — |
| **F8** | Virtual gage + downstream early warning | fig12 + fig10 + fig25(a) | `#fig-vq` | — |
| **F9** | Bedload as a bounded hypothesis (frequency) | fig11 + fig6 | `#fig-spec` | — |

## Supplement (chronological / workflow order)

| SI | Content | From | Workflow |
|---|---|---|---|
| Text S1 | Anthropogenic-noise screening | — | 04 |
| Text S2 | Flood-wave lag estimation + validation | — | 21 |
| Text S3 | Satellite MNDWI water-threshold ensemble | — | 19 |
| Text S4 | Virtual-discharge inversion + NSE skill | — | 12 |
| Text S5 | 30–50 Hz lower bedload edge / Hertzian | — | 14 |
| Text S6 | Coarse / wood-rich band-lowering caveat | — | — |
| S1 | Traffic-noise screening | figS_traffic | 04 |
| S2 | Full P–Q scatter grid (all stations) | fig3 | 02 |
| S3 | Full event hysteresis loops | fig4 | 02 |
| S4 | Per-AR bedload, source→downstream | fig7 | 05 |
| S5 | Time-resolved exponent b(t) | fig8 | 08 |
| S6 | 30–50 Hz lower bedload edge | fig13 | 14 |
| S7 | Stage–discharge rating, 4 gages | fig15 | 17 |
| S8 | SAR wetted-illumination time series | fig20 | 19 |
| S9 | Nisqually reorg-timing (confounded) | fig22-nis | 21 |
| S10 | Bedload animation (HTML/repo only) | — | 06 |
| S11 | Virtual-discharge animation (HTML only) | — | 13 |

## Section map

- **Introduction** — spine framing, seismic scalings (eq-water/eq-bed), three questions.
- **Data & Study Area** (F1) — network, channel-pattern gradient; the Dec-2025 AR floods fold in
  as a *Flood driver* subsection.
- **Methods** — core, fully re-codeable (band-power proxy; robust P–Q fit + HI; matched-discharge
  baseline + logistic step + bootstrap; lag-correction summary; broken-stick + n_eff; attenuation
  kernel; satellite change + W-illumination). Peripheral methods → SI text pointers.
- **Results** — 4.1 backbone (F2) · 4.2 threshold + rating geometry (F3) · 4.3 braided distributed
  source (F4) · 4.4 satellite corroboration, two basins (F5) · 4.5 **event-scale timing + mechanism**
  (F6, spine) · 4.6 domain (F7) · 4.7 bedload bounded hypothesis (F9) · 4.8 distributed sensing +
  early warning (F8).
- **Discussion** — synthesis; rating-curve mapping (short); new capabilities / 2026 what-if;
  **Limitations** subsection.
- **Conclusions**; **Open Research** — workflow→figure table updated to F1–F9 / S1–S11.

## Satellite basemap

`workflows/28_fetch_basemaps.py` (network, one-time) pulls Sentinel-2 L2A true-color (B04/B03/B02,
cloud-masked, clear pre-flood window) for the Puyallup braidplain, Nisqually braidplain, and the
corridor; caches RGB + extent to `notebooks/data/braid_cache/*.npz`. `riverseis/basemap.py` provides
a shared underlay so F1/F5 render identically. Geospatial analysis panels render in **UTM 10N**
(EPSG:32610, the mask CRS) for correct co-registration; the basemap is illustrative, the mask is
quantitative. Raw pull stays out of `figures-from-cache`; cached arrays are committed for offline
rebuild.

## Tracking

GitHub issues labelled `restructure`: one tracking issue + (A) basemap cache, (B) composite figures,
(C) section restructure, (D) methods re-codeability, (E) supplement reorganization.
