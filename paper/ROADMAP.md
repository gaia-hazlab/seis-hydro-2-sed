# Story roadmap — seis-hydro-2-sed

*Strategic inventory of every research thread we have explored on the December-2025
Mt. Rainier atmospheric-river floods, the evidence and assumptions behind each, and
the figure(s) attached. Purpose: see all options on one page, then choose the single
main story for the paper. This is a planning document, not the manuscript.*

Last updated: 2026-06-22.

::: {.callout-tip}
## For the reader (orientation)
This page exists so we can choose **one** spine for the paper from everything the
project currently supports. We are not short of results — we have ~18 figures across
a dozen distinct lines of evidence — so the risk is a diffuse paper that does many
things adequately and none decisively. The list below is deliberately *flat and
even-handed*: each thread gets the same treatment regardless of how much we like it,
so the comparison is honest.

**How each thread is scored.** Every thread carries five fields:

- **Claim** — the one sentence we would have to defend in review.
- **Evidence (figs)** — the specific figures that back it (labels match the
  manuscript exactly; see the inventory table).
- **Key assumptions** — what must be true for the claim to hold. This is where most
  threads are vulnerable; read these as the reviewer's first line of attack.
- **Strength** — a blunt editorial judgment (foundational / strong / medium / weak),
  *not* a statistical statement. "Strong" means novel + well-supported + hard to
  refute; "weak" means the data underdetermine the claim even if it is plausible.
- **What would close the gap** — the single most decisive missing piece. Several of
  these point at the same three pending inputs (see *open dependencies* at the end).

**How threads relate.** They are not independent. T1 (scaling) underpins everything.
T4 (frequency rigor) and T9 (braided source) are *constraints* that bound or
reinterpret the more ambitious threads (T3 bedload, T5 threshold, T7 virtual gage) —
treating them as caveats rather than separate papers is deliberate. T10 (the event)
is the container the rest hang from. The "candidate main stories" section at the end
recombines the threads three ways and argues for one.

**What this document is not.** It is not the manuscript and not a literature review
(those are the *Manuscript* and *Reviews* parts of this book). Strength labels are
provisional and will move as the three open dependencies resolve.
:::

---

## Figure inventory (label → file → one-line content)

*(Click any figure name to open it.)*

| Label | Figure (click to view) | Content |
|---|---|---|
| `@fig-map` | [fig1_transect_map.png](figures/fig1_transect_map.png) | PyGMT transect: DEM, NHD rivers, gages (diamonds), SNOTEL, OPERA flood, station status (hollow = no signal) |
| `@fig-scaling` | [fig2_scaling_exponent.png](figures/fig2_scaling_exponent.png) | Scaling exponent *b* per station, source→downstream; no-signal hollow/grey |
| `@fig-scatter` | [fig3_pq_scatter.png](figures/fig3_pq_scatter.png) | log P vs log Q scatter + power-law fits |
| `@fig-hyst` | [fig4_hysteresis.png](figures/fig4_hysteresis.png) | Per-event P–Q hysteresis loops |
| `@fig-ts` | [fig5_event_timeseries.png](figures/fig5_event_timeseries.png) | Ridgeline event timeseries, ordered by distance downstream |
| `@fig-bltime` | [fig6_bedload_time.png](figures/fig6_bedload_time.png) | 5–15 Hz "bedload" power ×pre-flood median through the ARs |
| `@fig-bl` | [fig7_bedload_per_AR.png](figures/fig7_bedload_per_AR.png) | Per-AR average bedload by station (AR2 dominates) |
| `@fig-bt` | [fig8_b_of_time.png](figures/fig8_b_of_time.png) | Time-resolved exponent *b(t)* vs turbulence baseline |
| `@fig-atten` | [fig9_attenuation.png](figures/fig9_attenuation.png) | PNW Q(f) attenuation correction, e-folding distance |
| `@fig-ew` | [fig10_early_warning.png](figures/fig10_early_warning.png) | Upstream seismic leads downstream stage peak (~36 h) |
| `@fig-spec` | [fig11_spectra.png](figures/fig11_spectra.png) | Flood-vs-quiet PSD: 50-sps source (≤25 Hz) vs 200-sps lowland |
| `@fig-vqval` | [fig12_virtual_q.png](figures/fig12_virtual_q.png) | Seismic virtual discharge vs co-located gage, per station |
| `@fig-edge` | [fig13_bedload_ch.png](figures/fig13_bedload_ch.png) | 30–50 Hz bedload edge from 100-sps UW broadband |
| `@fig-thr` | [fig14_threshold.png](figures/fig14_threshold.png) | Broken-stick transport-onset threshold Qc |
| `@fig-rating` | [fig15_rating.png](figures/fig15_rating.png) | Stage–discharge rating Q=C(h−h0)^β at 4 gages |
| `@fig-braid` | [fig16_braided_hysteresis.png](figures/fig16_braided_hysteresis.png) | Braided-reach onset diagnostic (geometric vs transport) |
| `@fig-braidsat` | [fig19_braid_change.png](figures/fig19_braid_change.png) | Sentinel-2/-1 braid change Nov→Jan + predicted per-station geometric baseline drift |
| `@fig-traffic` | [figS_traffic_noise.png](figures/figS_traffic_noise.png) | Traffic-noise contamination control (supplement) |
| `@fig-vq` | [virtual_q_animation.gif](figures/virtual_q_animation.gif) | **Embedded GIF**: space–time distributed discharge field |
| — | [bedload_animation.gif](figures/bedload_animation.gif) | Repo-only multidisciplinary GIF (SNOTEL + gage + bedload) |

---

## The threads

Each thread: **Claim** · **Evidence (figs)** · **Key assumptions** · **Strength** · **What would close the gap**.

### T1 — Seismic→discharge scaling is the calibration backbone (turbulence)
- **Claim:** 5–15 Hz seismic power follows a stable power law P ∝ Q^b, b≈1.4–1.7, consistent with the Gimbert (2014) turbulent-flow prediction (not 5/4; baseline ~0.9–1.4).
- **Evidence:** `@fig-scaling`, `@fig-scatter`, `@fig-ts`.
- **Assumptions:** single-thread line source at fixed r; response removal valid; lag constant; turbulence (not bedload) dominates the band.
- **Strength:** **Strong / foundational.** r up to 0.94. Everything else rests on this.
- **Gap:** baseline b uncertain (1.0 vs 1.4); confounded with attenuation geometry.

### T2 — Multi-river generalization (two basins, not one reach)
- **Claim:** The method works in a *second independent basin* — Nisqually (CC.GTWY r=0.89, UW.LON r=0.91, steepest b≈2.2) — not just the Puyallup.
- **Evidence:** `@fig-scaling`, `@fig-map`, `@fig-vqval`.
- **Assumptions:** cross-basin comparability of gages and station geometry.
- **Strength:** **Strong & undersold.** Generalization is a genuine novelty vs single-reach prior work.
- **Gap:** only 2 basins; CC networks for other western drainages downloaded but not all analyzed.

### T3 — Time-dependent bedload across the ARs (bounded hypothesis)
- **Claim:** 5–15 Hz excess power peaks in AR2 (≈110–180× background near source), 3× AR1/AR3, despite AR2 not being the largest discharge → progressive de-armoring / cross-pulse supply effect; b(t) rises above baseline during AR1–AR2.
- **Evidence:** `@fig-bltime`, `@fig-bl`, `@fig-bt`, `bedload_animation.gif`.
- **Assumptions:** **the 5–15 Hz band carries bedload at all** — directly contradicted by T4. Excess-over-baseline is bedload, not roughness/stage-geometry change.
- **Strength:** **Weak as literal bedload; strong as phenomenology.** The AR2 anomaly is real and interesting whatever its cause.
- **Gap:** no access to the canonical 30–80 Hz band at the stations that see the river (T4). Needs ground truth.

### T4 — Are we even sampling the right frequencies? (the honesty thread)
- **Claim:** No station *both* reaches the bedload band (30–80 Hz) *and* senses the river: 50-sps CC stations cap at 25 Hz (turbulence only); the one 200-sps lowland station is too far/urban. Grain size is amplitude-coded (∝D³), not band-coded below ~100 Hz. Bedload is therefore a frequency-bounded hypothesis.
- **Evidence:** `@fig-spec`, `@fig-edge`.
- **Assumptions:** PNW attenuation; Hertzian contact-corner scaling; UW.LON 30–50 Hz edge is tentative.
- **Strength:** **Strong & important.** This is the rigor that reframes the whole paper turbulence-led.
- **Gap:** the 2026 CH (500 sps) archive would settle it — not available via FDSN for 2025.

### T5 — Transport-onset threshold Qc
- **Claim:** Broken-stick fits find a slope-steepening Qc (≈35 m³/s upper Puyallup; ≈70–90 Nisqually/Carbon), matching slope-dependent critical Shields stress.
- **Evidence:** `@fig-thr`.
- **Assumptions:** the break is a *transport* onset. **Now contested by T9** — at PR01 it is geometric, not bedload.
- **Strength:** **Medium, partly undermined.** Physically motivated but mechanism-ambiguous.
- **Gap:** out-of-sample recurrence test (March 2026, running); geometric-vs-transport separation (done for PR cluster, T9).

### T6 — Attenuation correction (PNW Q)
- **Claim:** Using PNW-inferred Q(f)=Q0 f^η (Q0≈25, η≈0.5 → Q(10Hz)≈74–80) gives e-folding r_e≈780 m, far larger than Tsai's Q=20 (210 m) — sets how far a station "sees."
- **Evidence:** `@fig-atten`.
- **Assumptions:** regional Q applies to the Rainier edifice; single-scattering geometric spreading.
- **Strength:** **Medium / supporting.** Enables the detectability map (T11) and the braided-source argument (T9).
- **Gap:** Q not calibrated locally; no site terms.

### T7 — Seismic virtual discharge & the distributed nowcast map
- **Claim:** Each clean station is a virtual gage (Q_seis = 10^((logP−a)/b)); combining gages + virtual gages yields a space–time discharge field — distributed, gap-tolerant streamflow monitoring of an ungaged basin.
- **Evidence:** `@fig-vqval` (PR03/PR02 NSE 0.88/0.86), `@fig-vq` (the embedded GIF).
- **Assumptions:** rating stable & transferable in time. **T9 flags braided-reach drift** (PR01 NSE 0.70).
- **Strength:** **Strong & marketable.** The "seismic network = distributed gage array" message is the most novel application angle.
- **Gap:** cross-event transfer (March 2026 out-of-sample test pending).

### T8 — Downstream early warning (lead time + amplitude)
- **Claim:** Upstream seismic/stage lead the downstream flood-peak stage by ~36 h; downstream amplitude ~4× upstream → a seismic early-warning role for the undammed, lahar-prone corridor. Answers "why did downstream rise each AR but not upstream/seismic" (local sensing + basin wetting + tributary superposition).
- **Evidence:** `@fig-ew`, `@fig-ts`.
- **Assumptions:** lead time generalizes beyond this one compound event.
- **Strength:** **Strong & high-impact (hazard angle).** But n=1 event.
- **Gap:** needs the March-2026 (and ideally more) events to show the lead is robust, not coincidental.

### T9 — Braided-reach: a distributed, non-stationary source (NEW)
- **Claim:** The Puyallup source cluster sits on an aggrading glacial-outwash braidplain. The single-thread model fails: distributed source, nonlinear stage-partitioning, H^7/3 suppression/flattening, avulsion. PR01's "cleanest Qc" is **geometric (reversible loops, |HI|≤0.06, no supply decay), not bedload**; a coherent +0.2-log baseline drift AR1→AR3 across all 3 stations = channel migration.
- **Evidence:** `@fig-braid`, supported by `@fig-thr`, `@fig-scaling`, `@fig-vqval`. **`@fig-braidsat`: Sentinel-2/-1 pre/post (Nov→Jan) confirms the avulsion prediction directionally** — PR01's nearest active thread moved ~50 m closer and PR01 shows the largest predicted geometric drift (median Δlog₁₀P≈+0.28), matching the observed +0.2-log cross-AR drift in sign and rank.
- **Assumptions:** baseline drift is geometric, not a basin-supply trend; Lawler HI captures the relevant loop sense.
- **Strength:** **Strong & differentiating.** Recasts T3/T5 and warns T7 — a caution most of the (single-thread) literature has not had to make. Now satellite-corroborated.
- **Gap:** the satellite magnitude is threshold-sensitive at 10 m (sign/order robust, magnitude not) — a ≈3 m or a December time-series (per-AR) test would calibrate it; whether it generalizes to other braided reaches.

### T10 — The December-2025 compound AR event itself (the narrative spine)
- **Claim:** A weak pre-AR primed the basin, then three strong pulses; upstream stayed flat while downstream rose monotonically (cumulative wetting + rising freezing level + tributary superposition).
- **Evidence:** `@fig-map` (OPERA flood extent), `@fig-ts`, `@fig-ew`, `@fig-bltime`.
- **Assumptions:** AR window definitions; gage representativeness.
- **Strength:** **Strong as connective tissue.** This is the "dramatic event" container the braided thread (and all others) belong to.
- **Gap:** the event is the case study; not a standalone scientific claim.

### T11 — Detectability map: where river seismology works (and doesn't)
- **Claim:** Of 13 stations, 6 usable (r≥0.7), 2 marginal, 3 no-signal; hollow triangles document where it failed — itself a deployment guide.
- **Evidence:** `@fig-map`, `@fig-scaling`, `@fig-traffic` (traffic-contaminated controls).
- **Assumptions:** r-threshold classification; one event's classification generalizes.
- **Strength:** **Medium / valuable methods contribution.** Honest negative results.
- **Gap:** depends on station siting; sample of one network.

### T12 — Future capability: rising CC sample rates / 500-sps CH
- **Claim:** CC channels are increasing sample rate (some 500 sps as of 2026) → future multi-sensor flood + sediment monitoring; a direct 30–80 Hz bedload test becomes possible.
- **Evidence:** station table in `@fig-map`/config; `@fig-edge` as the best 2025 proxy.
- **Assumptions:** 500-sps archives will be retained and river-proximal.
- **Strength:** **Forward-looking / discussion only.** Not a 2025 result.
- **Gap:** 2026 CH data (Bradley/CVO route); March-2026 high-rate check.

---

## Candidate main stories (pick one spine)

**Option A — "A seismic network is a distributed gage array" (application-led).**
Spine: T1→T2→T7→T8, with T11 as method and T4/T9 as honest caveats. Sells the virtual-discharge map + early warning as the headline capability for an ungaged hazard basin.
*Strongest, most novel, most fundable. Figures: map, scaling, virtual-Q GIF, early warning, rating.*

**Option B — "What river seismology can and cannot do on a glacial braidplain" (rigor-led).**
Spine: T4→T9→T5→T11. Headline = the frequency-sampling and braided-source limits that bound bedload claims; turns the negative results and braided thread into the contribution.
*Most defensible, most original physically. Figures: spectra, braided diagnostic, threshold, detectability map.*

**Option C — "The December-2025 ARs seen seismically" (event/hazard-led).**
Spine: T10 as container, hanging T1/T3/T8 off it. A case-study narrative of a dramatic compound event.
*Most accessible, least novel methodologically. Figures: map+OPERA, timeseries, bedload-time, early warning.*

**Recommendation:** lead with **A** (distributed virtual-gage network + early warning) as the headline, fold **B**'s rigor (T4, T9) in as the "scope and limits" section that makes A credible, and use **C** (T10) as the framing event. T3/T5/T6/T12 become supporting/discussion. The braided thread (T9) is the credibility hinge between A's ambition and B's honesty — it is why the ratings are reach-specific.

**Open dependencies that could shift the choice:** March-2026 out-of-sample (tests T5, T7, T8 robustness); ~~PlanetScope pre/post (confirms T9 avulsion)~~ — **done via Sentinel-2/-1 (`@fig-braidsat`), directionally confirmed**; 2026 CH 500-sps (could promote T3/T12 from hypothesis to result).
