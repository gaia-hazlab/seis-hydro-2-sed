# Scientific & Code Review — Seismic Bedload Transport from the Dec-2025 Mt. Rainier Floods
*Revisit of the December 2025 "river rumble" work — June 2026*

This document revisits the scientific premise, audits the physical equations and their
implementation in [`notebooks/utils.py`](notebooks/utils.py) and
[`scripts/run_river_rumble_batch.py`](scripts/run_river_rumble_batch.py), recomputes the
key time-series diagnostics, and assesses whether there is a defensible new finding.

---

## 0. Verdict (TL;DR)

- **The premise is sound and under-exploited.** Discharge is measured; the *interesting,
  unclaimed* science is in the **deviation of the seismic–discharge scaling from the
  turbulent-flow baseline**, and in **hysteresis**. Nobody has tied the Dec-2025
  Rainier floods + debris flows to a mountain-to-sea bedload signal — the niche is real.
- **The "5/4 exponent" target is not correct as stated.** It is **not** a Gimbert/Tsai/Lamb
  result. The turbulent-flow (water) seismic power scales as **P ∝ H^(7/3) ∝ u\*^(14/3)**,
  which through hydraulic geometry gives **P ∝ Q^b with b ≈ 0.9–1.4 (≈ linear)**. Use the
  **locally-calibrated water exponent (~1)** as the baseline; **a steeper slope is the
  bedload signature**, not a deviation from 5/4. (5/4 is only a coincidental midpoint.)
- **The recomputed data already show the bedload signature** at the Rainier glacial-river
  stations: at **CC.PR03 (Puyallup nr Electron)** the slope **steepens from b≈1.6 (2–8 Hz)
  to b≈2.0 (10–30 Hz)** — well above the ~1 turbulence baseline, with high correlation.
  That super-linear, frequency-increasing exponent **is** a new, defensible result — *once
  the pipeline bugs below are fixed*, because the current products are contaminated.
- **The current implementation is a purely empirical band-power-vs-discharge regression.**
  There is **no Gimbert/Tsai forward model** (no Green's function, geometric spreading,
  attenuation, shear-stress or grain-size physics). That is fine as a first pass but the
  `science-plan.md` conceptual integral oversells what is coded.

---

## 1. Scientific premise (revisited)

The setting is excellent for the claim. The CC stations sit on **Mt. Rainier's glacial,
sediment-charged rivers**:

| Seismic | River / USGS gage | Notes |
|---|---|---|
| CC.PR01–PR05, CARB, CRBN, GNOB… | **Puyallup nr Electron 12092000** | glacial, debris-flow prone |
| CC.* | **Carbon nr Fairfax 12094000** | drains Carbon Glacier, very high sediment |
| CC.* | **Nisqually nr National 12082500** | drains Nisqually Glacier |
| UW.BHW | **Snohomish nr Monroe 12150800** | large lowland river — **negative control** |

The Dec-2025 atmospheric-river floods and debris flows mobilized large sediment volumes.
Because **discharge Q(t) is measured**, the seismic data are most valuable not for
predicting Q but for the **residual / scaling-deviation** that encodes bedload transport
efficiency and supply — exactly as `science-plan.md` argues. That framing is correct and
is the right thing to pitch.

---

## 2. The physics, and the exponent question

### 2.1 Turbulent flow (water) — Gimbert, Tsai & Lamb (2014, JGR-ES, Eq. 43)
Kolmogorov inertial-subrange turbulence → fluctuating bed tractions → Rayleigh waves with
1/√r spreading and exponential intrinsic attenuation e^(−2πf r /(v_u Q)). The final PSD scales as

> **P_water(f) ∝ H^(7/3) · sin(θ)^(7/3) · f^(4/3+5ξ) · (path/attenuation term)** , with ξ≈0.48.

Key scalings the code should respect:
- **Depth:** P ∝ **H^(7/3)** (verbatim in GTL14, p. 2221).
- **Slope:** P ∝ **S^(7/3)** (≈ const at a station).
- **Shear velocity:** u\*² = gH sinθ ⇒ P ∝ **u\*^(14/3)**.
- **Distance / Q:** enter **exponentially**, not as a power law.
- The model is parameterized in **u\* and H, NOT in U** — a "F ∝ U³" water law is a
  *different* simplified convention, not GTL14.

**Discharge exponent (the derivation GTL14 omits):** with S≈const, P ∝ H^(7/3); at-a-station
hydraulic geometry H ∝ Q^f (Leopold–Maddock f≈0.3–0.45; Tsai-2012 supplement gives up to
f≈0.6 at fixed width/slope). Hence

> **P_water ∝ Q^(7/3 · f) ≈ Q^0.9 … Q^1.4**, centered near **b ≈ 1 (linear)**.

→ **The right baseline is b≈1 (or, better, the *measured* low-band exponent for that
station), not 5/4.** GTL14 contains no 5/4, no 14/3, no analytic b at all; they validate
numerically per transect against the 2013 Colorado controlled flood.

### 2.2 Bedload — Tsai, Minchew, Lamb & Ampuero (2012, GRL, Eq. 7, 13)
Grain impacts as impulses → PSD per grain:

> **P_bed(f,D) ∝ (impact rate) · f³ · m² w² / (ρ_s² v_c³ v_u²) · χ(attenuation)** , integrated over the GSD.

Key results:
- **Linear in bedload flux:** P_bed ∝ q_b …
- **…but ∝ D³** (or D² on rough beds): the signal is **dominated by the coarse tail**
  (D₉₄), so a shifting grain-size distribution biases any flux inversion. This is the
  model's biggest vulnerability.
- Flux itself is **threshold-controlled and strongly nonlinear**: q_b ∝ (τ\*−τ\*_c)^1.5
  (Fernández-Luque & van Beek). So **P_bed vs Q is super-linear and steepens above the
  entrainment threshold** — and is **multivalued (hysteretic)** in Q.
- Caveat (Luong et al. 2024, JGR): the bedrock-saltation model under-predicts flux by 1–2
  orders at shallow flow (ignores rolling/sliding).

### 2.3 Separating the two
- **Frequency:** turbulence ~**1–10/20 Hz**; bedload **>15 Hz** (analysis bands 30–80 Hz in
  recent work). **The bands overlap** — clean separation by frequency alone is approximate.
- **Slope vs Q:** turbulence ≈ linear & single-valued; bedload **steeper & hysteretic**.
- **Hysteresis** (Hsu 2011; Roth 2014/2016; Cook 2018 Science GLOF):
  - **Clockwise** (more power on the *rising* limb): supply exhaustion, armor break-up,
    proximal/readily-available sediment.
  - **Counter-clockwise** (more power on the *falling* limb): **delayed / distal delivery**,
    migrating sediment pulse, bank collapse, tributary input — i.e., precisely the
    "delayed storage / sediment-pulse" idea worth pitching.
  - **Caution (Roth 2016):** boundary-roughness change can mimic supply-driven hysteresis.

---

## 3. Implementation critique (prioritized)

### Tier 1 — these can flip the conclusions
1. **Silent response-removal fallback → unit mixing.**
   [`_remove_instrument_response`](notebooks/utils.py#L1141) does `except: return tr`
   (raw counts). When metadata attach fails on a cached day, band power jumps by orders of
   magnitude. **This is the cause of the absurd dynamic ranges in the saved CSVs**
   (PR03 2–8 Hz spans 7×10²⁶). Fix: **fail loudly / drop the day**, never silently return
   counts; verify response attached before integrating.
2. **Lag selection is spurious.** [`estimate_constant_lag_seconds`](notebooks/utils.py#L1080)
   maximizes correlation of *log10 levels* over ±24 h. On trended storm hydrographs the
   correlation is dominated by the multi-day trend, so it locks onto **large, unphysical
   lags** — the saved `fit_parameters.csv` has **τ = −18 h** (proxy *leading* discharge;
   the notebook itself flags negative lags as metadata errors). Fix: detrend / first-
   difference (or band-pass to event timescales) before the lag scan, and **restrict to
   physically plausible lags** from gage↔station geometry and celerity (~1–4 m/s), with the
   correct sign for upstream vs downstream gages.
3. **Gage pairing by haversine only.** [`load_station_gage_pairs`](notebooks/utils.py#L180)
   picks the nearest gage by great-circle distance, ignoring whether it is on the *same
   river* or up/downstream. PR01 alone has **9 candidate gages**. This injects timing and
   scaling error. Fix: constrain to the same NHD reach / flowline, record upstream/downstream.

### Tier 2 — bias the slope
4. **Whole-day clipping is too aggressive.**
   [`clip_trace_days_on_stalta_impulses`](notebooks/utils.py#L1229) clips an **entire UTC
   day** to ±σ if *any* STA/LTA trigger fires that day. One teleseism nukes a day of real
   river signal, flattening the slope (plausibly why UW.BHW flow b=0.54 and 10–30 Hz r=0.06).
   Fix: clip only the **triggered windows**, not the day.
5. **Regression dilution.** OLS of log P on log Q attenuates the slope toward 0 when log Q
   is noisy/measured-with-error. Fix: report **Theil–Sen / Deming (total least squares)** and
   bin-averaged slopes; expect the true exponent to be *higher* than OLS.
6. **No transport threshold.** A single power law over all Q mixes below-threshold (no
   transport) with above-threshold data. Fit **piecewise** about an entrainment threshold
   (the bedrock-diagnostics cell uses an arbitrary 60th-percentile of Q — tie it to τ\*_c instead).
7. **Low band under-resolved.** Welch `nperseg = min(win, 10·sr)` → 10 s segments give poor
   resolution and few bins in the 0.5–2 Hz "flow" band. Use longer segments for low bands.

### Tier 3 — robustness / reproducibility
8. **RSS `dropna(how="any")`** drops all times when any component is missing → gaps/bias.
9. **Hysteresis is plotted but never quantified** — no index, no per-event separation, so the
   "delayed storage vs boulder size" hypotheses cannot be tested as written.
10. **Stale, inconsistent products:** `fit_parameters.csv` has only UW.BHW at the spurious
    −18 h lag, while the figures/CSVs come from different older runs with different cleaning.
11. **No distance/attenuation correction** ⇒ absolute power is **not** comparable across
    stations; only per-station relative change is interpretable. (`science-plan.md` admits this.)
12. **No physical forward model** — the proxy is band-integrated PSD in arbitrary units;
    there is no path to absolute flux without GTL14/Tsai inversion. State this plainly.

---

## 4. Recomputed diagnostics (Dec-2025 event)

Recomputed from the existing aligned CSVs with robust 6·MAD log-space outlier rejection,
zero lag (the saved −18 h lag is spurious), Theil–Sen + OLS slope, and a Lawler hysteresis
index over ±3 days around the event peak. Saved to
[`notebooks/data/results/RECOMPUTED_diagnostics_2026review.csv`](notebooks/data/results/RECOMPUTED_diagnostics_2026review.csv).

| Station / band | r | **slope b** | HI (event) | Q_peak (m³/s) | interpretation |
|---|---|---|---|---|---|
| **CC.PR03 2–8 Hz** (Puyallup) | 0.93 | **1.57** | −0.05 | 323 | above turbulence baseline |
| **CC.PR03 10–30 Hz** | 0.81 | **2.06** | +0.03 | 323 | **bedload regime; slope steepens with f** |
| CC.PR02 (single band) | 0.75 | 1.95 | +0.07 (CW) | 323 | super-linear; mild clockwise |
| UW.BHW 0.5–2 Hz (Snohomish) | 0.56 | 0.54 | n/a | 3000 | shallow — large lowland river / contamination |
| UW.BHW 10–30 Hz | **0.06** | 0.07 | — | 3000 | **no bedload signal (good negative control)** |

*(These are the pre-fix, as-found numbers; the bedload signal shrinks after the Tier-1 fixes — see the clean values in the manuscript.)*

> **Update (2026):** UW.BHW (lowland Snohomish) was used here only as an early
> negative control. It was **subsequently removed from the analysis** as too far
> from any bedload source to be informative; the published transect uses the
> internal source→downstream decay (CC.PR0x → CC.SIFT → CC.TRON) as the contrast
> instead. This chapter retains UW.BHW as a record of the development process.

Event peak at Puyallup nr Electron: **2025-12-09 ~03:30 UTC, ≈323 m³/s** (11,400 cfs).

---

## 5. Is there a new finding?

**Yes — a candidate, with caveats.** At the Rainier glacial-river stations the high-frequency
seismic power scales **super-linearly with discharge (b≈2)** and the **exponent increases
with frequency (1.6 → 2.0 from the flow band to the bedload band)**, far above the ~1
turbulent-flow baseline. That frequency-dependent steepening is the textbook bedload
signature (Tsai 2012: P_bed ∝ q_b with q_b ∝ (τ\*−τ\*_c)^1.5, plus D³ grain leverage). The
lowland negative control (UW.BHW, r≈0.06 in the bedload band) confirms the high-frequency
response is **not** a generic flow artifact. Differing hysteresis sign between stations
hints at event-scale changes in transport efficiency / a migrating post-debris-flow pulse —
the "delayed delivery" story.

**The honest caveat:** these slopes are computed from products contaminated by the Tier-1
bugs (unit-mixing from silent response failures, spurious −18 h lag, day-clipping). The
numbers are *suggestive and physically coherent* but **not yet publication-grade**. The
finding becomes solid only after re-running with the fixes in §3.

---

## 6. Recommended next steps (in order)

1. **Fix Tier-1 bugs** (response fallback, lag scan, gage pairing) and **re-run** the
   Rainier stations (PR01–05, CARB/CRBN, CRYS, Nisqually/Carbon gages).
2. **Calibrate the per-station water exponent** in a clean low band (1–5 Hz) → that is the
   baseline b_water; define **β_excess = b_high − b_water** as the bedload diagnostic
   (replaces the incorrect "deviation from 5/4").
3. **Quantify hysteresis per event** (Lawler index + enclosed area) across the station array
   and test for **downstream pulse migration** (Roth 2014 style) — the mountain-to-sea pitch.
4. **Threshold-aware, total-least-squares fits**; report slope CIs.
5. **(Optional) Implement a minimal Tsai/GTL14 forward model** for one well-instrumented
   reach to move from a proxy to an absolute flux estimate (with explicit GSD assumptions).

*Sources: Gimbert, Tsai & Lamb 2014 JGR-ES (Eq. 43); Tsai, Minchew, Lamb & Ampuero 2012 GRL
(Eq. 7, 13); Hsu, Finnegan & Brodsky 2011 GRL; Roth et al. 2014 EPSL / 2016 JGR; Cook et al.
2018 Science (GLOF); Luong et al. 2024 JGR-ES; Matthews et al. 2024 ESPL.*


## 7. Fluvial Hydrologist review
Committed (424a115) — the Nisqually two-basin result is integrated and the book renders clean. The migration-scales-with-break-steepness relationship (PR01 −50 m/Δb+1.7; UW.LON −240 m/Δb+3.1) is now a real, if two-point, argument that these breaks are geometric.

Now, switching hats. Here is my honest evaluation as a fluvial hydrologist / geomorphologist asked to referee this work.

Does this work generate new fluvial-science knowledge? A skeptical reading
Bottom line up front: This is a strong observational-methods contribution. As fluvial-process science, most of what it reports is the recovery of already-known fluvial behavior with a new sensor, not new knowledge about how rivers work. There is one genuinely novel caution for a subfield, one tantalizing-but-unproven candidate for new process insight, and a large untapped opportunity that, if exploited, would be the real fluvial-science contribution.

What is NOT new fluvial knowledge (most of it)
P ∝ Q^b turbulent-flow scaling — established (Gimbert 2014; Bakker 2020; Ogiso 2021). New site, not new physics.
Virtual discharge / distributed gaging — Ogiso (2021) already did this for ungauged basins. Network density is incremental.
Rain-on-snow generated the floods (warm-AR thread) — textbook PNW hydrometeorology. The SNOTEL panel confirms; it doesn't teach a hydrologist anything new.
~36 h upstream-to-downstream lead — this is essentially the kinematic flood-wave travel time from Electron (RM 41) to Puyallup (RM 10), which any hydraulic-routing model gives. New sensing of a known celerity, not a new hydrological insight.
Braided channels migrate / avulse and the active thread controls the local signal — this is the definition of a braided river. A geomorphologist already knows the source is distributed and non-stationary.
The one genuinely useful (semi-novel) point — but it's aimed at seismologists, not us
The thesis that a seismic P–Q break is a fluvial regime transition (geometry), not necessarily a bedload-transport onset is a real corrective — but it corrects an over-interpretation made by the environmental-seismology community, not a gap in fluvial knowledge. We already know that at-a-station width/depth partitioning, overbank activation, and anabranch switching produce non-linear stage–hydraulics. So this is valuable methodological hygiene for fluvial seismology, and the "braided reach is a distinct calibration regime" caution is a fair, modestly-new point for that small subfield. It is not a discovery about rivers.

The tantalizing candidate that is not yet knowledge
The AR2 "de-armoring / cross-pulse supply" effect (most bedload mobilized in AR2 despite AR2 not being the largest pulse) would be a genuine sediment-supply insight — supply-limited hysteresis across a compound event is real geomorphology. But the paper itself concedes the band is turbulence-dominated and the signal may be geometric, so this stays a bounded hypothesis. As written, it's a hint, not a result.
The migration-vs-break-steepness relationship is the most novel quantitative claim, but it is n = 2, with an uncontrolled seasonal (post-flood January) confound and a threshold-sensitive magnitude. Suggestive, not established.
Where the real new fluvial knowledge is hiding (and is currently left on the table)
The seismic network's unique affordance is continuous, sub-daily, spatially-distributed observation — something neither gages nor satellites provide. Braided-reach reorganization is normally seen only as before/after snapshots (your own Sentinel test) or inferred post-hoc. With 10-minute seismic illumination at several stations you could, in principle, answer questions a geomorphologist cannot currently answer:

When during the hydrograph does the active thread switch/avulse — on the rising limb, at peak, or on recession? Does channel reorganization lead or lag peak discharge?
Does the avulsion propagate through the cluster (a migrating front) and at what celerity?
Is the cross-AR baseline drift monotonic or stepwise (continuous lateral reworking vs discrete avulsion events)?
Resolving the timing and kinematics of braided-channel reorganization at event scale would be a genuine geomorphic discovery. The data already contain it (the baseline-drift timing, the per-AR loops); the paper currently reports the existence of migration, not its dynamics.

Verdict
As environmental-seismology / remote-sensing methods: solid, publishable, JGR-ES-appropriate — distributed glacial-river monitoring + a seismic–satellite discrimination of geometric vs bed-mechanical breaks.
As fluvial-geomorphology knowledge: thin as framed. It confirms known process with a new instrument and issues a useful caution to one subfield.
To make it a fluvial-science contribution: pivot the spine from "what the break is" (a static classification) to "the event-scale dynamics of braided-channel reorganization, resolved seismically" — i.e., use the continuous timing the sensor uniquely provides. That reframes the work from "a clever new gauge" to "a new way to watch a braided river rearrange itself during a flood."

## 8. New result — TIMING of braided-channel reorganization (the geomorphic angle, tested)
*Workflow [`workflows/21_braided_reorg_timing.py`](workflows/21_braided_reorg_timing.py)
`--basin puyallup|nisqually` →
[`config/braided_reorg_timing_puyallup.json`](config/braided_reorg_timing_puyallup.json) /
[`_nisqually.json`](config/braided_reorg_timing_nisqually.json) +
`paper/figures/fig22_braided_reorg_timing[_nisqually].png`. This is the §7 recommendation, executed.*

**Method.** The cross-AR baseline drift (workflow 18's `baseline_offset`, previously only
3 per-AR medians) is resolved continuously by tracking the seismic residual
`r(t) = log10 P − (a + b·log10 Q)` at **matched discharge**: for reference levels the
compound hydrograph crosses repeatedly during the event (60–160 m³ s⁻¹ — inter-pulse
troughs + limbs), sample `r` at every crossing and reference each level to its own
**rising-limb** value. The pooled series `c(t)` is a sequence of same-stage snapshots, so
any change is channel/source geometry, not discharge. Onset = first sustained >2σ
departure; the dominant step is a logistic fit (midpoint `t50`, width `4τ`, magnitude Δ)
with a bootstrap CI. (The matched-LOW-FLOW baseline, Q<median, only *brackets* the step —
Q never returns to baseflow between pulses — so the matched-Q crossings are what resolve
the transition.)

**Result — reorganization LAGS peak discharge; it never leads.** Peak Q at Electron is
2025-12-09 03:30 UTC. The rising-limb and pre-flood baselines are flat. The drift emerges
on the **falling limb of the main pulse and the later recessions**:

| Station | onset (lag vs peak) | dominant step `t50` (lag) | end-state (log₁₀) | character | R² | verdict |
|---|---|---|---|---|---|---|
| **CC.PR01** | — | — | −0.01 | — | 0.34 | **reversible / transient** — thread migrated and re-occupied (no persistent step) |
| **CC.PR02** | 12-09 16:25 (**+13 h**) | 12-12 01:31 (**+70 h**, CI +69…+72) | +0.30 | **stepwise** (≈11 h) | 0.66 | persistent (+) |
| **CC.PR03** | 12-09 23:50 (+20 h) | 12-12 05:06 (**+74 h**, CI +72…+75) | +0.21 | **stepwise** (≈12 h) | 0.72 | persistent (+) |

**Three genuinely new, gage-/satellite-invisible statements:**
1. **Lead/lag:** channel reorganization **lags peak discharge** — onset ~13 h into the
   recession of the main pulse, the persistent step ~3 days later on the final recession.
   It is a *recession* phenomenon (falling-stage bank collapse / thread abandonment),
   not a rising-limb or at-peak one.
2. **Stepwise, not gradual:** the persistent shifts complete in ~half-day transitions
   (τ ≈ 3 h), i.e. **discrete avulsion/thread-switch events**, not continuous lateral
   reworking.
3. **Spatial heterogeneity over ~1–2 km:** PR01 is **reversible** (the satellite-confirmed
   big-migration site moved and partly returned) while PR02/PR03 are **irreversible** —
   the braided reach does not reorganize as one body. The PR02→PR03 persistent steps are
   near-synchronous (Δt≈4 h, within timing uncertainty), so the data show a
   **recession-triggered local response**, not a resolvable laterally-propagating front.

**Honest caveats.** n = 3 stations; the compound hydrograph densely samples matched-Q only
at ~3 epochs (AR1 limbs, the AR1/AR2 trough, the final recession), so `t50` is pulled toward
the largest (final-recession) increment — the **onset** (+13 h) is the more robust
"when it starts," `t50` (+70 h) the "when it's mostly done." Band is 5–15 Hz (turbulence +
low bedload); the residual is geometry-dominated by construction but a within-event roughness
change could co-vary. Still: this is the first *event-scale timing* of braided reorganization
from the array, and it is the defensible seed of the reframed fluvial-science spine.

### 8b. Second basin — Nisqually (UW.LON, CC.GTWY): the clean signature does NOT reproduce
Same diagnostic, paired to **Nisqually nr National 12082500** (main peak 2025-12-10 14:15 UTC,
425 m³ s⁻¹ — here the *second* AR pulse is the larger one, so the rising-limb reference uses
the first pulse and lead/lag is measured against the later peak). The bounded, persistent
**positive** Puyallup step (+0.2…0.3 log, on the recession) **does not appear**:

| Station | b | matched-Q behavior | end-state (log₁₀) | R² | verdict |
|---|---|---|---|---|---|
| **UW.LON** | 2.7 | crashes ~1.5–1.8 log on the AR1 recession, **recovers** on the AR2 rise | −0.48 | 0.05 | **reversible / unresolved** — no persistent step (hysteresis/timing on a steep reach) |
| **CC.GTWY** | 1.6 | large monotone **negative** drift after the peak | −1.50 | 0.51 | persistent (−) **but ⚠ supply-confounded** |

**Why it differs — the rain→snow supply shutoff (mechanism, per MD).** Matched-Q controls for
discharge but not for *what sources that discharge*. After the warm AR the freezing level drops
and precipitation falls (and stays) as **snow**, so drainage-wide runoff — and the sediment it
mobilizes — collapses (Q falls 425 → ~40 baseflow). Storm flow is sediment-loaded (loud); the
post-event snowmelt-free recession delivers the *same* Q sediment-starved (quiet). So the bed
goes quiet **at equal Q** for a supply reason, not a geometric one. The workflow measures this
basin-wide as the raw post-event decline (`supply_shutoff_decline`, here ≈ −0.7 log at GTWY,
−1.1…−1.6 log at the Puyallup stations) and auto-flags any **negative** persistent step that is
same-signed with it. GTWY's −1.5 trips the flag → it **cannot be cleanly attributed to channel
reorganization**. The higher, colder Nisqually converts more of its catchment to snow, so the
shutoff is the dominant post-event signal there.

**Why the Puyallup result survives this and the Nisqually one doesn't.** The supply shutoff is
**negative**; the Puyallup steps are **positive** — opposite sign — so the geometric step is
measured *against* a headwind and is conservative (the true step is if anything larger). At
Nisqually the candidate step is itself negative, the *same* sign as the shutoff, so the two are
inseparable. Net: the event-scale reorganization-timing diagnostic is **cleanly interpretable in
the low-elevation, sediment-charged braided Puyallup reach, but confounded at the higher,
snow-dominated Nisqually** — a real methodological boundary that reinforces the PR cluster
(satellite-confirmed) as the clean geometric end-member. UW.LON's large *reversible* swings are a
separate caution: on a steep (b≈2.7) reach the matched-Q residual is sensitive to limb/timing and
supply hysteresis, so its steep P–Q break is **not** the braided-geometric end-member PR01 is.

### 8c. Lag-corrected cross-check — the Nisqually swings are largely a gage-geometry artifact
`workflow 21` now estimates a constant **P–Q flood-wave lag** per station (limb-timescale
cross-correlation, 36 h high-pass to avoid trend-locking) and shifts the gage discharge to the
station before forming the residual — but **only where a real travel time exists** (gage ≥3 km
away); at a co-located gage the estimated lag is sub-resolution noise and is not applied. The
implied celerities are physical and validate the correction:

| Station | gage distance | lag | implied celerity | effect on matched-Q |
|---|---|---|---|---|
| CC.PR01/02/03 | **0.2–1.9 km (co-located)** | ~0 (not applied) | — | none — result unchanged |
| CC.GTWY | 12.8 km downstream | −105 min | **2.0 m/s** | end-state −1.50 → **−0.57** |
| UW.LON | 20.9 km downstream | −180 min | **1.9 m/s** | swing span 2.1 → 1.6; still unresolved |

The Puyallup gage (Electron) is essentially **on** the PR cluster, so the clean result needs no
correction. The Nisqually gage (National) is **13–21 km downstream**, so gage Q lags the discharge
passing the station by the wave travel time; correcting it **collapses GTWY's apparent −1.5 drift
to −0.57** — now *within* the rain→snow supply-shutoff envelope (−0.7), i.e. no geometric signal
survives — and shrinks UW.LON's swings, which remain unresolved (sigmoid Δ disagrees with the
end-state sign → flagged reversible). A **constant** lag cannot fully fix a 20 km path whose
celerity varies with stage, so the far-gage Nisqually matched-Q is **intrinsically** limited — the
lag check converts "the method fails at Nisqually" into the sharper, correct statement: *the
Nisqually gage is too far downstream to time a 10-min source process against.*

---

## 9. Why the two basins differ — hydrology & geomorphology of the two source reaches
*Requested review. Tagged by confidence: ✔ = verified from data this session; ◐ = inferred /
needs a quick check. The throughline: **the very properties that make the Nisqually a more
dramatic braided site also make it harder to read seismically against a distant gage.***

| Property | Puyallup head (CC.PR01/02/03) | Nisqually (CC.GTWY, UW.LON) | Consequence for the seismic method |
|---|---|---|---|
| **Gage geometry** ✔ | gage **co-located** (0.2–1.9 km) | gage **13–21 km downstream** | the single biggest difference: clean vs lag-confounded matched-Q (§8c) |
| **Active-channel width** ✔ | **17–44 m** (optical wet width) | **190–624 m at UW.LON** (×3.3 Nov→post) | wide braidplain ≫ compact channel → the line-source / fixed-r model fails worse; source is hundreds of m of shifting anabranches |
| **Wet area in AOI** ✔ | 52,600 m² | 249,000–503,000 m² | source spread over ~5–10× the area → far more non-stationary geometry |
| **Station elevation** ✔ | 461–648 m | 617–**853 m** | higher Nisqually converts more of its catchment rain→snow → larger supply-shutoff confound |
| **Peak discharge (Dec-2025)** ✔ | 323 m³ s⁻¹ (Electron) | **425 m³ s⁻¹** (National) | bigger flow; combined with the wide section → deeper hydraulic-geometry effects |
| **Scaling exponent b** ✔ | 1.1–1.9 (PR cluster) | **2.7 (UW.LON, steepest), 1.6 (GTWY)** | steep b → matched-Q residual hypersensitive to small Q/timing error (amplifies the gage-lag artifact) |
| **Predicted geometric drift** ✔ | +0.07…+0.28 log (PR01 largest) | **+0.51 log (UW.LON)** | Nisqually's geometric change is *larger*, not smaller — it is unmeasurable cleanly, not absent |
| **Drainage flank / aspect** ◐ | NW flank of Rainier | S flank | different storm exposure, freezing-level behavior, glacier melt timing |
| **Source glaciers / network** ◐ | Puyallup + Tahoma + S. Tahoma Glaciers; several merging threads | Nisqually Glacier (+ Wilson); different confluence topology | more/fewer distributed inputs change source stationarity |
| **Drainage area** ◐ | smaller (consistent with lower peak Q) | larger (consistent with 425 vs 323 m³ s⁻¹) | larger basin → longer, more dispersive flood wave to the gage |
| **Valley confinement** ◐ | narrower, more incised/confined reach | wide, unconfined outwash plain (consistent with the width data) | confinement keeps the Puyallup source compact and stationary |

**Synthesis.** The Puyallup PR cluster is a *Goldilocks* reach for source-resolved river
seismology: a **compact (tens-of-m) channel** with a **co-located gage**, low enough to stay
rain-driven through the event. The Nisqually stations sit on a **wide (hundreds-of-m) high-
elevation braidplain gauged 13–21 km downstream** — every one of those differences (width,
elevation/snow, gage distance, steep b) pushes the matched-Q diagnostic from clean toward
confounded, even though the *physical* reorganization there is larger. This is not a failure of
the idea; it is its **domain of applicability**, and stating it is a genuine methods contribution
(which braided-reach competitors — Gangemi 2026 — have not made).

---

## 10. Master update plan — figures, threads, and the spine  ⟦TAG: REORG-2BASIN-2026-06-25⟧
*Everything below is the plan of record for the next pass. Commit-tagged so progress is trackable.
Principle the user set: **every thread T1–T13 ships — as the main spine or as supplementary** —
so the question is allocation and figure-readiness, not what to cut.*

### 10a. Figure-by-figure update checklist
Status: ✅ current · ⚠ needs refresh to the NWIS/flood-window/lag numbers · ➕ new this session.

| Fig | Update needed |
|---|---|
| fig1 map | ⚠ add the gage→station distance annotation (co-located vs 13–21 km) — it is now load-bearing |
| fig2 scaling-b | ⚠ confirm break markers + b values match the final flood-window fits (UW.LON 2.7, GTWY 1.6) |
| fig3 P–Q scatter | ⚠ apply the per-station lag to the Nisqually scatter; re-fit |
| fig4 hysteresis | ⚠ re-draw Nisqually loops lag-corrected (the raw loops are inflated by the 20 km lag) |
| fig5 event timeseries | ✅ keep; annotate AR1/AR2/AR3 + warm/cold tail |
| fig6/7/8 bedload(t), per-AR, b(t) | ⚠ re-confirm against complete NWIS; label band as turbulence+low-bedload |
| fig9 attenuation | ✅ supporting |
| fig10 early-warning | ⚠ recompute lead with lag-aware celerity (we now have 1.9–2.0 m/s in-basin) |
| fig11 spectra / fig13 30–50 Hz | ✅ the frequency-limit (T4) figures |
| fig12 virtual-Q | ⚠ note braided-drift + lag caveat on the Nisqually virtual gage |
| fig14 threshold Qc | ✅ two-exponent + ΔBIC |
| fig15 rating | ✅ |
| fig16 braided hysteresis | ⚠ supersede/extend with the new timing view (fig22) |
| fig19 braid satellite | ⚠ add the Nisqually panel (UW.LON W ×3.3) beside Puyallup |
| fig21 warm-AR snow | ⚠ retitle to the **rain→snow supply-shutoff** mechanism; tie to fig22 confound |
| ➕ **fig22 reorg-timing** | ✅ DONE (×2 basins) + flood-wave lag correction — lead the geomorphic-dynamics result |
| ➕ **fig23 domain panel** | ✅ DONE — two-basin **domain-of-applicability** (gage-distance × channel-width × elevation → clean/confounded); `workflows/22_domain_panel.py` |
| figS traffic / GIFs | ✅ supplement |

**Reproducibility hardening (done this pass).** `figures-from-cache` now rebuilds **every**
figure offline, including the satellite fig19/fig20: workflow 19 caches its derived Sentinel
mndwi/active-channel rasters to committed `notebooks/data/braid_cache/*.npz` (+ station pixels
json, December-series csv) and replays them with `--from-cache`, so the flaky Planetary Computer
API is never needed to rebuild the paper. Workflows 21 (both basins) + 22 are wired into the
offline Makefile path. Only raw seismic (slow) and a live PC refresh remain on the `repro` path.

### 10b. Thread disposition (all 13 ship)
- **Spine-eligible (lead candidates):** T9 braided source-model breakdown · **NEW reorg-timing**
  (event-scale dynamics) · T5 two-regime break · T4 frequency rigor.
- **Framing / payoff (main text, supporting):** T1 scaling backbone · T2 two-basin (now the
  *domain* contrast) · T7 virtual-Q field · T8 ~36 h early warning · T10 the event · T13 warm-AR
  rain→snow (now also the supply-shutoff confound).
- **Methods / supplement:** T6 attenuation · T11 detectability map · T12 future high-rate · T3
  time-dependent bedload (bounded hypothesis, supplement).

### 10c. Three candidate spines (pick one; the other two become section arcs)
**SPINE 1 — "Watching a braided river rearrange itself during a flood" (geomorphic-dynamics-led). ★ recommended.**
Lead with the **event-scale reorganization TIMING** (fig22): seismics resolve *when* the active
thread reorganizes — it **lags peak, is recession-driven and stepwise** — which gages and
before/after satellites cannot. T9 (source-model breakdown) + T5 (the static break signature) +
the **two-basin domain result** (§9, why it is clean at Puyallup, confounded at Nisqually) form the
body; everything else supplements. *Most novel; matches the referee's "pivot to dynamics" and the
open gap (Gangemi 2026 declines a braided model). Journal: JGR-ES.*

**SPINE 2 — "The domain of source-resolved river seismology on a glacial braidplain" (rigor/methods-led).**
Headline = honest boundaries: T4 (frequency/Nyquist) + T9 (braided source) + the **method-domain
law** (gage co-location, channel width, snow fraction set where matched-Q / virtual-Q works) + T11
(detectability). Turns the negative Nisqually result and the lag/width/snow analysis into the
contribution. *Most defensible; least "exciting."*

**SPINE 3 — "A seismic network as a distributed flood & sediment observatory" (application/hazard-led).**
T1+T2+T7 virtual-Q field + T8 early warning for the lahar-prone corridor, with reorg-timing +
braided + the domain result as the **reliability caveat**. *Most fundable; the survey flagged the
virtual-gage concept as the most saturated (Ogiso 2021), so it must be framed as density+dynamics,
not concept.*

### 10d. Execution order (next session)
1. Regenerate all ⚠ figures from the complete-NWIS + flood-window + lag pipeline; verify numbers.
2. Build the proposed **fig23 domain panel** (the §9 table as a figure).
3. Lock **Spine 1**; demote T1/T2/T7/T8 to framing/payoff; move T3/T6/T11/T12 to supplement.
4. Reconcile manuscript text (Abstract/Intro/Discussion/Conclusions) to the dynamics spine.
5. Out-of-sample March-2026 event as the robustness close (break recurrence + lead + rating transfer).