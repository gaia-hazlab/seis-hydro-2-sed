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
