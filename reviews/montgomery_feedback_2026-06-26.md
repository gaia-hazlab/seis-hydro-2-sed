# Co-author feedback — D. Montgomery (2026-06-26) — plan & first results

Substantive feedback from David Montgomery (incoming co-author). Tracked as GitHub
issues **#8–#14** (label `coauthor-feedback`). Several items are testable with data in
hand; the two highest-value (M5 hazard timing, M7 the clogging mechanism) already have
supportive preliminary results below.

| ID | Item | Issue | Feasibility now | Status |
|---|---|---|---|---|
| M1 | Planform map: single-thread/meandering (past CC.TRON) vs braided (PR, UW.LON) | #8 | Medium (extend fig1; needs planform traces) | open |
| M2 | Stage–discharge ratings → overbank / geometry change | #9 | **Now** (stage+Q at gages; fig15 base) | open |
| M3 | Trace braid/bank boundaries; zoom WAY in on PR01 | #10 | Placeholder now (Sentinel 10 m); final needs ≈3 m | open |
| M4 | Width–stage (dW/dH) on rising vs falling limbs | #11 | Partial (stage continuous; width coarse epochs) | **done (partial)** |
| M5 | Bedload initiation vs peak Q vs peak stage — hazard | #12 | **Now** — see results | **done** |
| M6 | Sustained falling limb → flux (beyond threshold) | #13 | **Now** (proxy+stage on recessions) | **done** |
| M7 | Slow AR3 recession → deposition → braid clogging → avulsion | #14 | **Now (partial)** — see results | open (prelim) |

## Preliminary results (Puyallup / Electron gage 12092000; existing data)

### M5 — bedload initiation leads peak discharge (hazard early-warning)
- Peak Q = 323 m³/s and **peak stage both at 12-09 03:30** (no loop-rating offset at
  15-min resolution at this gage).
- On-channel **CC.PR03 5–15 Hz power crosses (pre-flood base + 0.5 log) at 12-08 21:05
  — ~6.4 h *before* peak Q**. Transport initiation is flagged on the rising limb,
  ahead of the peak → a concrete hazard early-warning signal.
- *Next:* tie onset to the transport threshold $Q_c$; per-station/per-AR; map the
  spatial lead (upstream initiation → downstream peak; links the ~36 h early-warning
  thread).

### M7 — the slow-recession clogging mechanism (supports DM's hypothesis)
- **AR3's falling limb is slower & more sustained than AR1's:** dQ/dt −7.5 vs
  −11.4 m³/s/h; **32 h (AR3) vs 20 h (AR1)** to drain.
- **The dominant reorganization step lands on the slow AR3 recession (12-12), not the
  rapid AR1 recession** (CC.PR02 step 12-12 01:31; CC.PR03 12-12 05:06; AR3 falls
  12-11 16h → 12-13). Exactly DM's prediction: gravel deposits during the *slow* fall,
  clogs the active braid, and forces the switch.
- **Why this matters:** it supplies the avulsion *mechanism* the pre-submission review
  flagged as missing (FGR-2/V2, #3) — deposition → superelevation/aggradation →
  avulsion (Slingerland & Smith 2004; Jerolmack & Mohrig 2007), here recession-driven.
- *Next (to demonstrate):* (1) matched-Q baseline drift concentrates on the AR3 slow
  fall; (2) a *deposition* signature (declining bedload-band power at sustained moderate
  stage); (3) satellite braid-change at PR01 showing the abandoned/newly-dry thread;
  (4) write the deposition→clogging→avulsion mechanism into §sec-braided.

*Analysis is reproducible from the committed gage CSV + `results/*_timeseries.csv`; a
workflow script for M5/M7 is the natural next step.*

## Built & integrated (2026-06-27)
- **`workflows/24_hazard_timing_clogging.py` → fig25** (`@fig-hazard`), wired into the
  offline pipeline; committed the small USGS gage CSVs (stage) for offline rebuild.
- **§sec-reorg integration committed** (`cada2f3`): the deposition→clogging→avulsion
  mechanism (Slingerland & Smith 2004; Jerolmack & Mohrig 2007; Mohrig 2000) + the
  source-reach hazard lead. **Closes reviewer FGR-2 / V2 (#3).**

## Confirmation analysis (2026-06-27)
- **Satellite, PR01 (verifies the manuscript claim).** Within 300 m of CC.PR01:
  **0 persistent, 71 newly-wet, 20 newly-dry** pixels (Nov→Jan). The near-station
  channel relocated completely — old thread abandoned (newly-dry), new thread occupied
  closer (newly-wet) = avulsion. ✓ (M7, #14)
- **Spatial lead (M5, #12).** Transport-band onset **propagates downstream**: SIFT
  (15 km) 19:45 → Electron cluster (22–23 km) 21:05–22:00 → STYX (27 km) 23:40 — a ~4 h
  sweep (~0.8 m s⁻¹). Added to the manuscript hazard paragraph. ✓
- **Deposition signature (inconclusive).** AR3 falling-limb $b$ steeper than rising
  (PR02 1.18→1.58; PR03 0.69→1.36) — superficially sediment-waning on the slow
  recession — but the avulsion *step* lies on the same falling limb and contaminates the
  fit. Suggestive, **not demonstrated**; manuscript keeps "parsimonious, not unique."
  *Open:* a cleaner deposition test (e.g. bed-level/DoD or grain-size evidence) for the
  final. (M6, #13 / M7, #14)

## M6 — what a sustained falling limb says about flux (2026-06-27)
Honest, modest result (text integration in §sec-reorg, no new figure):
- A binary transport threshold is **uninformative on the recession**: the source-station
  Qc (13-28 m3/s) is exceeded the entire multi-day fall, so it labels the whole falling
  limb "transport-on" and carries no flux evolution.
- The flux/reorganization information lives in the **recession SHAPE** of the seismic
  power (rapid AR1 vs sustained AR3; fig25b) - which is what the M7 clogging mechanism
  uses. Per-AR loops near-reversible (|HI|<=0.06) -> geometric/transport, not strong
  supply-exhaustion hysteresis.
- NOT supported: "sustained -> more cumulative flux" (cumulative 5-15 Hz energy is
  dominated by AR1's larger magnitude: PR02 957 vs 660; PR03 693 vs 481). The clean
  signal is recession-shape dependence, not integrated energy.

## M4 — width–stage hysteresis, rising vs falling limb (2026-06-26)
`workflows/27_width_stage.py` → fig28 (`@fig-width`), wired into figures-from-cache;
text in §sec-reorg after the recession-deposition mechanism. Offline from the SAR
December series + Electron gage stage.
- **Rising limb widens steeply:** SAR width proxy CC.PR01 → **2.7×** pre-flood as
  Electron stage climbs 5.0→7.9 ft; CC.PR02 → 1.7×.
- **Falling limb does NOT retrace** (counter-clockwise loop): at equal stage, recession
  width < rising width; secant dW/dH gentler on the fall (PR01 +0.65 rise vs +0.55 fall
  ft⁻¹; PR02 +0.38 vs +0.21).
- **PR01 ends at 0.6× pre-flood width — narrower than before the flood**; PR02 returns
  to ~baseline. Residual narrowing concentrated at PR01 = the same station the change
  map flags newly-dry/abandoned (fig22b) → independent geometric corroboration of the
  avulsion. Directly answers DM's "dW/dH on rising vs falling limbs."
- *Caveat (in caption):* 5 SAR epochs, area-ratio proxy (not metric width),
  epoch-averaged stage; ≥3 m repeat width series would sharpen dW/dH.
