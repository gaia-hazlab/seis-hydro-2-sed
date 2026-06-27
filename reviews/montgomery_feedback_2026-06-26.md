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
| M4 | Width–stage (dW/dH) on rising vs falling limbs | #11 | Partial (stage continuous; width coarse epochs) | open |
| M5 | Bedload initiation vs peak Q vs peak stage — hazard | #12 | **Now** — see results | open (prelim) |
| M6 | Sustained falling limb → flux (beyond threshold) | #13 | **Now** (proxy+stage on recessions) | open |
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
