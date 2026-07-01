# Denolle-group pre-submission review — 2026-07-01

**Manuscript:** "Braided-channel reorganization during storms, resolved seismically" (Denolle & Lipovsky)
**Target:** JGR-Earth Surface — research article
**Method:** 9-subagent registry (Abstract, Introduction, Methods, Results, Discussion, Conclusions, Figures/Data, Reproducibility, Citation-Diversity) → 8-criterion synthesis. Profile: default.
**Note:** Advisory. Every finding requires human judgment before submission.

GitHub issues opened from this review carry the `manuscript-review` label (see tracking issue).

---

## Summary
Scientifically **sound and genuinely novel**: a near-channel seismic network resolves the *event-scale timing* of braided-channel reorganization on Mt. Rainier's glacial rivers, reinterpreting a P–Q scaling steepening as a **geometric channel-reorganization regime transition, not a bedload onset**, with above-field-norm reproducibility infrastructure. Every headline claim is backed by a shown figure; assumptions handling is exemplary; hedging discipline is a model. **Not yet submission-ready — for completion/consistency reasons, not scientific ones:** unfinished §Data placeholder text, a grammatically broken abstract sentence, an unsupported "~36 h" number in the Conclusions, and open AGU compliance items (LICENSE, code DOI, uncommitted caches). All mechanically fixable.

**Readiness: Revise before submission** (no fatal issues).

## Strengths
1. Sound evidence→claim chain — every headline claim maps to a shown figure; geometric-vs-bed-mechanical break separated by an independent satellite test + gage rating geometry.
2. Exemplary assumptions handling — single-thread idealization, its braided breakdown, supply-rich caveat, turbulence/bedload degeneracy all stated with failure modes.
3. Honest overreach discipline — nowcast = "proof of concept, not skill"; two-reach scaling self-flagged "suggestive, not established (n=2)"; three alternatives explicitly rebutted.
4. Reproducibility above field norm — pixi.lock, Dockerfile, fixed seeds, config-driven parameters, committed core caches, one-command offline rebuild, raw-data DOIs.
5. Citation hygiene + real novelty — 0% self-citation, 1916–2026 temporal spread, broad venue mix, high reference-combination novelty (seismic P–Q physics × unsteady-flow rating theory × braided-avulsion geomorphology).

## MAJOR findings (must fix before submitting)
1. **[C5/S-AB.12]** Abstract central sentence grammatically broken — doubled copula "…the P–Q scaling *is* a fluvial regime transition" (L18–20).
2. **[C2/S-ME.10]** §Data unfinished placeholder text: "(REF)" (L181), "The satellite images ..." (L220), "The snotel and gage stations." (L222); "seismig"/"Rainer" typos (L182).
3. **[C4/C7/S-CO.8]** Conclusions "leads the downstream flood-peak stage by ~36 h" (L1104) — appears nowhere in the body (§sec-ew: ~5–7 h transport-onset lead, ~8 h routing lag); overclaims vs the paper's own "not forecast skill".
4. **[C5/S-IN.11]** Two stray unfinished fragments in the Introduction: "Dynamic braiding in pro-glacial rivers." (L58) and "Geophysical sensing in mountain areas provide…" (L61–62, S-V error).
5. **[C2/S-ME.2]** Symbol collision: `Q` = both discharge and seismic quality factor in `exp(−2πf r/(v_c Q))` (L320–325, 495). Relabel the quality factor.
6. **[C3/S-RP.7]** No LICENSE file — hard AGU/FAIR-Reusable fail.
7. **[C3/S-RP.R1]** 3 infrasound npz (spectrogram_UW_LON, infrasound_specgram_CC, coherence_CC_PR03) git-ignored → S11–S13 not offline-reproducible as implied; 39/40/41 absent from `figures-from-cache`.
8. **[C3/S-RP.R2]** Open-Research build table mis-names 3 scripts: `26_rating_geometry`→`25_`, `28_width_stage`→`27_`, `25_hazard_clogging`→`24_hazard_timing_clogging` (L1157, 1161, 1163).
9. **[C8]** Missing Author Contributions/CRediT, Funding, Acknowledgments, Competing Interests, AI-use disclosure; placeholder Zenodo DOI (`10.5281/zenodo.XXXXXXX`).
10. **[C4/C7/S-DI.9]** Infrasound–coherence result (γ²<0.01; γ²≈0.2) introduced inside a Limitations bullet with figures only in the supplement — surface in Results or explicitly frame as a supporting test.

## MINOR findings
1. **[C5/S-RE.8]** Figure-order inversion — F5 `fig-braidsat` cited (L455) before F4 `fig-braid` (L532).
2. **[C5/S-FD.3]** Supplement numbering: text "S1–S11" (L1150) vs table S1–S13, with S8/S9 out of order after S13.
3. **[C2/S-ME.5, S-RE.7]** ~18 central numbers (b, Q_c, β, r_e, celerity, Δr) as point estimates with no CI, though the scaling table advertises bootstrap CIs.
4. **[C5/S-FD.4]** ~7 figure captions analytical rather than descriptive.
5. **[C5/S-FD]** F5b (`fig-pr01zoom`) lacks (a)/(b) labels and has an internal "…pending Planet E&R approval" note baked into the figure title.
6. **[C6/S-CD.11]** 9 defined-but-uncited refs (orphaned `williams1989`; topically-core `cui2003`/`lisle2001`) — cite or cut.
7. **[C1/C5/S-CO.9,.11]** Conclusions "early warning" register stronger than abstract's "proof-of-concept"; future-work thin (doesn't name March-2026 test).
8. **[C5/S-IN, S-ME]** Undefined-at-first-use: SNOTEL, RM, sps; mixed citation style (prose "Gimbert et al. 2014" vs `@`-keys at L86/L90).
9. **[C3/S-CD.12/.14]** `anderson2026` load-bearing preprint (14 cites); renamed/DOI-mismatched keys (`shakti2021`, `beason2022`, `piantini2022`); `jones1916` no DOI.

## Criterion tiers
- C1 Novelty: **Good** · C2 Methods/Soundness: **Fair** · C3 Reproducibility: **Good** · C4 Evidence–Conclusion: **Fair** · C5 Presentation: **Fair** · C6 Literature: **Good** · C7 Impact: **Good** · C8 Ethics/Compliance: **Fair→Poor**

## Diversity signals (surfaced, not scored; inferred fields need human check)
Self-cite 0% (0/51); temporal 1916–2026 (bimodal); no venue monopoly (top ~16%); geographic base US + W-Europe + 1 Japan (narrow global-South; typical for sub-field); interdisciplinary reach HIGH (the novelty — not padding).

## Items requiring human verification
- DOI/URL resolution: 6 data DOIs; `anderson2026`/`gangemi2026` preprint DOIs; `jones1916` (none); renamed keys `shakti2021`/`beason2022`/`piantini2022`.
- Whether released code hard-codes any parameter the text omits.
- `~36 h` provenance (typo / stale / missing-from-body?).
- v_c≈578 m/s phase-velocity plausibility; transferred attenuation Q(f) — specialist checks.
- Geographic/identity inferences (not confirmed via OpenAlex).
