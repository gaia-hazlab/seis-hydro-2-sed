# Fluvial-geomorphology review — §sec-braided + reorganization-timing + domain

*Reviewer: `fluvial-geomorph-reviewer` skill (domain reviewer standing in for a
fluvial geomorphologist / river engineer). Scope: `paper/paper.qmd` §sec-braided
("The braided-reach problem"), §sec-reorg, §sec-domain. Target venue: JGR-Earth
Surface. Author home field: environmental seismology. Date: 2026-06-26.*

> **Advisory.** Every finding needs author judgment before acting. Severity:
> Blocking = a specialist would disbelieve the result; Major = must address;
> Minor = polish. Status: `open` until addressed; update the table as you act.

## Verdict
Strong, process-driven work a JGR-ES geomorphology audience will largely welcome:
mechanism-led (geometric vs bed-mechanical), it confronts the single-thread-model
breakdown most fluvial-seismology papers ignore, and it cross-validates the seismic
inference against independent satellite imagery with honest limits. Two things hold
back a clean acceptance — a **logical gap in the "geometric, not bedload" argument**
(reversible hysteresis rules out *supply-limited* transport, not transport per se, and
this basin is supply-*un*limited) and a **near-total absence of the braided/avulsion
canon** behind claims that invoke exactly those mechanisms. Both are fixable with
citation and wording, not new analysis.

## Strengths (keep)
- Names the reach and attacks the single-thread assumption head-on (the discipline's
  top pet-peeve, turned into the contribution).
- Mechanism over correlation; correct driver ($H^{7/3}$/depth, not velocity).
- Independent cross-validation (satellite channel-change ↔ seismic baseline drift).
- Honest uncertainty (MNDWI spread, 10 m resolution, n=3, onset-vs-t50).
- Hysteresis used as a supply *diagnostic*, not merely reported.

## Findings (tracker)

| ID | Sev | Finding | Fix (one line) | Status |
|---|---|---|---|---|
| FGR-1 | Major | "Reversible hysteresis ⇒ geometric, not bedload" only rules out **supply-limited** transport; this volcaniclastic basin is **supply-unlimited**, where a real bedload onset is also reversible & non-declining. | State the supply regime; rest the positive geometric case on satellite + $H^{7/3}$ suppression; demote reversibility to "rules out supply-limited transport." | open |
| FGR-2 | Major | Avulsion invoked but the avulsion canon is absent; the "aggrading braidplain" premise actually *predicts* avulsion via superelevation. | Ground avulsion in Slingerland & Smith (2004), Mohrig (2000), Jerolmack & Mohrig (2007); tie recession trigger to falling-stage bank failure. | open |
| FGR-3 | Major | Only the *seismic* braided studies cited (Piantini, Burtin); the braided-geomorphology canon is missing ("home-field-only" tell). | Cite Ashmore (1991; 2013), Ashworth/Best/Bristow, Bertoldi, Mosley (1983); Tal & Paola (2007) for bank/vegetation control. | open |
| FGR-4 | Major | "Migration tracks break steepness" overclaims on **n=2**. | Downgrade to "consistent with … a suggestive two-point result requiring more reaches." | open |
| FGR-5 | Major | Single-event / magnitude–frequency context absent for the timing result. | One sentence placing the Dec-2025 AR in flood-frequency context; bound the lag claim to this event; cite the planned March-2026 recurrence test (Wolman & Miller 1960). | open |
| FGR-6 | Minor | "Migration" (gradual) vs "avulsion" (abrupt) used interchangeably. | Pick the term the evidence supports per claim, or state you can't distinguish and what would. | open |
| FGR-7 | Minor | Bed character / grain size of the braidplain unstated. | One sentence on bed material (poorly-sorted glacial outwash, supply-rich, debris-flow-fed). | open |
| FGR-8 | Minor | "Stepwise = discrete avulsion" — alternatives (bar dissection, chute cutoff, single thread abandonment) not acknowledged. | Note the family of geometric events the step could represent. | open |

**Tracked as GitHub issues** (label `manuscript-review`): FGR-1 → #2, FGR-2 → #3,
FGR-3 → #4, FGR-4 → #5, FGR-5 → #6. Close each from the fixing commit
(`... fixes #N`). Minors (FGR-6–8) and Verify items (V1–V4) are tracked in this
report only.

### Detail
- **FGR-1 (lead finding).** Reversible loops (|HI|≤0.06) and non-declining peak power
  rule out *supply-exhaustion* transport, but in a supply-unlimited braidplain a
  genuine bedload onset would also be reversible and show no across-AR decline — so the
  hysteresis test does not, by itself, discriminate geometric from a supply-unlimited
  transport onset. The positive geometric case must lean on the satellite corroboration
  + the $H^{7/3}$-suppression mechanism + the supply-rich context (which makes a
  discrete transport *onset* unexpected). Stating the supply regime explicitly
  *strengthens* the geometric reading. (concepts §9–§10, §18.)
- **FGR-2.** The avulsion claim is Exner-consistent with the "aggrading braidplain"
  premise (aggradation → superelevation → avulsion); make that derivation explicit and
  cite the avulsion literature rather than asserting "falling-stage bank collapse /
  thread abandonment" bare. (concepts §7, §13.)
- **FGR-3 / FGR-5.** Canon gaps; "home-field-only citations" and missing
  magnitude–frequency are the two fastest specialist objections. (culture_norms;
  literature_canon.)

## Terminology & units
Mostly clean ("avulse and migrate," "anabranch," "celerity ≈2 m/s," $H^{7/3}$). Tighten
migration-vs-avulsion per claim (FGR-6); either ground or soften "bank collapse" as the
recession mechanism. Units/dimensionless quantities consistent; "$b=0.79$ below the
turbulence baseline" is the right framing.

## Literature the reviewer expects (missing, by theme)
- Avulsion/aggradation: Slingerland & Smith (2004); Mohrig (2000); Jerolmack & Mohrig (2007).
- Braided dynamics & hydraulic geometry: Ashmore (1991; 2013); Ashworth/Best/Bristow; Bertoldi; Mosley (1983); Egozi & Ashmore.
- Bank/vegetation control: Tal & Paola (2007).
- Magnitude–frequency: Wolman & Miller (1960).
- Falling-stage bank failure: Thorne; Rinaldi & Darby.
- (Already good: Piantini 2022, Burtin 2011 for the seismic-braid bridge; Anderson & Shean / Czuba for study-area sediment.)

## Cross-discipline framing
To make a braided-river specialist nod: (1) state the **supply regime** (unlimited → a
transport "onset" is not expected → the break is parsimoniously geometric); (2) embed
the avulsion claim in **superelevation/aggradation theory** (turns an assertion into a
derivation from your own premise); (3) engage the **braided-geomorphology canon** so it
reads as a contribution *to* fluvial geomorphology. Concede the n=2 "relationship" and
single-event scope explicitly — geomorphologists trust authors who bound their own claims.

## Verify-before-submitting
- V1 — "First seismic resolution of event-scale timing of braided reorganization": confirm no DAS/dense-array (or Piantini follow-up) has done timing; soften "to our knowledge" if unsure.
- V2 — "Falling-stage bank collapse / thread abandonment" as *the* mechanism: verify against avulsion/bank-failure literature; it is one of several recession-phase events.
- V3 — $H^{7/3}$ suppression ⇒ exponent flattening: present as a tested prediction (PR01 $b=0.79$ is consistent), citing braided hydraulic geometry.
- V4 — Migration-vs-steepness "relationship": confirm you are not implying a fit on n=2.
