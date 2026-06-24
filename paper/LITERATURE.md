# Annotated Literature & Positioning
*Seismic bedload transport on Mt. Rainier glacial rivers during the December 2025 atmospheric-river floods*

Compiled June 2026. DOIs marked *(unverified)* must be confirmed before submission.

---

## 1. Foundational physics (forward models)

- **Burtin et al. (2008)**, *JGR Solid Earth* 113, B05301, doi:10.1029/2007JB005034 — first showed >1 Hz ambient noise along the Trisuli (Himalaya) tracks river hydrology and bedload.
- **Tsai, Minchew, Lamb & Ampuero (2012)**, *GRL* 39, L02404, doi:10.1029/2011GL050255 — **bedload forward model.** PSD ∝ (impact rate)·f³·m²w², i.e. **linear in flux q_b, ∝ D³ in grain size** (coarse tail / D₉₄ dominates). Our high-frequency band rests on this.
- **Schmandt et al. (2013)**, *GRL* 40, 4858, doi:10.1002/grl.50953 — Grand Canyon controlled flood: bedload 15–45 Hz, fluid tractions <1 Hz, air–fluid waves ~6–7 Hz are spectrally separable.
- **Gimbert, Tsai & Lamb (2014)**, *JGR-ES* 119, 2209, doi:10.1002/2014JF003201 — **turbulent-flow forward model.** P_water ∝ H^(7/3) S^(7/3) ∝ u\*^(14/3); turbulence occupies a lower band than bedload. Through hydraulic geometry → **P ∝ Q^(0.9–1.4)**, i.e. the ~linear "water baseline." *No analytic 5/4 exponent exists in this paper.*

## 2. Inversion / quantification (the standard toolchain)

- **Roth et al. (2016)**, *JGR-ES* 121, 725, doi:10.1002/2015JF003782 — Erlenbach; linear spectral inversion separates turbulence/rain from bedload, validated against impact-plate geophones.
- **Dietze (2018)**, *ESurf* 6, 669, doi:10.5194/esurf-6-669-2018 — **`eseis` R package**, de-facto environmental-seismology toolbox (implements Tsai/Gimbert models). Consider porting/benchmarking against it.
- **Dietze et al. (2019)**, *WRR* 55, doi:10.1029/2019WR026072 — joint inversion of bedload flux **and** flow depth from spectra.
- **Bakker et al. (2020)**, *JGR-ES* 125, doi:10.1029/2019JF005416 — **the key prior result for our diagnostic.** Low f (~10 Hz) → exponent ~1.5 (matches turbulence ~1.4); >30 Hz → strongly nonlinear with Q but ~linear with sediment flux. **Establishes "exponent rises with frequency above the turbulence baseline ⇒ bedload."** Power governed by ~D₉₅, ∝ D³.
- **Lagarde et al. (2021)**, *WRR* 57, doi:10.1029/2020WR028700 — quantifies Green's-function and coarse-tail GSD uncertainty in inversion.
- **Nasr et al. (2024)**, *ESurf* 12, 117, doi:10.5194/esurf-12-117-2024 — hydroacoustic analogue; distributed-source inversion.

## 3. Recent advances (2022–2026)

- **Cook & Dietze (2022)**, *Annu. Rev. Earth Planet. Sci.* 50, 183, doi:10.1146/annurev-earth-032320-085133 — authoritative review; best single entry point.
- **Antoniazza et al. (2023)**, *JGR-ES* 128, e2022JF007000, doi:10.1029/2022JF007000 — **24-station watershed-scale network**, maps where/when coarse sediment mobilizes; resolves multiple discrete pulses. Closest "array/network" precedent.
- **"Sounding out the river" / Chmiel et al. (2023/24)**, *ESPL*, doi:10.1002/esp.5940 — joint seismic + hydroacoustic bedload in a lowland alluvial river.
- **Luong et al. (2024)**, *JGR-ES* 129, doi:10.1029/2024JF007761 — modifies Tsai model for inelastic impacts + rolling/sliding; **30–80 Hz optimal bedload band**; Tsai model under-predicts flux 1–2 orders at shallow flow.
- **Luong et al. (2026)**, *WRR*, doi:10.1029/2025WR040371 *(recent)* — **hybrid empirical model: seismic power + shear stress jointly predict bedload flux.** Directly relevant to our empirical-scaling approach.
- **Roth et al. (2025)**, *Seismica* (DAS, "A River on Fiber") — distributed acoustic sensing for spatially-continuous fluvial monitoring; the emerging successor to point sensors.
- **Rickenmann et al. (2025)**, *ESPL* 50(5), doi:10.1002/esp.70059 — long-term multi-site impact-plate bedload synthesis (calibration backbone).
- **Nicoletti et al. (2026)**, arXiv:2604.17913 — full numerical waveform synthesis from grain-scale dynamics + turbulence.

### Added from the June-2026 positioning survey (competitors to differentiate)

- **Ogiso, Uchida & et al. (2021)**, *Prog. Earth Planet. Sci.* 8, 50, doi:10.1186/s40645-021-00448-1 — **the key prior art for our virtual-discharge framing.** Seismic noise as a *virtual gage for ungauged basins*; single-station power law $Q=A(E-E_0)^b$, validated on three floods incl. Typhoon Hagibis (Oct 2019), 1–2 Hz optimal after transient/background removal. *Our distinction: a **distributed multi-station field**, not single-station — network density is the contribution, not the P–Q concept.*
- **Gangemi et al. (2026)**, *EGUsphere* preprint egusphere-2026-1534 — **the closest competitor.** Opportunistic permanent stations on the **braided Tagliamento**; applies the single-thread Tsai/Gimbert framework *qualitatively only* (correlation/hysteresis/polarization) and explicitly states a braided source model "remains speculative and needs further investigation." *Cite to show the braided source-model gap is open; we occupy it with the satellite corroboration they lack.* (Preprint — track toward publication.)
- **Cook et al. (2021)**, *Science* 374, 87, doi:10.1126/science.abj1227 — Chamoli: regional seismic network detected/tracked the catastrophic flow; ~min-scale downstream lead. Early-warning precedent (different mechanism: debris flow).
- **Eibl et al. (2020)**, *Nature Comms* 11, 2504 (PMC7237689) — subglacial jökulhlaup tremor leads gage stage by **>20 h** in all four Skáftá cases. Early-warning precedent (different mechanism: outburst). *Together these cap "early warning" as a standalone novelty — deploy our ~36 h AR-flood lead as an applied payoff, not the spine.*
- **McLaughlin et al. (2024/2025)**, *JGR-ES* doi:10.1029/2024JF008159 (Arroyo de los Pinos) — bedload band 30–80 Hz; bedload dominates only for $r<100$ m, $f>30$ Hz. Supports the Nyquist caveat — but note the band is **grain-size/site-dependent** (Taiwan typhoon systems saw bedload at 5–15 Hz), so state #3 as *grain-size-conditional*, not a hard universal cap.

## 4. Sediment pulses & hysteresis (the "tracking" theme)

- **Roth et al. (2014)**, *EPSL* 404, 144, doi:10.1016/j.epsl.2014.07.019 *(unverified)* — **THE blocking citation.** Multi-station array along Chijiawan R. (Taiwan), hysteresis metric Ψ tracks a **downstream-migrating coarse sediment pulse** after dam removal. Must distinguish our work from this.
- **Roth et al. (2017)**, *JGR-ES* 122, doi:10.1002/2016JF004062 — **critical caveat:** hysteresis arises from bedload **and** boundary-roughness change; seismic power often tracks stage more than measured bedload. Must defend against.
- **Schmandt et al. (2017)**, *Geology* 45, 299, doi:10.1130/G38639.1 — 76-sensor array, 700 m reach, reach-scale bedload during augmentation flood.
- **Hassan et al. (2023)**, *WRR*, doi:10.1029/2023WR035406 — flume: supply timing controls bedload hysteresis sense.
- **Nativ et al. (2025)**, *GRL*, doi:10.1029/2024GL113784 — **"Stationary boulders increase river seismic frequency via turbulence."** Direct counter-mechanism: high-f power need not be bedload. Must address explicitly.
- Hysteresis sense conventions: clockwise ⇒ supply exhaustion / proximal / armor break-up on rising limb; counter-clockwise ⇒ delayed/distal delivery, pulse arrival on falling limb (Reid 1985; Kuhnle 1992; Mao 2014).

## 5. Sediment-pulse physics (translation vs dispersion)

- **Lisle et al. (2001)**, *ESPL* 26, 1409 — bed-material waves evolve dominantly by **dispersion**, not translation.
- **Cui et al. (2003a,b)**, *WRR* 39(9), doi:10.1029/2002WR001803 / 10.1029/2002WR001805 — sediment pulses in mountain rivers; dispersion dominates at Fr > ~0.4.
- **Sklar et al. (2009)**, *WRR* 45, W08439, doi:10.1029/2008WR007346 — translation/dispersion partitioning for gravel augmentation.
- **Czuba & Foufoula-Georgiou** — network structure controls pulse aggregation/dispersion en route to the sink.

## 6. Mt. Rainier / Puyallup / Puget Sound context

- **Czuba et al. (2011)**, USGS FS 2011-3083, doi:10.3133/fs20113083 — ~6.5 Mt/yr sediment to Puget Sound.
- **Czuba et al. (2012)**, USGS OFR 2012-1242 — **aggradation 1984–2009 up to 2.3 m (Puyallup), 0.6 m (Carbon); Nisqually nr National 0.13 m/yr.** Median-pulse transit times: Nisqually ~70 yr, Puyallup ~80 yr, Carbon ~300 yr, White ~60 yr. Aggradation concentrates at the confined→lowland slope break.
- **Magirl et al. (2010)**, USGS SIR 2010-5240 — lower Puyallup/White/Carbon conveyance loss from deposition.
- **Anderson (2025)**, USGS (Pierce Co.), data DOI 10.5066/P149MBYG — channel change & sediment transport through 2022 (lidar differencing); most recent aggradation update.
- **Beason et al. (2022)**, *ESPL*, doi:10.1002/esp.5274 — proglacial erosion rates, four Rainier basins, 1960–2017.
- **Hoblitt et al. (1998)**, USGS OFR 98-428 — Rainier lahar hazards (Osceola, Electron mudflows down the Puyallup).
- USGS/PNSN **debris-flow seismic detection** at Rainier (Tahoma Creek) — *detection* of mass flows, **not** continuous bedload quantification (our distinction).
- **December 2025 AR floods (confirmed):** two back-to-back atmospheric rivers ~Dec 8–12 2025; statewide emergency (Gov. Ferguson, ~Dec 10); Carbon R. nr Fairfax highest in 19 yrs; Puyallup record flooding; Mt Rainier NP closed indefinitely (debris flows, washouts). Sources: WA Mil. Dept; WA State Standard 2025-12-10; NASA SVS #5596; NPS MORA news. *(News/agency only — no peer-reviewed post-event analysis yet.)*

---

## 6b. Distance attenuation & the "seismic reach" (correction methods)

The central confound for a multi-station transect: high-frequency (bedload) power
is attenuated with distance from the channel.

- **Physical term** (Tsai 2012 Eq. 3; Gimbert 2014 ψβ(f)): amplitude
  $\propto r^{-1/2} e^{-\pi f r/(v_u Q)}$; power $\propto r^{-1} e^{-2\pi f r/(v_c Q)}$.
  e-folding distance $r_e = v_c Q/(2\pi f)$. With $Q\approx20$, $v_c(f)=1295\,(f)^{-0.374}$ m/s
  (Tsai): $r_e\approx316$ m at 5 Hz, **~210 m at the 5–15 Hz band center, ~13 m at 50 Hz**.
  So bedload (≳30 Hz) is essentially gone beyond a few hundred metres while turbulence
  (1–10 Hz) survives kilometres — bedload dominates only for $r<\sim100$ m and $f>\sim30$ Hz
  (Gimbert 2014).
- **Solutions used in the field:**
  1. **Active-source / hammer calibration** of the Green's function (Bakker et al. 2020
     doi:10.1029/2019JF005416; Lagarde et al. 2021 doi:10.1029/2020WR028700) — measures
     $v_c(f), Q$ locally; improves spectrogram fit by ~1 order of magnitude. *Not available to us.*
  2. **Amplitude-decay source location (ASL)**: $A_i=A_t\,\frac{S_i}{R_i}e^{-\pi f t_i/Q}$,
     grid-searched (Battaglia & Aki; Walter et al. 2017 nhess-17-939; Burtin et al. 2008/2010;
     packaged in **`eseis` `spatial_amplitude/_track`**, Dietze 2018). Needs ≥3–4 stations + assumed $v_c,Q$.
  3. **Spectral-decay inversion** of $Q,v_c$ from the data (trades off strongly with grain size).
  4. **Per-station normalization** (what we use) — simplest, but folds site+distance+source together; not transferable.
  5. **DAS** (Roth et al. 2025, Seismica) sidesteps standoff by sensing in/along the channel.
- **Our first-order result (fig 9):** the same-source PR cluster (PR03/PR01/PR02 at 0.19/0.71/1.9 km)
  shows the 5–15 Hz power **~distance-independent over 0.2–2 km — far weaker decay than the Q=20
  prediction** (which would give ~10⁴× drop to PR02). Interpretation: at these mid frequencies the
  band retains less-attenuated lower-frequency (turbulence) energy, and site response + crude
  channel distances preclude a data-driven $Q$.
- **PNW-inferred $Q$ (adopted instead of $Q=20$):** $Q(f)=Q_0 f^{\eta}$, **$Q_0\approx25$, $\eta\approx0.5$**
  → $Q(10\,\text{Hz})\approx80$ (range 40–240), from Cascade coda-Q (Havskov et al. 1989 BSSA, $Q_0{\approx}63 f^{0.9}$),
  Lg-Q (Erickson et al. 2004 BSSA, $152 f^{0.76}$, lower near the arc), and Mt. St. Helens **edifice**
  attenuation (Tusa et al. 2006 GJI, $Q_p{\approx}30$ under the crater; De Siena et al. 2016 EPSL). This gives
  $r_e\approx780$ m at the 5–15 Hz center (vs ~210 m for $Q=20$) — less attenuation, but bedload (30–80 Hz)
  still reaches only tens–hundreds of m. We restrict bedload claims to near-channel stations and treat farther
  ones as *upper-distance bounds*. A clean correction needs active-source calibration or a higher (30–80 Hz)
  band recorded close to the channel — which requires **≥160-sps instruments** (our CC broadband stations are
  50 sps, Nyquist 25 Hz; see the frequency-sampling caveat in §Methods).

## 7. Novelty matrix (what to claim, what to defend)

| Claim | Strength | Blocking prior work | Framing |
|---|---|---|---|
| Mountain-to-sea longitudinal transect tracking a pulse | **Moderate** | Roth 2014; Schmandt 2017; Antoniazza 2023 | Claim novelty in **catchment-length scale (tens of km, glacial source → lowland)**, not "first multi-station pulse tracking." |
| Named **atmospheric-river** flood + debris-flow event | **Strong** | typhoon/GLOF/windstorm studies (different storm type) | First **Pacific AR-tied** bedload seismology; argue AR hydrology (rain-on-snow, glacial melt, sustained Q) gives a distinct signature. |
| **Cascades glacial-river** bedload seismology | **Strong** | Rainier debris-flow *detection* (USGS/PNSN) | Distinguish continuous **bedload quantification** from catastrophic mass-flow *detection*. |
| Frequency-dependent scaling-exponent diagnostic | **Weak** | Bakker 2020; Gimbert 2014; Tagliamento 2026 preprint | Frame as **field-validation/application at transect scale**, not invention. |

**Lead framing (June-2026 survey-revised):** *single-thread fluvial-seismology
source-model **breakdown in a braided glacial-outwash reach**, diagnosed by
hysteresis reversibility + cross-flood baseline drift and corroborated by
Sentinel-2/-1 channel-migration imagery* — a seismic+satellite joint
channel-change diagnosis no 2023–2026 paper has published. The distributed
virtual-gage field is the network framing device (Ogiso 2021 owns the
single-station concept); the Nyquist/bedload limit is a grain-size-conditional
bounding caveat; the ~36 h lead is the applied payoff.
*(Superseded earlier lead: "first transect-scale bedload seismology on Cascades
glacial rivers during a named AR flood" — still true, but transect/AR-first is
weaker than braided-breakdown-first given Ogiso 2021 and the saturation of the
virtual-gage and early-warning niches.)*

**Three reviewer objections to pre-empt:**
1. *"Just Gimbert 2014 + Bakker 2020."* → We add the transect, the AR/Rainier system, and a multi-station β(f) map; we field-calibrate the water baseline per station rather than assuming it.
2. *"Nativ 2025: high-f can be boulder-driven turbulence, not bedload."* → Use hysteresis + temporal evolution + the negative control (lowland Snohomish, r≈0) to argue against a static-roughness origin.
3. *"Roth 2017: roughness change, not bedload, makes hysteresis."* → Report hysteresis sense per event and tie to the debris-flow supply chronology, not stage alone.

**Reconcile:** our water baseline came out ~1 (or the measured low band), Gimbert predicts ~1.4 — state which baseline we adopt and why (empirical per-station calibration).

**Methodological caveat discovered in re-analysis (2026):** the **0.5–2 Hz band is the oceanic secondary microseism**, not river turbulence — at PR01 it is *anti-correlated* with discharge (b=−0.29, r=−0.36). The turbulence baseline must be drawn from **1–5 / 2–8 Hz**; 0.5–2 Hz should never be auto-selected as the flow band. Also note **spatial decay**: PR01 (0.71 km from channel) gives a shallower bedload exponent (b≈1.2) than PR03 (0.19 km, b≈1.66) — consistent with distance attenuation of the high-frequency bedload signal, a transect-scale prediction worth testing explicitly.

**Target journal:** **JGR: Earth Surface** (field flagship; Gimbert/Roth/Bakker/Antoniazza live there) or **ESurf** (open, public review, same audience). **GRL** if compressed to a first-detection letter; **Seismica** for an open methods/detection framing.
