# Exploration & next directions (June 2026)
*Three threads: frequency/bedload validity (#4), sediment-catalog basis (#5), and a new early-warning framing.*

---

## 1. Are we sampling the right frequencies for bedload? — mostly **no** (#4)

**Spectral evidence (`fig11_spectra.png`).** Median ground-velocity PSD, flood (10 Dec) vs quiet (03 Dec):

- **CC.PR03 (50 sps, source):** the flood raises power by ~10–20 dB **broadband from ~1 to 25 Hz** — the turbulence band — and is hard-capped at the **25 Hz Nyquist**. We never see 30–80 Hz.
- **UW.UPS (200 sps, lowland, can reach 100 Hz):** flood ≈ quiet across the whole band, including 30–80 Hz — i.e. **no river signal**; 54 km from the source the spectrum is a flat urban/instrument noise floor.
- **Net:** *no station in the network both reaches 30–80 Hz and sees the river.* The canonical bedload band is inaccessible here.

**Grain-size/impact assumptions for a low-frequency bedload claim (Cascade glacial rivers).**

- Intrinsic impact frequency ~ Hertzian contact-time corner: for a 1 m boulder it is **~200–300 Hz**; to put the corner at 10–20 Hz needs **12–28 m clasts** — physically impossible. So grain *contact* does not place bedload in our band.
- Attenuation *shifts the apparent peak down*: Tsai (2012, Fig. 1) shows the PSD peak moving from ~15–25 Hz at 200 m to ~5–8 Hz at 600 m. So distant bedload energy **can** leak into 5–15 Hz — but **non-uniquely** (degenerate with turbulence, which peaks at 5–7 Hz; Gimbert 2014).
- These rivers are **boulder/cobble-rich** (Vallance & Scott: lahar clasts to ~2 m; Beason: cobble–boulder beds), favoring **rolling/sliding** transport, which **violates the saltation assumption** of the Tsai model (Luong 2024 had to add inelastic/rolling modes; underpredicts flux 1–2 orders at shallow flow).
- Roth et al. (2017): in a comparable low-frequency setting, seismic–discharge hysteresis was driven by **turbulence + boundary roughness, not bedload** — a direct cautionary precedent.
- **No published D₅₀/D₈₄** for the Puyallup/Carbon/Nisqually channels — a real data gap.

**Verdict.** The robust, defensible signal at 1–15 Hz is **turbulent-flow (drainage) noise ∝ discharge**; a bedload contribution there is *possible but non-unique and unproven*. To claim bedload we would need (i) a near-channel ≥160-sps station to reach 30–80 Hz, (ii) a grain-size survey, (iii) the threshold-onset + hysteresis + $Q^{>1.4}$ scaling tests jointly. **Recommended reframing of the paper:** lead with *turbulent-flow seismology of a compound AR flood across a glacial-river transect* (solid), and present bedload as a **hypothesis bounded by the frequency limitation** rather than the headline.

---

## 2. A landslide/avalanche catalog as an "initial volume of possible erosion" (#5)

**Ground-truth catalogs that exist (for you to add):**

| Source | What | Use |
|---|---|---|
| **WA DNR WGS — WASLID** | statewide landslide inventory GIS (downloadable) | baseline polygons/volumes |
| **WGS Dec-2025 Landslide Clearinghouse** | event-specific (279 slides statewide; sparse for Pierce/Rainier) | event mapping |
| **Beason (NPS) Rainier debris-flow chronology** | the definitive Rainier debris-flow/aggradation record (must digitize) | source-area history |
| **NWAC** (Northwest Avalanche Center) | snow-avalanche observations | avalanche supply |
| **NASA COOLR / Global Landslide Catalog** | global, news-seeded | near-useless at this scale |

⚠️ **Reality check:** Mt. Rainier NP characterized its **Dec-2025 impacts as "minor"**; the **record flooding was downstream** (Orting levee, Puyallup/Carbon record stage), and the Carbon River access loss (Fairfax Bridge) was **April 2025**, pre-dating the flood. So the dramatic "headwater sediment pulse" premise may be overstated — the story is more **downstream conveyance/aggradation** than a Rainier-proper failure.

**Empirical/theoretical basis to turn a catalog into a sediment estimate.** A catalog of failures gives a *supply* volume, not a *transported-bedload* volume; the chain is:

$$
V_{\text{bedload}} \;\approx\; \underbrace{\alpha A^{\gamma}}_{\text{volume–area (Larsen 2010, }\gamma\approx1.1\text{–}1.3)} \;\times\; \underbrace{\text{SDR}}_{\text{delivery to channel, }0.1\text{–}0.5} \;\times\; \underbrace{f_{\text{bed}}}_{\text{bedload fraction, }0.05\text{–}0.15}
$$

- **Volume–area scaling** (Larsen et al. 2010; Guzzetti) converts mapped failure *area* → volume.
- **Sediment delivery ratio** (connectivity; Cavalli IC, Heckmann 2018): only a fraction of failed material reaches the channel.
- **Bedload fraction**: most delivered sediment moves as suspended/wash load; bedload is the coarse minority.
- **Horizon dependence:** a catalog is an **upper bound on single-event supply** but a **lower bound on multi-year transportable load** (material is evacuated over years–decades; Czuba 2012 transit times: Puyallup ~80 yr).

**So:** a catalog is a legitimate *independent, order-of-magnitude initial supply estimate*. Anchors: Czuba (2012) basin yields; Tahoma Creek debris-flow volumes ~10⁶–10⁷ m³; historical Puyallup bedload ~10⁴ m³/yr. State it explicitly as supply-side, with the connectivity/grain-size caveats, **not** as measured transport.

---

## 3. New framing — early warning: predicting the **downstream flood peak timing and amplitude**

This is the strongest new angle, and the data already support it (`fig10_early_warning.png`).

**Timing (observed):**

- Mainstem peak propagates **Electron (RM 41, 09 Dec 03:00) → Orting (RM 30, +36 h) → Puyallup (RM 10, +70 h)**.
- **Upstream seismic (CC.PR03, 5–15 Hz) leads the downstream Puyallup discharge peak by ~36 h.**
- *Caveat:* the 70 h is **not** pure flood-wave routing (celerity over ~50 km would be hours); the downstream gage integrates the **White River + tributaries** peaking later from their own rain. The lead = routing **+** progressive basin filling — but for warning purposes the lead time is what matters, and it is **tens of hours**.

**Amplitude:** Puyallup peak (1254 m³/s) is **4.1× the Electron peak** (306) — set by drainage-area growth + tributary confluence. Predicting downstream *amplitude* from upstream signals needs a **learned/routed relation** (one event cannot fit it), which is the ML target below.

**Why seismic adds value over gage-routing alone:** it senses **ungaged reaches** and sediment-laden flows (lahars) that rating curves miss, is **distributed and continuous**, and degrades gracefully when gages fail in floods — relevant for **natural, undammed** rivers and the Orting/Puyallup lahar-hazard corridor.

**Prior art & the gap:** Cook (2018, GLOF tracking), Walter/Coviello (debris-flow ASL lead time), the Rainier lahar-detection system, Burtin (2008, noise ∝ Q). **Gap:** no one forecasts downstream **clear-water stage/peak timing+amplitude** from **upstream seismic** with hours–day lead for a natural river.

### Proposed ML approach
- **Targets:** downstream discharge/stage $Q_{\text{down}}(t+\Delta)$ at lead $\Delta \in [6,48]$ h; specifically **peak-timing** and **peak-amplitude**.
- **Features:** multi-band seismic PSD (0.5–2 microseism, 2–10 turbulence, 15–45 high-f) at upstream stations + their time-lags; upstream stage/Q; SNOTEL precip + antecedent (rain-on-snow, SWE); static basin attributes.
- **Model:** **encoder–decoder LSTM** (Kratzert/Nearing "Google flood" lineage) for sequence-to-sequence forecasting; optionally a **hybrid differentiable routing** layer (Muskingum–Cunge; Bindas 2024) for physical interpretability and extrapolation.
- **Training discipline:** **multi-basin / multi-event pretraining is essential** — single-basin LSTMs fail (Kratzert 2019). Pretrain across PNW rivers and years, fine-tune on the Puyallup. Validate on **held-out years/events**; report peak-timing error, peak-amplitude error, NSE/KGE.
- **Data to collect (multi-year):** UW/CC + regional continuous waveforms **2016–present**; USGS NWIS 15-min stage/Q at all corridor gages; SNOTEL precip; an **event catalog of prior ARs** (e.g., 2006, 2009, 2015, 2019, 2021, 2022) to build training samples; ideally **2–3 near-channel ≥160-sps stations** to also resolve the bedload band.

**One-line pitch:** *Upstream river seismic noise forecasts downstream flood-peak timing (and, once trained, amplitude) for an undammed glacial river — a continuous, gap-tolerant complement to sparse stream gages in a lahar-hazard corridor.*

---

### New figures
- `fig10_early_warning.png` — downstream peak propagation + upstream-seismic lead (~36 h).
- `fig11_spectra.png` — flood-vs-quiet PSD showing the turbulence band, the 25 Hz Nyquist wall, and the absent 30–80 Hz river signal.

---

## 4. Grain size, boulders, large wood, and the choice of frequency band

**Can a *lower* band tune in to boulders/logs?** Short answer: **not by a
frequency-selective mechanism — grain size is encoded in *amplitude*, not in the
band.** In the Tsai (2012) model an impact is a near-instantaneous impulse (flat
spectrum), so to first order the spectral *shape* is grain-size-independent below
the Hertzian contact corner $f_c\approx1/t_c$, with $t_c\propto D\,w^{-1/5}$, i.e.
$f_c\propto1/D$. Computed corners (rock, $w\approx2$ m/s): D=1 cm → ~30 kHz,
10 cm → ~3 kHz, 0.5 m → ~570 Hz, 1 m → ~290 Hz, **2 m → ~140 Hz**. So even a 2 m
boulder's spectrum is still *flat* through our ≤50 Hz window — boulders and cobbles
share the same low-frequency shape and differ only in **amplitude** (per-impact
$\propto D^6$; flux-integrated PSD $\propto D^3$). A low band therefore acts as a
**coarse-transport amplitude proxy**, not a grain-size *discriminator*.

**Where low frequency genuinely carries coarse-clast / wood information:**
- **Boulder fronts of debris flows / lahars** radiate down to ~2 Hz and are
  dominated by the bouldery snout (force $\propto D^3 v^3$; Lai et al. 2018; USGS
  HF debris-flow model) — directly relevant to Rainier's lahar hazard.
- **Large woody debris (tree trunks):** low density and modulus give longer contact
  times → lower $f_c$, and floating/grinding contacts plausibly emit emergent
  low-frequency tremor. **This is a genuine literature gap** — no dedicated seismic
  source model for transported logs exists — and an opportunity given Rainier's
  old-growth wood.
- Burtin et al. (2008) is the closest precedent: 1–20 Hz tracked coarse transport
  (via hysteresis) in a steep incised reach.

**Implication:** the route to grain size here is **multi-band *amplitude*** (the
$D^3$ weighting toward the coarse tail), plus targeting the **debris-flow/lahar
boulder-front and wood low-frequency signatures**, rather than expecting a clean
"boulder band." Confirming any of it still needs a near-channel ≥160-sps station
and a grain-size survey. (Refs: Tsai 2012; Gimbert 2014; Bakker 2020; Lagarde 2021;
Lai 2018; USGS debris-flow model; Spreitzer 2022; Ruiz-Villanueva 2023.)

## 5. Threshold discharge Qc for transport onset (`fig14_threshold.png`)

Bedload is threshold-gated, so the seismic-power–discharge relation should steepen
above a critical discharge $Q_c$. A continuous broken-stick fit of $\log_{10}P$ vs
$\log_{10}Q$ (5–15 Hz) per station gives candidate thresholds:

| station (basin) | Qc (m³/s) | b below → above | onset? |
|---|---|---|---|
| UW.LON (Nisqually) | ~89 | 1.2 → **4.3** | yes (strong) |
| CC.GTWY (Nisqually) | ~93 | 1.1 → 2.7 | yes |
| CC.TRON (Carbon/Orting) | ~68 | −0.7 → 1.9 | yes |
| CC.PR01/PR02/SIFT (Puyallup) | ~35–37 | →1.8–2.0 | yes (modest) |
| CC.PR03, CC.STYX | — | slope *flattens* | no break |

So **several stations show a discharge threshold (~35 m³/s on the upper Puyallup,
~70–90 m³/s on the Nisqually/Carbon) above which the scaling steepens** —
consistent with a transport/process onset, strongest on the Nisqually (LON, GTWY).
Caveats: the band is turbulence-dominated, so this is a *candidate* onset, not a
confirmed bedload entrainment threshold; PR03/STYX show no upward break (low-Q
slope is noise-floor). Physical anchor: the slope-dependent critical Shields stress
$\tau^*_c=0.15\,S^{0.25}$ (Lamb et al. 2008) predicts *higher* $Q_c$ on steeper,
coarser reaches — qualitatively matching the higher Nisqually thresholds. The
March-2026 second event (queued) will test whether these $Q_c$ values recur
out-of-sample.

### Added figures
- `fig12_virtual_q.png`, `virtual_q_animation.gif` — seismic virtual discharge.
- `fig13_bedload_ch.png` — 30–50 Hz edge (100-sps UW; CC 500-sps CH is a 2026, not-yet-archived capability).
- `fig14_threshold.png` — broken-stick transport-onset thresholds.
