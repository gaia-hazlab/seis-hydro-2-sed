# Independent topographic / LiDAR data for the Puyallup braidplain (deposition evidence hunt)

Searched 2026-06-27 (for the M7 deposition / DEM-of-Difference question, #14). Goal:
independent channel-change/deposition evidence to corroborate the seismic
slow-recession clogging mechanism. **Bottom line: the aggrading/reorganizing *regime*
is independently documented (2002–2022 lidar), but NO survey yet brackets the
Dec-2025 event — a post-event lidar over Electron is the decisive missing constraint.**

Geographic note: **Electron sits ~6 mi (downstream/west) outside the Mt. Rainier
National Park boundary**, so in-park-only datasets (2021 NPS SfM, etc.) do **not**
cover the PR braidplain; the watershed-wide USGS lidar products do.

## Ranked, confirmed datasets
1. **★ USGS Puyallup watershed release v2.0 — DOI 10.5066/P149MBYG** (`anderson2025`)
   + companion paper **Anderson (2026), EarthArXiv doi:10.31223/X5HR0N** (`anderson2026`).
   Repeat aerial-lidar **DEMs-of-Difference, 2002–2022**, upper Puyallup/Carbon/Mowich
   (includes the Electron reach); + 2023 lower-river cross-sections; + 1990s
   photogrammetry. Quantifies ~**1.3 ± 0.3 Myd³** valley-reach accumulation 2004–2020
   (~80,000 yd³/yr) with net-erosional deglaciating headwaters. **Best independent
   corroboration of the aggrading/reorganizing regime — but ends 2022 (pre-event).**
2. **WA DNR Lidar Portal** — <https://lidarportal.dnr.wa.gov/>. Action item: pull the
   2002–2022 epoch tiles over 46.90–46.92 °N / 122.03–122.05 °W to build an
   **Electron-specific** DoD, and check for any 2025–2026 post-flood refresh (not
   confirmed remotely).
3. **Anderson & Shean (2022) proglacial DEMs — DOI 10.5066/P9056ZNG** (esp.5274):
   HSfM 1960–2017, four Rainier proglacial basins (incl. Nisqually headwaters);
   *upstream* of Electron — sediment-supply context only.
4. **Czuba/Magirl cross-sections** — SIR 2010-5240 (156 sections, 1984→2009; lower
   Puyallup/White/Carbon aggradation up to 7.5 ft) and OFR 2012-1242 (basin-wide
   sediment transport; incl. Nisqually). Long-baseline aggradation, lower river.

## Honest gaps
- **No confirmed post-Dec-2025 lidar** in public catalogs as of June 2026 → no
  event-spanning DoD exists yet. (USGS "Puyallup Flood Alert" page exists; a lower-
  Nisqually flood-profile product is catalogued; no repeat-topo epoch found.)
- **Nisqually** is poorly served by DoD products (excluded from the Puyallup release).

## How it's used in the manuscript (commit 99321cc)
- §sec-braided "aggrading glacial-outwash braidplain" is now *grounded* in the
  2002–2022 lidar record [@anderson2025; @anderson2026; @czuba2012] — not assumed.
- The M7 mechanism paragraph states the decisive independent test = a DoD spanning the
  flood; the lidar confirms the regime, post-event acquisition is the open constraint.
