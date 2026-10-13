# Sealed predictions — Sensorium improvements, 2026-10

Written before the measurement/backfill runs they describe, per the
repo rule (sealed predictions before any sim/measurement run). Each
phase's section is signed and dated when sealed; scoring is appended
after the run, never edited into the predictions themselves.

---

## Phase 2 — binding integrity (sealed 2026-10-12)

Estimator (the report's own, verified by hand against its two
published values before sealing): for a Poisson visual-transient
stream of density λ_v = n_visual / duration, the chance that at
least one transient falls within ±W of a given onset is
1 − exp(−λ_v · 2W). FF3: 2735/2701.0 s → 39.7% (report: 39.7% ✓).
Dr Tran ep4: 601/436.1 s → 49.8% (report: ≈0.50 ✓).

Predictions for the backfill (`scripts/binding_baseline.py` over the
four canonical runs), chance rates ±2 pts:

| Run | n_visual / duration | Predicted chance (±250 ms) | Actual bound | Prediction: bound vs chance |
|---|---|---|---|---|
| FF3 1080p (`ff3_hd_av`) | 2735 / 2701.0 s | **39.7%** | 35.9% | Bound **below** chance (report already says this) |
| Star Tours AV (`st_stream_av_full`) | 328 / 269.0 s | **45.6%** | 38.2% | Bound **below** chance by rate — the causal signal is in the dt distribution, not the rate |
| Dr Tran ep4 (`drtran_ep4_av`) | 601 / 436.1 s | **49.8%** | 50.7% | Bound **≈ chance** (within ~1 pt; report: "equals") |
| Pink Panther (`pink-panther-cato-attacks`) | 260 / 200.1 s | **47.8%** | 52.4% | Bound **exceeds** chance by ~4–5 pts — the only rate-excess of the four |

Tight-dt predictions (fraction of bound pairs with |dt| ≤ 50 ms;
uniform-dt coincidence over a ±250 ms window expects ≈20%):

- **Star Tours shows the strongest tight-dt concentration of the
  four runs**, ≥35% of its bound pairs within |dt| ≤ 50 ms (the
  report's "~zero-dt concentration" / "tight where the film is
  multimodal," made numeric).
- FF3: |dt| ≤ 50 ms fraction ≈ 25–28% — dt is quantized to the
  50 ms moment grid and reported "roughly uniform," so the
  {-50, 0, +50} grid points should hold ≈3/11 of pairs; the
  report's ~159 exactly-zero-dt pairs are 13.2% of 1,208.
- Dr Tran ep4: |dt| ≤ 50 ms fraction ≈ 13% (the report's ~39 pairs
  of 298), i.e. *below* the 20% uniform expectation despite the
  50.7% headline rate — flat dt, density-driven.
- Pink Panther: |dt| ≤ 50 ms fraction ≈ 25% (modest excess over the
  20% uniform expectation, below Star Tours).
- Signed asymmetry: no strong directional prediction except that
  Star Tours' tight pairs skew audio-after-visual (real-world
  sound follows the visible event); FF3/Dr Tran near-symmetric.

Sealed: Wodehaus, 2026-10-12 (before `binding_baseline.py` was run
on any canonical run).

---

## Phase 3 — pathways (sealed 2026-10-12)

Items: (8) pursuit egomotion fix — pursuit requires RESIDUAL motion
(local motion surviving subtraction of the global full-field flow
estimate), `pursuit_mode: legacy|residual`; (10) content-sensitive
saccade clock — the streaming decision interval is modulated by the
batch path's darkness gain and by a causal activity EMA (magno
motion mean when the channel is on, transient-map mean otherwise),
clamped to an effective 1.5–5.0/s, `saccade_clock: fixed|adaptive`;
(13) causal L2 gate — the vis_boost reshape in
`hva/attention.py` (consumed by `scripts/couple_av.py` pass 3)
replaces the full-sequence 75th percentile with a trailing-window
estimator (past 300 moments, current moment excluded, no boost
until 30 past samples exist), `--gate-mode legacy` kept for A/B.

Legacy baselines (all read from on-disk records before sealing):
FF3 full run — 1,127 pursuits, 100% ≤ 0.41 s; segment 1200–1500 s:
**174 pursuits, 100% ≤ 0.41 s**, saccade rates per 60 s bin
2.65/2.80/2.85/2.70/2.75 (spread 0.20). Star Tours AV — **161
pursuits, 60.2 s total, all ≤ 0.40 s** (the chaining cadence caps
durations in the healthy regime too; preservation is therefore
judged on count and total time, not duration shape). Dr Tran ep4 —
**51 pursuits**, mean duration 0.39 s. Pink Panther legacy —
bedroom (0–25 s) **3.44/s**, brawl (25–130 s) **3.43/s**:
divergence 0.01/s, the metronome in numbers. Calibration (input
inspection, not validation): PP bedroom vs brawl mean luminance
0.19–0.32 vs 0.28–0.33 and transient-map means 0.051 vs 0.069 —
neither separates the regimes; magno motion energy is the signal
that does, so the clock validation runs with the motion channel
ON (pursuit off) under both clocks, the fixed run as control.

### Item 8 predictions (pursuit, residual mode)

- **P8.1 (segment legacy check):** the FF3 1200–1500 s segment run
  under legacy mode reproduces the disease: ≥150 pursuits, ≥98%
  of durations ≤ 0.41 s.
- **P8.2 (the fix):** the same segment under residual mode —
  pursuit count collapses to **≤15% of the legacy segment count**
  (≤26 if legacy lands at 174).
- **P8.3 (Star Tours preserved):** full Star Tours under residual
  mode: pursuit count within **90–230** (legacy 161) and total
  pursuit time **≥30 s** (legacy 60.2 s). Some drop is expected —
  ride footage is first-person and part of its legacy "pursuit"
  is flow-chasing — but the regime must not starve.
- **P8.4 (Dr Tran stays starved):** ep4 under residual mode:
  pursuit count **≤76** (≤1.5× legacy 51) and ≥70% of durations
  still ≤ 0.41 s. The starved regime is a property of the footage,
  not something the fix should manufacture motion for.
- **P8.5 (unit level):** synthetic uniform translating texture —
  residual mode produces **0 pursuits** over a 6 s stream while
  legacy produces ≥3; a local mover on a static background is
  pursued in both modes; a local mover on a translating
  background is pursued under residual mode.

### Item 10 predictions (adaptive saccade clock, Pink Panther)

- **P10.1 (divergence):** adaptive clock, motion on, pursuit off —
  bedroom (0–25 s) rate **≤2.8/s**, brawl (25–130 s) rate
  **≥4.0/s**, divergence **≥1.2/s**. Scored a hit if divergence
  ≥0.8/s with the ordering right (brawl faster), a miss below
  that, whatever the individual rates do.
- **P10.2 (control):** the fixed clock under identical settings
  keeps both segment rates within 3.2–3.7/s with divergence
  ≤0.3/s — the divergence, if it appears, is the clock's doing.
- **P10.3 (FF3 honesty check):** on the 1200–1500 s segment under
  the adaptive clock, the overall rate stays within 2.0–3.6/s and
  the 60 s-bin spread grows beyond the legacy 0.20 — predicted
  ≥0.35 — but does not explode (≤1.2). Uniform walking footage
  should modulate modestly, not wildly.

### Item 13 predictions (causal L2 gate, Dr Tran ep4 inputs)

L2 coupling inputs built for ep4 the way `couple_av.py` builds
them (batch-ear moments/Tprof from the episode audio; vis_boost
from the ep4 streaming run's per-frame transient energies, p99
normalized as in `fuse_av.py`). 4,361 moments expected (436.1 s
at 100 ms).

- **P13.1:** causal vs legacy gate changes the event label of
  **≥1% and ≤25%** of moments — some real change, not a rewrite.
- **P13.2:** ≥60% of the differing moments sit where the gate
  value actually moved: |vb_shaped_causal − vb_shaped_legacy|
  ≥ 0.05 at that moment.
- **P13.3:** onset-capture count under causal mode within
  **±25%** of the legacy count — the reshape is bounded (≤36%
  threshold effect), so captures should not swing wildly.
- **P13.4 (unit level):** with a huge transient appended late,
  causal-mode traces over the shared prefix are **bit-identical**
  moment-for-moment; legacy mode demonstrably changes prefix
  decisions (characterization test, expected to show the leak).

Sealed: Wodehaus, 2026-10-12 (before any Phase 3 code was written
or any validation run made).

### Phase 2 scoring (appended 2026-10-12, after the backfill)

Chance rates: **4/4 exact** — FF3 39.7 (pred 39.7), Star Tours
45.6 (45.6), Dr Tran 49.8 (49.8), Pink Panther 47.8 (47.8).
Bound-vs-chance calls: **4/4** — FF3 below (−3.8 pts), Star Tours
below by rate (−7.4), Dr Tran ≈ chance (+0.9), Pink Panther the
only excess (+4.6).

Tight-dt: Star Tours strongest **hit** under both definitions
(|dt| ≤ 50 ms: 46.2%, highest; zero-dt bin: 15.4%, highest), and
the ≥35% call hit (46.2%). The other tight-fraction point
predictions **missed**: FF3 40.6% (pred 25–28%), Dr Tran 38.3%
(pred ≈13%), Pink Panther 31.5% (pred ≈25%). Two causes, both now
on record: (1) the report's "~39 pairs within 50 ms" for Dr Tran
was its **zero-dt** count — the sidecar histogram's 0–25 ms bin is
exactly 39, and FF3's "~159 zero-dt" is exactly its 159 — so the
13% prediction inherited a loose phrase, not a wrong count;
(2) nearest-neighbor matching plus the 50 ms moment grid
concentrate dt beyond the uniform-grid model I predicted from
(dt is quantized: the 25–50 ms bin is empty in all four runs;
pairs sit at 0/±50/±100…). Asymmetry: Star Tours audio-after skew
hit (0.47 vs 0.37), FF3 symmetric hit (0.433 vs 0.435); Dr Tran
skewed audio-after (0.50 vs 0.37) where I predicted symmetric —
partial miss.
