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

### Phase 3 scoring (appended 2026-10-12, after validation)

All media runs used the flagship settings (640×360, motion-weight
1.0, 50 ms moments) with audio off; pursuit and clock behavior do
not depend on the audio path. The FF3 segment was re-encoded from
`ff3_full_hd.mp4` (1200–1500 s, `input/ff3_seg_1200_1500.mp4`).

**Item 8 — residual pursuit: 4/5 hit, default flipped to
residual.**
- P8.1 **HIT** — segment under legacy: 159 pursuits (pred ≥150),
  100% of durations ≤ 0.41 s (pred ≥98%). The disease reproduces
  on the re-encoded segment (the original run's window count was
  174; the segment is a re-encode, not the identical frames).
- P8.2 **MISS** — segment under residual: **41 pursuits = 25.8%**
  of legacy, against the sealed ≤15%. The collapse is large
  (−74%) but short of the sealed bar. The survivors are not the
  translation disease — phase correlation cancels translation
  only; camera rotation/zoom and genuinely local movers leave
  residual motion by definition. (Cross-check under residual +
  adaptive clock: 16 pursuits, 10.1% — inside the sealed band,
  but that is not the comparison that was sealed, and it is not
  counted as the hit.)
- P8.3 **HIT** — Star Tours under residual: 120 pursuits (band
  90–230), 45.1 s total (pred ≥30 s; legacy 161 / 60.2 s).
- P8.4 **HIT** — Dr Tran ep4 under residual: 49 pursuits (pred
  ≤76; legacy 51), 100% of durations ≤ 0.41 s (pred ≥70%). The
  starved regime is unchanged, as sealed.
- P8.5 **HIT** — unit level: uniform translating texture gives 0
  residual pursuits vs ≥3 legacy (observed 0 vs 4); local mover
  pursued in both modes on static backgrounds (uniform and
  textured); mover on a translating background is pursued under
  residual with the mover's direction, not the flow's.
- Disposition: the flip to `pursuit_mode="residual"` as the
  driver default was made despite the P8.2 miss — the mechanism
  is exact where it is defined (translation), both preservation
  regimes held, and the sealed threshold was a round number the
  real footage's rotational content does not respect. Recorded
  here so the judgment call is visible, not laundered.

**Item 10 — adaptive saccade clock: 2/3 hit, default stays
fixed.**
- P10.1 **MISS** — Pink Panther under the adaptive clock:
  bedroom 4.04/s (pred ≤2.8), brawl 3.99/s (pred ≥4.0, a
  rounding-distance hit on its own), divergence 0.05/s (sealed
  hit bar ≥0.8). Two measured causes, both on me: (1) the
  activity statistic saturates — whole-frame motion means are
  asleep 0.0130, sneak 0.0210, brawl 0.0258, and everything
  above ACT_REF×1.6 (= 0.016) clips to the same busy factor, so
  the bedroom (whose back half is 15 s of continuous sneaking —
  the beat map said "stealthy, coiled movement" and I sealed a
  "quiet bedroom" prediction anyway) and the brawl land on the
  same interval; (2) the clip's total motion-mean dynamic range
  is only ~2.5×, narrower than the design assumed. The clock
  did modulate where the signal separated: the late segment
  (motion mean 0.0105) slowed to 3.09/s vs 3.41/s fixed. Unit
  level the mechanism is exact (dark-static 1.43/s vs
  bright-busy 5.0/s, bounds held, fixed-clock control flat).
- P10.2 **HIT** — fixed-clock control under identical motion-on
  settings: bedroom 3.44/s, brawl 3.43/s, divergence 0.01
  (pred: both 3.2–3.7, divergence ≤0.3).
- P10.3 **HIT** — FF3 segment under the adaptive clock: overall
  3.15/s (band 2.0–3.6); 60 s-bin rates 2.83/3.00/2.93/3.12/3.85,
  spread 1.02 (pred ≥0.35 and ≤1.2; legacy spread 0.20).
- Disposition: `saccade_clock` ships opt-in, default "fixed" —
  the flip condition was the headline divergence and it missed.
  The follow-up the data points to (a peak/top-decile motion
  statistic instead of the whole-frame mean, plus a wider busy
  range) is a NEW experiment needing its own sealed predictions,
  not a re-roll of this one.

**Item 13 — causal L2 gate: 3/4 hit, causal is the default.**
(Validation: `scripts/l2_gate_compare.py` on Dr Tran ep4 inputs,
4,361 moments; also `scripts/couple_av.py` end-to-end on its
native Star Tours inputs: causal 41 captures / 66 switches vs
legacy 38 / 68 — same family, no breakage.)
- P13.1 **HIT** — event-label diffs 624 moments = 14.31% (sealed
  band 1–25%). The wider trace divergence including cf_bin is
  39.99% — the attended-band trajectory cascades from event
  diffs; the sealed wording was event labels and is scored as
  written.
- P13.2 **MISS** — only 21.6% of event-differing moments have
  |Δshaped| ≥ 0.05 (sealed ≥60%). Cause: state cascade — one
  flipped decision changes cf/habituation/IOR for later moments,
  whose events then differ where the two gates agree exactly;
  and small gate deltas (0.01–0.05) flip borderline captures, a
  channel my "diffs live where the gate moved a lot" model
  underweighted. The gate's own causality is exact (P13.4);
  this prediction was about trace geography and it was wrong.
- P13.3 **HIT** — captures 212 legacy → 196 causal = −7.5%
  (sealed ±25%).
- P13.4 **HIT** — unit level: causal traces over a shared prefix
  are bit-identical under a late appended transient (both the
  shaped series and full AuditoryAttention traces); the legacy
  leak is pinned by a characterization test that must keep
  passing while legacy mode exists.

Suite after Phase 3: **282 passed, 1 warning, exit 0** (full
run captured to file and read from the file, per the house rule
on unreliable background digests; 267 at Phase 2 + 15 new tests:
7 pursuit, 3 clock, 5 gate). One pre-existing test was pinned to
`pursuit_mode="legacy"` explicitly when the default flipped —
its subject is the pursuit mechanism as originally validated.

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

---

## Phase 4 — fixtures: the quiet/near-silent regime (sealed 2026-10-12)

Item 11's missing regime, built as a GENERATED fixture so it is
reproducible from the repo alone: `scripts/make_fixtures.py`
renders `input/quiet_room_60s.mp4` deterministically via ffmpeg
lavfi — 60 s, 640×360 @ 30 fps, near-black base (0x050505, luma
≈ 0.020, below DARK_LUM 0.03) with ONE slow faint brightening
(gaussian, peak +0.045 luma, centered t = 42 s); audio is a
seeded brown-noise floor at ≈ −58 dBFS with exactly ONE planted
event: an 880 Hz chime, 0.6 s with 50 ms fades, ≈ −30 dBFS peak,
at t = 30.0 s.

Runs that score these predictions: (R1) streaming flagship
settings (motion-weight 1.0, pursuit on — residual is the driver
default, audio on, 50 ms moments) under the FIXED clock →
`output/quiet_room_60s/`; (R2) identical under the ADAPTIVE
clock → `output/quiet_room_60s_adaptive/`; (R3) the batch ear
(cochlea → salience → moments → `AuditoryAttention.run`, no
vis_boost — the construction `scripts/couple_av.py` uses) on the
same audio, because the silence-dwell rule lives only in the
batch ear; the streaming ear has no dwell rule to engage.

- **P4.1 (onsets):** R1 reports **1–2 audio onsets** total, the
  strongest within **±0.5 s of t = 30.0 s**. The RMS floor
  (1e-3) keeps the −58 dBFS floor from onsetting; the chime at
  ~0.03 peak clears it exactly once (a second pick from the
  chime's decay is the allowed miss-by-one).
- **P4.2 (pursuits):** **0 pursuits** in R1 and R2. The fixture
  has no coherent local motion; the brightening is global and
  slow, and residual mode subtracts global flow by construction.
- **P4.3 (motion / binding):** R1 `motion_energies` col-1 mean
  **< 0.002** (≥6× below Dr Tran ep4's 0.0125, the quietest
  existing corpus run); visual transients **≤ 3**; **n_av_bound
  = 0** — the chime (30 s) and the brightening (~42 s) sit 12 s
  apart, far outside the ±250 ms binding window.
- **P4.4 (fixed clock is darkness-blind):** R1 saccade rate
  stays in the metronome band **2.5–3.5/s** (n_saccades
  150–210 over 60 s) despite the near-dark scene.
  `_decision_interval_ms` returns the base interval
  unconditionally in fixed mode — darkness_gain is consulted
  only by the adaptive branch — so any rate drop in R1 falsifies
  my reading of the driver.
- **P4.5 (adaptive clock: darkness observable):** R2 saccade
  count **≤ 65% of R1's** and **≤ 114** (rate ≤ 1.9/s); median
  inter-saccade interval **≥ 500 ms** (vs ~290 ms fixed).
  Mechanism: luminance EMA ≈ 0.02 → darkness_gain ≈ 1.67; the
  activity EMA sits at the BUSY_MIN clip (0.70) → interval
  ≈ 681 ms → clipped at CLOCK_MAX_MS 667 ms ≈ 1.5/s.
- **P4.6 (batch-ear dwell rule engages):** R3 over the 600
  moments: event **"dwell-quiet" on ≥ 75%** of moments;
  onset-capture + switch events **≤ 6 total**, all within
  **2 s after the chime**; **≥ 1 onset-capture within 1.0 s**
  of t = 30.0 s (the chime's temporal contrast dominates an
  otherwise silent profile); longest run of consecutive
  non-capture moments **≥ 250** (25 s).

Sealed: Wodehaus, 2026-10-12 (before `make_fixtures.py` was
written, before the fixture was generated, before any Phase 4
run).
