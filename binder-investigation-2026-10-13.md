# Binder investigation — why FF3 (and Star Tours) bind BELOW the Poisson chance baseline at ±250 ms

Date: 2026-10-13. Scope: `streaming/bind_av.py` (greedy one-to-one matcher, ±250 ms,
`dt = audio − visual`), `scripts/binding_baseline.py` (Poisson chance
`1 − exp(−λ_v·2W)`), and the binding artifacts on disk. Every number below was
read/computed by the investigator from the JSON artifacts named — brief-supplied
figures were treated as leads and re-derived.

---

## SEALED PREDICTIONS — written BEFORE the shifted-null run (2026-10-13)

Written after reading the binder code and the headline counts only, before any
null-model computation and before any burstiness statistic was computed.

Reasoning sketch (from the code): the Poisson baseline answers "what fraction of
audio onsets would have ≥1 visual transient within ±W if visual transients were a
uniform Poisson process?" The binder answers a different question: "what fraction
of audio onsets WIN a distinct visual transient under greedy one-to-one matching?"
A transient that is the only candidate for two onsets is counted twice by the
baseline's expectation and once by the binder. At λ_a ≈ 1.24/s (FF3) many onsets
share candidate transients, so the binder's rate should sit below the Poisson
figure even on pure noise. Additionally both streams live on the 50 ms moment
grid: |dt| can only be 0, 50, 100, … ms, so the inclusive tight window |dt| ≤ 50 ms
contains **three** grid slots (−50, 0, +50), an effective discrete width the
continuous `2W = 100 ms` formula does not price in — the tight-window Poisson
chance should therefore UNDERSTATE the grid chance.

- **SP.1 (FF3 wide shifted null):** a circular time-shift null (visual stream
  shifted by ≥5 s offsets, multiples of 50 ms, same greedy matcher) will give a
  mean bound rate at ±250 ms **below** the Poisson 0.3973, in **0.32–0.37**, and
  the observed 0.3593 will fall **inside** the null distribution (within its
  central ~90%). MISS if null mean > 0.385 or observed is below the null's 5th
  percentile.
- **SP.2 (FF3 tight shifted null):** the shifted-null tight rate (|dt| ≤ 50 ms)
  will be **0.11–0.15**, above the continuous-Poisson 0.0963; the observed
  0.1457 will sit at or only slightly above the null mean (residual tight excess
  ≤ ~3 pts), i.e. most of the apparent +4.9 pt tight excess is grid pricing.
  MISS if null tight mean < 0.10 or observed exceeds null mean by > 5 pts.
- **SP.3 (Star Tours):** shifted-null wide rate **0.35–0.44** (Poisson: 0.4564);
  observed 0.3824 inside the null. Tight: null **0.13–0.17**, observed 0.1765 at
  or modestly above the null mean — a small genuine tight excess survives for
  Star Tours but not the headline −7.4 pt wide deficit. MISS if ST observed wide
  falls below the null 5th percentile.
- **SP.4 (burstiness):** both streams in both materials are over-dispersed vs
  Poisson: inter-event-interval CV > 1 and 60 s-window Fano factor > 2 for at
  least the visual streams. MISS if all CVs ≤ 1.
- **SP.5 (dt = 0 slot):** in the real FF3/ST pairings the dt = 0 count will exceed
  the shifted-null mean dt = 0 count (true on-grid synchrony), even where the
  wide-window rate does not beat the null. MISS if real dt = 0 ≤ null mean for both.

Sealed: Wodehaus (subagent), 2026-10-13, before running
`binder-investigation` null/burstiness computations.

---

## Evidence trail (verified from artifacts)

(Sections below appended after the sealed predictions, from computations whose
outputs are captured in `binder-investigation-2026-10-13-output.txt`
(moved 2026-10-14 to `improvements-2026-10/logs/binder-investigation/`).

## SEALED AMENDMENT — maximum-matching diagnostic (written before running it, 2026-10-13)

Question: how much of the one-to-one loss is the greedy strength-order *stealing*
(a strong onset taking a transient that was a weaker onset's only candidate,
while the strong onset had alternatives) vs a hard ceiling (at-least-one count)?

- **SM.1:** maximum bipartite matching (|dt| ≤ 250 ms) for FF3 lands in
  **1,215–1,250** pairs (greedy: 1,208) — greedy loses only ~0.5–3.5% to
  ordering; its rate stays **below** Poisson chance 0.3973.
- **SM.2:** for Star Tours, maximum matching lands in **91–96** (greedy: 91),
  rate still below Poisson 0.4564 and at/below the at-least-one ceiling 0.4034.

Sealed before the Hopcroft–Karp computation below.

## Evidence trail (verified from artifacts)

Files read directly: `output/ff3_hd_av/av_binding.json` +
`output/ff3_hd_av/av_binding_baseline.json`,
`output/ff3_hd_av_2026-10-13/av_binding.json` (+ its `audio_events.json`,
`run_report.json`), `output/st_stream_av_full/av_binding.json` +
`av_binding_baseline.json`, and the control runs' `av_binding.json` under
`output/exp-2026-10-13/` and `output/drtran_ep4_av/`. Computations captured in
`binder-investigation-2026-10-13-output.txt`,
`binder-investigation-2026-10-13-output2.txt`,
`binder-investigation-2026-10-13-output3.txt`
(moved 2026-10-14 to `improvements-2026-10/logs/binder-investigation/`).

### Headline numbers — verified, not inherited

| Material | nA | nV | bound | bound rate | Poisson chance | excess | tight rate (\|dt\|≤50) | tight Poisson | tight fraction of bound | λ_v /s | dur s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FF3 (Oct 4 = 2026 rerun, pair-identical) | 3362 | 2735 | 1208 | 0.3593 | 0.3973 | −0.0380 | 0.1457 | 0.0963 | 0.4056 | 1.0126 | 2701.0 |
| Star Tours | 238 | 328 | 91 | 0.3824 | 0.4564 | −0.0741 | 0.1765 | 0.1148 | 0.4615 | 1.2192 | 269.0 |
| Dr Tran ep4 | 588 | 601 | 298 | 0.5068 | 0.4980 | +0.0088 | 0.1939 | 0.1287 | 0.3826 | 1.3782 | 436.1 |
| Pink Panther (motion-on) | 170 | 260 | 89 | 0.5235 | 0.4778 | +0.0457 | 0.1647 | 0.1219 | 0.3146 | 1.2995 | 200.1 |
| YTPMV (music) | 362 | 534 | 276 | 0.7624 | 0.6853 | +0.0772 | 0.3840 | 0.2064 | 0.5036 | 2.3120 | 231.0 |
| dark-busy (planted) | 54 | 56 | 27 | 0.5000 | 0.3729 | +0.1271 | 0.1852 | 0.0891 | 0.3704 | 0.9333 | 60.0 |

FF3 details verified from the sidecar/pairs: dt values are exactly
{−250,−200,…,0,…,250} ms (counter: dt=0 → 159; ±50 → 178/153; ±100 → 120/126;
±150 → 106/98; ±200 → 68/81; ±250 → 54/65); asymmetry 523 after / 526 before /
159 simultaneous; mean |dt| 108.7 ms, median 100 ms. Star Tours: dt=0 → 14;
43 after / 34 before; mean |dt| 103.3 ms.

Oct 4 vs 2026-10-13 FF3: `bound_pairs`, `unbound_audio`, `unbound_visual`, and
`audio_events.json` are **exactly identical** (verified by direct comparison).

### Q1(a) — the deficit decomposes exactly: baseline counts non-exclusively, the binder matches exclusively

The Poisson figure is the probability that **≥1** transient falls in an onset's
window — a transient serving five onsets counts five times in that expectation.
The binder is one-to-one. Decomposing each deficit (all quantities computed from
the stored event lists; they sum exactly):

**FF3 (−3.80 pts):** Poisson 0.3973 → empirical at-least-one rate **0.3923**
(1,319 of 3,362 onsets have ≥1 candidate; non-uniformity net effect **−0.49 pt**)
→ bound 0.3593 (one-to-one exclusivity loss **−3.30 pts**, 111 onsets).
Of the exclusivity loss: maximum bipartite matching (Hopcroft–Karp, sealed
SM.1/SM.2) reaches only **1,239 pairs (0.3685)** — so greedy ordering costs
31 pairs (**0.92 pt**, 2.6% of bound) and the remaining **2.38 pts** are a hard
structural ceiling: in visual-quiet stretches several onsets share too few
transients, and no matcher can bind them.

**Star Tours (−7.41 pts):** Poisson 0.4564 → at-least-one **0.4034** (96 of 238;
non-uniformity **−5.31 pts**) → bound 0.3824 (exclusivity **−2.10 pts**, 5 onsets;
max matching 92, so greedy costs exactly 1 pair / 0.42 pt).

Saturation behaviour confirms the mechanism: re-binding FF3 at wider windows,
the deficit vs Poisson grows monotonically (−3.3 pts at 250 ms, −19.7 at
500 ms, −34.6 at 1,000 ms) — the at-least-one formula saturates toward 1.0 while
real matching is capped by *local* transient supply (global cap nV/nA = 0.814
for FF3). The baseline, not the binder, is what breaks at the wide window.

Contention context: mean visual candidates per onset ≈ 0.55 in both films;
13.8% of FF3 onsets have ≥2 candidates; 11.1% of FF3 transients are contested
by ≥2 onsets (Star Tours 5.5%). Audio onsets themselves never cluster tighter
than the window: minimum audio inter-onset interval is **0.250 s in both films**
(the onset picker is effectively refractory at the window scale), so competition
is driven by the *visual* side's clustering.

### Q1(b) — burstiness, and the rate-matched non-uniform null (sealed SP.1–SP.5)

Burstiness (from the stored event times):

| Stream | IEI CV | IEI median | max IEI | Fano (10 s) | Fano (60 s) | frac IEI ≤0.5 s |
|---|---|---|---|---|---|---|
| FF3 audio | 0.858 | 0.55 s | 7.9 s | 0.89 | 2.05 | 0.468 |
| FF3 visual | **1.879** | 0.40 s | **26.2 s** | **5.30** | **14.80** | 0.612 |
| ST audio | 0.998 | 0.70 s | 7.5 s | 1.08 | 0.67 | 0.371 |
| ST visual | **2.026** | 0.35 s | **16.9 s** | **6.22** | **13.89** | 0.661 |

The visual streams are violently non-Poisson (CV ≈ 2, 60 s Fano ≈ 14–15, long
dead stretches next to cut-dense bursts — 12 of FF3's 270 ten-second windows
contain zero transients); the audio streams are near-Poisson by IEI. A uniform
Poisson baseline misprices this in two directions at once (Jensen: clustering
alone lowers the at-least-one rate; co-location of the two streams' bursts
raises it) — which is why a *stream-preserving* null is the right instrument.

**Shifted-null design:** circularly shift the visual stream by 200 seeded
offsets (uniform in [10 s, duration−10 s], multiples of 50 ms, wrap modulo the
duration in grid slots), preserving each stream's full IEI/burst structure and
the 50 ms grid, re-run the **same greedy matcher**. One methodological find was
required first: re-binding from the rounded JSON times with the current code
gives 1,226 FF3 pairs, not 1,208 — the run files are only reproduced **exactly**
(1,208/1,208 and 91/91 pairs) when visual times carry their runtime float form
(`slot × 0.05`, binary dust and all), because on-grid ties are decided by that
dust. The null below is therefore run slot-faithfully; its matcher reproduces
both files exactly at zero shift. (This dust sensitivity is itself a finding —
see Verdict/Recommendations.)

**Shifted-null results (200 shifts, seed 20261013):**

| Statistic | FF3 observed | null mean ± sd | [p5, p95] | obs percentile | ST observed | null mean ± sd | [p5, p95] | obs percentile |
|---|---|---|---|---|---|---|---|---|
| wide bound rate | 0.3593 | 0.3554 ± 0.0064 | [0.3447, 0.3656] | 74 | 0.3824 | 0.4341 ± 0.0331 | [0.3782, 0.4832] | 6.5 |
| tight rate \|dt\|≤50 | 0.1457 | 0.1461 ± 0.0058 | [0.1374, 0.1556] | 44 | 0.1765 | 0.1802 ± 0.0230 | [0.1429, 0.2143] | 46.5 |
| dt = 0 count | 159 | 164.9 ± 12.4 | [143, 184] | 28.5 | 14 | 14.4 ± 3.8 | [9, 21] | 40.5 |
| mean \|dt\| (ms) | 108.7 | 105.4 ± 2.1 | [102.1, 108.9] | 94 | 103.3 | 103.8 ± 7.1 | [91.8, 114.0] | 49 |

**FF3's observed binding is, statistic for statistic, a draw from its own
shifted null.** The wide rate sits *above* the null mean; the tight rate is at
the null mean to four decimal places of practical significance; there is no
dt = 0 spike (observed is *below* the null mean); if anything the real dt's are
slightly *wider*-spread than the null's (mean |dt| percentile 94) — no hidden
tight synchrony the wide window was masking. Star Tours: tight rate, dt = 0 and
mean |dt| all dead on the null; the wide rate is borderline low (percentile
6.5, just inside the 90% band) — its segment structure (below) is the story,
not a matcher effect.

Why is Star Tours' wide rate low even against shifts? Per-10 s segment counts
of the two streams **anti-correlate** in Star Tours (**r = −0.288**), are
uncorrelated in FF3 (**r = −0.001**), and **co-locate** in every above-chance
control: Pink Panther +0.456, YTPMV +0.562, dark-busy +0.578, Dr Tran +0.057.
That is the shared signature of the two below-chance materials: **their audio
and visual bursts do not live in the same segments** (FF3 neutrally, Star Tours
negatively — dialogue/music stretches with few transients alternate with
motion stretches whose continuous ride noise yields few picked onsets), while
every above-chance material puts its bursts together. Circular shifting
destroys segment alignment, so the null prices "bursts exist but are
unrelated"; Star Tours' real arrangement is slightly worse than unrelated.

### Q1(c) — dt quantization and the empty 25–50 ms bin

Both streams' stored times are 100% on the 50 ms grid (verified: 0 off-grid
times in either film, either stream), so dt takes only multiples of 50 ms. The
histogram edges (0, 25, 50, 100, 250) are continuous-time edges applied to a
discrete variable: the [25, 50) bin is **structurally empty** — no dt can ever
land in it — and the bin labelled [50, 100) in fact contains exactly the
|dt| = 50 ms pairs (331 in FF3), while [0, 25) contains exactly dt = 0 (159).
The zero is not missing data and not a dead zone in the material; it is a
labelling artifact of `DT_BIN_EDGES_MS` in `bind_av.py`.

The same discreteness misprices the tight-window chance: the continuous
formula prices a 100 ms-wide window, but on the grid the inclusive window
|dt| ≤ 50 ms contains **three slots** (−50, 0, +50), the discrete analogue of a
150 ms width. Grid-corrected chance `1 − exp(−λ_v·(2W + 50 ms))`: FF3 tight
0.0963 → **0.1409** (observed 0.1457: true excess **+0.5 pt**, not +4.9);
Star Tours 0.1148 → **0.1671** (observed 0.1765: **+0.9 pt**, not +6.2). The
shifted null (0.1461 / 0.1802) brackets the grid correction from the other
side and agrees: essentially all of the celebrated tight excess in both films
is grid pricing. (At the wide window the grid correction runs the other way —
0.3973 → 0.4270 for FF3 — making the wide deficit *larger* against a
grid-Poisson, which is exactly why the matcher-in-the-loop null, not either
closed form, is the verdict-grade baseline.) For calibration, grid-corrected
tight excesses across the corpus: YTPMV **+9.1 pts** (genuine), dark-busy
+5.5, Dr Tran +0.7, Star Tours +0.9, FF3 +0.5, Pink Panther −1.2.

### Q1(d) — what the improvement phases changed for binding: nothing upstream, additive reporting only

Verified from the artifacts: between FF3 Oct 4 and the 2026-10-13 rerun,
`audio_events.json` is identical, the binding input sets (bound ∪ unbound, both
streams) are identical, and the bound pairs are identical. `frame_energies.npy`
and `audio_features.npy` are byte-identical (md5); `motion_energies.npy`,
`fixations.npy`, `scanpath.npy` **differ** — the vision-pathway upgrades
(residual pursuit etc.) changed the motion/gaze products, but binding's visual
input derives from `driver.energies` totals (`run_stream.py`: `e_tot` →
`visual_transients`), i.e. from the unchanged frame energies, so the transient
list could not move. In code, the matcher in today's `bind_av.py` is
algorithm-identical to the original commit 1172fa2 (2026-10-03), and commit
a6688e5 (Phase 2) states "matcher unchanged" while adding the baseline fields —
which is why the Oct 4 file needs the sidecar and the 2026 file carries
`bound_rate`/`chance_rate_poisson`/`tight_windows`/`dt_summary` inline. Net:
the phases changed binding's *reporting*, not its inputs or its matching.

### Q2 — Star Tours' "stronger dt clustering" is mostly a denominator artifact

Premise check first: Star Tours does **not** lead the corpus on tight binding
in absolute terms. Its tight *rate* (0.1765 of onsets) is below YTPMV (0.3840),
Dr Tran (0.1939) and dark-busy (0.1852); its tight *fraction of bound* (0.4615)
is below YTPMV's 0.5036. What it leads FF3 on is the fraction — and the
fraction is high because the denominator is depleted: Star Tours is missing
its *loose* pairs (the segment anti-co-location suppresses wide-window
coincidences), so the pairs that remain are disproportionately the tight ones.
Against its own shifted null its tight rate is ordinary (percentile 46.5),
its dt = 0 count is ordinary (14 vs null 14.4), its mean |dt| is the null's
(103.3 vs 103.8 ms).

The ride-causation hypothesis (picture-synced impacts/blasts vs FF3's loosely
coupled exertion audio) gets only directional, sub-noise support: Star Tours'
tight excess over grid chance (+0.9 pt) is ~double FF3's (+0.5 pt); its
asymmetry tilts audio-after-visual (43 vs 34; FF3 is symmetric 523/526) as
sound-reacts-to-picture would predict; its dt = 0 share of bound (15.4%) edges
FF3's (13.2%). But every one of those sits inside the shifted null's spread
(at n = 238 the null sd on the tight rate is ±2.3 pts), so the corpus cannot
currently *establish* a causal-sync advantage for the ride film — an advantage
of up to ~+2 pts could hide under the noise, and none larger. FF3's exertion
story, by contrast, is positively consistent: segment correlation ≈ 0, every
binding statistic at null, mean |dt| slightly wide of null — breathing and
footsteps that simply do not time-lock to picture transients at 50 ms
resolution, in a film whose genuine AV events (hits, impacts) are too small a
minority of 3,362 onsets to move the aggregate.

### Sealed-prediction scoring

- **SP.1 HIT** — FF3 shifted-null wide mean 0.3554 (sealed 0.32–0.37, below
  Poisson 0.3973 ✓); observed 0.3593 inside [p5, p95], percentile 74.
- **SP.2 HIT** — tight null mean 0.1461 (sealed 0.11–0.15, above the continuous
  0.0963 ✓); observed at the null mean (residual −0.0004, sealed ≤ ~3 pts ✓).
- **SP.3 PARTIAL** — direction right (ST null 0.4341 < Poisson 0.4564; tight
  null 0.1802 vs sealed band 0.13–0.17, just above it); observed wide rate is
  borderline-low vs the null (percentile 6.5) rather than comfortably inside —
  the segment anti-correlation (found post-hoc, r = −0.288) is the extra
  mechanism the sealed model lacked.
- **SP.4 HIT as stated** — visual streams: CV 1.88 / 2.03, Fano-60 s 14.8 /
  13.9. (Audio streams are *not* IEI-overdispersed: CV 0.86 / 1.00 — the
  burstiness is a visual-stream property, and the memo's mechanism sections
  say so.)
- **SP.5 MISS** — no dt = 0 excess in either film: FF3 159 vs null mean 164.9;
  Star Tours 14 vs 14.4. Filed as a miss; it strengthens the verdict (there is
  no exact-synchrony signal in either material for the wide window to hide).
- **SM.1 / SM.2 HIT** — max matching FF3 1,239 (sealed 1,215–1,250; rate
  0.3685 < 0.3973 ✓); Star Tours 92 (sealed 91–96; 0.3866 < 0.4564 ✓).

## Verdict

**FF3's below-chance wide-window rate is a measurement artifact of the
baseline, layered over a material property — with only a minor binder defect
in the tail. It is not evidence of failed or anti-binding.**

Proportions, from the exact decomposition of FF3's −3.8 pts: **~87% baseline
artifact** — the Poisson figure is a non-exclusive, uniform-rate, continuous-
time at-least-one probability, and the binder is an exclusive matcher over a
bursty, grid-quantized stream; any one of those three mismatches pushes the
observed rate below the "chance" line on pure noise, and the shifted null
proves the point empirically (observed rate is *above* the null mean).
**~13% material** — the small non-uniformity residual (−0.5 pt) is FF3's real
burst geography. Within the exclusivity loss, the genuine binder defect
(greedy strength-order stealing) is worth **31 pairs / 0.9 pt / 2.6%** — real,
minor, and not the headline. Star Tours' −7.4 pts splits differently:
**~72% material** (segment anti-co-location of its two streams, −5.3 pts) over
the same baseline artifact and a 1-pair greedy loss.

And the positive statement: against a null that respects the streams, **FF3
shows no detectable audio-visual synchrony excess at any window** (wide
percentile 74, tight 44, dt = 0 below null mean, mean |dt| slightly wide of
null), and Star Tours shows the same at tight windows with a borderline-low
wide rate. The two "below-chance" films share exactly what the above-chance
controls lack: **no segment-level co-location of audio and visual bursts**
(r ≈ 0.00 and −0.29 vs +0.46…+0.58 for the controls) — plus, in FF3's case, an
audio stream of exertion/ambience that does not time-lock to picture.

### Recommended changes

1. **Baseline model (the fix that matters):** make the circular time-shift
   null (grid-faithful, matcher-in-the-loop, ≥200 shifts) the primary chance
   context in `binding_summary` reporting, with the Poisson figure retained
   but relabelled for what it is — a *non-exclusive uniform-Poisson
   at-least-one rate*, an upper reference, not the chance rate of a one-to-one
   matcher. Every "below chance" reading in the corpus to date traces to this
   mislabel.
2. **Reporting practice:** quote tight-window excess against the
   grid-corrected chance `1 − exp(−λ_v·(2W + 50 ms))` (or the shift null), and
   report absolute tight *rates* alongside fraction-of-bound — the fraction
   inflates exactly when wide binding is depleted (the Star Tours effect).
3. **dt histogram:** align bins to the grid (slots at 0, 50, 100, 250 ms,
   labelled as exact slots). The [25, 50) bin is structurally empty and the
   "[50, 100)" bin currently mislabels exactly-50 ms pairs.
4. **Reproducibility (minor defect, worth fixing while here):** bound pairs
   are not reproducible from the stored artifacts — re-binding the rounded
   JSON times yields 1,226 vs the filed 1,208, because on-grid ties are
   decided by sub-µs float dust in the runtime visual times (`slot × 0.05`).
   Quantizing both streams to integer slot indices inside `bind()` would make
   ties deterministic and artifacts self-reproducing. Until then,
   `bound_pairs` in the run files are the authoritative record.
5. **Matcher:** maximum matching would recover ≤31 FF3 pairs (+0.9 pt) and
   1 Star Tours pair. Not worth changing the matcher's semantics for the rate;
   note it and leave the greedy matcher alone.
6. **Nothing** needs to change in the pathways or the materials: the phases
   provably did not move binding's inputs, and FF3/Star Tours are not broken
   runs — they are chance-level-binding materials read against a baseline that
   was never the chance rate of the procedure being run.
