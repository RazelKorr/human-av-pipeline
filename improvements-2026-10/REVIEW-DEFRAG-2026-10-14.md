# Review + defrag — 2026-10-14

Commission: review and defrag the Sensorium (human-av-pipeline),
select six fixes, implement them, validate each, and write this
review of what was done and why. Full chain, one pass.

**Baseline (captured before any change):** full suite
`.venv/bin/python -m pytest tests/ -q` →
**303 passed** (log:
`improvements-2026-10/logs/defrag-2026-10-14/pytest_baseline.log`).
Tree at start: master `bb8d6dd`, clean.

All diagnostic numbers below come from scripts whose output is
captured in `improvements-2026-10/logs/defrag-2026-10-14/`.
Nothing in this review is copied from a run digest.

---

## Audit findings (recorded before any repair)

**A1. Binder reporting still implements the pre-investigation
framing.** `binder-investigation-2026-10-13.md` proved the Poisson
figure is a non-exclusive, uniform-rate, continuous-time
at-least-one rate — not the chance rate of the one-to-one matcher
— and recommended (recs 1–3): a circular time-shift null
(matcher-in-the-loop) as the primary chance context, grid-corrected
tight-window chance `1 − exp(−λ_v·(2W + 50 ms))`, and grid-aligned
dt histogram bins. None of it was implemented: `binding_summary`
still reports only `chance_rate_poisson` / `excess_over_chance`,
tight windows carry only the continuous-Poisson chance, and
`DT_BIN_EDGES_MS = (0, 25, 50, 100, 250)` still contains the
structurally empty [25, 50) bin (both streams live on the 50 ms
grid, so no |dt| can land in it) and mislabels the exactly-50 ms
pairs as "[50, 100)".

**A2. PX.1 (from EXPERIMENTS-2026-10-13): the streaming ear
false-alarms on a −36 dBFS brown-noise bed** — 54 onsets vs 12
planted bursts (all 12 matched). Diagnosed this pass
(`px1_diagnose.log`, `px1_features.log`, `px1_candidates.log`,
`px1_sustain.log`, `px1_relflux.log`, `px1_conc.log`):

- The filed 54 picks reproduce *exactly* from
  `input/dark_busy_60s.mp4` through the ear path
  (`decode_audio_track → moment_features → pick_onsets`).
- The 12 true burst picks are clean in hindsight (strength
  29.2–34.4; flux/threshold ratio ≥1.77). The 42 bed picks sit at
  15.1–19.6, ratios 1.01–1.30.
- Cause: `pick_onsets`' local bar is `median + k·MAD` with the
  MAD **unscaled**. For a Gaussian bulk, MAD ≈ 0.6745σ, so k=3
  puts the bar at ≈ median + 2.0σ — roughly the 97.7th percentile
  per moment, and the bed's flux is essentially independent
  across moments (lag-≥2 autocorrelation ≈ 0). Over 1,200
  moments a fixed ~2% tail exceedance, thinned by the local-max
  and min-gap rules, yields the observed ~42 false picks. The
  false-alarm *count* is set by the tail probability and the
  duration, not by events. The sealed PX.1 model ("the adaptive
  threshold rides a smooth bed") implicitly assumed the bar sat
  at the 3σ the parameter name suggests.
- Rejected alternatives, measured not argued: a strength or
  flux/threshold-ratio margin (distributions overlap once bed
  picks near bursts are excluded — and a pure σ-rescaling to
  k=3σ̂ or 4σ̂ cuts FF3's stored-feature pick count by 28–45%,
  an unmeasured sensitivity change to real material); a sustain
  gate (separates this fixture perfectly — bursts sustain
  ≥5.34× local median RMS, bed picks ≤1.23× — but FF3's real
  onsets have median sustain 1.19 and the gate would cut
  ~55–62% of them); a spectral-concentration gate (bursts
  top-5-bin share ≥0.56 vs bed ≤0.475, but YTPMV's real music
  onsets have median concentration 0.133 — the gate would cut
  87.5% of them); relative flux (overlaps: bursts ≥0.453, bed
  ≤0.491). Every per-pick statistic that separates this bed
  from these bursts fails on some real material. The bar's
  *units* are the defect that generalizes.

**A3. bind() pair output is representation-dependent (binder
memo rec 4).** Measured this pass on current code
(`bind_repro_before.log`): rebinding FF3's stored streams
(3,362 onsets / 2,735 transients from
`output/ff3_hd_av_2026-10-13/av_binding.json`) gives **1,228**
pairs from the stored ms-rounded times and **1,229** from the
slot×0.05 float form of the same events, with **71 pairs**
differing between the two representations; neither reproduces
the filed 1,208. Sub-µs float dust decides on-grid ties.

**A4. run_stream's audio features are not on the master grid
for non-20 Hz-divisor frame rates.** `ChunkedFeeder` cuts chunks
at *frame timestamps* (`feeder.py`: `t_s − chunk_t0 >= chunk_s`),
and `run_stream.py` slices the audio track at
`round(chunk.t0_s × SR)` and reshapes each chunk into 50 ms
moments starting there. When the frame interval doesn't divide
50 ms evenly into the chunk boundary (FF3 run: fps 29.97,
processed 2700.9983 s), each chunk's audio moments sit on a grid
phase-shifted by (t0 mod 50 ms) from the master grid the
features claim to share with vision, and chunk ends truncate or
zero-pad against the next chunk's restart. Evidence
(`ff3_decode_check.log`, `ff3_wav_check.log`,
`ff3_decode_lag.log`): the filed
`output/ff3_hd_av_2026-10-13/audio_features.npy` differs from a
whole-track `moment_features` computation in 53,931/54,019
moments (flux corr 0.37–0.64 depending on source), while two
independent decodes of the same audio (the mp4 track and the
pre-extracted `input/ff3_audio16k.wav`) agree with *each other*
exactly. Distributions and pick counts stay close (3,362 filed
vs 3,348 whole-track) — which is why nothing downstream noticed:
the binding window is ±250 ms and the phase error is ≤25 ms.
(30 fps material like the generated fixtures lands chunk
boundaries on exact 2.0 s marks and is unaffected — the filed
dark-busy features are true-grid, verified by A2's exact
reproduction.)

**A5. `scripts/binding_baseline.py` sidecars are stamped
wrong.** Every sidecar it writes carries `"generated":
"2026-10-12"` (hardcoded) and an `estimator` string describing
only the Poisson estimator. After Fix 1, the pair-only backfill
path also cannot produce the shift-null context (it needs the
event streams, not just pairs) — a limitation the sidecar
nowhere states.

**A6. The public record still teaches the mislabel.**
`streaming/STREAMING-REPORT.md` presents FF3's 35.9% bound vs
39.7% Poisson as "below chance" (e.g. line 1230, line 1242)
with no pointer to the 2026-10-13 investigation that decomposed
the deficit (~87% baseline artifact) or to the corrected
reporting. The report is a dated log — history stays; it needs
a dated addendum, per its own convention.

**Defrag (organization, no semantics):** the binder
investigation's three captured outputs
(`binder-investigation-2026-10-13-output{,2,3}.txt`) sit at the
repo root. They belong with the 2026-10 investigation logs.
No TODO/FIXME/XXX markers exist anywhere in hva/hvp/hvm/
streaming/scripts. `output/` (4.2 GB) is untracked and
gitignored, as are `input/`, `handoff/`, root `logs/`.

**Noted, deliberately not fixed (see Left open):**
`visual_transients` (bind_av.py) uses a *global* median+MAD bar
— the design `pick_onsets`' docstring says failed for audio in
2026-10-03 — but no measured failure exists for the visual
picker, and changing it would move every run's transient counts;
`poisson_chance_rate`'s wide-window figure also wants the grid
treatment in principle, but the memo shows the grid correction
runs the "wrong" way at wide windows and the shift null
supersedes both closed forms there.

---

## The six fixes — sealed predictions (written before implementing)

**Fix 1 — Binder reporting (memo recs 1–3).** Add to
`streaming/bind_av.py`: (i) `shift_null_context()` — circular
time-shift null, grid-faithful (shifts are multiples of the
50 ms moment grid), matcher-in-the-loop (the same greedy
matcher), 200 seeded shifts — reported from `bind()` as a
`shift_null` field with null mean/sd/p5/p95 and observed
percentile for the wide bound rate, the |dt|≤50 ms tight rate,
the dt=0 count, and mean |dt|; (ii) grid-corrected chance
`chance_rate_grid_corrected` (+ excess against it) in each
tight window; (iii) the Poisson fields relabelled via an
explicit `chance_rate_poisson_label` stating it is a
non-exclusive uniform-Poisson at-least-one rate, an upper
reference, not the matcher's chance rate; (iv) dt histogram
edges aligned to the grid — (0, 50, 100, 250) — each bin
annotated with the exact grid slots it contains on a 50 ms
grid. Existing field names and the matcher are unchanged;
`binding_summary` (pair-only backfill) gains the grid-corrected
tight fields and the label, and states the shift null is not
computable from pairs alone.
*Sealed:* on FF3's stored streams, `bind()` reports a shift-null
wide mean in **0.34–0.37** with the observed rate's percentile
inside [5, 95]; FF3 tight grid-corrected chance =
**0.1409** (formula-determined from λ_v = 2735/2701.0);
the new histogram partitions n_bound exactly on synthetic and
stored inputs; the two existing tests that pin the old bin
edges / bin fractions are updated to the grid-aligned contract
and nothing else in the suite moves.

**Fix 2 — PX.1: pick_onsets' MAD gets its consistency
constant.** Multiply the local MAD term by **1.4826** (the
standard MAD→σ consistency constant for a Gaussian bulk), so
k means robust sigmas as the sealed models assumed. One
constant, named and commented; nothing else in the picker
changes.
*Sealed* (from the candidate replica in `px1_candidates.log`,
which reproduced the current code's outputs exactly at
scale 1.0 — the validation below reruns through the real code):
dark-busy ear path → **21 total picks, 12/12 planted matched**
(±0.5 s), i.e. bed false alarms 42 → 9; bright-silent → **0**;
FF3 stored-features recount → **2,409** (from 3,362 — the filed
count was produced by a bar at 2.0σ̂ while documented as k=3;
the delta is reported, not hidden); a new regression test
(synthetic brown-noise bed + planted bursts) pins the behavior;
full suite green.

**Fix 3 — bind() matches on ms-quantized times (memo rec 4).**
Quantize both streams' times to integer milliseconds at
`bind()` entry (the output already rounds to ms; the streams
are produced on a 50 ms grid, so this removes only sub-µs
representation dust) and match on the quantized values.
*Sealed:* rebinding FF3's stored streams in both
representations (ms-rounded and slot-dust float form) yields
**identical pair sets, 1,228 pairs** (the pre-fix rounded-form
result), symmetric difference **0** (pre-fix: 71).

**Fix 4 — run_stream slices audio on the master grid.**
Replace the per-chunk `round(t0×SR)` slicing with a moment
cursor: each chunk emits the master-grid moments whose start
falls in [t0, t1), sliced from the track at exact moment-sample
offsets; `prev_mag` threading unchanged.
*Sealed:* a flagship run on a 29.97 fps clip
(`input/ff3_seg_1200_1500.mp4`) writes `audio_features.npy`
**exactly equal** (max abs diff 0.0) to a whole-track
`moment_features` computation over the same span; the
dark-busy flagship run (30 fps, previously aligned) keeps its
ear-path onset count (21 after Fix 2 — grid change is a no-op
there).

**Fix 5 — binding_baseline.py sidecar truthfulness.** Stamp
`generated` with the actual run date; extend the `estimator`
string to cover the grid-corrected tight chance; add an
explicit field noting the shift-null context requires the
event streams and is absent from pair-only backfills.
*Sealed:* running it on a scratch copy of a stored run
produces a sidecar dated 2026-10-14 carrying the note, with
all summary numbers unchanged from the current script's output
on the same input.

**Fix 6 — STREAMING-REPORT addendum.** Append a dated
2026-10-14 addendum (append-only; no historical text edited)
stating the corrected chance framing: the Poisson figure is a
non-exclusive upper reference, FF3's deficit decomposes to
~87% baseline artifact per the binder investigation, and
current reporting (Fix 1) carries the shift-null context and
grid-corrected tight chance.
*Sealed:* `git diff` on the file shows additions only.

---

## Results

### Fix 1 — Binder reporting (commit 2669698, with Fix 3)

`validate_fix1_fix3.log`, FF3 stored streams (3,362 onsets /
2,735 transients, duration 2701.0 s), `bind()` with defaults:
shift-null wide mean **0.3609** (sealed 0.34–0.37 ✓), observed
bound rate 0.3659 at percentile **77.0** (sealed within [5, 95]
✓). Tight window: grid-corrected chance **0.1409** (sealed
exactly ✓), observed tight rate 0.1452 vs null mean 0.1456,
percentile **45.5** — the null's center, replicating the
investigation's finding (0.1457 vs 0.1461) from a different
implementation. dt=0 count: observed 158, null mean 163.97,
percentile 33.5. Mean |dt|: percentile 95.5 — the one statistic
sitting high in its null, consistent with the investigation's
~13% material component (real pairs sit slightly farther out
than shifted chance puts them). New histogram partitions
n_bound exactly: [0,50): 158, [50,100): 330, [100,250]: 742,
sum 1,230. Targeted tests: `tests/test_binding_baseline.py`
**9 passed** (`pytest_fix1_fix3_targeted.log`), including the
two contract updates (histogram edges; clustered-bin assertion)
and four new tests.

### Fix 2 — PX.1 (commit 76ef553)

`validate_fix2.log`, real code path (decode → moment_features →
pick_onsets, defaults): dark-busy **21 picks, planted 12/12, bed
false alarms 9** (was 42; sealed 21 / 12/12 ✓). Bright-silent
**0** ✓. FF3 stored-features recount **2,409** (sealed exactly
✓; filed 3,362 under the 2.0σ bar — reported here, not hidden).
Consequence check (`ytpmv_rebind_fix2.log`): YTPMV recount drops
to 212 onsets (filed 362 bound-stream onsets), but rebound under
the fixed ear its genuine signal survives and cleans up — tight
excess over grid-corrected chance **+10.8 pts**, shift-null
tight percentile **97.0**, wide percentile 96.5. Full flagship
confirmation: run_stream on dark-busy writes 21 audio events,
12/12 planted (`fix4_dark_runstream.log`). Regression test
`test_pick_onsets_brown_noise_bed_px1` added; `tests/test_audio.py`
**18 passed**. One honest caveat: every filed onset count in the
corpus (FF3 3,362, YTPMV, Star Tours, Dr Tran) was produced
under the unscaled-MAD bar; counts from runs after this fix are
not directly comparable to filed ones. That is the cost of the
bar meaning what k=3 always claimed, and it is why the change
is one named constant with the experiment trail behind it.

### Fix 3 — bind() ms quantization (commit 2669698, with Fix 1)

`validate_fix1_fix3.log`: rebinding FF3's stored streams in the
ms-rounded and slot-dust representations now yields **identical
pair sets, symmetric difference 0** (sealed ✓; pre-fix 71,
`bind_repro_before.log`). Sealed pair *count* missed: predicted
1,228 (the pre-fix float matcher's rounded-form result), actual
**1,230**. Mechanism: the integer matcher compares the window
edge exactly (250 ≤ 250 binds; the float matcher sometimes
excluded exact-edge pairs on representation error) and resolves
true ties deterministically. The defect — representation
dependence — is eliminated; the count difference is the
correction, recorded rather than rounded away. Neither rebind
reproduces the filed 1,208, and cannot: the filed run's float
forms are not recoverable from ms-rounded artifacts. Future runs
are self-consistent by construction.

### Fix 4 — run_stream master-grid audio (commit 1dfc7db)

`validate_fix4.log`: flagship run on the 29.97 fps FF3 segment
(60 s) writes `audio_features.npy` **exactly equal** to the
whole-track computation — max abs diff **0.0** across all 1,200
moments (sealed ✓) — and its 42 audio events match whole-track
pick_onsets exactly. Dark-busy flagship (30 fps): 21 onsets,
12/12 planted — no-op on aligned material, as sealed
(`fix4_dark_runstream.log`, `fix4_runstream.log`). Downstream
note: FF3-class runs (non-20 Hz-divisor frame rates) re-run
after this fix will produce audio features that differ from the
filed ones — the filed ones were phase-rotated; onset *counts*
barely move (the binding window dwarfs the ≤25 ms phase error),
but feature-level comparisons across the fix boundary are
invalid, same caveat class as Fix 2.

### Fix 5 — Sidecar truthfulness (commit 885a13d)

`validate_fix5.log`, scratch copy of the FF3 run: sidecar
`generated` = **2026-10-14** ✓, `shift_null_note` present,
estimator string covers the grid-corrected chance, and every
summary field is identical to a fresh `binding_summary` call on
the same artifacts (sealed ✓). The sidecar's tight-50
grid-corrected chance reads 0.1409, matching Fix 1's live
computation.

### Fix 6 — STREAMING-REPORT addendum (commit b728e47)

`git diff` on `streaming/STREAMING-REPORT.md`: **35 insertions,
0 deletions** (sealed: additions only ✓). The addendum states
the corrected framing, the investigation's decomposition, and
the implemented reporting, and leaves every historical number
standing as written.

### Baseline vs final suite

Baseline (pre-change, `pytest_baseline.log`): **303 passed**.
Final (`pytest_final.log`): **308 passed, exit 0** — the +5 are
this pass's new tests (4 in `tests/test_binding_baseline.py`,
1 in `tests/test_audio.py`); no pre-existing test failed, and
the only pre-existing tests modified are the two
`test_binding_baseline.py` assertions pinning the superseded
histogram contract, updated under Fix 1.

## Defrag actions taken

- Binder investigation outputs moved from the repo root to
  `improvements-2026-10/logs/binder-investigation/` (git mv;
  history preserved); the memo's three references to their
  location updated in place.
- This commission's diagnostics and validation scripts/logs live
  in `improvements-2026-10/logs/defrag-2026-10-14/`. Note: the
  repo's `.gitignore` `logs/` rule leaves that directory
  untracked (the same convention the 2026-10-13 experiment logs
  already followed) — the files are on disk and referenced by
  path from this review; only the review file itself and the
  code commits are in git.
- Nothing was deleted. `output/` (4.2 GB, untracked, gitignored)
  was left exactly as found.

## Left open

1. **`visual_transients` global threshold.** bind_av.py's visual
   picker uses a global median + k·MAD bar — the design the
   audio picker's docstring records as failed for audio in
   2026-10-03. No measured failure exists for the visual picker,
   and changing it moves every run's transient counts, so it was
   not touched on suspicion alone. Wants: a measured failure (or
   a sealed prediction on a fixture with a loud stationary
   visual bed) before any change. Its MAD is also unscaled
   (k=4.0 ≈ 2.7σ̂) — if a future change scales it, that is a
   sensitivity change to be validated like Fix 2, not a
   consistency freebie.
2. **Filed-corpus recounts under the fixed ear.** Fix 2 changes
   onset counts on all real material (FF3 3,362 → 2,409 on
   stored features; YTPMV rebound and fine). Star Tours and
   Dr Tran were not recounted — their binding verdicts were
   "at chance" under both the old bar and (for the tight window)
   the investigation's corrections, but a clean recount + rebind
   under the fixed ear and new reporting is a defined, bounded
   follow-up if the corpus numbers are ever republished.
3. **Adaptive saccade clock adjudication** (carried from
   EXPERIMENTS-2026-10-13): still needs fixtures with
   order-of-magnitude motion contrast; the Pink Panther clip
   cannot adjudicate. Unchanged by this pass.
4. **Wide-window grid effects.** The shift null supersedes the
   closed forms at the wide window, and the investigation showed
   the naive grid correction runs the wrong way there; no
   grid-corrected wide Poisson figure was added, deliberately.
   If a closed-form wide reference is ever wanted, derive it
   from the slot structure explicitly rather than by analogy
   with the tight window.
5. **Phase-5-era doc numbers.** STREAMING-REPORT and
   improvements-2026-10 summaries quote historical suite counts
   (159/240/282/301/303) — all true when written, all dated;
   left as history by the same convention as Fix 6.
6. **PX.1 residual bed alarms.** 9 false alarms remain on the
   dark-busy fixture at the corrected 3σ̂ bar — the expected tail
   of a correctly-set bar over 1,200 moments, not a defect
   signature. If a future fixture demands fewer, the honest
   lever is k itself (with the real-material recount that
   implies), not another per-pick gate.

---

## Addendum 2026-10-14 (commission 2) — left-open items 1–3, resolved

Appended, append-only; the text above stands as written. Full
detail: `improvements-2026-10/REVIEW-WILD-2026-10-14.md`;
captured logs: `improvements-2026-10/logs/wild-2026-10-14/`.

**Item 1 — visual_transients: INDICTED and fixed.** The sealed
fixture the left-open item asked for (stationary noisy energy
bed, 10 planted transients) returned 41 picks / 31 false alarms
under the unscaled bar — the PX.1 shape, one channel over.
`visual_transients` now scales its MAD by
`VISUAL_MAD_TO_SIGMA` (1.4826, the same constant as the ear):
fixture 11 picks, 10/10 planted, 1 false; dark-busy corpus
transients 56 → 53. Counts across this boundary are not
comparable, same caveat class as Fix 2.

**Item 2 — Star Tours / Dr Tran recounts under the fixed ear**
(method identical to the FF3/YTPMV recounts above; the method
reproduces FF3's 2,409 from stored features exactly). Filed →
recount: Star Tours onsets **238 → 103**, bound 91 → 37,
bound rate 0.3592, tight-50 rate 0.1942 vs grid-corrected
chance 0.1671, shift-null tight percentile **81.0** (wide
10.5). Dr Tran ep4 onsets **588 → 377**, bound 298 → 199,
bound rate 0.5279, tight-50 rate 0.1963 vs grid chance
0.1868, shift-null tight percentile **87.0** (wide 91.5).
Both verdicts unchanged: at chance under the corrected
reporting, as they were under both earlier bars. The filed
counts above in this review (3,362-era) and these recounts
are not comparable across the PX.1 boundary — that caveat
applies wherever any of these numbers are cited.

**Item 3 — adaptive clock: adjudicated.** See EXPERIMENTS-
2026-10-13 addendum (commission 2, R3) and the wild review.
