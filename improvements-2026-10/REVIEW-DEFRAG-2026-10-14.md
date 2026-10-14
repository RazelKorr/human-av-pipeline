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

## Results (appended as each fix lands)

### Baseline vs final suite

(to be filled from the captured final log)

## Defrag actions taken

(to be filled)

## Left open

(to be filled)
