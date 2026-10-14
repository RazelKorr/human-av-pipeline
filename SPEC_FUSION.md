# SPEC_FUSION.md -- how the ears and the eyes become one perceiver

> **Specs rewritten 2026-10-13.** This spec now describes the streaming
> generation the 2026-10 improvement phases actually built, not only the
> batch generation it originally described. Where the two generations
> differ, both are named and labelled. Every architectural claim below
> is traceable to code in `hvm/`, `hva/`, `streaming/`, `scripts/`, or a
> phase artifact in `improvements-2026-10/`.

Three levels, in order of increasing commitment. Level 1 shares a
clock. Level 2 lets the systems tug on each other. Level 3 gives them
one shared landscape to stand on.

**A fourth meeting point now exists, and it is the one the flagship
reviews actually use:** the streaming binder (`streaming/bind_av.py`,
§ "Streaming binding" below) — coincidence binding on the 50 ms master
grid, with a Poisson chance baseline shipped in its own output. The
flagship streaming runner does **not** use the Level 3 joint map, and
its audio is passive (it observes; it never steers vision). Levels
1–3 remain real, battery-validated paths; they are not the flagship
path.

None of this is validated multisensory integration. It is a
working hypothesis about architecture: where in the pipeline the
senses meet, and what each meeting point can and cannot explain.

## Level 1 -- shared timeline (`scripts/fuse_av.py`)

The two pipelines run independently and their outputs are indexed by
media time. One joint 10 Hz feed: gaze position, audio focus, both
event streams.

The one honest finding at this level: visual content trails audio by
about two 100 ms moments when actual foveated frames are combined.
Ears are faster than eyes. The feed pairs by media time and documents
the lag instead of hiding it.

## Level 2 -- reshaped coupling (`scripts/couple_av.py`)

Bidirectional, but each system keeps its own machinery:

- **Audio -> vision:** spatial audio onsets add an alerting gain that
  tugs saccade landings toward the panned side. Landing error on 26
  spatial onsets: 28.1 px coupled vs 32.9 px uncoupled (224-space).
- **Vision -> audio:** visual transients lower the auditory capture
  bar, up to 36% off -- but only transients above the sequence's 75th
  percentile get a vote. The film is always wiggling a little; only
  the genuinely busy moments make you listen harder.

~~The 75th-percentile gate is computed over the full sequence, which is
noncausal. A rolling/causal estimator is open work.~~ **Done
(Phase 3, 2026-10-12) — the causal gate is now the default.** The
reshape lives in `hva/attention.py` (`shaped_vis_boost(..., gate_mode=
"causal")`, consumed by `scripts/couple_av.py --gate-mode causal`,
also the default there):

- **Causal (default):** the 75th percentile is a *trailing* estimator —
  moments strictly before k, at most 300 of them (`GATE_WINDOW`; 30 s
  at the 100 ms grid), current moment excluded. Moment k gets no boost
  until 30 past moments exist (`GATE_WARMUP`) — insufficient evidence
  cannot lower the capture bar. A transient late in the sequence
  cannot change any earlier gate value (unit-verified: causal traces
  over a shared prefix are bit-identical when a huge transient is
  appended late).
- **Legacy (kept for A/B only):** the pre-2026-10-12 full-sequence
  percentile, which leaks future transients into early decisions; a
  characterization test pins that leak for as long as legacy exists.
- Validation (`scripts/l2_gate_compare.py`, Dr Tran ep4, 4,361
  moments): the causal gate changed the event label of 14.31% of
  moments (sealed band 1–25%) and onset captures 212 → 196 (−7.5%,
  sealed ±25%). One sealed prediction missed: only 21.6% of
  event-differing moments sit where the gate value moved by ≥0.05
  (sealed ≥60%) — state cascade (one flipped decision changes
  cf/habituation/IOR downstream) plus small gate deltas flipping
  borderline captures. The gate's own causality is exact; the miss
  was about trace geography, and it is on record in
  `improvements-2026-10/PREDICTIONS.md`. End-to-end on Star Tours
  inputs: causal 41 captures / 66 switches vs legacy 38 / 68 — same
  family, no breakage.

## Level 3 -- the joint priority map (`hvm/`)

Level 2's cross-biases are replaced by one shared 56x56 spatial
priority map. Both systems write to it; both read from it.

**Writes** (`hvm/priority.py`, `JointPriorityMap.step`):

- Vision writes normalized retinotopic salience (static + transient),
  weight 1.0.
- Audio writes per-frequency-bin transient salience, splatted at each
  bin's ILD-derived azimuth. Audio has no elevation cue, so its
  contributions are centered vertically. Mono/centered audio becomes a
  central alerting contribution. Weight 0.7.
- Both contributions decay with a 300 ms time constant. The map is a
  leaky integrator, not a frame buffer.

**Reads:**

- Saccades are selected from the joint map (closed loop in
  `hvm/driver.py`). Read-time oculomotor priors -- inhibition of
  return (1500 ms), the human amplitude prior (~5-15 deg) -- are
  applied at read time, never written to the map. Deliberately no POI
  pull: the map's own 300 ms persistence is the memory here.
- Auditory attention reads spatial gains back: a frequency bin whose
  azimuth projects onto a salient map column gets its transient
  evidence amplified (`gains_for_pans`, gain 1 + 1.5 * column-max).
  The gained salience and gained transient peaks drive the attention
  module's actual decisions -- capture thresholds and switch targets,
  not just a rescaled side channel. (An earlier version scaled only
  the transient profile while the detector thresholded on ungained
  statistics; captures came out identical 26/26. The map has to touch
  the decision variables or it is decoration.)

**Provisional constants** (all guesses, all need a scientific audit):
`W_VIS=1.0`, `W_AUD=0.7`, `TAU_MS=300.0`, `AUD_SPREAD_PX=6.0`,
`READ_GAIN_K=1.5`, `SPEECH_BOOST=1.0`. Audio write strength is mean-normalized across
frequency bins so a broadband transient can't drown vision by bin
count alone.

**Speech gating** (`--transcript` on the runner, `hva/transcribe.py`
`speech_presence`, `JointPriorityMap.step(..., speech=...)`): when
speech is present, the auditory write is scaled by
`(1 + SPEECH_BOOST * speech)`, so audition's vote doubles during
speech (0.7 -> 1.4) -- enough to outvote vision in an equal-strength
conflict. The voice captures the map; that is the cocktail-party
direction. Attenuation across inputs falls out of the shared map's
normalization: when the ears get louder, everything else gets
relatively quieter. There is deliberately no global visual-suppression
knob -- real-media data (Dr Tran: title-card dwells during dense
narration) shows vision keeps working while speech runs, so muting
vision during speech would be wrong. Speech presence is binary from
word spans plus a 300 ms exponential hangover (same clock as the
map's persistence; a guess). Stream-level, not bin-level: every bin
is scaled equally, so a loud non-speech transient during speech gets
boosted too. The real fix is stream separation -- identifying which
bins carry the voice -- recorded as the v1 gap.

**Validation** (`hvm/battery.py`, synthetic, deterministic, 6/6):

- M1 congruence: matched flash-left/click-left makes one combined
  peak; mismatched makes two competing peaks (ratio 1.59). Orienting
  begins at 300 ms either way.
- M2 audio alone: left click peaks at map x=10.0.
- M3 vision alone: right flash peaks at map x=42.0.
- M4 conflict: equal-strength flash-left/click-right -- vision wins
  under current weighting (peak x=14.0).
- M5 audio read path: left-panned bin gain 1.38, right-panned 1.00.
- M6 speech gating: equal-strength flash-right/click-left -- without
  speech the peak sits at x=42.0 (vision wins, the ventriloquism
  direction); with speech present it flips to x=10.0 (speech-gated
  audio wins).

**Real media** (`scripts/run_level3.py`, Star Tours 62 s):

- Joint vs vision-only closed loops: audio moved the priority peak
  >8 px in 6.0% of moments; saccade landings displaced mean 15.6 px,
  32 of 207 landings moved >16 px.
- Map-gained auditory attention: captures 26 -> 23, switches 61 -> 87.
  The map routes more than it amplifies.
- Spatial specificity check: at capture moments, rank correlation
  between the transient profile and the spatial gain profile is 0.26
  vs 0.12 for shuffled gains (n=23, suggestive, not conclusive).

## Streaming binding (`streaming/bind_av.py`) — the flagship meeting point

This is how the flagship streaming runs (`streaming/run_stream.py`)
join the senses. It is deliberately weaker than Level 3: no shared
map, no steering in either direction.

- **Inputs.** Audio onsets from the streaming ear
  (`hva/online.py: pick_onsets`) and visual transients from the vision
  driver's per-frame energies (local maxima of the per-moment max
  |energy diff|, thresholded at median + 4·MAD) — both on the 50 ms
  master grid.
- **Matching.** A ±250 ms coincidence window (`COINCIDENCE_MS`).
  Each onset binds to at most one transient and vice versa — greedy
  by descending audio strength, nearest in window. Output pairs carry
  `dt_s = audio_t − visual_t`. Unbound onsets/transients are
  characterized, not discarded.
- **The chance baseline is part of the output (Phase 2, 2026-10-12).**
  Since that phase, `bind()` also writes, additively (the matcher and
  the original fields are unchanged): the **Poisson chance rate**
  `1 − exp(−λ_v · 2W)` with `λ_v = n_visual / duration_s`; the excess
  of the bound rate over chance; bound counts/rates at the tight
  windows (±50 ms, ±100 ms) with their own chance rates; and a dt
  summary (histogram over |dt|, signed asymmetry). Pre-Phase-2 runs
  were backfilled with sidecars by `scripts/binding_baseline.py`
  (which never modifies the originals).

### How binding claims must be stated

**A raw bound rate is a coincidence rate, not a causal-binding
measure, until it is read against its Poisson chance rate.** The
canonical runs, as backfilled and scored in Phase 2:

| Run | Bound (±250 ms) | Poisson chance | Excess |
|---|---|---|---|
| FF3 1080p (`ff3_hd_av`) | 35.9% (1,208 / 3,362) | 39.7% | −3.8 pts |
| Star Tours AV (`st_stream_av_full`) | 38.2% | 45.6% | −7.4 pts |
| Dr Tran ep4 (`drtran_ep4_av`) | 50.7% | 49.8% | +0.9 pts |
| Pink Panther (`pink-panther-cato-attacks`) | 52.4% | 47.8% | +4.6 pts |

Only Pink Panther's *rate* exceeds chance. That does not make the
others signal-free: the causal signal, where it exists, lives in the
**dt distribution** — the fraction of bound pairs at tight dt
(|dt| ≤ 50 ms: Star Tours 46.2% of bound, the strongest of the four;
FF3 40.6%; Dr Tran 38.3%; Pink Panther 31.5%) and in signed asymmetry
(Star Tours' tight pairs skew audio-after-visual, as real-world sound
following a visible event should).

**dt is quantized to the 50 ms grid.** Bound pairs sit at
0 / ±50 / ±100 / … ms: the 25–50 ms histogram bin is empty in all
four canonical runs, and the "zero-dt" pairs are exactly the
histogram's 0–25 ms bin (FF3: 159). Any reading of dt finer than the
50 ms grid is reading the quantization, not the footage. Strength
weighting and causal (audio-after-visual) scoring beyond the shipped
summary remain future work — the reviews reason that way; the code
now produces the baseline, tight windows, and dt summary they reason
from, and no more than that.

## What Level 3 is not

- ILD-derived pan is not HRTF localization. No elevation, no
  front/back disambiguation, no room model.
- The map is not validated multisensory binding. Congruence effects
  in the battery are architectural consequences, not neural claims.
- Audio attention is still frequency-band attention, not auditory
  object or stream perception. (The Phase 5 stem-stream experiment —
  SPEC_AUDIO.md, "Stem streams" — is a first, leaky object layer at
  the label/transcript level; it is experimental, not wired into any
  fusion path, and its onset/feature level does not form clean
  objects.)
- Streaming binding is not Level 3 and is not causal proof: it is a
  coincidence counter whose output must be read against its own
  Poisson chance baseline and dt distribution (see above).
- All coupling constants are provisional. The 300 ms persistence, the
  0.7 auditory weight, the 1.5 read gain -- none of these are fit to
  human data.

## Entry points

- `python3 streaming/run_stream.py V --outdir O [--motion-weight 1.0
  --pursuit]` -- the flagship streaming run: chunked vision + passive
  streaming-ear audio + `bind_av` binding, writing `run_report.json`
  (with its `streaming` block), `audio_events.json`, and
  `av_binding.json` (with the chance baseline) to `O`.
- `python3 scripts/binding_baseline.py RUN_DIR [...]` -- backfill the
  Poisson chance baseline as an `av_binding_baseline.json` sidecar
  for a pre-Phase-2 run; originals are never modified.
- `python3 scripts/run_level3.py --video V --mono M --stereo S
  --seconds N --outdir O` -- full Level 3 closed loop on real media;
  prints the report, saves maps/peaks/traces to `O`.
- `python3 scripts/watch.py` -- the L1 watch-and-review prosthesis
  (predates the shared map; event proximity, not binding).
- `python3 scripts/perceive.py` -- still-image perceptual reports.
