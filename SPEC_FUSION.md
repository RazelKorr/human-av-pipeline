# SPEC_FUSION.md -- how the ears and the eyes become one perceiver

Three levels, in order of increasing commitment. Level 1 shares a
clock. Level 2 lets the systems tug on each other. Level 3 gives them
one shared landscape to stand on.

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

The 75th-percentile gate is computed over the full sequence, which is
noncausal. A rolling/causal estimator is open work.

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

## What Level 3 is not

- ILD-derived pan is not HRTF localization. No elevation, no
  front/back disambiguation, no room model.
- The map is not validated multisensory binding. Congruence effects
  in the battery are architectural consequences, not neural claims.
- Audio attention is still frequency-band attention, not auditory
  object or stream perception.
- All coupling constants are provisional. The 300 ms persistence, the
  0.7 auditory weight, the 1.5 read gain -- none of these are fit to
  human data.

## Entry points

- `python3 scripts/run_level3.py --video V --mono M --stereo S
  --seconds N --outdir O` -- full Level 3 closed loop on real media;
  prints the report, saves maps/peaks/traces to `O`.
- `python3 scripts/watch.py` -- the L1 watch-and-review prosthesis
  (predates the shared map; event proximity, not binding).
- `python3 scripts/perceive.py` -- still-image perceptual reports.
