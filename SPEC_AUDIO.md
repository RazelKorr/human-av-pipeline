# Human Auditory Attention Pipeline (HAP) — SPEC

> **Specs rewritten 2026-10-13.** This spec now describes the streaming
> generation the 2026-10 improvement phases actually built, not only the
> batch generation it originally described. Where the two generations
> differ, both are named and labelled. Every architectural claim below is
> traceable to code in `hva/`, `scripts/`, `streaming/`, or a phase
> artifact in `improvements-2026-10/`.

Experimental demonstration, not a validated model of hearing. Built 2026-09-30.

## What it is

A pipeline that listens to audio the way the vision pipeline watches video:
sound goes in, and out comes a trace of where "attention" was, moment by
moment, plus a list of events (captures, switches, quiet dwelling).

**Two generations, two clocks — read this first.** The batch ear
described in "The stages" below (`cochlea → salience → moments →
attention`) runs on **100 ms perceptual moments**, the clock it shares
with the batch vision pipeline, and it is what the Level 1/2/3 fusion
scripts and `hva/stream.py` use. The **flagship streaming runs do not
use that stack**: they use the streaming ear, `hva/online.py` (§ "The
streaming ear"), on the **50 ms / 20 Hz master grid** it shares with
the streaming vision path. `hva/cochlea.py`, `hva/salience.py`, and
`hva/attention.py` are exercised by the batch/fusion paths and their
battery, but they did not produce the FF3 or Star Tours flagship
reviews. A reader who misses this distinction will form a materially
wrong picture of what produced those reviews (audit 2026-10-12,
Lane 1).

## The stages

**1. Cochlea (`hva/cochlea.py`).** The raw waveform becomes a spectrogram:
20ms analysis windows, 5ms hops, 64 frequency bins spaced logarithmically
from 50Hz to 8kHz (log spacing because pitch perception is logarithmic --
an octave is a doubling, and the ear treats octaves as equal steps). Output
is in absolute dBFS. Deliberately NOT normalized per file: normalizing
every file to its own loudest peak makes digital silence look like huge
meaningful events. The auditory nerve has an absolute threshold; so does
this.

**2. Salience (`hva/salience.py`).** Three channels, computed per
time-frequency bin:
- *Intensity* -- how loud is this bin right now.
- *Frequency contrast* -- how much does this bin stand out from its
  neighbors (a whistle among rumbles).
- *Temporal contrast* -- how much did this bin just change (onsets).

Weights: intensity 0.4, frequency contrast 0.3, temporal contrast 0.6
(change is the most attention-grabbing -- the ears are the early-warning
system). A provisional hearing floor (-60 dBFS) zeroes temporal contrast
where the source is essentially silent, because log-domain differences
explode on near-silence: tiny numerical wiggles become tens of fake dB.

**3. Moments (`hva/moments.py`).** Twenty 5ms frames are integrated into
each 100ms moment: mean spectrum, mean salience, a loudness proxy (averaged
in linear power first -- averaging in log domain made tonal input look
silent, which is exactly backwards), and Tmax, the strongest short-term
change (masked below the hearing floor, same reason as above). The 100ms
grain is a working choice, partly so audio shares a clock with vision; the
exact auditory integration grain is debated in the literature.

**4. Attention (`hva/attention.py`).** A frequency-band controller. Focus
is a center-frequency bin, and it moves two ways:
- *The interrupt (express path).* A sharp transient -- Tmax above an
  adaptive threshold (the 95th percentile of recent Tmax, floored at 6dB
  so a click in true quiet can still capture) -- yanks focus to the
  transient's band in ~50ms. (A 4x-running-median rule was tried first;
  in busy real-world audio the median itself sits high and the interrupt
  went completely deaf -- zero captures in 62s of Star Tours. The
  percentile rule fires on the most transient 5% of moments whatever the
  scene.) During the 100ms after any switch the bar triples (the
  attentional blink). Repetition habituates: the third identical onset
  within a second is expected, not news (stimulus-specific adaptation,
  and it keeps rhythmic scenes from thrashing attention). The interrupt
  never captures to a phantom: if the transient's band profile is
  all-zero below the hearing floor, there is no target.
- *Cross-modal (Level 2).* Two couplings, both bidirectional with
  vision: audio onsets tug saccades toward the panned side (ILD from
  the stereo mix) with a brief alerting gain; visual transients lower
  the auditory capture bar -- but only the top quartile of visual
  transient strength gets a vote (a linear boost let the film's constant
  wiggling double the capture count; the gate keeps it honest).
- *The scheduled path (the decision clock).* At most every 200ms -- the
  ears' analog of the ~3Hz saccade clock -- attention reconsiders: the most
  salient band wins, unless it's within 1.5 bins of where we already are.
  The attended band gets a processing gain (1.5x), which is what holds a
  stream at a cocktail party, and a distance cost discourages long jumps.
  Inhibition of return keeps it from ping-ponging.

Two rules with RazelKorr's fingerprints on them:
- *In quiet, dwell.* If the moment is below the silence threshold, don't
  go hunting among noise -- hold still. (From his description of trying to
  comprehend something difficult in the dark: more effort, less
  exploration, longer dwelling.)
- *Post-switch refractory.* For 100ms after a switch lands, the scheduled
  path holds -- the new band gets a moment to establish itself. (An
  earlier version scaled the whole salience map by 0.3; that changed
  nothing, because argmax is scale-invariant. The refractory is the honest
  mechanism.)

## The streaming ear (`hva/online.py`) — the flagship generation

The ear's magno channel: coarse, cheap, always-on — the auditory answer
to `hvp/motion.py`. Per **50 ms** perceptual moment (the master grid
shared with streaming vision) it computes exactly three floats:

- **RMS energy**,
- **spectral flux** (onset strength: the sum of positive spectral
  differences, computed with an NFFT of 1024 at 16 kHz), and
- **spectral centroid** (brightness).

Design decisions, from the module itself:

- Flux is **absolute**, not loudness-relative: a roar over an
  already-loud bed is still a big transient, and relative normalization
  was measured (2026-10-03) to rank the hyperspace roars 170th/202nd.
  Loudness-invariance instead comes from the adaptive threshold in
  onset picking.
- **Onsets (`pick_onsets`)** are peak-picked with a *local* threshold —
  median + 3·MAD of flux over a ±2 s window, an absolute floor of 0.1,
  an RMS floor of 1e-3 (below which digital silence does not onset on
  encode noise), local maxima in a ±2-moment neighbourhood, and a
  250 ms minimum gap enforced greedily by descending strength.
- Moment *features* are computed per chunk (streaming-pure; the caller
  threads the previous chunk's last magnitude frame through for flux
  continuity across chunk boundaries). The onset *decision* is post-hoc
  over the finished flux curve, mirroring how the visual event map is
  produced at end of run. That is documented in the module, not hidden.
- In `streaming/run_stream.py` this ear is **passive by design**: its
  features observe the master grid and supply onsets for binding
  (SPEC_FUSION) and for attended transcription, but they never steer
  the vision driver or render — the video path is bit-identical with
  audio on or off. Cost in the FF3 flagship run: 0.0356 ms per moment.

## Interpretation (`hva/interpret.py`) — labels and two-tier transcription

Undocumented in earlier versions of this spec; it is the FF3
interpretation pathway.

- **CLAP zero-shot labelling** (`label_window`): sound-event labels per
  window from the local CLAP model (weights
  `models/laion-clap-630k-audioset-best.pt`, 48 kHz).
- **Two-tier transcription** (`transcribe_two_tier`): faster-whisper
  **base** first; any segment whose `avg_logprob` falls below
  `LOW_CONF_THRESHOLD = −0.8` escalates to **medium**. The threshold is
  a single source of truth — the mishearing model imports it rather
  than restating it. On FF3, 110 interpretation windows were
  transcribed two-tier, 47 escalated, 0 errors; the base tier's
  "We're in Marvel!" hallucination at 1649.7 s is the tier system
  working as designed (the medium tier resolved the window to
  non-lexical vocalizations).
- Music features are also computed here; see the module for the
  inventory. Interpretation runs are driven by the `scripts/interpret_*.py`
  / FF3 driver scripts, not by the streaming runner itself.

## Bugs found and fixed (the real lessons)

1. **argmax of an all-zero field is 0, and 0 is a valid bin.** Twice this
   produced phantom jumps to bin 0 (50Hz): once when post-switch
   suppression + fresh inhibition-of-return crushed all salience to zero,
   and once when the interrupt's target profile was unmasked and the
   noisiest near-silent bin won. Guards: never schedule a switch when
   max(sal) <= 0; mask the temporal-contrast profile below the hearing
   floor (log diffs explode on silence).
2. **The decision clock was missing.** Without rate-limiting, the scheduled
   path re-decided every 100ms and attention thrashed between bands.
3. **Scale-invariant suppression.** See above -- multiplying a salience map
   by a constant doesn't change its argmax.

## The battery (all synthetic, all passing 2026-09-30, stable across runs)

- **A0:** a click in quiet is captured, ~0ms latency (express path).
- **A1:** an unattended stream glides 550->700Hz unnoticed (inattentional
  deafness -- the visual change-blindness analog).
- **A1b:** the same glide in the ATTENDED stream is noticed (~0.5-0.8s).
- **A2:** two simultaneous pip streams; the louder (cued) one holds
  attention 80-100% of the time (cocktail-party selection).
- **A3:** a loud click captures attention away from a continuing
  background (onset capture vs. sustained input).
- **A4:** post-switch refractory -- a new stream appearing during the
  100ms after a switch is held off (visible "refractory" in the trace),
  then wins promptly; control (no recent switch) orients immediately.

Known limitation: A2 uses simultaneous streams. Interleaved streams (taking
turns) need temporal prediction -- auditory streaming proper -- which is
future work, not v1. The frequency-switch controller is also not the full
story: real auditory attention is object-, stream-, time- and
space-selective, not just frequency-selective.

## What's next (status 2026-10-13)

The Star Tours 0-62s audio run and the quiet-dwell A/B are done (see
above). The open problem stands, with one experimental first step now
on record: auditory streaming / object formation — the pipeline knows
the waveform cold but can't tell the Starspeeder's engines from the
score (see "Stem streams" below for what the demucs experiment did and
did not change about that). Frequency-band attention is not the full
story: interleaved-stream segregation (taking-turns streams) needs
temporal prediction, and real auditory attention is object-, stream-,
time- and space-selective, not just frequency-selective.

## Star Tours 0-62s run (2026-09-30)

Ran end to end: 620 moments, cochleagram + salience + trace + event list
in `output/star_tours_62s/dwell/`, overview plot as `overview.png`.

- 26 onset-captures, 61 scheduled switches, 63 refractory holds.
- First 45s are restless: attention ping-pongs between bass (~100Hz)
  and midrange (~1-5kHz, plausibly effects vs. dialogue).
- Longest single dwell: 45.2s -> 55.5s parked at ~98Hz, a full ten
  seconds on a low rumble -- the run ends at the first hyperspace jump,
  so this reads as the jump sequence building.
- The cochleagram shows a constant harmonic drone at the low bins for
  the entire clip. Note on source: the input is the ride FILM's
  soundtrack, not cabin audio -- the film's mix itself emulates engine
  hum and environmental noise, and does it well enough that the pipeline
  treats it as the real thing (which is the sound designers' whole
  point).
- A/B of the quiet-dwell rule: dwell vs. no-dwell traces were
  IDENTICAL. Minimum moment loudness sat exactly at the silence
  threshold for the whole clip -- the rule never triggers on this ride.
  It matters for scenes with real silence, not this one.

Known limitation stands: the pipeline knows the waveform cold but can't
tell the Starspeeder's engines from the score -- auditory streaming /
object formation is the big v1 gap.

## Speech channel (2026-09-30): hva/transcribe.py

The v1 gap above has a first step. A local whisper model
(faster-whisper, base, int8 CPU) transcribes the 16 kHz mono track to
timestamped segments with per-word times and confidence scores,
aligned to the 100 ms moment grid (`align_to_moments`). Speech onsets
-- a segment starting after >= 0.4 s of silence -- become first-class
attention events (`speech_events`), so the attention system can treat
linguistic onsets the way it treats acoustic captures.

**Speech-gated attention (same night).** `speech_presence` turns the
transcript into a per-moment scalar in [0,1] -- binary word-span
presence plus a 300 ms exponential hangover so the auditory gain
releases on the map's own clock instead of chattering between words.
`scripts/run_level3.py --transcript` passes it to
`JointPriorityMap.step(..., speech=...)`, which scales the auditory
write by `(1 + SPEECH_BOOST * speech)`. Attenuation across inputs
falls out of the shared map's normalization rather than a separate
suppression knob: the Dr Tran data shows vision keeps working during
dense narration (title-card dwells), so globally muting vision during
speech would be wrong. Stream-level, not bin-level -- every frequency
bin is scaled equally, so a loud non-speech transient during speech
gets boosted too; stream separation (which bins carry the voice) is
the recorded v1 gap.

Why: the whole point of speech. Frequency-band attention hears
*transients*; dialogue-driven media is steered by *words*. The Dr Tran
runs showed the narrator's voice relocating visual attention (joint
peak yanked 170 px to the speaking screen) while the audio pipeline
itself had no representation of what was said.

Honest limits, stated up front:
- v1 is verbatim transcription -- superhuman. No human catches every
  word. Per-segment avg_logprob and per-word probabilities are
  recorded — and, since Phase 5 (2026-10-12), they are the input to
  an actual mishearing model; see the next section. The verbatim
  text remains the only transcript claim either way.
- "Perfect" hearing still misinterprets: transcription is not
  comprehension. Word sense, sarcasm, reference are not modeled.
  Deferred explicitly, not forgotten.
- No diarization in v1: turns are approximated from segment
  boundaries + gaps, not voice identity.

## The mishearing model (`hva/mishear.py`) — Phase 5, 2026-10-12

**Experimental. Not wired into the flagship streaming path.** The
module embodies the standing rule: *transcription claims carry
logprobs or they don't ship.*

- **Gate.** Only segments with `avg_logprob` below the interpret.py
  escalation threshold (−0.8, imported — one source of truth) are
  modelled, and within them only words with probability below 0.5
  (at most 3 words per segment, at most 5 alternatives each).
  Segments **without** a logprob are never modelled.
- **Method.** A self-contained Metaphone-lite phonetic key generates
  candidates from the whisper tokenizer vocabulary (exact-key tier,
  similarity 1.0; key edit-distance-1 tier, 0.5), rescored
  `(1 − word_prob) × phonetic similarity × frequency prior`. Scores
  are relative ranks, not probabilities. Candidate lists include
  sub-word BPE fragments ("maravil", "ine") — labelled model output,
  ranked by the same score; a wordlist-grade filter is future work.
- **Output, and the non-negotiable rule.** Per segment:
  `verbatim` (byte-identical; the input is never mutated),
  `confidence`, and `modeled_mishearings` — every record labelled
  MODEL OUTPUT with a disclaimer. **The model annotates doubt with
  phonetic alternatives; it NEVER replaces the verbatim transcript.**
- **What it cannot do, demonstrated.** On the FF3 1649.7 s window the
  base tier's "Marvel" (prob 0.091) models to *memorable / marble /
  maravil* — plausible lexical neighbours. None of the alternatives,
  for any word in the window, recovers what the medium tier found:
  non-lexical screams ("BLEH", "NOOOO"). The model proposes lexical
  neighbours of the words the base tier *thought* it heard; it cannot
  know the sound was non-lexical. It annotates doubt; it does not
  correct hallucinations into truth. That limitation is the phase's
  headline result (`output/ff3_hd_av/mishear_demo_1649.json`;
  tests `tests/test_mishear.py`, 8/8).

## Stem streams (`scripts/stem_streams.py`) — Phase 5, 2026-10-12

**Experimental. Not wired into the flagship streaming path.** A first
auditory-object layer, with its leakage caveats stated, not smoothed
over.

Demucs stems (vocals / no_vocals, via the restored
`--separate-vocals` path) each run through the streaming ear's
features + onset picking (`hva/online.py`), two-tier transcription,
and CLAP aggregation, emitting one stream record per stem in a single
JSON. Stems are demucs **model output**: a stem stream describes what
demucs isolated, not ground truth about the original mix.

- **At the label/transcript level, the stems separate voice from
  non-voice convincingly.** Synthetic mix (Star Tours vocals excerpt
  + a planted tone sequence at −6 dB, ground truth known): vocals-stem
  token containment 0.647 against the pre-mix transcript (bar 0.6;
  the famous lines survive, the opening line is garbled, and the
  transcript fragments into 13 micro-segments); no_vocals stem CLAP
  top-1 "singing" (a thin margin over "silence") with an empty
  transcript. Real-world demo (62 s of Star Tours): the vocals stem
  recovers 44 words of genuine dialogue, all segments at
  `avg_logprob ≥ −0.8`; the no_vocals stem transcribes 0 words with
  CLAP top-1 "music".
- **At the onset/feature level, they do NOT yet form clean objects —
  the stems leak.** Tone-burst times appear in the vocals stem's
  onsets and speech times in the no_vocals stem's; each stem
  approximates the union minus a little. Worse in the wild: on the
  real Star Tours demo the stems' onset counts (68 / 60) **exceed**
  the un-separated mix (54) — separation artifacts manufacture
  transients, so the per-stem onset sparsity seen on the synthetic
  mix is a synthetic-mix result, not a wild one.
- **Cross-cutting caveat.** This faster-whisper's per-segment
  `avg_logprob` is near-uniform within a run (−0.554 across all 13
  synthetic segments), so confidence discrimination in these outputs
  is coarse; the garbage-flag heuristic inherits that coarseness.

Verdict, in the phase summary's words: a first auditory-object layer
exists; it is leaky, and the leak is measured. The v1 gap named above
is narrowed at the label/transcript level, not closed.

First transcript (Dr Tran Ep 17, first 30 s): 14 segments, 5 speech
events, 80.9% of moments carrying speech. The narrator's affected
delivery bends the model ("Dr. Dre" for "Dr. Tran", "eye-time machine"
for "a time machine") but the content lands: "This one comes from AJ,
who writes," ... "do you have a time machine and a laser cannon?" ...
"Stop it! Just stop sending me letters!" -- the fan-mail frame the
visual analysis hypothesized, confirmed in the words.

Setup friction, recorded so nobody repeats it: faster-whisper will
not install on OS-managed system Pythons (PEP 668 + Debian RECORD
fights) -- use the pipeline venv. Its HF download client chokes on
some egress proxy configs (httpx URL parse failure); the model files
were fetched with curl -L (HF now redirects to a CDN) into
models/faster-whisper-base. Note: the Systran faster-whisper-{tiny,
base, small, medium} repos predate the ctranslate2 4.x model format --
they ship vocabulary.txt while ctranslate2>=4.0 demands
vocabulary.json. The vocab is identical across whisper sizes, so the
fix is large-v3's vocabulary.json dropped into the base model dir.
Also: the large-v3 preprocessor_config.json floating around claims
feature_size 128 -- whisper is 80; the wrong value breaks the encoder
shape. The repo-local preprocessor_config.json is corrected to 80.
