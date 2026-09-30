# Human Auditory Attention Pipeline (HAP) — SPEC

Experimental demonstration, not a validated model of hearing. Built 2026-09-30.

## What it is

A pipeline that listens to audio the way the vision pipeline watches video:
sound goes in, and out comes a trace of where "attention" was, moment by
moment, plus a list of events (captures, switches, quiet dwelling). The
clock is shared with vision: 100ms perceptual moments, so the two can
eventually be fused.

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

## What's next

Run the Star Tours 0-62s audio end to end: cochleagram, salience map,
attention trace, event list. A/B the quiet-dwell rule. Then think about
what "stream" means before calling frequency switching the analog of
saccades.

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

Why: the whole point of speech. Frequency-band attention hears
*transients*; dialogue-driven media is steered by *words*. The Dr Tran
runs showed the narrator's voice relocating visual attention (joint
peak yanked 170 px to the speaking screen) while the audio pipeline
itself had no representation of what was said.

Honest limits, stated up front:
- v1 is verbatim transcription -- superhuman. No human catches every
  word. Per-segment avg_logprob and per-word probabilities are
  recorded as the hook for a future mishearing model.
- "Perfect" hearing still misinterprets: transcription is not
  comprehension. Word sense, sarcasm, reference are not modeled.
  Deferred explicitly, not forgotten.
- No diarization in v1: turns are approximated from segment
  boundaries + gaps, not voice identity.

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
