# Real-scene grounding test (2026-09-30)

First test of "Wodehaus, what do you see?" against real visual content
instead of a black frame. 45 s of Dr. Tran ep17 (cartoon room: window
with mountains, jack-o'-lantern upper-right, low table with a box, a
small character in blue) with two TTS hails mixed in:

- t=2.0 s: "Hey Wodehaus, what do you see?" (soundtrack ducked to 0 under it)
- t=25.0 s: "Wodehaus, look left." (soundtrack ducked to 0 under it)

Full loop: video/audio stream -> Whisper -> turn detection ->
Pollinations (POST) -> reply text. Run: `scripts/run_conversation.py
--src /tmp/ground/ground_test.mp4 --seconds 45 --llm free`.

## What the model said

"What do you see?" (x3, plus one partial hail):

- "I'm looking at the center of the frame. The most salient point
  right now is center. I've made 20 saccades so far."
- "I'm looking at the left of the frame, and the most salient point
  right now is the upper right corner. I've made 45 saccades so far."

"Look left": "Looking left." (bias written to the shared map via the
LOOK backstop path).

## Verdict: honest, thin, no object hallucination

The answers report exactly what the perceptual snapshot contains --
gaze location, most-salient-point position, saccade count -- and do
not invent objects. The model did NOT claim to see a pumpkin, a
character, or a window. The "upper right corner" salient point is
*consistent* with the jack-o'-lantern's location (not proven to be
tracking it; the snapshot carries positions, not identities).

This is the correct epistemic ceiling for a priority-map perceptual
state with no object recognition. The gap is real and now empirically
confirmed rather than merely asserted: a human asking "what do you
see?" expects object-level description, and the system cannot give
it. Named-object recognition remains the missing piece, and the
replies must keep declining to name things.

## Turn-detection issues found (fed into the phantom-turn defense)

1. **Duplicate turns from window re-stitching.** One hail produced
   four turns (t=4.1, 6.0, 9.0, 13.4 s). Overlapping 30 s Whisper
   windows re-transcribed the same audio with slightly different word
   timings, defeating the `_words_seen` watermark. Same-utterance
   re-fires must be suppressed by temporal overlap with the previous
   turn's span, not by the word-count watermark alone.
2. **Background-dialogue bleed.** The t=13.4 s turn glued cartoon
   dialogue onto the hail ("I could be taking some makeup, blah,
   blah... It's time for Dr. Trey family"). The LLM ignored it, but
   turn text should ideally end at the addressed utterance.
3. **VAD vs. soundtrack (test artifact, documented).** The file-based
   setup puts the video's soundtrack on the "mic", so EnergyVAD read
   cartoon dialogue as user speech and every reply was dropped as
   stale during synthesis. In the real deployment the mic carries
   only the user, so this is a harness limitation, not a product bug
   -- but it confirms the VAD cannot distinguish *whose* speech it
   hears, which matters for the Zoom far-end-audio case.

## Defense implemented and validated (same day)

`TurnDetector` now has two evidence-based guards (no phrase
blacklists), wired into `run_conversation.py` via per-tick
`EnergyVAD.hot` history:

- **Acoustic agreement:** the turn's addressed span must show speech
  energy in the acoustic record (default >= 20% of the span). The
  original phantom ("Woodhouse, look left" over bit-exact silence)
  is suppressed as `no_speech`; a control with real speech in the
  same span fires normally.
- **Re-fire suppression:** a turn whose addressed span overlaps the
  previous fired turn's span (+1 s margin) is suppressed as `refire`.

Re-running this clip: 6 turns -> 2 turns (one per hail), 3
suppressed as refire, 0 no_speech. Suppressions are logged and
counted in the end-of-stream summary.

## Test-harness lessons

- A hail mixed at 2.5x over 0.4x cartoon audio still loses the name:
  the rolling 30 s window transcribed "Hey Wodehaus" as "Hey, who's
  that?" Competing speech destroys the address token even when a 5 s
  isolated clip transcribes it fine. The hail needs a clean channel
  (soundtrack ducked to 0), which matches the real use case: a person
  speaking to a mic, not overpowering a soundtrack.
- ffmpeg `volume='if(between(t,...))'` evaluates the expression once
  at init, not per frame. Use `volume=0:enable='between(t,...)'`
  after `asetpts=PTS-STARTPTS` for sample-accurate ducking.
