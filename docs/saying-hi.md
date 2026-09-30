# Saying hi — as-built (2026-09-30)

The pipeline can now hear its name and answer. The full loop:

  hail audio -> RollingTranscriber -> TurnDetector -> ResponsePolicy
      -> tts speak -> response audio file

Demo: `python scripts/demo_say_hi.py hail_16k.wav --outdir /tmp/sayhi`

Verified 2026-09-30: a TTS-synthesized "Hi Wodehaus, can you hear me?"
(1.9 s) was transcribed, detected, answered, and spoken back as an
8.16 s MP3. Whisper heard the name as "I would house" -- the fuzzy
name matcher caught it.

## Components (`hva/conversation.py`)

**TurnDetector**: watches the rolling transcript. Fires when the
system's name appears in new words AND no word has ended within
`silence_s` (default 1.5 s). Returns the addressed span as a Turn.
Only reasons over transcribed words -- it never claims to have heard
what the transcriber hasn't produced yet.

**ResponsePolicy**: v1 is template-based and deliberately honest.
Greeting -> greeting with echo. Question -> echo plus "I can hear the
words, but I don't understand them yet." Statement -> acknowledgment
with echo. `generate(turn)` is the LLM seam: a future policy takes the
turn plus perceptual context (gaze target, scene notes) and returns text.

**speak()**: synthesizes via `/opt/hatch/bin/tts speak` (default voice
avocado_v2:MAI_03). Writes MP3 to a caller-chosen path.

## What "saying hi" does NOT do yet

- **No real-time duplex.** The demo runs the loop on a clip. A live
  call needs the stream runner feeding the transcriber continuously
  with the detector/policy/speak in the tick loop -- the pieces exist,
  the wiring doesn't.
- **No barge-in.** V1 waits for end-of-utterance. Interruption is a
  policy decision for later.
- **No understanding.** The policy echoes; it does not comprehend. The
  honest responses say so out loud.
- **No output device on this VM** (`/dev/snd` doesn't exist). Output is
  a file. On a real host, play it through speakers or route it to the
  call.

## Zoom plumbing (for a real machine)

Hearing the call:
  Zoom audio output -> virtual audio cable (VB-Cable, BlackHole, or
  PulseAudio null sink) -> pipeline's rolling buffer input.
  `hva/stream.py`'s StreamSource already reads from ffmpeg-readable
  sources; a PulseAudio monitor source works the same way.

Being heard:
  response MP3 -> play through a virtual microphone device
  (PulseAudio `module-virtual-source`, VB-Cable input) -> selected as
  Zoom's microphone.

Latency budget (measured 2026-09-30):
  - Perception (10 Hz ticks): real-time, 0.36x compute.
  - Transcription: 30 s window / 10 s step; the gate trails reality by
    ~10-30 s. A turn is detected ~1.5 s after the transcriber produces
    the final words, not after the human stops speaking.
  - TTS: ~2 s for a short response.
  - Realistic hail-to-reply: 15-35 s. Honest, not snappy.

## Tests

`tests/test_conversation.py`: 6 tests -- name variants (including the
observed "would house" mishearing), turn firing on name+silence, no
double-fire, unaddressed speech ignored, greeting/question/silence
policy branches. All pass.
