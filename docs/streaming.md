# Streaming input & live presence — vision notes

Long-term direction: the emulator should run on live streams, not just
uploaded files — and eventually hold its own in a Zoom call. What's
already there, what's missing, and what "better than what they ship"
means.

## What already works (2026-09-30)

- **Any-format decode**: `hva.transcribe.load_audio` reads wav/mp3/mp4 via
  PyAV. The decode path is no longer the bottleneck.
- **Vocal isolation as a stage**: `--separate-vocals` routes music-heavy
  mixes through Demucs before transcription. Live music + speech is the
  norm, not the exception.
- **Moment grid**: the whole pipeline already ticks at 10 Hz perceptual
  moments. A stream is just moments that haven't happened yet.
- **Stateful map**: `JointPriorityMap` integrates with decay; it doesn't
  need the whole timeline upfront. The closed loop is already online in
  structure, just not in plumbing.

## What streaming actually requires

1. **Chunked decode, rolling buffer.** Replace "decode N seconds, then
   process" with: PyAV reads packets continuously, frames append to a
   ring buffer (~5 s video, ~30 s audio). The 10 Hz tick pulls the
   newest complete moment. Backpressure: if inference lags, drop to
   moment-skipping (never buffer unboundedly — a 30 s late "perception"
   is a failure, not a delay).
2. **Online transcription.** faster-whisper is batch-oriented. Options:
   run it on rolling 30 s windows with 5 s overlap and stitch segments
   (dedup by word timestamps), or swap in a streaming ASR (Whisper
   streaming wrappers, or a dedicated online model). The stitching
   approach keeps the current model and prompt machinery.
3. **Online Demucs.** mdx_extra is non-causal (needs the full clip).
   For live use: causal separators (Demucs streaming mode exists but
   quality drops), or run separation on the 30 s transcription window
   as a lookahead stage — perception lags ~30 s on speech content,
   stays real-time on the priority map. Honest tradeoff, document it.
4. **Stateful everything.** `run_level3.py` is a batch script: it
   decodes the whole video, builds all arrays, then loops. The live
   version is a tick function: `tick(frame, audio_chunk, dt)` updating
   one persistent `JointPriorityMap`, one `AuditoryAttention`, one
   transcript buffer. The math doesn't change; the scaffolding does.
5. **Clock discipline.** Uploads have perfect timestamps. Streams have
   jitter. The moment grid needs a real clock (monotonic) with
   interpolation when a frame arrives late, not assumption of 100 ms
   spacing.

## Zoom-call presence ("saying hi")

Two halves, very different difficulty:

- **Hearing the call (easy-ish).** A virtual microphone (PulseAudio
  null sink / BlackHole on Mac) feeding the rolling buffer. This is
  plumbing, not research.
- **Being seen/heard (the real project).** A virtual camera output
  means rendering *something*: at minimum a visualization of the
  priority map / attention state (honest — "this is what I'm
  attending to"), at most an avatar. Speaking means TTS wired to a
  response policy, which is a whole second system (dialogue manager,
  turn-taking from the speech-event detector — which already exists:
  `speech_events`). Don't fake it: a "hi" that's just a triggered
  sample is a parlor trick. The real thing is the attention system
  deciding *when* to speak from what it heard.

## What "better than what they ship" means

Stock video-call AI: transcribes everything, understands nothing about
*attention* — no notion of what was salient, what captured the ear,
when the voice overrode the eyes. This pipeline's edge is the joint
priority map: it doesn't just hear the call, it has a gaze. It can say
"you looked away when the bass dropped" or "everyone's voice pulled my
attention at 0:42" — perceptual claims, not transcript summaries.
That's the moat. Everything else (transcription, TTS, virtual cam) is
commodity plumbing around it.

## Suggested build order

1. Tick-function refactor of the Level 3 loop (no behavior change on
   uploads; proves the online structure).
2. Rolling-buffer file streamer: feed an mp4 as if live, verify the
   tick output matches the batch output.
3. Windowed online transcription with stitching.
4. Real stream source (RTMP/test stream), clock discipline.
5. Virtual mic input; map visualization as virtual camera.
6. Turn-taking + TTS: the actual "hi".

None of this claims validated human perception — same honesty bar as
the rest of the repo. It's an emulator with known, documented
deviations, and the deviations are the interesting part.
