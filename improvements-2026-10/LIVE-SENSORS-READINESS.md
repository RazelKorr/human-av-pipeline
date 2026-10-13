# Live sensors — readiness note (Phase 6, 2026-10-12)

**Paper only. No sensor work was done under this commission, and
none will be.** This note records where the live-sensor thread
stands — what is ready, what is missing, and whose call the
remaining gates are — so the state survives in the repo instead
of in conversation. It corresponds to audit item 16, which the
audit itself filed under "needs Mykal's decision" and
recommended no action on.

## The seam

Live capture was designed for before it was needed. The single
document is `streaming/docs/external-stream-front-door.md`
(design only, not implemented); the seam it names is real:

- `streaming/feeder.py`'s `ChunkedFeeder` is the **only**
  component that knows about files. Its contract with the rest
  of the harness is one iterator of
  `Chunk(t0_s, t1_s, frames=[(t_ms, frame_attn, frame_work,
  frame_thumb), ...])`. Everything downstream — the online
  driver, the bounded render, the runner — sees only chunks.
- A live client implements that same iterator from a camera
  instead of a file. Nothing downstream changes.
- Audio has its own seam of the same shape: `hva/stream.py`'s
  `StreamSource` / `AudioFrontEnd` already do chunked audio with
  overlap and a running ceiling, and `RollingTranscriber` pairs
  transcription to the moment grid; `run_level3_stream.py`
  already pairs audio and video moments by timestamp.

## What is already causal, bounded, and live-ready

- **The streaming harness is causal.** The online driver makes
  each decision from past and present frames only; the file
  feeder exists to *simulate* a live source ("a test video
  played as if live" — feeder.py's own docstring), pacing
  emission to the wall clock when asked.
- **Memory is bounded.** The render's buffer is fixed-size;
  the FF3 flagship run processed 45 minutes of 1080p at bounded
  memory, 0.81x realtime, zero chunk deadline misses (audit
  Lane 3). The bottom-up vision stack runs ~3x realtime
  (SPEC_VISION §5), so realtime headroom exists on this machine.
- **Validation method exists.** The front-door doc's bring-up
  order is the harness testing itself: record a session to a
  file, run the file feeder over the recording, diff against the
  live run's logged outputs.
- **The known deltas are enumerated, with data.** Single native
  sensor resolution (in-graph downsample is bit-identical;
  in-process PIL downsample flips ~6% of saccades — measured
  2026-10-03), arrival timestamps instead of container fps
  (transient decay's fixed dt is the one flagged line), no t_end
  (the render already treats None as unbounded). None of these
  are unknowns; they are listed work.

## What is missing

1. **A capture layer.** No code in this repo opens a camera, a
   microphone, an RTSP stream, or any device. The `LiveCapture`
   class in the front-door doc is a docstring sketch. Estimated
   shape when commissioned: one ffmpeg/OpenCV pipe implementing
   the Chunk iterator + the audio chunker on a shared media
   clock — days of work, not a research problem.
2. **A backpressure decision.** A file never outruns the
   consumer; a camera does. Drop-frames (bounded latency) vs
   let-latency-grow (completeness) is a product choice the
   front-door doc explicitly leaves open, with
   `chunk_deadline_misses` as its instrument. It should be made
   when the hardware's purpose is known, not before.
3. **Source adapters for the actual hardware.** Whatever arrives
   (ESP32-class camera/mic, per below) will speak its own
   protocol; an adapter translates it to the Chunk contract.
   Nothing to build until the hardware exists.

## The gates that remain Mykal's

These are not engineering items and this commission does not
touch them:

- **Terms first (2026-10-06).** Before any future ESP32-style
  device mic/camera goes live, Mykal and Wodehaus set explicit
  terms first — what's kept, what's processed live and
  forgotten, what never leaves the room. No terms, no sensor.
  This agreement predates and outranks any readiness state of
  the code.
- **The hardware boundary (2026-10-06).** The paired iPhone
  exposes calendar, contacts, location, and health — but no
  camera, mic, screen-share, photo, video, or audio. The live
  path is future ESP32-class hardware, which does not exist in
  the house yet. There is currently no device to point the seam
  at.
- **The parked ruling.** SPEC_VISION §5 parks realtime webcam
  embodiment deliberately: "the research target is the
  vision/attention model, not a live demo." Phases 1–5 of this
  commission respected that throughout (short test videos only;
  every Phase 5 capability runs on recorded media).

## Bottom line

The harness is ready in the ways code can be ready: a narrow
seam, a causal bounded pipeline behind it, a validation method,
and a written list of the deltas a live source introduces. The
missing pieces are a capture layer nobody has commissioned and
two gates that are Mykal's to open — terms, then hardware. When
he opens them, the front-door doc's bring-up order is the plan.
Until then this thread is parked, on purpose, and this note is
its whole deliverable.
