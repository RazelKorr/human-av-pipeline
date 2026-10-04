# External-stream front door (design only — not implemented)

How an RTSP / webcam / other live capture client would plug into the
streaming harness in place of the file feeder. No live sensors are
touched by this design; it is a sketch for when hardware arrives.

## The seam

`ChunkedFeeder` is the only component that knows about files. Its
contract with the rest of the harness is narrow:

  yields Chunk(t0_s, t1_s, frames=[(t_ms, frame_attn, frame_work, frame_thumb), ...])

where `frame_attn` is 224x224x3 float32 RGB, `frame_work` is WxHx3
float32 RGB, and `frame_thumb` is the motion-thumbnail (96x54 default,
WxH float32 RGB, None when no thumb output was requested), all in 0..1,
with `t_ms` media timestamps in nondecreasing order. Everything
downstream (driver, render, runner) only ever sees chunks.

A live capture client implements the same contract. Concretely:

```python
class LiveCapture:                      # design sketch, not code
    def __init__(self, src, work_wh=(640, 360), chunk_s=2.0):
        # src: "0" (webcam index), "rtsp://…", "udp://…"
        # one ffmpeg/openCV pipe at work_wh; downsample to 224 in-process
    def __iter__(self):                 # yields Chunk, forever
        ...
```

## What changes vs the file feeder

1. **Single capture, in-process or graph downsample.** The file feeder
   now runs ONE ffmpeg process with a 3-output split filter graph
   (224px attention + work res + motion thumb), bit-identical to the old
   two-pipe decode; `--dual-decode` keeps the original two-pipe path for
   reproducibility. A sensor has one native resolution; the 224px
   attention input becomes a downsample of the captured frame. The
   downsample question now has measured data (2026-10-03): in-process
   PIL bilinear diverged on 6% of saccades (close-call argmax flips,
   max 11.4 work-px fixation displacement, energies to 2.0e-02) vs a
   libswscale-in-graph downsample, which was bit-identical on every
   metric. A live client should prefer the in-graph path (one ffmpeg
   process, split filter) and re-validate against the split default.

2. **Timestamps come from arrival, not the container.** The file feeder
   labels frames from the container's fps. A live client stamps
   `t_ms` from a monotonic clock at capture. Frame intervals will
   jitter; the driver and render already tolerate irregular spacing
   (they index by timestamp, never by frame count) — but the transient
   decay uses a fixed `dt = 1000/fps`. For jittery sources, replace it
   with the measured inter-frame dt (one-line change in
   OnlineAttentionDriver; flagged, not made, because it changes the
   validated math).

3. **No t_end.** The file feeder knows the media duration and the
   driver drops saccades past it. A live stream has no end: pass
   `t_end_ms=inf` and let the decision clock run. The render's
   `pull()` already treats `t_end_ms=None` as unbounded.

4. **Backpressure policy (the real design decision).** A file never
   outruns the consumer; a camera does. When a chunk's processing
   exceeds its media duration, the harness must choose:
   - **drop frames** (keep latency bounded; lose moments), or
   - **let latency grow** (keep every moment; fall behind realtime).
   The runner's `chunk_deadline_misses` counter is the instrument for
   this choice. For a gadget that converses, bounded latency wins;
   for a gadget that reviews, completeness wins. Decide per product.

5. **Audio joins here.** This harness is vision-only (the v2 path).
   The Level-3 streaming code (`hva/stream.py`: StreamSource,
   AudioFrontEnd, RollingTranscriber) already solves chunked audio
   with overlap and a running ceiling. A live A/V client pairs one
   audio chunker with the video chunker on the same media clock;
   moments pair by timestamp as they already do in
   `run_level3_stream.py`.

## What does NOT change

- `OnlineAttentionDriver` and `OnlineVisionPipeline` are
  source-agnostic; they never touch the filesystem or the network.
- The bounded buffer, the t_end cap logic, the moment grid, and the
  output artifacts are identical.
- Validation stays the same: record a live session to a file, run the
  file feeder over the recording, and diff against the live run's
  logged outputs. The harness is its own test fixture.

## Bring-up order (when hardware exists)

1. Record 60 s from the target camera to a file.
2. Run the file feeder + compare against a live-capture dry run of
   the same recording; quantify the downsample divergence (item 1).
3. Point the live client at the real camera; watch
   `chunk_deadline_misses` and pick the backpressure policy (item 4).
4. Only then: audio pairing (item 5).
