# Streaming input — as-built (2026-09-30)

The emulator now runs on streams, not just uploaded files. A file played
as if live, an ffmpeg-readable URL, eventually a microphone: all of them
are just 10 Hz ticks that haven't happened yet.

## What was built

**`hva/stream.py`** — the online sensory front-end:

- `StreamSource`: yields a `Tick` per 100 ms from a file or URL.
  Video is decoded by an ffmpeg rawvideo pipe running the SAME filter
  chain as the batch decoder (`fps=10,scale=224:224,format=gray`) --
  bit-exact with `decode_gray` by construction. Audio is demuxed via
  PyAV, resampled to 16 kHz stereo, and binned by sample count.
  Moments pair video frame m with audio [m*1600,(m+1)*1600).
  `realtime=True` paces ticks to the wall clock (times `speed`).
- `VisionFrontEnd`: incremental retinotopic salience. The exact batch
  math (previous frame + decaying transient channel), stateful instead
  of a precomputed array.
- `AudioFrontEnd`: chunked DSP. 10 s chunks with 1 s overlap run the
  exact batch STFT/transient/ILD code; only the central 10 s of moments
  are emitted, so chunk edges never touch the output. The Tprof ceiling
  is a running percentile-99 over history (batch uses the whole run;
  inject the batch ceiling for exact verification).
- `RollingTranscriber`: windowed faster-whisper (30 s window, 10 s
  step), stitched by start time. Speech presence per tick uses only
  already-transcribed words -- the gate honestly trails reality by the
  transcription lag.

**`hvm/online.py`** — `OnlineLevel3.tick(...)`: the Level 3 closed loop
as a stateful tick function. Batch (`driver.py`) and streaming both
delegate to the same decision logic; the math didn't change, only the
scaffolding.

**`scripts/run_level3_stream.py`** — the stream runner:

- `--src`: file or PyAV-readable URL.
- `--realtime`, `--speed`: wall-clock pacing.
- `--seconds`: stream length for files.
- `--verify joint_maps.npz`: compares stream peaks against batch peaks.
- `--inject-ceiling`: batch Tprof p99 ceiling for exact verification.
- `--transcribe`: rolling Whisper on the stream.
- Saves joint/vision-only maps and transcripts.

## Verification (2026-09-30)

62-second Star Tours clip, 620 moments:

- Stream vs batch joint peaks: **max difference 0.0000 map-px**.
- Stream vs batch vision-only peaks: **max difference 0.0000 map-px**.
- Saccade-count equivalence: **206/206** (stream vs batch, exact).
- Identical statistics: 113 moments (18.2%) audio-moved, mean 13.6 px,
  max 188.2 px.
- **6 regression tests** in `tests/test_streaming.py` (all pass).

The stream is bit-exact with the batch. Three bugs were found and fixed
during verification:

1. **`decode_gray` fps bug.** The vf chain lacked `fps={fps}`, so ffmpeg
   emitted native-fps frames while the loop read/labeled them at 10 fps
   -- for a 30 fps source, a 62 s run watched the first ~21 s of video
   stretched across the whole timeline, desynced from audio. Every
   Level 3 batch run before this fix had the bug. (The fix had to be
   applied to BOTH copies of `run_video.py`: the batch imports from
   `human-av-pipeline/scripts/`, not `human-vision-pipeline/scripts/`.)
2. **PyAV audio plane padding.** `bytes(frame.planes[0])` includes padded
   samples beyond `frame.samples`. Reading the whole plane produced
   garbage/misaligned audio. Fixed by slicing to `samples * channels`.
3. **PyAV reformat vs ffmpeg scaler.** `frame.reformat(format="gray")`
   differed by ~2 LSB from ffmpeg's `scale+format=gray`. The salience
   normalization amplified this on dark frames. Replaced with an ffmpeg
   rawvideo pipe using the exact batch filter chain.

## What streaming does NOT do yet

- **Demucs** is non-causal (needs the whole file). The live path
  transcribes the raw mix. Perception lags speech by ~10-30 s; the
  priority map stays real-time. Documented cost of causality, not a bug.
- **Realtime wall-clock (measured)**: 20 s of video in 21.9 s wall-clock,
  compute at 0.36x realtime -- keeps pace. See `docs/saying-hi.md` latency budget.
- **Rolling Whisper latency (measured)**: 30 s clip -> 2 windows, 4 segments,
  15.6 s processing -- keeps up with realtime.
- **"Saying hi"** has shipped: turn detection (`TurnDetector`), response policy
  (`ResponsePolicy` v3), non-blocking TTS (`AsyncTTS`), live duplex + barge-in.
  Remaining: mic capture, output-device plumbing on real hardware, echo handling.

## Design notes for live use

- Backpressure: if inference lags, the tick loop should drop to
  moment-skipping, never buffer unboundedly. A 30 s late "perception"
  is a failure, not a delay. (Not yet implemented -- the current code
  processes every tick.)
- Clock discipline: uploads have perfect timestamps; live streams have
  jitter. The moment grid will need a real monotonic clock with
  interpolation when frames arrive late. (Current code assumes 100 ms
  spacing from the source.)
- A stalled video stalls emission until EOF. Correct for files; live
  video stalls are a documented v1 limitation.

None of this claims validated human perception -- same honesty bar as
the rest of the repo. It's an emulator with known, documented
deviations, and the deviations are the interesting part.
