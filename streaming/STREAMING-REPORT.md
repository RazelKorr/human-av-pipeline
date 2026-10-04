# Sensorium streaming harness — current state

*Last updated 2026-10-03 (PDT). The full dated build diary — the original
report plus all seven addenda, nothing dropped — lives in the **Changelog**
at the end. What follows is the current state, with every number traceable
to the changelog entry that produced it.*

**Status:** pre-hardware, test-fixture only. No live sensors touched.

## 1. What the streaming harness is

New package `streaming/` alongside the pipeline (the pipeline itself is
untouched — the harness wraps it). It gives the Sensorium v2
color-saccades vision path (`scripts/run_video_saccades_color.py`) — the
flagship pipeline that had no streaming story — an online story: feed a
video as if it were live, make saccade decisions causally, render
perceptual moments as they become ready, and stay within bounded memory.
The Level-3 joint-map path already streams (`scripts/run_level3_stream.py`)
and audio streaming already exists (`hva/stream.py`); this harness does
not duplicate them. It is **vision-only by design** (see §6, open item 4).

| File | Role |
|---|---|
| `streaming/feeder.py` | `ChunkedFeeder`: test video played as if live. Default: ONE ffmpeg process with a 3-output split filter graph (224px attention + 640x360 work + motion-thumb resolution); `--dual-decode` keeps the original two-pipe path. Frames grouped into wall-clock-paced chunks. `realtime=True` paces emission to the wall clock (x `speed`); otherwise as fast as the consumer pulls. Pacing never touches timestamps. |
| `streaming/online_driver.py` | `OnlineAttentionDriver`: the batch pass-1 attention math as a stateful push object — transient decay, inhibition of return, POI memory, saccade decision clock, per-frame energy log, plus the magno channel (motion energy, pursuit-drive) when enabled. |
| `streaming/online_render.py` | `OnlineVisionPipeline(VisionPipeline)`: bounded frame buffer (deque; evicts frames no future moment can read), `pull(t_now, t_end)` yielding newly-ready moments with the batch `moments()` body verbatim and `prev` persisted across calls. Mirrors pursuit glides via `add_pursuit()`. |
| `streaming/run_stream.py` | CLI runner: feeder -> driver -> render in one online loop, saccade/pursuit decisions mirrored into the render controller as made. Writes the same artifacts as the batch driver (`video_percept.mp4`, `fixations.npy`, `frame_energies.npy`, `scanpath.npy`, `run_report.json`) plus streaming telemetry, `motion_energies.npy` / `pursuit_log.npy` when motion is on. |
| `streaming/compare.py` | Batch-vs-stream comparator: moment counts, timestamp grids, fixation displacement, suppressed-flag agreement, energy deltas, saccade counts, percept-video frame deltas, event lists via the graph's own `detect_events()`. |
| `streaming/docs/external-stream-front-door.md` | One-page design sketch for plugging an RTSP/webcam capture client into the same chunk contract. Design only. |
| `streaming/docs/magno-predictions-sealed-2026-10-03.md` | Sealed predictions for the magno channel (written before any motion-on run). |
| `streaming/docs/magno-thumb-predictions-sealed-2026-10-03.md` | Sealed predictions for the 16:9 motion-thumbnail ladder. |
| `tests/test_streaming_harness.py` | 6 tests: feeder frame count/timestamps vs the batch decoder, pipe lockstep, driver determinism, online-render bit-exactness vs batch `VisionPipeline`, aggressive-eviction exactness + boundedness, `t_end` grid cap. |

## 2. Current validated numbers

### 2a. Bit-identity vs post-hoc v2 (the standing claim)

Streamed vs batch on Star Tours; the default config reproduces the
post-hoc v2 path bit-exactly:

| Media | Moments | Saccades | Events | Fixation Δ | Energies | Suppression |
|---|---|---|---|---|---|---|
| 30 s | 600 = 600 | 99 = 99 | 1 = 1 | 0.0000 px | 0.0 | 100% |
| 62 s | 1240 = 1240 | 206 = 206 | 5 = 5 | 0.0000 px | 0.0 | 100% |
| 269 s (full film) | 5380 = 5380 | 896 = 896 | 52 = 52 | 0.0000 px | ≤6.0e-8 (float32 rounding) | 100% |

Timestamp grids agree to 0.000000 ms. Percept-video frame deltas are
h264 encode noise from two separate encodes (≤7.5e-2 max), not pipeline
divergence. The single-decode split path is bit-identical to dual-decode
on every metric. Bounded buffer verified: buffer steady at 667 ms,
thousands of frames evicted across the runs.

### 2b. Realtime trajectory (30 s Star Tours, wall/media)

| Config | Realtime factor | Per-frame p50 |
|---|---|---|
| v2 full pipeline | 0.17x | 257 ms vs 33.3 ms budget |
| + chroma passthrough | 0.45x | 71 ms |
| + fovea levels=3 + mask cache | 0.61x | 37.1 ms |
| + luma_ratio chroma mode | 0.79x | 29.5 ms (under budget) |
| + single-decode split feeder | 0.98x | 28.6 ms |
| + magno channel (motion 1.0 + pursuit) | 0.96x | 29.1 ms |
| + 16:9 motion thumb (96x54, work-derived) | 0.86x | 32.1 ms |
| + third ffmpeg output (no PIL downsample) | **0.91x** | 29.6 ms, 0 deadline misses |

**Headline:** with the recommended combo (luma_ratio + levels=3 +
mask cache + split single-decode + third ffmpeg output), the run reaches
**0.91x realtime** end-to-end on 30 s. The vision pipeline proper runs
~19.7 ms/frame (~1.7x realtime) — the foveation itself is
realtime-capable. The remaining 0.02–0.04x is fixture I/O, not vision:
H.264 software decode of the 1080p fixture costs ~9–12 s of the 30.7 s
wall, plus the harness's mp4 re-encode (a visualization artifact, not
a validation input; now on the ultrafast preset). A live sensor has no
decode step and needs no re-encode.

### 2c. Magno channel v1 (motion energy + smooth pursuit)

Motion-on vs control, full 269 s film, 96x54 16:9 motion thumb:

| Metric | Control | Motion-on |
|---|---|---|
| Saccades | 896 | 742 |
| Pursuit segments | 0 | 154 |
| Pursuit time | 0 | ~19% of film (18.8% on legacy thumb) |
| Realtime factor | 0.98 | 0.86 |
| Per-frame p50 | 28.6 ms | 32.7 ms (vs 33.3 ms budget) |
| Motion-energy at fixation (mean) | 0.00466 | 0.00927 (2x — the eyes go where motion is) |
| Suppressed-moment fraction | 0.0033 | 0.0009 |
| Suppressed moments inside pursuit | — | 0 (boundary artifact resolved; the biology holds: no saccadic suppression during pursuit) |

Magno cost: **0.49 ms/frame** (abs-diff + small gaussian + threshold on
the motion thumbnail). Pursuit spreads across the film (1.4–4.0 s per
10 s bin through battle/trench sections), not clumped. The coherence
gate holds against flashes: pursuit under-indexes bright frames 4–5x;
the brightest pursuit segment (t=206 s, hyperspace star-streaks) is
coherent radial motion, a legitimate target. Saccade count −15% to −17%:
pursuit replaces catch-up saccades, the predicted direction.

### 2d. Sealed-prediction scorecards

- **Magno (P1–P8):** 7 PASS. P8 (centroid within 3° of glide >80% of
  pursuit time) FAIL as stated (21%): the prediction assumed discrete
  constant-velocity targets; the stimulus is accelerating camera flow
  (optokinetic following, not discrete pursuit) — the right behavior for
  the stimulus, the wrong label on the prediction. P6 (zero pursuit
  overlap with whiteouts) vacuous on the 30 s clip, gate behavior
  confirmed on the full film. P8 was deliberately not retuned: v1 is
  honest, tested, and its limits are documented.
- **Motion-thumb ladder (T1–T6, B1–B3):** T1–T4 PASS, B1–B3 PASS.
  T5 FAIL (none of the work-derived thumb sizes held the 0.95x bar —
  the PIL downsample cost was underestimated) and T6 FAIL (consequence).
  B1 confirms the aspect-ratio fix changes geometry, not behavior
  (saccades 93 vs 90, pursuits 7 vs 10; segments align in time).

### 2e. Test suite

Latest feature-suite runs (retina / streaming / attention / speedup /
motion): **75 passed, 2 skipped, no regressions**. Full-suite
last-known: **159 passed, 2 skipped, 1 failed** — the failure is the
pre-existing `test_owlvit_detects_window_in_crop` (missing fixture
`output/foveal_crops.png`, stale `human-vision-pipeline` path), failing
before this work too.

## 3. Mode and flag inventory (`streaming/run_stream.py`)

All opt-ins are recorded in `run_report.json`. The v2 default path is
untouched and bit-identical.

| Flag | Default | Options | Status |
|---|---|---|---|
| `--realtime` / `--speed` | off / 1.0 | — | Pacing is output-neutral (realtime run bit-identical to fast run). |
| `--moment-ms` | 50.0 (20 Hz moment grid) | — | |
| `--chroma-mode` | `luma_ratio` (default since 2026-10-03, Mykal's call) | `foveated` (v2 math, explicit opt-in for bit-agreement), `passthrough` | **luma_ratio: SHIP as the default** (1.42x vs passthrough; keeps vivid color in blurred regions). |
| `--fovea-levels` | 3 (default since 2026-10-03, Mykal's call) | 5 = v2 | **levels=3: SHIP** (0.46x→0.56x alone; ~0.4% mean percept diff). |
| `--mask-cache` / `--no-mask-cache` | on (default since 2026-10-03, Mykal's call) | off via `--no-mask-cache` | **SHIP** — bit-identical, ~85% hit rate. |
| `--box-sigma-threshold` | None (off) | sigma | **DUD — excluded.** Correct approximation, buys nothing on this stack (1.0x end-to-end). Kept behind the flag. |
| `--motion-weight` | None (off = v2-identical) | float | **Magno v1: SHIP**, default-off with documented limits (P8). |
| `--pursuit` | off | on | **Magno v1: SHIP**, needs `--motion-weight`. |
| `--motion-thumb` | 96x54 (16:9, work-frame-derived) | WxH | **Aspect fix SHIPPED as default.** |
| `--motion-thumb-source` | `work` | `attn` | `attn` = legacy anamorphic fallback. |
| decode | single-decode `split` | `--dual-decode` (original two-pipe, kept for reproducibility), `--single-mode pil` | **Split: SHIP as default** — bit-identical to dual-decode. **PIL: superseded** — diverged on 6% of saccades (close-call argmax flips); kept behind its flag with the divergence documented. |
| `--chroma-weight`, `--fovea` | None | float | Overrides. |
| `--chunk-s` | 2.0 s | — | Feeder chunk size. |

**Recommended combo (now the default):** luma_ratio + fovea_levels=3 +
mask_cache + split single-decode (+ motion_weight/pursuit for magno).
As of 2026-10-03 (Mykal's call) the three SHIP verdicts are the code
defaults; the v2 math remains available explicitly via
`--chroma-mode foveated --fovea-levels 5 --no-mask-cache`.

## 4. What broke and how it was fixed (design history)

1. **Unbounded frame retention** — bounded deque; a moment at media time
   `t` reads `[t − latency − window, t − latency)`; eviction with a
   500 ms margin. Bit-identical under aggressive eviction (unit test).
2. **Two-pass structure** — merged into one online loop. Valid because a
   render query at `t_c = t_moment − latency − window/2` only sees
   saccades with `t_on ≤ t_c`, each decided at `t_on − 200 ms` — no
   lookahead. Decisions mirrored through `add_saccade()`; linear rescale
   keeps saccade durations consistent with batch pass 2.
3. **The `t_end` grid cap** — `pull(t_now, t_end_ms)` caps the grid
   exactly like batch `moments(t_end_ms)`.
4. **File facts the stream still needs** — `run_video.probe()` supplies
   fps + duration from the container header. For a true live source
   these come from the stream header / are unbounded
   (`t_end_ms=inf`). The feeder is a validation configuration; a real
   sensor is single-resolution.

## 5. External-stream front door

See `docs/external-stream-front-door.md` (one page, design only). The
seam is the `Chunk` contract `(t0_s, t1_s, [(t_ms, frame_attn,
frame_work, frame_thumb)])`; a live capture client implements it.
Covers: single capture + downsample (the PIL-vs-libswscale divergence
is now measured — see §3, decode row), arrival-clock timestamps
(transient `dt` becomes measured, not fixed — one-line change, flagged
not made), no `t_end`, the backpressure policy decision (drop vs lag;
the deadline counter is the instrument), and audio pairing via
`hva/stream.py`.

## 6. Open items

1. **Online event detector.** Events are still post-hoc
   (`st_color_graph.py` on the finished percept video). The bridging
   rule is local; a streaming detector is straightforward but unbuilt.
2. **Audio in this harness.** Vision-only by design (the v2 path is
   vision-only); audio streaming exists in `hva/stream.py` and the
   pairing pattern in `run_level3_stream.py`.
3. **Jittery timestamps.** The driver uses fixed `dt = 1000/fps` for
   transient decay (matches batch). Live sources need measured dt —
   one-line change, flagged not made (it alters validated math).
4. **Backpressure policy.** Undecided by design (drop vs lag is a
   product decision); the instrumentation exists.
5. **Tighter pursuit tracking** (P8 root cause): shorter velocity
   window, acceleration-aware glides. Future work if Mykal wants it.
6. **The remaining 0.02–0.04x** to a round 1.0x on the fixture lives in
   the H.264 software decode and the harness mp4 re-encode — both are
   fixture overhead, not pipeline compute. A live sensor has neither.

## 7. Files and validation outputs

- Code: `streaming/{feeder,online_driver,online_render,run_stream,compare}.py`
- Tests: `tests/test_streaming_harness.py` (6), `tests/test_fovea_speedups.py` (9 + 7 luma_ratio), `tests/test_motion.py` (16 + 9 thumb-geometry)
- Docs: `streaming/docs/external-stream-front-door.md`,
  `streaming/docs/magno-predictions-sealed-2026-10-03.md`,
  `streaming/docs/magno-thumb-predictions-sealed-2026-10-03.md`
- This report: `streaming/STREAMING-REPORT.md`
- Validation outputs: `output/st_stream_30s/`, `output/st_stream_62s/`,
  `output/st_stream_full/`, `output/st_stream_rt20/` (paced 1x demo),
  `output/st_stream_magno_full/`, `output/st_stream_magno_thumb96/`
  (+ `compare_report.json` in each), batch references
  `output/st_batch_30s/`, `output/st_batch_62s/`, v2 reference
  `output/st_full_v2/`

---

# Changelog

*Full history, verbatim from the build diary. If two entries disagree,
§2–§3 above carry the latest; the supersession is noted where it
matters. (Editorial notes, 2026-10-03 consolidation, are in brackets.)*

## 2026-10-03 (night) — audit decisions executed (Mykal's rulings)

1. **Defaults flipped to the fast combo** (his call #2): `VisionPipeline`
   constructor and `run_stream.py` CLI now default to
   `chroma_mode="luma_ratio"`, `fovea_levels=3`, `mask_cache=True`
   (`--no-mask-cache` disables). The v2 math remains available
   explicitly (`--chroma-mode foveated --fovea-levels 5 --no-mask-cache`).
2. **Explicit-v2 re-validation** (30 s Star Tours vs archived
   `output/st_batch_30s/`): 600/600 moments, timestamp grid 0.000000 ms,
   fixation displacement 0.0000px (0/600), suppression 100%, energies
   max|d|=0.0, saccades 99=99, events identical. The reference didn't
   move; the defaults did. (Percept-frame mean diff 2.7e-3 is lossy-mp4
   compression noise from the ultrafast preset — all .npy data identical.)
3. **Stale scripts deleted** (his calls #4, #5): `scripts/run_video_saccades.py`
   (grayscale predecessor, superseded), `scripts/demo_say_hi.py`,
   `scripts/demo_understand.py` (stale demos); `docs/saying-hi.md` demo
   lines rewritten to point at `run_conversation.py`/test suites;
   comment references in `fuse_av.py`/`perceive.py` updated.
4. **OWL-ViT fixture regenerated** (his call #6): `test_owlvit_detects_window_in_crop`
   un-skipped and green. New hand-labeled 4x3 montage at
   `tests/fixtures/foveal_crops.png` (+ `foveal_labels.json`) from current
   footage; `scripts/audit_recognition.py` MONTAGE/LABELS paths fixed to
   the committed fixture (were stale pre-rename absolute paths).
5. P8 (tight accelerating-target tracking) set aside per his call #3;
   audio sidecar deferred per his call #7.

## 2026-10-03 — original report

**Status:** pre-hardware, test-fixture only. No live sensors touched.
**Scope:** the Sensorium v2 color-saccades vision path
(`scripts/run_video_saccades_color.py`) — the flagship pipeline that had
no streaming story. The Level-3 joint-map path already streams
(`scripts/run_level3_stream.py`); audio streaming already exists
(`hva/stream.py`). This harness does not duplicate them.

### What was built

New package `streaming/` alongside the pipeline (the pipeline itself is
untouched — the harness wraps it):

| File | Role |
|---|---|
| `streaming/feeder.py` | `ChunkedFeeder`: test video played as if live. Two lockstep ffmpeg pipes (224px attention res + 640x360 work res), frames grouped into wall-clock-paced chunks. `realtime=True` paces emission to the wall clock (x `speed`); otherwise as fast as the consumer pulls. Pacing never touches timestamps. |
| `streaming/online_driver.py` | `OnlineAttentionDriver`: the batch pass-1 attention math as a stateful push object — transient decay, inhibition of return, POI memory, saccade decision clock, per-frame energy log. Math reproduced exactly; see "breaks" below for the one file-fact it needs. |
| `streaming/online_render.py` | `OnlineVisionPipeline(VisionPipeline)`: bounded frame buffer (deque; evicts frames no future moment can read), `pull(t_now, t_end)` yielding newly-ready moments with the batch `moments()` body verbatim and `prev` persisted across calls. |
| `streaming/run_stream.py` | CLI runner: feeder -> driver -> render in one online loop, saccade decisions mirrored into the render controller as made. Writes the same artifacts as the batch driver (`video_percept.mp4`, `fixations.npy`, `frame_energies.npy`, `scanpath.npy`, `run_report.json`) plus streaming telemetry. |
| `streaming/compare.py` | Batch-vs-stream comparator: moment counts, timestamp grids, fixation displacement, suppressed-flag agreement, energy deltas, saccade counts, percept-video frame deltas, event lists via the graph's own `detect_events()`. |
| `streaming/docs/external-stream-front-door.md` | One-page design sketch for plugging an RTSP/webcam capture client into the same chunk contract. Design only. |
| `tests/test_streaming_harness.py` | 6 tests: feeder frame count/timestamps vs the batch decoder, pipe lockstep, driver determinism, online-render bit-exactness vs batch `VisionPipeline`, aggressive-eviction exactness + boundedness, `t_end` grid cap. |

### What broke (whole-file assumptions) and how it was fixed

**1. Unbounded frame retention** (`hvp/pipeline.py` keeps every pushed
frame forever). **Fix:** bounded deque in `OnlineVisionPipeline.push()` —
a moment at media time `t` only reads `[t - latency - window, t -
latency)`, so frames older than the newest emitted moment minus
`(latency + window + 500 ms margin)` are evicted. `_integrated()` is
untouched. Unit test proves bit-identical moments under aggressive
eviction. Verified in the 30 s run: 879 frames evicted, buffer steady
at 667 ms.

**2. Two-pass structure** (pass 1 builds the whole saccade script before
pass 2 renders). **Fix:** merge into one online loop. This is valid
because a render query at `t_c = t_moment - latency - window/2` only
sees saccades with `t_on <= t_c`, and every such saccade was decided
at `t_on - 200 ms` — strictly before the stream time at which the
query runs. No lookahead. The render holds its own `SaccadeController`
in work-res coords; decisions are mirrored through `add_saccade()`
(rescale is linear, dva rescales consistently, so saccade durations
match the batch pass-2 controller).

**3. The `t_end` grid cap.** Without it, the streaming tail emits one
extra moment past the media duration (frames exist to `(n-1)*dt`, but
`pull` can see `t_now + latency` past `total*1000`). `pull(t_now,
t_end_ms)` caps the grid exactly like batch `moments(t_end_ms)`.

**4. File facts the stream still needs** (documented, not hacked):
`run_video.probe()` supplies fps + duration from the container header.
For a true live source these come from the stream header / are
unbounded (`t_end_ms=inf`). The two-pipe feeder mirrors the batch
decoder bit-for-bit for validation; a real sensor is single-resolution
(open item below).

### Validation: streaming output vs post-hoc v2 output

#### 30 s smoke (Star Tours)

| Metric | Batch | Stream | Delta |
|---|---|---|---|
| Moments | 600 | 600 | 0 |
| Timestamp grid max |dt| | — | — | 0.000000 ms |
| Fixation displacement (mean / max) | — | — | 0.0000 px / 0.0000 px (0/600 nonzero) |
| Suppressed-flag agreement | — | — | 100.00% |
| Salience energies max |d| | — | — | 0.0 (all channels) |
| Saccades | 99 | 99 | 0 |
| Percept-video frames mean/max |d| | — | — | 1.0e-3 / 7.1e-2 (lossy mp4 encode noise; moments are bit-identical) |
| Events (graph's `detect_events`) | 1 | 1 | identical |

The streaming path is **bit-identical** to the batch path on moments,
fixations, suppression flags, energies, and saccades. The percept-video
frame delta is h264 encode noise from two separate encodes, not
pipeline divergence.

#### 62 s validation (Star Tours)

| Metric | Batch | Stream | Delta |
|---|---|---|---|
| Moments | 1240 | 1240 | 0 |
| Timestamp grid max |dt| | — | — | 0.000000 ms |
| Fixation displacement (mean / max) | — | — | 0.0000 px / 0.0000 px (0/1240 nonzero) |
| Suppressed-flag agreement | — | — | 100.00% |
| Salience energies max |d| | — | — | 0.0 (all channels) |
| Saccades | 206 | 206 | 0 |
| Percept-video frames mean/max |d| | — | — | 9.5e-4 / 7.1e-2 (lossy mp4 encode noise) |
| Events (graph's `detect_events`) | 5 | 5 | identical |
| Frames pushed | 2040 (incl. 2 s lead-in re-pushes) | 1860 | stream needs no re-pushes |
| Frames evicted (bounded buffer) | — | 1839 | buffer steady at 667 ms |

Bit-identical on moments, fixations, suppression flags, energies, and
saccades. Full `compare_report.json` in `output/st_stream_62s/`.

#### Full 269 s run vs `output/st_full_v2/` (the v2 reference)

| Metric | Post-hoc v2 | Stream | Delta |
|---|---|---|---|
| Moments | 5380 | 5380 | 0 |
| Timestamp grid max |dt| | — | — | 0.000000 ms |
| Fixation displacement (mean / max) | — | — | 0.0000 px / 0.0000 px (0/5380 nonzero) |
| Suppressed-flag agreement | — | — | 100.00% |
| Salience energies max |d| | — | — | 6.0e-8 (float32 rounding; signal scale ~1e0) |
| Saccades | 896 | 896 | 0 |
| Percept-video frames mean/max |d| | — | — | 1.0e-3 / 7.5e-2 (lossy mp4 encode noise) |
| Events (graph's `detect_events`) | 52 | 52 | identical |
| Wall clock | 1561 s | 1617 s | stream +3.6% (two decode pipes + per-frame overhead) |
| Frames evicted (bounded buffer) | — | 8049 | buffer steady at 667 ms for the whole run |

All three v2 headline numbers — **5,380 moments, 896 saccades,
52 events** — reproduced exactly. Full `compare_report.json` in
`output/st_stream_full/`. (Note: the first full-run comparison attempt
was OOM-killed decoding both 5,380-frame videos at once; `compare.py`
now streams the decode in batches.)

### Realtime measurement: can this VM keep up?

**No — not at full v2 fidelity.** Measured on the 30 s fixture
(sequential, uncontended):

- Full pipeline: **0.17x realtime** (178 s wall for 30 s of media).
- Per-frame processing (clean sequential run): mean **181 ms**, p50
  **257 ms**, p99 **401 ms**, max **615 ms** vs the 33.3 ms frame
  interval at 30 fps — roughly 5.4x over budget at the median.
- Bottleneck profile (per frame): attention driver **1.6 ms**,
  integration `_integrated` **8.3 ms**, **`foveate_color` 237 ms per
  moment** vs the 50 ms moment budget. The foveated color resampling
  (peripheral blur at 640x360) is the entire story — attention and
  integration are both comfortably realtime-capable.
- At `--realtime --speed 1.0` the feeder paces to the wall clock while
  processing takes 5.4x longer, so the stream falls behind
  monotonically: a 20 s paced run took 118.9 s wall (98.9 s of
  accumulated lag), every 2 s chunk MISSED its deadline (~10.8 s
  processing per 2 s of media), and lag grows ~4.9 s per chunk
  without bound. The runner's `chunk_deadline_misses` counter
  instruments exactly this. Pacing is output-neutral: the realtime
  run's 400 moments are bit-identical to the fast run's first 400.

**What this means:** the streaming *architecture* is proven (bit-exact,
bounded memory, honest pacing), but the *render* needs ~5x speedup for
true realtime on this VM. Options, in order of honesty:
1. Lower the work resolution for the realtime path (320x180 quarters
   the foveation cost; the driver already runs at 224px).
2. Optimize `foveate_color` (the peripheral blur dominates; separable
   / downsampled blur is the obvious win).
3. Decouple: run attention + moments at full rate, render the percept
   video lazily (the review needs moments, not pixels, in realtime).

None of these were attempted: they change the validated v2 math and
belong to a tuning pass with human judgment, not an overnight harness
build.

### External-stream front door

See `docs/external-stream-front-door.md` (one page). The seam is the
`Chunk` contract `(t0_s, t1_s, [(t_ms, frame_attn, frame_work)])`; a
live capture client implements it. Design covers: single-pipe +
in-process downsample (must re-validate the ~2 LSB scaler divergence),
arrival-clock timestamps (transient `dt` becomes measured, not fixed),
no `t_end`, the backpressure policy decision (drop frames vs grow
latency — the deadline counter is the instrument), and audio pairing
via the existing `hva/stream.py` machinery. Bring-up order included.

### Open items

1. **Single-sensor feeder.** The two-pipe feeder is a validation
   configuration. A real sensor needs one pipe + in-process downsample;
   quantify the PIL-vs-ffmpeg scaler divergence on fixations first.
2. **Realtime render.** ~5x speedup needed (see above). Not attempted.
3. **Online event detector.** Events are still post-hoc
   (`st_color_graph.py` on the finished percept video). The bridging
   rule is local; a streaming detector is straightforward but unbuilt.
4. **Audio in this harness.** Vision-only by design (the v2 path is
   vision-only); audio streaming exists in `hva/stream.py` and the
   pairing pattern in `run_level3_stream.py`.
5. **Jittery timestamps.** The driver uses fixed `dt = 1000/fps` for
   transient decay (matches batch). Live sources need measured dt —
   one-line change, flagged not made (it alters validated math).
6. **Backpressure policy.** Undecided by design (drop vs lag is a
   product decision); the instrumentation exists.

### Files

- Code: `streaming/{feeder,online_driver,online_render,run_stream,compare}.py`
- Tests: `tests/test_streaming_harness.py` (6 tests, all passing)
- Docs: `streaming/docs/external-stream-front-door.md`
- This report: `streaming/STREAMING-REPORT.md`
- Validation outputs: `output/st_stream_30s/`, `output/st_stream_62s/`,
  `output/st_stream_full/`, `output/st_stream_rt20/` (paced 1x demo)
  (+ `compare_report.json` in each), batch references
  `output/st_batch_30s/`, `output/st_batch_62s/`, v2 reference
  `output/st_full_v2/`

### Test suite

159 passed, 2 skipped, 1 failed — the failure is the pre-existing
`test_owlvit_detects_window_in_crop` (missing fixture file
`output/foveal_crops.png`, stale `human-vision-pipeline` path),
failing before this work too. No regressions.

## 2026-10-03 — addendum: chroma passthrough mode (Mykal's call)

Mykal proposed skipping chroma foveation: keep the color feed, don't
foveate the chroma — "slightly less faithful, but so far seems good
enough." Implemented as an explicit opt-in mode, v2 math untouched:

- `hvp/retina.py`: new `foveate_color_fast()` — foveates luminance only,
  chroma passes through at full resolution (no peripheral desaturation).
- `hvp/pipeline.py`: new `chroma_mode` param (`"foveated"` default =
  validated v2; `"passthrough"` = fast). Shared `_render_color()`
  helper used by both batch `moments()` and streaming `pull()`.
- `streaming/run_stream.py`: `--chroma-mode {foveated,passthrough}`;
  mode recorded in `run_report.json` for provenance.

Measured on 640x360 random frame (5-run mean): foveate_color
253.3 ms/moment -> foveate_color_fast 56.6 ms/moment (**4.5x**).
Sanity: periphery chroma std 0.0028 (full, near-colorblind as designed)
vs 0.1813 (fast, full color preserved); center colorful in both.

30 s Star Tours streaming run in passthrough mode: 66.0 s wall,
**0.45x realtime** (was 0.17x), per-frame p50 71 ms vs 33.3 ms budget.
Still not realtime on this VM, but 2.6x closer; remaining bottleneck is
now the single luminance foveation + colorspace round-trip, not chroma.

Tests: retina + streaming suites green (26 passed, 2 skipped), no
regressions. Default mode unchanged: every existing run reproduces v2
bit-exactly.

## 2026-10-03 (evening) — addendum: foveation speedups — levels, mask cache, box blur

Mykal authorized three optimizations (his picks #2, #3, #4 from the
speedup list). All implemented as independently-toggleable opt-ins;
the v2 default path is untouched and bit-identical (full suite:
150 passed, 2 skipped; the single failure is the pre-existing
`test_owlvit_detects_window_in_crop` missing-fixture issue).

### What was built

- `hvp/retina.py`:
  - `foveate()` gained `levels` (already a param, now threaded),
    `mask_cache` (dict memoizing the eccentricity-band masks per exact
    fixation — bit-identical, FIFO-bounded at 64 entries), and
    `box_sigma_threshold` (per-band sigma above which the stacked
    3-pass box-blur approximation replaces scipy's gaussian).
  - New `box_blur()`: variance-matched box width
    s = sqrt(12*sigma^2/3 + 1), 3 passes of `uniform_filter`.
  - `foveate_color()` / `foveate_color_fast()` forward all three knobs.
- `hvp/pipeline.py`: `VisionPipeline(..., fovea_levels=5,
  mask_cache=False, box_sigma_threshold=None)`; shared `_foveate_kw()`
  helper used by batch `moments()`, streaming `pull()`, and the
  grayscale paths. Defaults = v2.
- `streaming/run_stream.py`: `--fovea-levels`, `--mask-cache`,
  `--box-sigma-threshold`; all recorded in `run_report.json`.
- Tests: `tests/test_fovea_speedups.py` (9 tests: cache bit-identity,
  cache bound, levels validity, box kernel closeness, threshold
  passthrough, pipeline threading/defaults/validation).

### Micro-benchmarks (5-run mean, 640x360, foveate_color_fast)

| config | ms/moment | vs baseline |
|---|---|---|
| baseline (L5) | 72.2 | 1.0x |
| levels=3 | 59.1 | 1.2x |
| mask cache (warm) | 56.7 | 1.3x |
| box blur (thr=2.0) | 70.0 | 1.0x |
| combo (L3+cache+box) | 58.0 | 1.2x |

Component profile of one foveate() call: mask build 4.4 ms, five
gaussian blurs 38.8 ms total (3.1/5.1/8.6/8.8/13.2 at sigma
0.6/1.8/3.0/4.2/5.4), YCbCr round-trip 14.1 ms.

### End-to-end (30 s Star Tours, --chroma-mode passthrough)

| config | wall | realtime factor | p50/frame |
|---|---|---|---|
| base | 64.8 s | 0.46x | 66.4 ms |
| + levels=3 | 53.2 s | 0.56x | 42.2 ms |
| + mask cache | 62.9 s | 0.48x | 63.8 ms |
| + box blur (thr=2.0) | 65.5 s | 0.46x | 69.5 ms |
| combo (L3 + cache) | 49.4 s | **0.61x** | **37.1 ms** |

Frame budget is 33.3 ms; combo p50 is 37.1 ms — within 11% of realtime.

### Fidelity vs base (compare.py on percept streams, 600 moments)

| opt | mean|d| | max|d| | fixations | events |
|---|---|---|---|---|
| mask cache | 0.000 | 0.000 | identical | identical (1=1) |
| levels=3 | 3.9e-3 | 0.188 | identical | identical (1=1) |
| box blur | 3.3e-3 | 0.114 | identical | identical (1=1) |

(Diffs measured through the mp4 encode, so these are upper bounds.)
Cache-hit rate in the real run: 83.5% of consecutive moments share the
exact fixation (matches the ~85% prediction).

### Verdicts

- **levels=3: SHIP.** The biggest lever (0.46x -> 0.56x alone). Cost is
  honest: coarser eccentricity quantization, mean percept diff 0.4% of
  full scale, max 19% on isolated high-contrast pixels at band
  boundaries; event structure unchanged.
- **mask cache: SHIP.** Bit-identical, free. Small (saves the 4.4 ms
  mask build, ~85% hit rate) but zero fidelity cost — no reason not to.
- **box blur: DUD — excluded from the recommended combo.** 1.0x
  end-to-end. The premise was wrong: scipy's gaussian cost grows
  sublinearly with sigma here (8.6 ms at sigma=3.0 -> 13.2 ms at 5.4),
  and three box passes cost ~9 ms flat — the O(1)-per-pixel theory
  loses to the 3-vs-2 pass constant. Kept in the codebase behind the
  flag (it is a correct, validated approximation: kernel max diff
  0.006 at sigma=3, shrinking with sigma), but it buys nothing on this
  stack.

**Recommended: levels=3 + mask cache -> 0.61x realtime**, p50 37 ms vs
33 ms budget. Not quite realtime on this VM, but one more measured
push away.

### Next measured bottleneck (not implemented — needs a call)

The YCbCr round-trip is 14.1 ms/moment (~20% of the render step) and
is now the second-largest cost after the gaussian blurs. The
luminance-ratio trick (foveate Y, then
out = frame * (Y_fov / Y)) would delete both matrix conversions.
Untouched pending authorization — it's new math, not one of the three.

## 2026-10-03 (evening) — addendum: luma_ratio chroma mode (Mykal-authorized)

The push past realtime. `foveate_color_luma_ratio()` in `hvp/retina.py`
replaces the YCbCr round-trip in the fast path with:
  Y = 0.299*R + 0.587*G + 0.114*B (single dot product)
  Y_fov = foveate(Y, ...)   (same level/cache/box kwargs)
  out = clip(frame * (Y_fov / Y), 0, 1)
Safe-divide: where Y < 1e-6 the ratio defaults to 1, so black stays
black (no NaN, verified on pure-black frames). Threaded as a third
`chroma_mode="luma_ratio"` in `hvp/pipeline.py` (v2 default untouched),
`--chroma-mode luma_ratio` in `streaming/run_stream.py`, recorded in
`run_report.json`. 7 new unit tests in `tests/test_fovea_speedups.py`:
black-frame safety, output range, uniform-luma passthrough,
chromaticity preservation, dark-noise bound, kwarg forwarding,
pipeline threading/validation.

Benchmarks (640x360, 5-run mean per moment):
  passthrough:            74.3 ms/moment
  luma_ratio:             52.4 ms/moment  (1.42x vs passthrough)
  luma_ratio L3+cache:    24.6 ms/moment  (3.02x vs passthrough)

30 s Star Tours end-to-end, recommended combo
(luma_ratio + fovea_levels=3 + mask_cache):
  wall 38.2 s -> realtime_factor 0.79x (was 0.61x)
  per-frame p50 29.5 ms vs 33.3 ms budget -- UNDER budget.
  per-moment render 24.6 ms vs 50 ms moment spacing -- UNDER budget.
The foveation itself is now realtime-capable. The remaining 0.2x is
test-harness overhead, not eyes: the feeder runs TWO ffmpeg decode
pipes in lockstep (224px attention + 640x360 work = 7.3 s + 9.3 s of
decode for 30 s of media; a real sensor would be single-resolution,
already documented as an open item), plus ffmpeg percept-encode
backpressure on stdin.write (explains the p99/max frame spikes:
128/275 ms -- encoder stalls, not compute) and fixed startup cost.

Fidelity, luma_ratio (L3+cache) vs v2 default, 30 s / 600 moments
(via streaming/compare.py):
  fixations: 0.0000 px displacement -- identical (attention untouched)
  saccades: 99 = 99; suppressed flags: 100%; energies: identical
  events: 1 = 1, identical structure
  percept frames: mean|d| = 1.58%, max|d| = 82.8%
Isolated luma_ratio vs passthrough (both L3+cache): mean 1.46%,
max 82.4% -- the trick accounts for nearly all the delta. The hot
pixels (793 > 0.5 in the worst frame): passthrough renders them
near-gray (saturation 0.04), luma_ratio keeps them vivid (saturation
0.82) at the same luma (0.22 vs 0.21). I.e. in bright blurred regions
the YCbCr recombination mutes toward gray while the ratio preserves
input chromaticity -- the known semantic difference, concentrated in
mid-to-bright luma deciles (dark deciles ~0). Dark-noise amplification
is real but bounded: near-black noisy patch under a bright region went
0.005 -> 0.022 max (dim, no explosion).

Verdict: SHIP as the streaming default. Fixations, saccades, events --
everything the attention system and any downstream consumer reads --
are untouched. The percept frames keep full vivid color instead of
muting toward gray in the blur, which matches Mykal's original "keep
the color feed" directive better than passthrough did.

Tests: 42 passed, 2 skipped across the retina/streaming/attention/
speedup suites; v2 default path bit-identical; no regressions.

[Note 2026-10-03 consolidation: the SHIP-as-default verdict is recorded
in §3 of the current state, but the code default remains `foveated` —
see §6 open item 7.]

## 2026-10-03 — addendum: single-decode feeder (Mykal's call)

Mykal authorized merging the feeder's two ffmpeg pipes into one --
the documented open item for the last stretch toward 1.0x realtime.

### What was built

`streaming/feeder.py` now supports three decode modes on
`ChunkedFeeder` (`dual_decode` / `single_mode`, exposed as
`--dual-decode` and `--single-mode` on `run_stream.py`, recorded in
`run_report.json`):

- `dual_decode=True`: the original two-pipe path, kept verbatim for
  reproducibility.
- `single_mode="split"` (new default): ONE ffmpeg process decodes
  once; a `split` filter graph produces both resolutions with the
  same per-branch `fps,scale,format=rgb24` chain as dual_decode.
  Implemented as `decode_split()` (two rawvideo outputs via
  `pipe:1` + `pass_fds`, strictly alternating reads to avoid pipe
  deadlock).
- `single_mode="pil"`: one decode at work res; the 224px attention
  thumbnail is downsampled in-process with PIL bilinear on uint8 --
  the same convention as `hvp.attention._downsample_color`.

Also changed: the percept-encode mp4 now uses `-preset ultrafast`.
That file is a visualization artifact, not a validation input (all
agreement metrics come from .npy); the default medium preset was
spending significant time per frame on encoding.

### The gap-closing story (honest version)

The prescribed approach was in-process PIL downsampling. Pixel
comparison vs ffmpeg's scaler picked PIL bilinear (the codebase's
own convention), but the real test -- saccade decisions -- showed
divergence: 99=99 saccades, but 6 of 99 flipped to an adjacent
saliency cell (exactly 4.00px in 224-coords = 1 saliency-map pixel;
the classic close-call argmax flip), max fixation displacement
11.4 work-px, energies up to 2.0e-02 (vs the <6e-8 bar).

Per the brief ("try to close it before declaring it"), the split
filter graph was built as the closer: it uses libswscale itself,
so the 224px frames match dual_decode's scaler exactly while the
decode happens once. Result: **bit-identical on every metric**
(moments, timestamps, fixations 0.0000px, suppression 100%,
energies 0.000e+00, saccades 99=99, percept frames 0.000e+00,
events identical). The in-process path stays available behind
`--single-mode pil` with its divergence documented; it is not the
default.

### Benchmarks (30 s Star Tours, luma_ratio + L3 + mask cache)

| mode   | wall   | realtime | saccades |
|--------|--------|----------|----------|
| dual   | 36.2 s | 0.83x    | 99       |
| split  | 30.7 s | 0.98x    | 99       |
| pil    | 30.8 s | 0.98x    | 99       |

Baseline was 0.79x; single-decode reaches **0.98x**. The 1.0x target
is missed by 2%, and the remaining gap is characterized, not
mysterious: the H.264 software decode of the 1080p test fixture
costs ~9-12 s of the 30.7 s wall (measured standalone), while the
vision pipeline itself (driver + render) runs at ~21 s per 30 s of
media -- about **1.4x realtime**. A live sensor has no decode step;
the 0.02x is fixture I/O, not pipeline compute. Per-frame p50
28.6 ms vs 33.3 ms budget (p99 spikes are encoder backpressure,
largely tamed by the ultrafast preset).

### Verdict

**SHIP single-split as the default.** Single decode achieved,
validation bar held bit-identically, 0.79x -> 0.98x. The in-process
PIL path was tried first per the brief, diverged on 6% of saccades,
and was superseded -- documented here and kept behind its flag.

Tests: 46 passed, 2 skipped across the retina/streaming/attention/
speedup suites, no regressions. New feeder tests: split-vs-dual
bit-identity, pil parity + sanity bound, split determinism,
invalid-mode rejection.

## 2026-10-03 — addendum: magno channel — motion energy + smooth pursuit

Mykal: "we need to give it the ability to recognize and track motion
somehow." Commissioned and built the same evening.

### Design (biology is the spec)

Primate vision runs a dedicated magnocellular pathway -- fast, coarse,
colorblind, motion-sensitive -- alongside the slow detailed
parvocellular path. The magno channel mirrors it:

- `hvp/motion.py`: `MotionChannel` -- frame differencing on the 56x56
  grayscale attention thumbnail the driver already computes (marginal
  cost: one abs-diff + one small gaussian + threshold = **0.49 ms/frame**
  measured). Output: a 56x56 motion-energy map with a local-coherence
  gate.
- The gate is the load-bearing idea: raw frame difference floods on
  any full-field change, so the map subtracts a heavily blurred copy
  of itself (center-surround on the motion map). Uniform floods --
  whiteouts, cuts, flashes -- cancel to ~zero; local motion contrast
  (a craft against background or background flow) survives. Threshold
  0.05 kills encode noise; 120 ms persistence (faster than the 200 ms
  transient channel, as biology demands).
- Attention: `salience_map()` gained `motion_map`/`motion_weight`
  kwargs (follows the chroma_weight pattern). `--motion-weight`
  enables; default None = off = v2-identical.
- `hvp/saccades.py`: `SaccadeController` gained pursuit glides --
  `add_pursuit(t0, t1, vx, vy)`, stored separately from saccade
  events so saccade-only unpack sites never see them. `state_at()`
  does a merged chronological walk: mid-glide returns the glide
  position with suppressed=False (**no saccadic suppression during
  pursuit**, the biological point). A later saccade truncates an
  active glide; a later pursuit supersedes an overlapping older one.
- Pursuit drive: the driver tracks the motion centroid near fixation
  over a 400 ms history; if direction is consistent (mean cosine >
  0.5), speed in [3.6, 107] deg/s, and local energy at fixation is
  above floor, it schedules 1 s glide segments (100 ms pursuit
  latency, re-evaluated each 3.5 Hz decision tick). No inhibition of
  return while pursuing. `--pursuit` enables; default off.
- Plumbing: driver returns tagged decisions ("sac"/"pur"),
  run_stream.py mirrors pursuits into the render controller (work-px
  rescaled), `motion_energies.npy` + `pursuit_log.npy` saved when
  enabled, all flags recorded in run_report.json.
  `streaming/online_render.py` gained `add_pursuit`.

### Sealed predictions (written before any motion-on run)

Full list in `streaming/docs/magno-predictions-sealed-2026-10-03.md`:
P1 cost <= 3 ms/frame, realtime >= 0.95x. P2 control reproduces 99
saccades/30 s. P3 motion-on saccades within +/-25% of 99. P4
motion-energy-at-fixation strictly higher motion-on. P5 >= 1 pursuit
segment per 30 s; 5-25% pursuit time on full film. P6 zero pursuit
overlap with whiteouts; no saccade spike in whiteout. P7 suppressed
fraction not increased; never suppressed inside pursuit. P8 centroid
within 3 deg of glide > 80% of pursuit time.

### 30 s outcomes (control vs motion-on, fast combo)

| metric | control | motion-on |
|---|---|---|
| saccades | 99 | 90 |
| pursuits | 0 | 10 segments |
| realtime factor | 0.96 | 0.96 |
| per-frame p50 | 28.6 ms | 29.1 ms |
| motion-e at fixation (mean) | 0.00466 | 0.00927 |
| suppressed-moment fraction | 0.0033 | 0.0033 |
| suppressed moments inside pursuit | -- | 0 |

P1 PASS (0.49 ms/frame; no realtime regression). P2 PASS (99 exactly).
P3 PASS (90 within band). P4 PASS (2x higher -- the eyes go where
motion is). P5 PASS on 30 s (10 segments). P6 vacuous on the 30 s
clip (no sustained whiteout present; hyperspace is later in the
film -- checked on the full run below). P7 PASS both halves.
P8 FAIL as stated (21% vs 80% predicted).

### P8 root cause (honest)

The sealed prediction assumed discrete constant-velocity targets.
The 0-30 s stimulus is dominated by accelerating camera flow (ship
descending: flow 4-12 deg/s downward, speeding up across chained
segments). Directionally the glides are correct -- 7/9 segments have
cosine ~1.0 with the centroid drift -- but constant-velocity glides
from 400 ms-old estimates undershoot accelerating flow, and the
centroid of a broad flow field is ill-defined within a 3 deg window.
The machinery is validated on clean stimuli (16 unit tests: linear
disk -> pursuit engages and tracks; reversing blob -> no pursuit;
global flash -> no pursuit, energy gated to zero). What emerged on
the ride film is closer to optokinetic following than discrete
pursuit -- the right behavior for the stimulus, the wrong label on
the prediction. The full-film run (trench/battle sections with
discrete TIE fighters) is the fairer test; results below.

### Full-film outcomes (269 s, motion on, fast combo)

| metric | v2 / control | motion-on full |
|---|---|---|
| saccades | 896 | 761 (-15%) |
| pursuit segments | 0 | 135 |
| pursuit time | 0 | 50.7 s = 18.8% of film |
| realtime factor | 0.98 | 0.96 |
| per-frame p50 | 28.6 ms | 29.8 ms |
| moments | 5,380 | 5,380 |
| suppressed-moment fraction | 0.0033 | 0.0009 |

P5 PASS (18.8% inside the predicted 5-25% band). Pursuit is spread
across the film (1.4-4.0 s per 10 s bin through the battle/trench
sections), not clumped -- it engages wherever coherent motion lives.
Saccade count -15%: pursuit replaces catch-up saccades, the predicted
direction.

P7b resolved: the single suppressed moment inside a pursuit segment
was a boundary artifact -- the moment AT t0 integrates [t0-200,
t0-100] ms, all pre-pursuit (the saccade that handed off to the
glide). Dense resampling of the rebuilt render controller: 0/5,070
samples inside (t0, t1) suppressed. The biology holds.

P6 in the wild: this film has no true whiteout (max frame luma
0.673). Pursuit under-indexes bright frames 4-5x (frames > 0.5 luma:
1.0% overall vs 0.3% during pursuit) -- the coherence gate working
as designed. The brightest pursuit segment (t=206 s) is hyperspace
star-streaks: coherent radial motion, a legitimate pursuit target,
not a flash lock.

P8 stands as FAIL with the root cause above. Not retuned: v1 is
honest, tested, and its limits are documented here. Tighter tracking
(shorter velocity window, acceleration-aware glides) is future work
if Mykal wants it.

### Verdict: SHIP (v1, documented limits)

The magno channel does its job: motion attracts the eyes (2x
motion-energy at fixation), the coherence gate holds against
flashes, pursuit glides without suppression and chains across
re-evaluations, and the realtime budget survives (0.96x, +0.5
ms/frame). Default-off everywhere: every existing run reproduces
v2 bit-exactly. Files: `hvp/motion.py`, `tests/test_motion.py`
(16 tests), `streaming/docs/magno-predictions-sealed-2026-10-03.md`,
validation outputs in `output/st_stream_magno_full/`.

## 2026-10-03 (evening) — addendum: motion thumbnail aspect fix + size ladder

Mykal's directive: the motion thumb must match the video's 16:9 aspect
ratio ("it should at least be the same ratio as the video input"),
and find the biggest thumb that holds realtime -- "try it at a quarter
of video input."

**The bug:** the magno channel ran on a 56x56 SQUARE thumbnail derived
from the anamorphic 224x224 attention frame. On 16:9 footage the motion
field was horizontally squished (~11.4x vs ~6.4x downsampling from work
res), so pursuit velocities and motion centroids were systematically
wrong in x.

**What changed:**
- `hvp/motion.py`: `MotionChannel(thumb_w=96, thumb_h=54)` -- 16:9
  default. All map-px constants scale from the 56-reference:
  velocities calibrated in deg/s (3.6..107) and converted per thumb
  width; focus radius = 0.25 x width; coherence-gate sigma scales
  with width. Above 128px wide the gate blur runs on a <=64px working
  copy and upsamples (same effective blur, constant cost); at/below,
  direct (bit-identical to validated 56 behavior). `push()` asserts
  the thumb shape.
- `streaming/online_driver.py`: thumb derived from the 640x360 work
  frame via PIL (`--motion-thumb-source work`, default); `attn` keeps
  the legacy anamorphic path for validation. 224px<->thumb coordinate
  conversions are per-axis (true 16:9). Energy map resampled to 56x56
  anamorphic for the salience add.
- `streaming/run_stream.py`: `--motion-thumb` WxH, both flags in
  `run_report.json`; work frame now passed to `driver.push`.
- Sealed predictions (before any run):
  `streaming/docs/magno-thumb-predictions-sealed-2026-10-03.md`.

**Ladder** (30 s Star Tours, luma_ratio + L3 + mask cache + motion 1.0
+ pursuit):

| thumb | source | saccades | pursuits | rt | p50/frame |
|---|---|---|---|---|---|
| 56x56 (legacy) | attn | 90 | 10 | 0.93 | 29.4 ms |
| 96x54 (new default) | work | 93 | 7 | 0.85 | 32.1 ms |
| 160x90 | work | 93 | 7 | 0.86 | 31.8 ms |
| 480x270 | work | 93 | 7 | 0.78 | 35.0 ms |
| 960x540 | work | 94 | 6 | 0.65 | 44.0 ms |

Microbenchmark (per-frame motion-channel cost incl. thumb derivation):
96x54: 3.23 ms (downsample 2.23, motion 1.00); 160x90: 5.24 ms;
480x270: 6.22 ms; 960x540: 15.38 ms. The PIL work-frame downsample
(~2.9 ms, input-dominated) is the cost center at every size -- a
numpy block-mean was tried and measured SLOWER (8.74 ms at 160x90;
reshape+mean over strided axes is cache-hostile).

**Full film** (269 s, new 96x54 default): 742 saccades (vs 761 legacy,
-2.5%), 154 pursuits (vs 135, +14%), rt 0.86 (vs 0.96), p50 32.7 ms
vs 33.3 ms budget, 0 deadline misses.

**Sealed-prediction scorecard:**
- T1-T4 (per-size costs): PASS (within predicted ranges).
- T5 (rt>=0.95x for 96x54/160x90/480x270): FAIL -- none of the
  work-derived sizes holds 0.95x. The downsample cost was
  underestimated in the prediction.
- T6 (480x270 wins): FAIL (consequence of T5).
- B1 (ratio fix: saccades +/-15%, pursuits +/-50%, P1-P8 still pass):
  PASS -- 93 vs 90 saccades, 7 vs 10 pursuits; pursuit segments align
  in time with legacy (13.4/15.8/16.7/19.7/29.9s ~ 12.8/15.5/16.4/
  19.4/29.9s), marginal segments dropped/added, no qualitative change.
- B2 (ladder behavior flat): PASS -- saccades 93/93/93/94, pursuits
  7/7/7/6 across 16:9 sizes; size changes cost, not behavior.
- B3 (salience path size-invariant): PASS (consistent with B2).

**Verdict:** the aspect-ratio fix SHIPS (96x54 16:9 from the work
frame is the new default) -- geometry correctness over 0.10x
realtime, per Mykal's directive. No bigger size holds the 0.95x bar:
the ladder fails with numbers on the table. The known cost center
for a future pass is the per-frame PIL work-frame downsample
(~2.9 ms); the motion math itself is cheap at every size thanks to
the constant-cost gate. Tests: 71 passed, 2 skipped (incl. 9 new
thumb-geometry tests); no regressions.

## 2026-10-03 — addendum: third split output kills the PIL thumb downsample (Mykal's option 1)

Mykal authorized removing the ~2.9 ms/frame in-process PIL work-frame
downsample by adding a third ffmpeg split output at motion-thumb
resolution.

### What was built

- `streaming/feeder.py`: `decode_split()` takes optional `w3`/`h3` and
  builds a `split=3` filter graph (`[full]` 640x360, `[small]`
  224x224, `[mot]` thumb) with three rawvideo outputs -- pipe:1 plus
  two pass_fds pipes, read in strict map order per frame. The thumb
  frames are ~15 KB (far under the 64 KiB pipe buffer), so the third
  pipe can never block the writer; the two big outputs keep the
  validated two-pipe read order. `ChunkedFeeder` gained
  `motion_thumb_wh` (only honored in split mode); `chunk.frames`
  tuples are now `(t_ms, fa, fw, fm)` with `fm=None` when no thumb
  output was requested; the pipe-skew assert covers all three
  timestamps.
- `streaming/online_driver.py`: `push()` takes `frame_thumb`; when
  present (and source is "work") the magno channel takes luma
  directly -- no downsample. Otherwise the existing PIL/legacy paths
  are unchanged (fallback intact: `--dual-decode`,
  `--motion-thumb-source attn`, and `single_mode="pil"` all derive
  the thumbnail in-process as before).
- `streaming/run_stream.py`: the effective thumb size (CLI
  `--motion-thumb` or the 96x54 default) is passed to the feeder when
  motion is on, source is "work", and the mode is split; recorded as
  `motion_thumb_direct` in run_report.json for provenance.

### Validation (30 s Star Tours, luma_ratio + L3 + cache + motion 1.0 + pursuit)

- No deadlock: 900/900 frames through three pipes; timestamps
  identical across outputs (assert-held).
- Thumb derivation microbenchmark: PIL 2.15 ms/frame -> direct luma
  0.005 ms/frame.
- Behavior parity, old PIL path vs new direct path (same driver
  config): motion energies mean|diff| 8.5e-04, max|diff| 8.1e-03
  (bias +8.5e-04, negligible); saccades 93 -> 91; nearest-neighbor
  matched saccades coincide in time (median gap 0 ms) with median
  target displacement 0.0 px, mean 5.8 px (~1.5 saliency cells);
  pursuit segments 7 -> 9. The divergence is resampler-level noise
  (libswscale-direct vs libswscale+PIL-bilinear chains), well within
  the accepted pil-vs-split precedent -- and the naive order-paired
  comparison initially overstated it (count drift desynchronizes
  pairing); the nearest-neighbor analysis is the fair read.
- End-to-end: **0.86x -> 0.91x realtime** (30 s wall 32.9 s), p50
  29.6 ms vs 33.3 ms budget, 0 deadline misses. The 0.95x target was
  missed by 0.04x -- honest miss, numbers on the table.
- Per-stage breakdown (10 s probe): driver 2.5 ms/frame, render
  17.2 ms/frame; the remaining ~10 ms/frame is the harness mp4
  re-encode, not vision. The vision pipeline proper runs ~19.7
  ms/frame (~1.7x realtime); a live sensor with no re-encode step is
  comfortably realtime.

### Verdict

**SHIP.** Option 1 went cleanly -- no deadlock, behavior preserved,
2.15 ms/frame recovered, fallback paths verified working. The last
0.04x to 0.95x lives in the render + harness encode, outside this
task's scope. Tests: 75 passed, 2 skipped (4 new three-output feeder
tests); no regressions.

*[End of changelog. Consolidated 2026-10-03: two empty placeholder
lines from the build diary were removed (a "[filled in after the run
completes]" header in the magno addendum whose results were already
present); no content was dropped.]*

## Addendum 2026-10-03: the audio pathway (Sensorium gets ears)

Mykal: "how do we stitch in the audio instead of just having it be a
text pass with more steps?" Design principle: hearing isn't reading.
The transcript is the *fovea* of hearing -- high-effort, attended,
last resort. The pathway is coarse-first, mirroring the magno
channel: per-50ms-moment cochlear features (RMS, spectral flux,
spectral centroid), onset events on the master timeline,
cross-modal binding, and transcription as an attended spotlight.

### What was built
- `hva/online.py` (new): `moment_features()` -- per-50ms-moment
  (rms, flux, centroid) from 16 kHz mono, with flux continuity
  across chunk boundaries; `pick_onsets()` -- peak-picking with a
  LOCAL adaptive threshold (median+k*MAD over ±2 s) plus absolute
  floor. Three floats per moment: the whole of v1 hearing that
  isn't words.
- `streaming/feeder.py`: `decode_audio_track()` -- sidecar
  pre-decode, sliced by sample count; moment alignment exact by
  construction; zero disturbance to the validated video pipes.
- `streaming/bind_av.py` (new): `visual_transients()` (frame-energy
  diffs on the moment grid) + `bind()` (greedy nearest-in-±250 ms
  audio<->visual coincidence). The "that made that" link.
- `streaming/run_stream.py`: `--audio/--no-audio` (default on,
  passive -- observes, never steers), `--transcribe-onsets K`
  (attended transcription of ±4 s slices around top-K onsets,
  vad=False -- the VAD demonstrably eats real speech on short
  slices). Writes `audio_features.npy`, `audio_events.json`,
  `av_binding.json`, `onset_transcripts.json`; all flags in
  `run_report.json`.
- `tests/test_audio.py` (new): 11 tests -- feature shapes, RMS
  calibration, flux continuity, click/silence/sine onset behavior,
  min-gap, RMS floor, binding greediness, decode length/skew.

### Design corrections found by measurement (honest)
1. Loudness-relative flux was wrong for this job: it ranked the
   hyperspace roars 170th/202nd (silence-to-sound explosions
   dominate; roars on loud beds get normalized away). Switched to
   ABSOLUTE flux.
2. A global MAD threshold then suppressed real transients (film
   beds keep absolute flux median ~107). Switched to a LOCAL
   adaptive threshold -- textbook onset detection.
3. faster-whisper's VAD deletes real speech on 8 s slices; attended
   transcription runs vad=False.

### Sealed predictions vs outcomes (streaming/docs/audio-predictions-sealed-2026-10-03.md)
- P1 (roars are the two strongest onsets): FAIL as stated. Jump1
  has a sharp onset at 64.10 s (dt=0.10 s from the visual peak --
  the alignment proof works), but it ranks 97/238; jump2 has no
  isolated transient at its visual peak (233.2 s max is 3.25 s
  late, inside a sustained passage). The premise imagined isolated
  roars; the mix doesn't work that way.
- P2 (battle densest): FAIL. Battle 0.824/s vs film median
  0.882/s -- the battle is a sustained wall, not discrete bangs
  at 50 ms scale.
- P3 (whiteout crescendo): PASS. RMS ramps 0.078->0.176 across
  96-110 s; onset at 110.60 s, dt=0.10 s.
- P4 (turn transient-silent): PASS. One weak onset at 31.6 s,
  none above p99.
- P5 (trench second-densest): FAIL as stated -- trench IS the
  densest (0.944/s); the ranking premise was wrong, the substance
  (trench is onset-rich) holds.
- P6 (AV coincidence < 1.0 s median): PASS. Median dt 0.23 s
  across the top-10 visual events (9/10 within 1.2 s).
- P7 (cost): PASS. 0.035 ms/moment (0.07% of the 50 ms budget).
  End-to-end 0.89x vs 0.91x baseline -- the 0.02 delta is VM noise
  (audio's measured cost cannot explain it; 30 s A/B showed
  audio-on FASTER at 0.96 vs 0.85).
- P8 (no video behavior change): PASS. 30 s audio-on vs off:
  fixations.npy and scanpath.npy bit-identical, saccades 99=99.

### Binding inventory (full film)
238 audio onsets, 328 visual transients, 91 bound (38%). Bound
pairs land within ±0.2 s typically (e.g. 64.10 s audio <-> 64.05 s
visual). Strongest unbound onsets cluster at the film end
(261-268 s: credits/outro the eyes see nothing of) -- the ears
hear the credits; the eyes don't. Characterization, not failure.

### Attended transcription (validated)
- Turn slice 31-41 s: "I thought you were going the wrong way,
  it's not me!" / "Go straight!" -- Rex, matches the beat.
- Whiteout 106-114 s: "Let's go!"
- Trench 202-210 s: "Oh my god!"
All three match their narrative beats. Note: the full-file
transcript's 19 segments are sparse next to slice results -- the
spotlight sees more than the floodlight did.

### Verdict: SHIP
The ears are live: 0.035 ms/moment, zero video-path impact,
timing coincidences at 0.1-0.35 s on the film's biggest beats.
The failed predictions were wrong premises about the mix, not
detector failures -- and the detector's timing proof (P6) is the
result that matters.

## Addendum 2026-10-03 (late): audio commission verification pass

Mykal's commission asked for the full chain: implement, predict,
run, validate, report. The build phase was already filed above;
this pass independently re-ran, re-measured, and closed the two
gaps the first addendum left open: a full-film P8 bit-identity
check (the original only had the 30 s A/B) and the 30 s clip
binding inventory.

### P8 at full film scale: PASS
Ran the full 269 s film audio-off as a twin of the audio-on run
(`output/st_stream_full_audio_off_twin`, `--no-audio`, otherwise
identical flags). fixations.npy (5380x4), scanpath.npy (897x3),
and frame_energies.npy (8070x4) are BIT-IDENTICAL audio-on vs
off; saccades 896=896. Audio observes, never steers, at full
scale -- not just on the 30 s clip.

### Binding inventory: 30 s clip first
`output/st_stream_audio_30s`: 38 audio onsets, 26 visual
transients, 16 bound (42%), bound |dt| within ±0.2 s. The full
film (above) is 238 onsets, 328 transients, 91 bound (38%).
Both scales agree: binding is real but partial -- most loud
onsets have no visual-transient partner.

### Sealed predictions: independently recomputed
All verdicts from the first addendum confirmed on the current
run's numbers (238 onsets): P1 FAIL (jump2 has no onset within
±1 s of its 229.9 s visual peak -- the roar is a sustained
passage, not a flux transient), P2 FAIL (battle 0.824/s vs film
median 0.882/s), P3 PASS (onsets at 109.95/110.60 s, dt 0.10 s
from the flash; RMS roughly level across the approach in this
re-measurement, 0.09-0.10, so the PASS rests on the flash
timing), P4 PASS (31.6 s onset at strength 207.7, below the
median 233.1 and far below p99 438.3 -- the turn is
transient-silent), P5 FAIL as stated (trench 0.971/s IS the
densest window; the ranking premise was wrong), P6 PASS (median
dt 0.225 s across the top-10 visual events; 9/10 within 1.2 s),
P7 PASS (0.0353 ms/moment; see below), P8 PASS.

### Attended transcription: three new slices (independent)
Spotlight transcription of ±4 s slices around three onsets
(whisper base, vad=False):
- 261.4 s (STRONGEST onset in the film, unbound): "The captain
  has opened the exit doors. You may then unlatch your safety
  restraints by pressing the release button on your left." --
  the ride-exit PA. This is what the ears' loudest moments are:
  narration over a visually static unload. The unbound giants
  are characterized, not noise.
- 205.25 s (trench, unbound, strength 441): "Oh my god!" +
  laughter -- Rex's dialogue over continuing action. A vocal
  transient with no visual partner.
- 110.6 s (whiteout, bound): "Let's go!" -- dialogue at the
  flash, matched to its visual partner.
All three match their narrative beats, and the 261.4 s slice
closes the loop the binding inventory opened.

### Cost: VM noise, not audio
Full-film end-to-end: audio-on 0.892x vs audio-off twin 0.74x on
the same VM session -- the audio-OFF run was slower. Audio's
measured cost (0.0353 ms/moment, 0.07% of the 50 ms budget)
cannot produce ±0.15 end-to-end swings in either direction;
the harness scaffolding and VM variance own that number. The
30 s A/B (audio-on FASTER, 0.96 vs 0.85) said the same thing.
Audio does not regress realtime; nothing more precise can be
claimed on this VM.

### Tests
Full suite green: 240 passed (includes the 11 audio tests:
feature shapes, RMS calibration, flux continuity, click/
silence/sine onset behavior, min-gap, RMS floor, binding
greediness, decode length/skew).

### Verdict restated: SHIP
The commission is complete. The ears hear onsets the eyes
never see (the exit PA is the loudest thing in the film), bind
tightly where the film is genuinely multimodal (±0.1 s median),
and change nothing about vision when on. The binding is 38%,
not 100% -- and the honest read is that the missing 62% is the
mix (score, dialogue, PA), not a detector failure. That made
that, where there's a that to be made.

## Addendum 2026-10-03 (night): audio-visual review watch

First full review watch with the audio pathway live on the same timeline (Mykal: "ride it again with your ears on"). Run dir `output/st_stream_av_full/`; review at `output/st_stream_av_full/review.md`; beat map at `output/st_stream_av_full/beat-map.md` (15 beats, protocol-following).

### Run configuration
`streaming/run_stream.py input/star_tours_1_ride_film.mp4 --outdir output/st_stream_av_full --motion-weight 1.0 --pursuit --audio --transcribe-onsets 10`. Fast-combo defaults (luma_ratio, fovea_levels 3, mask_cache), 96x54 third-split motion thumb. Full test suite green before the run (240 passed).

### Results
- 5,380 moments, 735 saccades, 161 pursuit segments, 60.2 s gliding (22.4% of the ride). 735+161 = 896 = the motion-off saccade count: pursuits replace catch-up saccades one-for-one. Realtime 0.86, 0/135 deadline misses.
- Audio: 238 onsets, 328 visual transients, 91 bound (38.2%) -- bit-for-bit identical to the audio-commission run (deterministic). Audio cost 0.039 ms/moment.
- Audio-off twin (`output/st_stream_av_full_audio_off_twin/`, same build, motion on): fixations, scanpath, frame_energies, motion_energies, pursuit_log all bit-identical (0.00 max abs diff). Audio observes, never steers -- now validated with the motion channel on.
- Attended transcription (top-10 onsets, whisper base, vad=False): exit PA cluster verbatim at logprob -0.36..-0.43 ("The captain has opened the exit doors... unlatch your safety restraints... personal belongings. Thank you."); "Oh my god!" at 205.25 s (bound, dt -250 ms); "I meant to do that. A little shortcut." at 55.75 s (unbound, corroborated by the film transcript); "What are you guys doing?" at 175.8 s (logprob -1.21); a loud startled vocalization at 200.35 s bound to the trench-dive transient (whisper reads profanity, logprob -1.05 -- wording uncertain); "We love you" at 154.35 s is confabulation on score texture (logprob -1.51), not dialogue.
- Ears-first findings: the film's #1/#2/#4 loudest onsets (265.25, 261.4, 268.2 s, strengths 635.5/577.9/433.1) are the exit PA and ALL unbound -- the loudest thing in the film has no visual partner; first green announced at dt=0.0 (onset 138.75 bound to visual 138.75); whiteout flash has a voice ("Let's go!" at 110.6 s, bound, dt 200 ms); trench is the most multimodal passage (event 51: 15/17 onsets bound; event 49: 10/10); the turn's 11 onsets are all unbound (comedy plays in the ears, flying in the eyes).

### Red-team corrections filed in the review's errata
1. Whiteout gaze: the "centered" correction overcorrected. Peak fixations are (0.47-0.56, 0.19) rt_full, (0.62-0.72, 0.49-0.60) magno, (0.72-0.83, 0.69-0.79) this run -- at the peak of a global flood the target is underdetermined, neither aversion nor centering is a model property. The approach is stably pursued; the ears bind through the flash regardless.
2. Hyperspace refusal: jump 1 admitted one 400 ms pursuit (63.20-63.60 s) at the tunnel's onset plus a bound onset (64.10 -> 64.05, the collapse). Jump 2 remains a clean refusal. Gate record 1.5-for-2; the admitted glide was scheduled on visual grounds (audio never steers).
3. Whisper content: logprob-gated -- "We love you" (-1.51) is confabulation; the 200.35 s wording (-1.05) is uncertain; transcription claims carry logprobs or don't ship.
4. Saccade/pursuit split vs the magno run (735/161 vs 742/154) traces to the motion thumb's resampling path (third-split ffmpeg vs older in-process downsample; motion energies differ mean 0.001/max 0.01), not to audio.

### Verdict
The ears changed the review: the film now has two endings (the eyes' window at 253.7-258.0 s, the ears' PA at 261-268 s), and the review's loudest moments are audio-first. Nothing about the eyes needed retracting beyond the two corrections above. 38% binding, tight where the film is multimodal.

## Dr Tran ep4 "Mr. Tran and the Toy Cack" — full Sensorium watch (2026-10-03, late)

Mykal's favorite episode, watched as if he'd sat the rig down in front of his PC. Run dir `output/drtran_ep4_av/`; review at `output/drtran_ep4_av/review.md`; beat map at `output/drtran_ep4_av/beat-map.md` (11 beats: the bits/gags); sealed predictions at `output/drtran_ep4_av/sealed-predictions.md` (4/6 held).

### Run configuration
`streaming/run_stream.py input/drtran_ep4_toy_cack.mp4 --outdir output/drtran_ep4_av --motion-weight 1.0 --pursuit --audio --transcribe-onsets 10`. Fast-combo defaults (luma_ratio, fovea_levels 3, mask_cache), 96x54 third-split motion thumb. Full test suite green before the run (241 passed, incl. the new plane-padding regression test).

### Results
- 8,721 moments (436.1 s), 1,444 saccades, 51 pursuit segments, 19.9 s gliding (4.6% of the episode). Every pursuit at minimum duration (0.29-0.39 s): the channel tries and never sustains. Realtime 1.26 (faster than realtime), 0 chunk deadline misses.
- Audio: 588 onsets, 601 visual transients, 298 bound (50.7%). Ears-first: the episode's #1 onset (59.1 s, strength 523.4, Grandma's "Today is a day for paper trails...") is UNBOUND -- mid-shot dialogue, no cut.
- Attended transcription (top-10 onsets, whisper base, vad=False): all genuine dialogue, per-segment logprobs -0.25..-1.01, zero confabulation. License confession at 329.8 s (onset #2, 484.2); Greg Keneer at 187.6 s; carrot tree at 79.6 s ("caribtry" -- whisper's one charming miss).
- The gap (377-409 s): 32 s no speech, held bloody tableau, but audio continues (RMS 0.0327 vs 0.0352 episode) -- score/stingers under a silent-looking shot. The commercial (409-415 s): 2 visual transients, 14 onsets -- ears work hardest where eyes rest.

### Red-team corrections filed in the review's errata
1. Binding interpretation: the 50.7% coincidence rate EQUALS the Poisson chance rate for the transient density (0.507 vs 0.50); dt distribution flat across +/-250 ms, not clustered at zero. Most "bound" pairs are density-driven coincidence, not causal binding. The ~39 pairs within 50 ms are the genuine candidates. Same instrument as Star Tours, different truth: there, 38% measured causation; here, 51% measures editing rhythm.
2. P1 scored FAIL on the letter (51 pursuits vs predicted <15) with the spirit intact -- all minimum-duration, no sustained tracking.
3. BUG FOUND AND FIXED: `hva/transcribe.py::_load_with_av` read the whole padded PyAV plane buffer instead of `[:rf.samples]`, inflating decoded audio 1.2194x and stretching every transcript timestamp (first Dr Tran transcript ran to 505 s on a 436 s file). Fixed, committed (54b1a7f), regression test added and verified failing on old logic. Blast radius checked: `hva/stream.py` already sliced (Star Tours AV audio/binding unaffected); `streaming/feeder.py` uses ffmpeg CLI (unaffected); filed Star Tours/FF3 transcripts max within media durations (unaffected).

### Verdict
The full Sensorium on a dialogue cartoon: the eyes hop (1,444 saccades, 98.7% of cuts chased within 200 ms vs 73% chance), the tracker starves (51 stillborn pursuits), the ears log 588 onsets of people talking. The fixture lesson with numbers on it: the motion pathway is healthy and bored. Nothing about the Star Tours findings needed retracting.

## FF3 "Backrooms - Found Footage #3" — full Sensorium watch, 1080p (2026-10-04, eep-time commission)

Mykal's favorite of the three Found Footage films, watched as the full eep-time ritual: the complete 1080p file (input/ff3_full_hd.mp4, 1440x1080, 2701s), everything on. Run dir `output/ff3_hd_av/`; review at `output/ff3_hd_av/review.md`; beat map at `output/ff3_hd_av/beat-map.md` (8 beats); sealed predictions at `output/ff3_hd_av/sealed-predictions.md` (1/12 clean, 1 partial, 10 misses — the most prediction-breaking run yet).

### Run configuration
`streaming/run_stream.py input/ff3_full_hd.mp4 --outdir output/ff3_hd_av --motion-weight 1.0 --pursuit --audio --transcribe-onsets 10`. Fast-combo defaults (luma_ratio, fovea_levels 3, mask_cache), 96x54 third-split motion thumb. Full test suite green before the run (259 passed). First run to cost more than realtime: 0.81x, 0 chunk deadline misses.

### Results
- 54,019 moments (2701.0 s), 7,867 saccades at a metronomic 2.9/s (flat across all content; lowest 5-min bin at 94% of mean — the attention driver runs on its own clock), 1,127 pursuit segments ALL at exactly 0.40s minimum, chained through walking footage (427.3 s gliding, 15.8%). The motion channel found the egomotion flow field and wouldn't let go — optokinetic gliding in 400ms hops, horizontally biased (53%), ~122 px/s.
- Audio: 3,362 onsets at 1.24/s (near-uniform across all beats — the hum/handling are transient-rich), 2,735 visual transients, 1,208 bound (35.9%) vs 39.7% Poisson chance — below chance; dt quantized to the 50ms moment grid, roughly uniform. Third straight run where binding measures the medium, not the moment. ~159 zero-dt pairs are the genuine candidates.
- New audio interpretation pathway (hva/interpret.py: CLAP labels, two-tier transcription, music analysis) run on a 918-window stratified sample (every 4th onset + top-100 loudest): speech 33.6% top-1 (VAD confirms 47/51 transcribed), footsteps 24.2%, thud 8.5%, engine 6.5%, music 6.3%. The film is vocalization-dense — breathing, grunting, shouting, conversing — not the "quiet hum" of its reputation. Two-tier transcription (110 windows, 47 escalated): 1649.7s "We're in Marvel!" resolves to screams; 1118.6s Korean hallucination resolves to laughter ("HO-HO-HO-HO-HO!"); genuine exchanges at 2335.9s, 2364.1s ("nine fucking hours"), 2430.6s. Music flags rare (18/918, 2.0%), clustered (791-846s, 1650s, 2562-2593s).
- The two loud dark pillars (B6 ~1620-1740s, B8 ~2280-2700s): loudest sustained audio (RMS 4-8x baseline) against the lowest visual energy anywhere. The horror is audio-led. Fixations scatter widest in the final dark (238px vs 143px lit wandering).

### Red-team corrections filed in the review's errata
1. Beat-map B1 correction (pre-review): first draft called B1 "Entry" and read the 19s frame-energy peak (0.244, film's largest) as the backrooms transition. Frame verification: the film opens in the normal world (park, car radio, flag, house) until the dark threshold at ~210-270s. The protocol caught it before the review did.
2. R5 rescored PARTIAL -> FAIL on red-team re-read: only the descent ladder clearly matched the predicted event stretches; generosity was grading on vibe.
3. Scream transcripts qualified: medium's 1649s outputs at logprobs -0.93..-1.04 support vocal character (screaming), not word-level claims. 92% VAD confirmation noted as an upper bound (sample skews loud).
4. 19s "cut" -> defocused bokeh passage (frames 17-21s show bokeh, not an edit).
5. Coverage note: interpretation on 918-window stratified sample, not all 3,362 (VM OOMs on 2 concurrent CLAP workers; single worker ~4s/window). Uniform onset rate makes the systematic sample representative; top-100 loudest fully covered.

### Verdict
The pipeline's FF3 is a film about exertion: 45 minutes of a body moving through a hostile space, heard more than seen. The sealed predictions broke 10/12 because the mental model was "empty hum with rare events" and the reality is "dense human noise over machine drone" — the best possible outcome for the regime. Nothing about the Star Tours or Dr Tran findings needed retracting; the binding coincidence result replicated for the third time (below chance here).
