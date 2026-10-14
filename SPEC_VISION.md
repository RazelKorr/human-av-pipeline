# Human Vision Pipeline (HVP) — Spec

> **Specs rewritten 2026-10-13.** This spec now describes the streaming
> generation the 2026-10 improvement phases actually built, not only the
> batch generation it originally described. Where the two generations
> differ, both are named and labelled. Every architectural claim below is
> traceable to code in `hvp/`, `streaming/`, or a phase artifact in
> `improvements-2026-10/` / `streaming/STREAMING-REPORT.md`.

A functional emulation of human temporal vision: not a model of what the
retina *is*, but of how seeing *behaves in time*. If the timing budget is
right, the pipeline should trip over the same illusions a human trips over.
That is the entire validation strategy.

## 0. Two generations

- **Batch generation (v1/v2, still present and battery-validated):** the
  stages in §2, emitting 100 ms perceptual moments at 10 Hz.
- **Streaming generation (the flagship, which produced the recent
  full-film reviews):** `streaming/run_stream.py` — a chunked, causal,
  bounded-memory runner emitting **50 ms moments at 20 Hz** (see §2a).
  The temporal-integration window is still 100 ms; only the moment
  *spacing* changed. The FF3 flagship run's `run_report.json`
  (`output/ff3_hd_av/`) records `"moment_ms": 50.0, "moment_hz": 20.0`,
  `"integration_window_ms": 100.0`, and a `streaming` block with
  `chunk_s: 2.0`, 80,927 frames evicted, a final buffer of 700.7 ms,
  0 chunk deadline misses, and a 0.81x realtime factor over 2,701 s.

## 1. The baseline: human timing budget

All values are the demonstrative baseline — consensus numbers from vision
science, not fitted parameters.

| Stage | Value | Notes |
|---|---|---|
| Phototransduction (photon → retinal signal) | ~35 ms | rods/cones → ganglion cells |
| Retina → usable cortical signal | ~65 ms | LGN → V1; total pipeline latency **100 ms** |
| Recognition / categorization | ~150 ms | the classic 150 ms categorization limit |
| Temporal integration window (Bloch's law) | 100 ms | luminance sums over ~100 ms |
| Perceptual moment (alpha-cycle discretization) | 100 ms batch / 50 ms streaming | modeling assumption: moments discretized at ~10 Hz, after the alpha cycle; the streaming flagship (§2a) samples moments every 50 ms (20 Hz) over the same 100 ms integration window |
| Saccade rate | ~3.5 Hz | 3–4 ballistic jumps per second |
| Saccade duration | 21 + 2.2·A ms | main sequence; A = amplitude in degrees |
| Saccadic suppression | full | effectively blind mid-saccade; displacement suppressed |
| Fovea (high-acuity) diameter | ~2° | everything else is low-res + filled in |
| Acuity falloff | e₂ ≈ 2.5° | eccentricity constant for the falloff curve |
| Flicker fusion (foveal) | ~60 Hz | higher in periphery; CRT-era 60 Hz standard |

Net effect: **you live ~100–200 ms in the past**, see ~10 discrete moments
per second, and only ever inspect ~2° sharply at a time.

## 2. Pipeline stages

```
photon stream (high-fps input)
  → TemporalIntegrator   (100 ms sliding boxcar; the retina's summation)
  → SaccadeController    (scripted or salience-driven fixation; suppression windows)
  → Retina               (foveated sampling at current fixation)
  → PerceptualMoments    (quantize to 10 Hz; 100 ms pipeline delay)
  → percept stream       (what the "observer" gets)
```

### TemporalIntegrator
Holds the raw input stream. Each perceptual moment's neural signal is the
mean of input over `[t − 200 ms, t − 100 ms]` (100 ms integration window,
100 ms pipeline latency).

### SaccadeController
Scripted fixation sequences for tests: `[(t0, x0, y0), (onset1, x1, y1), …]`.
Saccade duration follows the main sequence. During a saccade the percept
*holds* the pre-saccadic frame (suppression of displacement) — no smear,
no update.

### Retina
Eccentricity-dependent blur around the fixation point:
`σ(e) = σmax · max(0, e − fovea) / (max(0, e − fovea) + e₂)`.
Sharp inside the fovea, saturating blur in the far periphery.

### PerceptualMoments
Output sampled every 100 ms. This discretization is what produces temporal
aliasing (wagon-wheel) and flicker fusion in the battery.

## 2a. The streaming generation (flagship)

```
file played as if live
  → ChunkedFeeder        (streaming/feeder.py; 2 s chunks by default;
                          one ffmpeg process, split filter graph giving
                          224px attention + work-res + motion-thumb
                          frames; --dual-decode keeps the old two-pipe
                          path, --single-mode pil is a documented
                          divergent alternative)
  → OnlineAttentionDriver (streaming/online_driver.py; the batch v2
                          pass-1 attention math as a causal push object —
                          transient decay, inhibition of return (1.5 s),
                          POI memory (12 s), saccade decision clock —
                          plus the magno/motion channel, §2b)
  → OnlineVisionPipeline  (streaming/online_render.py; VisionPipeline
                          subclass with a BOUNDED frame buffer)
  → percept stream        (50 ms moments, 20 Hz)
```

- **Bounded buffer / eviction.** `OnlineVisionPipeline.push()` keeps a
  deque and evicts any frame older than (newest emitted moment −
  latency − integration window − 500 ms margin), because no future moment
  can read it. `_integrated()` is untouched, so eviction is
  bit-identical: the report records buffer steady at ~667 ms in the
  validation runs and 700.7 ms at the end of the FF3 flagship run, with
  thousands of frames evicted. Moments are pulled as they become ready
  (`pull(t_now, t_end)`; the `t_end` cap stops the tail emitting one
  moment past the media duration).
- **Causal, one loop.** Driver and render run in one online loop;
  saccade/pursuit decisions are mirrored into the render controller as
  they are made. A render query only ever sees decisions made strictly
  before the stream time at which it runs — no lookahead. Streamed
  output is bit-identical to the post-hoc v2 path on moments, fixations,
  suppression flags, energies, and saccades (30 s / 62 s / full-film
  validated; `streaming/STREAMING-REPORT.md` §2a).
- **Colour is the flagship path, not an add-on.** The driver runs on
  colour frames: `hvp/attention.py` computes chromatic centre-surround
  salience on colour-opponent axes (R/G, B/Y opponency) alongside the
  luminance channels, and the render foveates colour. The streaming
  default chroma mode is `luma_ratio` (foveate luma, rescale RGB by the
  foveated-luma ratio; no YCbCr round-trip), with `fovea_levels=3` and
  the eccentricity-mask cache on — the "fast combo" defaults recorded
  in the FF3 `run_report.json` (`chroma_mode: luma_ratio`,
  `fovea_levels: 3`, `mask_cache: true`). The validated v2 math remains
  available explicitly (`--chroma-mode foveated --fovea-levels 5
  --no-mask-cache`); `passthrough` (chroma unfolded) is the third mode.

## 2b. The magno channel: motion energy and smooth pursuit

The old version of this spec never mentioned motion at all. Motion is
now a headline pathway of every motion-on flagship run.

- **MotionChannel (`hvp/motion.py`).** Frame differencing on a coarse,
  colourblind 16:9 thumbnail (default 96x54, derived from the
  work-resolution frame — never the anamorphic 224x224 attention frame),
  with a local-coherence gate (the motion map minus a heavily blurred
  copy of itself) so uniform floods — flashes, cuts, whiteouts — cancel
  to ~zero while local motion contrast survives. Threshold 0.05,
  persistence τ = 120 ms. Its energy map feeds salience when a
  `--motion-weight` is given (the FF3 flagship run used 1.0).
- **Pursuit.** When the motion centroid near fixation holds a consistent
  direction and a trackable speed (3.6–107 deg/s over a 400 ms history),
  the driver schedules pursuit glides instead of a saccade; there is no
  saccadic suppression during a glide and no inhibition of return while
  pursuing.
- **Residual-motion pursuit is the default pursuit pathway**
  (`OnlineAttentionDriver(pursuit_mode="residual")`, the driver default
  since Phase 3, 2026-10-12). Residual mode estimates the global
  translation by phase correlation, motion-compensates the previous
  frame by that shift, and only then differences — so pure camera
  motion (egomotion) cancels and what survives is motion that differs
  from the world's. Compensation is confidence-gated (correlation peak
  and structural-coverage floors; below them the plain diff stands).
  `legacy` mode — pursuit on the raw coherence-gated diff — is kept
  behind `--pursuit-mode legacy` for A/B comparison. Note the precise
  scope of "default": residual is the default *mode when pursuit is
  enabled*; pursuit itself, and the motion weight, are still enabled
  per run (`--motion-weight`, `--pursuit` on `run_stream.py`).
- **Why residual became the default, with the miss on record.** Under
  legacy mode on FF3, all 1,127 pursuits were exactly the 0.40 s minimum
  duration — the channel locked onto the egomotion flow field and
  hop-sustained instead of tracking objects. On the Phase 3 validation
  segment (FF3 1200–1500 s) residual mode cut pursuits 159 → 41
  (−74%), Star Tours was preserved (120 pursuits, 45.1 s, vs legacy
  161 / 60.2 s), and Dr Tran ep4 stayed starved (49 vs legacy 51).
  The sealed prediction for the segment was ≤15% of legacy; 25.8%
  **missed** that bar — the survivors are camera rotation/zoom and
  genuinely local movers, which translation compensation does not (and
  should not) cancel. The default flip was made with that miss recorded
  in `improvements-2026-10/PREDICTIONS.md`, not laundered past it.

## 3. Validation battery (pass = behaves like a human)

- **T0 latency**: step input 0→1 at t=1000 ms. First moment reflecting it
  must land 100–250 ms later. (You live in the past.)
- **T1 flicker fusion**: square-wave patch at 5/15/30/60/120 Hz. Temporal
  variance of the percept stream must be high at 5–15 Hz and collapse by
  60 Hz. (CRT fusion.)
- **T2 wagon-wheel**: 8-spoke wheel, spoke-pass frequency 9 Hz against the
  10 Hz moment clock. Measured spoke angle across moments must drift
  *backward* (aliasing to −1 Hz). (Temporal sampling artifact.)
- **T3 saccadic suppression**: 20 ms full-field flash during a scripted
  saccade vs. during fixation. Percept deviation in the saccade case must
  be <25% of the fixation case. (Blind mid-jump.)
- **T4 change blindness**: flicker paradigm — scenes A/B alternate (500 ms
  each), one bar of 64 flipped, with or without a 200 ms blank mask
  between. Salience-driven saccades (static center-surround + persistent
  transient channel, inhibition of return, 200 ms saccade latency).
  Without the mask the flip's transient captures the eyes in ~2 cycles;
  with the mask the transient is swamped and detection takes an order of
  magnitude longer (mean 22 vs 2 cycles; 5/6 masked runs never find it in
  25). (The invisible-gorilla family: inattentional vs. change blindness.)

## 4. Deliberate non-goals

- No rods/cones split, no cortical magnification beyond the blur
  falloff. The baseline is *temporal*; spatial detail is the minimum
  needed to make the tests meaningful.
  *(Correction, 2026-10-13: earlier versions of this section also listed
  "no color opponency" as a non-goal. That is false about the current
  system — colour-opponent chroma salience and colour foveation are
  implemented and are the flagship v2/streaming path; see §2a. The
  non-goal described the v1 temporal baseline only.)*
- No claim about qualia. Functional emulation only.
- The T4 salience search can settle into attractor loops among the most
  salient distractors (inhibition of return decays faster than the loop
  period), so masked search is slower than a human's — reported cycles
  are a demonstration of the mask/no-mask contrast, not fitted human RT.
- Express saccades (~100 ms latency to sudden onsets) are not modeled;
  all saccades pay the full 200 ms visually-guided latency.

## 5. Parked directions (2026-09-29, RazelKorr's call)

- Realtime webcam embodiment ("Wodehaus seeing through a camera") is
  parked. The bottom-up stack already runs ~3x realtime, so it is
  technically feasible — it is simply not the goal right now. The
  research target is the vision/attention model, not a live demo.
  (2026-10-03 note: the streaming harness (`streaming/`, see
  `streaming/STREAMING-REPORT.md`) now gives the v2 path an online
  story — bit-identical to batch, bounded memory, 0.91x realtime
  end-to-end on the test VM with vision compute proper at ~1.7x
  realtime — and the external-stream front door is designed. The
  live-sensor embodiment itself remains parked; the harness is the
  seam for it.)
- Foveated sampling of the full-res image (full-res foveal cutout +
  low-res periphery, instead of rendering megapixels just to blur them)
  is parked as a hardware problem. Emulation is possible; held for now.
- **Live sensors are paper-only (Phase 6, 2026-10-12).** No code in
  this repo opens a camera, a microphone, an RTSP stream, or any
  device. The readiness note
  (`improvements-2026-10/LIVE-SENSORS-READINESS.md`) records the seam
  only: `ChunkedFeeder` is the only file-aware component, and a live
  client would implement the same `Chunk` iterator from a camera
  instead of a file — that client does not exist; the front-door doc
  (`streaming/docs/external-stream-front-door.md`) is a design sketch.
  Two gates remain the user's, and outrank any code readiness: the
  2026-10-06 terms-first agreement (explicit terms for any future
  device mic/camera before it goes live) and the hardware boundary
  (the paired iPhone exposes no camera/mic; the live path would be
  future ESP32-class hardware that does not exist in the house yet).
  Nothing in this spec should be read as describing live sensing as
  implemented.

## 6. Input handling (2026-09-29)

- The pipeline accepts any video size and aspect ratio. Decode defaults
  to native resolution; `--width/--height` overrides the working size.
- Geometry model: square pixels, frame width subtends FIELD_WIDTH_DEG,
  so one degrees-per-pixel scale serves both axes (vertical FOV derived).
  The old square squash is gone — 16:9 stays 16:9.
- The salience grid stays 56x56 over any frame via per-axis scales
  (sx, sy); the thumbnailer is bilinear (PIL), no divisibility needed.

## 7. Noted refinements

- **Darkness behavior (2026-09-29, RazelKorr's observation):** in near-black
  frames the model kept saccading on salience noise; human eyes don't.
  His framing: "there is an increase in focus to try and comprehend
  something difficult, but not much." Implemented as a luminance gate
  on the saccade decision clock (`darkness_gain` in hvp/saccades.py):
  below 0.03 mean luminance the decision interval ramps up to 3x,
  i.e. longer dwells in the dark. A/B on Star Tours 0-31s (dark cut at
  ~14-19s): dark-frame decisions 16 -> 7, mean dark dwell 286ms ->
  631ms; bright-frame behavior unchanged (87/88 decisions, 286ms).
  `--no-dark-gate` recovers the baseline.
- **Adaptive saccade clock — OPT-IN ONLY (Phase 3, 2026-10-12).** The
  streaming driver's fixed decision clock measured near-flat across
  content regimes (FF3 2.9/s in every beat; Pink Panther 3.44/s bedroom
  vs 3.43/s brawl) — a metronome with salience steering *where*, not
  *when*. The adaptive clock (`OnlineAttentionDriver(saccade_clock=
  "adaptive")`, `--saccade-clock adaptive`) modulates the interval by
  the batch path's `darkness_gain` and a causal activity EMA (magno
  motion mean when the channel is on, transient-map mean otherwise),
  clipped to 200–667 ms (an effective 1.5–5.0/s). It is unit-verified
  and it modulated where its signal separated (quiet-room fixture:
  94 saccades vs 199 fixed), **but its headline prediction missed**:
  on Pink Panther the bedroom/brawl divergence was 0.05/s against a
  sealed hit bar of ≥0.8/s, because the whole-frame activity statistic
  saturates. Disposition: `saccade_clock` ships opt-in, default
  `"fixed"`. Do not describe the adaptive clock as the default. The
  follow-up the data points to (a top-decile motion statistic) is a
  new experiment, not a re-roll.
- **OCR is a seam only — signage is human-read today (`hvp/ocr.py`,
  Phase 5, 2026-10-12).** The module wraps tesseract behind a pluggable
  engine interface (`read_words`; frames/timestamps in,
  `[{t, text, confidence, bbox}]` out, confidence-floored at 40) and
  is **not wired into the flagship streaming path**. Validation: exact
  matches on a sterile synthetic fixture (PIL-rendered text down to
  10 px), but on real Star Tours frames at known-signage moments it
  read **no real sign** — garbage tokens on the bay signage and
  texture pareidolia (up to confidence 80) on machinery. Engine
  confidence does not make texture reads trustworthy. Per
  `REVIEW-PROTOCOL.md` §6, diegetic signage in footage stays
  human-read; this module is the seam a future engine plugs into,
  not a capability the pipeline has.
