# Human Vision Pipeline (HVP) — Spec

A functional emulation of human temporal vision: not a model of what the
retina *is*, but of how seeing *behaves in time*. If the timing budget is
right, the pipeline should trip over the same illusions a human trips over.
That is the entire validation strategy.

## 1. The baseline: human timing budget

All values are the demonstrative baseline — consensus numbers from vision
science, not fitted parameters.

| Stage | Value | Notes |
|---|---|---|
| Phototransduction (photon → retinal signal) | ~35 ms | rods/cones → ganglion cells |
| Retina → usable cortical signal | ~65 ms | LGN → V1; total pipeline latency **100 ms** |
| Recognition / categorization | ~150 ms | the classic 150 ms categorization limit |
| Temporal integration window (Bloch's law) | 100 ms | luminance sums over ~100 ms |
| Perceptual moment (alpha-cycle discretization) | 100 ms | conscious perception updates at ~10 Hz |
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

- No color opponency, no rods/cones split, no cortical magnification
  beyond the blur falloff. The baseline is *temporal*; spatial detail is
  the minimum needed to make the tests meaningful.
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
- Foveated sampling of the full-res image (full-res foveal cutout +
  low-res periphery, instead of rendering megapixels just to blur them)
  is parked as a hardware problem. Emulation is possible; held for now.

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
