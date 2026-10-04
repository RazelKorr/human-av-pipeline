# Sealed predictions: motion thumbnail size experiment (2026-10-03)

Written BEFORE implementation or benchmarking. Mykal's directive: the
motion thumb must match the video's 16:9 aspect ratio; find the biggest
thumb that holds realtime (rt >= 0.95x end-to-end).

## Background

The magno channel was validated on a 56x56 SQUARE thumbnail derived
from the anamorphic 224x224 attention frame. On 16:9 footage that is
horizontally squished: the motion field the eyes use is anisotropic
(~11.4x horizontal vs ~6.4x vertical downsampling from work res).
This experiment: 16:9 thumbs derived from the 640x360 work frame,
parameterized size, benchmark ladder.

Baselines (30 s Star Tours, luma_ratio + levels=3 + mask cache +
motion on + pursuit): 0.96x realtime end-to-end, 0.49 ms/frame motion
cost at 56x56, 99 saccades / 10 pursuit segments on the 30 s clip
(magno sealed predictions P1-P8).

## Timing predictions

Per-frame motion cost ~= PIL work-frame downsample + abs-diff +
coherence gate (gate blur computed at <=64px working width, then
upsampled -- constant cost by design) + threshold + history copy.

- T1: 96x54 (new small 16:9 default, ~5.2k px): ~1.5-2.5 ms/frame.
- T2: 160x90 (quarter of work res, 14.4k px): ~2-3 ms/frame.
- T3: 480x270 (quarter-dimension of 1080p, 129.6k px): ~3-5 ms/frame.
- T4: 960x540 (quarter-pixel of 1080p, 518.4k px): ~8-14 ms/frame
  (PIL upsample 640x360 -> 960x540 dominates).
- T5: end-to-end rt >= 0.95x holds for 96x54, 160x90, 480x270.
  960x540 FAILS the bar (current headroom: p50 29.8 ms vs 33.3 ms
  budget = ~3.5 ms; 8+ ms does not fit).
- T6: predicted winner (biggest passing thumb): 480x270. 160x90 is
  the safe fallback if T3's PIL cost surprises.

## Behavior predictions

- B1 (ratio fix alone): 96x54-from-work-frame vs legacy 56x56-from-224
  on the 30 s clip -- saccade counts within +/-15%, pursuit segment
  count within +/-50% (small N), and the magno sealed predictions
  P1-P8 still PASS qualitatively. The fix corrects geometry, not
  sensitivity.
- B2 (ladder): bigger thumbs resolve finer motion detail, so velocity
  estimates get less noisy -- pursuit segment count stays flat or
  rises slightly, saccade counts within +/-10% across sizes. NO
  qualitative change in what the eyes do: motion is still motion,
  the local-coherence gate still kills global floods at every size
  (unit-tested at each size before benchmarking).
- B3: the salience motion term is resized to 56x56 anamorphic for the
  salience add at every thumb size, so saccade targeting should be
  nearly size-invariant; any behavior delta comes through the
  pursuit velocity path, not the salience path.

## What would falsify the approach

- If the ratio fix alone flips P6 (pursuit on flashes) or P8-class
  behavior qualitatively, the geometry change is not neutral and the
  legacy path stays default pending investigation.
- If NO size above 96x54 holds 0.95x, the verdict is "small 16:9
  thumb ships, ladder fails" -- reported with numbers, not worked
  around with extra optimization in this pass.
