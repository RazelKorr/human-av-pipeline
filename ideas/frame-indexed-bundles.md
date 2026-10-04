# Frame-indexed sensorium bundles (review addressing scheme)

Filed 2026-10-01, from Mykal: for direct video input, review each frame
individually in sequence with the bundle of associated sensorium output
attached. Each frame collects the sensorium data it participated in —
the moments it fed, fixation, audio, salience — so the review still sees
the *full* movie while compressing the perceptual bloat but keeping the
data.

## The idea

Today the review's primary timeline is the moment stream (10Hz foveated
frames). The proposal flips the addressing: the primary timeline is the
*movie's own frames*, in sequence, and each frame carries a bundle of the
sensorium output associated with it. The review walks frames; the
perception rides along as annotation.

## Why it works

- The review stays anchored to the film's timeline — the thing the human
  knows cold (the ground truth) — with perception as annotation layers,
  not as a competing timeline.
- It preserves continuity: the turn-direction question (Star Tours,
  2026-10-01) lived *between* keyframes, in unbroken motion. A
  frame-sequential index can't skip it the way sampling can.
- Compression without loss: the bundle per frame is compact (moment ids,
  fixation coords, audio features, salience stats, event id). The full
  moment video stays on disk; the index just stops requiring you to
  re-watch 27,009 images to find anything.

## The subtlety (honest)

Frames are sub-perceptual: at 30fps each frame is ~33ms, shorter than the
100ms integration window that defines a perceptual moment. So "reviewing
each frame" means reviewing slices finer than the pipeline's own "now" —
the bundle for three consecutive frames will mostly point at the same
moment. That's fine: it's an index, not new data. But the moment remains
the perceptual atom; the frame index is an addressing scheme over it,
not a replacement for it.

## Implementation sketch

- In `VisionPipeline.moments()`, log per moment the source-frame indices
  (or timestamps) that fell in its integration window. The pipeline
  already knows `_times`; this is a small addition.
- Emit `frame_index.json` per run: frame_index -> {moment_ids,
  fixation_xy, suppressed, audio_rms, salience_stats, event_id}.
- The dense-log and review tooling then walk frames in order, expanding
  any frame into its full sensorium context on demand.
- Composes with the event map: events are moment ranges, which map to
  frame ranges through the same index.

## Status

**Built 2026-10-02** (overnight commission): `scripts/build_frame_index.py`
assembles `frame_index.json` from a run's `fixations.npy`,
`frame_energies.npy`, `run_report.json`, the audio wav, and the event
metrics. Per source frame: t_ms, contributing moment ids (analytic from
the integration window), nearest-moment fixation/suppression, audio RMS,
salience-channel energies, event index. First built for the
`st_full_v2` Star Tours run (color-native attention, 20 Hz moments).
