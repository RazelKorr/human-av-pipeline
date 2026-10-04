# Sealed audio predictions — Star Tours full film (269 s)

Written 2026-10-03, BEFORE the first audio-on streaming run. Visual
event references from output/st_stream_magno_thumb96/event_map.json
(57 events) and the filed reviews. The audio onset detector (spectral-
flux peak-picking on 50 ms moments) has not yet run on this film.

## P1 — hyperspace roars are the two strongest onsets
The two hyperspace jumps (visual peaks 64.2 s, span 62.6–64.5 s; and
229.9 s, span 229.3–232.1 s) will produce the two strongest audio
onsets in the film. Each roar onset lands within ±0.5 s of its visual
peak. This is the alignment proof: if onsets land systematically
early/late, the audio track is misaligned, not the film.

## P2 — battle is a cluster, not a bang
The battle (visual peak 166.3 s, span 151.0–167.5 s, largest visual
change signal 102.21) will show the HIGHEST onset rate of any 17 s
window — explosions are many transients, not one. Predict onset rate
in 151–168 s > 2× the film median window rate.

## P3 — whiteout crescendo
The whiteout (visual peak 110.5 s, span 96.2–110.8 s): predict a
rising RMS ramp across the 96–110 s approach with the onset at the
flash within ±1.0 s of 110.5 s. The ears should hear a crescendo
where the reviews say the eyes stay centered.

## P4 — the turn is silent (as a transient)
The corrected first turn (~30–32 s, left yaw into the maintenance
bay): predict NO strong isolated onset tied to the maneuver itself.
Engine/whoosh is continuous, not transient. A big onset exactly at
the turn would be suspicious (score, not the maneuver) and counts
against the detector's specificity.

## P5 — trench is the second-loudest window
Trench run (204.2–221.7 s, peak 206.2 s): second-highest onset
density after the battle (weapons fire, engine strain).

## P6 — audio-visual coincidence
For the top-10 visual events by peak_diff, the median |nearest audio
onset − visual peak| will be < 1.0 s. Transients drive both
modalities; the big ones should coincide.

## P7 — cost
Audio feature extraction < 2 ms per 50 ms moment (< 4% of the moment
budget). End-to-end realtime factor with audio on will not regress
more than 0.02 below the audio-off 0.91× baseline.

## P8 — no video behavior change
With audio on, saccade count, fixations, and pursuit segments will be
BIT-IDENTICAL to audio-off. Audio is passive in v1: it observes,
never steers.

## Scoring (after the run)
Each prediction PASS/FAIL with the measured number. P1 doubles as
the track-alignment check: systematic offset → measure it, correct
it, re-run before scoring the rest.
