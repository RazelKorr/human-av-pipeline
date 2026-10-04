# Magno channel: sealed predictions (2026-10-03 ~18:00 PDT)

Written BEFORE any motion-on validation run. These are the falsifiable
claims; the outcomes go in the STREAMING-REPORT.md addendum afterward.
No post-hoc rationalization: if a prediction fails, it fails on record.

Setup: Star Tours `input/star_tours_1_ride_film.mp4`, streaming driver,
fast combo (luma_ratio, fovea_levels=3, mask_cache, single-decode split).
Control = motion-off (defaults). Treatment = --motion-weight 1.0 --pursuit.

P1 (cost): motion channel adds <= 3 ms/frame mean when on. End-to-end
realtime factor stays >= 0.95x (baseline 0.98x).

P2 (control sanity): motion-off 30 s run reproduces the validated
numbers: 99 saccades, same fixations as the recorded baseline.

P3 (saccade count): motion-on 30 s saccade count within +/-25% of 99.
Motion attracts the eyes but does not multiply them.

P4 (where the eyes go): mean motion-energy-at-fixation is strictly
higher for motion-on than motion-off at matched timestamps. (The
channel works iff the eyes actually go where motion is.)

P5 (pursuit engages): >= 1 pursuit segment in the 30 s clip (it
contains fast action sections). Pursuit covers 5-25% of total time
on the full 269 s run.

P6 (whiteout immunity): zero pursuit segments overlapping
hyperspace-whiteout frames (full-frame flashes must not lock the
eyes -- this is the failure mode the coherence gate exists for).
Saccade rate during whiteout seconds does not spike vs control.

P7 (suppression): suppressed-moment fraction does not increase vs
control. suppressed is never True inside a pursuit segment
(biology: no saccadic suppression during pursuit).

P8 (pursuit quality): during pursuit segments, the motion centroid
near fixation stays within 3 deg of the glide position for > 80%
of pursuit time (the eyes actually track, not drift).
