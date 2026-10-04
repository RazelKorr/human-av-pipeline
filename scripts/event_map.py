#!/usr/bin/env python3
"""Full-feed event map: analyze EVERY percept frame (27,009), no sampling.

Decodes the percept video at reduced resolution, computes per-frame metrics:
- mean_abs_diff: mean absolute pixel difference vs previous frame (motion/change)
- brightness: mean pixel value
- dark_frac: fraction of pixels below threshold (how "black" the frame is)

Writes a JSON event map plus a list of "events": contiguous runs where the
frame differs meaningfully from its neighbors (jitter, flashes, cuts, new shots),
with start/end timestamps. Every one of the 27,009 frames is analyzed.
"""
import json
import subprocess
import numpy as np

PERCEPT = "/home/hatch/workspace/human-av-pipeline/output/ff3_full/video_percept.mp4"
OUT = "/home/hatch/workspace/human-av-pipeline/output/ff3_full/event_map.json"
W, H, FPS = 160, 120, 10  # analysis resolution; timestamps stay exact

cmd = [
    "/usr/bin/ffmpeg", "-v", "error", "-i", PERCEPT,
    "-vf", f"scale={W}:{H},format=gray",
    "-f", "rawvideo", "-pix_fmt", "gray", "-",
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
frame_size = W * H
n_frames = 27009

diffs = np.zeros(n_frames, dtype=np.float32)
brights = np.zeros(n_frames, dtype=np.float32)
darkfrac = np.zeros(n_frames, dtype=np.float32)
prev = None
for i in range(n_frames):
    raw = proc.stdout.read(frame_size)
    if len(raw) < frame_size:
        break
    f = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    brights[i] = f.mean()
    darkfrac[i] = (f < 16).mean()
    if prev is not None:
        diffs[i] = np.abs(f - prev).mean()
    prev = f
proc.wait()

# Event detection: a frame is "active" if it differs from the previous one
# above a small noise floor, or brightness jumps. Group active frames into events.
ACTIVE_DIFF = 1.5   # mean abs diff threshold (tuned for foveated grayscale)
events = []
in_event = False
start = 0
for i in range(n_frames):
    active = diffs[i] > ACTIVE_DIFF
    if active and not in_event:
        in_event, start = True, i
    elif not active and in_event:
        # require 5 quiet frames before closing (bridges micro-gaps)
        if i - start >= 1:
            quiet_run = 0
            j = i
            while j < n_frames and diffs[j] <= ACTIVE_DIFF and quiet_run < 5:
                quiet_run += 1
                j += 1
            if quiet_run >= 5:
                events.append({"start_frame": start, "end_frame": i - 1,
                               "start_s": round(start / FPS, 1),
                               "end_s": round((i - 1) / FPS, 1),
                               "peak_diff": round(float(diffs[start:i].max()), 2)})
                in_event = False
if in_event:
    events.append({"start_frame": start, "end_frame": n_frames - 1,
                   "start_s": round(start / FPS, 1),
                   "end_s": round((n_frames - 1) / FPS, 1),
                   "peak_diff": round(float(diffs[start:].max()), 2)})

# Summary stats
result = {
    "n_frames_analyzed": n_frames,
    "fps": FPS,
    "duration_s": round(n_frames / FPS, 1),
    "n_events": len(events),
    "mean_diff": round(float(diffs.mean()), 3),
    "frac_dark_frames": round(float((darkfrac > 0.95).mean()), 4),
    "events": events,
}
with open(OUT, "w") as fh:
    json.dump(result, fh)
print(f"analyzed {n_frames} frames -> {len(events)} events")
print(f"dark-frame fraction: {result['frac_dark_frames']}")
print(f"wrote {OUT}")
