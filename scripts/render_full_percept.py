"""Render the full closed-loop run as a percept video.

Streams the 269 s ride film in 45 s chunks (memory-bounded), foveates each
10 Hz perceptual moment at the 7-degree tool fovea using the final saccade
script from the closed-loop run, concatenates, and muxes the source audio
delayed by 150 ms -- the center of the pipeline's [t-200, t-100] ms
integration window -- so what you hear lines up with what the agent saw.

Output: output/closed_loop/full_ride_percept.mp4
"""

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_video
from hvp.pipeline import VisionPipeline
from hvp import baseline as B

VSIZE = 224
FOVEA_DEG = 7.0
OUT = os.path.join(ROOT, "output", "closed_loop")
VIDEO = os.path.join(ROOT, "input", "star_tours_1_ride_film.mp4")
BOUNDS = [0, 45000, 90000, 135000, 180000, 225000, 269033]
AUDIO_DELAY_MS = 150  # center of the [t-200, t-100] integration window


def main():
    st = json.load(open(os.path.join(OUT, "full_s6", "state.json")))
    script = [(0.0, 112.0, 112.0)] + [(float(t), float(x), float(y))
                                      for t, x, y in st["script"]]
    print(f"saccade script: {len(script)} fixations", flush=True)
    meta = run_video.probe(VIDEO)
    dva = B.FIELD_WIDTH_DEG / VSIZE

    chunk_paths = []
    for i in range(len(BOUNDS) - 1):
        T0, T1 = BOUNDS[i], BOUNDS[i + 1]
        dec0 = max(0, T0 - 1000)  # 1 s lead-in for integration + hold seed
        dur = (T1 - dec0) / 1000.0
        pipe = VisionPipeline((VSIZE, VSIZE), dva, script,
                              fovea_radius_deg=FOVEA_DEG)
        n = 0
        for t, f in run_video.decode_gray(VIDEO, dur, meta["fps"],
                                          VSIZE, VSIZE,
                                          t0=dec0 / 1000.0):
            pipe.push(f, t)
            n += 1
        percepts = [p for t, p, m in pipe.moments(T1, t_start_ms=dec0)
                    if T0 <= t < T1]
        cp = os.path.join(OUT, f"_chunk_{i}.mp4")
        run_video.write_mp4(percepts, cp)
        chunk_paths.append(cp)
        print(f"chunk {i}: {n} frames in, {len(percepts)} moments "
              f"[{T0/1000:.0f}-{T1/1000:.0f}s]", flush=True)
        del pipe, percepts

    lst = os.path.join(OUT, "_chunks.txt")
    with open(lst, "w") as fh:
        for cp in chunk_paths:
            fh.write(f"file '{cp}'\n")
    silent = os.path.join(OUT, "_silent.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe",
                    "0", "-i", lst, "-c", "copy", silent], check=True)

    final = os.path.join(OUT, "full_ride_percept.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", silent, "-i", VIDEO,
         "-map", "0:v", "-map", "1:a",
         "-af", f"adelay={AUDIO_DELAY_MS}|{AUDIO_DELAY_MS}",
         "-c:v", "copy", "-c:a", "aac", "-shortest", final], check=True)
    for cp in chunk_paths + [silent, lst]:
        os.remove(cp)
    print(f"wrote {final}", flush=True)


if __name__ == "__main__":
    main()
