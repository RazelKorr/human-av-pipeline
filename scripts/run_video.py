"""Run a real video file through the human vision pipeline.

Usage: python3 scripts/run_video.py <video.mp4> [--seconds N] [--width W --height H]

Decodes to grayscale at native fps, pushes frames with millisecond
timestamps (center fixation), pulls the 10 Hz percept stream, and writes:
  output/video_percept.mp4  -- what the pipeline "sees" (10 fps moments)
  output/video_compare.png  -- original vs percept at sample moments
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from hvp.pipeline import VisionPipeline
from hvp import baseline as B

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,duration,width,height",
         "-of", "json", path],
        capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    return {"fps": float(num) / float(den), "duration": float(s["duration"]),
            "w": s["width"], "h": s["height"]}


def decode_gray(path, seconds, fps, w=None, h=None, t0=0.0):
    """Yield (t_ms, HxW float32 frame in 0..1). Assumes CFR (see probe).

    w/h: working resolution. None -> native video dimensions (probed),
    so any input size parses without issue. Non-square is fine.
    t0: start offset in seconds (input seeking)."""
    if w is None or h is None:
        meta = probe(path)
        w, h = meta["w"], meta["h"]
    n = int(seconds * fps)
    # NOTE (2026-09-30): the vf chain MUST include fps={fps}. Without it,
    # ffmpeg emits native-fps frames while this loop reads/labels them at
    # `fps` -- for a 30fps source that silently watches only the first
    # third of the video stretched across the whole timeline, desynced
    # from the audio. Every Level 3 batch run before this fix had that bug.
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", path,
           "-t", str(seconds),
           "-vf", f"fps={fps},scale={w}:{h},format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = w * h
    base = t0 * 1000.0
    for i in range(n):
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        f = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
        yield base + i * 1000.0 / fps, f.astype(np.float32) / 255.0
    # Don't bare-wait: if ffmpeg produced more frames than we read, its
    # stdout pipe is full and it will never exit on its own (2026-09-30:
    # wedged a Level 3 run for 8+ minutes in proc.wait()).
    try:
        proc.stdout.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def write_mp4(frames, path, fps=10):
    h, w = frames[0].shape
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
           "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write((np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=None,
                    help="working width px; default = native video width")
    ap.add_argument("--height", type=int, default=None,
                    help="working height px; default = native video height")
    ap.add_argument("--fovea", type=float, default=None,
                    help="fovea radius in degrees; default = human baseline "
                         "(baseline.FOVEA_RADIUS_DEG)")
    args = ap.parse_args()

    meta = probe(args.video)
    print(f"video: {meta['w']}x{meta['h']} @ {meta['fps']:.2f} fps, "
          f"{meta['duration']:.1f}s", flush=True)

    w = args.width or meta["w"]
    h = args.height or meta["h"]
    # Square pixels: one degrees-per-pixel scale serves both axes; the
    # vertical field of view is derived (dva_per_px * h).
    dva = B.FIELD_WIDTH_DEG / w
    pipe = VisionPipeline((h, w), dva, [(0, w / 2, h / 2)],
                          fovea_radius_deg=args.fovea)

    n_in, t_end = 0, 0.0
    originals = {}
    for t, f in decode_gray(args.video, args.seconds, meta["fps"], w, h):
        pipe.push(f, t)
        n_in += 1
        t_end = t
        if n_in % int(meta["fps"]) == 0:  # keep 1 original per second
            originals[t] = f
    print(f"pushed {n_in} frames", flush=True)

    moments = list(pipe.moments(t_end))
    print(f"pulled {len(moments)} perceptual moments", flush=True)
    percepts = [p for _, p, _ in moments]
    write_mp4(percepts, os.path.join(OUT, "video_percept.mp4"))
    print("wrote output/video_percept.mp4", flush=True)

    # comparison strip: original vs percept near sample moments
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = np.linspace(0, len(moments) - 1, 6).astype(int)
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for j, i in enumerate(idx):
        t, p, m = moments[i]
        o = originals[min(originals, key=lambda k: abs(k - t))]
        axes[0][j].imshow(o, cmap="gray", vmin=0, vmax=1)
        axes[0][j].set_title(f"camera t={t:.0f}ms")
        axes[0][j].axis("off")
        axes[1][j].imshow(p, cmap="gray", vmin=0, vmax=1)
        axes[1][j].set_title(f"percept t={t:.0f}ms"
                             + (" (held)" if m["suppressed"] else ""))
        axes[1][j].axis("off")
    fig.suptitle("Star Tours ride film: camera truth vs the pipeline's "
                 "percept stream (foveated, 100 ms latency + 100 ms moments)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "video_compare.png"), dpi=110)
    plt.close(fig)
    print("wrote output/video_compare.png", flush=True)


if __name__ == "__main__":
    main()
