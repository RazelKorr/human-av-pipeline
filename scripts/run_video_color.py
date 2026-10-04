"""Full-length COLOR vision pass: every frame through the human vision pipeline.

Color variant of run_video_long.py. Decodes RGB at native fps, pushes every
frame with millisecond timestamps (center fixation), pulls the 10 Hz percept
stream. The pipeline foveates luminance sharply and chroma with a tiny
fovea + heavy peripheral blur (see hvp/retina.foveate_color) — sharp color
only at fixation, desaturating into the periphery, the way cones work.

Usage:
  python3 scripts/run_video_color.py <video.mp4> --outdir output/st_color
      [--seconds N] [--width W --height H] [--chunk SEC] [--leadin SEC]

Writes into --outdir:
  video_percept.mp4  -- what the pipeline "sees" (10 fps color moments)
  video_compare.png  -- original vs percept at sample moments
  run_report.json    -- params, timings, chunk log
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_video
from hvp.pipeline import VisionPipeline
from hvp import baseline as B


def decode_color(path, seconds, fps, w=None, h=None, t0=0.0):
    """Yield (t_ms, HxWx3 float32 RGB frame in 0..1). CFR assumed (see probe)."""
    if w is None or h is None:
        meta = run_video.probe(path)
        w, h = meta["w"], meta["h"]
    n = int(seconds * fps)
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", path,
           "-t", str(seconds),
           "-vf", f"fps={fps},scale={w}:{h},format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = w * h * 3
    base = t0 * 1000.0
    for i in range(n):
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        f = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        yield base + i * 1000.0 / fps, f.astype(np.float32) / 255.0
    proc.stdout.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=None,
                    help="process this many seconds; default = full duration")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--chunk", type=float, default=20.0,
                    help="seconds per chunk (default 20)")
    ap.add_argument("--leadin", type=float, default=2.0,
                    help="lead-in seconds pushed before each chunk (default 2)")
    ap.add_argument("--fovea", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    chunks_dir = os.path.join(args.outdir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, "manifest.json")
    done_chunks = set()
    if os.path.exists(manifest_path):
        try:
            done_chunks = set(json.load(open(manifest_path))["done"])
            print(f"resuming: {len(done_chunks)} chunks already done",
                  flush=True)
        except Exception:
            pass
    meta = run_video.probe(args.video)
    total = args.seconds or meta["duration"]
    total = min(total, meta["duration"])
    print(f"video: {meta['w']}x{meta['h']} @ {meta['fps']:.2f} fps, "
          f"{meta['duration']:.1f}s; processing {total:.1f}s (COLOR)",
          flush=True)

    w = args.width or meta["w"]
    h = args.height or meta["h"]
    dva = B.FIELD_WIDTH_DEG / w
    fps = meta["fps"]

    out_mp4 = os.path.join(args.outdir, "video_percept.mp4")

    def save_manifest():
        with open(manifest_path, "w") as f:
            json.dump({"done": sorted(done_chunks)}, f)

    t0_wall = time.time()
    n_in = n_moments = 0
    chunk_log = []
    t = 0.0
    chunk_i = 0
    n_chunks = int((total + args.chunk - 1e-9) // args.chunk) + 1
    while t < total:
        c_end = min(t + args.chunk, total)
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_i:04d}.mp4")
        if chunk_i in done_chunks and os.path.exists(chunk_path):
            print(f"chunk {chunk_i}: [{t:.0f}-{c_end:.0f}s] skipped "
                  f"(already done)", flush=True)
            t = c_end
            chunk_i += 1
            continue
        c0_wall = time.time()
        pipe = VisionPipeline((h, w, 3), dva, [(0, w / 2, h / 2)],
                              fovea_radius_deg=args.fovea)
        push_start = max(0.0, t - args.leadin)
        push_secs = (c_end - push_start)
        pushed = 0
        for t_ms, f in decode_color(args.video, push_secs, fps,
                                    w, h, t0=push_start):
            pipe.push(f, t_ms)
            pushed += 1
        n_in += pushed
        ff = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "10",
             "-i", "pipe:0", "-pix_fmt", "yuv420p", chunk_path],
            stdin=subprocess.PIPE)
        kept = 0
        start_ms = 0.0 if chunk_i == 0 else t * 1000.0 + 100.0
        for t_ms, p, m in pipe.moments(c_end * 1000.0, start_ms):
            ff.stdin.write((np.clip(p, 0, 1) * 255).astype(np.uint8).tobytes())
            kept += 1
        ff.stdin.close()
        ff.wait()
        n_moments += kept
        del pipe
        import gc
        gc.collect()
        c1_wall = time.time()
        done_chunks.add(chunk_i)
        save_manifest()
        chunk_log.append({"chunk": chunk_i, "t_start": t, "t_end": c_end,
                          "wall_s": round(c1_wall - c0_wall, 1)})
        print(f"chunk {chunk_i}/{n_chunks}: [{t:.0f}-{c_end:.0f}s] "
              f"pushed={pushed} moments={kept} wall={c1_wall - c0_wall:.1f}s",
              flush=True)
        t = c_end
        chunk_i += 1

    # stitch chunks into one percept mp4
    with open(os.path.join(chunks_dir, "concat.txt"), "w") as f:
        for i in range(chunk_i):
            f.write(f"file 'chunk_{i:04d}.mp4'\n")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", os.path.join(chunks_dir, "concat.txt"),
         "-c", "copy", out_mp4], check=True)
    wall_total = time.time() - t0_wall
    print(f"pushed {n_in} frames, pulled {n_moments} moments, "
          f"wall {wall_total:.0f}s", flush=True)
    print(f"wrote {out_mp4}", flush=True)

    # comparison strip: 6 sample times, original vs percept (color)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sample_ts = np.linspace(total * 0.05, total * 0.95, 6)

    def grab(src, t_s):
        cmd = ["ffmpeg", "-v", "error", "-ss", str(t_s), "-i", src,
               "-frames:v", "1", "-vf", f"scale={w}:{h},format=rgb24",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        out = subprocess.run(cmd, capture_output=True).stdout
        return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3) / 255.0

    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for j, t_s in enumerate(sample_ts):
        o = grab(args.video, t_s)
        p = grab(out_mp4, round(t_s * 10) / 10.0)
        axes[0][j].imshow(o, vmin=0, vmax=1)
        axes[0][j].set_title(f"camera t={t_s:.0f}s")
        axes[0][j].axis("off")
        axes[1][j].imshow(p, vmin=0, vmax=1)
        axes[1][j].set_title(f"percept t={t_s:.0f}s")
        axes[1][j].axis("off")
    fig.suptitle("color: camera truth vs the pipeline's percept stream "
                 "(foveated color, 100 ms latency + 100 ms moments)")
    fig.tight_layout()
    cmp_path = os.path.join(args.outdir, "video_compare.png")
    fig.savefig(cmp_path, dpi=110)
    plt.close(fig)
    print(f"wrote {cmp_path}", flush=True)

    report = {
        "video": os.path.basename(args.video),
        "source_meta": meta,
        "processed_seconds": total,
        "working_res": [w, h],
        "decode_fps": fps,
        "chunk_s": args.chunk,
        "leadin_s": args.leadin,
        "frames_pushed": n_in,
        "moments_pulled": n_moments,
        "wall_clock_s": round(wall_total, 1),
        "color": True,
        "chunks": chunk_log,
    }
    with open(os.path.join(args.outdir, "run_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote run_report.json", flush=True)


if __name__ == "__main__":
    main()
