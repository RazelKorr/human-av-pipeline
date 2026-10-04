"""Full wiring: salience-driven saccades + foveated COLOR vision.

Two passes, like run_video_saccades.py, but the render pass is color and
the attention driver is color-native:
  Pass 1: attention driver (hvp.attention, 224px) watches the decoded
          COLOR frames and picks saccade targets at ~3.5 Hz with
          200 ms latency -> fixation script (in 224px coords).
          Salience = luminance center-surround + transient channel
          + chromatic center-surround (R/G, B/Y opponency)
          - inhibition of return - current-fixation disk.
  Pass 2: the fixation script is rescaled to the working resolution and
          VisionPipeline renders the foveated COLOR percept stream
          along the scanpath (sharp luminance, color-blind periphery),
          chunked so long videos fit in memory. Per-moment fixations are
          logged for the tracking panel.

Temporal resolution: --moment-ms sets the perceptual-moment spacing
(default 50 ms = 20 Hz). The 100 ms integration window is unchanged
(Bloch's-law temporal integration); finer spacing means overlapping
windows, resolving saccade flights (~50 ms) that a 100 ms grid swallows.

Usage:
  python3 scripts/run_video_saccades_color.py <video.mp4> --outdir OUT
      [--seconds N] [--width W --height H] [--chunk SEC] [--fovea DEG]
      [--moment-ms MS] [--chroma-weight W]

Writes into --outdir:
  video_percept.mp4  -- color percept stream along the scanpath
  fixations.npy      -- per-moment (t_ms, fx, fy, suppressed) in work-px
  frame_energies.npy -- per-frame (t_ms, lum_e, chroma_e, trans_e),
                        the salience-channel energies for frame bundles
  scanpath.png       -- color frames with gaze trail + color percepts
  run_report.json
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
from run_video_color import decode_color
from scipy.ndimage import gaussian_filter
from hvp import attention as A
from hvp import baseline as B
from hvp.pipeline import VisionPipeline
from hvp.saccades import SaccadeController

VSIZE = 224  # pass-1 decode size; matches attention.SIZE


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--chunk", type=float, default=20.0)
    ap.add_argument("--leadin", type=float, default=2.0)
    ap.add_argument("--fovea", type=float, default=None,
                    help="fovea radius deg; default = human baseline")
    ap.add_argument("--moment-ms", type=float, default=50.0,
                    help="perceptual-moment spacing in ms; default 50 (20 Hz). "
                         "Integration window stays 100 ms (human baseline).")
    ap.add_argument("--chroma-weight", type=float, default=None,
                    help="weight of the chromatic salience channel; "
                         "default = hvp.attention.CHROMA_WEIGHT")
    args = ap.parse_args()
    moment_ms = args.moment_ms
    moment_hz = 1000.0 / moment_ms

    os.makedirs(args.outdir, exist_ok=True)
    chunks_dir = os.path.join(args.outdir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    meta = run_video.probe(args.video)
    fps = meta["fps"]
    total = args.seconds or meta["duration"]
    total = min(total, meta["duration"])
    print(f"video: {meta['w']}x{meta['h']} @ {fps:.2f} fps, "
          f"{meta['duration']:.1f}s; processing {total:.1f}s", flush=True)

    # ---------------- pass 1: attention driver -> fixation script ----------------
    print("pass 1: attention driver (color-native, 224px)…", flush=True)
    # Stream color frames (no retained list: 8070 x 224x224x3 float32
    # would be ~5 GB). Per frame: luma thumbnail for the static/transient
    # channels, color thumbnail for the chroma channel.
    t_end = total * 1000.0
    dt = 1000.0 / fps
    dva1 = B.FIELD_WIDTH_DEG / VSIZE
    cx = cy = VSIZE / 2.0
    controller = SaccadeController([(0, cx, cy)], dva1)
    inhib = []
    scanpath = [(0.0, cx, cy)]
    pois = []
    trans = np.zeros((A.SMALL, A.SMALL), dtype=np.float32)
    prev_small = None
    decay = np.exp(-dt / A.TRANS_TAU_MS)
    next_decision = 100.0
    energies = []  # (t_ms, lum_e, chroma_e, trans_e) for frame bundles
    n_frames = 0

    for t, f in decode_color(args.video, total, fps, VSIZE, VSIZE):
        n_frames += 1
        small_rgb = A._downsample_color(f)
        # luma thumbnail for the luminance static + transient channels
        lum = (small_rgb @ np.array([0.299, 0.587, 0.114],
                                    dtype=np.float32))
        small = lum
        if prev_small is not None:
            trans = np.maximum(trans * decay, np.abs(small - prev_small))
        prev_small = small
        static_e = float(np.abs(
            small - gaussian_filter(small, 6)).mean())
        chroma_map = A.chroma_salience(small_rgb)
        energies.append((t, static_e, float(chroma_map.mean()),
                         float(trans.mean())))
        if t >= next_decision:
            sal_base = A.salience_map(small, trans, t, controller, inhib,
                                      small_rgb=small_rgb,
                                      chroma_weight=args.chroma_weight)
            fx0, fy0, _ = controller.state_at(t)
            yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
            dist_deg = np.hypot(xx - fx0 / A.SCALE,
                                yy - fy0 / A.SCALE) * A.SCALE * dva1
            sal = sal_base + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
            for (px, py, pt, ps) in pois:
                age = t - pt
                if age < 12000.0:
                    sal = sal + (ps * np.exp(-age / 6000.0)
                                 * A._blob((A.SMALL, A.SMALL), px, py, 12.0))
            iy, ix = np.unravel_index(int(np.argmax(sal)), sal.shape)
            tx, ty = (ix + 0.5) * A.SCALE, (iy + 0.5) * A.SCALE
            t_on = t + B.SACCADE_LATENCY_MS
            if t_on < t_end:
                controller.add_saccade(t_on, tx, ty)
                scanpath.append((t_on, tx, ty))
                strength = float(np.clip(sal_base[iy, ix] / 2.0, 0.2, 1.5))
                pois.append((float(ix), float(iy), t, strength))
                pois = [(x, y, it, s) for (x, y, it, s) in pois
                        if t - it < 12000.0][-10:]
            inhib = [(x, y, it) for (x, y, it) in inhib if t - it < 1500.0]
            inhib.append((ix, iy, t))
            next_decision = t + 1000.0 / B.SACCADE_RATE_HZ

    n_sac = len(scanpath) - 1
    print(f"driver made {n_sac} saccades over {n_frames} frames", flush=True)
    energies_arr = np.array(energies, dtype=np.float32)
    np.save(os.path.join(args.outdir, "frame_energies.npy"), energies_arr)
    print(f"wrote frame_energies.npy ({len(energies_arr)} frames)", flush=True)
    import gc
    gc.collect()

    # ---------------- pass 2: chunked color render along scanpath ----------------
    w, h = args.width, args.height
    sx, sy = w / VSIZE, h / VSIZE
    script2 = [(t, x * sx, y * sy) for (t, x, y) in controller.script]
    dva2 = B.FIELD_WIDTH_DEG / w
    print(f"pass 2: color render at {w}x{h}, {moment_hz:.0f} Hz moments, "
          f"along {n_sac} saccades…", flush=True)

    out_mp4 = os.path.join(args.outdir, "video_percept.mp4")
    manifest_path = os.path.join(args.outdir, "manifest.json")
    done_chunks = set()
    if os.path.exists(manifest_path):
        try:
            done_chunks = set(json.load(open(manifest_path))["done"])
        except Exception:
            pass

    def save_manifest():
        with open(manifest_path, "w") as f:
            json.dump({"done": sorted(done_chunks)}, f)

    t0_wall = time.time()
    n_in = n_moments = 0
    fix_log = []  # (t_ms, fx, fy, suppressed)
    t = 0.0
    chunk_i = 0
    n_chunks = int((total + args.chunk - 1e-9) // args.chunk) + 1
    while t < total:
        c_end = min(t + args.chunk, total)
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_i:04d}.mp4")
        if chunk_i in done_chunks and os.path.exists(chunk_path):
            # re-derive fixations for skipped chunks from the script
            ctl = SaccadeController(script2, dva2)
            k0 = 0 if chunk_i == 0 else int(t * 1000.0 / moment_ms) + 1
            k1 = int(c_end * 1000.0 / moment_ms)
            for k in range(k0, k1 + 1):
                mt = k * moment_ms
                tc = mt - B.PIPELINE_LATENCY_MS - B.INTEGRATION_WINDOW_MS / 2.0
                fx, fy, in_sac = ctl.state_at(max(tc, 0.0))
                fix_log.append((mt, fx, fy, bool(in_sac)))
                n_moments += 1
            print(f"chunk {chunk_i}: [{t:.0f}-{c_end:.0f}s] skipped "
                  f"(already done)", flush=True)
            t = c_end
            chunk_i += 1
            continue
        c0_wall = time.time()
        ctl = SaccadeController(script2, dva2)
        pipe = VisionPipeline((h, w, 3), dva2, script2,
                              fovea_radius_deg=args.fovea,
                              moment_ms=moment_ms)
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
             "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{moment_hz:.1f}",
             "-i", "pipe:0", "-pix_fmt", "yuv420p", chunk_path],
            stdin=subprocess.PIPE)
        kept = 0
        start_ms = 0.0 if chunk_i == 0 else t * 1000.0 + moment_ms
        for t_ms, p, m in pipe.moments(c_end * 1000.0, start_ms):
            ff.stdin.write((np.clip(p, 0, 1) * 255).astype(np.uint8).tobytes())
            fx, fy = m["fixation"]
            fix_log.append((t_ms, float(fx), float(fy),
                            bool(m["suppressed"])))
            kept += 1
        ff.stdin.close()
        ff.wait()
        n_moments += kept
        del pipe
        gc.collect()
        c1_wall = time.time()
        done_chunks.add(chunk_i)
        save_manifest()
        print(f"chunk {chunk_i}/{n_chunks}: [{t:.0f}-{c_end:.0f}s] "
              f"pushed={pushed} moments={kept} wall={c1_wall - c0_wall:.1f}s",
              flush=True)
        t = c_end
        chunk_i += 1

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

    fix_arr = np.array(fix_log, dtype=np.float32)
    np.save(os.path.join(args.outdir, "fixations.npy"), fix_arr)
    print(f"wrote fixations.npy ({len(fix_arr)} moments)", flush=True)

    # scanpath figure: color camera frames + gaze trail + color percepts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def grab(src, t_s):
        cmd = ["ffmpeg", "-v", "error", "-ss", str(t_s), "-i", src,
               "-frames:v", "1", "-vf", f"scale={w}:{h},format=rgb24",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        out = subprocess.run(cmd, capture_output=True).stdout
        return np.frombuffer(out, dtype=np.uint8).reshape(h, w, 3) / 255.0

    sample_t = np.linspace(total * 0.1, total * 0.9, 4)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for j, st in enumerate(sample_t):
        fr = grab(args.video, st)
        ax = axes[0][j]
        ax.imshow(fr, vmin=0, vmax=1)
        trail = [(x * sx, y * sy) for (tt, x, y) in scanpath if tt <= st * 1000]
        if len(trail) > 1:
            xs, ys = zip(*trail)
            ax.plot(xs, ys, "c-", lw=1.2, alpha=0.8)
            ax.plot(xs[:-1], ys[:-1], "co", ms=3, alpha=0.8)
        ax.plot(trail[-1][0], trail[-1][1], "r+", ms=10, mew=1.5)
        ax.set_title(f"camera t={st:.0f}s + gaze")
        ax.axis("off")
        p = grab(out_mp4, round(st * moment_hz) / moment_hz)
        axes[1][j].imshow(p, vmin=0, vmax=1)
        axes[1][j].set_title(f"color percept t={st:.0f}s")
        axes[1][j].axis("off")
    fig.suptitle("Star Tours, full wiring: salience saccades at 3.5 Hz + "
                 "foveated color (sharp luma, color-blind periphery)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "scanpath.png"), dpi=110)
    plt.close(fig)
    print("wrote scanpath.png", flush=True)

    report = {
        "video": os.path.basename(args.video),
        "source_meta": meta,
        "processed_seconds": total,
        "working_res": [w, h],
        "n_saccades": n_sac,
        "frames_pushed": n_in,
        "moments_pulled": n_moments,
        "wall_clock_s": round(wall_total, 1),
        "color": True,
        "saccades": True,
        "color_attention": True,
        "moment_ms": moment_ms,
        "moment_hz": moment_hz,
        "chroma_weight": (A.CHROMA_WEIGHT if args.chroma_weight is None
                          else args.chroma_weight),
        "integration_window_ms": B.INTEGRATION_WINDOW_MS,
    }
    with open(os.path.join(args.outdir, "run_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote run_report.json", flush=True)


if __name__ == "__main__":
    main()
