"""Watch a real video the way the model predicts a human does: salience-driven saccades.

First pass: an attention driver (static center-surround + transient channel
+ inhibition of return, from hvp.attention) watches the decoded frames and
picks saccade targets at ~3.5 Hz with 200 ms latency, building a fixation
script. Second pass: VisionPipeline renders the 10 Hz foveated percept
stream along that scanpath.

Usage: python3 scripts/run_video_saccades.py <video.mp4> [--seconds N]
                                                          [--fovea DEG]

Outputs:
  output/video_saccade_percept.mp4 -- percept stream (10 fps moments)
  output/video_scanpath.png        -- frames with gaze trail + percepts
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_video
from hvp import attention as A
from hvp import baseline as B
from hvp.pipeline import VisionPipeline
from hvp.saccades import SaccadeController

OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

VSIZE = 224  # decode size; matches attention.SIZE so salience math lines up


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fovea", type=float, default=7.0,
                    help="fovea radius in degrees (default 7: the wide tool "
                         "setting; use 1.0 for the human baseline)")
    args = ap.parse_args()

    meta = run_video.probe(args.video)
    fps = meta["fps"]
    dt = 1000.0 / fps
    print(f"video: {meta['w']}x{meta['h']} @ {fps:.2f} fps", flush=True)

    frames = [(t, f) for t, f in
              run_video.decode_gray(args.video, args.seconds, fps,
                                    VSIZE, VSIZE)]
    t_end = frames[-1][0]
    print(f"decoded {len(frames)} frames", flush=True)

    # ---- first pass: attention driver builds the fixation script ----
    dva = B.FIELD_WIDTH_DEG / VSIZE
    cx = cy = VSIZE / 2.0
    controller = SaccadeController([(0, cx, cy)], dva)
    inhib = []
    scanpath = [(0.0, cx, cy)]
    pois = []  # points of interest: (x, y, t_noted, strength) in small-px
    trans = np.zeros((A.SMALL, A.SMALL), dtype=np.float32)
    prev_small = None
    decay = np.exp(-dt / A.TRANS_TAU_MS)
    next_decision = 100.0

    for t, f in frames:
        small = A._downsample(f)
        if prev_small is not None:
            trans = np.maximum(trans * decay, np.abs(small - prev_small))
        prev_small = small
        if t >= next_decision:
            sal_base = A.salience_map(small, trans, t, controller, inhib)
            # Human amplitude prior: saccades land ~5-15 deg out, not
            # anywhere on the field. Additive bonus keeps inhibited
            # (negative) locations ordered correctly.
            fx0, fy0, _ = controller.state_at(t)
            yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
            dist_deg = np.hypot(xx - fx0 / A.SCALE,
                                yy - fy0 / A.SCALE) * A.SCALE * dva
            sal = sal_base + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
            # Local search: recent points of interest pull the next
            # several saccades toward their neighborhoods. Inhibition
            # of return still covers the exact visited spots, so the
            # net effect is exploration *around* POIs, not re-fixation.
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
                # Note the point of interest, weighted by how strongly
                # it won the salience competition.
                strength = float(np.clip(sal_base[iy, ix] / 2.0, 0.2, 1.5))
                pois.append((float(ix), float(iy), t, strength))
                pois = [(x, y, it, s) for (x, y, it, s) in pois
                        if t - it < 12000.0][-10:]
            inhib = [(x, y, it) for (x, y, it) in inhib if t - it < 1500.0]
            inhib.append((ix, iy, t))
            next_decision = t + 1000.0 / B.SACCADE_RATE_HZ

    print(f"driver made {len(scanpath) - 1} saccades", flush=True)

    # ---- second pass: render the percept stream along the scanpath ----
    pipe = VisionPipeline((VSIZE, VSIZE), dva, controller.script,
                          fovea_radius_deg=args.fovea)
    for t, f in frames:
        pipe.push(f, t)
    moments = list(pipe.moments(t_end))
    print(f"pulled {len(moments)} perceptual moments", flush=True)
    run_video.write_mp4([p for _, p, _ in moments],
                        os.path.join(OUT, "video_saccade_percept.mp4"))
    print("wrote output/video_saccade_percept.mp4", flush=True)

    # ---- figure: gaze trail over frames + the percept at those times ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_t = np.linspace(4000, t_end - 1000, 4)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for j, st in enumerate(sample_t):
        fr = min(frames, key=lambda tf: abs(tf[0] - st))[1]
        ax = axes[0][j]
        ax.imshow(fr, cmap="gray", vmin=0, vmax=1)
        trail = [(x, y) for (tt, x, y) in scanpath if tt <= st]
        if len(trail) > 1:
            xs, ys = zip(*trail)
            ax.plot(xs, ys, "c-", lw=1.2, alpha=0.8)
            ax.plot(xs[:-1], ys[:-1], "co", ms=3, alpha=0.8)
        ax.plot(trail[-1][0], trail[-1][1], "r+", ms=10, mew=1.5)
        ax.set_title(f"camera t={st:.0f}ms + gaze")
        ax.axis("off")

        mt, p, m = min(moments, key=lambda mp: abs(mp[0] - st))
        axes[1][j].imshow(p, cmap="gray", vmin=0, vmax=1)
        axes[1][j].set_title(f"percept t={mt:.0f}ms"
                             + (" (saccade hold)" if m["suppressed"] else ""))
        axes[1][j].axis("off")
    fig.suptitle("Star Tours, watched by the attention driver: salience "
                 "saccades at 3.5 Hz + POI local search, 200 ms latency, "
                 "fovea 7 deg")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "video_scanpath.png"), dpi=110)
    plt.close(fig)
    print("wrote output/video_scanpath.png", flush=True)


if __name__ == "__main__":
    main()
