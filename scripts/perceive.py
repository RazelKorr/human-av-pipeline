"""Look at an image the way a human would: salience-driven saccades.

The prosthesis entry point. Hand it a picture, get back a plain-language
report of where human-like attention goes, in order, and what it probably
never really sees. People are bad at enumerating their own perceptual
experience -- this enumerates it for them.

A still image is treated as a static scene watched for --seconds. The
attention driver (the same one as run_video_saccades.py) picks saccade
targets at ~3.5 Hz with 200 ms latency; inhibition of return keeps it
exploring instead of re-fixating. Nothing moves, so the transient channel
stays silent -- every saccade here is earned by brightness and contrast.

Usage: python3 scripts/perceive.py <image> [--seconds 5] [--fovea 7.0]
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from hvp import attention as A
from hvp import baseline as B
from hvp.saccades import SaccadeController
from scipy.ndimage import gaussian_filter

VSIZE = 224  # matches attention.SIZE so salience math lines up
FPS = 10.0


def describe_pos(x, y, size=VSIZE):
    col = "left" if x < size / 3 else "center" if x < 2 * size / 3 else "right"
    row = "upper" if y < size / 3 else "middle" if y < 2 * size / 3 else "lower"
    if col == "center" and row == "middle":
        return "center"
    return f"{row} {col}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--fovea", type=float, default=7.0,
                    help="fovea radius in degrees")
    args = ap.parse_args()

    img = Image.open(args.image).convert("L").resize((VSIZE, VSIZE))
    frame = np.asarray(img, dtype=np.float32) / 255.0

    dt = 1000.0 / FPS
    n_frames = int(args.seconds * FPS)
    t_end = n_frames * dt

    # ---- attention driver (same machinery as run_video_saccades.py) ----
    dva = B.FIELD_WIDTH_DEG / VSIZE
    cx = cy = VSIZE / 2.0
    controller = SaccadeController([(0, cx, cy)], dva)
    inhib = []
    fixations = [(0.0, cx, cy, "starting fixation")]
    pois = []
    trans = np.zeros((A.SMALL, A.SMALL), dtype=np.float32)
    prev_small = None
    decay = np.exp(-dt / A.TRANS_TAU_MS)
    next_decision = 100.0

    for i in range(n_frames):
        t = i * dt
        small = A._downsample(frame)
        if prev_small is not None:
            trans = np.maximum(trans * decay, np.abs(small - prev_small))
        prev_small = small
        if t >= next_decision:
            sal_base = A.salience_map(small, trans, t, controller, inhib)
            fx0, fy0, _ = controller.state_at(t)
            yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
            dist_deg = np.hypot(xx - fx0 / A.SCALE,
                                yy - fy0 / A.SCALE) * A.SCALE * dva
            sal = sal_base + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
            poi_pull = np.zeros_like(sal)
            for (px, py, pt, ps) in pois:
                age = t - pt
                if age < 12000.0:
                    pull = (ps * np.exp(-age / 6000.0)
                            * A._blob((A.SMALL, A.SMALL), px, py, 12.0))
                    poi_pull += pull
                    sal = sal + pull
            iy, ix = np.unravel_index(int(np.argmax(sal)), sal.shape)
            tx, ty = (ix + 0.5) * A.SCALE, (iy + 0.5) * A.SCALE
            # Why did this target win? Static brightness/contrast vs
            # pull from a previously noted point of interest.
            static_here = float(A._norm(
                np.abs(small - gaussian_filter(small, 6)))[iy, ix])
            if poi_pull[iy, ix] > 0.3 * sal[iy, ix]:
                why = "exploring around something noted a moment ago"
            elif static_here > 0.5:
                why = "drawn to a bright, high-contrast detail"
            else:
                why = "best of a quiet neighborhood"
            t_on = t + B.SACCADE_LATENCY_MS
            if t_on < t_end:
                controller.add_saccade(t_on, tx, ty)
                fixations.append((t_on / 1000.0, tx, ty, why))
                strength = float(np.clip(sal_base[iy, ix] / 2.0, 0.2, 1.5))
                pois.append((float(ix), float(iy), t, strength))
                pois = [(x, y, it, s) for (x, y, it, s) in pois
                        if t - it < 12000.0][-10:]
            inhib = [(x, y, it) for (x, y, it) in inhib if t - it < 1500.0]
            inhib.append((ix, iy, t))
            next_decision = t + 1000.0 / B.SACCADE_RATE_HZ

    # ---- coverage: what ever fell inside the fovea? ----
    fovea_px = args.fovea / dva
    gy, gx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
    covered = np.zeros((A.SMALL, A.SMALL), dtype=bool)
    for (_, fx, fy, _) in fixations:
        d = np.hypot((gx + 0.5) * A.SCALE - fx,
                     (gy + 0.5) * A.SCALE - fy)
        covered |= d < fovea_px
    frac = covered.mean()
    # Least-covered third of the scene, for the "likely missed" line.
    thirds = {}
    for name, (xs, ys) in {
            "upper": ((0, 56), (0, 19)), "middle": ((0, 56), (19, 37)),
            "lower": ((0, 56), (37, 56))}.items():
        thirds[name] = covered[ys[0]:ys[1], xs[0]:xs[1]].mean()
    neglected = min(thirds, key=thirds.get)

    # ---- the report ----
    name = os.path.basename(args.image)
    print(f"Perceptual report: {name}")
    print(f"(static scene, watched {args.seconds:.1f}s the way a human would; "
          f"nothing moves, so every saccade is brightness/contrast earning it)")
    print()
    print(f"Fixations ({len(fixations)}):")
    prev = None
    for (ts, fx, fy, why) in fixations:
        label = describe_pos(fx, fy)
        glance = ""
        if prev is not None:
            dx, dy = fx - prev[0], fy - prev[1]
            if max(abs(dx), abs(dy)) > 25 and label == describe_pos(*prev):
                # Same coarse cell, but the eyes did move: say where.
                parts = []
                if dy < -25:
                    parts.append("up")
                elif dy > 25:
                    parts.append("down")
                if dx < -25:
                    parts.append("left")
                elif dx > 25:
                    parts.append("right")
                glance = f" (a glance {'-'.join(parts)} inside it)"
        print(f"  {ts:5.1f}s  {label:12s}{glance} -- {why}")
        prev = (fx, fy)
    print()
    print(f"Coverage: {100 * frac:.0f}% of the scene fell inside the fovea "
          f"at some point.")
    if thirds[neglected] < 0.3:
        print(f"Likely missed: the {neglected} band stayed peripheral the "
              f"whole time -- nothing there ever won the salience "
              f"competition.")


if __name__ == "__main__":
    main()
