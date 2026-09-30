"""Top-down attention demo: recognition contextualizes the next saccades.

Pass 1 (bottom-up only) produced fixations; originally a human reader
(the "cortex") labeled 12 foveal crops. Now the loop classifies the
foveal crop live with CLIP (hvp/recognize.py); --recognizer manual
reproduces the old hand-labeled behavior for comparison.

  brightness captures -> fovea identifies -> identity sets the search.

Label -> computable bias (all on the 56x56 salience map):
  gate        -> seek_opening:    darkness near the POI (find the way through)
  gate-edge   -> seek_opening (weaker)
  dark        -> move_on:         suppress the neighborhood (nothing here)
  light-strip -> follow_row:      brightness along the horizontal band
  distant-light -> approach_light: brightness near the POI
  sign        -> seek_referent:   brightness near the POI (what is it for?)
  lit-floor   -> look_up:         brightness above the POI
  windows     -> follow_row

Context becomes available after recognition (live: classified once per
fixation, ~310 ms on CPU) and decays over ~2.5 s -- it contextualizes
the next several saccades.
"""

import json
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
from hvp.recognize import (DARK_STREET_VOCAB, FovealClassifier,
                           foveal_crop_pil)
from hvp.saccades import SaccadeController

OUT = os.path.join(ROOT, "output")
VSIZE = 224


def _bias(label, small, px, py):
    S = (A.SMALL, A.SMALL)
    bright = A._norm(small)
    dark = A._norm(1.0 - small)
    yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
    if label in ("gate", "gate-edge"):
        w = 1.2 if label == "gate" else 0.8
        return w * dark * A._blob(S, px, py, 14.0), "seek opening"
    if label == "dark":
        return -1.0 * A._blob(S, px, py, 10.0), "move on"
    if label in ("light-strip", "windows"):
        band = np.exp(-((yy - py) ** 2) / (2 * 6.0 ** 2))
        return 1.0 * bright * band, "follow the row"
    if label == "distant-light":
        return 1.2 * bright * A._blob(S, px, py, 10.0), "approach light"
    if label == "sign":
        return 1.2 * bright * A._blob(S, px, py, 14.0), "seek referent"
    if label == "lit-floor":
        return 0.8 * bright * (yy < py).astype(np.float32), "look up"
    return np.zeros(S, np.float32), "none"


def manual_recognizer(labels, dva):
    """Adapter reproducing the hand-labeled behavior.

    recognize(t, fx, fy, frame) -> (label, conf, lx, ly, age) | None.
    Fires when the eyes are within 5 deg of a labeled location, 150 ms
    to 4 s after the label's timestamp.
    """
    def recognize(t, fx, fy, frame):
        best = None
        for L in labels:
            d = np.hypot(fx - L["x"], fy - L["y"]) * dva
            age = t - L["t"]
            if d < 5.0 and 150.0 < age < 4000.0:
                if best is None or age < best[0]:
                    best = (age, L)
        if best is None:
            return None
        age, L = best
        return L["label"], 1.0, L["x"], L["y"], age
    return recognize


def live_recognizer(classifier, vocab, threshold=0.30, move_deg=2.0, dva=None):
    """Adapter classifying the foveal crop at each new fixation.

    Classifies once per fixation (reuses the label while the eyes stay
    within move_deg); returns None below threshold ("unknown" is honest).
    """
    vocab_pairs = [(name, prompt) for name, prompt in vocab.items()]
    state = {"x": None, "y": None, "t": None,
             "label": None, "conf": 0.0}

    def recognize(t, fx, fy, frame):
        st = state
        moved = (st["x"] is None or
                 np.hypot(fx - st["x"], fy - st["y"]) * dva > move_deg)
        if moved:
            crop = foveal_crop_pil(frame, fx, fy)
            label, conf = classifier.classify(crop, vocab_pairs)
            st.update(x=fx, y=fy, t=t, label=label, conf=conf)
        if st["conf"] < threshold:
            return None
        return st["label"], st["conf"], st["x"], st["y"], t - st["t"]
    return recognize


def drive(frames, fps, recognize, use_topdown):
    dt = 1000.0 / fps
    dva = B.FIELD_WIDTH_DEG / VSIZE
    t_end = frames[-1][0]
    c = SaccadeController([(0, VSIZE / 2, VSIZE / 2)], dva)
    inhib, pois = [], []
    trans = np.zeros((A.SMALL, A.SMALL), dtype=np.float32)
    prev = None
    decay = np.exp(-dt / A.TRANS_TAU_MS)
    next_decision = 100.0
    scanpath = [(0.0, VSIZE / 2, VSIZE / 2)]
    events = []  # (t, label, conf, what) when a context fires

    for t, f in frames:
        small = A._downsample(f)
        if prev is not None:
            trans = np.maximum(trans * decay, np.abs(small - prev))
        prev = small
        if t >= next_decision:
            sal_base = A.salience_map(small, trans, t, c, inhib)
            fx0, fy0, _ = c.state_at(t)
            yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
            dist_deg = np.hypot(xx - fx0 / A.SCALE,
                                yy - fy0 / A.SCALE) * A.SCALE * dva
            sal = sal_base + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
            for (px, py, pt, ps) in pois:
                age = t - pt
                if age < 12000.0:
                    sal = sal + ps * np.exp(-age / 6000.0) * A._blob(
                        (A.SMALL, A.SMALL), px, py, 12.0)
            # --- top-down: recognition contextualizes the next saccades ---
            if use_topdown:
                fx, fy, _ = c.state_at(t)
                rec = recognize(t, fx, fy, f)
                if rec is not None:
                    label, conf, lx, ly, age = rec
                    if age < 4000.0:
                        bmap, what = _bias(label, small,
                                           lx / A.SCALE, ly / A.SCALE)
                        sal = sal + conf * bmap * np.exp(-age / 1500.0)
                        events.append((t, label, conf, what))
            iy, ix = np.unravel_index(int(np.argmax(sal)), sal.shape)
            t_on = t + B.SACCADE_LATENCY_MS
            if t_on < t_end:
                c.add_saccade(t_on, (ix + 0.5) * A.SCALE, (iy + 0.5) * A.SCALE)
                scanpath.append((t_on, (ix + 0.5) * A.SCALE,
                                 (iy + 0.5) * A.SCALE))
                pois.append((float(ix), float(iy), t,
                             float(np.clip(sal_base[iy, ix] / 2, 0.2, 1.5))))
                pois = [p for p in pois if t - p[2] < 12000.0][-10:]
            inhib = [(x, y, it) for (x, y, it) in inhib if t - it < 1500.0]
            inhib.append((ix, iy, t))
            next_decision = t + 1000.0 / B.SACCADE_RATE_HZ
    return scanpath, events


def find_labels():
    """foveal_labels.json: local output first, then the vision repo."""
    for p in (os.path.join(OUT, "foveal_labels.json"),
              os.path.expanduser("~/workspace/human-vision-pipeline/"
                                  "output/foveal_labels.json")):
        if os.path.exists(p):
            return p
    raise SystemExit("foveal_labels.json not found "
                     "(needed for --recognizer manual)")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--recognizer", choices=["live", "manual"],
                    default="live",
                    help="live: CLIP classifies each foveal crop; "
                         "manual: the 12 hand labels (comparison)")
    args = ap.parse_args()

    meta = run_video.probe(args.video)
    frames = [(t, f) for t, f in
              run_video.decode_gray(args.video, args.seconds,
                                    meta["fps"], VSIZE, VSIZE)]
    dva = B.FIELD_WIDTH_DEG / VSIZE
    if args.recognizer == "manual":
        labels = json.load(open(find_labels()))
        recognize = manual_recognizer(labels, dva)
        print(f"{len(frames)} frames, {len(labels)} labeled fixations",
              flush=True)
    else:
        recognize = live_recognizer(FovealClassifier(), DARK_STREET_VOCAB,
                                    dva=dva)
        print(f"{len(frames)} frames, live CLIP recognizer", flush=True)

    sp_bottom, _ = drive(frames, meta["fps"], recognize, use_topdown=False)
    sp_top, events = drive(frames, meta["fps"], recognize, use_topdown=True)
    print(f"bottom-up: {len(sp_bottom)} fixations, "
          f"top-down: {len(sp_top)} fixations", flush=True)
    print(f"context fired {len(events)} times:", flush=True)
    seen = set()
    for t, lbl, conf, what in events:
        key = (lbl, what)
        if key not in seen:
            seen.add(key)
            print(f"  t={t:.0f}ms: '{lbl}' (conf {conf:.2f}) -> {what}",
                  flush=True)

    # divergence: fraction of top-down fixations >5 deg from any
    # bottom-up fixation near the same time
    dva = B.FIELD_WIDTH_DEG / VSIZE
    div = 0
    for t, x, y in sp_top[1:]:
        near = [(tt, xx, yy) for (tt, xx, yy) in sp_bottom
                if abs(tt - t) < 1500.0]
        if near:
            dmin = min(np.hypot(x - xx, y - yy) * dva for _, xx, yy in near)
            if dmin > 5.0:
                div += 1
    print(f"{div}/{len(sp_top)-1} top-down fixations diverged >5 deg "
          f"from the bottom-up path", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sample_t = np.linspace(4000, frames[-1][0] - 1000, 4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for j, st in enumerate(sample_t):
        fr = min(frames, key=lambda tf: abs(tf[0] - st))[1]
        ax = axes[j]
        ax.imshow(fr, cmap="gray", vmin=0, vmax=1)
        for sp, col, lab in ((sp_bottom, "cyan", "bottom-up only"),
                             (sp_top, "magenta", "+ top-down")):
            trail = [(x, y) for (tt, x, y) in sp if tt <= st]
            if len(trail) > 1:
                xs, ys = zip(*trail)
                ax.plot(xs, ys, "-", color=col, lw=1.0, alpha=0.7,
                        label=lab if j == 0 else None)
        ax.set_title(f"t={st:.0f}ms")
        ax.axis("off")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Same eyes, different minds: brightness-only gaze (cyan) "
                 "vs gaze re-biased by what was recognized (magenta)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "video_topdown.png"), dpi=110)
    plt.close(fig)
    print("wrote output/video_topdown.png", flush=True)


if __name__ == "__main__":
    main()
