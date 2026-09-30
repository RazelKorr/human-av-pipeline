"""Closed-loop attention: Wodehaus as the visual cortex.

The eyes are salience-driven (brightness, transients, POI local search).
The cortex is the operator: at each segment boundary the driver dumps
foveal crops at its fixations; the operator reads them and writes a
context file -- per-fixation label, search directive, and a bias spec
in a small computable vocabulary:

  {"kind": "darkness"|"brightness",
   "shape": "blob"|"band"|"upper"|"full",
   "sigma": float, "weight": float}

At the next segment's saccade decisions, active contexts (150 ms after
their fixation landed, decaying over ~2.5 s) re-bias the salience map.
Recognition -> directive -> gaze. The loop is closed through the reader.

Usage:
  python3 closed_loop.py VIDEO --t0 0 --t1 10000 --state-in DIR --state-out DIR --crops DIR --context JSON_or_none
State dirs carry the driver across segments (saccade script, inhibition,
POIs, transient map). Crops go to DIR; context JSON is written by hand.
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
from hvp.saccades import SaccadeController, darkness_gain

OUT = os.path.join(ROOT, "output", "closed_loop")


def load_state(d):
    st = {"script": [], "inhib": [], "pois": [], "trans": None,
          "prev": None, "next_decision": 100.0, "scanpath": [],
          "decisions": []}
    if d and os.path.exists(os.path.join(d, "state.json")):
        meta = json.load(open(os.path.join(d, "state.json")))
        st.update({k: meta[k] for k in
                   ("script", "inhib", "pois", "next_decision",
                    "scanpath", "decisions") if k in meta})
        st["trans"] = np.load(os.path.join(d, "trans.npy"))
        st["prev"] = np.load(os.path.join(d, "prev.npy"))
    return st


def save_state(st, d):
    os.makedirs(d, exist_ok=True)
    json.dump({k: st[k] for k in
               ("script", "inhib", "pois", "next_decision",
                "scanpath", "decisions")},
              open(os.path.join(d, "state.json"), "w"))
    np.save(os.path.join(d, "trans.npy"), st["trans"])
    np.save(os.path.join(d, "prev.npy"), st["prev"])


def context_bias(spec, small, px, py):
    kind = spec["kind"]
    shape = spec["shape"]
    base = A._norm(small if kind == "brightness" else 1.0 - small)
    S = (A.SMALL, A.SMALL)
    yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
    if shape == "blob":
        m = A._blob(S, px, py, spec.get("sigma", 12.0))
    elif shape == "band":
        m = np.exp(-((yy - py) ** 2) / (2 * spec.get("sigma", 6.0) ** 2))
    elif shape == "upper":
        m = (yy < py).astype(np.float32)
    else:
        m = np.ones(S, np.float32)
    return spec.get("weight", 1.0) * base * m


def run_segment(frames, fps, st, contexts, dark_gate=True):
    dt = 1000.0 / fps
    h, w = frames[0][1].shape
    # Square pixels: one degrees-per-pixel scale serves both axes.
    dva = B.FIELD_WIDTH_DEG / w
    sx, sy = w / A.SMALL, h / A.SMALL  # frame px per salience-grid cell
    t_end = frames[-1][0]
    c = SaccadeController([(0, w / 2, h / 2)], dva)
    for (t_on, x, y) in st["script"]:
        c.add_saccade(t_on, x, y)
    inhib = [tuple(i) for i in st["inhib"]]
    pois = [tuple(p) for p in st["pois"]]
    trans = (st["trans"] if st["trans"] is not None
             else np.zeros((A.SMALL, A.SMALL), np.float32))
    prev = st["prev"]
    decay = np.exp(-dt / A.TRANS_TAU_MS)
    next_decision = st["next_decision"]
    scanpath = [tuple(s) for s in st["scanpath"]] or [(0.0, w / 2.0, h / 2.0)]
    decisions = [list(d) for d in st["decisions"]]
    log = []

    for t_np, f in frames:
        t = float(t_np)
        small = A._downsample(f)
        lum = float(small.mean())
        if prev is not None:
            trans = np.maximum(trans * decay, np.abs(small - prev))
        prev = small
        if t >= next_decision:
            sal_base = A.salience_map(small, trans, t, c, inhib, sx, sy)
            fx0, fy0, _ = c.state_at(t)
            yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
            dx = (xx - fx0 / sx) * sx
            dy = (yy - fy0 / sy) * sy
            dist_deg = np.hypot(dx, dy) * dva
            sal = sal_base + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
            for (px, py, pt, ps) in pois:
                age = t - pt
                if age < 12000.0:
                    sal = sal + ps * np.exp(-age / 6000.0) * A._blob(
                        (A.SMALL, A.SMALL), px, py, 12.0)
            # --- top-down: the cortex fires when the eyes land where it
            # has left a directive. Recognition -> directive -> gaze.
            for ctx in contexts:
                d = np.hypot(fx0 - ctx["x"], fy0 - ctx["y"]) * dva
                last = ctx.get("_fired", -1e9)
                if d < 5.0 and t - last > 3000.0:
                    ctx["_fired"] = float(t)
                    log.append((t, ctx["label"], ctx["directive"]))
            for ctx in contexts:
                t_land = ctx.get("_fired")
                if t_land is not None:
                    age = t - t_land
                    if 150.0 < age < 2500.0:
                        sal = sal + context_bias(
                            ctx["bias"], small,
                            ctx["x"] / sx, ctx["y"] / sy
                        ) * np.exp(-age / 1500.0)
            iy, ix = np.unravel_index(int(np.argmax(sal)), sal.shape)
            t_on = float(t + B.SACCADE_LATENCY_MS)
            tx, ty = float((ix + 0.5) * sx), float((iy + 0.5) * sy)
            if t_on < t_end:
                c.add_saccade(t_on, tx, ty)
                scanpath.append((t_on, tx, ty))
                st["script"].append((t_on, tx, ty))
                pois.append((float(ix), float(iy), t,
                             float(np.clip(sal_base[iy, ix] / 2, 0.2, 1.5))))
                pois = [p for p in pois if t - p[2] < 12000.0][-10:]
            inhib = [(x, y, it) for (x, y, it) in inhib if t - it < 1500.0]
            inhib.append((int(ix), int(iy), t))
            dec_ms = (1000.0 / B.SACCADE_RATE_HZ *
                      (darkness_gain(lum) if dark_gate else 1.0))
            decisions.append([t, lum, dec_ms])
            next_decision = t + dec_ms

    st["inhib"] = [list(i) for i in inhib]
    st["pois"] = [list(p) for p in pois]
    st["trans"] = trans
    st["prev"] = prev
    st["next_decision"] = next_decision
    st["scanpath"] = [list(s) for s in scanpath]
    st["decisions"] = decisions[-20000:]
    return st, log


def dump_crops(frames, scanpath, t0, t1, d, every=8):
    os.makedirs(d, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = frames[0][1].shape
    in_seg = [(int(i), float(tt), float(x), float(y))
              for i, (tt, x, y) in enumerate(scanpath)
              if t0 <= tt < t1][::every]
    crops = []
    for n, (i, tt, x, y) in enumerate(in_seg):
        fr = min(frames, key=lambda tf: abs(tf[0] - (tt + 60)))[1]
        r = 48
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r)
        crop = fr[y0:y1, x0:x1]
        pad = np.zeros((2 * r, 2 * r), np.float32)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        big = np.kron(pad, np.ones((2, 2), np.float32))
        crops.append({"n": n, "fix": i, "t": tt, "x": x, "y": y})
        plt.imsave(os.path.join(d, f"crop_{n:02d}.png"), big, cmap="gray",
                   vmin=0, vmax=1)
    n = len(crops)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes)
    for cr in crops:
        ax = axes[cr["n"] // cols][cr["n"] % cols]
        ax.imshow(plt.imread(os.path.join(d, f"crop_{cr['n']:02d}.png")),
                  cmap="gray")
        ax.set_title(f"#{cr['n']} fix={cr['fix']} t={cr['t']:.0f}ms")
        ax.axis("off")
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"Foveal crops t={t0:.0f}-{t1:.0f}ms -- cortex reading")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "sheet.png"), dpi=110)
    plt.close(fig)
    json.dump(crops, open(os.path.join(d, "crops.json"), "w"), indent=1)
    print(f"wrote {n} crops to {d}", flush=True)
    return crops


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=10000.0)
    ap.add_argument("--state-in", default=None)
    ap.add_argument("--state-out", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--context", default=None,
                    help="JSON list written by the cortex")
    ap.add_argument("--width", type=int, default=None,
                    help="working width px; default = native video width")
    ap.add_argument("--height", type=int, default=None,
                    help="working height px; default = native video height")
    ap.add_argument("--no-dark-gate", action="store_true",
                    help="disable the luminance gate on saccade rate "
                         "(baseline for A/B comparison)")
    ap.add_argument("--metrics-out", default=None,
                    help="write per-decision [t_ms, lum, interval_ms] JSON")
    args = ap.parse_args()

    meta = run_video.probe(args.video)
    dur = (args.t1 - args.t0) / 1000.0
    allf = [(t, f) for t, f in run_video.decode_gray(
        args.video, dur, meta["fps"], args.width, args.height,
        t0=args.t0 / 1000.0)]
    frames = allf
    st = load_state(args.state_in)
    n0 = len(st["decisions"])
    contexts = json.load(open(args.context)) if args.context else []
    st, log = run_segment(frames, meta["fps"], st, contexts,
                          dark_gate=not args.no_dark_gate)
    save_state(st, args.state_out)
    if args.metrics_out:
        json.dump(st["decisions"][n0:],
                  open(args.metrics_out, "w"))
    crops = dump_crops(allf, [tuple(s) for s in st["scanpath"]],
                       args.t0, args.t1, args.crops)
    seen = set()
    for t, lbl, dirc in log:
        if (lbl, dirc) not in seen:
            seen.add((lbl, dirc))
            print(f"  t={t:.0f}ms [{lbl}] {dirc}", flush=True)
    print(f"segment {args.t0:.0f}-{args.t1:.0f}ms: "
          f"{len(st['scanpath'])} fixations total", flush=True)


if __name__ == "__main__":
    main()
