"""Level 1 audio-visual fusion: a single synchronized feed.

Combines the human-vision scanpath with the human-audio attention trace
into one joint moment stream at 10 Hz, indexed by MEDIA time.

Timing note: both systems quantize to the 100ms grid, but visual
perception carries ~100ms of pipeline latency (phototransduction +
retina-to-cortex). The saccade script itself runs on media time, so gaze
positions pair directly with audio moments by media timestamp. If you
later pull the visual *percept frames* (the foveated content), remember
they trail their timestamp by ~2 moments -- content-aligned pairs are
visual[k] <-> audio[k-2]. This mirrors biology: ears are faster than
eyes, and the multisensory binding window absorbs the difference.

Output: output/fused/av_joint.jsonl -- one entry per 100ms of media time:
  {t_s, gaze_x, gaze_y, saccades, audio_cf_bin, audio_cf_hz, audio_event}

Usage: python3 scripts/fuse_av.py [--seconds 62] [--rebuild-scanpath]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from hva import cochlea as C

OUTDIR = os.path.join(ROOT, "output", "fused")
SCANPATH_NPZ = os.path.join(OUTDIR, "visual_scanpath_62s.npz")

# Visual content lags its timestamp by two moments (200ms of pipeline
# latency at a 100ms grain). Audio moment j covers [j*100,(j+1)*100)ms;
# visual moment k covers [(k-2)*100,(k-1)*100)ms. Content-aligned:
VIS_LAG_MOMENTS = 2


def build_visual_scanpath(video_path, seconds, audio_onsets=None):
    """First-pass attention driver (from run_video_saccades.py): builds the
    saccade script for the segment.

    audio_onsets: optional list of (t_s, pan, conf) for Level 2
    audio->vision coupling. A recent onset tugs gaze: a spatial blob on
    the panned side plus a brief global alerting gain (the "pip and pop"
    effect).

    Returns (scanpath, controller, vis_trans): vis_trans is per-100ms
    visual transient strength in [0,1] (Level 2 vision->audio coupling --
    a flash makes you listen harder)."""
    import run_video
    from hvp import attention as A
    from hvp import baseline as B
    from hvp.saccades import SaccadeController

    VSIZE = 224
    meta = run_video.probe(video_path)
    fps = meta["fps"]
    dt = 1000.0 / fps
    print(f"video: {meta['w']}x{meta['h']} @ {fps:.2f} fps", flush=True)
    frames = [(t, f) for t, f in
              run_video.decode_gray(video_path, seconds, fps, VSIZE, VSIZE)]
    t_end = frames[-1][0]
    print(f"decoded {len(frames)} frames", flush=True)

    dva = B.FIELD_WIDTH_DEG / VSIZE
    cx = cy = VSIZE / 2.0
    controller = SaccadeController([(0, cx, cy)], dva)
    inhib = []
    scanpath = [(0.0, cx, cy)]
    pois = []
    trans = np.zeros((A.SMALL, A.SMALL), dtype=np.float32)
    prev_small = None
    decay = np.exp(-dt / A.TRANS_TAU_MS)
    next_decision = 100.0
    # per-100ms visual transient strength (mean frame-difference)
    n_mom = int(seconds * 10) + 2
    vis_trans = np.zeros(n_mom)
    vis_count = np.zeros(n_mom)

    for t, f in frames:
        small = A._downsample(f)
        if prev_small is not None:
            trans = np.maximum(trans * decay, np.abs(small - prev_small))
        prev_small = small
        m = int(t // 100.0)
        if m < n_mom:
            vis_trans[m] += float(trans.mean())
            vis_count[m] += 1
        if t >= next_decision:
            sal_base = A.salience_map(small, trans, t, controller, inhib)
            fx0, fy0, _ = controller.state_at(t)
            yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
            dist_deg = np.hypot(xx - fx0 / A.SCALE,
                                yy - fy0 / A.SCALE) * A.SCALE * dva
            sal = sal_base + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
            for (px, py, pt, ps) in pois:
                age = t - pt
                if age < 12000.0:
                    sal = sal + (ps * np.exp(-age / 6000.0)
                                 * A._blob((A.SMALL, A.SMALL), px, py, 12.0))
            if audio_onsets:
                # Level 2, audio->vision: recent onsets tug gaze.
                for (ot, opan, oconf) in audio_onsets:
                    age = t - ot * 1000.0
                    if 0.0 <= age < 400.0:
                        w = oconf * np.exp(-age / 200.0)
                        # spatial: blob on the panned side
                        bx = (0.5 + opan * 0.4) * A.SMALL
                        sal = sal + 2.0 * w * A._blob(
                            (A.SMALL, A.SMALL), bx, A.SMALL / 2.0, 12.0)
                        # alerting: brief global gain
                        sal = sal * (1.0 + 0.3 * w)
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

    print(f"driver made {len(scanpath) - 1} saccades", flush=True)
    nz = vis_count > 0
    vis_trans[nz] /= vis_count[nz]
    # normalize to [0,1] by a robust ceiling (99th percentile)
    ceil = float(np.percentile(vis_trans[nz], 99)) if np.any(nz) else 1.0
    vis_trans = np.clip(vis_trans / max(ceil, 1e-9), 0, 1)
    return scanpath, controller, vis_trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=62.0)
    ap.add_argument("--rebuild-scanpath", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- visual scanpath (cached) ----
    if os.path.exists(SCANPATH_NPZ) and not args.rebuild_scanpath:
        z = np.load(SCANPATH_NPZ, allow_pickle=True)
        scanpath = [tuple(s) for s in z["scanpath"]]
        print(f"loaded cached scanpath ({len(scanpath)} saccades)")
    else:
        video = os.path.join(VIS_ROOT, "input",
                             "star_tours_1_ride_film.mp4")
        scanpath, _, _ = build_visual_scanpath(video, args.seconds + 0.5)
        np.savez(SCANPATH_NPZ,
                 scanpath=np.array(scanpath, dtype=np.float64))
        print(f"saved scanpath to {SCANPATH_NPZ}")

    # gaze lookup: for media time t_ms, the active fixation (x, y)
    sacc_t = np.array([s[0] for s in scanpath])
    sacc_x = np.array([s[1] for s in scanpath])
    sacc_y = np.array([s[2] for s in scanpath])

    def gaze_at(t_ms):
        i = int(np.searchsorted(sacc_t, t_ms, side="right")) - 1
        i = max(0, i)
        return float(sacc_x[i]), float(sacc_y[i])

    # ---- audio trace ----
    audio_trace = []
    with open(os.path.join(AUD_ROOT, "output", "star_tours_62s", "dwell",
                           "trace.txt")) as fh:
        for line in fh.readlines()[1:]:
            p = line.split()
            audio_trace.append({
                "t0": float(p[0]), "cf_bin": float(p[1]),
                "event": p[4] if len(p) > 4 else None})
    n_audio = len(audio_trace)
    print(f"audio moments: {n_audio}")

    # ---- joint feed ----
    # Media window [T, T+100)ms pairs audio moment T/100 with visual
    # content from the same window. Visual moment k covers
    # [(k-2)*100,(k-1)*100), so the gaze for window T comes from the
    # scanpath at media time T (the saccade script runs on media time).
    out_path = os.path.join(OUTDIR, "av_joint.jsonl")
    n_joint = 0
    with open(out_path, "w") as out:
        for j in range(n_audio):
            T_ms = j * 100.0
            gx, gy = gaze_at(T_ms + 50.0)  # mid-window fixation
            # saccades landing inside this media window
            sac = [(round(float(s[0]) / 1000.0, 2), round(float(s[1]), 1),
                    round(float(s[2]), 1))
                   for s in scanpath
                   if T_ms <= s[0] < T_ms + 100.0]
            a = audio_trace[j]
            cf_hz = float(C.bin_to_hz(
                int(np.clip(round(a["cf_bin"]), 0, 63))))
            entry = {
                "t_s": round(j * 0.1, 2),
                "gaze_x": round(gx, 1), "gaze_y": round(gy, 1),
                "gaze_space": "224x224",
                "saccades": sac,
                "audio_cf_bin": round(a["cf_bin"], 1),
                "audio_cf_hz": round(cf_hz, 0),
                "audio_event": a["event"],
            }
            out.write(json.dumps(entry) + "\n")
            n_joint += 1
    print(f"wrote {n_joint} joint moments to {out_path}")
    print("lag note: visual content trails audio by "
          f"{VIS_LAG_MOMENTS} moments; gaze is reported at media time.")


if __name__ == "__main__":
    main()
