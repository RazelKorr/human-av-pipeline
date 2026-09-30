"""Level 2: bidirectional cross-modal coupling.

Round-trip:
  Pass 1: audio onsets (uncoupled) + per-onset pan from the stereo mix.
  Pass 2: visual driver with audio bias -- recent onsets tug saccades
          toward the panned side + a brief global alerting gain.
          Records per-moment visual transient strength.
  Pass 3: audio attention re-run with vis_boost -- visual transients
          lower the auditory capture bar.

Then compares coupled vs uncoupled: do saccades land closer to the
panned onset side? Does audio capture more where vision is busy?

Outputs (output/fused/):
  av_coupled.jsonl   -- joint feed from the coupled run
  couple_report.txt  -- comparison numbers
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from hva import cochlea as C
from hva import salience as S
from hva import moments as M
from hva import attention as AT
from hva import spatial as SP
from fuse_av import build_visual_scanpath

OUTDIR = os.path.join(ROOT, "output", "fused")
DWELL = os.path.join(ROOT, "output", "star_tours_62s", "dwell")


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- Pass 1: audio onsets + pan ----
    sr, l, r = SP.load_stereo(
        os.path.join(ROOT, "input", "star_tours_0-62s_44k_stereo.wav"))
    pan, conf = SP.moment_pan(l, r, sr)
    onsets = []  # (t_s, pan, conf)
    with open(os.path.join(DWELL, "trace.txt")) as fh:
        for line in fh.readlines()[1:]:
            p = line.split()
            if len(p) > 4 and p[4] == "onset-capture":
                t = float(p[0])
                m = min(int(t * 10), len(pan) - 1)
                onsets.append((t, float(pan[m]), float(conf[m])))
    print(f"Pass 1: {len(onsets)} audio onsets with pan")

    # ---- Pass 2: visual driver with audio bias ----
    video = os.path.join(ROOT, "input", "star_tours_1_ride_film.mp4")
    scanpath, controller, vis_trans = build_visual_scanpath(
        video, 62.5, audio_onsets=onsets)
    np.savez(os.path.join(OUTDIR, "visual_scanpath_coupled.npz"),
             scanpath=np.array(scanpath, dtype=np.float64),
             vis_trans=vis_trans)
    print(f"Pass 2: {len(scanpath)-1} saccades (coupled)")

    # ---- Pass 3: audio re-run with vis_boost ----
    Sg = np.load(os.path.join(DWELL, "cochleagram.npy"))
    sal = np.load(os.path.join(DWELL, "salience.npy"))
    t = np.load(os.path.join(DWELL, "times_s.npy"))
    moms = M.moments(Sg, sal, t)

    def _tprof(i):
        sl = Sg[i * 20:(i + 1) * 20]
        if len(sl) < 2:
            return np.zeros(C.N_BINS)
        d = np.abs(np.diff(sl, axis=0))
        d[np.maximum(sl[:-1], sl[1:]) < S.HEAR_FLOOR_DBFS] = 0.0
        return d.mean(axis=0)

    Tprof = np.stack([_tprof(i) for i in range(len(moms))])
    vb = vis_trans[:len(moms)]
    att = AT.AuditoryAttention(cf_init=32.0)
    trace = att.run(moms, Tprof, vis_boost=vb)
    n_cap = sum(1 for m in trace if m["event"] == "onset-capture")
    n_sw = sum(1 for m in trace if m["event"] == "switch")
    print(f"Pass 3: audio re-run -> {n_cap} captures, {n_sw} switches")

    # ---- joint coupled feed ----
    sacc_t = np.array([s[0] for s in scanpath])

    def gaze_at(t_ms):
        i = max(0, int(np.searchsorted(sacc_t, t_ms, side="right")) - 1)
        return float(scanpath[i][1]), float(scanpath[i][2])

    out_path = os.path.join(OUTDIR, "av_coupled.jsonl")
    with open(out_path, "w") as out:
        for j, m in enumerate(trace):
            T_ms = j * 100.0
            gx, gy = gaze_at(T_ms + 50.0)
            sac = [(round(float(s[0]) / 1000.0, 2),
                    round(float(s[1]), 1), round(float(s[2]), 1))
                   for s in scanpath if T_ms <= s[0] < T_ms + 100.0]
            entry = {
                "t_s": round(j * 0.1, 2),
                "gaze_x": round(gx, 1), "gaze_y": round(gy, 1),
                "saccades": sac,
                "audio_cf_bin": round(float(m["cf_bin"]), 1),
                "audio_cf_hz": round(float(C.bin_to_hz(
                    int(np.clip(round(m["cf_bin"]), 0, 63)))), 0),
                "audio_event": m["event"],
                "vis_trans": round(float(vb[j]), 3),
            }
            out.write(json.dumps(entry) + "\n")
    print("wrote", out_path)

    # ---- comparison: uncoupled vs coupled ----
    # (a) Do saccades land closer to the panned onset side?
    # For each onset, find the first saccade within 600ms after it and
    # measure its x vs the panned side (pan<0 -> expect x < 112).
    def landing_error(sp):
        errs = []
        st = np.array([s[0] for s in sp])
        for (ot, opan, oconf) in onsets:
            if oconf < 0.3:
                continue
            ot_ms = ot * 1000.0
            cand = [s for s in sp if ot_ms < s[0] <= ot_ms + 600.0]
            if not cand:
                continue
            x = cand[0][1]
            # expected side: pan -1 -> x~22, pan +1 -> x~202 (224 space)
            expect = (0.5 + opan * 0.4) * 224.0
            errs.append(abs(x - expect))
        return errs

    z0 = np.load(os.path.join(OUTDIR, "visual_scanpath_62s.npz"),
                 allow_pickle=True)
    sp0 = [tuple(s) for s in z0["scanpath"]]
    e0 = landing_error(sp0)
    e1 = landing_error(scanpath)
    # (b) audio capture counts
    cap0 = sw0 = 0
    with open(os.path.join(DWELL, "trace.txt")) as fh:
        for line in fh.readlines()[1:]:
            p = line.split()
            if len(p) > 4:
                if p[4] == "onset-capture":
                    cap0 += 1
                elif p[4] == "switch":
                    sw0 += 1

    rep = [
        "Level 2 coupling report (Star Tours 0-62s)",
        f"audio onsets driving vision: {len(onsets)}",
        "",
        "(a) saccade landing error vs panned onset side (px, 224 space):",
        f"  uncoupled: n={len(e0)} mean={np.mean(e0):.1f}",
        f"  coupled:   n={len(e1)} mean={np.mean(e1):.1f}",
        "",
        "(b) audio attention events:",
        f"  uncoupled: {cap0} captures, {sw0} switches",
        f"  coupled:   {n_cap} captures, {n_sw} switches",
    ]
    rep_path = os.path.join(OUTDIR, "couple_report.txt")
    with open(rep_path, "w") as fh:
        fh.write("\n".join(rep) + "\n")
    print("\n".join(rep))
    print("wrote", rep_path)


if __name__ == "__main__":
    main()
