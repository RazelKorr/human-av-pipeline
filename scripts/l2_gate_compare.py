"""A/B compare the Level 2 vis_boost gate modes (audit item 13).

Builds L2 coupling inputs the way scripts/couple_av.py builds them
(batch-ear moments + Tprof from a 16 kHz mono wav; vis_boost from a
streaming run's per-frame transient energies, p99-normalized as in
scripts/fuse_av.py), then runs AuditoryAttention twice -- causal
vs legacy gate -- and reports where the traces diverge:

  - event-label diffs (count, % of moments)
  - onset-capture counts per mode
  - how many diffs sit where the shaped gate value actually moved
    (|delta shaped| >= 0.05): diffs should live there if the gate
    is the cause

Usage:
  python3 scripts/l2_gate_compare.py --wav input/drtran_ep4_16k.wav \
      --frame-energies output/drtran_ep4_av/frame_energies.npy
"""
import argparse
import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from hva import cochlea as C
from hva import salience as S
from hva import moments as M
from hva import attention as AT


def load_wav16(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        return np.frombuffer(w.readframes(n), dtype=np.int16
                             ).astype(np.float32) / 32768.0


def build_moments(wav_path):
    x = load_wav16(wav_path)
    t, f, Sg = C.stft_log(x)
    feat = S.salience_map(Sg)
    moms = M.moments(Sg, feat["sal"], t)

    def _tprof(i):
        sl = Sg[i * 20:(i + 1) * 20]
        if len(sl) < 2:
            return np.zeros(C.N_BINS)
        d = np.abs(np.diff(sl, axis=0))
        d[np.maximum(sl[:-1], sl[1:]) < S.HEAR_FLOOR_DBFS] = 0.0
        return d.mean(axis=0)

    Tprof = np.stack([_tprof(i) for i in range(len(moms))])
    return moms, Tprof


def build_vis_boost(frame_energies_path, n_moms):
    """Per-frame (t_ms, static, chroma, trans) -> per-100ms-moment
    transient strength, p99-normalized like fuse_av.py."""
    fe = np.load(frame_energies_path)
    t_ms, trans = fe[:, 0], fe[:, 3]
    vb = np.zeros(n_moms)
    cnt = np.zeros(n_moms)
    idx = np.clip((t_ms / 100.0).astype(int), 0, n_moms - 1)
    np.add.at(vb, idx, trans)
    np.add.at(cnt, idx, 1.0)
    nz = cnt > 0
    vb[nz] /= cnt[nz]
    ceil = float(np.percentile(vb[nz], 99)) if np.any(nz) else 1.0
    return np.clip(vb / max(ceil, 1e-9), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--frame-energies", required=True)
    ap.add_argument("--out", default=None,
                    help="optional JSON summary path")
    args = ap.parse_args()

    moms, Tprof = build_moments(args.wav)
    vb = build_vis_boost(args.frame_energies, len(moms))
    print(f"moments: {len(moms)}  vis_boost nonzero: "
          f"{int((vb > 0).sum())}")

    traces = {}
    shaped = {}
    for mode in ("legacy", "causal"):
        att = AT.AuditoryAttention(cf_init=32.0)
        traces[mode] = att.run(moms, Tprof, vis_boost=vb,
                               gate_mode=mode)
        shaped[mode] = AT.shaped_vis_boost(vb, mode)

    n = len(moms)
    diff_idx = [k for k in range(n)
                if traces["legacy"][k]["event"]
                != traces["causal"][k]["event"]
                or traces["legacy"][k]["cf_bin"]
                != traces["causal"][k]["cf_bin"]]
    event_idx = [k for k in range(n)
                 if traces["legacy"][k]["event"]
                 != traces["causal"][k]["event"]]
    dshaped = np.abs(shaped["causal"] - shaped["legacy"])
    on_gate = sum(1 for k in diff_idx if dshaped[k] >= 0.05)
    event_on_gate = sum(1 for k in event_idx if dshaped[k] >= 0.05)
    caps = {mode: sum(1 for m in traces[mode]
                      if m["event"] == "onset-capture")
            for mode in traces}
    switches = {mode: sum(1 for m in traces[mode]
                          if m["event"] == "switch")
                for mode in traces}
    summary = {
        "n_moments": n,
        "diff_moments": len(diff_idx),
        "diff_pct": round(100.0 * len(diff_idx) / n, 2),
        "diff_on_moved_gate": on_gate,
        "diff_on_moved_gate_frac": (round(on_gate / len(diff_idx), 3)
                                    if diff_idx else None),
        "event_diff_moments": len(event_idx),
        "event_diff_pct": round(100.0 * len(event_idx) / n, 2),
        "event_diff_on_moved_gate": event_on_gate,
        "event_diff_on_moved_gate_frac": (
            round(event_on_gate / len(event_idx), 3)
            if event_idx else None),
        "captures": caps,
        "switches": switches,
        "diff_first_t_s": (round(diff_idx[0] * 0.1, 1)
                           if diff_idx else None),
        "diff_last_t_s": (round(diff_idx[-1] * 0.1, 1)
                          if diff_idx else None),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
