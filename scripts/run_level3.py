"""Level 3 on real media: the joint priority map watches Star Tours.

Per 100 ms moment:
  - vision writes retinotopic salience (static + transient) to the map
  - audio writes per-bin transient salience, splatted at each bin's
    ILD-derived pan from the stereo track
A second, vision-only map (w_aud=0) runs in parallel so we can measure
what the audio actually contributed: how far it moved the peak, and when.

Then the read paths:
  - saccades are driven from the joint map (closed loop)
  - auditory attention runs on transients gained by the map
    (baseline: ungained), and capture counts are compared.

Outputs (output/level3/, gitignored):
  joint_maps.npz, peak_tracks.npz, level3.png, report printed to stdout.

Usage: python3 scripts/run_level3.py [--seconds 62]
"""

import os
import sys
import wave

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_video
from hvp import attention as A
from hvp import baseline as B
from hva import cochlea as C
from hva import salience as S
from hva import moments as M
from hva import attention as AT
from hvm.driver import run_closed_loop
from hvm.priority import JointPriorityMap, SIZE

OUTDIR = os.path.join(ROOT, "output", "level3")
VIDEO = os.path.join(os.path.dirname(ROOT), "human-vision-pipeline",
                     "input", "star_tours_1_ride_film.mp4")
MONO = os.path.join(ROOT, "input", "star_tours_0-62s_16k.wav")
STEREO = os.path.join(ROOT, "input", "star_tours_0-62s_44k_stereo.wav")


def load_mono_16k(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        return np.frombuffer(w.readframes(n),
                             dtype=np.int16).astype(np.float32) / 32768.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=62.0)
    args = ap.parse_args()
    seconds = args.seconds
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- vision: per-moment salience (sensory part only) ----
    print("vision: decoding...", flush=True)
    frames = [(t, f) for t, f in
              run_video.decode_gray(VIDEO, seconds, 10.0, 224, 224)]
    n_mom = min(int(seconds * 10), len(frames))
    vis_sal = np.zeros((n_mom, SIZE, SIZE), dtype=np.float32)
    trans = np.zeros((SIZE, SIZE), dtype=np.float32)
    prev = None
    decay = np.exp(-100.0 / A.TRANS_TAU_MS)
    for m in range(n_mom):
        small = A._downsample(frames[m][1])
        if prev is not None:
            trans = np.maximum(trans * decay, np.abs(small - prev))
        prev = small
        static = A._norm(np.abs(small - gaussian_filter(small, 6)))
        vis_sal[m] = A._norm(static + 1.5 * A._norm(trans))
    print(f"vision: {n_mom} moments", flush=True)

    # ---- audio: per-moment transient profile (mono) ----
    print("audio: mono transient profile...", flush=True)
    x = load_mono_16k(MONO)
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

    n_a = min(len(moms), n_mom)
    Tprof = np.stack([_tprof(i) for i in range(n_a)])
    vis_sal = vis_sal[:n_a]
    n_mom = n_a
    # robust clip normalization so one crash doesn't define the scale
    ceil = float(np.percentile(Tprof, 99))
    Tprof_n = np.clip(Tprof / max(ceil, 1e-9), 0, 1)

    # ---- audio: per-bin pan from stereo ILD ----
    print("audio: stereo ILD pans...", flush=True)
    with wave.open(STEREO, "rb") as w:
        n = w.getnframes()
        raw = np.frombuffer(w.readframes(n),
                            dtype=np.int16).astype(np.float32) / 32768.0
    L = resample_poly(raw[0::2], 160, 441)
    R = resample_poly(raw[1::2], 160, 441)
    _, _, S_l = C.stft_log(L)
    _, _, S_r = C.stft_log(R)
    pan_bin = np.zeros((n_mom, C.N_BINS), dtype=np.float32)
    for m in range(n_mom):
        sl = slice(m * 20, (m + 1) * 20)
        ild = (S_l[sl] - S_r[sl]).mean(axis=0)  # dB; + = left louder
        pan_bin[m] = -np.clip(ild / 12.0, -1.0, 1.0)  # -1 = left

    # ---- closed loop: joint map vs vision-only map ----
    print("level 3: closed loop...", flush=True)
    dva = B.FIELD_WIDTH_DEG / 224.0

    def stim(m):
        return vis_sal[m], (Tprof_n[m], pan_bin[m])

    def stim_vis(m):
        return vis_sal[m], None

    joint = run_closed_loop(stim, n_mom / 10.0, dva)
    visonly = run_closed_loop(stim_vis, n_mom / 10.0, dva)

    jp = np.array([p[:2] for p in joint["peaks"]])
    vp = np.array([p[:2] for p in visonly["peaks"]])
    disp = np.hypot(jp[:, 0] - vp[:, 0], jp[:, 1] - vp[:, 1]) * (224.0 / SIZE)
    moved = disp > 8.0
    top3 = np.argsort(disp)[-3:][::-1]

    print()
    print(f"Level 3 report ({n_mom / 10.0:.0f}s, {n_mom} moments)")
    print(f"  joint-map saccades: {len(joint['scanpath']) - 1}; "
          f"vision-only saccades: {len(visonly['scanpath']) - 1}")
    print(f"  audio moved the peak >8px in {moved.sum()} moments "
          f"({100 * moved.mean():.1f}%)")
    print(f"  mean displacement: {disp.mean():.1f}px; "
          f"max {disp.max():.1f}px at t={top3[0] * 0.1:.1f}s")
    print("  top-3 audio-moved moments:")
    for m in top3:
        print(f"    t={m * 0.1:5.1f}s  displaced {disp[m]:5.1f}px "
              f"(joint peak map-px {jp[m, 0]:.0f},{jp[m, 1]:.0f})")

    # ---- read path: map-gained auditory attention ----
    probe = JointPriorityMap()
    gains = np.ones_like(Tprof)
    for m in range(n_mom):
        probe.map = joint["maps"][m]
        gains[m] = probe.gains_for_pans(pan_bin[m])
    att = AT.AuditoryAttention(cf_init=32.0)
    tr_base = att.run(moms[:n_mom], Tprof)
    att2 = AT.AuditoryAttention(cf_init=32.0)
    tr_gain = att2.run(moms[:n_mom], Tprof * gains)
    cap = lambda tr: sum(1 for e in tr if e["event"] == "onset-capture")
    swi = lambda tr: sum(1 for e in tr if e["event"] == "switch")
    print(f"  audio captures: baseline {cap(tr_base)}, "
          f"map-gained {cap(tr_gain)}; "
          f"switches: {swi(tr_base)} vs {swi(tr_gain)}")

    # ---- save ----
    np.savez(os.path.join(OUTDIR, "joint_maps.npz"),
             maps=np.stack(joint["maps"][::1]),
             peaks_v=np.array(joint["peaks"]),
             peaks_visonly=np.array(visonly["peaks"]),
             disp=disp)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ts = np.arange(n_mom) * 0.1
        ax[0].plot(ts, jp[:, 0] * 224 / SIZE, label="joint peak x")
        ax[0].plot(ts, vp[:, 0] * 224 / SIZE, label="vision-only peak x",
                   alpha=0.6)
        ax[0].set_ylabel("peak x (224-px)")
        ax[0].legend(fontsize=8)
        ax[1].plot(ts, disp, color="tab:red")
        ax[1].axhline(8, color="k", ls="--", lw=1)
        ax[1].set_ylabel("displacement (px)")
        ax[1].set_xlabel("t (s)")
        fig.suptitle("Level 3: what audio did to the priority map")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTDIR, "level3.png"), dpi=90)
        print("  saved output/level3/")
    except ImportError:
        print("  (matplotlib missing; skipped plot)")


if __name__ == "__main__":
    main()
