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
       python3 scripts/run_level3.py --video V --mono M --stereo S \
           --transcript output/transcripts/ep.json --seconds N
The --transcript JSON (from scripts/transcribe.py) enables the
speech-gated joint run: a third closed loop where the auditory map
write is scaled by smoothed speech presence.
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
from hva import transcribe as TR
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
        ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(n),
                            dtype=np.int16).astype(np.float32) / 32768.0
        if ch == 2:  # fold stereo down to mono
            raw = raw.reshape(-1, 2).mean(axis=1)
        return raw


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=62.0)
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--mono", default=MONO)
    ap.add_argument("--stereo", default=STEREO,
                    help="44.1kHz stereo wav for ILD pan; falls back to "
                         "--mono when omitted (centered audio)")
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--transcript", default=None,
                    help="transcript JSON from scripts/transcribe.py; "
                         "enables the speech-gated joint run")
    args = ap.parse_args()
    seconds = args.seconds
    outdir = args.outdir
    stereo_path = args.stereo or args.mono
    os.makedirs(outdir, exist_ok=True)

    # ---- vision: per-moment salience (sensory part only) ----
    print("vision: decoding...", flush=True)
    frames = [(t, f) for t, f in
              run_video.decode_gray(args.video, seconds, 10.0, 224, 224)]
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
    x = load_mono_16k(args.mono)
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
    with wave.open(stereo_path, "rb") as w:
        n = w.getnframes()
        ch = w.getnchannels()
        sr = w.getframerate()
        raw = np.frombuffer(w.readframes(n),
                            dtype=np.int16).astype(np.float32) / 32768.0
    import math as _math
    g = _math.gcd(16000, sr)
    up, down = 16000 // g, sr // g
    if ch == 2:
        # stereo at sr Hz -> 16k per channel
        L = resample_poly(raw[0::2], up, down)
        R = resample_poly(raw[1::2], up, down)
    else:
        # mono fallback: resample to 16k, centered pan
        L = resample_poly(raw, up, down)
        R = L.copy()
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

    # ---- speech gating: the voice gets a louder vote ----
    speech = None
    joint_sp = None
    seg_text = None
    if args.transcript:
        import json as _json
        with open(args.transcript) as f:
            tr = _json.load(f)
        segments = tr["segments"] if isinstance(tr, dict) else tr
        print("speech: gating auditory write from transcript...",
              flush=True)
        speech = TR.speech_presence(segments, n_mom)
        joint_sp = run_closed_loop(
            stim, n_mom / 10.0, dva,
            speech_fn=lambda m: float(speech[m]))
        seg_text = [(s["start"], s["text"]) for s in segments]
        print(f"speech: {100 * (speech > 0.5).mean():.1f}% of moments "
              f"speech-active", flush=True)

    jp = np.array([p[:2] for p in joint["peaks"]])
    vp = np.array([p[:2] for p in visonly["peaks"]])
    disp = np.hypot(jp[:, 0] - vp[:, 0], jp[:, 1] - vp[:, 1]) * (224.0 / SIZE)
    moved = disp > 8.0
    top3 = np.argsort(disp)[-3:][::-1]

    print()
    print(f"Level 3 report ({n_mom / 10.0:.0f}s, {n_mom} moments)")
    print(f"  joint-map saccades: {len(joint['scanpath']) - 1}; "
          f"vision-only saccades: {len(visonly['scanpath']) - 1}")
    # saccade landings pair by index (same clock, same times)
    js = np.array(joint["scanpath"])
    vs = np.array(visonly["scanpath"])
    n_sac = min(len(js), len(vs))
    land = np.hypot(js[:n_sac, 1] - vs[:n_sac, 1],
                    js[:n_sac, 2] - vs[:n_sac, 2])
    print(f"  saccade landing displacement: mean {land.mean():.1f}px, "
          f"max {land.max():.1f}px; "
          f"{(land > 16).sum()} of {n_sac} landings moved >16px")
    print(f"  audio moved the peak >8px in {moved.sum()} moments "
          f"({100 * moved.mean():.1f}%)")
    print(f"  mean displacement: {disp.mean():.1f}px; "
          f"max {disp.max():.1f}px at t={top3[0] * 0.1:.1f}s")
    print("  top-3 audio-moved moments:")
    for m in top3:
        print(f"    t={m * 0.1:5.1f}s  displaced {disp[m]:5.1f}px "
              f"(joint peak map-px {jp[m, 0]:.0f},{jp[m, 1]:.0f})")

    if joint_sp is not None:
        # What did the speech gate change, over and above audio?
        # Compare the gated joint map against the ungated joint map:
        # same stimuli, same clock -- only the auditory gain differs.
        sp = np.array([p[:2] for p in joint_sp["peaks"]])
        gdisp = (np.hypot(sp[:, 0] - jp[:, 0], sp[:, 1] - jp[:, 1])
                 * (224.0 / SIZE))
        gmoved = gdisp > 8.0
        gtop = np.argsort(gdisp)[-3:][::-1]
        print()
        print(f"  speech gating moved the peak >8px in {gmoved.sum()} "
              f"moments ({100 * gmoved.mean():.1f}%)")
        print(f"  mean gated displacement: {gdisp.mean():.1f}px; "
              f"max {gdisp.max():.1f}px at t={gtop[0] * 0.1:.1f}s")
        print("  top-3 speech-gated moments (with transcript):")
        for m in gtop:
            t = m * 0.1
            said = next((txt for (st, txt) in seg_text
                         if st <= t < st + 8.0), "")
            print(f"    t={t:5.1f}s  moved {gdisp[m]:5.1f}px "
                  f"speech~{speech[m]:.2f}  \"{said[:72]}\"")
        # The read path also runs off the gated map: recompute gains
        # from the gated peaks so the auditory attention comparison
        # below reflects what the ears actually heard.
        gated_for_read = joint_sp
    else:
        gated_for_read = joint

    # ---- read path: map-gained auditory attention ----
    # The map doesn't just rescale the transient profile -- the gained
    # profile has to drive the attention module's *decisions*. The
    # onset detector thresholds on m["Tmax"] and the switcher argmaxes
    # m["sal"], so feeding the raw moments with a scaled Tprof changes
    # nothing (2026-09-30: captures came out 26/26 identical). Build
    # gained moment dicts: same timing/loudness, spatially amplified
    # salience and transient peaks.
    probe = JointPriorityMap()
    gains = np.ones_like(Tprof)
    for m in range(n_mom):
        probe.map = gated_for_read["maps"][m]
        gains[m] = probe.gains_for_pans(pan_bin[m])
    moms_gain = []
    for k, m in enumerate(moms[:n_mom]):
        g = dict(m)
        g["sal"] = (m["sal"] * gains[k]).astype(np.float32)
        g["Tmax"] = float((Tprof[k] * gains[k]).max())
        moms_gain.append(g)
    att = AT.AuditoryAttention(cf_init=32.0)
    tr_base = att.run(moms[:n_mom], Tprof)
    att2 = AT.AuditoryAttention(cf_init=32.0)
    tr_gain = att2.run(moms_gain, Tprof * gains)
    cap = lambda tr: sum(1 for e in tr if e["event"] == "onset-capture")
    swi = lambda tr: sum(1 for e in tr if e["event"] == "switch")
    n_cb, n_cg = cap(tr_base), cap(tr_gain)
    n_sb, n_sg = swi(tr_base), swi(tr_gain)
    print(f"  audio captures: baseline {n_cb}, "
          f"map-gained {n_cg}; "
          f"switches: {n_sb} vs {n_sg}")
    div = [k for k in range(n_mom)
           if tr_base[k]["event"] != tr_gain[k]["event"]]
    if div:
        print(f"  first divergences at " +
              ", ".join(f"{moms[d]['t0']:.1f}s" for d in div[:5]))

    # ---- save ----
    evcode = {"onset-capture": 1, "switch": 2, "refractory": 3,
              "dwell-quiet": 4}
    enc = lambda tr: np.array([evcode.get(e["event"], 0) for e in tr],
                              dtype=np.int8)
    tgt = lambda tr: np.array([e["target"] if e["target"] is not None else -1
                               for e in tr], dtype=np.float32)
    np.savez(os.path.join(outdir, "joint_maps.npz"),
             maps=np.stack(joint["maps"][::1]),
             peaks_v=np.array(joint["peaks"]),
             peaks_visonly=np.array(visonly["peaks"]),
             peaks_gated=(np.array(joint_sp["peaks"]) if joint_sp
                          is not None else np.array([])),
             speech=(speech if speech is not None else np.array([])),
             disp=disp,
             gains=gains,
             pan_bin=pan_bin,
             Tprof=Tprof,
             events_base=enc(tr_base),
             events_gain=enc(tr_gain),
             cf_base=np.array([e["cf_bin"] for e in tr_base]),
             cf_gain=np.array([e["cf_bin"] for e in tr_gain]),
             tgt_base=tgt(tr_base),
             tgt_gain=tgt(tr_gain))
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
        fig.savefig(os.path.join(outdir, "level3.png"), dpi=90)
        print(f"  saved {outdir}/")
    except ImportError:
        print("  (matplotlib missing; skipped plot)")


if __name__ == "__main__":
    main()
