"""Watch a video the way a human would, then review it like a speedster.

Ties all three systems together: the vision pipeline watches the frames,
the audio pipeline listens to the soundtrack, and the L1 fusion puts them
on one timeline. Out comes a plain-language perceptual review -- where
attention went, what grabbed it, what it missed -- produced in a fraction
of the clip's runtime. Sixty seconds of video, reviewed in seconds.

The bottlenecks are the point: 10 Hz moments, a fovea, an attentional
blink, inhibition of return. This doesn't see the video; it sees what a
human would have seen, which is less -- and that's what makes the review
human-shaped. (Roz works past his perceptual limits magically; Wodehaus
gets the standard ones and doesn't mind.)

Usage: python3 scripts/watch.py <video.mp4> [--seconds N] [--wav track.wav]
  --wav : use this 16 kHz mono wav instead of extracting audio from the video.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import fuse_av
from perceive import describe_pos
from hva import cochlea as C
from hva import salience as S
from hva import moments as M
from hva import attention as AT


def load_wav_16k_mono(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(
            np.float32) / 32768.0
    return x


def audio_trace_for_wav(wav_path):
    x = load_wav_16k_mono(wav_path)
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
    att = AT.AuditoryAttention(cf_init=32.0)
    return att.run(moms, Tprof)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--wav", default=None)
    args = ap.parse_args()
    t_start = time.time()

    seconds = args.seconds

    # ---- eyes ----
    print("eyes: building scanpath...", flush=True)
    scanpath, _, _ = fuse_av.build_visual_scanpath(args.video, seconds + 0.5)
    sacc_t = np.array([s[0] for s in scanpath])

    def gaze_at(t_ms):
        i = int(np.searchsorted(sacc_t, t_ms, side="right")) - 1
        i = max(0, i)
        return float(scanpath[i][1]), float(scanpath[i][2])

    # ---- ears ----
    print("ears: listening...", flush=True)
    tmp = None
    if args.wav:
        wav_path = args.wav
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        wav_path = tmp.name
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", args.video,
             "-ac", "1", "-ar", "16000", "-t", str(seconds), wav_path],
            check=True)
    try:
        trace = audio_trace_for_wav(wav_path)
    finally:
        if tmp is not None:
            os.unlink(wav_path)

    n = min(len(trace), int(seconds * 10))
    trace = trace[:n]

    # ---- joint moments ----
    joint = []
    for j in range(n):
        T_ms = j * 100.0
        gx, gy = gaze_at(T_ms + 50.0)
        sac = [s for s in scanpath if T_ms <= s[0] < T_ms + 100.0]
        a = trace[j]
        joint.append({"t_s": round(j * 0.1, 2), "gx": gx, "gy": gy,
                      "saccades": sac, "cf_hz": a["cf_hz"],
                      "loud": a["loud"], "event": a["event"]})
    elapsed = time.time() - t_start

    # ---- the review ----
    n_sac = len(scanpath) - 1
    n_cap = sum(1 for m in joint if m["event"] == "onset-capture")
    n_swi = sum(1 for m in joint if m["event"] == "switch")
    sac_rate = n_sac / seconds
    busy = "restless" if sac_rate > 3.0 else "steady" if sac_rate > 1.5 else "calm"

    name = os.path.basename(args.video)
    print()
    print(f"WATCH REPORT: {name} ({seconds:.0f}s of video, "
          f"watched in {elapsed:.1f}s -- {seconds / max(elapsed, 1e-9):.0f}x "
          f"realtime)")
    print(f"The shape of it: {busy} eyes ({n_sac} saccades), "
          f"{n_cap} audio captures, {n_swi} scheduled audio switches.")
    print()

    # Standout multisensory moments: a capture landing near a saccade.
    # Coincidence, not proven binding -- no shuffled baseline here.
    standout = []
    for m in joint:
        if m["event"] != "onset-capture":
            continue
        tc = m["t_s"]
        near = [s for s in scanpath if abs(s[0] / 1000.0 - tc) <= 0.3]
        if near:
            s = near[0]
            standout.append((m["loud"], tc, m["cf_hz"],
                             describe_pos(m["gx"], m["gy"]),
                             describe_pos(s[1], s[2])))
    standout.sort(reverse=True)
    print("Standout moments (ear and eye coincided):")
    if standout:
        for (loud, tc, hz, greg, sreg) in standout[:5]:
            print(f"  {tc:5.1f}s  sharp sound at {hz:5.0f} Hz while looking "
                  f"{greg} -- eye jumped to {sreg}")
    else:
        print("  none -- the senses never landed on the same instant.")
    print()

    # Where the eyes lived.
    regions = {}
    for m in joint:
        r = describe_pos(m["gx"], m["gy"])
        regions[r] = regions.get(r, 0) + 1
    top = sorted(regions.items(), key=lambda kv: -kv[1])[:3]
    print("Where the eyes lived: "
          + "; ".join(f"{r} ({c * 0.1:.0f}s)" for r, c in top) + ".")
    neglected = min(regions, key=regions.get)
    print(f"Likely missed: {neglected} -- the eyes barely went there.")
    print()

    # The audio story.
    cf_vals = [m["cf_hz"] for m in joint]
    dom = float(np.median(cf_vals))
    # Longest stretch without an event (dwelling).
    best, cur, cur0 = (0, 0, 0), 0, 0
    for j, m in enumerate(joint):
        if m["event"] in (None, "dwell-quiet"):
            if cur == 0:
                cur0 = j
            cur += 1
        else:
            if cur > best[0]:
                best = (cur, cur0)
            cur = 0
    if cur > best[0]:
        best = (cur, cur0)
    print(f"The audio story: attention centered around {dom:.0f} Hz, "
          f"captured {n_cap} times by sharp onsets; longest uninterrupted "
          f"dwell {best[0] * 0.1:.1f}s starting at {best[1] * 0.1:.1f}s.")


if __name__ == "__main__":
    main()
