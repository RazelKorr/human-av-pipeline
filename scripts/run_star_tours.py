"""Run the human auditory pipeline on the Star Tours 0-62s audio.

Produces: cochleagram, salience map, attention trace, event list.
Usage: python3 scripts/run_star_tours.py [--no-dwell] [--outdir DIR]
  --no-dwell : disable the quiet-dwell rule (A/B comparison).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hva import cochlea as C, salience as S, moments as M
from hva import attention as AT
from hva import battery as B  # for run_pipeline pieces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dwell", action="store_true")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    tag = "nodwell" if args.no_dwell else "dwell"
    outdir = args.outdir or os.path.join(
        os.path.dirname(__file__), "..", "output", "star_tours_62s", tag)
    os.makedirs(outdir, exist_ok=True)

    wav = os.path.join(os.path.dirname(__file__), "..", "input",
                       "star_tours_0-62s_16k.wav")
    print("loading", wav)
    import wave
    with wave.open(wav, "rb") as w:
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(
            np.float32) / 32768.0
    print("samples:", len(x), "dur_s:", round(len(x) / 16000.0, 1))

    if args.no_dwell:
        # A/B: allow the scheduled path to chase salience even in quiet
        AT.SILENCE_LOUD = -1.0

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
    trace = att.run(moms, Tprof)

    # --- save arrays -------------------------------------------------
    np.save(os.path.join(outdir, "cochleagram.npy"), Sg)
    np.save(os.path.join(outdir, "salience.npy"), feat["sal"])
    np.save(os.path.join(outdir, "freqs_hz.npy"), f)
    np.save(os.path.join(outdir, "times_s.npy"), t)

    # --- event list --------------------------------------------------
    events = [(round(m["t0"], 2), m["event"], round(m["cf_bin"], 1),
               round(C.bin_to_hz(int(np.clip(round(m["cf_bin"]), 0, 63))), 0))
              for m in trace if m["event"] not in (None,)]
    with open(os.path.join(outdir, "events.txt"), "w") as fh:
        fh.write("t_s event cf_bin cf_hz\n")
        for e in events:
            fh.write("%7.2f %-14s %5.1f %7.0f\n" % e)
    print("events:", len(events))
    from collections import Counter
    print(Counter(m["event"] for m in trace if m["event"]))

    # --- trace summary ------------------------------------------------
    with open(os.path.join(outdir, "trace.txt"), "w") as fh:
        fh.write("t0 cf_bin cf_hz loud event\n")
        for m in trace:
            hz = C.bin_to_hz(int(np.clip(round(m["cf_bin"]), 0, 63)))
            fh.write("%6.2f %6.1f %7.0f %.2f %s\n" % (
                m["t0"], m["cf_bin"], hz, m["loud"], m["event"]))
    print("moments:", len(trace))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
