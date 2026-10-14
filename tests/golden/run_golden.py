"""Golden-fixture regression pack (W3, commission 2, 2026-10-14).

Why this exists: the corpus's flagship numbers (onset counts,
transient counts, binding rates) have repeatedly moved silently
when a shared constant changed -- the PX.1 bar rescale moved
every filed onset count, and there was no small, fast check
that screamed at the moment it happened. This pack freezes
three tiny canonical cases -- the ear (moment_features +
pick_onsets), the eye (visual_transients), and the binder
(bind, shift null and all) -- against expected outputs, so any
drift in the flagship math fails loudly here, in seconds,
instead of surfacing months later as a corpus inconsistency.

The cases are synthetic and seeded (fixed seeds, in code):
regenerating them is deterministic. The expectations were
frozen with --bless at creation; re-blessing is a deliberate
act that belongs in the commit message that changes the math.

Usage:
  python tests/golden/run_golden.py           # check (exit 1 on diff)
  python tests/golden/run_golden.py --bless   # re-freeze expectations
Also wired into the suite as tests/test_golden.py, and behind
`sensorium golden`. (Loaded by file path: `tests` is not an
importable package in this repo.)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EXPECTED_DIR = os.path.join(HERE, "expected")

SEED = 20261014


def case_ear():
    """30 s synthetic waveform: seeded noise bed + 6 planted
    bursts -> moment features -> onset picks."""
    from hva.online import SPM, moment_features, pick_onsets
    rng = np.random.default_rng(SEED)
    sr = 16000
    n = sr * 30
    x = (rng.normal(0.0, 0.004, n)).astype(np.float32)
    for k, t0 in enumerate([2.0, 6.5, 11.0, 17.25, 23.0, 28.0]):
        i = int(t0 * sr)
        burst = (np.sin(2 * np.pi * (660 if k % 2 else 880)
                        * np.arange(sr // 4) / sr) * 0.25)
        env = np.hanning(len(burst)).astype(np.float32)
        x[i:i + len(burst)] += (burst * env).astype(np.float32)
    blocks = x[: (n // SPM) * SPM].reshape(-1, SPM)
    rms, flux, _cent, _ = moment_features(blocks)
    picks = pick_onsets(flux, rms)
    return {"n_onsets": len(picks),
            "onsets": [[round(t, 3), round(s, 4)] for t, s in picks]}


def case_eye():
    """The R1 miniature: stationary noisy energy bed, 10
    planted transients -> visual_transients."""
    from streaming.bind_av import visual_transients
    rng = np.random.default_rng(SEED)
    n = int(30.0 * 120.0)
    t = np.arange(n) / 30.0
    e = 1.0 + 0.1 * np.sin(2 * np.pi * t / 60.0)
    e = e + rng.normal(0.0, 0.05, n)
    for tp in [7.5 + 11.0 * k for k in range(10)]:
        i = int(round(tp * 30.0))
        e[i:i + 2] += 1.0
    vt = visual_transients(e, t * 1000.0)
    return {"n_transients": len(vt),
            "transients": [[round(ts, 3), round(s, 4)] for ts, s in vt]}


def case_binder():
    """Fixed literal streams -> bind() with its seeded shift
    null. Exercises the matcher, the summary, and the null in
    one frozen artifact."""
    from streaming.bind_av import bind
    onsets = [(1.0, 3.0), (2.05, 2.0), (4.5, 4.0), (7.0, 1.5),
              (9.95, 2.5), (13.0, 3.5), (16.5, 2.0), (20.0, 5.0)]
    vtrans = [(1.05, 2.0), (2.0, 1.0), (4.75, 3.0), (6.0, 1.0),
              (10.0, 2.0), (13.2, 1.5), (18.0, 1.0), (20.1, 4.0),
              (23.0, 1.0), (26.5, 2.0)]
    return bind(onsets, vtrans, duration_s=30.0)


CASES = {"ear": case_ear, "eye": case_eye, "binder": case_binder}


def _canon(obj):
    return json.dumps(obj, sort_keys=True, indent=1)


def check():
    """Run all cases, diff against frozen expectations.
    Returns a list of human-readable result lines; empty diff
    list means the pack is green."""
    lines, diffs = [], []
    for name, fn in CASES.items():
        got = fn()
        path = os.path.join(EXPECTED_DIR, f"{name}.json")
        if not os.path.exists(path):
            diffs.append(f"{name}: NO EXPECTATION FILE ({path})")
            lines.append(f"golden {name}: MISSING EXPECTATION")
            continue
        with open(path) as f:
            want = json.load(f)
        if _canon(got) == _canon(want):
            lines.append(f"golden {name}: MATCH")
        else:
            diffs.append(f"{name}: output differs from frozen "
                         f"expectation ({path})")
            lines.append(f"golden {name}: DIFF")
    return lines, diffs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--bless" in argv:
        for name, fn in CASES.items():
            path = os.path.join(EXPECTED_DIR, f"{name}.json")
            with open(path, "w") as f:
                json.dump(fn(), f, indent=1, sort_keys=True)
            print(f"golden {name}: blessed -> {path}")
        return 0
    lines, diffs = check()
    for line in lines:
        print(line)
    if diffs:
        for d in diffs:
            print(f"DIFF: {d}", file=sys.stderr)
        return 1
    print("golden pack: all cases match their frozen expectations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
