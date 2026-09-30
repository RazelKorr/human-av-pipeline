"""Audit D: anaphora over tracks.

Replays the live-run sightings (same stream as audits A/B) into an
ObjectMemory, then checks pronoun and 'the left one' resolution:

  'look at the windows' ; 'look at it again' -> same bias peak
  'look at the left one' ; 'the right one'   -> the two window tracks
  'look at it' after 'the right one'         -> stays on the right track
  'look at it' with no prior region          -> honest miss, no bias
  'look at that gate'                        -> the gate track
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hva.understanding import DialogueState, ObjectMemory, understand

LOG = "/tmp/ground4/out2/conversation.log"
LINE = re.compile(
    r"\[t=\s*([\d.]+)s\] RECOGNIZED '([^']+)' \(([\d.]+)\) at \((\d+),(\d+)\)")


def replay():
    mem = ObjectMemory()
    for line in open(LOG):
        m = LINE.search(line)
        if not m:
            continue
        t_s, label, conf, fx, fy = (m.group(1), m.group(2), m.group(3),
                                   m.group(4), m.group(5))
        mem.add(label, int(fx) / 4.0, int(fy) / 4.0,
                t_ms=float(t_s) * 1000.0, conf=float(conf))
    return mem


def peak_x(bias):
    return divmod(int(bias.argmax()), 56)[1]


def main():
    mem = replay()
    ok = True
    t = 40_000.0

    wtracks = sorted(mem.locate_all("windows", t), key=lambda tr: tr.x)
    print(f"windows tracks at x={[round(tr.x, 1) for tr in wtracks]}")
    if len(wtracks) < 2:
        print("SKIP: only one windows track; cannot test left/right")
        sys.exit(0)
    lx, rx = round(wtracks[0].x), round(wtracks[-1].x)

    d = DialogueState()
    _, b1 = understand("wodehaus look at the windows", memory=mem,
                       t_now_ms=t, dialogue=d)
    r2, b2 = understand("wodehaus look at it again", memory=mem,
                        t_now_ms=t + 1000, dialogue=d)
    again_ok = (b2 is not None and int(b1.argmax()) == int(b2.argmax())
                and "windows" in r2)
    print(f"  'it again' same peak ({peak_x(b2)}): "
          f"{'OK' if again_ok else 'FAIL'}")
    ok &= again_ok

    r3, b3 = understand("wodehaus look at the left one", memory=mem,
                        t_now_ms=t + 2000, dialogue=d)
    left_ok = b3 is not None and abs(peak_x(b3) - lx) <= 2
    print(f"  'the left one' peak x={peak_x(b3)} (track {lx}): "
          f"{'OK' if left_ok else 'FAIL'}")
    ok &= left_ok

    r4, b4 = understand("wodehaus look at the right one", memory=mem,
                        t_now_ms=t + 3000, dialogue=d)
    right_ok = b4 is not None and abs(peak_x(b4) - rx) <= 2
    print(f"  'the right one' peak x={peak_x(b4)} (track {rx}): "
          f"{'OK' if right_ok else 'FAIL'}")
    ok &= right_ok

    r5, b5 = understand("wodehaus look at it", memory=mem,
                        t_now_ms=t + 4000, dialogue=d)
    chain_ok = b5 is not None and abs(peak_x(b5) - rx) <= 2
    print(f"  'it' chains off right-one, peak x={peak_x(b5)}: "
          f"{'OK' if chain_ok else 'FAIL'}")
    ok &= chain_ok

    d2 = DialogueState()
    r6, b6 = understand("wodehaus look at it", memory=mem,
                        t_now_ms=t + 5000, dialogue=d2)
    miss_ok = b6 is None and "'it' refers to" in r6
    print(f"  bare 'it' -> honest miss: {'OK' if miss_ok else 'FAIL'}")
    ok &= miss_ok

    r7, b7 = understand("wodehaus look at that gate", memory=mem,
                        t_now_ms=t + 6000, dialogue=d2)
    det_ok = b7 is not None and "Looking at the gate" in r7
    print(f"  'that gate' -> gate track: {'OK' if det_ok else 'FAIL'}")
    ok &= det_ok

    print(f"\nAUDIT D: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
