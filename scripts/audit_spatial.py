"""Audit B: spatial referring expressions.

Replays the live-run sightings (same stream as audit A) into an
ObjectMemory, then checks that spatial commands ground to the right
tracks:

  'look at the leftmost windows'  -> bias peaks at the left window track
  'look at the windows on the right' -> right window track
  'look at the nearest gate' (gaze near gate) -> the gate track
  'look at the leftmost red car' -> honest miss, no bias
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hva.understanding import ObjectMemory, understand

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


def peak_xy(bias):
    iy, ix = divmod(int(bias.argmax()), 56)
    return ix, iy


def main():
    mem = replay()
    ok = True
    t = 40_000.0

    # windows has 2 tracks; leftmost/rightmost must split them
    wtracks = sorted(mem.locate_all("windows", t), key=lambda tr: tr.x)
    print(f"windows tracks at x={[round(tr.x, 1) for tr in wtracks]}")
    if len(wtracks) < 2:
        print("SKIP: only one windows track; cannot test left/right")
        sys.exit(0)

    r1, b1 = understand("wodehaus look at the leftmost windows",
                        memory=mem, t_now_ms=t)
    px, _ = peak_xy(b1)
    left_ok = px < 28 and "leftmost windows" in r1
    print(f"  leftmost: peak x={px}, reply={r1!r} -> {'OK' if left_ok else 'FAIL'}")
    ok &= left_ok

    r2, b2 = understand("wodehaus look at the windows on the right",
                        memory=mem, t_now_ms=t)
    px, _ = peak_xy(b2)
    # relative, not absolute: the righter of the two tracks
    right_ok = px == round(wtracks[-1].x) and "right windows" in r2
    print(f"  on the right: peak x={px} vs tracks "
          f"{[round(tr.x) for tr in wtracks]} -> {'OK' if right_ok else 'FAIL'}")
    ok &= right_ok

    # nearest gate with gaze near the gate track
    gt = mem.locate("gate", t)
    r3, b3 = understand("wodehaus look at the nearest gate",
                        memory=mem, t_now_ms=t,
                        gaze_xy=(gt[0], gt[1]))
    px, py = peak_xy(b3)
    near_ok = (abs(px - gt[0]) < 6 and abs(py - gt[1]) < 6
               and "nearest gate" in r3)
    print(f"  nearest gate: peak=({px},{py}) vs track=({gt[0]:.1f},{gt[1]:.1f}) "
          f"-> {'OK' if near_ok else 'FAIL'}")
    ok &= near_ok

    # honest miss preserved
    r4, b4 = understand("wodehaus look at the leftmost red car",
                        memory=mem, t_now_ms=t)
    miss_ok = b4 is None and "don't know" in r4
    print(f"  miss: reply={r4!r} -> {'OK' if miss_ok else 'FAIL'}")
    ok &= miss_ok

    print("\nAUDIT B:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
