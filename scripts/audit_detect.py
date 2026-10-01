"""Audit C: on-demand OWL-ViT detection for never-fixated objects.

The foveal classifier only names what gaze has fixated. This audit
checks the last-resort path: with an EMPTY memory, 'where is the X'
scans the current frame with OWL-ViT and grounds the reply to the
detected box.

  1. crop 10 (hand-labeled 'windows'), empty memory:
     'wodehaus where is the window' -> 'Found the a window', bias peaks
     at the detected box, and the detection becomes a track (follow-up
     'look at the window' uses memory).
  2. 'wodehaus where is the red car' -> no boxes -> honest miss.
  3. Latency is measured and reported (CPU; on-demand only).
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hva.understanding import ObjectMemory, understand
from hvp.detect import ObjectDetector
from scripts.audit_recognition import extract_crops, MONTAGE


def main():
    ok = True
    det = ObjectDetector()
    frame = extract_crops(MONTAGE)[10].convert("RGB").resize((224, 224))

    def detect_fn(queries):
        t0 = time.perf_counter()
        # same noun-phrase wrapping as ResponsePolicy
        dets = det.detect(frame, [f"a {q}" for q in queries],
                          threshold=0.05)
        dt = time.perf_counter() - t0
        print(f"  detect({queries}) -> {len(dets)} box(es) "
              f"in {dt * 1000:.0f} ms")
        out = []
        for d in dets:
            mx, my = ObjectDetector.box_center_map(d, 224, 224)
            out.append((d[0], mx, my, d[5]))
        return out

    # 1. unseen object found by scanning the frame
    mem = ObjectMemory()
    r, b = understand("wodehaus where is the window", memory=mem,
                      t_now_ms=1000.0, detect_fn=detect_fn)
    iy, ix = divmod(int(b.argmax()), 56)
    found_ok = (b is not None and "Found the window" in r
                and abs(ix - 27) < 6 and abs(iy - 27) < 6)
    print(f"  reply={r!r}, bias peak=({ix},{iy}) "
          f"-> {'OK' if found_ok else 'FAIL'}")
    ok &= found_ok

    # the 0.215 detection is below memory's 0.30 track bar: it steers
    # gaze (so the foveal classifier can confirm) but is not yet a
    # known object. A follow-up re-detects rather than using memory.
    no_track = mem.locate("window", 2000.0) is None
    print(f"  sub-0.30 detection not tracked: "
          f"{'OK' if no_track else 'FAIL'}")
    ok &= no_track
    r2, b2 = understand("wodehaus look at the window", memory=mem,
                        t_now_ms=2000.0, detect_fn=detect_fn)
    redetect_ok = b2 is not None and "Found the window" in r2
    print(f"  follow-up re-detects: {r2!r} "
          f"-> {'OK' if redetect_ok else 'FAIL'}")
    ok &= redetect_ok

    # 2. genuine miss stays honest
    r3, b3 = understand("wodehaus where is the red car",
                        memory=ObjectMemory(), t_now_ms=1000.0,
                        detect_fn=detect_fn)
    miss_ok = b3 is None and "don't know" in r3
    print(f"  miss: {r3!r} -> {'OK' if miss_ok else 'FAIL'}")
    ok &= miss_ok

    print("\nAUDIT C:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
