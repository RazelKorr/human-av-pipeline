"""Audit A: sighting tracking across fixations.

Replays the RECOGNIZED events from the live conversation run
(/tmp/ground4/out2/conversation.log) into a fresh ObjectMemory and
checks:
  1. Nearby same-label sightings associate into tracks (fewer tracks
     than sightings for stable objects).
  2. locate() returns the track-smoothed position, not the last raw
     sighting (flicker suppression).
  3. locate_all() exposes multiple tracks of one label (enabler for
     spatial references, item B).
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hva.understanding import ObjectMemory

LOG = "/tmp/ground4/out2/conversation.log"
LINE = re.compile(
    r"\[t=\s*([\d.]+)s\] RECOGNIZED '([^']+)' \(([\d.]+)\) at \((\d+),(\d+)\)")


def main():
    mem = ObjectMemory()
    n = 0
    for line in open(LOG):
        m = LINE.search(line)
        if not m:
            continue
        t_s, label, conf, fx, fy = (m.group(1), m.group(2), m.group(3),
                                   m.group(4), m.group(5))
        mem.add(label, int(fx) / 4.0, int(fy) / 4.0,
                t_ms=float(t_s) * 1000.0, conf=float(conf))
        n += 1
    print(f"replayed {n} sightings -> {len(mem.tracks)} tracks")
    ok = True

    # 1. association happened (stable objects collapse)
    per_label: dict[str, int] = {}
    for tr in mem.tracks:
        per_label[tr.label] = per_label.get(tr.label, 0) + 1
    for label, ntr in sorted(per_label.items()):
        nsight = sum(1 for s in mem.sightings if s[0] == label)
        print(f"  {label:13s}: {nsight:3d} sightings -> {ntr} track(s)")
    if len(mem.tracks) >= n:
        print("FAIL: no association happened")
        ok = False

    # 2. flicker suppression: smoothed != last raw
    t_end = 40_000.0
    for label in ("windows", "light-strip"):
        raw = [s for s in mem.sightings if s[0] == label][-1]
        x, y, conf, age = mem.locate(label, t_end)
        if (x, y) == (raw[1], raw[2]):
            print(f"FAIL: {label} locate() == last raw sighting")
            ok = False
        else:
            print(f"  {label}: smoothed ({x:.1f},{y:.1f}) "
                  f"vs last raw ({raw[1]:.1f},{raw[2]:.1f})")

    # 3. multiple tracks of one label are addressable
    multi = [l for l, c in per_label.items() if c > 1]
    print(f"  labels with >1 track: {multi or 'none'}")
    for label in multi[:2]:
        trs = mem.locate_all(label, t_end)
        print(f"    {label}: {len(trs)} tracks, "
              f"hits={[t.hits for t in trs]}")

    print("\nAUDIT A:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
