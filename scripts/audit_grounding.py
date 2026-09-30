"""Audit 3: referring-expression grounding end to end.

Populates an ObjectMemory from REAL CLIP classifications of the 12
foveal crops (at their labeled positions), then checks that
"look at the X" steers the task bias to where X was seen -- and that
an unseen object gets the honest miss.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from audit_recognition import extract_crops, MONTAGE, LABELS
from hva.understanding import ObjectMemory, understand
from hvp.recognize import DARK_STREET_VOCAB, FovealClassifier


def main():
    manual = json.load(open(LABELS))
    crops = extract_crops(MONTAGE)
    vocab = [(n, p) for n, p in DARK_STREET_VOCAB.items()]
    clf = FovealClassifier()
    mem = ObjectMemory()
    print("sightings from live CLIP:")
    for i, crop in enumerate(crops):
        L = manual[i]
        label, conf = clf.classify(crop, vocab)
        mx, my = L["x"] / 4.0, L["y"] / 4.0
        mem.add(label, mx, my, t_ms=L["t"], conf=conf)
        print(f"  crop {i:2d} @{L['t']:.0f}ms -> '{label}' "
              f"conf {conf:.2f} (human said '{L['label']}')")
    print(f"\nknown: {mem.known_objects()}")

    ok = True

    # 1. grounded: "look at the windows" -> bias at a windows sighting
    reply, bias = understand("wodehaus look at the windows",
                             memory=mem, t_now_ms=30000.0)
    win = mem.locate("windows", 30000.0)
    if win is None:
        print("SKIP: no windows sighting to ground")
    else:
        iy, ix = divmod(bias.argmax(), 56)
        dx = abs(ix - win[0])
        print(f"\n'look at the windows' -> bias peak ({ix},{iy}), "
              f"sighting ({win[0]:.0f},{win[1]:.0f}), "
              f"reply: {reply}")
        if dx > 8:
            print("  FAIL: bias not at the sighting")
            ok = False
        elif "windows" not in reply:
            print("  FAIL: reply does not name the object")
            ok = False
        else:
            print("  OK: grounded")

    # 2. honest miss: unseen object names what's known
    reply2, bias2 = understand("wodehaus look at the red car",
                                memory=mem, t_now_ms=30000.0)
    print(f"\n'look at the red car' -> bias={bias2}, reply: {reply2}")
    if bias2 is not None or "red car" not in reply2:
        print("  FAIL: should be an honest miss with no bias")
        ok = False
    else:
        print("  OK: honest miss")

    print("\nAUDIT 3:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
