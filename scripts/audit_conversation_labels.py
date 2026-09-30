"""Audit 4: the label stream into conversation.

Feeds an ObjectMemory with REAL CLIP classifications of the 12 foveal
crops (at their labeled positions/times), then drives ResponsePolicy
exactly as the live runner does:

  1. "Hey Wodehaus, what do you see?" -> reply must name objects the
     classifier actually saw, with qualitative positions.
  2. "Hey Wodehaus, look at the windows" -> grounded reply + task bias
     peaked at the windows sighting.
  3. "Hey Wodehaus, look at the red car" -> honest miss, no bias.

The live tick-loop feeding itself is verified by
scripts/run_conversation.py --foveal-vocab dark-street (RECOGNIZED
lines in the log); see docs/recognition.md audit 4.
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from audit_recognition import extract_crops, MONTAGE, LABELS
from hva.conversation import ResponsePolicy, Turn, PerceptualState
from hva.understanding import ObjectMemory
from hvp.recognize import DARK_STREET_VOCAB, FovealClassifier


class _Loop:
    def __init__(self):
        self.scanpath = [(0.0, 112.0, 112.0)]

        class _Map:
            def peak(self):
                return (28.0, 28.0, 0.01)
        self.jmap = _Map()


def main():
    manual = json.load(open(LABELS))
    crops = extract_crops(MONTAGE)
    vocab = list(DARK_STREET_VOCAB.items())
    clf = FovealClassifier()
    pol = ResponsePolicy()
    pol.perceptual = PerceptualState(_Loop(), memory=pol.memory)
    for i, crop in enumerate(crops):
        L = manual[i]
        label, conf = clf.classify(crop, vocab)
        pol.memory.add(label, L["x"] / 4.0, L["y"] / 4.0,
                       t_ms=L["t"], conf=conf)
    print("memory:", pol.memory.known_objects())

    ok = True

    # 1. "what do you see?" names what was recognized
    r1 = pol.generate(Turn("Hey Wodehaus, what do you see?", 31.0))
    print("\nQ: what do you see?\nA:", r1)
    for name in pol.memory.known_objects():
        if name not in r1:
            print(f"  FAIL: recognized '{name}' not named")
            ok = False
    if ok:
        print("  OK: all recognized objects named")

    # 2. "look at the windows" grounds to the sighting
    r2 = pol.generate(Turn("Hey Wodehaus, look at the windows", 32.0))
    b2 = pol.take_bias()
    win = pol.memory.locate("windows", 32000.0)
    print("\nQ: look at the windows\nA:", r2)
    if win is None:
        print("  SKIP: no windows sighting")
    else:
        iy, ix = divmod(b2.argmax(), 56)
        print(f"  bias peak ({ix},{iy}) vs sighting "
              f"({win[0]:.0f},{win[1]:.0f})")
        if abs(ix - win[0]) > 8 or "windows" not in r2:
            print("  FAIL: not grounded")
            ok = False
        else:
            print("  OK: grounded")

    # 3. unseen object: honest miss
    r3 = pol.generate(Turn("Hey Wodehaus, look at the red car", 33.0))
    b3 = pol.take_bias()
    print("\nQ: look at the red car\nA:", r3)
    if b3 is not None or "red car" not in r3:
        print("  FAIL: should be an honest miss")
        ok = False
    else:
        print("  OK: honest miss")

    print("\nAUDIT 4:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
