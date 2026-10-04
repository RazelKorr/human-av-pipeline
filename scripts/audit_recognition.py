"""Audit 1: zero-shot CLIP vs the 12 hand-labeled foveal crops.

Extracts the crops from the montage, classifies each with label-derived
prompts, and reports top-1 agreement with the manual labels plus
per-crop latency.
"""
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from PIL import Image

from hvp.recognize import DARK_STREET_VOCAB, FovealClassifier

MONTAGE = os.path.join(REPO, "tests", "fixtures", "foveal_crops.png")
LABELS = os.path.join(REPO, "tests", "fixtures", "foveal_labels.json")

PROMPTS = dict(DARK_STREET_VOCAB)


def extract_crops(path):
    """4x3 grid, 330px cells; skip the title strip at each cell top."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    cw, ch = W // 4, H // 3
    crops = []
    for i in range(12):
        col, row = i % 4, i // 4
        x0, y0 = col * cw, row * ch
        crops.append(im.crop((x0 + 8, y0 + 62, x0 + cw - 8, y0 + ch - 4)))
    return crops


def main():
    manual = json.load(open(LABELS))
    crops = extract_crops(MONTAGE)
    vocab = [(name, PROMPTS[name]) for name in PROMPTS]
    clf = FovealClassifier()
    t0 = time.time()
    clf._ensure()  # downloads weights on first run
    print(f"model load: {time.time() - t0:.1f}s")
    hits = 0
    lat = []
    for i, crop in enumerate(crops):
        want = manual[i]["label"]
        t1 = time.time()
        dist = clf.distribution(crop, vocab)
        lat.append(time.time() - t1)
        got, conf = dist[0]
        mark = "OK " if got == want else "MISS"
        if got == want:
            hits += 1
        top3 = ", ".join(f"{n}:{p:.2f}" for n, p in dist[:3])
        print(f"crop {i:2d} want={want:13s} got={got:13s} {mark} [{top3}]")
    print(f"\n{hits}/12 top-1 agreement "
          f"({hits / 12:.0%}); mean latency {sum(lat) / len(lat) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
