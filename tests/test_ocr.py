"""Tests for hvp.ocr (Phase 5, audit item 17).

Engine behavior is tested against a stub (deterministic); the
real tesseract path is tested on rendered text and skips when
the binary is absent from the machine.
"""

import shutil

import numpy as np
import pytest

from hvp.ocr import StubEngine, ocr_frame, ocr_frames

CANNED = [
    {"text": "FUEL", "confidence": 92.0,
     "bbox": {"x": 10, "y": 10, "w": 80, "h": 30}},
    {"text": "smudge", "confidence": 12.0,
     "bbox": {"x": 10, "y": 50, "w": 60, "h": 20}},
    {"text": "", "confidence": 99.0,
     "bbox": {"x": 0, "y": 0, "w": 5, "h": 5}},
]


def _blank(h=90, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_stub_engine_confidence_filter():
    out = ocr_frame(_blank(), t_s=3.5, engine=StubEngine(CANNED))
    assert [r["text"] for r in out] == ["FUEL"]
    assert out[0]["t"] == 3.5
    assert out[0]["confidence"] == 92.0
    assert out[0]["bbox"]["w"] == 80 // 2  # 200px wide -> 2x prepare


def test_stub_engine_min_conf_override():
    out = ocr_frame(_blank(), engine=StubEngine(CANNED), min_conf=10)
    assert [r["text"] for r in out] == ["FUEL", "smudge"]


def test_ocr_frames_flattens_with_times():
    eng = StubEngine(CANNED)
    out = ocr_frames([(1.0, _blank()), (2.0, _blank())], engine=eng)
    assert [r["t"] for r in out] == [1.0, 2.0]


@pytest.mark.skipif(shutil.which("tesseract") is None,
                    reason="tesseract binary not installed")
def test_tesseract_reads_large_rendered_text():
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1280, 720), (5, 5, 5))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/"
                              "DejaVuSans-Bold.ttf", 64)
    d = ImageDraw.Draw(img)
    d.text((80, 300), "FUEL", fill=(255, 255, 255), font=font)
    out = ocr_frame(np.asarray(img), t_s=0.0)
    assert "FUEL" in {r["text"].upper() for r in out}
