"""Tests for foveated color vision (hvp/retina.foveate_color).

Color is opt-in: the grayscale path is unchanged. The color path foveates
luminance sharply and chroma with a tiny fovea + heavy peripheral blur,
matching cone distribution (color-blind periphery).
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hvp.retina import foveate, foveate_color, rgb_to_ycbcr, ycbcr_to_rgb
from hvp.pipeline import VisionPipeline


def test_ycbcr_roundtrip():
    rng = np.random.default_rng(0)
    rgb = rng.random((16, 24, 3), dtype=np.float32)
    rt = ycbcr_to_rgb(rgb_to_ycbcr(rgb))
    assert np.abs(rt - rgb).max() < 1e-4


def test_foveate_color_shape_and_range():
    rng = np.random.default_rng(1)
    img = rng.random((60, 80, 3), dtype=np.float32)
    out = foveate_color(img, (40, 30), 0.1)
    assert out.shape == (60, 80, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_foveate_color_rejects_non_rgb():
    with pytest.raises(ValueError):
        foveate_color(np.zeros((60, 80), dtype=np.float32), (40, 30), 0.1)


def test_color_saturated_at_fixation_desaturated_in_periphery():
    img = np.zeros((120, 160, 3), dtype=np.float32)
    img[55:65, 75:85, 0] = 1.0   # red square at fixation
    img[55:65, 5:15, 0] = 1.0    # red square far in periphery
    out = foveate_color(img, (80, 60), 0.1)
    center_sat = out[60, 80].max() - out[60, 80].min()
    periph_sat = out[60, 10].max() - out[60, 10].min()
    assert center_sat > 0.8, f"color should survive at fixation, got {center_sat}"
    assert periph_sat < 0.3, f"periphery should desaturate, got {periph_sat}"


def test_pipeline_color_moments_end_to_end():
    img = np.zeros((48, 64, 3), dtype=np.float32)
    img[20:28, 28:36] = (1.0, 0.2, 0.2)
    pipe = VisionPipeline((48, 64, 3), 0.1, [(0, 32, 24)])
    for i in range(40):
        pipe.push(img, i * 33.37)
    moms = list(pipe.moments(1200.0))
    assert len(moms) > 0
    assert moms[0][1].shape == (48, 64, 3)


def test_pipeline_grayscale_unchanged():
    img = np.zeros((48, 64), dtype=np.float32)
    img[20:28, 28:36] = 0.9
    pipe = VisionPipeline((48, 64), 0.1, [(0, 32, 24)])
    assert not pipe._color
    for i in range(40):
        pipe.push(img, i * 33.37)
    moms = list(pipe.moments(1200.0))
    assert moms[0][1].shape == (48, 64)
