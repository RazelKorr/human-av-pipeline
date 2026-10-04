"""Tests for color-native attention (hvp/attention.chroma_salience).

The chroma channel is opt-in: salience_map without small_rgb behaves
exactly like the luminance-only driver. With it, chromatically salient
targets pull saccades the way brightness does.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hvp import attention as A
from hvp import baseline as B
from hvp.saccades import SaccadeController
from hvp.pipeline import VisionPipeline


def _controller():
    dva = B.FIELD_WIDTH_DEG / A.SIZE
    return SaccadeController([(0, A.SIZE / 2.0, A.SIZE / 2.0)], dva)


def test_chroma_salience_achromatic_is_zero():
    gray = np.full((A.SMALL, A.SMALL, 3), 0.5, dtype=np.float32)
    cs = A.chroma_salience(gray)
    assert cs.shape == (A.SMALL, A.SMALL)
    assert np.abs(cs).max() < 1e-6


def test_chroma_salience_finds_color_boundary():
    # left half red, right half green: same luminance-ish, strong R/G edge
    img = np.zeros((A.SMALL, A.SMALL, 3), dtype=np.float32)
    img[:, :A.SMALL // 2, 0] = 0.8   # red
    img[:, A.SMALL // 2:, 1] = 0.8   # green
    cs = A.chroma_salience(img)
    mid = A.SMALL // 2
    edge = cs[:, mid - 3:mid + 3].max()
    flat = max(cs[:, :mid - 8].max(), cs[:, mid + 8:].max())
    assert edge > 0.5, "color boundary should be salient"
    assert flat < 0.2, "uniform color regions should not be salient"


def test_chroma_salience_blue_yellow_axis():
    img = np.zeros((A.SMALL, A.SMALL, 3), dtype=np.float32)
    img[:, :A.SMALL // 2, 2] = 0.8              # blue
    img[:, A.SMALL // 2:, 0] = 0.8              # yellow-ish (r+g)
    img[:, A.SMALL // 2:, 1] = 0.8
    cs = A.chroma_salience(img)
    assert cs.max() > 0.5


def test_salience_map_backward_compatible():
    # Without small_rgb the map must equal static + 1.5*transient
    # minus inhibition and the fixation disk (the old formula exactly).
    rng = np.random.default_rng(7)
    small = rng.random((A.SMALL, A.SMALL), dtype=np.float32)
    trans = rng.random((A.SMALL, A.SMALL), dtype=np.float32) * 0.1
    ctl = _controller()
    sal = A.salience_map(small, trans, 500.0, ctl, [])
    from scipy.ndimage import gaussian_filter
    static = A._norm(np.abs(small - gaussian_filter(small, 6)))
    transient = A._norm(trans)
    fx, fy, _ = ctl.state_at(500.0)
    expected = (static + 1.5 * transient
                - A._blob((A.SMALL, A.SMALL), fx / A.SCALE, fy / A.SCALE, 3.5))
    assert np.abs(sal - expected).max() < 1e-6


def test_chroma_channel_adds_to_salience():
    rng = np.random.default_rng(11)
    small = rng.random((A.SMALL, A.SMALL), dtype=np.float32) * 0.2 + 0.4
    trans = np.zeros((A.SMALL, A.SMALL), dtype=np.float32)
    ctl = _controller()
    base = A.salience_map(small, trans, 500.0, ctl, [])
    rgb = np.full((A.SMALL, A.SMALL, 3), 0.5, dtype=np.float32)
    rgb[20:30, 20:30, 0] = 0.9  # saturated red patch, low luma contrast
    rgb[20:30, 20:30, 1:] = 0.1
    col = A.salience_map(small, trans, 500.0, ctl, [], small_rgb=rgb)
    added = col - base
    assert added[25, 25] > 0.3, "chroma channel should boost the red patch"
    assert np.abs(added).max() < 3.0, "chroma should not dominate outright"


def test_color_target_beats_gray_distractor():
    # Gray blob: strong luminance contrast. Red blob: weak luminance
    # contrast, strong chroma contrast. With the chroma channel on, the
    # argmax should move to the red blob.
    lum = np.full((A.SMALL, A.SMALL), 0.5, dtype=np.float32)
    lum[8:16, 8:16] = 0.95          # bright gray distractor, top-left
    rgb = np.stack([lum, lum, lum], axis=-1).copy()
    rgb[40:48, 40:48, 0] = 0.75     # red target: luma ~0.45 vs bg 0.5
    rgb[40:48, 40:48, 1] = 0.15
    rgb[40:48, 40:48, 2] = 0.15
    lum2 = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    trans = np.zeros((A.SMALL, A.SMALL), dtype=np.float32)
    ctl = _controller()
    sal_gray = A.salience_map(lum2, trans, 500.0, ctl, [])
    iy0, ix0 = np.unravel_index(int(np.argmax(sal_gray)), sal_gray.shape)
    sal_col = A.salience_map(lum2, trans, 500.0, ctl, [], small_rgb=rgb,
                             chroma_weight=2.0)
    iy1, ix1 = np.unravel_index(int(np.argmax(sal_col)), sal_col.shape)
    d_gray = np.hypot(ix0 - 12, iy0 - 12)
    d_col = np.hypot(ix1 - 44, iy1 - 44)
    assert d_gray < 8, "luminance-only driver should pick the gray blob"
    assert d_col < 8, "color-native driver should pick the red target"


def test_pipeline_moment_ms_finer():
    # 50 ms spacing -> ~2x the moments of 100 ms over the same span,
    # same integration window.
    shape = (36, 48, 3)
    dva = 0.1
    script = [(0, 24.0, 18.0)]
    rng = np.random.default_rng(3)
    for mm, expected_min in ((100.0, 8), (50.0, 16)):
        pipe = VisionPipeline(shape, dva, script, moment_ms=mm)
        assert pipe.moment == mm
        assert pipe.window == B.INTEGRATION_WINDOW_MS  # window unchanged
        for i in range(40):
            pipe.push(rng.random(shape, dtype=np.float32), i * 25.0)
        moments = list(pipe.moments(1000.0))
        assert len(moments) >= expected_min, (mm, len(moments))
    # spacing check at 50 ms
    pipe = VisionPipeline(shape, dva, script, moment_ms=50.0)
    for i in range(40):
        pipe.push(rng.random(shape, dtype=np.float32), i * 25.0)
    ts = [t for t, _, _ in pipe.moments(1000.0)]
    gaps = np.diff(ts)
    assert np.allclose(gaps, 50.0), gaps[:5]


def test_downsample_color_shape():
    rng = np.random.default_rng(5)
    big = rng.random((240, 320, 3), dtype=np.float32)
    small = A._downsample_color(big)
    assert small.shape == (A.SMALL, A.SMALL, 3)
    assert small.min() >= 0.0 and small.max() <= 1.0
