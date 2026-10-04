"""Tests for the foveation speedups (2026-10-03, Mykal-authorized).

- mask_cache: must be BIT-IDENTICAL to uncached.
- fovea_levels: changes the math (opt-in), output stays valid.
- box_sigma_threshold: approximation — close to gaussian, not identical.
- VisionPipeline params thread through; defaults are v2 behavior.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from hvp.retina import foveate, foveate_color_fast, box_blur
from hvp.pipeline import VisionPipeline


@pytest.fixture
def frame():
    rng = np.random.default_rng(7)
    return rng.random((180, 320), dtype=np.float32)


DVA = 0.05
FIX = (160.0, 90.0)


def test_mask_cache_bit_identical(frame):
    ref = foveate(frame, FIX, DVA)
    cache = {}
    a = foveate(frame, FIX, DVA, mask_cache=cache)   # miss
    b = foveate(frame, FIX, DVA, mask_cache=cache)   # hit
    assert np.array_equal(ref, a)
    assert np.array_equal(a, b)
    # a different fixation is a separate key, still correct
    c = foveate(frame, (10.0, 10.0), DVA, mask_cache=cache)
    ref2 = foveate(frame, (10.0, 10.0), DVA)
    assert np.array_equal(c, ref2)
    assert len(cache) == 2


def test_mask_cache_bounded(frame):
    cache = {}
    for i in range(200):
        foveate(frame, (float(i), float(i)), DVA, mask_cache=cache)
    from hvp.retina import _MASK_CACHE_MAX
    assert len(cache) <= _MASK_CACHE_MAX


def test_mask_cache_color_path_bit_identical():
    rng = np.random.default_rng(11)
    rgb = rng.random((180, 320, 3), dtype=np.float32)
    ref = foveate_color_fast(rgb, FIX, DVA)
    cache = {}
    a = foveate_color_fast(rgb, FIX, DVA, mask_cache=cache)
    b = foveate_color_fast(rgb, FIX, DVA, mask_cache=cache)
    assert np.array_equal(ref, a)
    assert np.array_equal(a, b)


def test_fewer_levels_changes_math_but_stays_valid(frame):
    ref = foveate(frame, FIX, DVA)
    fast = foveate(frame, FIX, DVA, levels=3)
    assert fast.shape == ref.shape and fast.dtype == np.float32
    assert fast.min() >= 0.0 and fast.max() <= 1.0
    assert not np.array_equal(ref, fast)  # coarser quantization: real change
    # ...but a small one: same fovea, same periphery trend
    assert np.abs(fast - ref).mean() < 0.05


def test_box_blur_approximates_gaussian():
    imp = np.zeros((129, 129), dtype=np.float32)
    imp[64, 64] = 1.0
    for sig in (2.5, 3.0, 4.2, 5.4):
        g = gaussian_filter(imp, sig)
        x = box_blur(imp, sig)
        assert np.abs(x - g).max() < 0.01, f"sigma={sig}"
        # same total energy (both normalized kernels)
        assert abs(x.sum() - g.sum()) < 1e-5


def test_box_threshold_keeps_small_sigma_exact(frame):
    ref = foveate(frame, FIX, DVA)
    # threshold above every band sigma -> identical to v2
    same = foveate(frame, FIX, DVA, box_sigma_threshold=99.0)
    assert np.array_equal(ref, same)
    # threshold 2.0 -> peripheral bands approximated: close, not identical
    approx = foveate(frame, FIX, DVA, box_sigma_threshold=2.0)
    assert not np.array_equal(ref, approx)
    assert np.abs(approx - ref).max() < 0.05


def test_pipeline_defaults_are_fast_combo():
    # 2026-10-03: Mykal flipped the defaults to the fast streaming combo.
    p = VisionPipeline((90, 160), DVA, [(0, 80.0, 45.0)])
    assert p.fovea_levels == 3
    assert isinstance(p._mask_cache, dict)
    assert p.box_sigma_threshold is None
    assert p.chroma_mode == "luma_ratio"


def test_pipeline_explicit_v2_mode():
    # The validated v2 math stays available via explicit opt-in.
    p = VisionPipeline((90, 160), DVA, [(0, 80.0, 45.0)],
                       chroma_mode="foveated",
                       fovea_levels=5, mask_cache=False)
    assert p.fovea_levels == 5
    assert p._mask_cache is None
    assert p.box_sigma_threshold is None
    assert p.chroma_mode == "foveated"


def test_pipeline_params_threaded():
    p = VisionPipeline((90, 160, 3), DVA, [(0, 80.0, 45.0)],
                       chroma_mode="passthrough",
                       fovea_levels=3, mask_cache=True,
                       box_sigma_threshold=2.0)
    assert p.fovea_levels == 3
    assert isinstance(p._mask_cache, dict)
    assert p.box_sigma_threshold == 2.0
    rng = np.random.default_rng(3)
    for k in range(12):
        p.push(rng.random((90, 160, 3), dtype=np.float32), k * 50.0)
    moments = list(p.moments(600.0))
    assert len(moments) == 6  # default 100 ms moments over 600 ms
    assert all(m[1].shape == (90, 160, 3) for m in moments)
    # cache was actually used (one fixation -> one entry)
    assert len(p._mask_cache) == 1


def test_pipeline_fovea_levels_validation():
    with pytest.raises(ValueError):
        VisionPipeline((90, 160), DVA, [(0, 80.0, 45.0)], fovea_levels=0)


# --- luma_ratio mode (2026-10-03, Mykal-authorized) ---------------------------


def test_luma_ratio_black_frame_safe():
    from hvp.retina import foveate_color_luma_ratio
    blk = np.zeros((64, 96, 3), dtype=np.float32)
    out = foveate_color_luma_ratio(blk, (48.0, 32.0), DVA)
    assert out.shape == blk.shape
    assert not np.isnan(out).any()
    assert not np.isinf(out).any()
    assert np.all(out == 0.0)  # ratio defaults to 1, black stays black


def test_luma_ratio_output_range():
    from hvp.retina import foveate_color_luma_ratio
    rng = np.random.default_rng(21)
    rgb = rng.random((120, 200, 3), dtype=np.float32)
    out = foveate_color_luma_ratio(rgb, (100.0, 60.0), DVA)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not np.isnan(out).any()


def test_luma_ratio_uniform_luma_passthrough():
    from hvp.retina import foveate_color_luma_ratio
    # constant luma, varying chroma: foveate(constant) == constant, so
    # constant luma, varying chroma: foveate(constant) == constant, so
    # ratio == 1 everywhere and the frame passes through unchanged.
    # (kept away from the [0,1] clip so luma stays exactly uniform)
    rng = np.random.default_rng(22)
    chroma = rng.random((120, 200, 3), dtype=np.float32) * 0.4 + 0.3
    # normalize each pixel to luma 0.5: scale rgb so dot(w, rgb) == 0.5
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lum = chroma @ w
    rgb = (chroma * (0.5 / lum)[..., None]).astype(np.float32)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0  # no clipping involved
    out = foveate_color_luma_ratio(rgb, (100.0, 60.0), DVA)
    assert np.allclose(out, rgb, atol=1e-5)


def test_luma_ratio_preserves_chromaticity():
    from hvp.retina import foveate_color_luma_ratio
    # out is a per-pixel scalar multiple of the input, so hue is untouched
    rng = np.random.default_rng(23)
    rgb = rng.random((120, 200, 3), dtype=np.float32) * 0.9 + 0.05
    out = foveate_color_luma_ratio(rgb, (100.0, 60.0), DVA)
    lum_in = (rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
    lum_out = (out @ np.array([0.299, 0.587, 0.114], dtype=np.float32))
    # chromaticity is preserved wherever the [0,1] clip didn't intervene
    unclipped = (out > 1e-6) & (out < 1.0 - 1e-6)
    ok = unclipped.all(axis=-1) & (lum_in > 0.05) & (lum_out > 1e-4)
    for c in range(3):
        # channel ratios match the luma ratio wherever both are defined
        ratio_c = np.divide(out[..., c], rgb[..., c],
                            out=np.zeros_like(lum_in), where=rgb[..., c] > 1e-4)
        ratio_y = lum_out / np.maximum(lum_in, 1e-6)
        assert np.allclose(ratio_c[ok], ratio_y[ok], atol=1e-4)
    assert ok.mean() > 0.75  # invariant covers the unclipped bulk


def test_luma_ratio_dark_noise_bounded():
    from hvp.retina import foveate_color_luma_ratio
    # near-black noisy region under a bright one: luma bleeds down, the
    # ratio amplifies the dark noise — must stay dim, never explode
    rng = np.random.default_rng(24)
    frame = np.zeros((120, 200, 3), dtype=np.float32)
    frame[:60] = rng.random((60, 200, 3), dtype=np.float32) * 0.8 + 0.2
    frame[60:] = rng.random((60, 200, 3), dtype=np.float32) * 0.005
    out = foveate_color_luma_ratio(frame, (100.0, 30.0), DVA)
    dark = np.s_[80:110, 20:180]
    assert out[dark].max() < 0.1  # amplified, but stays dim
    assert not np.isnan(out).any()


def test_luma_ratio_forwards_speedup_kwargs():
    from hvp.retina import foveate_color_luma_ratio
    rng = np.random.default_rng(25)
    rgb = rng.random((120, 200, 3), dtype=np.float32)
    cache = {}
    a = foveate_color_luma_ratio(rgb, (100.0, 60.0), DVA,
                                 levels=3, mask_cache=cache)
    b = foveate_color_luma_ratio(rgb, (100.0, 60.0), DVA,
                                 levels=3, mask_cache=cache)
    assert np.array_equal(a, b)  # cache hit path works
    assert len(cache) == 1
    with pytest.raises(ValueError):
        foveate_color_luma_ratio(rgb[..., 0], (100.0, 60.0), DVA)


def test_pipeline_chroma_mode_luma_ratio():
    p = VisionPipeline((90, 160, 3), DVA, [(0, 80.0, 45.0)],
                       chroma_mode="luma_ratio",
                       fovea_levels=3, mask_cache=True)
    rng = np.random.default_rng(26)
    for k in range(12):
        p.push(rng.random((90, 160, 3), dtype=np.float32), k * 50.0)
    moments = list(p.moments(600.0))
    assert len(moments) == 6
    assert all(m[1].shape == (90, 160, 3) for m in moments)
    assert all(m[1].min() >= 0.0 and m[1].max() <= 1.0 for m in moments)
    with pytest.raises(ValueError):
        VisionPipeline((90, 160, 3), DVA, [(0, 80.0, 45.0)],
                       chroma_mode="bogus")


# --- input validation (2026-10-03 audit) ------------------------------------
# levels=0 used to fall through to a silent all-black frame; negative
# max_sigma_px produced degenerate bands. Both now fail loudly.


def test_foveate_levels_zero_raises(frame):
    with pytest.raises(ValueError):
        foveate(frame, FIX, DVA, levels=0)
    with pytest.raises(ValueError):
        foveate(frame, FIX, DVA, levels=-3)


def test_foveate_levels_noninteger_raises(frame):
    with pytest.raises(ValueError):
        foveate(frame, FIX, DVA, levels=2.5)


def test_foveate_negative_max_sigma_raises(frame):
    with pytest.raises(ValueError):
        foveate(frame, FIX, DVA, max_sigma_px=-1.0)


def test_box_blur_zero_sigma_is_identity(frame):
    assert np.array_equal(box_blur(frame, 0.0), frame)
    assert np.array_equal(box_blur(frame, -2.0), frame)


def test_box_blur_positive_sigma_blurs(frame):
    assert not np.array_equal(box_blur(frame, 2.0), frame)


def test_mask_cache_key_covers_all_mask_params(frame):
    # Same fixation, different foveation params -> distinct keys, each
    # bit-identical to its own uncached run (no stale-mask serving).
    cache = {}
    a = foveate(frame, FIX, DVA, mask_cache=cache, max_sigma_px=6.0)
    b = foveate(frame, FIX, DVA, mask_cache=cache, max_sigma_px=3.0)
    assert np.array_equal(a, foveate(frame, FIX, DVA, max_sigma_px=6.0))
    assert np.array_equal(b, foveate(frame, FIX, DVA, max_sigma_px=3.0))
    assert len(cache) == 2
    c = foveate(frame, FIX, DVA, mask_cache=cache, fovea_radius_deg=2.0)
    assert np.array_equal(c, foveate(frame, FIX, DVA, fovea_radius_deg=2.0))
    assert len(cache) == 3
