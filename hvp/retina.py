"""Retina: foveated sampling.

Only ~2 degrees are sharp. Everything else is progressively blurred and
filled in. This is why the pipeline is *worse* than a camera at pixel
recall and *better* at behaving like an observer.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from . import baseline as B


# --- Foveation speedups (2026-10-03, Mykal-authorized) -----------------------
# Three independently-toggleable optimizations for the foveate() bottleneck.
# All default to OFF / v2 behavior: the validated math is bit-identical
# unless a caller explicitly opts in.

# Bound on cached eccentricity-band sets (fixation holds are few; a
# streaming run only ever has a handful of live fixations).
_MASK_CACHE_MAX = 64


def _sigma_bands(h, w, fix, dva_per_px,
                 fovea_radius_deg, e2_deg, levels, max_sigma_px):
    """Eccentricity band masks for foveate(). Pure function of its args.

    Returns [(lo, hi, mask), ...], one per pyramid level. This is the
    per-moment recomputation the mask cache memoizes.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ecc_deg = np.hypot(xx - fix[0], yy - fix[1]) * dva_per_px

    sharp = np.clip(ecc_deg - fovea_radius_deg, 0.0, None)
    sigma = max_sigma_px * sharp / (sharp + e2_deg)

    bounds = np.linspace(0.0, max_sigma_px, levels + 1)
    bands = []
    for i in range(levels):
        lo, hi = bounds[i], bounds[i + 1]
        if i < levels - 1:
            mask = (sigma >= lo) & (sigma < hi)
        else:
            mask = sigma >= lo
        bands.append((lo, hi, mask))
    return bands


def _cached_bands(mask_cache, h, w, fix, dva_per_px,
                  fovea_radius_deg, e2_deg, levels, max_sigma_px):
    """Memoized _sigma_bands(). Bit-identical to the uncached path: the
    key holds the exact fixation floats, so a hit replays the exact
    computation that would have run. FIFO eviction keeps it bounded."""
    key = (fix[0], fix[1], h, w,
           fovea_radius_deg, e2_deg, levels, max_sigma_px, dva_per_px)
    bands = mask_cache.get(key)
    if bands is None:
        bands = _sigma_bands(h, w, fix, dva_per_px,
                             fovea_radius_deg, e2_deg, levels, max_sigma_px)
        if len(mask_cache) >= _MASK_CACHE_MAX:
            mask_cache.pop(next(iter(mask_cache)))
        mask_cache[key] = bands
    return bands


def box_blur(frame, sigma, passes=3):
    """Approximate a Gaussian blur with stacked box blurs.

    O(1) per pixel regardless of sigma — the classic realtime trick.
    ``passes`` box blurs of variance-matched odd width approximate a
    Gaussian of std ``sigma`` (variance of one odd-width-s box is
    (s^2-1)/12; s = sqrt(12*sigma^2/passes + 1)).
    A non-positive sigma is the identity (previously it returned a
    width-3 blur, which is not a zero-width Gaussian).
    """
    out = np.asarray(frame, dtype=np.float32)
    if sigma <= 0:
        return out
    s = int(round(np.sqrt(12.0 * sigma * sigma / passes + 1.0)))
    s = max(3, s | 1)  # odd, at least 3
    for _ in range(passes):
        out = uniform_filter(out, size=s, mode="reflect")
    return out


def foveate(frame, fix, dva_per_px,
            fovea_radius_deg=B.FOVEA_RADIUS_DEG,
            e2_deg=B.E2_DEG,
            levels=5, max_sigma_px=6.0,
            mask_cache=None, box_sigma_threshold=None):
    """Blur frame with eccentricity-dependent sigma around fixation point.

    frame: HxW float32 in [0, 1]. fix: (x, y) pixel fixation.
    sigma(e) = max_sigma * max(0, e - fovea) / (max(0, e - fovea) + e2):
    zero inside the fovea, saturating toward max_sigma_px in the periphery.

    Speedup knobs (all default to v2 behavior):
    - levels: fewer pyramid levels = fewer full-frame blurs (coarser
      eccentricity quantization; changes the math, so opt-in).
    - mask_cache: dict memoizing the eccentricity-band masks per
      fixation. Bit-identical to uncached; hits ~85% of moments in
      streaming (saccades ~3/s vs 20 moments/s).
    - box_sigma_threshold: above this per-band sigma, use the stacked
      box-blur approximation instead of scipy's gaussian (whose cost
      grows with sigma). Approximation, not bit-identical.
    """
    frame = np.asarray(frame, dtype=np.float32)
    h, w = frame.shape
    if int(levels) != levels or levels < 1:
        # levels=0 used to fall through to a silent all-black frame
        # (no bands -> out stays zeros); fail loudly instead.
        raise ValueError(f"levels must be an integer >= 1, got {levels!r}")
    levels = int(levels)
    if max_sigma_px < 0:
        raise ValueError(f"max_sigma_px must be >= 0, got {max_sigma_px!r}")
    fix = (float(fix[0]), float(fix[1]))
    if mask_cache is not None:
        bands = _cached_bands(mask_cache, h, w, fix, dva_per_px,
                              fovea_radius_deg, e2_deg,
                              levels, max_sigma_px)
    else:
        bands = _sigma_bands(h, w, fix, dva_per_px,
                             fovea_radius_deg, e2_deg,
                             levels, max_sigma_px)

    out = np.zeros_like(frame)
    for (lo, hi, mask) in bands:
        if not np.any(mask):
            continue
        sig = 0.5 * (lo + hi)
        if sig <= 0.3:
            blurred = frame
        elif box_sigma_threshold is not None and sig > box_sigma_threshold:
            blurred = box_blur(frame, sig)
        else:
            blurred = gaussian_filter(frame, sig)
        out += blurred * mask
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --- Color vision -----------------------------------------------------------
# Human color vision is fovea-centric: cones pack the central ~2 degrees and
# the periphery is nearly colorblind. So the faithful upgrade is NOT "color
# everywhere" — it is luminance foveated as above, with chroma blurred far
# more aggressively outside a tiny central spot. Desaturation with
# eccentricity, the way eyes actually work.

# BT.601 matrices, full-range.
_RGB2YCBCR = np.array([
    [0.29900,  0.58700,  0.11400],
    [-0.168736, -0.331264, 0.50000],
    [0.50000, -0.418688, -0.081312],
], dtype=np.float32)
_YCBCR_OFFSET = np.array([0.0, 0.5, 0.5], dtype=np.float32)


def rgb_to_ycbcr(frame):
    """HxWx3 float32 RGB [0,1] -> HxWx3 float32 YCbCr (Y in [0,1], C in [0,1])."""
    frame = np.asarray(frame, dtype=np.float32)
    return np.clip(frame @ _RGB2YCBCR.T + _YCBCR_OFFSET, 0.0, 1.0)


def ycbcr_to_rgb(frame):
    """Inverse of rgb_to_ycbcr."""
    frame = np.asarray(frame, dtype=np.float32)
    inv = np.linalg.inv(_RGB2YCBCR).astype(np.float32)
    return np.clip((frame - _YCBCR_OFFSET) @ inv.T, 0.0, 1.0)


def foveate_color(frame, fix, dva_per_px,
                  fovea_radius_deg=B.FOVEA_RADIUS_DEG,
                  e2_deg=B.E2_DEG,
                  levels=5, max_sigma_px=6.0,
                  chroma_fovea_deg=0.15, chroma_sigma_scale=3.0,
                  mask_cache=None, box_sigma_threshold=None):
    """Foveated color: sharp luminance, color-blind periphery.

    frame: HxWx3 float32 RGB in [0, 1]. fix: (x, y) pixel fixation.
    Luminance (Y) is foveated exactly like the grayscale path. Chroma
    (Cb, Cr) is foveated with a tiny fovea (~0.15 deg) and a much larger
    max sigma, so color survives only near fixation and desaturates
    into the periphery — matching cone distribution, and cheaper than
    full-res per-channel foveation.

    mask_cache / box_sigma_threshold are forwarded to each foveate()
    call (see foveate() for semantics).
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"foveate_color needs HxWx3 RGB, got {frame.shape}")
    kw = dict(mask_cache=mask_cache, box_sigma_threshold=box_sigma_threshold)
    ycc = rgb_to_ycbcr(frame)
    y = foveate(ycc[..., 0], fix, dva_per_px,
                fovea_radius_deg=fovea_radius_deg, e2_deg=e2_deg,
                levels=levels, max_sigma_px=max_sigma_px, **kw)
    cb = foveate(ycc[..., 1], fix, dva_per_px,
                 fovea_radius_deg=chroma_fovea_deg, e2_deg=e2_deg,
                 levels=levels, max_sigma_px=max_sigma_px * chroma_sigma_scale,
                 **kw)
    cr = foveate(ycc[..., 2], fix, dva_per_px,
                 fovea_radius_deg=chroma_fovea_deg, e2_deg=e2_deg,
                 levels=levels, max_sigma_px=max_sigma_px * chroma_sigma_scale,
                 **kw)
    return ycbcr_to_rgb(np.stack([y, cb, cr], axis=-1))


def foveate_color_fast(frame, fix, dva_per_px,
                       fovea_radius_deg=B.FOVEA_RADIUS_DEG,
                       e2_deg=B.E2_DEG,
                       levels=5, max_sigma_px=6.0,
                       mask_cache=None, box_sigma_threshold=None):
    """Fast color: foveated luminance, full-resolution chroma passthrough.

    Deliberate fidelity tradeoff for streaming/realtime use (added
    2026-10-03 at Mykal's call: "keep the color feed but don't foveate
    the chroma"). Luminance is foveated exactly like the grayscale path;
    chroma (Cb, Cr) passes through unfolded, so color does NOT
    desaturate with eccentricity the way human vision does. Roughly 3x
    cheaper than foveate_color (one blur pyramid instead of three).
    NOT the validated v2 math — use chroma_mode="foveated" (the
    default) when you need bit-agreement with the v2 percept stream.

    mask_cache / box_sigma_threshold are forwarded to the luminance
    foveate() call (see foveate() for semantics).
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"foveate_color_fast needs HxWx3 RGB, got {frame.shape}")
    ycc = rgb_to_ycbcr(frame)
    y = foveate(ycc[..., 0], fix, dva_per_px,
                fovea_radius_deg=fovea_radius_deg, e2_deg=e2_deg,
                levels=levels, max_sigma_px=max_sigma_px,
                mask_cache=mask_cache, box_sigma_threshold=box_sigma_threshold)
    return ycbcr_to_rgb(np.stack([y, ycc[..., 1], ycc[..., 2]], axis=-1))


# BT.601 luma weights (same as the Y row of _RGB2YCBCR above).
_LUMA_W = np.array([0.29900, 0.58700, 0.11400], dtype=np.float32)


def foveate_color_luma_ratio(frame, fix, dva_per_px,
                             fovea_radius_deg=B.FOVEA_RADIUS_DEG,
                             e2_deg=B.E2_DEG,
                             levels=5, max_sigma_px=6.0,
                             mask_cache=None, box_sigma_threshold=None):
    """Fastest color: foveate luma, rescale RGB by the foveated-luma ratio.

    Replaces the YCbCr round-trip in foveate_color_fast (added 2026-10-03
    at Mykal's authorization, as the push past realtime). Math:
        Y     = 0.299*R + 0.587*G + 0.114*B   (single dot product)
        Y_fov = foveate(Y, ...)
        out   = clip(frame * (Y_fov / Y), 0, 1)
    Safe-divide: where Y < 1e-6 (black regions) the ratio defaults to 1,
    so black stays black instead of NaN/exploding. Chromaticity is
    preserved by construction (every channel scaled by the same factor);
    only brightness is foveated. NOT the validated v2 math.

    mask_cache / box_sigma_threshold are forwarded to the luma
    foveate() call (see foveate() for semantics).
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"foveate_color_luma_ratio needs HxWx3 RGB, "
                         f"got {frame.shape}")
    y = frame @ _LUMA_W
    y_fov = foveate(y, fix, dva_per_px,
                    fovea_radius_deg=fovea_radius_deg, e2_deg=e2_deg,
                    levels=levels, max_sigma_px=max_sigma_px,
                    mask_cache=mask_cache, box_sigma_threshold=box_sigma_threshold)
    ratio = np.divide(y_fov, y, out=np.ones_like(y_fov),
                      where=y > 1e-6).astype(np.float32)
    return np.clip(frame * ratio[..., None], 0.0, 1.0).astype(np.float32)
