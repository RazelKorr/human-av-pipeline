"""Retina: foveated sampling.

Only ~2 degrees are sharp. Everything else is progressively blurred and
filled in. This is why the pipeline is *worse* than a camera at pixel
recall and *better* at behaving like an observer.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from . import baseline as B


def foveate(frame, fix, dva_per_px,
            fovea_radius_deg=B.FOVEA_RADIUS_DEG,
            e2_deg=B.E2_DEG,
            levels=5, max_sigma_px=6.0):
    """Blur frame with eccentricity-dependent sigma around fixation point.

    frame: HxW float32 in [0, 1]. fix: (x, y) pixel fixation.
    sigma(e) = max_sigma * max(0, e - fovea) / (max(0, e - fovea) + e2):
    zero inside the fovea, saturating toward max_sigma_px in the periphery.
    """
    frame = np.asarray(frame, dtype=np.float32)
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ecc_deg = np.hypot(xx - fix[0], yy - fix[1]) * dva_per_px

    sharp = np.clip(ecc_deg - fovea_radius_deg, 0.0, None)
    sigma = max_sigma_px * sharp / (sharp + e2_deg)

    bounds = np.linspace(0.0, max_sigma_px, levels + 1)
    out = np.zeros_like(frame)
    for i in range(levels):
        lo, hi = bounds[i], bounds[i + 1]
        if i < levels - 1:
            mask = (sigma >= lo) & (sigma < hi)
        else:
            mask = sigma >= lo
        if not np.any(mask):
            continue
        sig = 0.5 * (lo + hi)
        blurred = gaussian_filter(frame, sig) if sig > 0.3 else frame
        out += blurred * mask
    return np.clip(out, 0.0, 1.0).astype(np.float32)
