"""VisionPipeline: the full temporal stack.

photon stream -> TemporalIntegrator -> SaccadeController -> Retina
    -> PerceptualMoments -> percept stream

Input frames are pushed with millisecond timestamps. Perceptual moments are
pulled at moment_ms spacing (default 100 ms = 10 Hz, the human baseline).
Each moment's content is the integrated signal over the 100 ms window ending
100 ms ago (integration window + pipeline latency), foveated at the fixation
holding during that window, or held constant when the whole window fell
inside saccades (suppression: no clean samples to integrate).

Finer temporal resolution: pass moment_ms=50.0 for 20 Hz moments. The
integration window stays 100 ms (Bloch's-law temporal integration — the
honest human number); only the sampling gets finer, so consecutive moments
overlap. This resolves saccade flights (~50 ms) that a 100 ms grid swallows
whole, and makes the suppressed flag fire for real instead of ~never.
"""

import numpy as np

from . import baseline as B
from .retina import foveate, foveate_color, foveate_color_fast, \
    foveate_color_luma_ratio
from .saccades import SaccadeController


class VisionPipeline:
    def __init__(self, frame_shape, dva_per_px, script, fovea_radius_deg=None,
                 moment_ms=None, chroma_mode="luma_ratio",
                 fovea_levels=3, mask_cache=True, box_sigma_threshold=None):
        self.frame_shape = tuple(frame_shape)
        self.dva_per_px = dva_per_px
        self.controller = SaccadeController(script, dva_per_px)
        # None -> human baseline (B.FOVEA_RADIUS_DEG). A wider value is a
        # deliberate tool override: sharper than a human, not a human.
        self.fovea_radius_deg = (B.FOVEA_RADIUS_DEG if fovea_radius_deg is None
                                 else float(fovea_radius_deg))
        self.latency = B.PIPELINE_LATENCY_MS
        self.window = B.INTEGRATION_WINDOW_MS
        # None -> human baseline (B.PERCEPTUAL_MOMENT_MS = 100 ms, 10 Hz).
        # Finer (e.g. 50 ms) = overlapping integration windows, resolving
        # saccade flights. The window itself stays human.
        self.moment = (B.PERCEPTUAL_MOMENT_MS if moment_ms is None
                       else float(moment_ms))
        # chroma_mode: "luma_ratio" (default) = fast streaming combo:
        # foveate luma and rescale RGB by the foveated-luma ratio, no
        # YCbCr round-trip. "foveated" = validated v2 math, chroma
        # desaturates with eccentricity like human vision (explicit
        # opt-in for bit-agreement with the v2 reference). "passthrough"
        # = foveate luminance only, chroma unfolded.
        if chroma_mode not in ("foveated", "passthrough", "luma_ratio"):
            raise ValueError(f"chroma_mode must be 'foveated', "
                             f"'passthrough' or 'luma_ratio', "
                             f"got {chroma_mode!r}")
        self.chroma_mode = chroma_mode
        # foveation speedups (defaults = fast streaming combo; the
        # validated v2 behavior is available explicitly via
        # chroma_mode="foveated", fovea_levels=5, mask_cache=False):
        # - fovea_levels: blur-pyramid depth (3 = default; 5 = v2;
        #   fewer = coarser eccentricity quantization, changes the math).
        # - mask_cache: memoize eccentricity-band masks per fixation
        #   (bit-identical to uncached; ~85% hit rate in streaming).
        # - box_sigma_threshold: above this per-band sigma use the
        #   stacked box-blur approximation (approximation, not identical).
        self.fovea_levels = int(fovea_levels)
        if self.fovea_levels < 1:
            raise ValueError(f"fovea_levels must be >= 1, "
                             f"got {fovea_levels!r}")
        self._mask_cache = {} if mask_cache else None
        self.box_sigma_threshold = (None if box_sigma_threshold is None
                                    else float(box_sigma_threshold))
        # color path when frames are HxWx3; grayscale otherwise
        self._color = len(self.frame_shape) == 3 and self.frame_shape[2] == 3
        self._times = []   # list of float ms
        self._frames = []  # list of HxW (or HxWx3) float32

    def push(self, frame, t_ms):
        frame = np.asarray(frame, dtype=np.float32).reshape(self.frame_shape)
        self._times.append(float(t_ms))
        self._frames.append(frame)

    def _integrated(self, t_moment):
        """Mean of input over [t - latency - window, t - latency], excluding
        samples taken mid-saccade (cortical suppression blanks the smear,
        but the clean pre/post-saccadic samples still integrate).

        Returns (frame, frac_suppressed). A None frame means the whole
        window fell inside saccades: hold the previous percept.
        """
        a = t_moment - self.latency - self.window
        b = t_moment - self.latency
        in_window = [i for i, t in enumerate(self._times) if a <= t < b]
        if not in_window:
            return None, 0.0
        idx = [i for i in in_window
               if not self.controller.state_at(self._times[i])[2]]
        if not idx:
            return None, 1.0
        stack = np.stack([self._frames[i] for i in idx], axis=0)
        return (stack.mean(axis=0).astype(np.float32),
                1.0 - len(idx) / len(in_window))

    def _foveate_kw(self):
        """Keyword args threading the foveation speedups into foveate()."""
        return dict(levels=self.fovea_levels,
                    mask_cache=self._mask_cache,
                    box_sigma_threshold=self.box_sigma_threshold)

    def _render_color(self, neural, fx, fy):
        """Foveate one integrated color frame per the chroma_mode."""
        kw = dict(fovea_radius_deg=self.fovea_radius_deg, **self._foveate_kw())
        if self.chroma_mode == "passthrough":
            return foveate_color_fast(neural, (fx, fy), self.dva_per_px, **kw)
        if self.chroma_mode == "luma_ratio":
            return foveate_color_luma_ratio(neural, (fx, fy), self.dva_per_px,
                                            **kw)
        return foveate_color(neural, (fx, fy), self.dva_per_px, **kw)

    def moments(self, t_end_ms, t_start_ms=0.0):
        """Yield (t_ms, percept_frame, meta) for each moment.

        Moments are spaced self.moment ms apart (default 100 ms = 10 Hz).
        t_start_ms skips earlier moments (for chunked rendering); the
        caller should still push ~1 s of lead-in frames so the first
        kept moment has a full integration window and a seeded hold.
        """
        prev = np.zeros(self.frame_shape, dtype=np.float32)
        k = max(1, int(t_start_ms // self.moment))
        while k * self.moment <= t_end_ms:
            t = k * self.moment
            neural, frac_supp = self._integrated(t)
            t_c = t - self.latency - self.window / 2.0
            fx, fy, _ = self.controller.state_at(t_c)
            suppressed = neural is None
            if suppressed:
                percept = prev  # hold: no clean samples this window
            elif self._color:
                percept = self._render_color(neural, fx, fy)
                prev = percept
            else:
                percept = foveate(neural, (fx, fy), self.dva_per_px,
                                  fovea_radius_deg=self.fovea_radius_deg,
                                  **self._foveate_kw())
                prev = percept
            yield t, percept, {"suppressed": suppressed,
                              "supp_frac": frac_supp,
                              "fixation": (fx, fy)}
            k += 1
