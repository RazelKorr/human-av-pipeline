"""VisionPipeline: the full temporal stack.

photon stream -> TemporalIntegrator -> SaccadeController -> Retina
    -> PerceptualMoments -> percept stream

Input frames are pushed with millisecond timestamps. Perceptual moments are
pulled at 10 Hz. Each moment's content is the 100 ms integrated signal from
100-200 ms ago (integration window + pipeline latency), foveated at the
fixation holding during that window, or held constant if a saccade was in
flight (suppression).
"""

import numpy as np

from . import baseline as B
from .retina import foveate
from .saccades import SaccadeController


class VisionPipeline:
    def __init__(self, frame_shape, dva_per_px, script, fovea_radius_deg=None):
        self.frame_shape = tuple(frame_shape)
        self.dva_per_px = dva_per_px
        self.controller = SaccadeController(script, dva_per_px)
        # None -> human baseline (B.FOVEA_RADIUS_DEG). A wider value is a
        # deliberate tool override: sharper than a human, not a human.
        self.fovea_radius_deg = (B.FOVEA_RADIUS_DEG if fovea_radius_deg is None
                                 else float(fovea_radius_deg))
        self.latency = B.PIPELINE_LATENCY_MS
        self.window = B.INTEGRATION_WINDOW_MS
        self.moment = B.PERCEPTUAL_MOMENT_MS
        self._times = []   # list of float ms
        self._frames = []  # list of HxW float32

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

    def moments(self, t_end_ms, t_start_ms=0.0):
        """Yield (t_ms, percept_frame, meta) for each 10 Hz moment.

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
            else:
                percept = foveate(neural, (fx, fy), self.dva_per_px,
                                  fovea_radius_deg=self.fovea_radius_deg)
                prev = percept
            yield t, percept, {"suppressed": suppressed,
                              "supp_frac": frac_supp,
                              "fixation": (fx, fy)}
            k += 1
