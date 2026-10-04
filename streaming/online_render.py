"""OnlineVisionPipeline: VisionPipeline with a bounded frame buffer.

THE streaming break in hvp/pipeline.py: it retains every pushed frame
forever (unbounded memory -- fine for a 269 s file, fatal for a stream
that never ends).

The principled fix: a moment at media time t only ever reads frames
in [t - latency - window, t - latency). Frames older than the oldest
not-yet-emitted moment's window can be evicted with zero effect on
any future moment. push() therefore keeps a rolling buffer of
(latency + window + margin) ms; _integrated() is untouched and sees
identical windows, so moments are bit-identical to the batch run.

pull(t_now_ms, t_end_ms): yield newly-ready moments. A moment at media
time t is ready once all frames it needs have arrived, i.e. once the
latest pushed frame time t_last satisfies t_last >= t - latency
(frames arrive in timestamp order, so every frame with time < t -
latency has arrived). t_end_ms caps the grid exactly like the batch
moments(t_end_ms) call -- without it the tail would emit one extra
moment past the media duration.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from hvp.pipeline import VisionPipeline
from hvp.retina import foveate


class OnlineVisionPipeline(VisionPipeline):
    def __init__(self, *args, keep_margin_ms=500.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the unbounded lists with bounded deques. _integrated()
        # only enumerates and indexes them, so it works unchanged.
        self._times = deque()
        self._frames = deque()
        self._keep_margin = float(keep_margin_ms)
        self._k_next = 1
        self._prev = None
        self._t_emitted = 0.0
        self.n_evicted = 0

    def push(self, frame, t_ms):
        frame = np.asarray(frame, dtype=np.float32).reshape(self.frame_shape)
        t = float(t_ms)
        self._times.append(t)
        self._frames.append(frame)
        # Evict frames no future moment can read: anything older than
        # (newest emitted moment) - latency - window - margin.
        horizon = (self._t_emitted - self.latency - self.window
                   - self._keep_margin)
        while self._times and self._times[0] < horizon:
            self._times.popleft()
            self._frames.popleft()
            self.n_evicted += 1

    def add_saccade(self, t_onset_ms, x, y):
        """Mirror an attention-driver decision into the render
        controller (work-res coords). Saccades must be added in
        chronological order -- the driver decides them that way."""
        self.controller.add_saccade(t_onset_ms, x, y)

    def add_pursuit(self, t0_ms, t1_ms, vx, vy):
        """Mirror an attention-driver pursuit decision into the render
        controller (work-res coords)."""
        self.controller.add_pursuit(t0_ms, t1_ms, vx, vy)

    @property
    def buffer_ms(self):
        """Current buffer span in ms (0 if empty)."""
        if not self._times:
            return 0.0
        return self._times[-1] - self._times[0]

    def pull(self, t_now_ms, t_end_ms=None):
        """Return list of newly-ready (t_ms, percept, meta) moments.

        Ready = t_moment <= t_now + latency (all needed frames arrived)
        and t_moment <= t_end (grid cap, like batch moments(t_end_ms)).
        The percept hold logic (suppressed -> hold previous) is the
        batch moments() body verbatim, with `prev` persisted across
        calls instead of living in a generator.
        """
        cap = float(t_now_ms) + self.latency
        if t_end_ms is not None:
            cap = min(cap, float(t_end_ms))
        if self._prev is None:
            self._prev = np.zeros(self.frame_shape, dtype=np.float32)
        out = []
        while self._k_next * self.moment <= cap + 1e-9:
            t = self._k_next * self.moment
            neural, frac_supp = self._integrated(t)
            t_c = t - self.latency - self.window / 2.0
            fx, fy, _ = self.controller.state_at(t_c)
            suppressed = neural is None
            if suppressed:
                percept = self._prev  # hold: no clean samples this window
            elif self._color:
                percept = self._render_color(neural, fx, fy)
                self._prev = percept
            else:
                percept = foveate(neural, (fx, fy), self.dva_per_px,
                                  fovea_radius_deg=self.fovea_radius_deg,
                                  **self._foveate_kw())
                self._prev = percept
            out.append((t, percept,
                        {"suppressed": suppressed,
                         "supp_frac": frac_supp,
                         "fixation": (fx, fy)}))
            self._t_emitted = t
            self._k_next += 1
        return out
