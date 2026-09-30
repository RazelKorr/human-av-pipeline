"""Online (streaming) Level 3 driver: the same closed loop as hvm.driver,
but as a stateful tick() instead of a batch for-loop.

Batch mode precomputes every moment's sensory input, then indexes into
arrays. A live stream can't do that -- moments arrive one at a time and
never come back. OnlineLevel3 holds all the persistent state (joint map,
saccade controller, inhibition of return, decision clock) so each tick
does exactly what one batch-loop iteration did.

hvm.driver.run_closed_loop is now a thin wrapper over this class, so the
batch path and the streaming path share the decision code by construction
-- they cannot drift apart.

Contract for tick(t_ms, vis_sal, aud, speech):
  t_ms    float, stream clock in ms (monotonic; 100 ms per moment)
  vis_sal (56,56) float32 in ~[0,1], or None for a blind moment
  aud     (bin_sal_64, bin_pan_64) tuple, or None for a deaf moment
  speech  scalar in [0,1]: smoothed speech presence (the voice's vote)
"""

import numpy as np

from hvp import baseline as B
from hvp.attention import _blob
from hvp.saccades import SaccadeController
from hvm.priority import JointPriorityMap, SIZE

SCALE = 224.0 / SIZE  # map px -> 224-space px


class OnlineLevel3:
    """Stateful closed-loop Level 3. One instance per read path
    (joint, vision-only, speech-gated) -- same as the batch runs."""

    def __init__(self, dva_per_px, t_end_ms=float("inf")):
        self.dva = dva_per_px
        self.t_end_ms = t_end_ms
        self.jmap = JointPriorityMap()
        cx = cy = 224.0 / 2.0
        self.controller = SaccadeController([(0, cx, cy)], dva_per_px)
        self.inhib = []                      # (ix, iy, t_ms) inhib-of-return
        self.scanpath = [(0.0, cx, cy)]      # (t_on_ms, x_224, y_224)
        self.maps = []                       # joint map after each moment
        self.peaks = []                      # (x, y, value) in map px
        self.next_decision = 100.0
        self.n_ticks = 0

    def tick(self, t_ms, vis_sal, aud, speech=0.0):
        """Process one 100 ms moment arriving at stream clock t_ms."""
        t = float(t_ms)
        self.jmap.step(100.0, vis_sal=vis_sal, aud=aud,
                       speech=float(speech))
        self.maps.append(self.jmap.map.copy())
        self.peaks.append(self.jmap.peak())

        if t >= self.next_decision:
            sal = self.jmap.map.copy()
            for (ix, iy, it) in self.inhib:
                age = t - it
                if age < 1500.0:
                    sal -= np.exp(-age / 800.0) * _blob(
                        (SIZE, SIZE), ix, iy, 5.0)
            fx0, fy0, _ = self.controller.state_at(t)
            yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
            dist_deg = (np.hypot(xx - fx0 / SCALE, yy - fy0 / SCALE)
                        * SCALE * self.dva)
            sal = sal + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
            iy, ix = np.unravel_index(int(np.argmax(sal)), sal.shape)
            tx, ty = (ix + 0.5) * SCALE, (iy + 0.5) * SCALE
            t_on = t + B.SACCADE_LATENCY_MS
            if t_on < self.t_end_ms:
                self.controller.add_saccade(t_on, tx, ty)
                self.scanpath.append((t_on, tx, ty))
            self.inhib = [(x, y, it) for (x, y, it) in self.inhib
                          if t - it < 1500.0]
            self.inhib.append((ix, iy, t))
            self.next_decision = t + 1000.0 / B.SACCADE_RATE_HZ

        self.n_ticks += 1
        return self.peaks[-1]

    def result(self):
        return {"scanpath": self.scanpath, "maps": self.maps,
                "peaks": self.peaks, "jmap": self.jmap,
                "n_ticks": self.n_ticks}
