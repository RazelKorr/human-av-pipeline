"""Closed-loop Level 3 driver: the map is stepped per 100 ms moment,
and the saccade controller reads it for targets.

Contract for stim_fn(m):
  m -> (vis_sal, aud), where vis_sal is a (56,56) array in ~[0,1] or None,
  and aud is (bin_sal_64, bin_pan_64) or None.

Read-time (oculomotor, not sensory -- never written to the map):
  - inhibition of return, 1500 ms
  - the human amplitude prior (~5-15 deg saccades)
Deliberately no POI pull: the map's own 300 ms persistence is the memory
here. That is the architectural difference from the Level 1/2 driver.
"""

import numpy as np

from hvp import baseline as B
from hvp.attention import _blob
from hvp.saccades import SaccadeController
from hvm.priority import JointPriorityMap, SIZE

SCALE = 224.0 / SIZE  # map px -> 224-space px


def run_closed_loop(stim_fn, seconds, dva_per_px):
    """Returns dict(scanpath, maps, peaks). scanpath entries are
    (t_on_ms, x_224, y_224); maps[m] is the joint map after moment m;
    peaks[m] is (x, y, value) in map px."""
    jmap = JointPriorityMap()
    n_mom = int(seconds * 10)
    t_end = seconds * 1000.0
    cx = cy = 224.0 / 2.0
    controller = SaccadeController([(0, cx, cy)], dva_per_px)
    inhib = []
    scanpath = [(0.0, cx, cy)]
    maps, peaks = [], []
    next_decision = 100.0

    for m in range(n_mom):
        t = m * 100.0
        vis_sal, aud = stim_fn(m)
        jmap.step(100.0, vis_sal=vis_sal, aud=aud)
        maps.append(jmap.map.copy())
        peaks.append(jmap.peak())

        if t >= next_decision:
            sal = jmap.map.copy()
            for (ix, iy, it) in inhib:
                age = t - it
                if age < 1500.0:
                    sal -= np.exp(-age / 800.0) * _blob(
                        (SIZE, SIZE), ix, iy, 5.0)
            fx0, fy0, _ = controller.state_at(t)
            yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
            dist_deg = (np.hypot(xx - fx0 / SCALE, yy - fy0 / SCALE)
                        * SCALE * dva_per_px)
            sal = sal + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
            iy, ix = np.unravel_index(int(np.argmax(sal)), sal.shape)
            tx, ty = (ix + 0.5) * SCALE, (iy + 0.5) * SCALE
            t_on = t + B.SACCADE_LATENCY_MS
            if t_on < t_end:
                controller.add_saccade(t_on, tx, ty)
                scanpath.append((t_on, tx, ty))
            inhib = [(x, y, it) for (x, y, it) in inhib if t - it < 1500.0]
            inhib.append((ix, iy, t))
            next_decision = t + 1000.0 / B.SACCADE_RATE_HZ

    return {"scanpath": scanpath, "maps": maps, "peaks": peaks,
            "jmap": jmap}
