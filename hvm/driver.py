"""Closed-loop Level 3 driver: the map is stepped per 100 ms moment,
and the saccade controller reads it for targets.

Contract for stim_fn(m):
  m -> (vis_sal, aud), where vis_sal is a (56,56) array in ~[0,1] or None,
  and aud is (bin_sal_64, bin_pan_64) or None.

speech_fn(m), optional, returns a scalar in [0,1]: smoothed speech
presence for moment m. Kept separate from stim_fn so existing 2-tuple
stimuli keep working; the driver passes it to JointPriorityMap.step as
the speech gate.

Read-time (oculomotor, not sensory -- never written to the map):
  - inhibition of return, 1500 ms
  - the human amplitude prior (~5-15 deg saccades)
Deliberately no POI pull: the map's own 300 ms persistence is the memory
here. That is the architectural difference from the Level 1/2 driver.
"""

import numpy as np

from hvm.online import OnlineLevel3
from hvm.priority import SIZE

SCALE = 224.0 / SIZE  # map px -> 224-space px  (kept for import compat)


def run_closed_loop(stim_fn, seconds, dva_per_px, speech_fn=None):
    """Returns dict(scanpath, maps, peaks). scanpath entries are
    (t_on_ms, x_224, y_224); maps[m] is the joint map after moment m;
    peaks[m] is (x, y, value) in map px.

    Thin wrapper over hvm.online.OnlineLevel3 -- the batch path and the
    streaming path share the decision code, so they cannot drift apart.
    """
    loop = OnlineLevel3(dva_per_px, t_end_ms=seconds * 1000.0)
    n_mom = int(seconds * 10)
    for m in range(n_mom):
        vis_sal, aud = stim_fn(m)
        sp = float(speech_fn(m)) if speech_fn is not None else 0.0
        loop.tick(m * 100.0, vis_sal, aud, sp)
    return loop.result()
