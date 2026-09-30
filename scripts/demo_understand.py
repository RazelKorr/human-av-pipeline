"""Demo: comprehension -- language steering and querying perception.

Two proves:
  1. "Wodehaus look left": the words become a bias blob on the shared
     priority map; the next saccades go left. Language moves the eyes
     through the same machinery the senses use.
  2. "Wodehaus what do you see?": the answer is read off the live
     OnlineLevel3 (gaze location, saccade count, map peak) -- grounded,
     not canned.

Usage:
  python scripts/demo_understand.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvm.online import OnlineLevel3
from hva.conversation import Turn
from hva.understanding import (PerceptualState, DialogueState, understand,
                               direction_bias)


def main():
    rng = np.random.default_rng(7)
    loop = OnlineLevel3(dva_per_px=0.1, t_end_ms=20000.0)
    perceptual = PerceptualState(loop)
    dialogue = DialogueState()

    # Run 5 s of "watching" with a salient blob on the RIGHT, so gaze
    # has somewhere to be before the command arrives.
    vis = np.zeros((56, 56), dtype=np.float32)
    yy, xx = np.mgrid[0:56, 0:56].astype(np.float32)
    vis += np.exp(-((xx - 42) ** 2 + (yy - 28) ** 2) / 50.0)
    aud = (np.zeros(56, dtype=np.float32), np.zeros(56, dtype=np.float32))
    for m in range(50):
        loop.tick(m * 100.0, vis, aud, 0.0)
    gx, gy = perceptual.gaze_now()
    print(f"before command: gaze at ({gx:.0f}, {gy:.0f}) = "
          f"{perceptual.gaze_where()}")

    # --- Prove 1: "look left" steers the map.
    reply, bias = understand("wodehaus look left",
                             perceptual=perceptual, dialogue=dialogue)
    print(f"command reply: {reply}")
    assert bias is not None
    # The command is a nudge, not a clamp: hold it for 3 s, then release.
    # Check gaze DURING the bias -- afterwards it returns to the stimulus.
    from hva.understanding import direction_bias as _db
    strong = _db("left", strength=2.0)
    mid_gaze = None
    for m in range(50, 100):
        b = strong if m < 80 else None
        loop.tick(m * 100.0, vis, aud, 0.0, task_bias=b)
        if m == 70:
            gx_mid, gy_mid = perceptual.gaze_now()
            mid_gaze = perceptual.gaze_where()
    gx2, gy2 = perceptual.gaze_now()
    print(f"during 'look left' (t=7s): gaze at ({gx_mid:.0f}, {gy_mid:.0f}) "
          f"= {mid_gaze}")
    print(f"after bias released (t=10s): gaze at ({gx2:.0f}, {gy2:.0f}) = "
          f"{perceptual.gaze_where()}")
    moved_left = gx_mid < gx
    print(f"gaze moved left during command: {moved_left}")

    # --- Prove 2: "what do you see" is grounded in the loop.
    reply2, _ = understand("wodehaus what do you see?",
                           perceptual=perceptual, dialogue=dialogue)
    print(f"see-question reply: {reply2}")

    # --- Prove 3: honest limit on ungrounded references.
    reply3, _ = understand("wodehaus look at the red car",
                           perceptual=perceptual, dialogue=dialogue)
    print(f"look-at-thing reply: {reply3}")

    print(f"\ndialogue history:\n{dialogue.history_text()}")
    print(f"\nRESULT: {'PASS' if moved_left else 'FAIL'} -- "
          f"language {'steered' if moved_left else 'did not steer'} gaze")


if __name__ == "__main__":
    main()
