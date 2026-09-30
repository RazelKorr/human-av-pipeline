"""Level 3 battery: does the joint map integrate like multisensory
integration should?

All synthetic, all deterministic (no RNG anywhere).

  M1  spatial congruence: flash-left + click-left (matched) vs
      flash-left + click-right (mismatched). Matched should build a
      taller single peak; mismatched should show two competing peaks.
  M2  audio alone: a left click with no flash puts the map peak left.
  M3  vision alone: a right flash with no click puts the map peak right.
  M4  conflict: equal-strength flash-left + click-right. Vision should
      win the peak (the ventriloquism direction, per W_AUD < W_VIS).
  M5  audio read path: a left-peaked map gives left-panned bins a larger
      spatial gain than right-panned bins.
  M6  speech gating: flash-right + click-left at equal strength. Without
      speech, vision wins the peak (the ventriloquism direction, as in
      M4). With speech present, the speech-gated auditory boost should
      flip the peak to the click -- the voice captures the map.

Usage: python3 -m hvm.battery   (from the repo root)
"""

import numpy as np
from scipy import ndimage

from hvp import baseline as B
from hvp.attention import _blob
from hvm.driver import run_closed_loop
from hvm.priority import JointPriorityMap, SIZE

DVA = B.FIELD_WIDTH_DEG / 224.0
LEFT_X, RIGHT_X, MID_Y = 14.0, 42.0, 28.0


def _flash(x, y, amp=1.0, sigma=4.0):
    return amp * _blob((SIZE, SIZE), x, y, sigma)


def _click(b0, b1, amp=1.0, pan=0.0):
    bs = np.zeros(64, dtype=np.float32)
    bs[b0:b1] = amp
    return bs, np.full(64, pan, dtype=np.float32)


def _stim(vis=None, aud=None, on=(2, 7)):
    """stim_fn with a stimulus on moments [on0, on1)."""
    def fn(m):
        if on[0] <= m < on[1]:
            return vis, aud
        return None, None
    return fn


def _n_peaks(map2d, frac=0.5):
    m = map2d.max()
    if m <= 1e-9:
        return 0
    lab, n = ndimage.label(map2d > frac * m)
    return n


def _orient_time(scanpath, side="left"):
    for (t_on, x, y) in scanpath[1:]:
        if (x < 112.0) == (side == "left"):
            return t_on
    return float("inf")


def m1_congruence():
    vis = _flash(LEFT_X, MID_Y)
    matched = run_closed_loop(
        _stim(vis=vis, aud=_click(0, 64, pan=-0.8)), 1.0, DVA)
    mismatched = run_closed_loop(
        _stim(vis=vis, aud=_click(0, 64, pan=+0.8)), 1.0, DVA)
    pm = matched["maps"][4].max()
    px = mismatched["maps"][4].max()
    ratio = pm / max(px, 1e-9)
    n_matched = _n_peaks(matched["maps"][4])
    n_mismatched = _n_peaks(mismatched["maps"][4])
    t_matched = _orient_time(matched["scanpath"])
    t_mismatched = _orient_time(mismatched["scanpath"])
    ok = ratio > 1.3 and n_matched == 1 and n_mismatched == 2 \
        and t_matched <= t_mismatched
    return ok, (f"peak ratio matched/mismatched={ratio:.2f} "
                f"(need >1.3); peaks {n_matched} vs {n_mismatched} "
                f"(need 1 vs 2); orient {t_matched:.0f}ms vs "
                f"{t_mismatched:.0f}ms")


def m2_audio_alone():
    r = run_closed_loop(_stim(aud=_click(0, 64, pan=-0.8)), 1.0, DVA)
    x, y, v = r["peaks"][6]
    ok = x < SIZE / 2 and v > 0
    return ok, f"audio-only peak at x={x:.1f} (need <28), value={v:.2f}"


def m3_vision_alone():
    r = run_closed_loop(_stim(vis=_flash(RIGHT_X, MID_Y)), 1.0, DVA)
    x, y, v = r["peaks"][6]
    ok = x > SIZE / 2 and v > 0
    return ok, f"visual-only peak at x={x:.1f} (need >28), value={v:.2f}"


def m4_conflict():
    r = run_closed_loop(
        _stim(vis=_flash(LEFT_X, MID_Y, amp=1.0),
              aud=_click(0, 64, amp=1.0, pan=+0.8)), 1.0, DVA)
    x, y, v = r["peaks"][6]
    ok = x < SIZE / 2
    return ok, (f"conflict peak at x={x:.1f} (need <28: "
                f"vision wins the tug-of-war)")


def m5_audio_read():
    jmap = JointPriorityMap()
    aud = _click(20, 28, amp=1.0, pan=-0.8)
    for _ in range(5):
        jmap.step(100.0, aud=aud)
    g = jmap.gains_for_pans([-0.8, 0.8])
    ok = g[0] > 1.2 and g[1] < 1.15
    return ok, f"gains: left-panned bin {g[0]:.2f} (need >1.2), " \
               f"right-panned bin {g[1]:.2f} (need <1.15)"


def m6_speech_gating():
    vis = _flash(RIGHT_X, MID_Y, amp=1.0)
    aud = _click(0, 64, amp=1.0, pan=-0.8)
    stim = _stim(vis=vis, aud=aud)
    r0 = run_closed_loop(stim, 1.0, DVA)
    r1 = run_closed_loop(stim, 1.0, DVA, speech_fn=lambda m: 1.0)
    x0 = r0["peaks"][6][0]
    x1 = r1["peaks"][6][0]
    ok = x0 > SIZE / 2 and x1 < SIZE / 2
    return ok, (f"no-speech peak x={x0:.1f} (need >28: vision wins); "
                f"speech peak x={x1:.1f} (need <28: speech-gated "
                f"audio flips it)")


TESTS = [
    ("M1 congruence (matched > mismatched)", m1_congruence),
    ("M2 audio alone -> left peak", m2_audio_alone),
    ("M3 vision alone -> right peak", m3_vision_alone),
    ("M4 conflict -> vision wins", m4_conflict),
    ("M5 audio read path (spatial gains)", m5_audio_read),
    ("M6 speech gating flips a conflict", m6_speech_gating),
]


def run_all():
    print("Level 3 battery: joint priority map")
    results = []
    for name, fn in TESTS:
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 -- battery must not die
            ok, detail = False, f"exception: {e}"
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}\n"
              f"        {detail}")
    n = sum(results)
    print(f"{n}/{len(results)} passed")
    return all(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
