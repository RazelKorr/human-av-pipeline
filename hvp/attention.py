"""Attention: salience-driven saccades for the change-blindness (T4) test.

Closed-loop driver. The pipeline watches the flicker paradigm and picks
saccade targets from a salience map:

  salience = static center-surround contrast + transient channel
             - inhibition of return - current-fixation disk

The transient channel is why the mask works: each mask onset/offset floods
the whole field with change, swamping the local transient of the flipped
bar -- the mask doesn't remove the signal, it *masks* it. Without the mask,
the flip's local transient wins outright and the eyes jump straight there.

Detection (the behavioral metric, mirroring the human button-press):
  R1: at each B-phase onset T, if the eyes are on the change (within 3 deg)
      at T+120 ms, the change is seen.
  R2: if a saccade lands within 3 deg during B with >= 80 ms of B left,
      the change is seen.
Cycles-to-detection is counted in paradigm cycles, mask vs. no-mask.
"""

from collections import deque

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from . import baseline as B
from .saccades import SaccadeController
from . import stimuli as S

SIZE = 224          # scene resolution (px)
SMALL = 56          # salience-map resolution (px)
SCALE = SIZE / SMALL
INPUT_FPS = 200.0
DT_MS = 1000.0 / INPUT_FPS
DETECT_RADIUS_DEG = 2.0   # change must be foveated, not just nearby
TRANS_TAU_MS = 200.0      # transient persistence (the channel rings briefly)


def _downsample(frame):
    """Any-size frame -> SMALLxSMALL float32 thumbnail (bilinear).

    The old reshape trick required dimensions divisible by SMALL; this
    handles arbitrary input sizes and aspect ratios."""
    im = Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    small = im.resize((SMALL, SMALL), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255.0


def _norm(x):
    m = x.max()
    return x / m if m > 1e-9 else x


def _blob(shape, x0, y0, sigma):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))


def salience_map(small_now, trans_map, t_now, controller, inhib,
                 sx=SCALE, sy=SCALE):
    """salience = static center-surround + persistent transient
                 - inhibition of return - current-fixation disk.

    sx/sy: frame-pixels per grid cell along x/y. Default to the square
    SCALE for the synthetic tests; video callers pass per-axis scales
    so non-square frames map correctly."""
    static = _norm(np.abs(small_now - gaussian_filter(small_now, 6)))
    transient = _norm(trans_map)
    sal = static + 1.5 * transient

    for (ix, iy, it) in inhib:
        age = t_now - it
        if age < 1500:
            sal -= np.exp(-age / 800.0) * _blob((SMALL, SMALL), ix, iy, 5.0)

    fx, fy, _ = controller.state_at(t_now)
    sal -= _blob((SMALL, SMALL), fx / sx, fy / sy, 3.5)
    return sal


def _fixation_for_detection(controller, t):
    """Where the eyes are (or are about to be) at time t."""
    fx, fy, suppressed = controller.state_at(t)
    if suppressed:
        for (t_on, dur, _frm, to) in controller.events:
            if t_on <= t < t_on + dur:
                return to
    return (fx, fy)


def _b_phase_bounds(t_ms, use_mask):
    """(start, end) of the B-phase containing t_ms, or None."""
    if use_mask:
        k = int(t_ms // 1400.0)
        s, e = k * 1400.0 + 700.0, k * 1400.0 + 1200.0
    else:
        k = int(t_ms // 1000.0)
        s, e = k * 1000.0 + 500.0, k * 1000.0 + 1000.0
    return (s, e) if s <= t_ms < e else None


def run_trial(scene_a, scene_b, change_xy, use_mask=True, max_cycles=25):
    """Run one flicker-paradigm trial. Returns dict with cycles_to_detect."""
    small_a, small_b = _downsample(scene_a), _downsample(scene_b)
    small_m = np.full((SMALL, SMALL), 0.5, dtype=np.float32)
    dva = B.FIELD_WIDTH_DEG / SIZE
    detect_px = DETECT_RADIUS_DEG / dva
    cx = cy = SIZE / 2.0

    controller = SaccadeController([(0, cx, cy)], dva)
    inhib = []
    scanpath = [(0.0, cx, cy)]
    trans_map = np.zeros((SMALL, SMALL), dtype=np.float32)
    lag_buf = deque()  # ~25 ms lagged frames for the transient channel

    cycle = S.cycle_ms(use_mask)
    max_t = max_cycles * cycle
    b_onset0 = 700.0 if use_mask else 500.0
    appearances = [b_onset0 + k * cycle for k in range(max_cycles)]
    next_app = 0

    t = 0.0
    next_decision = 100.0
    detected_cycle = None
    decay = np.exp(-DT_MS / TRANS_TAU_MS)

    def small_at(tt):
        ph = S.flicker_phase_at(tt, use_mask)
        return small_a if ph == "A" else small_b if ph == "B" else small_m

    while t < max_t:
        f = small_at(t)
        lag_buf.append((t, f))
        while lag_buf and lag_buf[0][0] < t - 30.0:
            lag_buf.popleft()
        f_old = lag_buf[0][1]
        trans_map = np.maximum(trans_map * decay, np.abs(f - f_old))

        # R1: change-appearance check
        while next_app < len(appearances) and appearances[next_app] <= t:
            T = appearances[next_app]
            fx, fy = _fixation_for_detection(controller, T + 120.0)
            if np.hypot(fx - change_xy[0], fy - change_xy[1]) <= detect_px:
                detected_cycle = float(next_app + 1)
            next_app += 1

        # saccade decision
        if t >= next_decision:
            sal = salience_map(f, trans_map, t, controller, inhib)
            iy, ix = np.unravel_index(int(np.argmax(sal)), sal.shape)
            tx, ty = (ix + 0.5) * SCALE, (iy + 0.5) * SCALE
            t_on = t + B.SACCADE_LATENCY_MS
            controller.add_saccade(t_on, tx, ty)
            t_on_e, dur_e, _f, _to = controller.events[-1]
            t_land = t_on_e + dur_e
            scanpath.append((t_on_e, tx, ty))
            inhib = [(x, y, it) for (x, y, it) in inhib if t - it < 1500.0]
            inhib.append((ix, iy, t))
            next_decision = t + 1000.0 / B.SACCADE_RATE_HZ

            # R2: saccade landing check
            bounds = _b_phase_bounds(t_land, use_mask)
            if bounds and np.hypot(tx - change_xy[0], ty - change_xy[1]) <= detect_px \
                    and bounds[1] - t_land >= 80.0:
                detected_cycle = float(int(t_land // cycle) + 1)

        if detected_cycle is not None:
            break
        t += DT_MS

    return {
        "cycles_to_detect": detected_cycle if detected_cycle is not None else float(max_cycles),
        "found": detected_cycle is not None,
        "scanpath": scanpath,
        "scene_small": small_a,
        "change_xy_small": (change_xy[0] / SCALE, change_xy[1] / SCALE),
        "use_mask": use_mask,
    }
