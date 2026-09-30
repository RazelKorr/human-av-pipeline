"""Stimulus generators -- synthetic photon streams at high temporal resolution.

All generators yield (t_ms, frame) with frames as HxW float32 in [0, 1].
"""

import numpy as np
from PIL import Image, ImageDraw


def gen_flicker(duration_s, freq_hz, fps, size=64, duty=0.5):
    """Full-field square-wave flicker: 0/1 at freq_hz."""
    n = int(duration_s * fps)
    ts = np.arange(n) * 1000.0 / fps
    phase = (ts / 1000.0 * freq_hz) % 1.0
    vals = (phase < duty).astype(np.float32)
    for t, v in zip(ts, vals):
        yield t, np.full((size, size), v, dtype=np.float32)


def gen_wheel(duration_s, fps, size=256, n_spokes=8, rev_per_s=1.0,
              bg=0.15, fg=0.9):
    """N-spoke wheel rotating at rev_per_s revolutions per second."""
    n = int(duration_s * fps)
    cx = cy = size / 2.0
    radius = size * 0.42
    for i in range(n):
        t = i * 1000.0 / fps
        angle = 2.0 * np.pi * rev_per_s * (t / 1000.0)
        img = Image.new("L", (size, size), int(bg * 255))
        d = ImageDraw.Draw(img)
        # rim
        d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                  outline=int(fg * 255), width=3)
        for s in range(n_spokes):
            a = angle + 2.0 * np.pi * s / n_spokes
            x2 = cx + radius * np.cos(a)
            y2 = cy + radius * np.sin(a)
            d.line([cx, cy, x2, y2], fill=int(fg * 255), width=3)
        # hub
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=int(fg * 255))
        yield t, np.asarray(img, dtype=np.float32) / 255.0


def gen_flash(duration_s, fps, size=64, flash_at_ms=1000.0,
              flash_dur_ms=20.0, base=0.3, flash_val=0.8):
    """Uniform field with a brief full-field increment at flash_at_ms."""
    n = int(duration_s * fps)
    for i in range(n):
        t = i * 1000.0 / fps
        v = flash_val if flash_at_ms <= t < flash_at_ms + flash_dur_ms else base
        yield t, np.full((size, size), v, dtype=np.float32)


def gen_step(duration_s, fps, size=64, step_at_ms=1000.0):
    """0 -> 1 step at step_at_ms (latency probe)."""
    n = int(duration_s * fps)
    for i in range(n):
        t = i * 1000.0 / fps
        v = 1.0 if t >= step_at_ms else 0.0
        yield t, np.full((size, size), v, dtype=np.float32)


# ------------------------------------------------------------------
# Change-blindness (flicker paradigm) stimuli

def make_change_scene(size=224, grid=8, seed=0):
    """Two scenes differing by one bar's orientation (90-degree flip).

    A grid of short bars, half vertical / half horizontal at random, on a
    dark field. The changed bar is never in the central 2x2 cells (so the
    initial central fixation can't catch it for free).
    Returns (scene_a, scene_b, change_xy_px).
    """
    rng = np.random.default_rng(seed)
    cell = size / grid
    bar_len, bar_w = cell * 0.62, max(3, int(cell * 0.13))
    base = 0.2
    scene = np.full((size, size), base, dtype=np.float32)

    cells = [(r, c) for r in range(grid) for c in range(grid)]
    central = {(r, c) for r in (grid // 2 - 1, grid // 2)
               for c in (grid // 2 - 1, grid // 2)}
    candidates = [i for i, (r, c) in enumerate(cells) if (r, c) not in central]
    change_idx = int(rng.choice(candidates))

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    orientations = []
    for i, (r, c) in enumerate(cells):
        cx = (c + 0.5) * cell
        cy = (r + 0.5) * cell
        vertical = bool(rng.integers(0, 2))
        orientations.append(vertical)
        lum = 0.72 + rng.uniform(-0.06, 0.06)
        if vertical:
            m = (np.abs(xx - cx) < bar_w / 2) & (np.abs(yy - cy) < bar_len / 2)
        else:
            m = (np.abs(yy - cy) < bar_w / 2) & (np.abs(xx - cx) < bar_len / 2)
        scene[m] = lum

    scene_b = scene.copy()
    r, c = cells[change_idx]
    cx = (c + 0.5) * cell
    cy = (r + 0.5) * cell
    # erase bar A, draw it flipped in B
    m_old = (np.abs(xx - cx) < (bar_w / 2 if orientations[change_idx] else bar_len / 2)) & \
            (np.abs(yy - cy) < (bar_len / 2 if orientations[change_idx] else bar_w / 2))
    scene_b[m_old] = base
    flipped_vertical = not orientations[change_idx]
    if flipped_vertical:
        m_new = (np.abs(xx - cx) < bar_w / 2) & (np.abs(yy - cy) < bar_len / 2)
    else:
        m_new = (np.abs(yy - cy) < bar_w / 2) & (np.abs(xx - cx) < bar_len / 2)
    scene_b[m_new] = float(scene[m_old].mean()) if m_old.any() else 0.72

    change_xy = ((c + 0.5) * cell, (r + 0.5) * cell)
    return (np.clip(scene, 0, 1).astype(np.float32),
            np.clip(scene_b, 0, 1).astype(np.float32),
            change_xy)


def flicker_phase_at(t_ms, use_mask=True):
    """Phase of the flicker paradigm at time t: 'A', 'B', or 'mask'.

    Slow phases (500 ms) so a captured saccade can land inside B with
    time to spare -- the fast 240 ms version made even veridical
    capture miss the window, a grid artifact, not a model prediction.
    """
    a_dur, m_dur, b_dur = 500.0, 200.0, 500.0
    if use_mask:
        cycle = a_dur + m_dur + b_dur + m_dur
        t = t_ms % cycle
        if t < a_dur:
            return "A"
        if t < a_dur + m_dur:
            return "mask"
        if t < a_dur + m_dur + b_dur:
            return "B"
        return "mask"
    cycle = a_dur + b_dur
    return "A" if (t_ms % cycle) < a_dur else "B"


def cycle_ms(use_mask=True):
    return 1400.0 if use_mask else 1000.0
