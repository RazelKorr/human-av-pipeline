"""SaccadeController: ballistic eye jumps with mid-flight suppression,
plus smooth-pursuit glides for tracking moving targets.

Scripted fixation sequences drive the tests:
    script = [(0, x0, y0), (onset_ms_1, x1, y1), ...]
Entry 0 is the initial fixation. Each later entry is a saccade *onset*:
the eye jumps from the previous fixation to the new one, taking
21 + 2.2 * amplitude_deg ms (the main sequence), during which vision is
suppressed -- the percept holds the pre-saccadic frame, no smear.

Pursuit (added 2026-10-03, magno channel): when the attended target
moves coherently, the eyes glide with it instead of jumping. A
pursuit is (t0, t1, from_xy, (vx, vy)): fixation moves linearly at
(vx, vy) px/ms. There is NO saccadic suppression during pursuit --
the eyes stay online while tracking, which is the biological point.
A later saccade interrupts an active glide.
"""


# --- darkness behavior (RazelKorr, 2026-09-29) ---
# In near-black frames human eyes don't keep saccading on noise: focus
# (dwell) increases a little -- "trying to comprehend something difficult,
# but not much" -- while exploration drops. Modeled as a luminance gate
# on the saccade decision clock: darker frames space decisions out.
DARK_LUM = 0.03       # mean frame luminance below this counts as dark
DARK_SLOW_MAX = 3.0   # max decision-interval multiplier in full darkness

def darkness_gain(lum):
    """Saccade decision-interval multiplier for a frame of mean
    luminance `lum` (0..1). 1.0 in normal light, ramping to
    DARK_SLOW_MAX in full dark."""
    dark = min(max((DARK_LUM - lum) / DARK_LUM, 0.0), 1.0)
    return 1.0 + (DARK_SLOW_MAX - 1.0) * dark


class SaccadeController:
    def __init__(self, script, dva_per_px):
        if not script or script[0][0] != 0:
            raise ValueError("script must start with (0, x0, y0)")
        self.script = [(float(t), float(x), float(y)) for t, x, y in script]
        self.dva_per_px = dva_per_px
        # Precompute saccade events: (onset_ms, dur_ms, from_fix, to_fix)
        self.events = []
        for i in range(1, len(self.script)):
            t_on = self.script[i][0]
            fx0, fy0 = self.script[i - 1][1], self.script[i - 1][2]
            fx1, fy1 = self.script[i][1], self.script[i][2]
            amp_deg = ((fx1 - fx0) ** 2 + (fy1 - fy0) ** 2) ** 0.5 * dva_per_px
            dur = 21.0 + 2.2 * amp_deg
            self.events.append((t_on, dur, (fx0, fy0), (fx1, fy1)))
        # Pursuit glides: (t0_ms, t1_ms, from_xy, (vx_px_ms, vy_px_ms)).
        # Kept separate from saccade events so the saccade-only
        # unpack sites (detection logic) never see a glide tuple.
        self.pursuits = []

    @staticmethod
    def main_sequence_duration_ms(amp_deg):
        return 21.0 + 2.2 * amp_deg

    def add_saccade(self, t_onset_ms, x, y):
        """Append a saccade online (for closed-loop attention drivers).

        The saccade starts at t_onset_ms from wherever fixation is then
        (including mid-pursuit: the glide position). A saccade
        interrupts any active pursuit glide at its onset.
        Callers must add onsets in chronological order.
        """
        t_on = float(t_onset_ms)
        fx, fy, _ = self.state_at(t_on)
        amp_deg = ((x - fx) ** 2 + (y - fy) ** 2) ** 0.5 * self.dva_per_px
        dur = self.main_sequence_duration_ms(amp_deg)
        # Interrupt pursuit at the saccade onset.
        self.pursuits = [(a, min(b, t_on), f, v)
                         for (a, b, f, v) in self.pursuits if a < t_on]
        self.events.append((t_on, dur, (fx, fy),
                            (float(x), float(y))))
        self.script.append((t_on, float(x), float(y)))

    def add_pursuit(self, t0_ms, t1_ms, vx, vy):
        """Append a smooth-pursuit glide online.

        From wherever fixation is at t0_ms, glide at (vx, vy) px/ms
        until t1_ms. No suppression during the glide. A later pursuit
        supersedes an overlapping older one; a later saccade
        (add_saccade) interrupts the glide. Callers must add in
        chronological order.
        """
        t0, t1 = float(t0_ms), float(t1_ms)
        if t1 <= t0:
            raise ValueError(f"pursuit needs t1 > t0, got {t0}, {t1}")
        fx, fy, _ = self.state_at(t0)
        # Supersede any still-active older pursuit.
        self.pursuits = [(a, b, f, v) for (a, b, f, v) in self.pursuits
                         if b <= t0]
        self.pursuits.append((t0, t1, (fx, fy),
                              (float(vx), float(vy))))

    def state_at(self, t_ms):
        """Return (fix_x, fix_y, suppressed) at time t_ms.

        Mid-saccade -> pre-saccadic fixation, suppressed. Mid-pursuit
        -> glide position, NOT suppressed (the eyes stay online while
        tracking). Otherwise the latest completed fixation.
        """
        current_fix = (self.script[0][1], self.script[0][2])
        i = j = 0
        evs, pus = self.events, self.pursuits
        while True:
            e = evs[i] if i < len(evs) else None
            p = pus[j] if j < len(pus) else None
            if e is None and p is None:
                break
            if p is None or (e is not None and e[0] <= p[0]):
                t_on, dur, frm, to = e
                i += 1
                if t_ms < t_on:
                    break
                if t_ms < t_on + dur:
                    return frm[0], frm[1], True  # mid-saccade: hold
                current_fix = to
            else:
                t0, t1, frm, (vx, vy) = p
                j += 1
                if t_ms < t0:
                    break
                if t_ms < t1:
                    dt = t_ms - t0
                    return (frm[0] + vx * dt, frm[1] + vy * dt, False)
                current_fix = (frm[0] + vx * (t1 - t0),
                               frm[1] + vy * (t1 - t0))
        return current_fix[0], current_fix[1], False
