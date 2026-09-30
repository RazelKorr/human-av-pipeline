"""SaccadeController: ballistic eye jumps with mid-flight suppression.

Scripted fixation sequences drive the tests:
    script = [(0, x0, y0), (onset_ms_1, x1, y1), ...]
Entry 0 is the initial fixation. Each later entry is a saccade *onset*:
the eye jumps from the previous fixation to the new one, taking
21 + 2.2 * amplitude_deg ms (the main sequence), during which vision is
suppressed -- the percept holds the pre-saccadic frame, no smear.
"""

from . import baseline as B


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

    @staticmethod
    def main_sequence_duration_ms(amp_deg):
        return 21.0 + 2.2 * amp_deg

    def add_saccade(self, t_onset_ms, x, y):
        """Append a saccade online (for closed-loop attention drivers).

        The saccade starts at t_onset_ms from wherever fixation is then.
        Callers must add onsets in chronological order.
        """
        fx, fy, _ = self.state_at(t_onset_ms)
        amp_deg = ((x - fx) ** 2 + (y - fy) ** 2) ** 0.5 * self.dva_per_px
        dur = self.main_sequence_duration_ms(amp_deg)
        self.events.append((float(t_onset_ms), dur, (fx, fy),
                            (float(x), float(y))))
        self.script.append((float(t_onset_ms), float(x), float(y)))

    def state_at(self, t_ms):
        """Return (fix_x, fix_y, suppressed) at time t_ms."""
        # Find the latest saccade event at or before t
        current_fix = (self.script[0][1], self.script[0][2])
        for (t_on, dur, frm, to) in self.events:
            if t_ms < t_on:
                break
            if t_ms < t_on + dur:
                return frm[0], frm[1], True  # mid-saccade: hold pre-saccadic
            current_fix = to
        return current_fix[0], current_fix[1], False
