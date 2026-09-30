"""Perceptual moments for hearing, mirroring vision's 100ms boxcar.

The auditory system doesn't hear a continuous stream either -- it
integrates over tens-of-milliseconds windows (the exact grain is still
debated; 100ms here is the scaffold's working grain, matching the
vision pipeline's moments so the two can share a clock later).
Each moment collapses 20 cochlear frames (5ms each) into one: mean
spectrogram, mean salience, mean loudness. Attention decisions happen
at moment boundaries -- the auditory analog of the 3Hz saccade clock,
except the "eye" here is a frequency band, not a place on a screen.
"""

import numpy as np

from .salience import HEAR_FLOOR_DBFS as HFLOOR

MOMENT_MS = 100.0
FRAMES_PER_MOMENT = 20  # 20 x 5ms


def moments(S, sal, times):
    """S, sal: (n_frames, 64). times: frame times in seconds.
    Returns list of dicts, one per 100ms moment."""
    n = len(S)
    out = []
    for i in range(0, n - FRAMES_PER_MOMENT + 1, FRAMES_PER_MOMENT):
        sl = slice(i, i + FRAMES_PER_MOMENT)
        out.append({
            "t0": float(times[i]),
            "t1": float(times[i + FRAMES_PER_MOMENT - 1]),
            "spec": S[sl].mean(axis=0),
            "sal": sal[sl].mean(axis=0),
            "I": None,  # filled by caller if needed
            # loudness proxy: mean LINEAR power -> dBFS. (Mean of
            # log-power is dominated by the silence floor and reads pure
            # tones as silent -- energy must be averaged before the log.)
            "loud": float(np.clip((10.0 * np.log10(
                (10.0 ** (S[sl] / 10.0)).mean() + 1e-12) + 80.0) / 80.0,
                0, 1)),
            # Tmax: strongest short-term change, masked below the hearing
            # floor. (Log-domain diffs explode on near-silent bins --
            # tiny linear wiggles become tens of dB -- so the same floor
            # the salience T channel uses applies here.)
            "Tmax": float(np.where(
                np.maximum(S[sl][:-1], S[sl][1:]) < HFLOOR, 0.0,
                np.abs(np.diff(S[sl], axis=0))).max()),
        })
    return out
