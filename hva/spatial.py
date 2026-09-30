"""hva.spatial: where is the sound? (binaural cues from stereo)

The mono cochlea hears *what*; this module hears *where*. Two cues,
the same ones human hearing uses:

- ILD (interaural level difference): which ear is louder, per moment.
  Strong at high frequencies, robust in mixed content. This is the
  workhorse.
- ITD (interaural time difference): which ear got it first, from the
  cross-correlation lag around sharp onsets. Precise for transients,
  mostly a low-frequency cue in biology.

Output is a pan in [-1, +1] (-1 = hard left, +1 = hard right, 0 =
center), with a confidence that goes to zero in near-silence (no level,
no location -- don't hallucinate a direction for noise).

The pan feeds Level 2 cross-modal bias: a bang on the left tugs gaze
left (the "pip and pop" effect, Van der Burg et al. 2008).
"""
import numpy as np
from scipy.io import wavfile

# ILD (dB) that maps to full-scale pan. 12 dB is roughly a hard-panned
# source in a typical mix; beyond that it's just "very left".
PAN_FULL_DB = 12.0
# Below this broadband level (dBFS) the pan is untrustworthy.
PAN_FLOOR_DBFS = -50.0


def load_stereo(path):
    sr, d = wavfile.read(path)
    x = np.asarray(d, dtype=np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    peak = max(1.0, np.abs(x).max())
    return sr, x[:, 0] / peak, x[:, 1] / peak


def moment_pan(l, r, sr, moment_s=0.1):
    """Per-moment broadband pan from ILD.

    Returns (pan, conf): pan in [-1,1], conf in [0,1].
    """
    n = int(sr * moment_s)
    n_mom = (len(l) - n) // n + 1
    pan = np.zeros(n_mom)
    conf = np.zeros(n_mom)
    for m in range(n_mom):
        seg_l = l[m * n:(m + 1) * n]
        seg_r = r[m * n:(m + 1) * n]
        e_l = np.mean(seg_l ** 2) + 1e-12
        e_r = np.mean(seg_r ** 2) + 1e-12
        lvl = 10.0 * np.log10(max(e_l, e_r))
        ild = 10.0 * np.log10(e_l / e_r)  # + = left louder
        p = -np.clip(ild / PAN_FULL_DB, -1.0, 1.0)  # -1 = left
        # confidence ramps in above the floor
        c = float(np.clip((lvl - PAN_FLOOR_DBFS) / 20.0, 0.0, 1.0))
        pan[m] = p
        conf[m] = c
    return pan, conf


def onset_itd(l, r, sr, t_s, win_s=0.05, max_lag_ms=1.0):
    """Refine a transient's direction with ITD.

    Cross-correlates L/R in a short window around t_s; the lag of peak
    correlation is the interaural time difference. Returns lag in ms
    (+ = left leads, i.e. source on the left), or None if unclear.
    """
    n = int(sr * win_s)
    c = int(t_s * sr)
    a = max(0, c - n // 2)
    seg_l = l[a:a + n]
    seg_r = r[a:a + n]
    if len(seg_l) < n // 2:
        return None
    max_lag = int(sr * max_lag_ms / 1000.0)
    cc = np.correlate(seg_l, seg_r, mode="full")
    lags = np.arange(-len(seg_l) + 1, len(seg_l))
    m = np.abs(lags) <= max_lag
    if not np.any(m):
        return None
    best = lags[m][int(np.argmax(cc[m]))]
    # normalize by peak auto-correlation as a clarity check
    clarity = cc[m].max() / (np.sqrt((seg_l ** 2).sum()
                                     * (seg_r ** 2).sum()) + 1e-12)
    if clarity < 0.3:
        return None
    return float(best) / sr * 1000.0  # ms; + = left leads
