"""Auditory salience map, after Kayser, Petkov, Lippert & Logothetis
(2005): "Mechanisms for allocating auditory attention: an auditory
saliency map."

Three feature channels over the cochlear spectrogram S (frames x 64):
  I -- intensity: normalized loudness structure (the "brightness").
  F -- frequency contrast: spectral edges, |S - blur_freq(S)|
       (a pure tone in noise pops; the analog of a bright edge).
  T -- temporal contrast: onsets/offsets, |S - blur_time(S)|
       (a glass breaking grabs the ears the way a flash grabs the eyes).

Combined with weights, normalized 0..1. The argmax of this map is where
the ears want to go -- the analog of the visual salience map.
"""

import numpy as np
from scipy import ndimage

W_I = 0.4
W_F = 0.3
W_T = 0.6   # onsets dominate, as they do in biology
FREQ_SIGMA = 2.0    # bins
TIME_SIGMA = 6.0    # frames (30ms)
HEAR_FLOOR_DBFS = -60.0  # below this: neural noise, not signal.
    # Log-domain diffs explode at low levels (tiny linear wiggles become
    # tens of dB), so temporal contrast is masked below the floor.


def _norm(m):
    m = np.asarray(m, dtype=np.float32)
    lo, hi = m.min(), m.max()
    if hi - lo < 1e-9:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def salience_map(S):
    """S: (n_frames, 64) log-power spectrogram. Returns dict of
    feature maps plus combined 'sal' in 0..1, same shape."""
    I = _norm(S)
    F = _norm(np.abs(S - ndimage.gaussian_filter1d(
        S, FREQ_SIGMA, axis=1, mode="nearest")))
    Draw = np.abs(S - ndimage.gaussian_filter1d(
        S, TIME_SIGMA, axis=0, mode="nearest"))
    Draw[S < HEAR_FLOOR_DBFS] = 0.0
    T = _norm(Draw)
    sal = _norm(W_I * I + W_F * F + W_T * T)
    return {"I": I, "F": F, "T": T, "sal": sal}
