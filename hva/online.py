"""Online audio front end: the ear's magno channel.

Coarse, cheap, always-on — the auditory answer to hvp/motion.py.
Per 50 ms perceptual moment (the master grid shared with vision):
RMS energy, spectral flux (onset strength), spectral centroid
(brightness). Three floats per moment: the whole of v1 hearing
that isn't words.

Design notes:
- Moment features are computed per chunk (streaming-pure); only the
  onset *decision* (peak-picking) is post-hoc, mirroring how the
  visual event map is produced at end of run. This is documented,
  not hidden.
- Spectral flux is ABSOLUTE (sum of positive spectral diffs), not
  loudness-relative: a roar over an already-loud bed is still a big
  transient, and relative normalization inflates silence-to-sound
  transitions astronomically while suppressing exactly the big
  absolute events the predictions care about (measured 2026-10-03:
  relative flux ranked the hyperspace roars 170th and 202nd).
  Loudness-invariance comes from the adaptive median+MAD threshold
  in pick_onsets, not the flux itself. An RMS floor gate keeps
  digital silence from onsetting on encode noise.
- Flux continuity across chunk boundaries: the caller threads the
  previous chunk's last magnitude frame through (prev_mag).
"""

from __future__ import annotations

import numpy as np

SR = 16000          # analysis sample rate (matches hva.cochlea)
MOMENT_MS = 50.0    # master perceptual grid, shared with vision
SPM = int(SR * MOMENT_MS / 1000.0)  # 800 samples per moment
NFFT = 1024
FREQ_FLOOR = 1e-12
RMS_FLOOR = 1e-3    # below this: digital silence, not signal


def moment_features(x, prev_mag=None):
    """x: (n_moments, SPM) float32 mono @16kHz.

    Returns (rms, flux, centroid), each (n_moments,) float32, plus
    last_mag (513,) float32 to thread into the next chunk for flux
    continuity. flux[i] is the onset strength INTO moment i
    (change from moment i-1 to moment i); flux[0] is 0 unless
    prev_mag is given.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != SPM:
        raise ValueError(f"moment_features needs (n, {SPM}), got {x.shape}")
    n = x.shape[0]
    rms = np.sqrt((x.astype(np.float64) ** 2).mean(axis=1)).astype(np.float32)
    win = np.hanning(SPM).astype(np.float32)
    mag = np.abs(np.fft.rfft(x * win[None, :], n=NFFT)).astype(np.float32)
    if prev_mag is not None:
        mall = np.concatenate([np.asarray(prev_mag, dtype=np.float32)[None, :],
                               mag], axis=0)
    else:
        mall = mag
    # Absolute spectral flux: positive spectral change per moment.
    # Deliberately NOT loudness-normalized (see module docstring).
    d = np.diff(mall, axis=0)
    flux = np.maximum(d, 0.0).sum(axis=1).astype(np.float32)
    if prev_mag is None:
        flux = np.concatenate([np.zeros(1, dtype=np.float32), flux])
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR).astype(np.float32)
    tot = mag.sum(axis=1) + FREQ_FLOOR
    centroid = ((mag * freqs[None, :]).sum(axis=1) / tot).astype(np.float32)
    return rms, flux, centroid, mag[-1].copy()


def pick_onsets(flux, rms, moment_ms=MOMENT_MS, k=3.0,
                min_gap_ms=250.0, rms_floor=RMS_FLOOR,
                local_ms=2000.0, abs_floor=0.1):
    """Peak-pick onsets from a full flux curve (post-hoc, like the
    visual event map). Returns list of (t_s, strength) sorted by time.

    The threshold is LOCAL: median(flux) + k * MAD(flux) over a
    ±local_ms window. A global threshold fails on music-heavy audio
    (measured 2026-10-03: absolute flux has median ~107, so a global
    median+MAD bar suppressed real transients at 64.1 s and 108.8 s
    that sit below it). Candidates must also clear the RMS floor.
    Local maxima in a ±2-moment neighborhood; min-gap enforced
    greedily by descending strength.
    """
    from scipy.ndimage import median_filter
    flux = np.asarray(flux, dtype=np.float32)
    rms = np.asarray(rms, dtype=np.float32)
    n = len(flux)
    w = max(1, int(round(local_ms / moment_ms)))
    lmed = median_filter(flux, size=2 * w + 1, mode="reflect")
    lmad = median_filter(np.abs(flux - lmed), size=2 * w + 1,
                         mode="reflect") + 1e-12
    thr = lmed + k * lmad
    # Absolute floor: when the local signal is essentially zero (pure
    # tones, digital silence), the adaptive threshold collapses to
    # numerical noise (~1e-11) and everything fires. 0.1 sits ten
    # orders above numerical noise and below the smallest test
    # transient (0.41 for a 0.5 ms click); real film onsets are
    # hundreds.
    thr = np.maximum(thr, abs_floor)
    cand = [i for i in range(n)
            if flux[i] > thr[i] and rms[i] > rms_floor
            and flux[i] >= flux[max(0, i - 2):i + 3].max()]
    gap = max(1, int(round(min_gap_ms / moment_ms)))
    cand.sort(key=lambda i: -float(flux[i]))
    kept, taken = [], np.zeros(n, dtype=bool)
    for i in cand:
        # min separation: no kept peak within `gap` moments
        if taken[i]:
            continue
        kept.append(i)
        lo, hi = max(0, i - gap + 1), min(n, i + gap)
        taken[lo:hi] = True
    kept.sort()
    dt = moment_ms / 1000.0
    return [(i * dt, float(flux[i])) for i in kept]
