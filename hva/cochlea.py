"""Cochlea: the ear's front end, modeled the way retina.py models the eye's.

The basilar membrane is tonotopic -- pitch is place-coded, the way the
retina place-codes space. So the cochlea's output is a log-frequency
spectrogram: rows are time (5ms steps), columns are 64 log-spaced
frequency bins from 50 Hz to 8 kHz. That 64-bin vector is audition's
answer to vision's 56x56 salience thumbnail: the whole scene, coarse,
every few milliseconds.
"""

import numpy as np
from scipy import signal

SR = 16000          # analysis sample rate
WIN_MS = 20.0       # STFT window -- ~one auditory integration slice
HOP_MS = 5.0        # hop between slices
N_BINS = 64         # log-frequency bins (cf. SMALL=56 for vision)
FMIN = 50.0
FMAX = 8000.0


def _log_bin_edges(n_bins=N_BINS, fmin=FMIN, fmax=FMAX):
    return np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)


def bin_to_hz(bins):
    """Center frequency in Hz for bin index/indices."""
    edges = _log_bin_edges()
    return float(np.sqrt(edges[int(bins)] * edges[int(bins) + 1])) \
        if np.ndim(bins) == 0 else np.sqrt(edges[bins] * edges[bins + 1])


def stft_log(x, sr=SR):
    """x: mono float32 in -1..1. Returns (times_s, freqs_hz, S)
    with S = log-power spectrogram, shape (n_frames, N_BINS)."""
    x = np.asarray(x, dtype=np.float32)
    nperseg = int(sr * WIN_MS / 1000.0)
    noverlap = nperseg - int(sr * HOP_MS / 1000.0)
    f, t, Z = signal.stft(x, fs=sr, window="hann", nperseg=nperseg,
                          noverlap=noverlap, boundary="zeros", padded=True)
    power = np.abs(Z) ** 2  # (n_freq, n_frames)
    edges = _log_bin_edges()
    S = np.zeros((Z.shape[1], N_BINS), np.float32)
    for b in range(N_BINS):
        m = (f >= edges[b]) & (f < edges[b + 1])
        if m.any():
            S[:, b] = power[m].mean(axis=0)
    # log power, absolute dBFS (no per-file normalization -- the
    # auditory nerve has an absolute threshold, and per-file norming
    # turns digital silence into fake 70dB onsets)
    S = 10.0 * np.log10(S + 1e-10)
    return t, bin_to_hz(np.arange(N_BINS)), S


def loudness(S):
    """Per-frame loudness proxy: mean dBFS across bins, 0..1."""
    l = (S.mean(axis=1) + 80.0) / 80.0
    return np.clip(l, 0.0, 1.0)
