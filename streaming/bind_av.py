"""Audio-visual binding: the "that made that" link.

Correlates audio onsets (from hva.online.pick_onsets) with visual
transients (frame-energy diffs on the master 50 ms moment grid)
inside a coincidence window. A bound pair is one percept arriving
through two senses -- the whole point of a sensorium over two
stapled pipelines.

Unbound onsets are characterized, not discarded: a strong unbound
onset is a missed binding (or score/crowd bed the eyes never saw);
a weak one is texture.
"""

from __future__ import annotations

import numpy as np

MOMENT_MS = 50.0
COINCIDENCE_MS = 250.0  # ±window for audio<->visual coincidence


def visual_transients(frame_energies, frame_times_ms, moment_ms=MOMENT_MS,
                      k=4.0):
    """frame_energies: per-frame energies (driver.energies);
    frame_times_ms: matching timestamps. Returns (t_s, strength)
    transient peaks on the moment grid: local maxima of the
    per-moment max |energy diff|, thresholded at median + k*MAD.
    """
    e = np.asarray(frame_energies, dtype=np.float64)
    t = np.asarray(frame_times_ms, dtype=np.float64) / 1000.0
    if len(e) < 2:
        return []
    d = np.abs(np.diff(e))
    td = t[1:]  # diff i = change into frame i
    dt = moment_ms / 1000.0
    n_mom = int(np.ceil(t[-1] / dt)) + 1
    curve = np.zeros(n_mom, dtype=np.float64)
    idx = np.clip((td / dt).astype(int), 0, n_mom - 1)
    np.maximum.at(curve, idx, d)
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med))) + 1e-12
    thr = med + k * mad
    peaks = [i for i in range(n_mom)
             if curve[i] > thr
             and curve[i] >= curve[max(0, i - 2):i + 3].max()]
    return [(i * dt, float(curve[i])) for i in peaks]


def bind(onsets, vtrans, coincidence_ms=COINCIDENCE_MS):
    """onsets: [(t_s, strength)] audio. vtrans: [(t_s, strength)]
    visual. Returns dict with bound_pairs (audio_t, visual_t, dt,
    audio_strength, visual_strength), unbound_audio, unbound_visual.
    Each onset binds to at most one transient (nearest in window)
    and vice versa -- greedy by descending audio strength.
    """
    w = coincidence_ms / 1000.0
    pairs, used_v = [], set()
    unbound_a = []
    for at, astr in sorted(onsets, key=lambda p: -p[1]):
        best, bestd = None, w
        for vi, (vt, vstr) in enumerate(vtrans):
            if vi in used_v:
                continue
            d = abs(at - vt)
            if d <= bestd:
                best, bestd = vi, d
        if best is not None:
            used_v.add(best)
            vt, vstr = vtrans[best]
            pairs.append({"audio_t_s": round(at, 3),
                          "visual_t_s": round(vt, 3),
                          "dt_s": round(at - vt, 3),
                          "audio_strength": round(astr, 4),
                          "visual_strength": round(vstr, 4)})
        else:
            unbound_a.append({"t_s": round(at, 3),
                              "strength": round(astr, 4)})
    unbound_v = [{"t_s": round(vt, 3), "strength": round(vstr, 4)}
                 for vi, (vt, vstr) in enumerate(vtrans)
                 if vi not in used_v]
    pairs.sort(key=lambda p: p["audio_t_s"])
    return {
        "coincidence_ms": coincidence_ms,
        "n_audio_onsets": len(onsets),
        "n_visual_transients": len(vtrans),
        "n_bound": len(pairs),
        "bound_pairs": pairs,
        "unbound_audio": sorted(unbound_a, key=lambda o: o["t_s"]),
        "unbound_visual": sorted(unbound_v, key=lambda o: o["t_s"]),
    }
