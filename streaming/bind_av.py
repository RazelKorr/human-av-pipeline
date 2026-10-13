"""Audio-visual binding: the "that made that" link.

Correlates audio onsets (from hva.online.pick_onsets) with visual
transients (frame-energy diffs on the master 50 ms moment grid)
inside a coincidence window. A bound pair is one percept arriving
through two senses -- the whole point of a sensorium over two
stapled pipelines.

Unbound onsets are characterized, not discarded: a strong unbound
onset is a missed binding (or score/crowd bed the eyes never saw);
a weak one is texture.

Since 2026-10-12 the output also carries its own chance context
(additive fields only; the matching algorithm and the original
fields are unchanged): a raw bound rate is a coincidence rate, not
a causal-binding measure, until it is read against the Poisson
chance rate for the run's transient density -- the baseline the
project's reviews used to compute by hand (see STREAMING-REPORT
addenda: FF3 35.9% bound vs 39.7% chance; Dr Tran ep4 50.7% vs
~50%). `binding_summary` produces those fields from a pairs list,
so the same code serves live runs and the backfill script
(`scripts/binding_baseline.py`).
"""

from __future__ import annotations

import math

import numpy as np

MOMENT_MS = 50.0
COINCIDENCE_MS = 250.0  # ±window for audio<->visual coincidence
TIGHT_WINDOWS_MS = (50.0, 100.0)
# |dt| histogram bins in ms; the last bin's hi follows the
# coincidence window (250 ms at the default).
DT_BIN_EDGES_MS = (0.0, 25.0, 50.0, 100.0, 250.0)


def poisson_chance_rate(n_visual, duration_s, window_ms):
    """Chance that >=1 transient of a Poisson visual stream falls
    within +-window_ms of a given onset: 1 - exp(-lambda_v * 2W),
    lambda_v = n_visual / duration_s. This is the estimator the
    STREAMING-REPORT addenda used by hand (FF3: 2735 transients /
    2701.0 s at W=250 ms -> 0.397). Returns None when the density
    is undefined (no duration or no transients)."""
    if not duration_s or duration_s <= 0 or not n_visual:
        return None
    lam = n_visual / float(duration_s)
    return 1.0 - math.exp(-lam * 2.0 * window_ms / 1000.0)


def binding_summary(pairs, n_audio, n_visual, duration_s=None,
                    coincidence_ms=COINCIDENCE_MS, duration_source=None):
    """Additive context fields for a bind() result.

    pairs: bind()-style dicts with dt_s (audio_t - visual_t).
    Returns bound_rate, the Poisson chance rate for the coincidence
    window (plus the inputs used), bound counts/rates at the tight
    windows (TIGHT_WINDOWS_MS) alongside, and a dt summary:
    histogram over |dt| bins and signed asymmetry (fraction of
    pairs where the audio follows the visual, precedes it, or is
    simultaneous on the moment grid).
    """
    n_bound = len(pairs)
    bound_rate = (n_bound / n_audio) if n_audio else None

    if duration_s:
        lam = (n_visual / float(duration_s)) if n_visual else None
        chance = poisson_chance_rate(n_visual, duration_s, coincidence_ms)
    else:
        lam, chance = None, None

    tight = {}
    for w_ms in TIGHT_WINDOWS_MS:
        w = w_ms / 1000.0
        n_t = sum(1 for p in pairs if abs(p["dt_s"]) <= w)
        tight[str(int(w_ms))] = {
            "window_ms": w_ms,
            "n_bound": n_t,
            "bound_rate": round(n_t / n_audio, 4) if n_audio else None,
            "fraction_of_bound": round(n_t / n_bound, 4) if n_bound else None,
            "chance_rate_poisson": (
                round(poisson_chance_rate(n_visual, duration_s, w_ms), 4)
                if duration_s and n_visual else None),
        }

    edges = list(DT_BIN_EDGES_MS)
    edges[-1] = max(edges[-1], float(coincidence_ms))
    hist = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n_b = sum(1 for p in pairs
                  if lo <= abs(p["dt_s"]) * 1000.0 < hi
                  or (hi == edges[-1]
                      and abs(p["dt_s"]) * 1000.0 == hi))
        hist.append({"lo_ms": lo, "hi_ms": hi, "n": n_b,
                     "fraction": round(n_b / n_bound, 4) if n_bound else None})
    n_after = sum(1 for p in pairs if p["dt_s"] > 0)
    n_before = sum(1 for p in pairs if p["dt_s"] < 0)
    n_zero = n_bound - n_after - n_before
    abs_dt_ms = [abs(p["dt_s"]) * 1000.0 for p in pairs]

    return {
        "bound_rate": round(bound_rate, 4) if bound_rate is not None else None,
        "chance_rate_poisson": round(chance, 4) if chance is not None else None,
        "excess_over_chance": (
            round(bound_rate - chance, 4)
            if bound_rate is not None and chance is not None else None),
        "chance_inputs": {
            "n_visual_transients": n_visual,
            "duration_s": round(float(duration_s), 3) if duration_s else None,
            "duration_source": duration_source,
            "window_ms": coincidence_ms,
            "lambda_visual_per_s": round(lam, 5) if lam else None,
        },
        "tight_windows": tight,
        "dt_summary": {
            "n_pairs": n_bound,
            "histogram_abs_dt_ms": hist,
            "asymmetry": {
                "audio_after_visual": n_after,
                "audio_before_visual": n_before,
                "simultaneous": n_zero,
                "frac_audio_after_visual": (
                    round(n_after / n_bound, 4) if n_bound else None),
                "frac_audio_before_visual": (
                    round(n_before / n_bound, 4) if n_bound else None),
                "frac_simultaneous": (
                    round(n_zero / n_bound, 4) if n_bound else None),
            },
            "mean_abs_dt_ms": (
                round(float(np.mean(abs_dt_ms)), 1) if abs_dt_ms else None),
            "median_abs_dt_ms": (
                round(float(np.median(abs_dt_ms)), 1) if abs_dt_ms else None),
        },
    }


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


def bind(onsets, vtrans, coincidence_ms=COINCIDENCE_MS, duration_s=None):
    """onsets: [(t_s, strength)] audio. vtrans: [(t_s, strength)]
    visual. Returns dict with bound_pairs (audio_t, visual_t, dt,
    audio_strength, visual_strength), unbound_audio, unbound_visual.
    Each onset binds to at most one transient (nearest in window)
    and vice versa -- greedy by descending audio strength.

    duration_s (optional): the media duration the events span,
    used for the Poisson chance baseline. When omitted it is
    estimated from the latest event timestamp and flagged as such
    in chance_inputs.duration_source. The baseline fields are
    additive; the original fields are unchanged.
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
    result = {
        "coincidence_ms": coincidence_ms,
        "n_audio_onsets": len(onsets),
        "n_visual_transients": len(vtrans),
        "n_bound": len(pairs),
        "bound_pairs": pairs,
        "unbound_audio": sorted(unbound_a, key=lambda o: o["t_s"]),
        "unbound_visual": sorted(unbound_v, key=lambda o: o["t_s"]),
    }
    if duration_s:
        dur, dur_src = float(duration_s), "caller"
    else:
        latest = max([t for t, _ in onsets] + [t for t, _ in vtrans],
                     default=0.0)
        dur, dur_src = (latest if latest > 0 else None), \
            "max_event_time_estimate"
    result.update(binding_summary(pairs, len(onsets), len(vtrans),
                                  duration_s=dur,
                                  coincidence_ms=coincidence_ms,
                                  duration_source=dur_src))
    return result
