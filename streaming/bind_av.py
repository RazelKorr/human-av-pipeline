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
a causal-binding measure, until it is read against a chance
context for the run's transient density.

Since 2026-10-14 (binder investigation of 2026-10-13 implemented;
see `improvements-2026-10/REVIEW-DEFRAG-2026-10-14.md` Fix 1) the
PRIMARY chance context is the circular time-shift null
(`shift_null`): the visual stream is rotated by seeded grid
multiples and re-bound with the same greedy matcher, so the null
prices the one-to-one exclusivity, the streams' burst structure,
and the 50 ms grid exactly as the binder experiences them. The
Poisson figure is retained but labelled for what it is -- a
non-exclusive uniform-Poisson at-least-one rate, an upper
reference, NOT the chance rate of this matcher. (Every
"below chance" reading in the corpus traced to reading the
Poisson figure as the matcher's chance rate.) Tight windows also
carry the grid-corrected chance `1 - exp(-lambda_v * (2W + grid))`:
on the moment grid the inclusive window |dt| <= W contains the
slots -W, 0, +W, an effective width of 2W + one grid step that the
continuous formula does not price. `binding_summary` produces the
pair-only fields, so the same code serves live runs and the
backfill script (`scripts/binding_baseline.py`); the shift null
needs the event streams and is computed in `bind()` -- pair-only
summaries state its absence explicitly.
"""

from __future__ import annotations

import math

import numpy as np

MOMENT_MS = 50.0
COINCIDENCE_MS = 250.0  # ±window for audio<->visual coincidence
TIGHT_WINDOWS_MS = (50.0, 100.0)
# |dt| histogram bins in ms; the last bin's hi follows the
# coincidence window (250 ms at the default).
DT_BIN_EDGES_MS = (0.0, 50.0, 100.0, 250.0)


POISSON_LABEL = (
    "non-exclusive uniform-Poisson at-least-one rate: the "
    "probability that AT LEAST ONE visual transient of a "
    "homogeneous Poisson stream falls in the window. An upper "
    "reference, NOT the chance rate of the one-to-one greedy "
    "matcher (which cannot count one transient twice and whose "
    "streams are bursty and grid-quantized, not uniform). "
    "Primary chance context: the shift_null field.")


def poisson_chance_rate(n_visual, duration_s, window_ms):
    """Chance that >=1 transient of a Poisson visual stream falls
    within +-window_ms of a given onset: 1 - exp(-lambda_v * 2W),
    lambda_v = n_visual / duration_s. This is the estimator the
    STREAMING-REPORT addenda used by hand (FF3: 2735 transients /
    2701.0 s at W=250 ms -> 0.397). It is a non-exclusive upper
    reference, not the matcher's chance rate (see POISSON_LABEL).
    Returns None when the density is undefined (no duration or
    no transients)."""
    if not duration_s or duration_s <= 0 or not n_visual:
        return None
    lam = n_visual / float(duration_s)
    return 1.0 - math.exp(-lam * 2.0 * window_ms / 1000.0)


def grid_corrected_chance_rate(n_visual, duration_s, window_ms,
                               grid_ms=MOMENT_MS):
    """Poisson at-least-one chance for an INCLUSIVE window on the
    moment grid: 1 - exp(-lambda_v * (2W + grid)). The slots
    -W, 0, +W all pair, so the effective width exceeds the
    continuous 2W by one grid step -- at W = 50 ms on the 50 ms
    grid that is 150 ms of exposure, not 100 ms. (Binder
    investigation 2026-10-13, recommendation 2.) Still a
    non-exclusive Poisson reference; the shift null remains the
    matcher's chance context. Returns None when the density is
    undefined."""
    if not duration_s or duration_s <= 0 or not n_visual:
        return None
    lam = n_visual / float(duration_s)
    width_s = (2.0 * float(window_ms) + float(grid_ms)) / 1000.0
    return 1.0 - math.exp(-lam * width_s)


def binding_summary(pairs, n_audio, n_visual, duration_s=None,
                    coincidence_ms=COINCIDENCE_MS, duration_source=None):
    """Additive context fields for a bind() result.

    pairs: bind()-style dicts with dt_s (audio_t - visual_t).
    Returns bound_rate, the Poisson chance rate for the coincidence
    window (plus the inputs used, and POISSON_LABEL stating what
    that figure is and is not), bound counts/rates at the tight
    windows (TIGHT_WINDOWS_MS) alongside both the continuous
    Poisson chance and the grid-corrected chance, and a dt
    summary: histogram over grid-aligned |dt| bins (each bin
    annotated with the moment-grid slots it contains) and signed
    asymmetry (fraction of pairs where the audio follows the
    visual, precedes it, or is simultaneous on the moment grid).

    The shift-null context is NOT computable from pairs alone (it
    re-runs the matcher on time-shifted streams); summaries built
    from stored pairs carry shift_null = {available: False, ...}
    so the absence is stated, never silent. bind() attaches the
    real null.
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
        grid_chance = (grid_corrected_chance_rate(n_visual, duration_s,
                                                  w_ms)
                       if duration_s and n_visual else None)
        tight_rate = (n_t / n_audio) if n_audio else None
        tight[str(int(w_ms))] = {
            "window_ms": w_ms,
            "n_bound": n_t,
            "bound_rate": round(tight_rate, 4) if tight_rate is not None else None,
            "fraction_of_bound": round(n_t / n_bound, 4) if n_bound else None,
            "chance_rate_poisson": (
                round(poisson_chance_rate(n_visual, duration_s, w_ms), 4)
                if duration_s and n_visual else None),
            "chance_rate_grid_corrected": (
                round(grid_chance, 4) if grid_chance is not None else None),
            "excess_over_grid_chance": (
                round(tight_rate - grid_chance, 4)
                if tight_rate is not None and grid_chance is not None
                else None),
        }

    edges = list(DT_BIN_EDGES_MS)
    edges[-1] = max(edges[-1], float(coincidence_ms))
    hist = []
    for i_bin, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        n_b = sum(1 for p in pairs
                  if lo <= abs(p["dt_s"]) * 1000.0 < hi
                  or (hi == edges[-1]
                      and abs(p["dt_s"]) * 1000.0 == hi))
        # Moment-grid slots (multiples of MOMENT_MS) this bin
        # contains; the last bin also contains its right edge.
        slots = [s for s in range(int(lo), int(hi), int(MOMENT_MS))]
        if i_bin == len(edges) - 2 and int(hi) % int(MOMENT_MS) == 0:
            slots.append(int(hi))
        hist.append({"lo_ms": lo, "hi_ms": hi, "n": n_b,
                     "fraction": round(n_b / n_bound, 4) if n_bound else None,
                     "grid_slots_ms": slots})
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
        "chance_rate_poisson_label": POISSON_LABEL,
        "shift_null": {
            "available": False,
            "reason": ("requires the event streams (the null re-runs "
                       "the matcher on time-shifted streams); not "
                       "computable from stored pairs alone"),
        },
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


def _to_ms(t_s):
    """Event time (s) -> integer milliseconds. All matching in this
    module is done on ms-quantized times (Fix 3, 2026-10-14): the
    streams are produced on a 50 ms grid and every output already
    rounds to ms, so quantization discards only sub-microsecond
    float representation dust -- which otherwise decides on-grid
    ties, making the same events bind differently depending on
    how their times happen to be represented (binder investigation
    2026-10-13, recommendation 4)."""
    return int(round(float(t_s) * 1000.0))


def shift_null_context(onsets_ms, vtrans_ms, duration_s,
                       coincidence_ms=COINCIDENCE_MS, tight_ms=50,
                       n_shifts=200, seed=20261014, pairs=None):
    """Circular time-shift null for a bind() result (Fix 1).

    The visual stream is rotated by seeded multiples of the 50 ms
    moment grid (uniform in [10 s, duration - 10 s], the binder
    investigation's protocol) and re-bound with the same greedy
    one-to-one matcher, so the null prices matcher exclusivity,
    stream burst structure, and grid effects together. Observed
    stats come from `pairs` when given (bind()'s own output),
    else from an unshifted fast match. Returns a dict with, per
    statistic (wide bound rate, |dt| <= tight_ms rate, dt = 0
    count, mean |dt|): observed value, null mean/sd/p5/p95 and
    the observed value's percentile in the null distribution --
    or {available: False, reason} when the null is undefined.
    """
    import bisect
    import random

    if not duration_s or duration_s <= 0:
        return {"available": False,
                "reason": "no positive duration"}
    if not onsets_ms:
        return {"available": False, "reason": "no audio onsets"}
    if not vtrans_ms:
        return {"available": False, "reason": "no visual transients"}
    dur_ms = int(round(float(duration_s) * 1000.0))
    grid_ms = int(round(MOMENT_MS))
    lo_ms = 10000 if dur_ms > 20000 + grid_ms else grid_ms
    hi_ms = dur_ms - lo_ms
    if hi_ms <= lo_ms:
        return {"available": False,
                "reason": "duration too short for shifted nulls"}

    def _match(a_list, v_list):
        order = sorted(range(len(v_list)), key=lambda i: v_list[i][0])
        vt_sorted = [v_list[i][0] for i in order]
        used = set()
        dts = []
        for at, _s in sorted(a_list, key=lambda p: -p[1]):
            lo = bisect.bisect_left(vt_sorted, at - int(coincidence_ms))
            hi = bisect.bisect_right(vt_sorted, at + int(coincidence_ms))
            best, bestd = None, int(coincidence_ms)
            for j in range(lo, hi):
                if order[j] in used:
                    continue
                d = abs(at - vt_sorted[j])
                if d <= bestd:
                    best, bestd = order[j], d
            if best is not None:
                used.add(best)
                dts.append(at - v_list[best][0])
        return dts

    def _stats(dts):
        n = len(dts)
        return {
            "bound_rate": (n / len(onsets_ms)) if onsets_ms else None,
            "tight_rate": (sum(1 for d in dts if abs(d) <= tight_ms)
                           / len(onsets_ms)) if onsets_ms else None,
            "dt0_count": sum(1 for d in dts if d == 0),
            "mean_abs_dt_ms": (float(np.mean([abs(d) for d in dts]))
                               if dts else None),
        }

    if pairs is not None:
        obs_dts = [int(round(p["dt_s"] * 1000.0)) for p in pairs]
    else:
        obs_dts = _match(onsets_ms, vtrans_ms)
    observed = _stats(obs_dts)

    rng = random.Random(seed)
    span_steps = (hi_ms - lo_ms) // grid_ms
    null = {"bound_rate": [], "tight_rate": [], "dt0_count": [],
            "mean_abs_dt_ms": []}
    for _ in range(int(n_shifts)):
        shift = lo_ms + grid_ms * rng.randrange(span_steps + 1)
        v_shifted = [((vt + shift) % dur_ms, s) for vt, s in vtrans_ms]
        st = _stats(_match(onsets_ms, v_shifted))
        for key in null:
            null[key].append(st[key])

    def _block(key):
        vals = np.asarray([v for v in null[key] if v is not None],
                          dtype=np.float64)
        obs = observed[key]
        if len(vals) == 0 or obs is None:
            return {"observed": obs, "null_mean": None, "null_sd": None,
                    "null_p5": None, "null_p95": None,
                    "observed_percentile": None}
        return {
            "observed": (round(float(obs), 4)
                         if isinstance(obs, float) else obs),
            "null_mean": round(float(np.mean(vals)), 4),
            "null_sd": round(float(np.std(vals, ddof=1)), 4)
            if len(vals) > 1 else 0.0,
            "null_p5": round(float(np.percentile(vals, 5)), 4),
            "null_p95": round(float(np.percentile(vals, 95)), 4),
            "observed_percentile": round(
                float(100.0 * np.mean(vals <= obs)), 1),
        }

    return {
        "available": True,
        "method": ("circular time-shift of the visual stream by "
                   "seeded multiples of the 50 ms moment grid "
                   "(uniform in [10 s, duration - 10 s]); the same "
                   "greedy one-to-one matcher re-run per shift"),
        "n_shifts": int(n_shifts),
        "seed": seed,
        "bound_rate": _block("bound_rate"),
        "tight_rate_50ms": _block("tight_rate"),
        "dt0_count": _block("dt0_count"),
        "mean_abs_dt_ms": _block("mean_abs_dt_ms"),
    }


def bind(onsets, vtrans, coincidence_ms=COINCIDENCE_MS, duration_s=None,
         shift_null=True, n_shifts=200, null_seed=20261014):
    """onsets: [(t_s, strength)] audio. vtrans: [(t_s, strength)]
    visual. Returns dict with bound_pairs (audio_t, visual_t, dt,
    audio_strength, visual_strength), unbound_audio, unbound_visual.
    Each onset binds to at most one transient (nearest in window)
    and vice versa -- greedy by descending audio strength.

    duration_s (optional): the media duration the events span,
    used for the chance context. When omitted it is estimated
    from the latest event timestamp and flagged as such in
    chance_inputs.duration_source. The context fields are
    additive; the original fields are unchanged.

    Matching is done on ms-quantized times (Fix 3, 2026-10-14;
    see _to_ms). With shift_null=True (default) the result's
    shift_null field carries the circular time-shift null
    (Fix 1); pass n_shifts / null_seed to tune it, or
    shift_null=False to skip the computation.
    """
    a_ms = [(_to_ms(t), s) for t, s in onsets]
    v_ms = [(_to_ms(t), s) for t, s in vtrans]
    w_ms = int(round(float(coincidence_ms)))
    pairs, used_v = [], set()
    unbound_a = []
    for at, astr in sorted(a_ms, key=lambda p: -p[1]):
        best, bestd = None, w_ms
        for vi, (vt, vstr) in enumerate(v_ms):
            if vi in used_v:
                continue
            d = abs(at - vt)
            if d <= bestd:
                best, bestd = vi, d
        if best is not None:
            used_v.add(best)
            vt, vstr = v_ms[best]
            pairs.append({"audio_t_s": round(at / 1000.0, 3),
                          "visual_t_s": round(vt / 1000.0, 3),
                          "dt_s": round((at - vt) / 1000.0, 3),
                          "audio_strength": round(astr, 4),
                          "visual_strength": round(vstr, 4)})
        else:
            unbound_a.append({"t_s": round(at / 1000.0, 3),
                              "strength": round(astr, 4)})
    unbound_v = [{"t_s": round(vt / 1000.0, 3),
                  "strength": round(vstr, 4)}
                 for vi, (vt, vstr) in enumerate(v_ms)
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
        latest_ms = max([t for t, _ in a_ms] + [t for t, _ in v_ms],
                        default=0)
        dur = (latest_ms / 1000.0) if latest_ms > 0 else None
        dur_src = "max_event_time_estimate"
    result.update(binding_summary(pairs, len(onsets), len(vtrans),
                                  duration_s=dur,
                                  coincidence_ms=coincidence_ms,
                                  duration_source=dur_src))
    if shift_null:
        result["shift_null"] = shift_null_context(
            a_ms, v_ms, dur, coincidence_ms=coincidence_ms,
            n_shifts=n_shifts, seed=null_seed, pairs=pairs)
    return result
