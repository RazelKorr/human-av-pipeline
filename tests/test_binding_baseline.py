"""Binding chance-baseline tests (Phase 2, 2026-10-12).

The binder's raw bound rate is a coincidence rate; these tests pin
the Poisson chance machinery that now ships alongside it:

- the estimator reproduces the STREAMING-REPORT hand baselines;
- on independent synthetic Poisson streams, measured bound rate
  tracks the chance rate, and the tight-window fraction of bound
  pairs tracks the window-ratio expectation (uniform dt);
- on a synthetic stream with transients planted just after onsets,
  the tight-dt excess and the signed asymmetry show it clearly.

Deterministic seeds throughout.

2026-10-14 (defrag review Fix 1 + Fix 3): the Poisson figure is
now labelled as a non-exclusive upper reference; the primary
chance context is the circular time-shift null; tight windows
carry the grid-corrected chance; dt histogram bins are aligned
to the 50 ms moment grid; and bind() matches on ms-quantized
times so pair output no longer depends on float representation.
New tests pin each of those.
"""

import numpy as np

from streaming.bind_av import (bind, binding_summary,
                               grid_corrected_chance_rate,
                               poisson_chance_rate)


def _poisson_times(rng, rate_per_s, duration_s):
    n = rng.poisson(rate_per_s * duration_s)
    return sorted(rng.uniform(0.0, duration_s, size=n).tolist())


def test_estimator_reproduces_report_hand_baselines():
    # STREAMING-REPORT FF3 addendum: 2735 transients / 2701.0 s,
    # +-250 ms -> 39.7% chance (bound was 35.9%, below chance).
    assert abs(poisson_chance_rate(2735, 2701.0, 250.0) - 0.397) < 0.002
    # Dr Tran ep4 addendum: 601 / 436.1 s -> ~0.50 (bound 50.7%).
    assert abs(poisson_chance_rate(601, 436.1, 250.0) - 0.50) < 0.005
    # Degenerate inputs are undefined, not zero.
    assert poisson_chance_rate(0, 100.0, 250.0) is None
    assert poisson_chance_rate(10, 0.0, 250.0) is None


def test_bind_output_keeps_original_fields():
    b = bind([(1.0, 5.0), (3.0, 1.0)], [(1.02, 9.0), (9.0, 1.0)],
             duration_s=10.0)
    for key in ("coincidence_ms", "n_audio_onsets",
                "n_visual_transients", "n_bound", "bound_pairs",
                "unbound_audio", "unbound_visual"):
        assert key in b
    # New additive fields ride along.
    for key in ("bound_rate", "chance_rate_poisson", "chance_inputs",
                "tight_windows", "dt_summary"):
        assert key in b
    assert b["chance_inputs"]["duration_source"] == "caller"
    assert b["chance_inputs"]["n_visual_transients"] == 2


def test_poisson_streams_bound_rate_tracks_chance():
    rng = np.random.default_rng(20261012)
    duration = 1200.0
    onsets = [(t, 1.0) for t in _poisson_times(rng, 0.5, duration)]
    vtrans = [(t, 1.0) for t in _poisson_times(rng, 1.0, duration)]
    b = bind(onsets, vtrans, duration_s=duration)
    chance = poisson_chance_rate(len(vtrans), duration, 250.0)
    assert b["chance_rate_poisson"] == round(chance, 4)
    # Independent streams: bound rate should sit at chance (the
    # greedy 1:1 matching shaves a little; +-0.08 is ~4 sigma of
    # sampling noise at this n and covers the shave).
    assert abs(b["bound_rate"] - chance) < 0.08
    # Tight fraction of bound pairs ~ window ratio 100/500 = 0.20.
    tight50 = b["tight_windows"]["50"]["fraction_of_bound"]
    assert abs(tight50 - 0.20) < 0.08
    # dt histogram partitions the bound pairs; bins are aligned
    # to the 50 ms moment grid (2026-10-14: the old [25, 50) bin
    # was structurally empty for grid-produced streams).
    hist = b["dt_summary"]["histogram_abs_dt_ms"]
    assert sum(h["n"] for h in hist) == b["n_bound"]
    assert [(h["lo_ms"], h["hi_ms"]) for h in hist] == [
        (0.0, 50.0), (50.0, 100.0), (100.0, 250.0)]
    assert [h["grid_slots_ms"] for h in hist] == [
        [0], [50], [100, 150, 200, 250]]
    # No directional asymmetry in pure coincidence.
    asym = b["dt_summary"]["asymmetry"]
    assert abs(asym["frac_audio_after_visual"]
               - asym["frac_audio_before_visual"]) < 0.15


def test_clustered_stream_shows_tight_dt_excess():
    rng = np.random.default_rng(7)
    duration = 1200.0
    onset_ts = _poisson_times(rng, 0.5, duration)
    onsets = [(t, 1.0) for t in onset_ts]
    # 80% of onsets get a transient planted 10-30 ms BEFORE them
    # (the visible event lands a beat before its sound: dt =
    # audio_t - visual_t = +10..30 ms, audio after visual).
    planted = []
    for t in onset_ts:
        if rng.uniform() < 0.8:
            vt = t - rng.uniform(0.010, 0.030)
            if vt >= 0:
                planted.append((vt, 1.0))
    background = [(t, 1.0) for t in _poisson_times(rng, 0.15, duration)]
    vtrans = sorted(planted + background)
    b = bind(onsets, vtrans, duration_s=duration)
    tight50 = b["tight_windows"]["50"]["fraction_of_bound"]
    chance_tight = b["tight_windows"]["50"]["chance_rate_poisson"]
    # Clear excess: most bound pairs are the planted near-zero-dt
    # ones, far above both the uniform-dt expectation (0.20 of
    # bound) and the +-50 ms chance rate.
    assert tight50 > 0.55
    assert b["bound_rate"] > chance_tight + 0.30
    # The grid-aligned first bin [0, 50) holds the bulk of pairs
    # (planted |dt| is 10-30 ms, inside the first bin).
    hist = b["dt_summary"]["histogram_abs_dt_ms"]
    assert hist[0]["fraction"] > 0.55
    # Planted lags put the audio after the visual.
    assert b["dt_summary"]["asymmetry"]["frac_audio_after_visual"] > 0.6


def test_binding_summary_from_stored_pairs():
    # The backfill path: summary computed from persisted bound_pairs
    # alone (dt_s as stored, rounded to ms) matches a live bind.
    pairs = [{"audio_t_s": 1.0, "visual_t_s": 0.99, "dt_s": 0.01,
              "audio_strength": 1.0, "visual_strength": 1.0},
             {"audio_t_s": 2.0, "visual_t_s": 2.2, "dt_s": -0.2,
              "audio_strength": 1.0, "visual_strength": 1.0},
             {"audio_t_s": 3.0, "visual_t_s": 3.0, "dt_s": 0.0,
              "audio_strength": 1.0, "visual_strength": 1.0}]
    s = binding_summary(pairs, n_audio=10, n_visual=40,
                        duration_s=100.0, duration_source="test")
    assert s["bound_rate"] == 0.3
    assert s["tight_windows"]["50"]["n_bound"] == 2
    assert s["tight_windows"]["100"]["n_bound"] == 2
    assert s["dt_summary"]["asymmetry"]["simultaneous"] == 1
    assert s["dt_summary"]["asymmetry"]["audio_after_visual"] == 1
    assert s["dt_summary"]["asymmetry"]["audio_before_visual"] == 1


# ---------------------------------------------------------------------
# 2026-10-14: grid-corrected chance, shift null, ms quantization
# ---------------------------------------------------------------------

def test_grid_corrected_chance_prices_the_grid():
    # FF3 densities at W=50 ms: continuous Poisson says 0.0962;
    # pricing the inclusive grid window (2W + 50 ms = 150 ms of
    # exposure) gives 0.1409 -- the binder investigation's figure.
    cont = poisson_chance_rate(2735, 2701.0, 50.0)
    grid = grid_corrected_chance_rate(2735, 2701.0, 50.0)
    assert abs(cont - 0.0962) < 0.001
    assert abs(grid - 0.1409) < 0.001
    assert grid > cont
    # At the wide window the correction is proportionally small.
    assert grid_corrected_chance_rate(2735, 2701.0, 250.0) > \
        poisson_chance_rate(2735, 2701.0, 250.0)
    assert grid_corrected_chance_rate(0, 100.0, 50.0) is None
    # The tight-window fields ride in binding summaries.
    s = binding_summary([], n_audio=10, n_visual=2735,
                        duration_s=2701.0)
    tw = s["tight_windows"]["50"]
    assert abs(tw["chance_rate_grid_corrected"] - 0.1409) < 0.001
    assert "non-exclusive" in s["chance_rate_poisson_label"]


def test_shift_null_present_and_sane_on_independent_streams():
    rng = np.random.default_rng(20261012)
    duration = 1200.0
    onsets = [(t, 1.0) for t in _poisson_times(rng, 0.5, duration)]
    vtrans = [(t, 1.0) for t in _poisson_times(rng, 1.0, duration)]
    b = bind(onsets, vtrans, duration_s=duration)
    sn = b["shift_null"]
    assert sn["available"] is True
    assert sn["n_shifts"] == 200
    blk = sn["bound_rate"]
    # Independent streams: the observed rate is an ordinary draw
    # from its own shift null -- mean close, percentile central.
    assert abs(blk["observed"] - blk["null_mean"]) < 0.05
    assert 1.0 <= blk["observed_percentile"] <= 99.0
    assert sn["tight_rate_50ms"]["observed"] == \
        b["tight_windows"]["50"]["bound_rate"]


def test_shift_null_absent_states_why_from_pairs_only():
    s = binding_summary([], n_audio=1, n_visual=1, duration_s=10.0)
    assert s["shift_null"]["available"] is False
    assert "pairs" in s["shift_null"]["reason"]


def test_bind_ms_quantization_kills_representation_dust():
    # Same events, two float representations: ms-rounded times
    # (as stored in av_binding.json) and the runtime's slot x 0.05
    # form (sub-microsecond dust). Before Fix 3 these bound
    # differently (FF3: 71 pairs differed); now identical.
    rng = np.random.default_rng(99)
    duration = 300.0
    # On-grid event times (multiples of 0.05), as the pipeline
    # produces; the two representations below encode the SAME
    # events, one ms-rounded, one in slot x 0.05 float form.
    slots_a = sorted(rng.integers(0, int(duration / 0.05),
                                  size=150).tolist())
    slots_v = sorted(rng.integers(0, int(duration / 0.05),
                                  size=150).tolist())
    rounded_a = [(round(s * 0.05, 3), 1.0) for s in slots_a]
    rounded_v = [(round(s * 0.05, 3), 1.0) for s in slots_v]
    dust_a = [(s * 0.05, 1.0) for s in slots_a]
    dust_v = [(s * 0.05, 1.0) for s in slots_v]
    r1 = bind(rounded_a, rounded_v, duration_s=duration,
              shift_null=False)
    r2 = bind(dust_a, dust_v, duration_s=duration, shift_null=False)
    assert r1["n_bound"] == r2["n_bound"]
    assert r1["bound_pairs"] == r2["bound_pairs"]
    # Off-grid times keep their ms identity (quantization is to
    # milliseconds, not to the 50 ms grid).
    r3 = bind([(1.012, 1.0)], [(1.030, 1.0)], duration_s=10.0,
              shift_null=False)
    assert r3["bound_pairs"][0]["dt_s"] == -0.018
