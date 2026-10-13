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
"""

import numpy as np

from streaming.bind_av import bind, binding_summary, poisson_chance_rate


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
    # dt histogram partitions the bound pairs.
    hist = b["dt_summary"]["histogram_abs_dt_ms"]
    assert sum(h["n"] for h in hist) == b["n_bound"]
    assert [(h["lo_ms"], h["hi_ms"]) for h in hist] == [
        (0.0, 25.0), (25.0, 50.0), (50.0, 100.0), (100.0, 250.0)]
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
    # Bins 0-25 and 25-50 together hold the bulk of pairs.
    hist = b["dt_summary"]["histogram_abs_dt_ms"]
    assert hist[0]["fraction"] + hist[1]["fraction"] > 0.55
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
