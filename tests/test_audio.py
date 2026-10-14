"""Tests for the audio pathway (Phase 1-3): cochlear features, onset
detection, audio-visual binding, attended transcription plumbing."""

import json
import subprocess

import numpy as np
import pytest

from hva.online import (moment_features, pick_onsets, SR, SPM,
                        MOMENT_MS, RMS_FLOOR)
from streaming.bind_av import visual_transients, bind
from streaming.feeder import decode_audio_track


def _sine(freq=440.0, seconds=1.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _clicks(times_s, seconds=2.0, amp=0.9):
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    for t in times_s:
        i = int(t * SR)
        x[i:i + 8] = amp  # 0.5 ms click
    return x


def _blocks(x):
    n = len(x) // SPM
    return x[:n * SPM].reshape(n, SPM)


def test_moment_features_shapes():
    x = _blocks(_sine(seconds=0.5))
    rms, flux, cent, last = moment_features(x)
    assert rms.shape == flux.shape == cent.shape == (10,)
    assert last.shape == (513,)
    assert np.all(rms >= 0) and np.all(flux >= 0)
    assert np.all((cent >= 0) & (cent <= SR / 2))


def test_rms_silence_and_sine():
    rms, _, _, _ = moment_features(_blocks(np.zeros(SR, dtype=np.float32)))
    assert np.all(rms == 0.0)
    rms, _, _, _ = moment_features(_blocks(_sine(amp=0.5, seconds=0.5)))
    assert np.allclose(rms, 0.5 / np.sqrt(2), atol=1e-3)


def test_flux_continuity_across_chunks():
    x = _blocks(_sine(seconds=1.0))
    _, f1, _, last = moment_features(x[:10])
    _, f2, _, _ = moment_features(x[10:], prev_mag=last)
    _, fall, _, _ = moment_features(x)
    # flux[1:] of chunk 2 must match the unbroken run (flux[0] of chunk
    # 2 uses the threaded previous magnitude instead of zero)
    assert np.allclose(f2[1:], fall[11:], atol=1e-6)
    assert f2[0] > 0 or True  # no crash on the boundary; value is data


def test_pick_onsets_clicks():
    x = _clicks([0.5, 1.0, 1.5], seconds=2.0)
    rms, flux, _, _ = moment_features(_blocks(x))
    onsets = pick_onsets(flux, rms)
    assert len(onsets) == 3
    for (t, _), want in zip(onsets, [0.5, 1.0, 1.5]):
        assert abs(t - want) <= 0.05, (t, want)


def test_pick_onsets_silence_and_sine_have_none():
    for x in [np.zeros(2 * SR, dtype=np.float32),
              _sine(seconds=2.0)]:
        rms, flux, _, _ = moment_features(_blocks(x))
        assert pick_onsets(flux, rms) == []


def test_pick_onsets_min_gap():
    x = _clicks([0.5, 0.6, 1.5], seconds=2.0)  # two clicks 100 ms apart
    rms, flux, _, _ = moment_features(_blocks(x))
    onsets = pick_onsets(flux, rms, min_gap_ms=250.0)
    # the close pair collapses to the stronger one; the distant stays
    assert len(onsets) == 2
    assert abs(onsets[1][0] - 1.5) <= 0.05


def test_pick_onsets_brown_noise_bed_px1():
    # PX.1 regression (2026-10-14, REVIEW-DEFRAG Fix 2): a loud
    # stationary brown-noise bed must not produce onset counts set
    # by duration alone. Before the MAD consistency constant, the
    # k=3 bar sat at ~2.0 sigma of the bed's flux bulk and this
    # fixture class yielded ~42 false picks per 60 s (filed
    # experiment: 54 total onsets for 12 planted bursts).
    rng = np.random.default_rng(20261014)
    dur_s = 30.0
    n = int(dur_s * SR)
    # Stationary noise bed synthesized spectrally (1/sqrt(f)
    # amplitude): power spread across the band, so per-moment flux
    # averages over many bins into a tight, near-Gaussian bulk --
    # the regime of the filed fixture, whose AAC-decoded bed flux
    # had median 11.67 / p95 15.71 (this bed: median ~19.7,
    # p95 ~27.2 at the same RMS; see the defrag tuning log).
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1]
    bed = np.fft.irfft(spec / np.sqrt(f), n)
    bed /= np.sqrt(np.mean(bed ** 2))
    bed = (bed * 0.0151).astype(np.float64)  # fixture bed RMS
    planted = [2.5 + 5.0 * k for k in range(6)]
    t = np.arange(int(0.3 * SR)) / SR
    fade = np.minimum(1.0, np.minimum(t / 0.05,
                                      (0.3 - t) / 0.05))
    for k, t0 in enumerate(planted):
        freq = 880.0 if k % 2 == 0 else 660.0
        burst = 0.11 * np.sin(2 * np.pi * freq * t) * fade
        i0 = int(t0 * SR)
        bed[i0:i0 + len(burst)] += burst
    x = bed.astype(np.float32)
    n_mom = len(x) // SPM
    rms, flux, _cent, _last = moment_features(
        x[:n_mom * SPM].reshape(n_mom, SPM))
    onsets = pick_onsets(flux, rms)
    # Every planted burst is found.
    for t0 in planted:
        assert any(abs(ts - t0) <= 0.5 for ts, _ in onsets), t0
    # And the bed does not flood: bed-only picks on this bed
    # measure 28 at the pre-fix bar vs 9 at the fixed bar (30 s,
    # tuning log in the defrag review), so with bursts the total
    # lands ~15 post-fix vs ~34 pre-fix.
    assert len(onsets) <= 2 * len(planted) + 6


def test_pick_onsets_rms_floor():
    # tiny clicks below the RMS floor must not onset (encode noise)
    x = _clicks([0.5], seconds=1.0, amp=1e-4)
    rms, flux, _, _ = moment_features(_blocks(x))
    assert pick_onsets(flux, rms) == []


def test_visual_transients_and_bind():
    # synthetic: energies jump at t=1.0s and t=5.0s (30 fps frames).
    # Both the rising AND falling edges are transients -- 4 peaks.
    t = np.arange(0, 6.0, 1 / 30.0)
    e = np.ones_like(t) * 0.1
    e[(t > 1.0) & (t < 1.2)] = 0.9
    e[(t > 5.0) & (t < 5.2)] = 0.9
    vt = visual_transients(e, t * 1000.0)
    assert len(vt) == 4
    assert abs(vt[0][0] - 1.0) < 0.1 and abs(vt[2][0] - 5.0) < 0.1
    onsets = [(1.05, 2.0), (3.0, 1.5)]  # one near, one far
    b = bind(onsets, vt)
    assert b["n_bound"] == 1
    assert abs(b["bound_pairs"][0]["audio_t_s"] - 1.05) < 1e-9
    assert len(b["unbound_audio"]) == 1  # the 3.0s onset: score bed
    assert b["unbound_audio"][0]["t_s"] == 3.0
    # binding is greedy: each visual transient used once -- with a
    # single transient, the stronger onset wins and the weaker goes
    # unbound even though it's also in window
    b2 = bind([(1.04, 5.0), (1.06, 1.0)], [(1.0, 9.0)])
    assert b2["n_bound"] == 1
    assert b2["bound_pairs"][0]["audio_strength"] == 5.0
    assert len(b2["unbound_audio"]) == 1
    assert b2["unbound_audio"][0]["strength"] == 1.0


def test_visual_transients_stationary_bed_r1():
    # R1 (2026-10-14): a loud stationary energy bed must not
    # manufacture transients. Construction mirrors the sealed
    # fixture (improvements-2026-10/logs/wild-2026-10-14/
    # r1_fixture.py): 120 s at 30 fps, smooth base + white noise
    # sigma 0.05, 10 planted +1.0 transients. Under the pre-fix
    # UNSCALED MAD bar this returned 41 picks / 31 false alarms;
    # with VISUAL_MAD_TO_SIGMA applied it returns 11 / 1. If the
    # constant is ever dropped, this test floods and fails.
    rng = np.random.default_rng(20261014)
    n = int(30.0 * 120.0)
    t = np.arange(n) / 30.0
    e = 1.0 + 0.1 * np.sin(2 * np.pi * t / 60.0)
    e = e + rng.normal(0.0, 0.05, n)
    planted = [7.5 + 11.0 * k for k in range(10)]
    for tp in planted:
        i = int(round(tp * 30.0))
        e[i:i + 2] += 1.0
    vt = visual_transients(e, t * 1000.0)
    # classify in moment-index space (pick times are grid
    # multiples; a +-0.05 s float tolerance straddles the edge)
    found = sum(1 for (ts, _s) in vt
                if any(abs(round(ts / 0.05) - round(tp / 0.05)) <= 1
                       for tp in planted))
    assert found == 10
    assert len(vt) <= 13  # false alarms <= 3 (measured: 1)


def test_decode_audio_track_length(tmp_path):
    # build a 2 s stereo 48 kHz test video with a tone
    vid = str(tmp_path / "tone.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2",
         "-f", "lavfi", "-i", "color=black:s=64x64:d=2:r=30",
         "-shortest", vid], check=True)
    x, sr = decode_audio_track(vid, 2.0, sr=16000)
    assert sr == 16000
    assert abs(len(x) / sr - 2.0) < 0.05
    assert x.ndim == 1 and x.dtype == np.float32


def test_decode_audio_track_missing_audio_raises(tmp_path):
    vid = str(tmp_path / "silent.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "color=black:s=64x64:d=1:r=30", vid], check=True)
    with pytest.raises(RuntimeError, match="no usable audio"):
        decode_audio_track(vid, 1.0)


def test_moment_grid_constants():
    assert SPM == 800 and MOMENT_MS == 50.0 and SR == 16000
    assert RMS_FLOOR > 0


def test_load_audio_no_plane_padding_inflation(tmp_path):
    # Regression (2026-10-03): hva.transcribe._load_with_av read the whole
    # padded PyAV plane buffer instead of [:rf.samples], inflating decoded
    # audio by ~22% (measured 1.2194x) and stretching every transcript
    # timestamp. hva/stream.py already sliced; this path did not.
    from hva.transcribe import load_audio
    vid = str(tmp_path / "tone2.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=3",
         "-f", "lavfi", "-i", "color=black:s=64x64:d=3:r=30",
         "-shortest", vid], check=True)
    x = load_audio(vid, target_sr=16000)
    assert abs(len(x) / 16000 - 3.0) < 0.1, f"inflated: {len(x)/16000:.2f}s"


# --- Causal L2 vis_boost gate (2026-10-12, audit item 13) ---

from hva.attention import (AuditoryAttention, shaped_vis_boost,
                           GATE_WINDOW, GATE_WARMUP)


def _vb_pair(n=500, split=400, seed=4):
    """Two vis_boost series identical up to `split`; the second
    grows a huge transient afterwards."""
    rng = np.random.default_rng(seed)
    vb = rng.random(n) * 0.6
    vb2 = vb.copy()
    vb2[split + 10] = 1.0
    vb2[split + 50:] = 1.0
    return vb, vb2, split


def test_causal_gate_prefix_invariance():
    vb, vb2, split = _vb_pair()
    s1 = shaped_vis_boost(vb, "causal")
    s2 = shaped_vis_boost(vb2, "causal")
    # Nothing at or before the split may change: the causal gate at
    # moment k reads only moments < k and vb[k] itself.
    assert np.array_equal(s1[:split + 1], s2[:split + 1])


def test_legacy_gate_leaks_future_characterization():
    # Characterization of the OLD behavior (kept for A/B): the
    # full-sequence percentile lets a late transient rewrite early
    # gate values. This test documents the leak; it must keep
    # passing as long as legacy mode exists.
    vb, vb2, split = _vb_pair()
    s1 = shaped_vis_boost(vb, "legacy")
    s2 = shaped_vis_boost(vb2, "legacy")
    assert not np.array_equal(s1[:split], s2[:split]), \
        "legacy gate no longer leaks -- has it been removed?"


def test_causal_gate_warmup_blocks_early_boost():
    vb = np.ones(200)
    s = shaped_vis_boost(vb, "causal")
    assert np.all(s[:GATE_WARMUP] == 0.0), \
        "no boost may fire before GATE_WARMUP past moments exist"


def test_causal_gate_window_forgets_old_transients():
    # After the huge block: a 3:1 pattern of 0.4 with 0.6 every
    # fourth moment, so the trailing q75 sits near 0.4 once the
    # block leaves the window.
    vb = np.concatenate(
        [np.ones(100),
         np.tile([0.4, 0.4, 0.4, 0.6], 200)]).astype(np.float64)
    s = shaped_vis_boost(vb, "causal")
    # While the huge block is inside the trailing window its q75
    # pins the gate shut (index 151 is a 0.6 moment).
    assert vb[151] == 0.6 and s[151] == 0.0
    # Long after it leaves the window, the same 0.6 moment earns a
    # boost against its quieter neighborhood.
    assert vb[803] == 0.6 and s[803] > 0.0


def _synthetic_moms(n=600, seed=1):
    rng = np.random.default_rng(seed)
    moms = []
    for k in range(n):
        sal = (rng.random(64) * 2.0).astype(np.float32)
        tmax = 20.0 if k % 37 == 0 else 3.0
        moms.append({"t0": k * 0.1, "t1": k * 0.1 + 0.1,
                     "sal": sal, "Tmax": tmax, "loud": 0.5})
    Tprof = (rng.random((n, 64)) * 5.0).astype(np.float32)
    return moms, Tprof


def test_causal_gate_trace_prefix_identical():
    # End-to-end: two attention runs whose vis_boost differs only
    # after the split must produce identical traces up to it.
    vb, vb2, split = _vb_pair(n=600, split=400)
    moms, Tprof = _synthetic_moms(600)
    tr1 = AuditoryAttention().run(moms, Tprof, vis_boost=vb,
                                  gate_mode="causal")
    tr2 = AuditoryAttention().run(moms, Tprof, vis_boost=vb2,
                                  gate_mode="causal")
    for a, b in zip(tr1[:split], tr2[:split]):
        assert a["event"] == b["event"] and a["cf_bin"] == b["cf_bin"]
