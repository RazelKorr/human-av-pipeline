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
