"""Tests for scripts/stem_streams.py (Phase 5, audit item 14).

Only the deterministic, model-free machinery is under test here:
the planted tone sequence, the mix level arithmetic, and the
online features/onset stream over them. Demucs separation, CLAP
labeling, and transcription are exercised by the validation runs
(PREDICTIONS.md Phase 5), not by unit tests.
"""

import importlib.util
import os

import numpy as np

from hva.online import SR

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts",
                     "stem_streams.py")
_spec = importlib.util.spec_from_file_location("stem_streams", _PATH)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)


def test_tone_sequence_plants_energy_at_known_times():
    music = ss.tone_sequence(20.0)
    def rms_at(t):
        i = int(t * SR)
        return float(np.sqrt((music[i:i + SR // 10] ** 2).mean()))
    # Planted: bursts at 2/7/13, pad 9-12, thump at 16.
    assert rms_at(2.0) > 0.05
    assert rms_at(9.5) > 0.05
    assert rms_at(16.0) > 0.05
    # Quiet between events.
    assert rms_at(4.5) < 1e-4
    assert rms_at(18.5) < 1e-4


def test_build_mix_music_level():
    rng = np.random.default_rng(0)
    speech = (0.2 * rng.standard_normal(20 * SR)).astype(np.float32)
    mix, music = ss.build_mix(speech, 20.0, music_db=-6.0)
    s_rms = float(np.sqrt((speech ** 2).mean()))
    m_rms = float(np.sqrt((music ** 2).mean()))
    assert abs(20 * np.log10(m_rms / s_rms) - (-6.0)) < 0.5
    assert len(mix) == 20 * SR
    assert np.abs(mix).max() <= 1.0


def test_stream_features_onsets_find_planted_bursts():
    music = ss.tone_sequence(20.0)
    onsets, _rms = ss.stream_features(music)
    times = [t for t, _s in onsets]
    for planted in (2.0, 7.0, 13.0, 16.0):
        assert any(abs(t - planted) < 0.3 for t in times), planted
    rec = ss.analyze_mix(music)
    assert rec["n_onsets"] == len(times)
    assert rec["n_onsets"] >= 4
