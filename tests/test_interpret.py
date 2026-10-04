"""Tests for the audio "what" pathway (hva.interpret): sound-event
labeling, two-tier transcription, music features.

Model-dependent tests (CLAP weights, whisper models) skip cleanly
when the weights aren't present -- the suite must stay green on a
fresh checkout. Logic tests use fakes and synthetic audio.
"""

import os

import numpy as np
import pytest

from hva import interpret
from hva.interpret import (
    LABEL_VOCABULARY,
    LOW_CONF_THRESHOLD,
    _sanitize,
    interpret_window,
    is_music,
    label_window,
    music_features,
    transcribe_two_tier,
)

SR = 16000
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "models")
CLAP_CKPT = os.path.join(MODELS, "laion-clap-630k-audioset-best.pt")
WHISPER_BASE = os.path.join(MODELS, "faster-whisper-base")
WHISPER_MEDIUM = os.path.join(MODELS, "faster-whisper-medium")

needs_clap = pytest.mark.skipif(
    not os.path.isfile(CLAP_CKPT),
    reason="CLAP weights not downloaded (models/laion-clap-*.pt)")
needs_whisper = pytest.mark.skipif(
    not (os.path.isdir(WHISPER_BASE) and os.path.isdir(WHISPER_MEDIUM)),
    reason="whisper base/medium models not present")


def _noise_burst(seconds=2.0, amp=0.8):
    rng = np.random.default_rng(0)
    x = np.zeros(int(seconds * SR), dtype=np.float32)
    i = int(0.5 * SR)
    x[i:i + int(0.2 * SR)] = rng.standard_normal(int(0.2 * SR)
                                                 ).astype(np.float32) * amp
    return np.clip(x, -1, 1)


def _chord(seconds=3.0):
    t = np.arange(int(seconds * SR)) / SR
    x = sum(np.sin(2 * np.pi * f * t)
            for f in (261.63, 329.63, 392.0)) / 3.0
    return (0.5 * x).astype(np.float32)


# --- sanitize / edge cases (no models needed) ---------------------------

def test_sanitize_kills_nonfinite():
    x = np.array([0.5, np.nan, np.inf, -np.inf, -0.5], dtype=np.float32)
    y = _sanitize(x)
    assert np.all(np.isfinite(y))
    assert y[0] == 0.5 and y[4] == -0.5
    assert y[1] == 0.0 and y[2] == 0.0 and y[3] == 0.0


def test_sanitize_clips():
    y = _sanitize(np.array([2.0, -3.0], dtype=np.float32))
    assert float(y.max()) <= 1.0 and float(y.min()) >= -1.0


def test_label_window_empty_no_model_needed():
    assert label_window(np.zeros(0, dtype=np.float32)) == []


def test_transcribe_two_tier_empty():
    r = transcribe_two_tier(np.zeros(0, dtype=np.float32))
    assert r["segments"] == [] and r["n_escalated"] == 0


def test_music_features_silent_and_short():
    assert music_features(np.zeros(SR * 3, dtype=np.float32)) is None
    assert music_features(np.zeros(SR // 2, dtype=np.float32)) is None


def test_is_music_threshold():
    assert is_music([{"label": "music", "score": 0.9}])
    assert is_music([{"label": "singing", "score": 0.5}])
    assert not is_music([{"label": "music", "score": 0.1}])
    assert not is_music([{"label": "speech", "score": 0.9}])
    assert not is_music([])


def test_interpret_window_empty_schema_no_model_needed():
    rec = interpret_window(np.zeros(0, dtype=np.float32), t_s=1.5,
                           strength=100.0, transcribe=False)
    assert rec["t_s"] == 1.5
    assert rec["labels"] == []
    assert rec["transcript"] is None
    assert rec["music"] is None


# --- two-tier escalation logic (fakes: deterministic, no models) ---------

class _FakeWord:
    def __init__(self, start, end, word, prob):
        self.start, self.end, self.word = start, end, word
        self.probability = prob


class _FakeSeg:
    def __init__(self, start, end, text, avg_logprob):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob = avg_logprob
        self.words = [_FakeWord(start, end, text, 0.9)]


class _FakeModel:
    def __init__(self, segs):
        self._segs = segs
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        return self._segs, None


def _patch_loader(monkeypatch, base_segs, medium_segs):
    import hva.transcribe as ht
    base_m, med_m = _FakeModel(base_segs), _FakeModel(medium_segs)
    def fake_loader(name):
        return {"base": base_m, "medium": med_m}[name]
    monkeypatch.setattr(ht, "_load_model", fake_loader)
    return base_m, med_m


def test_two_tier_escalates_low_confidence(monkeypatch):
    low = _FakeSeg(0.0, 1.0, "fuzzy words", -1.5)
    fixed = _FakeSeg(0.0, 1.0, "clear words", -0.2)
    base_m, med_m = _patch_loader(monkeypatch, [low], [fixed])
    audio = np.zeros(SR * 2, dtype=np.float32)
    r = transcribe_two_tier(audio, low_conf_threshold=-0.8)
    assert r["n_escalated"] == 1
    assert med_m.calls == 1
    assert r["segments"][0]["tier"] == "medium"
    assert r["segments"][0]["text"] == "clear words"
    assert r["segments"][0]["avg_logprob"] == -0.2


def test_two_tier_keeps_confident_on_base(monkeypatch):
    good = _FakeSeg(0.0, 1.0, "clean speech", -0.3)
    base_m, med_m = _patch_loader(monkeypatch, [good], [])
    audio = np.zeros(SR * 2, dtype=np.float32)
    r = transcribe_two_tier(audio, low_conf_threshold=-0.8)
    assert r["n_escalated"] == 0
    assert med_m.calls == 0
    assert r["segments"][0]["tier"] == "base"


def test_two_tier_medium_silence_falls_back_to_base(monkeypatch):
    low = _FakeSeg(0.0, 1.0, "mumble", -2.0)
    base_m, med_m = _patch_loader(monkeypatch, [low], [])
    audio = np.zeros(SR * 2, dtype=np.float32)
    r = transcribe_two_tier(audio, low_conf_threshold=-0.8)
    # medium returned nothing: base segment kept, honestly labeled
    assert r["n_escalated"] == 0
    assert r["segments"][0]["tier"] == "base"
    assert r["segments"][0]["text"] == "mumble"


def test_two_tier_threshold_is_sealed_value():
    assert LOW_CONF_THRESHOLD == -0.8


# --- model-backed tests --------------------------------------------------

@needs_clap
def test_label_window_schema_and_scores():
    labels = label_window(_noise_burst(), top_k=5)
    assert len(labels) == 5
    assert all(set(l) == {"label", "score"} for l in labels)
    assert all(l["label"] in LABEL_VOCABULARY for l in labels)
    scores = [l["score"] for l in labels]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # softmax over the vocabulary: full-set mass ~1
    full = label_window(_noise_burst(), top_k=len(LABEL_VOCABULARY))
    assert abs(sum(l["score"] for l in full) - 1.0) < 1e-3


@needs_clap
def test_label_window_deterministic():
    a = _noise_burst()
    assert label_window(a) == label_window(a)


@needs_clap
def test_label_window_silence_says_silence():
    labels = label_window(np.zeros(SR * 2, dtype=np.float32), top_k=3)
    assert labels[0]["label"] == "silence"


@needs_whisper
def test_two_tier_real_speech_escalation_extremes():
    import wave
    # 8 s of real speech: Dr Tran ep4 onset window (Grandma, 59.1 s)
    wav_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "output",
        "drtran_ep4_av", "onset_59.1s.wav")
    if not os.path.isfile(wav_path):
        pytest.skip("drtran onset fixture not present")
    with wave.open(wav_path, "rb") as w:
        raw = w.readframes(w.getnframes())
    audio = (np.frombuffer(raw, dtype=np.int16).astype(np.float32)
             / 32768.0)
    r_all = transcribe_two_tier(audio, low_conf_threshold=0.0)
    r_none = transcribe_two_tier(audio, low_conf_threshold=-100.0)
    assert len(r_all["segments"]) > 0  # real speech detected
    assert r_all["n_escalated"] >= 1  # threshold 0 escalates everything
    assert r_none["n_escalated"] == 0
    assert all(s["tier"] == "base" for s in r_none["segments"])
    for s in r_all["segments"] + r_none["segments"]:
        assert set(s) >= {"start", "end", "text", "avg_logprob",
                          "tier", "words"}


def test_music_features_chord_shape():
    mf = music_features(_chord())
    assert mf is not None
    assert len(mf["chroma_mean"]) == 12
    assert len(mf["chroma_std"]) == 12
    assert mf["n_chroma_frames"] > 0
    # C major chord: pitch classes C, E, G (0, 4, 7) should dominate
    top3 = sorted(range(12), key=lambda i: -mf["chroma_mean"][i])[:3]
    assert set(top3) == {0, 4, 7}
