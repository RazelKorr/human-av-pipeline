"""Tests for hva.mishear (Phase 5, audit item 15).

The mishearing model proposes phonetic alternatives for
low-confidence words. Invariants under test:
- verbatim is byte-identical to the input text, input unmutated;
- high-confidence segments (and segments with no logprob) yield
  NO modeled mishearings;
- a low-prob word with a known phonetic neighbor yields that
  neighbor as a modeled alternative, labeled as model output.
"""

import copy
import os

import pytest

from hva.mishear import (build_doubt_track, candidate_readings,
                         model_segment, model_segments, phonetic_key)

VOCAB = {"their": 100, "there": 90, "they": 95, "night": 200,
         "knight": 210, "marvel": 300, "marble": 400, "calm": 500}


def _seg(text, lp, words):
    return {"start": 0.0, "end": 1.0, "text": text,
            "avg_logprob": lp, "words": words}


def test_phonetic_key_known_pairs():
    assert phonetic_key("their") == phonetic_key("there")
    assert phonetic_key("night") == phonetic_key("knight")
    assert phonetic_key("marvel") == phonetic_key("marble")
    assert phonetic_key("") == ""
    assert phonetic_key("!!!") == ""


def test_low_prob_word_yields_phonetic_neighbor():
    seg = _seg("their house", -1.0, [
        {"start": 0.0, "end": 0.4, "word": " their", "prob": 0.2},
        {"start": 0.4, "end": 0.9, "word": " house", "prob": 0.95},
    ])
    out = model_segment(seg, vocab=VOCAB)
    assert out["modeled"] is True
    assert len(out["modeled_mishearings"]) == 1
    entry = out["modeled_mishearings"][0]
    assert entry["word"] == "their"
    readings = {a["reading"]: a for a in entry["alternatives"]}
    assert "there" in readings
    assert readings["there"]["key_match"] == "exact"
    assert "there" in entry["modeled_reading"]
    assert out["model_output"] is True


def test_high_confidence_segment_yields_nothing():
    seg = _seg("their house", -0.2, [
        {"start": 0.0, "end": 0.4, "word": " their", "prob": 0.2},
        {"start": 0.4, "end": 0.9, "word": " house", "prob": 0.95},
    ])
    out = model_segment(seg, vocab=VOCAB)
    assert out["modeled"] is False
    assert out["modeled_mishearings"] == []


def test_segment_without_logprob_is_never_modeled():
    seg = _seg("their house", None, [
        {"start": 0.0, "end": 0.4, "word": " their", "prob": 0.1},
    ])
    seg["avg_logprob"] = None
    out = model_segment(seg, vocab=VOCAB)
    assert out["modeled"] is False


def test_confident_words_in_doubtful_segment_yield_nothing():
    seg = _seg("calm night", -1.2, [
        {"start": 0.0, "end": 0.4, "word": " calm", "prob": 0.9},
        {"start": 0.4, "end": 0.9, "word": " night", "prob": 0.85},
    ])
    out = model_segment(seg, vocab=VOCAB)
    assert out["modeled_mishearings"] == []


def test_verbatim_byte_identical_and_input_unmutated():
    text = "  We're in Marvel!  ".strip()
    seg = _seg(text, -0.877, [
        {"start": 0.0, "end": 0.3, "word": " We're", "prob": 0.4},
        {"start": 0.3, "end": 0.5, "word": " in", "prob": 0.9},
        {"start": 0.5, "end": 1.0, "word": " Marvel!", "prob": 0.3},
    ])
    before = copy.deepcopy(seg)
    out = model_segment(seg, vocab=VOCAB)
    assert out["verbatim"] == text
    assert out["verbatim"].encode() == text.encode()
    assert seg == before


def test_model_segments_preserves_order_and_count():
    segs = [_seg("calm", -0.1, []), _seg("night", -1.0, [
        {"start": 0.0, "end": 0.5, "word": " night", "prob": 0.2}])]
    out = model_segments(segs, vocab=VOCAB)
    assert len(out) == 2
    assert out[0]["modeled"] is False
    readings = {a["reading"] for a in
                out[1]["modeled_mishearings"][0]["alternatives"]}
    assert "knight" in readings


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(
        os.path.dirname(__file__), "..", "models",
        "faster-whisper-base", "tokenizer.json")),
    reason="local whisper tokenizer vocabulary not present")
def test_real_vocab_marvel_has_marble():
    alts = candidate_readings("Marvel", word_prob=0.3)
    exact = {a["reading"] for a in alts if a["key_match"] == "exact"}
    assert "marble" in exact
    assert "marvel" not in {a["reading"] for a in alts}


def test_doubt_track_is_parallel_and_labeled():
    # W5 (2026-10-14): the doubt track travels as one labeled
    # object, entries parallel to the segments by index,
    # verbatim byte-identical, input unmutated.
    segs = [
        _seg("their house", -1.0, [
            {"start": 0.0, "end": 0.4, "word": " their",
             "prob": 0.2},
            {"start": 0.4, "end": 0.9, "word": " house",
             "prob": 0.95}]),
        _seg("clear words", -0.1, [
            {"start": 0.0, "end": 0.5, "word": " clear",
             "prob": 0.99}]),
    ]
    before = copy.deepcopy(segs)
    track = build_doubt_track(segs, vocab=VOCAB)
    assert segs == before  # input unmutated
    assert track["track"] == "doubt"
    assert track["model_output"] is True
    assert "MODEL OUTPUT" in track["disclaimer"]
    assert track["n_segments"] == 2
    assert len(track["entries"]) == len(segs)
    assert track["entries"][0]["verbatim"] == segs[0]["text"]
    assert track["entries"][1]["verbatim"] == segs[1]["text"]
    assert track["n_modeled"] == 1  # only the low-conf segment
    assert track["entries"][1]["modeled"] is False
