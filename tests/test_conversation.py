"""Tests for the conversational turn-taking (hva/conversation.py)."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hva.conversation import (  # noqa: E402
    TurnDetector, ResponsePolicy, Turn, is_addressed,
)


def _seg(words):
    """Build a transcript segment from (word, start, end) triples."""
    ws = [{"word": w, "start": s, "end": e} for w, s, e in words]
    return {"start": ws[0]["start"], "end": ws[-1]["end"],
            "text": " ".join(w for w, _, _ in words), "words": ws}


def test_is_addressed_matches_whisper_variants():
    assert is_addressed("hi wodehaus")
    assert is_addressed("hello woodhouse")
    assert is_addressed("I would house, can you hear me?")  # observed 2026-09-30
    assert is_addressed("hey what house")
    assert not is_addressed("hi there, can you hear me?")
    assert not is_addressed("the house is blue")


def test_turn_detector_fires_on_name_plus_silence():
    det = TurnDetector(silence_s=1.5)
    segs = [_seg([("I", 0.0, 0.2), ("would", 0.24, 0.4),
                  ("house", 0.4, 0.7), ("hi", 0.9, 1.1)])]
    # Still talking: last word ended at 1.1, now is 1.5 (< 1.5s silence).
    assert det.update(segs, 1.5) == []
    # Silence elapsed: turn completes.
    turns = det.update(segs, 3.0)
    assert len(turns) == 1
    assert "house" in turns[0].text
    # Detector resets: no double-fire.
    assert det.update(segs, 5.0) == []


def test_turn_detector_ignores_unaddressed_speech():
    det = TurnDetector(silence_s=1.5)
    segs = [_seg([("hello", 0.0, 0.4), ("there", 0.5, 0.9)])]
    assert det.update(segs, 5.0) == []


def test_response_policy_greeting():
    pol = ResponsePolicy()
    reply = pol.generate(Turn("Hi wodehaus!", 1.0))
    assert reply is not None and "Hello" in reply


def test_response_policy_question_is_honest():
    pol = ResponsePolicy()
    reply = pol.generate(Turn("Wodehaus can you hear me?", 1.0))
    assert reply is not None
    assert "don't understand" in reply or "hear" in reply


def test_response_policy_silent_when_not_addressed():
    pol = ResponsePolicy()
    assert pol.generate(Turn("hello there", 1.0)) is None
