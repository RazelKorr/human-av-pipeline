"""Tests for the conversational turn-taking (hva/conversation.py)."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hva.conversation import (  # noqa: E402
    TurnDetector, ResponsePolicy, Turn, is_addressed, AsyncTTS,
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


def test_async_tts_keeps_tick_loop_running():
    """The tick loop must not stall while TTS synthesizes.

    A slow mock speak (0.4 s) runs in the background while we simulate
    10 Hz ticks. We should complete many ticks before the synthesis
    finishes, then collect it via poll_ready().
    """
    import time

    def slow_speak(text, outpath, voice="v"):
        time.sleep(0.4)
        with open(outpath, "w") as f:
            f.write(text)
        return outpath

    tts = AsyncTTS(speak_fn=slow_speak)
    try:
        out = "/tmp/async_tts_test.txt"
        tts.submit("hello", out)
        ticks = 0
        ready = []
        t0 = time.time()
        # Simulate 10 Hz ticks for up to 2 s.
        while time.time() - t0 < 2.0:
            ready += tts.poll_ready()
            ticks += 1
            time.sleep(0.05)  # 20 Hz test loop (faster than 10 Hz ticks)
            if ready:
                break
        # We kept ticking while synthesis ran: at least 4 ticks in 0.4 s
        # at 20 Hz, and we eventually got the result.
        assert ticks >= 4, f"tick loop stalled: only {ticks} ticks"
        assert len(ready) == 1
        assert ready[0][0] == out
        assert not tts.has_pending()
    finally:
        tts.shutdown()


def test_async_tts_failure_does_not_kill_loop():
    def bad_speak(text, outpath, voice="v"):
        raise RuntimeError("boom")

    tts = AsyncTTS(speak_fn=bad_speak)
    try:
        tts.submit("hi", "/tmp/async_tts_bad.mp3")
        import time
        time.sleep(0.2)
        # Should log the failure and return [], not raise.
        assert tts.poll_ready() == []
        assert not tts.has_pending()
    finally:
        tts.shutdown()
