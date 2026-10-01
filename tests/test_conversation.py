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
    assert is_addressed("World House, look to the left, please.")  # observed 2026-10-01
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


def _speech_frac(hot_spans):
    """speech(t0, t1) callable: 1.0 inside hot_spans, else 0.0."""
    def fn(t0, t1):
        hot = sum(max(0.0, min(t1, e) - max(t0, s)) for s, e in hot_spans)
        total = max(t1 - t0, 1e-9)
        return hot / total
    return fn


def test_turn_detector_suppresses_hallucination_over_silence():
    """Whisper's phantom 'Woodhouse, look left' over digital silence:
    the transcript claims speech where the acoustic record shows none.
    """
    det = TurnDetector(silence_s=1.5)
    segs = [_seg([("Woodhouse", 10.0, 10.4), ("look", 10.5, 10.7),
                  ("left", 10.8, 11.0)])]
    # No acoustic speech anywhere near the claimed span.
    turns = det.update(segs, 13.0, speech=_speech_frac([]))
    assert turns == []
    assert det.suppressed == [("no_speech", "Woodhouse look left", 13.0)]


def test_turn_detector_fires_when_acoustics_agree():
    det = TurnDetector(silence_s=1.5)
    segs = [_seg([("Woodhouse", 10.0, 10.4), ("hi", 10.5, 10.8)])]
    turns = det.update(segs, 13.0, speech=_speech_frac([(9.5, 11.5)]))
    assert len(turns) == 1
    assert turns[0].t_start == 10.0 and turns[0].t_end == 10.8
    assert det.suppressed == []


def test_turn_detector_guard_disabled_without_speech_fn():
    """Backward compatibility: no speech evidence, old behavior."""
    det = TurnDetector(silence_s=1.5)
    segs = [_seg([("Woodhouse", 10.0, 10.4), ("hi", 10.5, 10.8)])]
    assert len(det.update(segs, 13.0)) == 1


def test_turn_detector_suppresses_window_refire():
    """Overlapping windows re-emit the same utterance with shifted word
    timings; the re-emission's addressed span overlaps the fired turn's.
    Segments accumulate across updates, as the rolling transcriber does.
    """
    det = TurnDetector(silence_s=1.5)
    speech = _speech_frac([(0.0, 100.0)])
    segs = [_seg([("hey", 2.0, 2.2), ("Woodhouse", 2.3, 2.7),
                  ("hi", 2.8, 3.0)])]
    assert len(det.update(segs, 5.0, speech=speech)) == 1
    # Same audio re-transcribed with shifted timings and extra words:
    # new words arrive, but their span overlaps the fired turn's.
    segs = segs + [_seg([("hey", 2.1, 2.3), ("Woodhouse", 2.4, 2.8),
                         ("hi", 2.9, 3.1), ("hi", 3.2, 3.4)])]
    assert det.update(segs, 8.0, speech=speech) == []
    assert det.suppressed[-1][0] == "refire"


def test_turn_detector_allows_genuine_repeat():
    """A real second address, later and non-overlapping, still fires."""
    det = TurnDetector(silence_s=1.5)
    speech = _speech_frac([(0.0, 100.0)])
    segs = [_seg([("Woodhouse", 2.0, 2.4), ("left", 2.5, 2.8)])]
    assert len(det.update(segs, 5.0, speech=speech)) == 1
    segs = segs + [_seg([("Woodhouse", 20.0, 20.4), ("right", 20.5, 20.8)])]
    turns = det.update(segs, 23.0, speech=speech)
    assert len(turns) == 1
    assert "right" in turns[0].text


def test_turn_detector_second_turn_after_replacement_reemission():
    """Regression (2026-09-30): the rolling transcriber replaces a
    segment in place when a later window re-emits it with shifted
    timings. The replacement's words must count as new -- with the old
    word-count watermark they didn't, and the detector went deaf after
    the first turn (turns 2 and 3 of the agent-handoff test never
    fired)."""
    det = TurnDetector(silence_s=1.5)
    speech = _speech_frac([(0.0, 100.0)])
    seg1 = _seg([("Woodhouse", 4.7, 5.1), ("what", 5.2, 5.4),
                 ("see", 5.5, 5.8)])
    assert len(det.update([seg1], 9.0, speech=speech)) == 1
    # Later window re-emits utterance 1 with shifted timings: the
    # transcriber replaces the segment (new identity, same span).
    seg1b = _seg([("Woodhouse", 4.8, 5.2), ("what", 5.3, 5.5),
                  ("see", 5.6, 5.9)])
    assert det.update([seg1b], 12.0, speech=speech) == []
    assert det.suppressed[-1][0] == "refire"
    # A genuine second address arrives as a new segment: it must fire.
    seg2 = _seg([("Woodhouse", 28.5, 28.9), ("left", 29.0, 29.3)])
    turns = det.update([seg1b, seg2], 33.0, speech=speech)
    assert len(turns) == 1
    assert "left" in turns[0].text


def test_turn_detector_suppressions_carry_stream_time():
    """Suppressed turns record when they happened, so the end-of-run
    log doesn't stamp them all with the final timestamp."""
    det = TurnDetector(silence_s=1.5)
    segs = [_seg([("Woodhouse", 10.0, 10.4), ("look", 10.5, 10.7),
                  ("left", 10.8, 11.0)])]
    det.update(segs, 13.0, speech=_speech_frac([]))
    assert det.suppressed == [("no_speech", "Woodhouse look left", 13.0)]


def test_energy_vad_exposes_per_tick_presence():
    """update() still reports onsets; .hot reports per-tick presence."""
    import numpy as np
    from hva.conversation import EnergyVAD
    vad = EnergyVAD()
    quiet = np.zeros(1600, dtype=np.float32)
    loud = np.full(1600, 0.5, dtype=np.float32)
    assert vad.update(quiet) is False
    assert vad.hot is False
    # Three hot ticks -> one onset; hot is True on every hot tick.
    assert vad.update(loud) is False and vad.hot is True
    assert vad.update(loud) is False and vad.hot is True
    assert vad.update(loud) is True and vad.hot is True
    assert vad.update(quiet) is False and vad.hot is False


def test_conversation_runner_flushes_trailing_audio(tmp_path):
    """Regression: the runner once silently dropped the trailing partial
    audio chunk (13.8 s of audio -> 105 moments instead of 137), losing
    any turns in it. Runs the real script; transcription is disabled
    via a huge --tx-step so no Whisper model loads."""
    import re
    import shutil
    import subprocess
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    src = tmp_path / "tail.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d=13.8",
         "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=13.8",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(src)],
        check=True, timeout=120)
    outdir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "run_conversation.py"),
         "--src", str(src), "--seconds", "13.8", "--llm", "none",
         "--tx-step", "100", "--outdir", str(outdir)],
        capture_output=True, text=True, timeout=300, cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]
    m = re.search(r"stream ended: (\d+) moments", proc.stdout)
    assert m, proc.stdout[-2000:]
    n_mom = int(m.group(1))
    # 13.8 s at 10 Hz = 138 moments; without the flush() this was 105.
    assert n_mom >= 135, f"trailing audio lost: {n_mom} moments"


def test_response_policy_greeting():
    pol = ResponsePolicy()
    reply = pol.generate(Turn("Hi wodehaus!", 1.0))
    assert reply is not None and "Hello" in reply


def test_response_policy_question_is_honest():
    pol = ResponsePolicy()
    reply = pol.generate(Turn("Wodehaus can you hear me?", 1.0))
    assert reply is not None
    # Pinned to the actual template: honest "don't know" fallback.
    assert "don't know what to make of that yet" in reply


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


def test_async_tts_cancel_pending():
    """cancel_pending drops queued syntheses before they run."""
    import threading
    gate = threading.Event()

    def blocking_speak(text, outpath, voice="v"):
        gate.wait(timeout=5.0)  # block until released
        return outpath

    tts = AsyncTTS(max_workers=1, speak_fn=blocking_speak)
    try:
        # First submit occupies the worker; second queues behind it.
        tts.submit("first", "/tmp/async_tts_c1.mp3")
        tts.submit("second", "/tmp/async_tts_c2.mp3")
        import time
        time.sleep(0.2)  # let the first start, second stay queued
        dropped = tts.cancel_pending()
        assert dropped >= 1  # the queued one was cancelled
        gate.set()  # release the worker
        time.sleep(0.2)
        ready = tts.poll_ready()
        # Only the first (already running) completed.
        assert len(ready) == 1
        assert ready[0][1] == "first"
    finally:
        gate.set()
        tts.shutdown()


class _StubLLM:
    """Minimal stand-in for an hva.llm backend: canned reply, always on."""

    def __init__(self, reply: str):
        self._reply = reply

    @property
    def available(self) -> bool:
        return True

    def generate(self, payload: dict) -> str:
        return self._reply


def test_llm_look_backstop_steers_from_intent():
    """If the model drops the LOOK line on a look command, the rule-based
    intent classifier still steers the shared map. Rules hold the floor."""
    from hva.understanding import direction_bias
    import numpy as np
    pol = ResponsePolicy()
    pol.llm = _StubLLM("I'm looking at the upper right of the frame.")
    reply = pol.generate(Turn("Wodehaus, look left.", 1.0))
    assert reply == "I'm looking at the upper right of the frame."
    assert pol.pending_bias is not None
    assert np.array_equal(pol.pending_bias, direction_bias("left"))


def test_llm_look_line_still_wins_over_backstop():
    """An explicit LOOK line from the model takes precedence over the
    rule-based backstop."""
    from hva.understanding import direction_bias
    import numpy as np
    pol = ResponsePolicy()
    pol.llm = _StubLLM("Okay, looking over there.\nLOOK: right")
    reply = pol.generate(Turn("Wodehaus, look left.", 1.0))
    assert reply == "Okay, looking over there."
    assert pol.pending_bias is not None
    assert np.array_equal(pol.pending_bias, direction_bias("right"))


def test_llm_no_backstop_without_look_intent():
    """The backstop only fires on look commands; ordinary replies leave
    the shared map alone."""
    pol = ResponsePolicy()
    pol.llm = _StubLLM("I heard you say hi.")
    reply = pol.generate(Turn("Wodehaus, say hi.", 1.0))
    assert reply == "I heard you say hi."
    assert pol.pending_bias is None


# -------------------------------------- item 4: label stream tests

from hva.conversation import PerceptualState  # noqa: E402 (re-export)
from hva.understanding import ObjectMemory  # noqa: E402


class _FakeLoop:
    """Minimal OnlineLevel3 stand-in: one fixation, quiet map."""
    def __init__(self):
        self.scanpath = [(0.0, 112.0, 112.0)]

        class _Map:
            def peak(self):
                return (28.0, 28.0, 0.01)  # below the 0.05 mention threshold
        self.jmap = _Map()


def test_see_question_names_recognized_objects():
    pol = ResponsePolicy()
    pol.perceptual = PerceptualState(_FakeLoop(), memory=pol.memory)
    pol.memory.add("gate", 14.0, 28.0, t_ms=1000.0, conf=0.8)
    pol.memory.add("windows", 42.0, 20.0, t_ms=2000.0, conf=0.7)
    reply = pol.generate(Turn("Wodehaus, what do you see?", 3.0))
    assert reply is not None
    assert "gate" in reply and "windows" in reply
    # positions are qualitative, grounded in the sightings
    assert "left" in reply and "right" in reply


def test_see_question_no_objects_no_claim():
    pol = ResponsePolicy()
    pol.perceptual = PerceptualState(_FakeLoop(), memory=pol.memory)
    reply = pol.generate(Turn("Wodehaus, what do you see?", 3.0))
    assert reply is not None
    assert "recognized" not in reply


def test_policy_look_at_uses_memory():
    pol = ResponsePolicy()
    pol.perceptual = PerceptualState(_FakeLoop(), memory=pol.memory)
    pol.memory.add("gate", 14.0, 28.0, t_ms=1000.0, conf=0.9)
    reply = pol.generate(Turn("Wodehaus, look at the gate", 3.0))
    assert reply is not None and "gate" in reply
    bias = pol.take_bias()
    assert bias is not None
    iy, ix = divmod(bias.argmax(), 56)
    assert (ix, iy) == (14, 28)
    assert pol.take_bias() is None  # one-shot


def test_policy_look_at_unknown_no_bias():
    pol = ResponsePolicy()
    pol.perceptual = PerceptualState(_FakeLoop(), memory=pol.memory)
    pol.memory.add("gate", 14.0, 28.0, t_ms=1000.0, conf=0.9)
    reply = pol.generate(Turn("Wodehaus, look at the red car", 3.0))
    assert reply is not None and "don't see one" in reply
    assert "gate" in reply  # known objects still listed
    assert pol.take_bias() is None


def test_describe_includes_recognized_objects():
    mem = ObjectMemory()
    mem.add("gate", 14.0, 28.0, t_ms=1000.0, conf=0.8)
    p = PerceptualState(_FakeLoop(), memory=mem)
    d = p.describe()
    assert "gate" in d and "left" in d


def test_describe_without_memory_unchanged():
    p = PerceptualState(_FakeLoop())
    d = p.describe()
    assert "recognized" not in d


def test_policy_detector_wiring():
    from PIL import Image

    class StubDetector:
        def detect(self, frame, queries):
            assert queries == ["a gate"], queries
            return [("a gate", 150.0, 70.0, 170.0, 90.0, 0.55)]

    pol = ResponsePolicy()
    pol.detector = StubDetector()
    pol.frame_fn = lambda: Image.new("RGB", (224, 224))
    reply = pol.generate(Turn("Wodehaus where is the gate", 1.0))
    assert reply is not None and "Found the gate" in reply
    assert pol.pending_bias is not None
    iy, ix = divmod(int(pol.pending_bias.argmax()), 56)
    assert (ix, iy) == (40, 20)
    # the detection became a track in policy memory
    assert pol.memory.locate("gate", 2000.0) is not None


def test_turn_detector_does_not_fire_on_truncated_window_tail():
    """Regression (2026-10-01): a transcription window that cuts an
    utterance off mid-word makes the truncated tail look like a
    completed turn to a stream-time silence check (turn 1 fired as
    'World House Wattoop'). The detector must endpoint on transcript
    evidence -- the latest window has to extend past
    last_end + silence_s -- so the turn waits for a window that
    actually contains the utterance's end."""
    det = TurnDetector(silence_s=1.5)
    speech = _speech_frac([(0.0, 100.0)])
    # Window ending at t=6 cuts the utterance off: "what do you s-".
    cut = [_seg([("Woodhouse", 4.8, 5.2), ("what", 5.3, 5.5),
                 ("do", 5.6, 5.7), ("you", 5.8, 5.9), ("s", 5.9, 6.0)])]
    assert det.update(cut, 7.5, speech=speech, tx_time=6.0) == []
    # Next window (ends t=9) holds the whole utterance: fires, complete.
    full = [_seg([("Woodhouse", 4.8, 5.2), ("what", 5.3, 5.5),
                  ("do", 5.6, 5.7), ("you", 5.8, 6.0), ("see", 6.1, 6.7)])]
    turns = det.update(full, 9.0, speech=speech, tx_time=9.0)
    assert len(turns) == 1
    assert turns[0].text == "Woodhouse what do you see"
