"""Tests for hva/understanding.py -- comprehension short of an LLM."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hva.understanding import (  # noqa: E402
    Intent, classify, look_direction, direction_bias,
    PerceptualState, DialogueState, understand,
)
from hva.conversation import Turn  # noqa: E402


def test_classify_intents():
    assert classify("hi wodehaus") == Intent.GREETING
    assert classify("bye wodehaus") == Intent.FAREWELL
    assert classify("wodehaus who are you") == Intent.IDENTITY
    assert classify("wodehaus what do you see") == Intent.SEE_QUESTION
    assert classify("hey what are you looking at") == Intent.SEE_QUESTION
    assert classify("what do you hear") == Intent.HEAR_QUESTION
    assert classify("wodehaus look left") == Intent.LOOK_COMMAND
    assert classify("look at the red car") == Intent.LOOK_AT
    assert classify("thanks wodehaus") == Intent.THANKS
    assert classify("can you hear me") == Intent.YESNO_QUESTION
    assert classify("where is the exit") == Intent.WH_QUESTION


def test_look_direction():
    assert look_direction("wodehaus look left") == "left"
    assert look_direction("look up please") == "up"
    assert look_direction("look middle") == "center"
    assert look_direction("hello there") is None


def test_direction_bias_targets_map_side():
    b = direction_bias("left")
    assert b.shape == (56, 56)
    iy, ix = divmod(b.argmax(), 56)
    assert ix < 28 and iy == 28  # left middle
    b = direction_bias("up")
    iy, ix = divmod(b.argmax(), 56)
    assert iy < 28 and ix == 28  # top middle


class _FakeLoop:
    """Minimal OnlineLevel3 stand-in for PerceptualState tests."""
    def __init__(self):
        self.scanpath = [(0.0, 112.0, 112.0),
                         (1200.0, 40.0, 60.0),
                         (2400.0, 180.0, 150.0)]

        class _Map:
            def peak(self):
                return (35.0, 20.0, 0.5)
        self.jmap = _Map()


def test_perceptual_state_describes_gaze():
    p = PerceptualState(_FakeLoop())
    assert p.gaze_now() == (180.0, 150.0)
    assert p.gaze_where() == "lower right"
    d = p.describe()
    assert "lower right" in d and "2 saccades" in d


def test_understand_look_command_returns_bias():
    dlg = DialogueState()
    reply, bias = understand("wodehaus look left", perceptual=None,
                             dialogue=dlg)
    assert reply == "Looking left."
    assert bias is not None and bias.shape == (56, 56)
    assert dlg.last_region == "left"


def test_understand_look_at_is_honest():
    reply, bias = understand("wodehaus look at the red car")
    assert bias is None
    assert "don't know what things look like" in reply


def test_understand_see_question_uses_perception():
    p = PerceptualState(_FakeLoop())
    reply, _ = understand("wodehaus what do you see?", perceptual=p)
    assert "lower right" in reply  # grounded in the fake gaze, not canned


def test_understand_identity():
    reply, _ = understand("wodehaus who are you")
    assert "Wodehaus" in reply and "priority map" in reply


def test_dialogue_history_keeps_turns():
    dlg = DialogueState(max_turns=3)
    for i in range(5):
        dlg.add(f"turn {i}", Intent.STATEMENT, f"reply {i}")
    assert len(dlg.turns) == 3
    assert "turn 4" in dlg.history_text()


def test_task_bias_reaches_the_map():
    """Language -> perception: a bias array steers the joint map peak."""
    from hvm.online import OnlineLevel3
    loop = OnlineLevel3(dva_per_px=0.1, t_end_ms=5000.0)
    vis = np.zeros((56, 56), dtype=np.float32)
    aud = (np.zeros(56, dtype=np.float32), np.zeros(56, dtype=np.float32))
    # No bias: map stays near zero.
    loop.tick(0.0, vis, aud, 0.0)
    assert loop.jmap.map.max() < 1e-6
    # With a left bias: the map peak moves left.
    loop.tick(100.0, vis, aud, 0.0, task_bias=direction_bias("left"))
    px, py, _ = loop.jmap.peak()
    assert px < 28, f"bias did not move the peak left (px={px})"


def test_speaker_play_stop():
    """Speaker tracks playback; stop() marks interruption."""
    from hva.conversation import Speaker
    import tempfile
    # Fake a 2s audio file for duration probing.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    import wave
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 32000)  # 2 s of silence
    sp = Speaker()
    sp.play(path, now_s=10.0)
    assert sp.is_playing(11.0)
    assert not sp.is_playing(12.5)
    assert sp.check_finished(12.5)
    sp.play(path, now_s=20.0)
    sp.stop()  # barge-in
    assert sp.interrupted
    assert not sp.is_playing(20.5)


def test_energy_vad_fires_on_speech_onset():
    """VAD fires ~300 ms after energy jumps above the noise floor."""
    from hva.conversation import EnergyVAD
    vad = EnergyVAD()
    silence = np.zeros(1600, dtype=np.float32)
    speech = (np.random.default_rng(0).standard_normal(1600)
              .astype(np.float32) * 0.1)
    # Settle the floor on silence.
    for _ in range(20):
        assert vad.update(silence) is False
    # Speech: fires on the 3rd hot tick, then every 3 ticks while
    # speech continues (sustained speech keeps the onset signal alive).
    assert vad.update(speech) is False
    assert vad.update(speech) is False
    assert vad.update(speech) is True
    assert vad.update(speech) is False
    assert vad.update(speech) is False
    assert vad.update(speech) is True
    # Back to silence: floor re-adapts, no fire.
    for _ in range(10):
        assert vad.update(silence) is False


def test_llm_payload_and_look_split():
    from hva.llm import build_payload, split_look_command
    from hva.understanding import DialogueState
    d = DialogueState()
    p = build_payload("Wodehaus, what do you see?", perceptual=None,
                      dialogue=d)
    assert p["turn"] == "Wodehaus, what do you see?"
    assert p["perceptual_state"] is None
    assert p["system"] == "Wodehaus"
    text, direction = split_look_command("On it.\nLOOK: left")
    assert text == "On it." and direction == "left"
    text, direction = split_look_command("Just talking, no look here.")
    assert direction is None


def test_llm_fallback_without_key():
    """No key -> rule-based reply, no crash."""
    from hva.conversation import ResponsePolicy, Turn
    from hva.llm import ApiGenerator
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    policy = ResponsePolicy()
    policy.llm = ApiGenerator(api_key=None)
    assert not policy.llm.available
    reply = policy.generate(Turn("Hi Wodehaus", 1.0))
    assert isinstance(reply, str)
    assert reply is not None
    # Rule-based fallback for a greeting (not an LLM reply).
    assert "Hello" in reply


def test_llm_path_with_mock_and_look_bias():
    """Mocked API: reply text returned, LOOK line becomes task bias."""
    import numpy as np
    from hva.conversation import ResponsePolicy, Turn
    from hva.llm import ApiGenerator

    class FakeGen(ApiGenerator):
        available = True
        def generate(self, payload):
            assert "Wodehaus look left" in payload["turn"]
            return "Looking left now.\nLOOK: left"

    policy = ResponsePolicy()
    policy.llm = FakeGen(api_key="fake")
    reply = policy.generate(Turn("Wodehaus look left", 1.0))
    assert reply == "Looking left now."
    bias = policy.take_bias()
    assert bias is not None and bias.shape == (56, 56)
    # Bias peak sits left of center.
    assert np.unravel_index(bias.argmax(), bias.shape)[1] < 28


def test_llm_failure_falls_back_to_rules():
    """API error -> rule-based reply, bias still works."""
    from hva.conversation import ResponsePolicy, Turn
    from hva.llm import ApiGenerator

    class BrokenGen(ApiGenerator):
        available = True
        def generate(self, payload):
            raise ConnectionError("nope")

    policy = ResponsePolicy()
    policy.llm = BrokenGen(api_key="fake")
    reply = policy.generate(Turn("Wodehaus look right", 1.0))
    assert isinstance(reply, str)  # rule fallback answered
    assert policy.take_bias() is not None


def test_select_llm_backend_none():
    """none -> no backend, rule-based only."""
    from hva.llm import select_llm_backend
    backend, desc = select_llm_backend("none")
    assert backend is None
    assert "rule-based" in desc


def test_select_llm_backend_api_no_key():
    """api without key -> None, fallback described."""
    from hva.llm import select_llm_backend
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    backend, desc = select_llm_backend("api")
    assert backend is None
    assert "no key" in desc


def test_select_llm_backend_api_with_key():
    """api with key -> ApiGenerator, keyed."""
    from hva.llm import select_llm_backend
    backend, desc = select_llm_backend("api", api_model="test-model")
    # Uses env key if present; if not, backend is None. Either way,
    # the description must match the backend.
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        assert backend is not None
        assert "keyed" in desc
    else:
        assert backend is None
        assert "no key" in desc


def test_select_llm_backend_local_unreachable():
    """local with no server -> None, unreachable described."""
    from hva.llm import select_llm_backend
    # Port 9 is discard; nothing listens there.
    backend, desc = select_llm_backend(
        "local", llm_url="http://localhost:9")
    assert backend is None
    assert "unreachable" in desc


def test_select_llm_backend_auto_fallback():
    """auto with no server and no key -> None."""
    from hva.llm import select_llm_backend
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    backend, desc = select_llm_backend(
        "auto", llm_url="http://localhost:9")
    assert backend is None
    assert "none" in desc
