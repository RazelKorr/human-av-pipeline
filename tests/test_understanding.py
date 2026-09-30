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
