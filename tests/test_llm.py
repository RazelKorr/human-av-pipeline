"""Tests for hva.llm: the grounding payload the model reads."""
import pytest

from hva.llm import build_payload, split_look_command
from hva.understanding import DialogueState, ObjectMemory


def _mem():
    mem = ObjectMemory()
    mem.add("windows", 10.0, 20.0, 1000.0, 0.80)
    mem.add("gate", 40.0, 40.0, 1000.0, 0.90)
    return mem


def test_payload_lists_recognized_objects_with_regions():
    p = build_payload("wodehaus what do you see", memory=_mem(),
                      t_now_ms=5000.0)
    objs = p["recognized_objects"]
    assert objs is not None and len(objs) == 2
    assert any(o.startswith("windows (") for o in objs)
    assert any("gate (" in o and "confidence 0.90" in o for o in objs)


def test_payload_without_memory_says_none():
    p = build_payload("hello", dialogue=DialogueState())
    assert p["recognized_objects"] is None
    assert p["dialogue_history"] == ""


def test_payload_carries_dialogue_history():
    d = DialogueState()
    d.add("hi wodehaus", "greeting", "Hello!")
    p = build_payload("what now", dialogue=d)
    assert "hi wodehaus" in p["dialogue_history"]


def test_split_look_command():
    text, direction = split_look_command("Looking left.\nLOOK: left")
    assert (text, direction) == ("Looking left.", "left")
    text, direction = split_look_command("Just talking.")
    assert (text, direction) == ("Just talking.", None)
    text, direction = split_look_command("Hmm.\nLOOK: sideways")
    assert (text, direction) == ("Hmm.\nLOOK: sideways", None)


def test_policy_llm_path_receives_memory(monkeypatch):
    from hva.conversation import ResponsePolicy, Turn

    seen = {}

    class StubLLM:
        available = True

        def generate(self, payload):
            seen.update(payload)
            return "I see windows over there."

    pol = ResponsePolicy()
    pol.memory = _mem()
    pol.llm = StubLLM()
    reply = pol.generate(Turn("Wodehaus what do you see?", 5.0))
    assert reply == "I see windows over there."
    assert seen["recognized_objects"] is not None
    assert any("windows" in o for o in seen["recognized_objects"])


# --- AgentGenerator (file-handoff backend) ---

import json
import os

from hva.llm import AgentGenerator, select_llm_backend


def test_agent_backend_roundtrip_via_files(tmp_path):
    gen = AgentGenerator(handoff_dir=str(tmp_path), poll_s=0.01)
    assert gen.available
    # Operator answers before generate() even starts polling: the
    # prompt file must still be written, and the reply returned.
    (tmp_path / "turn-0001.response.txt").write_text("Hello back.\n")
    reply = gen.generate({"turn": "hello", "system": "Wodehaus"})
    assert reply == "Hello back."
    prompt_path = tmp_path / "turn-0001.prompt.json"
    assert prompt_path.exists()
    prompt = json.loads(prompt_path.read_text())
    assert prompt["payload"]["turn"] == "hello"
    assert "system" in prompt and "instruction" in prompt


def test_agent_backend_times_out(tmp_path):
    gen = AgentGenerator(handoff_dir=str(tmp_path), poll_s=0.01,
                         timeout_s=0.05)
    with pytest.raises(TimeoutError):
        gen.generate({"turn": "hello"})


def test_select_agent_backend():
    gen, desc = select_llm_backend("agent")
    assert isinstance(gen, AgentGenerator)
    assert "agent" in desc
