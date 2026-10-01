"""Tests for hva/understanding.py -- comprehension short of an LLM."""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from hva.understanding import (  # noqa: E402
    Intent, classify, look_direction, direction_bias,
    PerceptualState, DialogueState, understand,
    ObjectMemory, split_spatial, resolve_spatial_referent,
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
    assert classify("where is the exit") == Intent.LOOK_AT  # visual search


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
    assert "don't see one right now" in reply


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


def test_select_llm_backend_api_with_key(monkeypatch):
    """api with key -> ApiGenerator, keyed (deterministic via monkeypatch)."""
    from hva.llm import select_llm_backend, ApiGenerator
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    backend, desc = select_llm_backend("api", api_model="test-model")
    assert backend is not None
    assert isinstance(backend, ApiGenerator)
    assert backend.api_key == "sk-test-fake-key"
    assert "keyed" in desc


def test_select_llm_backend_api_explicit_key(monkeypatch):
    """ApiGenerator accepts key directly, not just via env."""
    from hva.llm import ApiGenerator
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    gen = ApiGenerator(api_key="sk-direct-key", model="test-model")
    assert gen.available
    assert gen.api_key == "sk-direct-key"


def test_select_llm_backend_local_reachable(monkeypatch):
    """local with reachable server -> LocalGenerator."""
    from hva.llm import select_llm_backend, LocalGenerator
    import urllib.request

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=3: FakeResp())
    backend, desc = select_llm_backend(
        "local", llm_url="http://localhost:8080")
    assert backend is not None
    assert isinstance(backend, LocalGenerator)
    assert "reachable" in desc


def test_select_llm_backend_auto_prefers_local(monkeypatch):
    """auto -> local when server reachable, even with API key set."""
    from hva.llm import select_llm_backend, LocalGenerator
    import urllib.request

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=3: FakeResp())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    backend, desc = select_llm_backend(
        "auto", llm_url="http://localhost:8080")
    assert isinstance(backend, LocalGenerator)
    assert "auto -> local" in desc


def test_select_llm_backend_auto_falls_to_api(monkeypatch):
    """auto -> api when local unreachable, no HF token, but key is set."""
    from hva.llm import select_llm_backend, ApiGenerator
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    backend, desc = select_llm_backend(
        "auto", llm_url="http://localhost:9")
    assert isinstance(backend, ApiGenerator)
    assert "auto -> api" in desc


def test_select_llm_backend_local_unreachable():
    """local with no server -> None, unreachable described."""
    from hva.llm import select_llm_backend
    # Port 9 is discard; nothing listens there.
    backend, desc = select_llm_backend(
        "local", llm_url="http://localhost:9")
    assert backend is None
    assert "unreachable" in desc


def test_select_llm_backend_auto_fallback(monkeypatch):
    """auto with no server, no HF token, no API key -> keyless free backend."""
    from hva.llm import select_llm_backend, PollinationsGenerator
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    backend, desc = select_llm_backend(
        "auto", llm_url="http://localhost:9")
    assert isinstance(backend, PollinationsGenerator)
    assert "auto -> free" in desc


def test_select_llm_backend_free():
    """free -> PollinationsGenerator, always available, keyless."""
    from hva.llm import select_llm_backend, PollinationsGenerator
    backend, desc = select_llm_backend("free")
    assert isinstance(backend, PollinationsGenerator)
    assert backend.available
    assert "keyless" in desc


def test_free_generator_generate_mocked(monkeypatch):
    """PollinationsGenerator POSTs an OpenAI-style body: model, system and
    user messages, prompt in the body (not the URL)."""
    import json
    import urllib.request
    from hva.llm import PollinationsGenerator, SYSTEM_PROMPT

    captured = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "  hello post ok  "}}],
            }).encode()

    def fake_urlopen(req, timeout=90.0):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["content_type"] = req.get_header("Content-type")
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    gen = PollinationsGenerator(model="openai")
    reply = gen.generate({"turn": "what do you see?"})
    assert reply == "hello post ok"  # stripped
    assert captured["url"] == "https://text.pollinations.ai/openai"
    assert captured["content_type"] == "application/json"
    body = captured["body"]
    assert body["model"] == "openai"
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert "what do you see?" in body["messages"][1]["content"]
    # The payload must not leak into the URL.
    assert "what do you see?" not in captured["url"]


def test_free_generator_retries_connection_drop_not_429(monkeypatch):
    """One retry on RemoteDisconnected; HTTP errors (e.g. 429) propagate
    immediately so the service's backoff signals are respected."""
    import http.client
    import json
    import urllib.request
    from urllib.error import HTTPError
    from unittest.mock import patch
    from hva.llm import PollinationsGenerator

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "recovered"}}]}).encode()

    calls = {"n": 0}
    def flaky(req, timeout=90.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http.client.RemoteDisconnected("drop")
        return FakeResp()
    with patch.object(urllib.request, "urlopen", flaky):
        assert PollinationsGenerator().generate({"turn": "hi"}) == "recovered"
    assert calls["n"] == 2

    calls2 = {"n": 0}
    def rate_limited(req, timeout=90.0):
        calls2["n"] += 1
        raise HTTPError(req.full_url, 429, "Too Many Requests", {}, None)
    try:
        with patch.object(urllib.request, "urlopen", rate_limited):
            PollinationsGenerator().generate({"turn": "hi"})
    except HTTPError as e:
        assert e.code == 429
    assert calls2["n"] == 1


def test_select_llm_backend_auto_prefers_api_over_free(monkeypatch):
    """auto -> api when keyed, even though the free backend is available."""
    from hva.llm import select_llm_backend, ApiGenerator
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    backend, desc = select_llm_backend(
        "auto", llm_url="http://localhost:9")
    assert isinstance(backend, ApiGenerator)
    assert "auto -> api" in desc


def test_select_llm_backend_hf_no_token(monkeypatch):
    """hf without token -> None, fallback described."""
    from hva.llm import select_llm_backend
    monkeypatch.delenv("HF_TOKEN", raising=False)
    backend, desc = select_llm_backend("hf")
    assert backend is None
    assert "no token" in desc


def test_select_llm_backend_hf_with_token(monkeypatch):
    """hf with token -> HFGenerator, tokened (deterministic)."""
    from hva.llm import select_llm_backend, HFGenerator
    monkeypatch.setenv("HF_TOKEN", "hf-test-fake-token")
    backend, desc = select_llm_backend("hf", hf_model="Qwen/Qwen3-8B")
    assert backend is not None
    assert isinstance(backend, HFGenerator)
    assert backend.api_token == "hf-test-fake-token"
    assert backend.model == "Qwen/Qwen3-8B"
    assert "tokened" in desc


def test_hf_generator_generate_mocked(monkeypatch):
    """HFGenerator.generate parses the OpenAI-style response."""
    import json
    import urllib.request
    from hva.llm import HFGenerator

    captured = {}

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({
                "choices": [{"message": {
                    "content": "I see you looking left.\nLOOK: left"}}]
            }).encode()

    def fake_urlopen(req, timeout=60.0):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data.decode())
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    gen = HFGenerator(api_token="hf-test-fake-token")
    assert gen.available
    reply = gen.generate({"turn": "what do you see?"})
    assert reply == "I see you looking left.\nLOOK: left"
    assert captured["url"] == \
        "https://router.huggingface.co/v1/chat/completions"
    assert captured["auth"] == "Bearer hf-test-fake-token"
    assert captured["body"]["model"] == "Qwen/Qwen3-8B"
    roles = [m["role"] for m in captured["body"]["messages"]]
    assert roles == ["system", "user"]


def test_hf_generator_generate_no_token():
    """HFGenerator.generate without token raises instead of calling."""
    from hva.llm import HFGenerator
    import os
    os.environ.pop("HF_TOKEN", None)
    gen = HFGenerator(api_token=None)
    assert not gen.available
    try:
        gen.generate({"turn": "hi"})
    except RuntimeError as e:
        assert "HF_TOKEN" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_select_llm_backend_auto_prefers_hf_over_api(monkeypatch):
    """auto -> hf when local is down, HF tokened, API keyed (free first)."""
    from hva.llm import select_llm_backend, HFGenerator
    monkeypatch.setenv("HF_TOKEN", "hf-test-fake-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    backend, desc = select_llm_backend(
        "auto", llm_url="http://localhost:9")
    assert isinstance(backend, HFGenerator)
    assert "auto -> hf" in desc


# --------------------------------------------- referring-expression tests

from hva.understanding import (  # noqa: E402
    ObjectMemory, extract_referent, object_bias, _normalize_name,
)


def test_normalize_name():
    assert _normalize_name("the windows") == "window"
    assert _normalize_name("a gate") == "gate"
    assert _normalize_name("light strip") == "lightstrip"  # label: light-strip
    assert _normalize_name("gate edge") == "gateedge"


def test_extract_referent():
    assert extract_referent("wodehaus look at the windows") == "windows"
    assert extract_referent("look at a red car please") == "red car"
    assert extract_referent("look toward the gate") == "gate"
    assert extract_referent("look left") is None


def test_object_memory_locate_most_recent():
    mem = ObjectMemory()
    mem.add("windows", 40.0, 20.0, t_ms=1000.0, conf=0.8)
    mem.add("gate", 10.0, 10.0, t_ms=2000.0, conf=0.9)
    mem.add("windows", 42.0, 22.0, t_ms=3000.0, conf=0.7)
    mx, my, conf, age = mem.locate("the windows", t_now=4000.0)
    assert (mx, my) == (41.0, 21.0)  # track-smoothed, not raw last
    assert conf == 0.75 and age == 1000.0
    assert mem.locate("car", t_now=4000.0) is None


def test_object_memory_rejects_unknown_and_low_conf():
    mem = ObjectMemory()
    mem.add("unknown", 1.0, 1.0, t_ms=1000.0, conf=0.9)
    mem.add("gate", 1.0, 1.0, t_ms=1000.0, conf=0.1)
    assert mem.known_objects() == []
    assert mem.locate("gate", t_now=2000.0) is None


def test_object_memory_expiry():
    mem = ObjectMemory(max_age_ms=5000.0)
    mem.add("gate", 10.0, 10.0, t_ms=1000.0, conf=0.9)
    assert mem.locate("gate", t_now=20000.0) is None  # too old
    assert mem.locate("gate", t_now=3000.0) is not None


def test_object_bias_targets_sighting():
    b = object_bias(40.0, 20.0)
    assert b.shape == (56, 56)
    iy, ix = divmod(b.argmax(), 56)
    assert (ix, iy) == (40, 20)


def test_look_at_grounded():
    mem = ObjectMemory()
    mem.add("windows", 40.0, 20.0, t_ms=1000.0, conf=0.8)
    reply, bias = understand("wodehaus look at the windows",
                             memory=mem, t_now_ms=2000.0)
    assert "windows" in reply
    assert bias is not None and bias.shape == (56, 56)
    iy, ix = divmod(bias.argmax(), 56)
    assert (ix, iy) == (40, 20)


def test_look_at_grounded_plural():
    mem = ObjectMemory()
    mem.add("gate", 10.0, 30.0, t_ms=1000.0, conf=0.9)
    reply, bias = understand("look at the gates", memory=mem,
                             t_now_ms=2000.0)
    assert bias is not None  # 'gates' -> 'gate'
    assert "gate" in reply


def test_look_at_ungrounded_lists_known():
    mem = ObjectMemory()
    mem.add("windows", 40.0, 20.0, t_ms=1000.0, conf=0.8)
    mem.add("gate", 10.0, 10.0, t_ms=2000.0, conf=0.9)
    reply, bias = understand("wodehaus look at the red car",
                             memory=mem, t_now_ms=3000.0)
    assert bias is None
    assert "don't see one" in reply
    assert "windows" in reply and "gate" in reply


def test_look_at_no_memory_honest():
    reply, bias = understand("wodehaus look at the red car")
    assert bias is None
    assert "don't see one" in reply


def test_is_addressed_wootouse_variant():
    from hva.understanding import is_addressed
    assert is_addressed("Hey, WooTouse.")  # observed 2026-09-30


def test_resolve_referent_drops_trailing_narration():
    from hva.understanding import resolve_referent
    mem = ObjectMemory()
    mem.add("windows", 40.0, 20.0, t_ms=1000.0, conf=0.8)
    sighting, matched = resolve_referent("windows the", mem, 2000.0)
    assert sighting is not None and matched == "windows"
    sighting, matched = resolve_referent("red car the", mem, 2000.0)
    assert sighting is None and matched == "red car the"


def test_look_at_with_trailing_narration_grounds():
    mem = ObjectMemory()
    mem.add("windows", 40.0, 20.0, t_ms=1000.0, conf=0.8)
    reply, bias = understand(
        "Contact A Wood House Look at the windows The",
        memory=mem, t_now_ms=2000.0)
    assert "windows" in reply and bias is not None
    iy, ix = divmod(bias.argmax(), 56)
    assert (ix, iy) == (40, 20)


def test_track_associates_nearby_sightings():
    mem = ObjectMemory()
    mem.add("gate", 40.0, 20.0, t_ms=1000.0, conf=0.8)
    mem.add("gate", 42.0, 21.0, t_ms=2000.0, conf=0.6)  # near -> same track
    mem.add("gate", 10.0, 50.0, t_ms=3000.0, conf=0.7)  # far -> new track
    assert len(mem.tracks) == 2
    tr = mem.locate_all("gate", 4000.0)[0]
    assert tr.hits == 2  # the nearby pair wins on hits


def test_track_smooths_position():
    mem = ObjectMemory()
    mem.add("windows", 40.0, 20.0, t_ms=1000.0, conf=0.8)
    mem.add("windows", 44.0, 20.0, t_ms=2000.0, conf=0.8)
    x, y, conf, age = mem.locate("windows", 2500.0)
    assert x == 42.0  # EMA with smooth=0.5, not the raw last sighting
    assert age == 500.0


def test_track_expiry():
    mem = ObjectMemory(max_age_ms=1000.0)
    mem.add("gate", 40.0, 20.0, t_ms=1000.0, conf=0.8)
    assert mem.locate("gate", 1500.0) is not None
    assert mem.locate("gate", 2500.0) is None
    assert mem.locate_all("gate", 2500.0) == []


def test_locate_prefers_hits_over_recency():
    mem = ObjectMemory()
    mem.add("dark", 5.0, 5.0, t_ms=1000.0, conf=0.5)
    mem.add("dark", 6.0, 5.0, t_ms=2000.0, conf=0.5)
    mem.add("dark", 6.0, 6.0, t_ms=3000.0, conf=0.5)
    mem.add("dark", 50.0, 50.0, t_ms=4000.0, conf=0.9)  # newer, 1 hit
    x, y, conf, age = mem.locate("dark", 4500.0)
    assert (x, y) != (50.0, 50.0)  # 3-hit track wins over newer 1-hit


def test_split_spatial_prefix():
    assert split_spatial("leftmost windows") == ("leftmost", "windows")
    assert split_spatial("the nearest gate") == ("nearest", "gate")


def test_split_spatial_suffix():
    assert split_spatial("the windows on the left") == ("left", "windows")
    assert split_spatial("gate at the top") == ("top", "gate")
    assert split_spatial("the gate") == (None, "the gate")


def test_spatial_leftmost_rightmost():
    mem = ObjectMemory()
    mem.add("windows", 10.0, 20.0, t_ms=1000.0, conf=0.8)
    mem.add("windows", 45.0, 20.0, t_ms=2000.0, conf=0.8)
    left, _ = resolve_spatial_referent("windows", "leftmost", mem, 3000.0)
    right, _ = resolve_spatial_referent("windows", "rightmost", mem, 3000.0)
    assert left.x < right.x
    assert left.x == 10.0 and right.x == 45.0


def test_spatial_on_the_left_suffix():
    mem = ObjectMemory()
    mem.add("windows", 10.0, 20.0, t_ms=1000.0, conf=0.8)
    mem.add("windows", 45.0, 20.0, t_ms=2000.0, conf=0.8)
    pick, _ = resolve_spatial_referent("windows", "left", mem, 3000.0)
    assert pick.x == 10.0


def test_spatial_nearest_uses_gaze():
    mem = ObjectMemory()
    mem.add("gate", 5.0, 5.0, t_ms=1000.0, conf=0.8)
    mem.add("gate", 50.0, 50.0, t_ms=2000.0, conf=0.8)
    near, _ = resolve_spatial_referent("gate", "nearest", mem, 3000.0,
                                       gaze_xy=(48.0, 48.0))
    assert near.x == 50.0
    far, _ = resolve_spatial_referent("gate", "farthest", mem, 3000.0,
                                      gaze_xy=(48.0, 48.0))
    assert far.x == 5.0


def test_spatial_miss_stays_honest():
    reply, bias = understand("look at the leftmost red car",
                             memory=ObjectMemory(), t_now_ms=1000.0)
    assert bias is None
    assert "don't see one" in reply


def test_spatial_grounded_reply():
    mem = ObjectMemory()
    mem.add("windows", 45.0, 20.0, t_ms=1000.0, conf=0.8)
    mem.add("windows", 10.0, 20.0, t_ms=2000.0, conf=0.8)
    reply, bias = understand("look at the leftmost windows",
                             memory=mem, t_now_ms=3000.0)
    assert bias is not None
    assert "leftmost windows" in reply
    # bias peaks at the left track
    peak = bias.argmax()
    assert peak % 56 < 28


def _fake_detect(queries):
    # a "gate" box centered at (160, 80)px -> map (40, 20)
    return [("gate", 40.0, 20.0, 0.55)] if "gate" in queries[0] else []


def test_detect_fallback_finds_unseen():
    mem = ObjectMemory()
    reply, bias = understand("wodehaus where is the gate",
                             memory=mem, t_now_ms=1000.0,
                             detect_fn=_fake_detect)
    assert bias is not None
    assert "Found the gate" in reply
    iy, ix = divmod(int(bias.argmax()), 56)
    assert (ix, iy) == (40, 20)
    # the detection becomes a track: follow-up uses memory, not detect
    reply2, bias2 = understand("wodehaus look at the gate",
                               memory=mem, t_now_ms=2000.0,
                               detect_fn=lambda q: [])
    assert "Looking at the gate" in reply2


def test_detect_not_used_for_spatial():
    mem = ObjectMemory()
    calls = []
    reply, bias = understand("wodehaus look at the leftmost gate",
                             memory=mem, t_now_ms=1000.0,
                             detect_fn=lambda q: calls.append(q) or [])
    assert bias is None and not calls
    assert "don't see one" in reply


def test_detect_exception_stays_honest():
    def boom(q):
        raise RuntimeError("nope")
    reply, bias = understand("wodehaus find the gate",
                             memory=ObjectMemory(), t_now_ms=1000.0,
                             detect_fn=boom)
    assert bias is None
    assert "don't see one" in reply


def test_find_and_where_is_intent():
    assert classify("wodehaus find the gate") == Intent.LOOK_AT
    assert classify("wodehaus where is the gate") == Intent.LOOK_AT
    assert extract_referent("wodehaus find the gate") == "gate"
    assert extract_referent("wodehaus where is the gate") == "gate"


def _anaphor_mem():
    from hva.understanding import ObjectMemory
    mem = ObjectMemory()
    mem.add("windows", 10.0, 20.0, 1000.0, 0.8)
    mem.add("windows", 11.0, 21.0, 2000.0, 0.8)
    mem.add("windows", 40.0, 20.0, 1500.0, 0.7)
    mem.add("gate", 28.0, 30.0, 1000.0, 0.9)
    return mem


def test_pronoun_repeats_last_region():
    from hva.understanding import DialogueState, understand
    mem, d = _anaphor_mem(), DialogueState()
    r1, b1 = understand("wodehaus look at the windows", memory=mem,
                        t_now_ms=5000.0, dialogue=d)
    r2, b2 = understand("wodehaus look at it again", memory=mem,
                        t_now_ms=6000.0, dialogue=d)
    assert b2 is not None and int(b1.argmax()) == int(b2.argmax())
    assert "windows" in r2


def test_pronoun_chains_off_spatial_pick():
    from hva.understanding import DialogueState, understand
    mem, d = _anaphor_mem(), DialogueState()
    understand("wodehaus look at the right one", memory=mem,
               t_now_ms=8000.0, dialogue=d)
    r, b = understand("wodehaus look at it", memory=mem,
                      t_now_ms=9000.0, dialogue=d)
    assert b is not None
    assert divmod(int(b.argmax()), 56)[1] == 40
    assert "rightmost windows" in r


def test_pronoun_with_no_referent_is_honest():
    from hva.understanding import DialogueState, understand
    mem, d = _anaphor_mem(), DialogueState()
    r, b = understand("wodehaus look at it", memory=mem,
                      t_now_ms=5000.0, dialogue=d)
    assert b is None and "'it' refers to" in r


def test_the_left_one_picks_among_last_label():
    from hva.understanding import DialogueState, understand
    mem, d = _anaphor_mem(), DialogueState()
    understand("wodehaus look at the windows", memory=mem,
               t_now_ms=5000.0, dialogue=d)
    r, b = understand("wodehaus look at the left one", memory=mem,
                      t_now_ms=6000.0, dialogue=d)
    assert b is not None
    assert divmod(int(b.argmax()), 56)[1] == 10
    assert "left windows" in r


def test_the_one_with_empty_memory_is_honest():
    from hva.understanding import DialogueState, ObjectMemory, understand
    d = DialogueState()
    r, b = understand("wodehaus look at the left one",
                      memory=ObjectMemory(), t_now_ms=5000.0, dialogue=d)
    assert b is None and "don't see one" in r


def test_determiner_that_gate():
    from hva.understanding import DialogueState, understand
    mem, d = _anaphor_mem(), DialogueState()
    r, b = understand("wodehaus look at that gate", memory=mem,
                      t_now_ms=5000.0, dialogue=d)
    assert b is not None and "Looking at the gate" in r
