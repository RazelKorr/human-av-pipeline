"""The LLM seam: ResponsePolicy.generate() backed by a real API.

Architecture:
  turn text + PerceptualState + DialogueState
      -> build_payload()            (the grounding; this is what makes the
                                     LM's words *about* something real)
      -> ApiGenerator.generate()    (Anthropic Messages API via urllib)
      -> reply text (+ optional task_bias parsed from the payload reply)

No key / API error -> falls back to the rule-based understand(). The
rules are the deterministic floor; the API is the ceiling. Nothing here
claims the repository independently contains Wodehaus -- it calls out to
a language model the operator provides.

Key source: the ANTHROPIC_API_KEY environment variable. Never in chat,
never in the repo.
"""
import json
import os
import urllib.request

SYSTEM_PROMPT = """\
You are the voice of Wodehaus, a perceptual system that watches and \
listens through a joint audio-visual priority map. You are given the \
speaker's words plus a snapshot of your own perceptual state: where \
your gaze is, what region is most salient, recent saccades, and the \
dialogue history.

Rules:
- Answer from the perceptual state you are given. If it says you are \
looking at the upper-left, say so; do not invent objects or people.
- You do not know what things look like yet (no object recognition). \
If asked about a named object, say so honestly.
- You are not human. Do not claim human perception, consciousness, \
or complete hearing/vision.
- Keep replies to one or two short sentences, conversational.
- If the perceptual state is absent or stale, say what you heard, \
not what you saw.

You may also steer perception: if the user tells you to look \
somewhere, include a line `LOOK: <left|right|up|down|center>` at the \
end of your reply. That line is stripped before speaking and converted \
to a bias on the shared priority map. Omit it when no look is requested."""


def build_payload(turn_text, perceptual=None, dialogue=None) -> dict:
    """Structured context: the grounding the LM reads."""
    state = perceptual.describe() if perceptual is not None else None
    history = (dialogue.history_text(n=6) if dialogue is not None else "")
    return {
        "turn": turn_text,
        "perceptual_state": state,
        "dialogue_history": history,
        "system": "Wodehaus",
    }


def split_look_command(reply: str):
    """Pull a trailing `LOOK: <direction>` line out of an LLM reply."""
    lines = reply.rstrip().splitlines()
    if lines and lines[-1].strip().upper().startswith("LOOK:"):
        direction = lines[-1].split(":", 1)[1].strip().lower()
        if direction in ("left", "right", "up", "down", "center"):
            return "\n".join(lines[:-1]).rstrip(), direction
    return reply, None


class LocalGenerator:
    """llama-server backend (OpenAI-compatible endpoint).

    Run: llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M --port 8080
    Then: policy.llm = LocalGenerator()
    The same payload goes out; the LOOK-line convention is parsed by
    the caller in ResponsePolicy, so local and API backends share it.
    """

    def __init__(self, base_url: str = "http://localhost:8080",
                 model: str = "local", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        try:
            req = urllib.request.Request(self.base_url + "/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, payload: dict) -> str:
        body = json.dumps({
            "model": self.model,
            "max_tokens": 200,
            "temperature": 0.6,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": ("Perceptual snapshot and dialogue:\n"
                             + json.dumps(payload, indent=1)
                             + "\n\nRespond to the turn.")},
            ],
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=body, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()


class ApiGenerator:
    """Anthropic Messages API backend for the generate() seam."""

    def __init__(self, api_key: str | None = None,
                 model: str = "claude-haiku-4-5-20251001",
                 timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, payload: dict) -> str:
        """Returns reply text (with any LOOK line still attached)."""
        if not self.available:
            raise RuntimeError("no ANTHROPIC_API_KEY")
        body = json.dumps({
            "model": self.model,
            "max_tokens": 200,
            "system": SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": (
                    "Perceptual snapshot and dialogue:\n"
                    + json.dumps(payload, indent=1)
                    + "\n\nRespond to the turn."
                ),
            }],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return "".join(
            b.get("text", "") for b in data.get("content", [])
            if b.get("type") == "text"
        ).strip()
