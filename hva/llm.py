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
import urllib.parse
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


class HFGenerator:
    """HuggingFace Inference Providers backend (OpenAI-compatible).

    Free tier: create an account at huggingface.co, generate a token
    at huggingface.co/settings/tokens with "Make calls to Inference
    Providers" permission, and export HF_TOKEN. No key / API error ->
    falls back to the rule-based understand(), same as the other
    backends. The rules are the floor; the hosted model is the ceiling.

    Default model is Qwen/Qwen3-8B -- the same family docs/local-llm.md
    recommends for the eventual local install, so hosted results
    transfer to the local weights later.
    """

    ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

    def __init__(self, api_token: str | None = None,
                 model: str = "Qwen/Qwen3-8B",
                 timeout: float = 60.0):
        self.api_token = api_token or os.environ.get("HF_TOKEN")
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_token)

    def generate(self, payload: dict) -> str:
        """Returns reply text (with any LOOK line still attached)."""
        if not self.available:
            raise RuntimeError("no HF_TOKEN")
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
            self.ROUTER_URL,
            data=body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_token}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"].strip()


class PollinationsGenerator:
    """Pollinations.ai classic text endpoint -- no account, no key.

    GET https://text.pollinations.ai/{prompt}?model=...&system=...
    Keyless and free, rate-limited by IP. available is always True:
    there is no credential to check, and any network/API failure
    falls back to the rule-based understand() via ResponsePolicy,
    same as the other backends.

    Trade-offs, stated plainly: a third party (pollinations.ai) sees
    the prompts, which include perceptual snapshots and dialogue.
    Fine for prototyping; not for anything sensitive. No SLA, no
    guaranteed model behind the label. The rules remain the floor.
    """

    BASE_URL = "https://text.pollinations.ai"

    def __init__(self, model: str = "openai", timeout: float = 90.0):
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return True  # keyless: nothing to validate without a request

    def generate(self, payload: dict) -> str:
        """Returns reply text (with any LOOK line still attached)."""
        user_text = ("Perceptual snapshot and dialogue:\n"
                     + json.dumps(payload, indent=1)
                     + "\n\nRespond to the turn.")
        query = urllib.parse.urlencode(
            {"model": self.model, "system": SYSTEM_PROMPT})
        url = (f"{self.BASE_URL}/{urllib.parse.quote(user_text)}?{query}")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode().strip()


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


def select_llm_backend(choice: str = "none",
                       llm_url: str = "http://localhost:8080",
                       api_model: str = "claude-haiku-4-5-20251001",
                       hf_model: str = "Qwen/Qwen3-8B",
                       free_model: str = "openai"):
    """Pick an LLM backend for ResponsePolicy.

    Returns (backend_or_None, description). Rule-based understand() is
    always the fallback; the backend is the ceiling. Default is "none"
    so a stray ANTHROPIC_API_KEY or HF_TOKEN never spends money or
    quota without an explicit flag.

    choice: "none" | "api" | "local" | "hf" | "free" | "auto"
      free = Pollinations.ai classic endpoint: no account, no key.
      auto tries local llama-server first, then HuggingFace (free tier)
      if tokened, then API if keyed, then the keyless free backend,
      else none. Free before paid; your own credentials before a
      public gateway.
    """
    if choice == "none":
        return None, "none (rule-based)"
    if choice == "local":
        gen = LocalGenerator(base_url=llm_url)
        return (gen if gen.available else None,
                f"local @ {llm_url} "
                f"({'reachable' if gen.available else 'unreachable, fallback'})")
    if choice == "hf":
        gen = HFGenerator(model=hf_model)
        return (gen if gen.available else None,
                f"hf {hf_model} "
                f"({'tokened' if gen.available else 'no token, fallback'})")
    if choice == "free":
        gen = PollinationsGenerator(model=free_model)
        return gen, f"free (pollinations.ai {free_model}, keyless)"
    if choice == "api":
        gen = ApiGenerator(model=api_model)
        return (gen if gen.available else None,
                f"api {api_model} "
                f"({'keyed' if gen.available else 'no key, fallback'})")
    # auto: local if reachable, else hf if tokened, else api if keyed,
    # else the keyless free backend, else none.
    local = LocalGenerator(base_url=llm_url)
    if local.available:
        return local, f"auto -> local @ {llm_url}"
    hf = HFGenerator(model=hf_model)
    if hf.available:
        return hf, f"auto -> hf {hf_model}"
    api = ApiGenerator(model=api_model)
    if api.available:
        return api, f"auto -> api {api_model}"
    free = PollinationsGenerator(model=free_model)
    if free.available:
        return free, f"auto -> free (pollinations.ai {free_model}, keyless)"
    return None, "auto -> none (no local server, no HF token, no API key)"
