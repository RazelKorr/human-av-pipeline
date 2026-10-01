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
import http.client
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
- Reporting where your gaze rests is not the same as seeing something \
there. If the snapshot names no salient point, the view looks blank -- \
say so instead of describing gaze alone.
- The recognized-objects list names things your foveal classifier has \
actually seen, with rough regions and confidences. You may refer to \
those objects. If asked about a named object that is NOT on the list, \
say honestly that you have not recognized it -- do not invent it.
- You are not human. Do not claim human perception, consciousness, \
or complete hearing/vision.
- Keep replies to one or two short sentences, conversational.
- If the perceptual state is absent or stale, say what you heard, \
not what you saw.

You may also steer perception: if the user tells you to look \
somewhere, include a line `LOOK: <left|right|up|down|center>` at the \
end of your reply. That line is stripped before speaking and converted \
to a bias on the shared priority map. Omit it when no look is requested."""


def build_payload(turn_text, perceptual=None, dialogue=None,
                  memory=None, t_now_ms=0.0) -> dict:
    """Structured context: the grounding the LM reads.

    perceptual.describe() gives gaze/salience; memory contributes the
    recognized objects with rough regions, so the model's words are
    about things the system actually saw. Without memory the model is
    told plainly that it has no object recognition.
    """
    from hva.understanding import _qualitative
    state = perceptual.describe() if perceptual is not None else None
    history = (dialogue.history_text(n=6) if dialogue is not None else "")
    objects = []
    if memory is not None:
        for label in memory.known_objects():
            sighting = memory.locate(label, t_now_ms)
            if sighting is None:
                continue
            mx, my, conf, _ = sighting
            objects.append(
                f"{label} ({_qualitative(mx * 4.0, my * 4.0)}, "
                f"confidence {conf:.2f})")
    return {
        "turn": turn_text,
        "perceptual_state": state,
        "recognized_objects": objects or None,
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


class AgentGenerator:
    """File-handoff backend: an operator (human or AI agent) in the loop.

    generate(payload) writes handoff/turn-NNNN.prompt.json and blocks
    until handoff/turn-NNNN.response.txt appears, then returns its
    text. The operator reads the prompt file, writes the reply file,
    and the conversation continues. The wait is bounded (default 15
    min); on timeout it raises and ResponsePolicy falls back to the
    rule-based understand() -- the rules are the floor, same as the
    other backends.

    Additive and opt-in: select with --llm agent. Nothing about the
    other backends changes. Handoff dir overridable via HANDOFF_DIR.
    """

    def __init__(self, handoff_dir: str | None = None,
                 poll_s: float = 2.0, timeout_s: float = 900.0):
        self.handoff_dir = handoff_dir or os.environ.get(
            "HANDOFF_DIR", "handoff")
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self._n = 0
        os.makedirs(self.handoff_dir, exist_ok=True)

    @property
    def available(self) -> bool:
        return True  # the operator is the credential

    def _paths(self, n: int):
        base = os.path.join(self.handoff_dir, f"turn-{n:04d}")
        return base + ".prompt.json", base + ".response.txt"

    def generate(self, payload: dict) -> str:
        """Returns reply text (with any LOOK line still attached)."""
        import time
        self._n += 1
        prompt_path, response_path = self._paths(self._n)
        with open(prompt_path, "w") as f:
            json.dump({"system": SYSTEM_PROMPT,
                       "payload": payload,
                       "instruction": "Respond to the turn."},
                      f, indent=1)
        print(f"[agent-llm] prompt -> {prompt_path}; "
              f"waiting for {response_path}", flush=True)
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            if os.path.exists(response_path):
                with open(response_path) as f:
                    text = f.read().strip()
                print(f"[agent-llm] response <- {response_path}",
                      flush=True)
                return text
            time.sleep(self.poll_s)
        raise TimeoutError(
            f"no agent response within {self.timeout_s:.0f}s "
            f"(wrote {prompt_path})")


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
    """Pollinations.ai OpenAI-compatible endpoint -- no account, no key.

    POST https://text.pollinations.ai/openai with a JSON body, parsed as
    an OpenAI chat completion. Keyless and free, rate-limited by IP.
    available is always True: there is no credential to check, and any
    network/API failure falls back to the rule-based understand() via
    ResponsePolicy, same as the other backends.

    Why POST and not the classic GET /text/{prompt}: the payload carries
    the perceptual snapshot and dialogue history, and those do not belong
    in a URL path or query string (proxy/server request logs). POST puts
    them in the body.

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
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
        }).encode()
        req = urllib.request.Request(
            f"{self.BASE_URL}/openai", data=body,
            headers={"Content-Type": "application/json"})
        # One retry on connection-level drops only (RemoteDisconnected etc.).
        # HTTP errors -- including 429/503 -- propagate immediately so the
        # service's backoff signals are respected, not fought.
        last_exc = None
        for _ in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()
            except (http.client.RemoteDisconnected, ConnectionResetError,
                    TimeoutError) as exc:
                last_exc = exc
        raise last_exc


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

    choice: "none" | "api" | "local" | "hf" | "free" | "agent" | "auto"
      free = Pollinations.ai OpenAI-compatible POST endpoint
             (keyless JSON body): no account, no key.
      agent = file-handoff backend (handoff/turn-NNNN.prompt.json ->
              turn-NNNN.response.txt): an operator answers each turn.
      auto tries local llama-server first, then HuggingFace (free tier)
      if tokened, then API if keyed, then the keyless free backend,
      else none. Free before paid; your own credentials before a
      public gateway.
    """
    if choice == "none":
        return None, "none (rule-based)"
    if choice == "agent":
        gen = AgentGenerator()
        return gen, f"agent (file handoff @ {gen.handoff_dir}/)"
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
