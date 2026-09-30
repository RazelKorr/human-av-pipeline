"""Conversational turn-taking for the streaming A/V pipeline.

The "saying hi" loop:

  1. TurnDetector watches the rolling transcript for the system's name
     plus end-of-utterance (silence after speech). When both hold, the
     turn is the system's.
  2. ResponsePolicy decides what to say. V3 is grounded via
     hva.understanding (reads the live perceptual loop) and LLM-backed
     via hva.llm (ApiGenerator/LocalGenerator, with rule-based fallback).
     It reports what it heard and saw; it does not pretend to understand
     beyond its intents. The generate() interface is the seam where the
     LLM plugs in.
  3. speak() synthesizes the response via the tts CLI, non-blocking:
     AsyncTTS runs synthesis on a background thread so the perceptual
     tick loop never stalls. If the user barges in during synthesis,
     the pending response is dropped as stale.

Hearing a call is plumbing (audio into the rolling buffer -- hva.stream
already does this). Being heard is a second system: this module plus a
speaker or virtual audio device on the host.

Live duplex: run_conversation.py wires the detector/policy/speak into
the tick loop with EnergyVAD barge-in. If speech starts mid-response,
playback stops and the turn is re-taken.

Design notes:
  - Name matching is fuzzy on purpose. Whisper hears "Wodehaus" as
    "woodhouse", "wodehouse", "vodehaus", etc. We match the consonant
    skeleton, not the spelling.
  - End-of-utterance uses transcript silence, not raw VAD, because the
    transcript is what the policy reasons over. The cost is the
    transcription lag (window/step); the policy never claims to have
    heard words that haven't been transcribed yet.
  - Barge-in is an EnergyVAD gate on the tick loop: speech energy above
    the adaptive floor during playback/synthesis marks the turn stale.
"""
import subprocess
import concurrent.futures
import threading

from hva.understanding import (  # noqa: F401  (re-exported for callers)
    NAME_PATTERNS, _normalize, is_addressed,
    DialogueState, PerceptualState, ObjectMemory, understand)


class Turn:
    """A completed user turn: the transcript text and its time span."""
    def __init__(self, text: str, t_end: float, t_start: float | None = None):
        self.text = text
        self.t_end = t_end
        # t_start is the addressed word's start; None = unknown (t_end).
        self.t_start = t_end if t_start is None else t_start

    def __repr__(self):
        return (f"Turn(t_start={self.t_start:.1f}s, t_end={self.t_end:.1f}s, "
                f"text={self.text!r})")


class TurnDetector:
    """Fires when the system is addressed AND the utterance has ended.

    update(segments, now_s, speech=None) -> list[Turn]. segments are the
    rolling transcript dicts ({start, end, text, words}). now_s is stream
    time. A turn completes when the name was heard and no word has ended
    within silence_s of now.

    Two hallucination/duplicate guards, both evidence-based (no phrase
    blacklists):

    - Acoustic agreement: when `speech` is given -- a callable
      (t0, t1) -> fraction of [t0, t1] with acoustic speech energy --
      the turn's addressed span must clear min_speech_frac. Whisper
      hallucinates prompt-colored speech ("Woodhouse, look left") over
      long digital silence; the transcript then claims speech where
      the acoustic record shows none, and the turn is suppressed.
    - Re-fire suppression: the rolling transcriber re-emits the same
      audio with slightly different word timings across overlapping
      windows, which defeats the word-count watermark and re-fires the
      same utterance. A turn whose addressed span overlaps the previous
      fired turn's span (plus refire_margin_s) is suppressed.

    Suppressed turns are recorded on self.suppressed as (reason, text)
    with reason "no_speech" or "refire".
    """

    def __init__(self, silence_s: float = 1.5,
                 min_speech_frac: float = 0.2,
                 refire_margin_s: float = 1.0):
        self.silence_s = silence_s
        self.min_speech_frac = min_speech_frac
        self.refire_margin_s = refire_margin_s
        self._addressed_at: float | None = None
        self._words_seen = 0  # word count watermark; only new words matter
        self._last_span: tuple[float, float] | None = None
        self.suppressed: list[tuple[str, str]] = []

    def _all_words(self, segments):
        for s in segments:
            for w in s.get("words", []):
                yield w

    def _suppress(self, reason: str, text: str):
        self.suppressed.append((reason, text))
        self._addressed_at = None

    def update(self, segments, now_s: float, speech=None) -> list[Turn]:
        words = list(self._all_words(segments))
        new_words = words[self._words_seen:]
        self._words_seen = len(words)
        if not words:
            return []

        # Did the name appear in the new words?
        new_text = " ".join(w.get("word", "") for w in new_words)
        if new_text and is_addressed(new_text):
            # Address starts at the first new word; the turn text runs
            # from there to the utterance end.
            self._addressed_at = new_words[0]["start"]

        if self._addressed_at is None:
            return []

        last_end = max(w["end"] for w in words)
        if now_s - last_end < self.silence_s:
            return []  # still talking

        # Utterance complete. Collect the addressed span.
        span = [w.get("word", "").strip()
                for w in words if w["start"] >= self._addressed_at - 0.01]
        text = " ".join(span).strip()
        addressed_at, span_end = self._addressed_at, last_end

        # Guard 1: same utterance re-emitted by an overlapping window.
        if (self._last_span is not None
                and addressed_at < self._last_span[1] + self.refire_margin_s):
            self._suppress("refire", text)
            return []

        # Guard 2: the transcript's speech claim must agree with the
        # acoustic record over the addressed span.
        if speech is not None:
            if speech(addressed_at, span_end) < self.min_speech_frac:
                self._suppress("no_speech", text)
                return []

        turn = Turn(text=text, t_end=span_end, t_start=addressed_at)
        self._last_span = (addressed_at, span_end)
        self._addressed_at = None
        return [turn]


class ResponsePolicy:
    """v3: grounded responses via hva.understanding, LLM-backed via hva.llm.

    generate(turn) -> str | None. When a PerceptualState is attached
    (policy.perceptual = PerceptualState(loop)), "what do you see?"
    consults the live map and "look left" steers it -- the bias array
    is stashed on policy.pending_bias for the tick loop to pick up
    with take_bias().

    Attach an LLM backend with policy.llm = ApiGenerator(...). generate()
    then sends build_payload() to the API and parses an optional trailing
    `LOOK: <direction>` line into a task bias. No key or any API error ->
    falls back to the rule-based understand(). The rules are the
    deterministic floor; the API is the ceiling.
    """

    def __init__(self, system_name: str = "Wodehaus"):
        self.system_name = system_name
        self.perceptual: PerceptualState | None = None
        self.dialogue = DialogueState()
        self.memory = ObjectMemory()  # fed by the foveal recognizer
        self.pending_bias = None  # 56x56 array for the tick loop, or None
        self.llm = None  # hva.llm.ApiGenerator or compatible

    def generate(self, turn: Turn) -> str | None:
        if not is_addressed(turn.text):
            return None
        if self.perceptual is not None:
            self.perceptual.memory = self.memory
        if self.llm is not None and self.llm.available:
            try:
                return self._generate_llm(turn)
            except Exception as e:  # API failure -> rule fallback
                print(f"[policy] LLM failed ({e}); using rule fallback",
                      flush=True)
        reply, bias = understand(turn.text,
                                 perceptual=self.perceptual,
                                 dialogue=self.dialogue,
                                 memory=self.memory,
                                 t_now_ms=turn.t_end * 1000.0)
        self.pending_bias = bias
        return reply

    def _generate_llm(self, turn: Turn) -> str:
        from hva.llm import build_payload, split_look_command
        from hva.understanding import direction_bias, look_direction
        payload = build_payload(turn.text,
                                perceptual=self.perceptual,
                                dialogue=self.dialogue)
        raw = self.llm.generate(payload)
        text, direction = split_look_command(raw)
        if direction is None:
            # Deterministic backstop: the rule-based intent classifier is
            # the floor. If the user gave a look command and the model
            # dropped the LOOK line, steer from the classified intent
            # anyway. The model is the ceiling; the rules hold the floor.
            direction = look_direction(turn.text)
        self.pending_bias = (direction_bias(direction)
                             if direction else None)
        # Keep the dialogue state in sync even on the LLM path.
        self.dialogue.add(turn.text, "llm", text, region=direction)
        return text or None

    def take_bias(self):
        """One-shot retrieval for the tick loop; clears after reading."""
        b = self.pending_bias
        self.pending_bias = None
        return b


def speak(text: str, outpath: str,
          voice: str = "avocado_v2:MAI_03") -> str:
    """Synthesize text to an audio file via the tts CLI. Returns outpath."""
    proc = subprocess.run(
        ["/opt/hatch/bin/tts", "speak", "--output", outpath,
         "--voice", voice, "--text-stdin"],
        input=text.encode("utf-8"), capture_output=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tts failed: {proc.stderr.decode()[:500]}")
    return outpath


def audio_duration(path: str) -> float:
    """MP3/WAV duration in seconds via ffprobe."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=30)
    return float(proc.stdout.strip())


class AsyncTTS:
    """Non-blocking TTS synthesis: keep the perceptual tick loop running.

    speak() blocks the calling thread for seconds while the tts CLI runs.
    In the live loop that stalls perception -- the 10 Hz tick stops while
    we synthesize. AsyncTTS moves synthesis to a background thread so the
    tick loop keeps stepping OnlineLevel3.

    Usage (tick thread only):
      tts = AsyncTTS()
      tts.submit("hello there", "/tmp/reply.mp3")  # non-blocking
      ...
      for outpath, text in tts.poll_ready():       # call each tick
          speaker.play(outpath, now_s)

    The worker thread only runs the subprocess; Speaker.play() stays on
    the tick thread (it just sets timers). poll_ready() re-raises
    synthesis failures as log lines, not exceptions, so one bad reply
    never kills the loop.
    """

    def __init__(self, max_workers: int = 1, speak_fn=None):
        self._ex = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)
        self._speak_fn = speak_fn or speak
        self._lock = threading.Lock()
        self._pending: dict = {}  # future -> (outpath, text)

    def submit(self, text: str, outpath: str, voice: str = "avocado_v2:MAI_03"):
        """Queue synthesis. Returns the Future (rarely needed)."""
        fut = self._ex.submit(self._speak_fn, text, outpath, voice)
        with self._lock:
            self._pending[fut] = (outpath, text)
        return fut

    def poll_ready(self):
        """Collect finished syntheses. Call from the tick thread."""
        with self._lock:
            items = list(self._pending.items())
        ready: list[tuple[str, str]] = []
        done: list = []
        for fut, (outpath, text) in items:
            if fut.done():
                done.append(fut)
                try:
                    fut.result()
                    ready.append((outpath, text))
                except Exception as e:  # one bad reply never kills the loop
                    print(f"[async-tts] synthesis failed: {e}", flush=True)
        with self._lock:
            for fut in done:
                self._pending.pop(fut, None)
        return ready

    def has_pending(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def cancel_pending(self):
        """Drop queued (not yet running) syntheses. Returns count dropped."""
        dropped = 0
        with self._lock:
            for fut in list(self._pending):
                if fut.cancel():
                    self._pending.pop(fut, None)
                    dropped += 1
        return dropped

    def shutdown(self):
        self._ex.shutdown(wait=False)


class Speaker:
    """Response playback state.

    On this VM there is no audio device, so 'playing' is simulated by
    tracking stream time against the file's duration. On a real host,
    pass a play_fn that starts real playback and a stop_fn that halts
    it; is_playing() then reflects the device. The barge-in logic only
    needs is_playing()/stop(), so the simulation and the real device
    are interchangeable.
    """

    def __init__(self, play_fn=None, stop_fn=None):
        self.play_fn = play_fn
        self.stop_fn = stop_fn
        self.playing_until: float | None = None
        self.current_file: str | None = None
        self.interrupted = False  # True if the last playback was cut short

    def play(self, path: str, now_s: float):
        dur = audio_duration(path)
        self.current_file = path
        self.playing_until = now_s + dur
        self.interrupted = False
        if self.play_fn is not None:
            self.play_fn(path)

    def is_playing(self, now_s: float) -> bool:
        return (self.playing_until is not None
                and now_s < self.playing_until)

    def stop(self):
        """Barge-in: halt playback now."""
        if self.stop_fn is not None:
            self.stop_fn()
        self.playing_until = None
        self.interrupted = True
        self.current_file = None

    def check_finished(self, now_s: float) -> bool:
        """True if playback completed naturally (not interrupted)."""
        if self.playing_until is not None and now_s >= self.playing_until:
            self.playing_until = None
            self.current_file = None
            return True
        return False


class EnergyVAD:
    """Fast speech-onset detector for barge-in.

    Per-tick RMS energy against an adaptive noise floor. Fires when
    energy exceeds the floor by `ratio` for `hangover` consecutive
    ticks (default 3 = 300 ms). This is the fast path -- the transcript
    lags 10-30 s, but barge-in needs to react in under half a second.

    Assumes the system's own playback does not leak into the mic
    (headphones / virtual routing). Echo cancellation is out of scope.
    """

    def __init__(self, ratio: float = 4.0, hangover: int = 3,
                 floor_alpha: float = 0.05, abs_floor: float = 1e-4):
        self.ratio = ratio
        self.hangover = hangover
        self.floor_alpha = floor_alpha
        self.abs_floor = abs_floor
        self.noise_floor = abs_floor
        self.hot_ticks = 0
        self.hot = False  # per-tick speech presence (no hangover)

    def _level(self, mono_100ms) -> bool:
        """True when this tick's energy clears the adaptive floor."""
        import numpy as np
        rms = float(np.sqrt(np.mean(np.asarray(mono_100ms) ** 2)) + 1e-12)
        if rms < self.noise_floor * self.ratio:
            # Quiet: adapt the floor toward it.
            self.noise_floor = ((1 - self.floor_alpha) * self.noise_floor
                                + self.floor_alpha * max(rms, self.abs_floor))
            return False
        return True

    def update(self, mono_100ms) -> bool:
        """Feed one 100 ms mono chunk. Returns True on speech onset."""
        self.hot = self._level(mono_100ms)
        if not self.hot:
            self.hot_ticks = 0
            return False
        self.hot_ticks += 1
        if self.hot_ticks >= self.hangover:
            self.hot_ticks = 0  # fire once per onset
            return True
        return False
