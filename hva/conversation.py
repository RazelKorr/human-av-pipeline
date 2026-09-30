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
import os
import select
import shlex
import shutil
import subprocess
import concurrent.futures
import threading
import time

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
        self.detector = None  # hvp.detect.ObjectDetector or compatible
        self.frame_fn = None  # () -> PIL 224x224 RGB frame or None

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
        gaze = None
        if self.perceptual is not None:
            gx, gy = self.perceptual.gaze_now()
            gaze = (gx / 4.0, gy / 4.0)
        detect_fn = None
        if self.detector is not None and self.frame_fn is not None:
            def detect_fn(queries):
                frame = self.frame_fn()
                if frame is None:
                    return []
                out = []
                # OWL-ViT wants noun phrases: "window" -> "a window".
                dets = self.detector.detect(
                    frame, [f"a {q}" for q in queries])
                for det in dets:
                    mx, my = self.detector.box_center_map(det)
                    out.append((det[0], mx, my, det[5]))
                return out
        reply, bias = understand(turn.text,
                                 perceptual=self.perceptual,
                                 dialogue=self.dialogue,
                                 memory=self.memory,
                                 t_now_ms=turn.t_end * 1000.0,
                                 gaze_xy=gaze,
                                 detect_fn=detect_fn)
        self.pending_bias = bias
        return reply

    def _generate_llm(self, turn: Turn) -> str:
        from hva.llm import build_payload, split_look_command
        from hva.understanding import direction_bias, look_direction
        payload = build_payload(turn.text,
                                perceptual=self.perceptual,
                                dialogue=self.dialogue,
                                memory=self.memory,
                                t_now_ms=turn.t_end * 1000.0)
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


TTS_BIN = "/opt/hatch/bin/tts"


def speak(text: str, outpath: str,
          voice: str = "avocado_v2:MAI_03") -> str:
    """Synthesize text to an audio file via the tts CLI. Returns outpath."""
    proc = subprocess.run(
        [TTS_BIN, "speak", "--output", outpath,
         "--voice", voice, "--text-stdin"],
        input=text.encode("utf-8"), capture_output=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tts failed: {proc.stderr.decode()[:500]}")
    return outpath


class StreamPlayback:
    """Handle for one tts --stream -> player pipeline.

    The pump thread reads MP3 bytes from the tts process stdout and
    writes each chunk to both the player stdin and the record file, so
    playback starts as soon as the first bytes are synthesized (no
    waiting for the full file) while output/conversation/reply_NN.mp3
    is still written for the audit record.
    """

    def __init__(self, tts_proc, player_proc, pump_thread, record_file,
                 outpath, stop_event):
        self.tts_proc = tts_proc
        self.player_proc = player_proc
        self.pump_thread = pump_thread
        self._record = record_file
        self.outpath = outpath
        self._stop_event = stop_event
        self._stopped = False

    def poll(self):
        """None while the player is alive, else its returncode."""
        return self.player_proc.poll()

    def wait(self, timeout=None):
        return self.player_proc.wait(timeout=timeout)

    def stop(self):
        """Pipe-kill: stop the player first, then the synthesizer.

        Idempotent. Signals the pump thread first (it wakes from its
        select within the timeout, so it never blocks on the tts
        pipe), terminates (then kills if needed) both processes, then
        closes the record file and reaps the children so no zombies
        linger.

        Note: the tts CLI can fork daemon children that inherit its
        stdout and outlive it, so the pump deliberately never waits
        on EOF from the tts pipe -- the stop event is what ends it.
        """
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        for p in (self.player_proc, self.tts_proc):
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        for p in (self.player_proc, self.tts_proc):
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                    p.wait(timeout=5)  # reap after kill; no zombie until GC
                except Exception:
                    pass
        # The pump wakes from select within its timeout once the event
        # is set, so this join is normally instant; the timeout is only
        # a backstop. The record file is closed after the pump is dead,
        # so it can never write to a closed file.
        self.pump_thread.join(timeout=5)
        try:
            self._record.close()
        except Exception:
            pass
        for stream in (self.tts_proc.stdout, self.tts_proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        for p in (self.tts_proc, self.player_proc):
            try:
                p.wait(timeout=0)
            except Exception:
                pass


def speak_stream(text: str, player_cmd, outpath: str,
                 voice: str = "avocado_v2:MAI_03",
                 first_byte_timeout: float = 15.0) -> StreamPlayback:
    """Synthesize text straight into an audio player via tts --stream.

    Spawns `tts speak --stream` (MP3 bytes on stdout) piped into
    `player_cmd` (a argv list like ["ffplay", "-nodisp", "-autoexit",
    "-"]). A pump thread tees each chunk to the player stdin and to
    `outpath`, preserving the record file. Returns a StreamPlayback
    handle immediately once the first bytes flow (or the tts process
    exits); playback start latency is first-byte latency, not
    full-synthesis latency.

    Raises RuntimeError like speak() if the tts process exits nonzero
    before producing any bytes.
    """
    tts_proc = subprocess.Popen(
        [TTS_BIN, "speak", "--stream", "--voice", voice, "--text-stdin"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    try:
        tts_proc.stdin.write(text.encode("utf-8"))
        tts_proc.stdin.close()
    except BrokenPipeError:
        pass  # tts died instantly; the handshake below reports it
    player_proc = subprocess.Popen(
        list(player_cmd), stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    record = open(outpath, "wb")
    state = {"bytes": 0}
    stop_event = threading.Event()

    def pump():
        # select + os.read, never a blocking full-size read: the tts
        # CLI can fork daemon children that inherit its stdout and
        # keep the pipe open after tts itself exits, so waiting on EOF
        # -- or on BufferedReader.read(n) filling n bytes -- can block
        # indefinitely. The stop event bounds every wait instead, and
        # no exception ever escapes the thread.
        fd = tts_proc.stdout.fileno()
        try:
            while not stop_event.is_set():
                try:
                    r, _, _ = select.select([fd], [], [], 0.5)
                except (OSError, ValueError):
                    break  # fd closed under us; nothing left to do
                if stop_event.is_set():
                    break
                if not r:
                    continue
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    break
                if not chunk:
                    break  # EOF: tts finished naturally
                state["bytes"] += len(chunk)
                try:
                    record.write(chunk)
                except ValueError:
                    break  # record closed after stop(); done
                try:
                    player_proc.stdin.write(chunk)
                except BrokenPipeError:
                    break  # player died; finally-block cleans up
        except Exception:
            pass
        finally:
            try:
                player_proc.stdin.close()  # EOF -> player drains and exits
            except Exception:
                pass
            try:
                record.flush()
            except Exception:
                pass

    pump_thread = threading.Thread(target=pump, daemon=True)
    pump_thread.start()
    handle = StreamPlayback(tts_proc, player_proc, pump_thread, record,
                            outpath, stop_event)

    # Failure handshake: wait for first bytes or tts exit, bounded.
    # Healthy tts produces first bytes in well under a second; this
    # keeps the speak() contract (fail fast, raise RuntimeError)
    # without blocking for the full synthesis.
    deadline = time.time() + first_byte_timeout
    while (state["bytes"] == 0 and tts_proc.poll() is None
           and time.time() < deadline):
        time.sleep(0.05)
    rc = tts_proc.poll()
    if state["bytes"] == 0 and rc is not None and rc != 0:
        # Bounded stderr drain: the tts CLI can fork daemon children
        # that inherit its stderr and outlive it, so an unbounded
        # read() here could block for a minute on a pipe that will
        # never see EOF -- stalling the tick loop on a failure path.
        # One short bounded wait is plenty for an error message.
        err = ""
        try:
            r, _, _ = select.select([tts_proc.stderr], [], [], 2.0)
            if r:
                err = os.read(tts_proc.stderr.fileno(), 65536).decode(
                    errors="replace")[:500]
        except Exception:
            pass
        handle.stop()
        raise RuntimeError(f"tts --stream failed: {err}")
    return handle


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


def _resolve_player_cmd(player_cmd):
    """Resolve the streaming audio player command, or None.

    Accepts an argv list or a shell-like string (shlex.split), falling
    back to the HVA_AUDIO_PLAYER environment variable. Returns None
    unless the binary resolves via shutil.which -- so a configured-but-
    missing player degrades to simulated playback instead of crashing.
    """
    if player_cmd is None:
        player_cmd = os.environ.get("HVA_AUDIO_PLAYER")
    if not player_cmd:
        return None
    parts = (shlex.split(player_cmd) if isinstance(player_cmd, str)
             else list(player_cmd))
    if not parts or shutil.which(parts[0]) is None:
        return None
    return parts


class Speaker:
    """Response playback state.

    On this VM there is no audio device, so 'playing' is simulated by
    tracking stream time against the file's duration. On a real host,
    pass a play_fn that starts real playback and a stop_fn that halts
    it; is_playing() then reflects the device. The barge-in logic only
    needs is_playing()/stop(), so the simulation and the real device
    are interchangeable.

    Streaming mode: pass player_cmd (or set HVA_AUDIO_PLAYER, e.g.
    "ffplay -nodisp -autoexit -") and playback goes through
    speak_stream -- tts --stream piped straight into the player, so
    playback starts at first-byte latency and stop() is a pipe-kill.
    The record file (output/conversation/reply_NN.mp3) is still
    written. Without a resolvable player binary, everything behaves
    exactly as before (simulated).
    """

    def __init__(self, play_fn=None, stop_fn=None, player_cmd=None):
        self.play_fn = play_fn
        self.stop_fn = stop_fn
        self.player_cmd = _resolve_player_cmd(player_cmd)
        # Explicit play_fn/stop_fn win over the streaming player.
        self.streaming = (self.player_cmd is not None
                          and play_fn is None and stop_fn is None)
        self.playing_until: float | None = None
        self.current_file: str | None = None
        self.interrupted = False  # True if the last playback was cut short
        self._stream: StreamPlayback | None = None

    def play(self, path: str, now_s: float):
        dur = audio_duration(path)
        self.current_file = path
        self.playing_until = now_s + dur
        self.interrupted = False
        if self.play_fn is not None:
            self.play_fn(path)

    def play_stream(self, text: str, outpath: str, now_s: float,
                    voice: str = "avocado_v2:MAI_03"):
        """Start streaming playback of text. No-op unless streaming.

        Spawns the tts --stream -> player pipeline (non-blocking) and
        returns the StreamPlayback handle. Raises RuntimeError if tts
        fails before producing bytes, like speak().
        """
        if not self.streaming:
            return None
        self.stop()  # never overlap two playbacks
        self.interrupted = False
        self._stream = speak_stream(text, self.player_cmd, outpath,
                                    voice=voice)
        self.current_file = outpath
        return self._stream

    def is_playing(self, now_s: float) -> bool:
        if self._stream is not None:
            return self._stream.poll() is None
        return (self.playing_until is not None
                and now_s < self.playing_until)

    def stop(self):
        """Barge-in: halt playback now (pipe-kill in streaming mode)."""
        if self._stream is not None:
            self._stream.stop()
            self._stream = None
        if self.stop_fn is not None:
            self.stop_fn()
        self.playing_until = None
        self.interrupted = True
        self.current_file = None

    def check_finished(self, now_s: float) -> bool:
        """True if playback completed naturally (not interrupted)."""
        if self._stream is not None:
            if self._stream.poll() is not None:
                # Natural finish: run the same deterministic shutdown
                # as stop() (join pump, close record file, reap tts) so
                # no fd or zombie waits on garbage collection.
                self._stream.stop()
                self._stream = None
                self.current_file = None
                return True
            return False
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
