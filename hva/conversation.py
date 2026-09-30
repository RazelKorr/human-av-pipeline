"""Conversational turn-taking for the streaming A/V pipeline.

The "saying hi" loop:

  1. TurnDetector watches the rolling transcript for the system's name
     plus end-of-utterance (silence after speech). When both hold, the
     turn is the system's.
  2. ResponsePolicy decides what to say. V1 is template-based and honest
     about its limits -- it reports what it heard, it does not pretend
     to understand. The generate() interface is the seam where an LLM
     plugs in later.
  3. speak() synthesizes the response via the tts CLI.

Hearing a call is plumbing (audio into the rolling buffer -- hva.stream
already does this). Being heard is a second system: this module plus a
speaker or virtual audio device on the host.

Design notes:
  - Name matching is fuzzy on purpose. Whisper hears "Wodehaus" as
    "woodhouse", "wodehouse", "vodehaus", etc. We match the consonant
    skeleton, not the spelling.
  - End-of-utterance uses transcript silence, not raw VAD, because the
    transcript is what the policy reasons over. The cost is the
    transcription lag (window/step); the policy never claims to have
    heard words that haven't been transcribed yet.
  - V1 does not barge in. It waits for the utterance to end. Interruption
    is a later policy decision, not a missing feature.
"""
import re
import subprocess

# Consonant skeletons the name is heard as. "Wodehaus" -> W-D-H-S.
# Whisper variants observed or expected: woodhouse, wodehouse, vodehaus,
# wodehaus, wood house, what house (mishear). We keep this list short and
# explicit -- a fuzzy matcher that matches everything matches nothing.
NAME_PATTERNS = [
    r"wodehaus",
    r"woodhouse",
    r"wodehouse",
    r"vodehaus",
    r"vodahaus",
    r"wood\s+house",
    r"would\s+house",   # Whisper heard "Hi Wodehaus" as "I would house"
    r"what\s+house",
]

GREETING_WORDS = {
    "hi", "hello", "hey", "greetings", "good morning", "good afternoon",
    "good evening", "howdy",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z\s]", "", text.lower())


def is_addressed(text: str) -> bool:
    """True if the system's name appears in the text (fuzzy)."""
    norm = _normalize(text)
    return any(re.search(p, norm) for p in NAME_PATTERNS)


class Turn:
    """A completed user turn: the transcript text and when it ended."""
    def __init__(self, text: str, t_end: float):
        self.text = text
        self.t_end = t_end

    def __repr__(self):
        return f"Turn(t_end={self.t_end:.1f}s, text={self.text!r})"


class TurnDetector:
    """Fires when the system is addressed AND the utterance has ended.

    update(segments, now_s) -> list[Turn]. segments are the rolling
    transcript dicts ({start, end, text, words}). now_s is stream time.
    A turn completes when the name was heard and no word has ended
    within silence_s of now.
    """

    def __init__(self, silence_s: float = 1.5):
        self.silence_s = silence_s
        self._addressed_at: float | None = None
        self._words_seen = 0  # word count watermark; only new words matter

    def _all_words(self, segments):
        for s in segments:
            for w in s.get("words", []):
                yield w

    def update(self, segments, now_s: float) -> list[Turn]:
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
        turn = Turn(text=text, t_end=last_end)
        self._addressed_at = None
        return [turn]


class ResponsePolicy:
    """V1: template responses, honest about perceptual limits.

    generate(turn) -> str | None. None means stay silent (not addressed
    to us -- the detector should already filter, this is the backstop).
    The method signature is the LLM seam: a future policy takes the turn
    plus perceptual context (gaze target, scene description) and returns
    text. V1 reports what it heard; it does not confabulate understanding.
    """

    def __init__(self, system_name: str = "Wodehaus"):
        self.system_name = system_name

    def generate(self, turn: Turn) -> str | None:
        if not is_addressed(turn.text):
            return None
        norm = _normalize(turn.text)
        if any(g in norm for g in GREETING_WORDS):
            return (f"Hello! This is {self.system_name}. "
                    f"I heard you say: {turn.text.strip()}")
        if "?" in turn.text:
            return (f"I heard your question: {turn.text.strip()} "
                    f"I can hear the words, but I don't understand "
                    f"them yet -- that's the next thing to build.")
        return (f"Yes, I'm here. I heard: {turn.text.strip()}")


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
