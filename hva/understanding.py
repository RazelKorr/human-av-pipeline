"""Comprehension for the streaming A/V pipeline -- as far as it goes
without stapling on a full language model.

The thesis: for THIS system, comprehension means grounding words in
its own perceptual state. It has eyes (a joint priority map, a saccade
controller, a scanpath), ears (transient profile, speech transcript),
and a clock. Understanding "what do you see?" means consulting the
map, not generating text about seeing. Understanding "look left"
means writing a bias into the map and letting the saccade machinery
do the rest.

Components:

  Intent -- rule-based classification of a turn's transcript.
  PerceptualState -- read-only view over an OnlineLevel3: where gaze
      is, where it has been, what is salient now.
  LanguageToPerception -- turns spatial commands into bias blobs on
      the 56x56 map ("look left" -> Gaussian at map-left).
  DialogueState -- turn history, so "what about that?" has context.
  understand() -- the entry point: (turn, perceptual_state, dialogue)
      -> (intent, entities, response_text, task_bias|None).

Honest limits, stated plainly:
  - No object recognition. "Look at the red car" gets "I don't know
    what a red car looks like yet" -- the bias channel needs a target
    the system can locate, and right now that's directions, not things.
  - No deep semantics. The intents are patterns, not understanding.
    The module reports what it did ("I looked left because you said
    left"), it does not pretend to grasp meaning.
  - Pronoun resolution is crude (last mentioned region). It will be
    wrong sometimes; the history is there to inspect.
"""
import re

import numpy as np

MAP = 56  # joint priority map size


# ------------------------------------------------------- name matching
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


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z\s]", "", text.lower())


def is_addressed(text: str) -> bool:
    """True if the system's name appears in the text (fuzzy)."""
    norm = _normalize(text)
    return any(re.search(p, norm) for p in NAME_PATTERNS)


# ---------------------------------------------------------------- intents

class Intent:
    GREETING = "greeting"
    FAREWELL = "farewell"
    IDENTITY = "identity"
    SEE_QUESTION = "see_question"      # what do you see / where looking
    HEAR_QUESTION = "hear_question"    # what do you hear
    LOOK_COMMAND = "look_command"      # look left/right/up/down/center
    LOOK_AT = "look_at_thing"          # look at the X (ungrounded)
    THANKS = "thanks"
    YESNO_QUESTION = "yesno_question"
    WH_QUESTION = "wh_question"
    STATEMENT = "statement"
    UNKNOWN = "unknown"


def classify(text: str) -> str:
    """Rule-based intent from a transcript string."""
    norm = _normalize(text)

    if re.search(r"\b(bye|goodbye|good bye|see you|good night)\b", norm):
        return Intent.FAREWELL
    # SEE_QUESTION before IDENTITY: "what are you looking at" contains
    # the "what are you" substring.
    if re.search(r"\bwhat do you see\b|\bwhat are you looking at\b|"
                 r"\bwhere are you looking\b|\bdescribe what you see\b|"
                 r"\bwhat can you see\b", norm):
        return Intent.SEE_QUESTION
    if re.search(r"\bwho are you\b|\bwhat are you\b|\byour name\b", norm):
        return Intent.IDENTITY
    if re.search(r"\bwhat do you hear\b|\bwhat did you hear\b|"
                 r"\bwhat was that sound\b|\bwhat are you hearing\b", norm):
        return Intent.HEAR_QUESTION
    m = re.search(r"\blook\s+(left|right|up|down|center|centre|middle)\b", norm)
    if m:
        return Intent.LOOK_COMMAND
    if re.search(r"\blook at\b|\blook toward", norm):
        return Intent.LOOK_AT
    if re.search(r"\bthank", norm):
        return Intent.THANKS
    if re.search(r"\b(hi|hello|hey|howdy|greetings)\b", norm):
        return Intent.GREETING
    if re.search(r"^(do|does|did|is|are|was|were|can|could|would|will|"
                 r"have|has|should)\b", norm) or norm.rstrip().endswith("?"):
        # "?" alone is weak; leading auxiliary is the real signal.
        if re.match(r"^(do|does|did|is|are|was|were|can|could|would|"
                     r"will|have|has|should)\b", norm):
            return Intent.YESNO_QUESTION
    if re.match(r"^(what|where|who|when|why|how)\b", norm):
        return Intent.WH_QUESTION
    if len(norm.split()) <= 4:
        return Intent.STATEMENT
    return Intent.UNKNOWN


def look_direction(text: str) -> str | None:
    """Extract the direction from a look command, or None."""
    m = re.search(r"\blook\s+(left|right|up|down|center|centre|middle)\b",
                  _normalize(text))
    if not m:
        return None
    d = m.group(1)
    return "center" if d in ("centre", "middle") else d


# ------------------------------------------------------- perceptual state

def _qualitative(x224: float, y224: float) -> str:
    """(x, y) in 224px -> 'upper left' etc."""
    horiz = "left" if x224 < 75 else "right" if x224 > 149 else "center"
    vert = "upper" if y224 < 75 else "lower" if y224 > 149 else "middle"
    if horiz == "center" and vert == "middle":
        return "center"
    return f"{vert} {horiz}".replace("middle ", "").replace(" center", "")


class PerceptualState:
    """Read-only view over a live OnlineLevel3.

    This is the comprehension substrate: every grounded answer comes
    from here, not from language statistics.
    """

    def __init__(self, loop):
        self.loop = loop  # OnlineLevel3

    def gaze_now(self) -> tuple[float, float]:
        """Current fixation in 224px (last saccade target)."""
        sp = self.loop.scanpath
        return (sp[-1][1], sp[-1][2]) if sp else (112.0, 112.0)

    def gaze_where(self) -> str:
        x, y = self.gaze_now()
        return _qualitative(x, y)

    def recent_saccades(self, n: int = 3) -> list[tuple[float, float, float]]:
        """Last n (t_ms, x224, y224)."""
        return [(t, x, y) for t, x, y in self.loop.scanpath[-n:]]

    def peak_now(self) -> tuple[float, float, float]:
        """Current map peak (x, y, value) in map px."""
        return self.loop.jmap.peak()

    def describe(self) -> str:
        """Plain-language summary of the current perceptual state."""
        x, y = self.gaze_now()
        px, py, pv = self.peak_now()
        n_sac = max(len(self.loop.scanpath) - 1, 0)
        parts = [f"I'm looking at the {_qualitative(x, y)} of the frame"]
        if pv > 0.05:
            parts.append(f"the most salient point right now is "
                         f"{_qualitative(px * 4.0, py * 4.0)}")
        parts.append(f"I've made {n_sac} saccades so far")
        return ". ".join(parts) + "."


# ------------------------------------------------- language -> perception

_DIRECTION_XY = {
    # (map_x, map_y) on the 56x56 grid
    "left": (14.0, 28.0),
    "right": (42.0, 28.0),
    "up": (28.0, 14.0),
    "down": (28.0, 42.0),
    "center": (28.0, 28.0),
}


def direction_bias(direction: str, strength: float = 1.0,
                   sigma: float = 8.0) -> np.ndarray:
    """A Gaussian bias blob on the 56x56 map for a look command.

    Written into the shared priority map via the task_bias channel;
    the saccade controller reads the combined map, so language steers
    gaze through the same machinery the senses use. The blob decays
    with the map -- a command is a nudge, not a clamp.
    """
    cx, cy = _DIRECTION_XY[direction]
    yy, xx = np.mgrid[0:MAP, 0:MAP].astype(np.float32)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return (strength * blob).astype(np.float32)


# ---------------------------------------------------------- dialogue

class DialogueState:
    """Turn history. Pronoun resolution is crude by design: 'that' and
    'there' refer to the last mentioned region or gaze target."""

    def __init__(self, max_turns: int = 10):
        self.turns: list[tuple[str, str, str]] = []  # (text, intent, reply)
        self.max_turns = max_turns
        self.last_region: str | None = None

    def add(self, text: str, intent: str, reply: str,
            region: str | None = None):
        self.turns.append((text, intent, reply))
        self.turns = self.turns[-self.max_turns:]
        if region:
            self.last_region = region

    def history_text(self, n: int = 4) -> str:
        return "\n".join(f"them: {t}\nme: {r}"
                         for t, _, r in self.turns[-n:])


# -------------------------------------------------------------- entry

def understand(turn_text: str,
               perceptual: PerceptualState | None = None,
               dialogue: DialogueState | None = None
               ) -> tuple[str, str | None]:
    """Classify and respond. Returns (response_text, task_bias|None).

    task_bias is a 56x56 array for the perception loop, or None.
    """
    intent = classify(turn_text)
    bias = None

    if intent == Intent.GREETING:
        reply = (f"Hello! I heard you say: {turn_text.strip()}")
    elif intent == Intent.FAREWELL:
        reply = "Goodbye! I'll keep watching."
    elif intent == Intent.IDENTITY:
        reply = ("I'm Wodehaus -- a perceptual system. I see through a "
                 "priority map my eyes and ears both write to, and I hear "
                 "through a rolling transcriber. I don't understand the "
                 "way you do yet.")
    elif intent == Intent.SEE_QUESTION:
        if perceptual is not None:
            reply = perceptual.describe()
        else:
            reply = "My eyes aren't connected right now."
    elif intent == Intent.HEAR_QUESTION:
        reply = ("I hear through a rolling transcriber -- right now I "
                 "can report words as they're transcribed, but I don't "
                 "track sound identity yet.")
    elif intent == Intent.LOOK_COMMAND:
        direction = look_direction(turn_text)
        bias = direction_bias(direction)
        if dialogue is not None:
            dialogue.last_region = direction
        reply = f"Looking {direction}."
    elif intent == Intent.LOOK_AT:
        reply = ("I don't know what things look like yet -- I can look "
                 "left, right, up, down, or center, but I can't find "
                 "objects by name.")
    elif intent == Intent.THANKS:
        reply = "You're welcome."
    elif intent in (Intent.YESNO_QUESTION, Intent.WH_QUESTION):
        # Grounded fallback: report the perceptual state honestly
        # instead of confabulating an answer.
        if perceptual is not None and intent == Intent.WH_QUESTION:
            reply = (f"I heard your question: {turn_text.strip()} "
                     f"I don't understand it well enough to answer, but "
                     f"here's where I am: {perceptual.describe()}")
        else:
            reply = (f"I heard your question: {turn_text.strip()} "
                     f"I can hear the words, but I don't understand "
                     f"them yet.")
    elif intent == Intent.STATEMENT:
        reply = f"I heard: {turn_text.strip()}"
    else:
        reply = (f"I heard you say: {turn_text.strip()} I don't know "
                 f"what to make of that yet.")

    if dialogue is not None:
        region = look_direction(turn_text)
        dialogue.add(turn_text, intent, reply, region=region)
    return reply, bias
