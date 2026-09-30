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
  - Object recognition exists but is narrow: zero-shot CLIP over a
    small vocabulary (hvp/recognize.py), 6/12 top-1 on the hand-labeled
    dark-scene set. "Look at the red car" when no car has been seen
    gets "I don't know what a red car looks like yet" plus the list of
    things actually recognized so far. Out-of-vocabulary objects are
    invisible by name.
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
    r"woo\s*touse",     # Whisper heard "Hey Wodehaus" as "Hey, WooTouse"
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
    if re.search(r"\blook at\b|\blook toward|\bfind the\b|"
                 r"\bwhere is\b|\bwheres\b|where's|\blocate the\b", norm):
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

    def __init__(self, loop, memory: "ObjectMemory | None" = None):
        self.loop = loop  # OnlineLevel3
        self.memory = memory  # ObjectMemory fed by the recognition loop

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
        if self.memory is not None:
            known = self.memory.known_objects()
            if known:
                bits = []
                for name in known[:5]:
                    s = self.memory.last_seen(name)
                    if s is not None:
                        bits.append(f"{name} "
                                    f"({_qualitative(s[0] * 4.0, s[1] * 4.0)})")
                    else:
                        bits.append(name)
                parts.append("I've recognized: " + ", ".join(bits))
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


def _gaussian_blob(cx: float, cy: float, strength: float = 1.0,
                   sigma: float = 8.0) -> np.ndarray:
    yy, xx = np.mgrid[0:MAP, 0:MAP].astype(np.float32)
    blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return (strength * blob).astype(np.float32)


def direction_bias(direction: str, strength: float = 1.0,
                   sigma: float = 8.0) -> np.ndarray:
    """A Gaussian bias blob on the 56x56 map for a look command.

    Written into the shared priority map via the task_bias channel;
    the saccade controller reads the combined map, so language steers
    gaze through the same machinery the senses use. The blob decays
    with the map -- a command is a nudge, not a clamp.
    """
    cx, cy = _DIRECTION_XY[direction]
    return _gaussian_blob(cx, cy, strength, sigma)


def object_bias(map_x: float, map_y: float, strength: float = 1.2,
                sigma: float = 8.0) -> np.ndarray:
    """A Gaussian bias blob at a recognized object's map location.

    Same channel as direction_bias -- "look at the windows" steers
    gaze to where windows were last seen, through the same machinery.
    """
    return _gaussian_blob(map_x, map_y, strength, sigma)


# ------------------------------------------------------ object memory

class Track:
    """One hypothesized object: sightings of the same label, associated
    across fixations by proximity in space and time.

    Position and confidence are exponentially smoothed, so a flickering
    classifier (light-strip/dark/light-strip across three fixations)
    yields a stable track instead of a jumping point.
    """
    _next_id = 0

    def __init__(self, label: str, map_x: float, map_y: float,
                 t_ms: float, conf: float, smooth: float = 0.5):
        Track._next_id += 1
        self.id = Track._next_id
        self.label = label
        self.smooth = smooth
        self.x = map_x
        self.y = map_y
        self.conf = conf
        self.created_t = t_ms
        self.last_t = t_ms
        self.hits = 1

    def update(self, map_x: float, map_y: float, t_ms: float,
               conf: float):
        a = self.smooth
        self.x = a * map_x + (1 - a) * self.x
        self.y = a * map_y + (1 - a) * self.y
        self.conf = a * conf + (1 - a) * self.conf
        self.last_t = t_ms
        self.hits += 1

    def age(self, t_now: float) -> float:
        return t_now - self.last_t


class ObjectMemory:
    """What the eyes have recognized, and where.

    The perceptual loop records (label, map_x, map_y, t_ms, conf) each
    time the foveal classifier fires; sightings of the same label near
    each other in space and time are associated into Tracks. Language
    resolves "look at the X" against the most recent confident track.
    Out-of-vocabulary or never-seen objects simply aren't here -- the
    honest miss.
    """

    def __init__(self, max_sightings: int = 200,
                 min_conf: float = 0.30, max_age_ms: float = 60000.0,
                 assoc_dist: float = 10.0):
        self.sightings: list[tuple[str, float, float, float, float]] = []
        self.tracks: list[Track] = []
        self.max_sightings = max_sightings
        self.min_conf = min_conf
        self.max_age_ms = max_age_ms
        self.assoc_dist = assoc_dist

    def add(self, label: str, map_x: float, map_y: float,
            t_ms: float, conf: float):
        if label == "unknown" or conf < self.min_conf:
            return
        self.sightings.append((label, map_x, map_y, t_ms, conf))
        self.sightings = self.sightings[-self.max_sightings:]
        self._associate(label, map_x, map_y, t_ms, conf)

    def _associate(self, label: str, map_x: float, map_y: float,
                   t_ms: float, conf: float):
        want = _normalize_name(label)
        best = None
        best_d2 = self.assoc_dist ** 2
        for tr in self.tracks:
            if _normalize_name(tr.label) != want:
                continue
            if t_ms - tr.last_t > self.max_age_ms:
                continue
            d2 = (tr.x - map_x) ** 2 + (tr.y - map_y) ** 2
            if d2 < best_d2:
                best, best_d2 = tr, d2
        if best is None:
            self.tracks.append(Track(label, map_x, map_y, t_ms, conf))
        else:
            best.update(map_x, map_y, t_ms, conf)

    def _live_tracks(self, t_now: float) -> list[Track]:
        return [tr for tr in self.tracks
                if 0 <= t_now - tr.last_t <= self.max_age_ms]

    def locate(self, name: str, t_now: float
               ) -> tuple[float, float, float, float] | None:
        """Best track of `name`: (map_x, map_y, conf, age_ms).

        Position/confidence are track-smoothed, not the last raw
        sighting. Most hits wins; ties break by recency.
        """
        tracks = self.locate_all(name, t_now)
        if not tracks:
            return None
        tr = tracks[0]
        return (tr.x, tr.y, tr.conf, tr.age(t_now))

    def locate_all(self, name: str, t_now: float) -> list[Track]:
        """All live tracks of `name`, best first (hits, then recency)."""
        want = _normalize_name(name)
        cands = [tr for tr in self._live_tracks(t_now)
                 if _normalize_name(tr.label) == want]
        cands.sort(key=lambda tr: (tr.hits, tr.last_t), reverse=True)
        return cands

    def all_live_tracks(self, t_now: float) -> list[Track]:
        """Every live track regardless of label, best first."""
        cands = self._live_tracks(t_now)
        cands.sort(key=lambda tr: (tr.hits, tr.last_t), reverse=True)
        return cands

    def last_seen(self, name: str):
        """Most recent sighting regardless of age: (map_x, map_y, conf)."""
        want = _normalize_name(name)
        for label, mx, my, t, conf in reversed(self.sightings):
            if _normalize_name(label) == want:
                return (mx, my, conf)
        return None

    def known_objects(self) -> list[str]:
        """Labels seen, most recent first, deduplicated."""
        seen: list[str] = []
        for label, *_ in reversed(self.sightings):
            if label not in seen:
                seen.append(label)
        return seen


def _normalize_name(name: str) -> str:
    """'the windows' -> 'window'; naive singularization for matching.

    Spaces are dropped too: the vocabulary's 'light-strip' normalizes
    to 'lightstrip', matching speech's 'light strip'.
    """
    n = _normalize(name).strip()
    n = re.sub(r"^(the|a|an)\s+", "", n)
    if n.endswith("s") and not n.endswith("ss"):
        n = n[:-1]
    return n.replace(" ", "")


def extract_referent(text: str) -> str | None:
    """Pull the X out of 'look at (the) X' / 'find (the) X' /
    'where is (the) X'."""
    m = re.search(r"\b(?:look\s+(?:at|toward(?:s)?)|find|locate|"
                  r"where(?:'s|\s+is))\s+(?:the\s+|a\s+|an\s+)?"
                  r"(.+?)(?:\s+please)?\s*$", _normalize(text))
    if not m:
        return None
    return m.group(1).strip() or None


_SPATIAL_SELECTORS = ("leftmost", "rightmost", "topmost", "uppermost",
                      "bottommost", "lowermost", "nearest", "closest",
                      "farthest", "furthest")


def split_spatial(candidate: str
                ) -> tuple[str | None, str]:
    """Split a spatial selector off a referent.

    'leftmost windows' -> ('leftmost', 'windows');
    'the windows on the left' -> ('left', 'windows').
    No selector -> (None, candidate).
    """
    norm = _normalize(candidate)
    if norm.startswith("the "):
        norm = norm[4:]
    words = norm.split()
    if words and words[0] in _SPATIAL_SELECTORS:
        return words[0], " ".join(words[1:]).strip()
    m = re.search(r"\b(?:on the|at the|in the)\s+"
                  r"(left|right|top|upper|bottom|lower|middle|center)"
                  r"\s*$", norm)
    if m:
        return m.group(1), norm[:m.start()].strip()
    return None, candidate


def _pick_by_spatial(tracks: list, spatial: str,
                     gaze_xy: tuple[float, float] | None = None):
    """One track from a list by a spatial selector. None when empty.

    Map coords: x 0=left 56=right, y 0=top 56=bottom. 'nearest' is
    measured from the current gaze (frame center when unknown).
    """
    if not tracks:
        return None
    gx, gy = gaze_xy if gaze_xy is not None else (28.0, 28.0)
    if spatial in ("leftmost", "left"):
        return min(tracks, key=lambda tr: tr.x)
    if spatial in ("rightmost", "right"):
        return max(tracks, key=lambda tr: tr.x)
    if spatial in ("topmost", "uppermost", "top", "upper"):
        return min(tracks, key=lambda tr: tr.y)
    if spatial in ("bottommost", "lowermost", "bottom", "lower"):
        return max(tracks, key=lambda tr: tr.y)
    if spatial in ("nearest", "closest"):
        return min(tracks,
                   key=lambda tr: (tr.x - gx) ** 2 + (tr.y - gy) ** 2)
    if spatial in ("farthest", "furthest"):
        return max(tracks,
                   key=lambda tr: (tr.x - gx) ** 2 + (tr.y - gy) ** 2)
    return min(tracks,  # "middle" / "center"
               key=lambda tr: (tr.x - 28.0) ** 2 + (tr.y - 28.0) ** 2)


def resolve_spatial_referent(name: str, spatial: str,
                             memory: "ObjectMemory", t_now: float,
                             gaze_xy: tuple[float, float] | None = None):
    """Pick one track of `name` by a spatial selector.

    Returns (track, matched_name); (None, name) when nothing matches.
    """
    track = _pick_by_spatial(memory.locate_all(name, t_now), spatial,
                             gaze_xy)
    return (track, name) if track is not None else (None, name)


_ONE_CANONICAL = {"left": "leftmost", "right": "rightmost",
                  "top": "topmost", "upper": "uppermost",
                  "bottom": "bottommost", "lower": "lowermost"}


def resolve_one_track(spatial: str, memory: "ObjectMemory",
                      t_now: float, label: str | None,
                      gaze_xy: tuple[float, float] | None = None):
    """'the left one': pick among the last-referenced label's tracks,
    or among every live track when no label is in play. Spatial
    queries are relative among known tracks, so the detector is
    never consulted here."""
    if label:
        tracks = memory.locate_all(label, t_now)
    else:
        tracks = memory.all_live_tracks(t_now)
    return _pick_by_spatial(tracks, spatial, gaze_xy)


_PRONOUNS = ("it", "that", "this", "there")
_DIRECTIONS = ("left", "right", "up", "down")


def split_anaphor(referent: str) -> tuple[str | None, str | None]:
    """Split anaphoric forms off a referent.

    'it' / 'it again' -> ('pronoun', None); 'that gate' ->
    ('det', 'gate'); 'the left one' -> ('one', 'left').
    Returns (None, None) for ordinary referents.
    """
    words = _normalize(referent).split()
    if not words:
        return None, None
    if words[0] in _PRONOUNS:
        if len(words) == 1 or words[1:] == ["again"]:
            return "pronoun", None
        if words[0] in ("that", "this"):
            # determiner + noun: 'that gate' -> 'gate'
            return "det", " ".join(words[1:])
        return None, None
    rest = words[1:] if words[0] == "the" else words
    if (len(rest) == 2 and rest[1] == "one"
            and rest[0] in _ONE_CANONICAL):
        return "one", rest[0]
    return None, None


def base_label(label: str | None) -> str | None:
    """Strip a spatial selector and direction words off a stored
    region name: 'leftmost windows' -> 'windows'; 'left' -> None."""
    if not label:
        return None
    if _normalize(label) in _DIRECTIONS:
        return None
    spatial, name = split_spatial(label)
    return name if spatial else label


def resolve_referent(candidate: str, memory: "ObjectMemory",
                     t_now: float):
    """Match a referent against memory, tolerating trailing narration.

    Whisper merges hails with surrounding speech ('look at the windows
    the'), so try the full candidate, then successively shorter
    leading prefixes; first memory hit wins. Returns
    (sighting, matched_name); (None, candidate) when nothing matches.
    """
    words = candidate.split()
    for n in range(len(words), 0, -1):
        name = " ".join(words[:n])
        sighting = memory.locate(name, t_now)
        if sighting is not None:
            return sighting, name
    return None, candidate


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
               dialogue: DialogueState | None = None,
               memory: ObjectMemory | None = None,
               t_now_ms: float = 0.0,
               gaze_xy: tuple[float, float] | None = None,
               detect_fn=None,
               ) -> tuple[str, str | None]:
    """Classify and respond. Returns (response_text, task_bias|None).

    task_bias is a 56x56 array for the perception loop, or None.
    memory is the ObjectMemory the loop feeds; t_now_ms anchors
    sighting ages; gaze_xy is the current fixation in map coords
    (for 'nearest'/'farthest'). detect_fn, when provided, is called
    as detect_fn([query]) -> [(label, map_x, map_y, conf)] and is the
    last resort for referents with no track in memory.
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
        referent = extract_referent(turn_text)
        sighting, matched = None, referent
        spatial_word = None
        anaphor, detail, store_region = None, None, None
        if referent:
            # Anaphora resolves against the dialogue state before the
            # ordinary memory path: 'it'/'that' -> last region,
            # 'that gate' -> 'gate', 'the left one' -> spatial pick.
            anaphor, detail = split_anaphor(referent)
            if anaphor == "det":
                referent, matched = detail, detail
                anaphor = None
            elif anaphor == "pronoun":
                last = dialogue.last_region if dialogue else None
                if last and _normalize(last) in _DIRECTIONS:
                    last = None  # 'look left' leaves no object to mean
                if last:
                    # verbatim: 'leftmost windows' re-enters the
                    # spatial path and lands on the same track
                    referent, matched = last, last
                    anaphor = None
        if memory is not None and referent and anaphor != "pronoun":
            if anaphor == "one":
                label = base_label(
                    dialogue.last_region if dialogue else None)
                track = resolve_one_track(detail, memory, t_now_ms,
                                          label, gaze_xy)
                if track is not None:
                    sighting = (track.x, track.y, track.conf,
                                track.age(t_now_ms))
                    matched = f"{detail} {track.label}"
                    # canonical form so a later 'it' re-resolves here
                    store_region = (f"{_ONE_CANONICAL[detail]} "
                                    f"{track.label}")
            else:
                spatial_word, name = split_spatial(referent)
                if spatial_word:
                    track, matched = resolve_spatial_referent(
                        name, spatial_word, memory, t_now_ms, gaze_xy)
                    if track is not None:
                        sighting = (track.x, track.y, track.conf,
                                    track.age(t_now_ms))
                        matched = ((spatial_word + " " + matched).strip()
                                   or matched)
                else:
                    sighting, matched = resolve_referent(referent, memory,
                                                         t_now_ms)
        if sighting is not None:
            mx, my, conf, age = sighting
            bias = object_bias(mx, my, strength=1.2 * conf)
            where = _qualitative(mx * 4.0, my * 4.0)
            if dialogue is not None:
                dialogue.last_region = store_region or matched
            reply = (f"Looking at the {matched} -- I saw it {where}.")
        else:
            # Last resort: scan the current frame for something never
            # fixated. Only when not a spatial/anaphoric query (those
            # are relative among known tracks) and a detector hook is
            # provided.
            found = None
            if (detect_fn is not None and referent and not spatial_word
                    and anaphor is None and memory is not None):
                try:
                    dets = detect_fn([referent])
                except Exception:
                    dets = None
                if dets:
                    label, mx, my, conf = dets[0]
                    memory.add(label, mx, my, t_now_ms, conf)
                    found = (label, mx, my, conf)
            if found is not None:
                label, mx, my, conf = found
                bias = object_bias(mx, my, strength=1.2 * conf)
                where = _qualitative(mx * 4.0, my * 4.0)
                if dialogue is not None:
                    dialogue.last_region = label
                reply = (f"Found the {referent} -- looking at it {where}.")
            elif anaphor == "pronoun":
                reply = ("I don't know what 'it' refers to yet -- "
                         "I haven't locked onto anything.")
            else:
                known = (memory.known_objects()
                         if memory is not None else [])
                if referent:
                    reply = (f"I don't know what a {referent} looks like yet")
                else:
                    reply = "I couldn't tell what you want me to look at"
                if known:
                    reply += (f" -- so far I've recognized: "
                              f"{', '.join(known)}")
                reply += "."
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
