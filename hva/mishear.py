"""Mishearing model: modeled alternatives for low-confidence words.

EXPERIMENTAL (Phase 5, 2026-10-12; audit item 15). Not wired into
the flagship streaming path.

The standing rule this module embodies: **transcription claims
carry logprobs or they don't ship.** A low-confidence segment is
doubtful data, and this module's job is to make the doubt
*legible* — for the words the transcriber itself was unsure of,
it proposes phonetically plausible alternative readings ("the
ear may have misheard X as Y"). Those alternatives are **MODEL
OUTPUT**, labeled as such in every record the module emits. They
are never a replacement for the verbatim text: the verbatim field
is a byte-identical copy of the input segment text, and nothing
in this module mutates its input.

What it cannot do (demonstrated on the FF3 1649.7 s window, see
improvements-2026-10/PREDICTIONS.md Phase 5 scoring): recover
what a sound *actually was* when the base tier hallucinated words
onto a non-lexical sound. It proposes lexical neighbors of the
words the base tier thought it heard. It annotates doubt; it
does not correct hallucinations into truth.

Method (self-contained, no new dependencies):
- Gate: only segments with avg_logprob below the interpret.py
  escalation threshold (LOW_CONF_THRESHOLD, imported — one source
  of truth) are modeled, and within them only words with prob
  below WORD_PROB_THRESHOLD. High-confidence segments yield no
  modeled mishearings, by construction.
- Phonetic key: a Metaphone-lite consonant skeleton
  (phonetic_key). Digraph folding (ph->f class, th->theta class,
  soft c/g), silent initial letters (kn/gn/pn/wr), silent gh,
  class folding for labials (b/p/f/v) and dentals (t/d),
  non-initial vowels dropped, doubles collapsed. Deliberately
  coarse: it is a candidate generator, not a phonetician.
- Candidates: words sharing the key (exact tier, similarity 1.0)
  or at key edit-distance 1 (near tier, similarity 0.5), looked
  up in the whisper tokenizer vocabulary (word-initial alphabetic
  tokens from models/faster-whisper-base/tokenizer.json; rank =
  lowest token id, a BPE-frequency proxy).
- Score = (1 - word_prob) * similarity * (0.5 + 0.5 * freq_prior)
  where freq_prior = 1 - rank/max_rank over the vocabulary.
  The (1 - word_prob) factor is the doubt: the less the
  transcriber believed its own word, the more room a modeled
  alternative gets. Scores are relative ranks, not probabilities.
"""

from __future__ import annotations

import json
import os
import re

from hva.interpret import LOW_CONF_THRESHOLD

WORD_PROB_THRESHOLD = 0.5   # below this, a word is "low-probability"
MAX_WORDS_PER_SEGMENT = 3   # model at most this many words/segment
MAX_ALTERNATIVES = 5        # per word
MODEL_TAG = "mishear-v1 (metaphone-lite)"
DISCLAIMER = ("MODEL OUTPUT - phonetic alternatives modeled by "
              "hva.mishear; not a transcript, not a correction. "
              "The verbatim text is the only transcript claim.")

# ---------------------------------------------------------------------
# Phonetic key (Metaphone-lite)
# ---------------------------------------------------------------------

def phonetic_key(word: str) -> str:
    """Coarse phonetic skeleton for a word. '' for non-words."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    # Silent initial letters: knight -> night, write -> rite.
    if w[:2] in ("kn", "gn", "pn"):
        w = w[1:]
    elif w[:2] == "wr":
        w = w[1:]
    # Digraphs and letter classes (order matters).
    w = w.replace("ph", "f")
    w = w.replace("sch", "sk")
    w = w.replace("ck", "k")
    w = w.replace("ch", "x")          # x = ch/sh class
    w = w.replace("sh", "x")
    w = w.replace("th", "0")          # 0 = theta class
    w = w.replace("dge", "j")
    w = w.replace("gh", "")           # silent gh (night -> nit)
    w = w.replace("q", "k")
    w = w.replace("x", "ks")
    w = w.replace("z", "s")
    w = re.sub(r"c([eiy])", r"s\1", w)   # soft c
    w = w.replace("c", "k")
    w = re.sub(r"g([eiy])", r"j\1", w)   # soft g (lite: over-fires
    #                                      on 'get'; accepted, the
    #                                      key only generates
    #                                      candidates)
    if w.endswith("mb"):
        w = w[:-1]                       # lamb -> lam
    # Class folding: all labials (b/p/f/v) to one class, dentals
    # (t/d) to one class — same gesture, confusable by ear.
    w = w.translate(str.maketrans({"b": "p", "v": "p", "f": "p",
                                   "d": "t"}))
    # Drop non-initial vowels (initial vowel is identity-bearing).
    if len(w) > 1:
        w = w[0] + re.sub(r"[aeiouy]", "", w[1:])
    # Collapse doubled consonants.
    w = re.sub(r"(.)\1+", r"\1", w)
    return w.upper()


def _levenshtein(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 1:
        return 2  # callers only ask "== 1?"
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------
# Candidate vocabulary (whisper tokenizer, word-initial tokens)
# ---------------------------------------------------------------------

_INDEX = None  # (key -> [(word, rank)], max_rank)


def _tokenizer_vocab_paths() -> list[str]:
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "models")
    return [os.path.join(root, "faster-whisper-base", "tokenizer.json"),
            os.path.join(root, "faster-whisper-medium",
                         "tokenizer.json")]


def build_index(vocab: dict[str, int]):
    """key -> [(word, rank)] from a {word: rank} vocabulary."""
    index: dict[str, list[tuple[str, int]]] = {}
    for word, rank in vocab.items():
        k = phonetic_key(word)
        if k:
            index.setdefault(k, []).append((word, rank))
    return index


def load_vocab() -> dict[str, int]:
    """{word: rank} from the local whisper tokenizer vocabulary.

    Word-initial alphabetic tokens only (the byte-level BPE 'Ġ'
    prefix marks a word start); rank = lowest token id across
    case variants, a rough frequency proxy (earlier BPE merges
    are more frequent). Raises FileNotFoundError naming the
    expected paths when no local model tokenizer exists.
    """
    for path in _tokenizer_vocab_paths():
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)["model"]["vocab"]
            words: dict[str, int] = {}
            for tok, idx in raw.items():
                t = tok.replace("Ġ", " ")
                if re.fullmatch(r" [A-Za-z]{2,}", t):
                    w = t.strip().lower()
                    if w not in words or idx < words[w]:
                        words[w] = int(idx)
            return words
    raise FileNotFoundError(
        "no whisper tokenizer.json under models/faster-whisper-*/; "
        "the mishearing model needs the local model vocabulary "
        f"(looked in {_tokenizer_vocab_paths()})")


def _index():
    global _INDEX
    if _INDEX is None:
        vocab = load_vocab()
        _INDEX = (build_index(vocab), max(vocab.values()))
    return _INDEX


def candidate_readings(word: str, vocab: dict[str, int] | None = None,
                       index=None, max_rank: int | None = None,
                       max_results: int = MAX_ALTERNATIVES,
                       word_prob: float = 0.0) -> list[dict]:
    """Ranked phonetic alternatives for one word.

    Returns [{reading, score, key_match, key}] — exact-key matches
    first, then edit-distance-1 keys; the word itself excluded.
    `vocab`/`index`/`max_rank` are injectable for tests; defaults
    use the whisper tokenizer vocabulary.
    """
    if vocab is None and index is None:
        index, max_rank = _index()
    elif index is None:
        index = build_index(vocab)
        max_rank = max(vocab.values()) if vocab else 1
    key = phonetic_key(word)
    if not key:
        return []
    target = word.strip().lower()
    cands: list[tuple[float, str, str, str]] = []

    def consider(w: str, rank: int, sim: float, tier: str):
        if w == target:
            return
        freq_prior = 1.0 - rank / max_rank if max_rank else 0.0
        score = (1.0 - word_prob) * sim * (0.5 + 0.5 * freq_prior)
        cands.append((score, w, tier, key if tier == "exact"
                      else phonetic_key(w)))

    for w, rank in index.get(key, []):
        consider(w, rank, 1.0, "exact")
    for k2, entries in index.items():
        if k2 != key and _levenshtein(key, k2) == 1:
            for w, rank in entries:
                consider(w, rank, 0.5, "near")
    cands.sort(key=lambda c: (-c[0], c[1]))
    return [{"reading": w, "score": round(score, 4),
             "key_match": tier, "key": k}
            for score, w, tier, k in cands[:max_results]]


# ---------------------------------------------------------------------
# Segment modeling
# ---------------------------------------------------------------------

_WORD_PUNCT = re.compile(r"^(\W*)(.*?)(\W*)$")


def _modeled_reading(words: list[dict], idx: int,
                     alternative: str) -> str:
    """Segment text with words[idx] replaced by `alternative`.

    Rebuilt from the word tokens (the words list is the segment's
    own tokenization); original capitalization of the replaced
    word's first letter is preserved.
    """
    parts = []
    for i, w in enumerate(words):
        tok = w.get("word", "")
        if i == idx:
            m = _WORD_PUNCT.match(tok.strip()) if tok.strip() else None
            lead = tok[:len(tok) - len(tok.lstrip())]
            core = tok.strip()
            if core[:1].isupper():
                alternative = alternative.capitalize()
            tok = lead + alternative
            if m and m.group(3):
                tok += m.group(3)
        parts.append(tok)
    return "".join(parts).strip()


def model_segment(segment: dict,
                  low_conf_threshold: float = LOW_CONF_THRESHOLD,
                  word_prob_threshold: float = WORD_PROB_THRESHOLD,
                  vocab: dict[str, int] | None = None) -> dict:
    """One transcript segment -> verbatim + modeled mishearings.

    Input format: hva.transcribe / interpret.two-tier segments —
    {start, end, text, avg_logprob, words: [{word, prob, ...}]}.
    The input dict is not mutated; `verbatim` is the input text,
    byte-identical.
    """
    verbatim = segment.get("text", "")
    lp = segment.get("avg_logprob")
    out = {
        "start": segment.get("start"),
        "end": segment.get("end"),
        "verbatim": verbatim,
        "avg_logprob": lp,
        "confidence": lp,
        "modeled": False,
        "modeled_mishearings": [],
        "model": MODEL_TAG,
        "model_output": True,
        "disclaimer": DISCLAIMER,
    }
    # No logprob -> no claim at all (the standing rule). Below
    # threshold only; high-confidence segments stay unmodeled.
    if lp is None or float(lp) >= low_conf_threshold:
        return out
    words = segment.get("words") or []
    low = [(i, w) for i, w in enumerate(words)
           if w.get("prob") is not None
           and float(w["prob"]) < word_prob_threshold
           and re.search(r"[A-Za-z]", w.get("word", ""))]
    low.sort(key=lambda iw: float(iw[1]["prob"]))
    entries = []
    for i, w in low[:MAX_WORDS_PER_SEGMENT]:
        core = _WORD_PUNCT.match(w["word"].strip())
        core_word = core.group(2) if core else w["word"].strip()
        alts = candidate_readings(core_word, vocab=vocab,
                                   word_prob=float(w["prob"]))
        if not alts:
            continue
        entries.append({
            "word_index": i,
            "word": w["word"].strip(),
            "word_prob": float(w["prob"]),
            "alternatives": alts,
            "modeled_reading": _modeled_reading(
                words, i, alts[0]["reading"]),
        })
    out["modeled"] = bool(entries)
    out["modeled_mishearings"] = entries
    return out


def model_segments(segments: list[dict], **kw) -> list[dict]:
    """model_segment over a segment list, order preserved."""
    return [model_segment(s, **kw) for s in segments]
