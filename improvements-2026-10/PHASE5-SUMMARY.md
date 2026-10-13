# Phase 5 + 6 summary — new capabilities & readiness (2026-10-12)

Audit Lane 7 items 14, 15, 17 (the "needs Mykal's call" items —
called: "do all of it") plus the Phase 6 readiness note. All
three capabilities are **experimental additions**: new modules /
a new script + tests + one demonstration run each. The flagship
streaming path's defaults were not rewired by any of it. Sealed
predictions were committed before any Phase 5 code (13d2726);
scoring is appended in PREDICTIONS.md — 2 hit + 1 hit-as-
limitation + 1 vacuous (item 15), 1 hit + 2 miss (item 17),
4 hit with leakage caveats (item 14).

## Item 15 — Mishearing model (`hva/mishear.py`) — DONE

For transcript segments below the interpret.py escalation
threshold (avg_logprob < −0.8, imported — one source of truth),
low-probability words (prob < 0.5) get phonetically plausible
alternative readings: a self-contained Metaphone-lite key,
candidates from the whisper tokenizer vocabulary, rescored
(1 − word_prob) × phonetic similarity × frequency prior.
Output per segment: `verbatim` (byte-identical, input never
mutated), confidence, and `modeled_mishearings` — every record
labeled MODEL OUTPUT. Segments without a logprob are never
modeled (transcription claims carry logprobs or they don't
ship).

Demonstration (FF3 1649.7 s window, the "We're in Marvel!"
hallucination; `output/ff3_hd_av/mishear_demo_1649.json`): with
vad=False the base decode reproduces the stored phase-1 record
(lp −0.875; with VAD on it hallucinates *differently* — "We're
in silence!" — a configuration-sensitivity finding in itself).
"Marvel" (prob 0.091) models to **memorable / marble /
maravil** (exact-key tier). **What it cannot do, demonstrated:**
none of the alternatives recovers what the medium tier found —
non-lexical screams ("BLEH", "NOOOO"). The model annotates
doubt around the words the base tier invented; it does not
correct hallucinations into truth. That limitation is the
item's headline result. Tests: tests/test_mishear.py, 8/8.

## Item 17 — OCR for diegetic signage (`hvp/ocr.py`) — DONE, capability NOT established

Engine situation: **no blocker.** tesseract 5.3.4 was already
installed at OS level; pytesseract 0.3.13 installed into the
venv cleanly and is pinned in requirements.txt; tesseract is
documented in SETUP.md §1 as a system prerequisite
(VM-ephemeral, same class as ffmpeg). The module is built
against a pluggable engine interface (`read_words`), so the
synthetic tests run on a stub and a future engine slots in
without touching callers. API: frames / video timestamps in,
[{t, text, confidence, bbox}] out, confidence-floored.

Validation, honestly: on the synthetic fixture (PIL-rendered
DejaVu text, 1280×720), exact match at **every** sealed size —
48/32/24/18/14/10 px — with total failure only at ≤8 px
(extension). The sealed failure-threshold prediction missed
because sterile rendered text is far easier than real signage.
On Star Tours frames at the known-signage moments, **no real
sign was read**: garbage tokens on the bay signage, texture
pareidolia (up to conf 80) on the closing near-miss frames.
Disposition: signage in footage stays human-read per
REVIEW-PROTOCOL §6. This module is a working seam with a
sterile-regime baseline (tests/test_ocr.py 4/4;
output/ocr_synthetic_validation.json,
output/ocr_star_tours_qualitative.json), not a claim that the
pipeline can read signs.

## Item 14 — Stem-stream experiment (`scripts/stem_streams.py`) — DONE

Demucs stems (the restored `--separate-vocals` path), each stem
through hva/online.py features + onset picking, two-tier
transcription, and CLAP aggregation — per-stem streams in one
JSON record.

Synthetic mix (Star Tours vocals excerpt + planted tone
sequence at −6 dB; ground truth pre-mix transcript): vocals-stem
token containment **0.647** (bar 0.6) — the famous lines survive
("I meant to do that / A little shortcut"), the opening line is
garbled and the transcript fragments into 13 micro-segments.
No_vocals stem: CLAP top-1 **"singing"** (thin margin over
"silence"), transcript empty. Onsets 19/19 vs the mix's 21 —
sparser by the sealed bar, but the stems **leak** (tone times
in the vocals onsets, speech times in the no_vocals onsets).

Real-world demo (62 s of Star Tours): vocals stem recovers 44
words of genuine dialogue at avg_logprob ≥ −0.8; no_vocals
stem transcribes 0 words with CLAP top-1 "music". But its
onset counts (68/60) **exceed** the un-separated mix (54) —
separation artifacts manufacture transients, so onset sparsity
is a synthetic-mix result, not a wild one. Cross-cutting: this
faster-whisper's per-segment avg_logprob is near-uniform within
a run (−0.554 across all 13 synth segments) — confidence
discrimination is coarse, flagged in PREDICTIONS.md. Outputs:
output/stem_streams_synth.json, output/stem_streams_st62.json.
Tests: tests/test_stem_streams.py, 3/3 (deterministic machinery
only).

**Verdict on the audit's item-14 framing** — "the pipeline
cannot tell engines from score": at the *label/transcript*
level the stems now separate voice from non-voice convincingly
(dialogue on one stem, music labels + no words on the other).
At the *onset/feature* level they do not yet form clean
objects. A first auditory-object layer exists; it is leaky,
and the leak is measured.

## Phase 6 — Live-sensors readiness (paper only) — DONE

`improvements-2026-10/LIVE-SENSORS-READINESS.md`. **No sensor
work was done or will be done under this commission.** The
note records: the seam (ChunkedFeeder is the only file-aware
component; hva/stream.py is the audio counterpart), what is
already causal/bounded/live-ready, what is missing (a capture
layer, a backpressure decision, hardware adapters), and the
gates that remain Mykal's — the 2026-10-06 terms-first
agreement and the hardware boundary (paired iPhone exposes no
camera/mic; live path is future ESP32-class hardware).

## Suite & commits

Full suite after Phase 5: **301 passed, 1 warning, exit 0**
(captured to file, read from the file; 286 at Phase 4 + 15 new
tests). Commits: 13d2726 (sealed predictions), b7e3b06 (item
15), f6384f0 (item 17), 5ebec0e (item 14), b1c333d (Phase 6),
plus this summary + scoring commit. Not pushed.
