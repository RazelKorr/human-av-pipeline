# Object recognition: model choice and chain (2026-09-30)

The "cortex" is no longer a human with a JSON file. What replaced it,
why, what it costs, and what it still cannot do.

## The pick: CLIP ViT-B/32, zero-shot, via transformers

- **Model:** `openai/clip-vit-base-patch32`, `transformers` 5.17.0,
  CPU, lazy load, 578 MB fp32 weights (HuggingFace cache,
  gitignored).
- **Why this one:**
  - Already reachable through `transformers` -- no new heavy
    dependency. (`open_clip_torch` was tried first and died on a
    broken `torchvision` operator registration; uninstalled.)
  - Zero-shot over a caller-supplied vocabulary: no object-specific
    training, no label collection beyond the audit set.
  - Fits the foveal-crop seam exactly: the pipeline already extracts
    a 96x96 crop around each fixation; the classifier consumes that
    crop and nothing else.
  - Text embeddings cached per vocabulary; ~300 ms per crop on this
    CPU; classification throttled to moved-gaze in the live loops.
  - Honest by construction: below `unknown_threshold=0.30` it
    returns `"unknown"` instead of forcing a label, and confidence
    is relative to the supplied vocabulary, never absolute.
- **Measured (scripts/audit_recognition.py, 12 hand-labeled foveal
  crops, dark gate/corridor scene):** 4/12 top-1 with naive prompts,
  **6/12** with tuned grayscale/discriminative prompts. Remaining
  misses are dark/ambiguous crops where the human labeler had
  temporal context the single crop lacks.

## What it is not

- Not general object detection: no bounding boxes, no segmentation.
- Not tracking: sightings are per-fixation; `ObjectMemory` keeps
  the last sighting per label for 60 s, nothing more.
- Not open-world: out-of-vocabulary objects are invisible by name.
  `GENERAL_VOCAB` is a 20-word starter list, unaudited.
- Not validated human-equivalent recognition. The audit is 12 crops
  from one dark scene. Say "6/12 on the dark-scene set", not
  "it recognizes objects".
- No color grounding, no speaker/source identity, no size/distance
  beyond the qualitative region words.

## Alternatives considered

- **SigLIP:** better zero-shot retrieval in the literature; not
  tried here -- CLIP's interface was sufficient and the crop seam
  was the priority. Revisit if a head-to-head audit warrants it.
- **Smaller CLIP variants (RN50, ViT-B/16):** same family, same
  interface; ViT-B/16 is slower per crop (4x the patches), RN50
  untested in this loop. Not worth the swap now.
- **Detector-based (YOLOv8, DETR, OWL-ViT):** the honest upgrade
  path. A detector returns boxes natively, which is what "where is
  the X" actually wants -- our current grounding classifies the
  *fixated* crop, so an object is only nameable after gaze has
  already found it. Fixed-vocabulary detectors (COCO 80) trade the
  zero-shot phrasing away; open-vocabulary detectors (OWL-ViT) keep
  it but cost more per frame. Deferred: heavier deps, and the crop
  seam was the fastest honest step.
- **Caption/VLM (BLIP-2, LLaVA):** richer descriptions, but
  autoregressive decoding per crop is much slower on this CPU than
  one contrastive forward pass (not benchmarked here -- estimate),
  and prone to fluent hallucination -- the opposite of the
  honest-unknown design. Not a fit for the 10 Hz loop.
- **DINOv2:** excellent visual features, but no zero-shot labels
  without a trained head; adds a training step we don't have data
  for.

## The chain (audited end to end)

```
frame -> foveal crop (96x96 @ fixation, 2x upscale)
      -> FovealClassifier (CLIP zero-shot, vocab-bound, unknown<0.30)
      -> ObjectMemory.add(label, map_x, map_y, t, conf)
      -> (a) live_recognizer: conf-scaled task bias in run_video_topdown
      -> (b) "look at the X": locate() -> object_bias() at the sighting
      -> (c) "what do you see?": PerceptualState.describe() names objects
      -> (d) ResponsePolicy: memory + t_now on every turn; LLM payload
          inherits objects via describe()
```

## Audits

1. **Recognition** (`scripts/audit_recognition.py`): 6/12 top-1,
   tuned prompts, CPU ~310 ms/crop. `tests/test_recognition.py`
   7/7 (model stubbed).
2. **Live cortex** (`scripts/run_video_topdown.py --recognizer live`,
   30 s Star Tours): 8/99 fixations diverge >5 deg from bottom-up;
   deterministic across runs (identical events, 1/26 divergence on
   8 s reruns).
3. **Grounding** (`scripts/audit_grounding.py`): memory fed by real
   CLIP output on the 12 crops; "look at the windows" lands the bias
   peak exactly on the windows sighting; "look at the red car" gets
   the honest miss with the known-object list.
4. **Conversation stream** (`scripts/audit_conversation_labels.py`
   PASS; live loop `scripts/run_conversation.py --foveal-vocab
   dark-street`, 40 s Star Tours + 2 TTS hails):
   - The tick loop fed the memory live: dozens of sightings -- gate,
     windows, dark, light-strip, distant-light, sign (up to 0.91),
     lit-floor (up to 0.88).
   - Policy-level: "what do you see?" ->
     "I've recognized: light-strip (right), windows (right), dark
     (lower right), distant-light (lower right), gate (center)."
   - Live loop: "Hey Woodhouse, look at the windows" (merged by
     Whisper with ride narration) ->
     "Looking at the windows -- I saw it left." with the task bias
     written to the shared map.
   - Two robustness fixes fell out of the live run: Whisper heard
     "Wodehaus" as "WooTouse" (added to NAME_PATTERNS per the file's
     observed-variant convention), and trailing narration in the
     turn text ("look at the windows the") is now tolerated by
     resolve_referent(), which tries successively shorter prefixes.
   - Honest limitation observed: the first hail's "what do you see?"
     arrived in a separate Whisper segment from its "Hey, WooTouse",
     so that turn fired as a greeting -- pre-existing turn
     segmentation behavior, not a recognition failure.

## Standing limits

Determinism holds for the pipeline; CLIP's outputs are
deterministic given weights and inputs on this CPU build, but model
weights are third-party artifacts outside this repo -- version them
by `HVP_CLIP_MODEL`, not by hash. If the weights change, re-run
audit 1.
