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

- Not general object detection in the loop: the 10 Hz path still
  classifies fixated crops. Boxes exist only via the on-demand
  OWL-ViT fallback (`hvp/detect.py`), ~2 s per frame on CPU.
- Not open-world: out-of-vocabulary objects are invisible by name to
  the classifier; the detector takes free-text queries but was only
  audited on "a window".
  `GENERAL_VOCAB` is a 20-word starter list, unaudited.
- Not validated human-equivalent recognition. The audit is 12 crops
  from one dark scene. Say "6/12 on the dark-scene set", not
  "it recognizes objects".
- No color grounding, no speaker/source identity, no size/distance
  beyond the qualitative region words and relative track selectors.

## Alternatives considered

- **SigLIP:** better zero-shot retrieval in the literature; not
  tried here -- CLIP's interface was sufficient and the crop seam
  was the priority. Revisit if a head-to-head audit warrants it.
- **Smaller CLIP variants (RN50, ViT-B/16):** same family, same
  interface; ViT-B/16 is slower per crop (4x the patches), RN50
  untested in this loop. Not worth the swap now.
- **Detector-based (YOLOv8, DETR, OWL-ViT):** was the honest
  upgrade path; built 2026-09-30 with OWL-ViT
  (`google/owlvit-base-patch32`, `hvp/detect.py`). Open-vocabulary
  like the CLIP pick, ~2 s/frame CPU, on-demand only: "where is the
  X" / "find the X" with no track scans the current frame, best box
  gets the gaze bias, >= 0.30 conf also becomes a track. Fixed-vocab
  detectors (COCO 80) were skipped -- they trade the zero-shot
  phrasing away.
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
          -> Track association: same label within 10 map-px / 60 s
             joins one Track; position/conf are EMA-smoothed (0.5)
      -> (a) live_recognizer: conf-scaled task bias in run_video_topdown
      -> (b) "look at the X": locate() -> object_bias() at the track
      -> (c) "the leftmost X" / "X on the right": split_spatial() +
             resolve_spatial_referent() pick among the label's tracks
             (min/max x/y, nearest/farthest from gaze, nearest center)
      -> (d) "where is the X" / "find the X" with no track:
             ObjectDetector (OWL-ViT, on-demand, ~2 s CPU) scans the
             current frame; best box -> bias + "Found the X";
             >= 0.30 conf also becomes a track
      -> (e) anaphora: "look at it again" re-resolves dialogue's last
             region verbatim (a spatial pick stays picked); "the left
             one" picks among the last label's tracks (or all live
             tracks); "that gate" strips the determiner; bare "it"
             with no referent is an honest miss
      -> (f) "what do you see?": PerceptualState.describe() names objects
      -> (g) ResponsePolicy: memory + t_now + gaze on every turn;
             policy.detector + policy.frame_fn wire the detection
             fallback; LLM payload inherits objects via describe()
             AND the recognition layer (build_payload's
             recognized_objects: label, region, track confidence)
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
5. **Tracking** (`scripts/audit_tracking.py`, replays the 90
   live-run sightings): 90 sightings -> 28 tracks; gate collapses
   18 -> 2 (one 17-hit track + one stray); `locate()` returns the
   EMA-smoothed position, never the last raw sighting.
6. **Spatial references** (`scripts/audit_spatial.py`, live-run
   tracks): "look at the leftmost windows" peaks the bias at the
   left window track; "the windows on the right" at the righter of
   the two tracks (selectors are relative among tracks, not absolute
   frame halves); "the nearest gate" from a gaze near the gate hits
   the gate track; "the leftmost red car" -> honest miss, no bias.
7. **On-demand detection** (`scripts/audit_detect.py`, real OWL-ViT
   weights, CPU): with an empty memory, "wodehaus where is the
   window" on a windows crop -> 5 boxes, top 0.215 ->
   "Found the window -- looking at it center.", bias peak (27,27)
   at the box center; the 0.215 detection is below the 0.30 track
   bar, so it steers gaze without becoming a known object (a
   follow-up re-detects); "where is the red car" -> 0 boxes ->
   honest miss. Latency: ~2 s per 224px frame warm, ~11 s cold --
   on-demand only, never in the 10 Hz loop. Model:
   `google/owlvit-base-patch32` (~350 MB), same zero-shot
   philosophy as the CLIP pick. Query phrasing matters:
   "a window" detects, "window" does not -- ResponsePolicy wraps
   queries as noun phrases.
8. **Detector pick** (`hvp/detect.py`, `tests/test_recognition.py`):
   crop 10 (hand-labeled "windows") -> "a window" top detection at
   0.215. Honest negative: crop 0 (hand-labeled "gate", a dark
   slatted crop) detects "a window" instead -- the crop is genuinely
   ambiguous at 224px, and the test asserts the window case rather
   than forcing the gate.

9. **Anaphora** (`scripts/audit_anaphora.py`, live-run replay):
   "look at the windows" ; "look at it again" repeats the bias peak;
   "the left one"/"the right one" split the two window tracks; "it"
   after "the right one" stays on the right track; bare "it" with no
   prior region -> honest miss, no bias; "look at that gate" hits the
   gate track. Anaphoric forms never consult the detector -- they are
   relative among known tracks.
10. **LLM payload grounding** (`tests/test_llm.py`, 5 tests):
    `build_payload()` now includes `recognized_objects` (label,
    qualitative region, track confidence) from the live memory, and
    the system prompt's stale "no object recognition" rule is
    replaced with list-membership honesty. The model is still the
    ceiling and the rules the floor; the payload is what makes the
    model's words about something real.

## Standing limits

Determinism holds for the pipeline; CLIP's outputs are
deterministic given weights and inputs on this CPU build, but model
weights are third-party artifacts outside this repo -- version them
by `HVP_CLIP_MODEL`, not by hash. If the weights change, re-run
audit 1.
