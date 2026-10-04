# Sensorium Review Protocol

Adopted 2026-10-03, after the turn-direction miss: two consecutive Star Tours reviews filed "the wrong turn goes to the RIGHT" as a flagship finding, when the narrative first turn (~30–32s, into the maintenance bay) is a LEFT yaw. The 36.0–38.5s strip the reviews measured was a later, separate rightward maneuver inside the bay. The direction reading was correct; the *identification* was wrong — the review named an event from its appearance instead of anchoring it to the story. Mykal caught it by re-watching. This protocol exists so the pipeline catches it first.

Core principle: **beats before events.**

## 1. Beat map first

Before any event pass, build the film's narrative chapter structure — the story's own beats (takeoff, wrong turn, maintenance bay, hyperspace, ice field, battle, trench, docking). Sources, in order:

- The transcript: dialogue marks beats ("wrong way... brakes!", tone shifts, scene-setting lines).
- The audio track: score changes, silence, dialogue density — the film tells you where chapters break.
- Visual evidence: major palette/location shifts, and especially **diegetic signage** — legible in-world labels ("MAINTENANCE BAY NO ADMITTANCE", "LAUNCH") are the film naming its own beats. Read them.

File as `beat-map.md` in the run's output folder, before the review is drafted. A beat map is a falsifiable document: each beat carries its timestamp span and the evidence that names it.

## 2. Map events onto beats, never the reverse

Detected events are assigned to existing beats. An event that fits no beat is flagged **unidentified** — it does not get to borrow a story name because it looks the part. This is the step that would have caught the miss: with the maintenance-bay beat anchored at ~30–34s by its own signage, the 36–38.5s strip could only ever have been "maneuver inside the maintenance bay," never "the turn."

## 3. Claims cite three things

Every interpretive claim in the review cites: (a) the timestamp strip, (b) the visual evidence, and (c) the beat anchor. The old reviews had (a) and (b). (c) is mandatory. A claim without a beat anchor is a draft, not a finding.

## 4. Red-team pass

Ported from the Module Standard. After the review draft, a dedicated adversarial re-read whose explicit job is to break the claims — above all the named beats. "Is this actually the turn?" is somebody's assigned question, asked on purpose, every time.

## 5. Inheritance rule

A claim carried forward from a prior review is re-verified against the new run's own evidence, or it is marked **inherited-unverified**. Flagship findings do not coast on previous watches.

## Mykal's role: appeals, not trial

The beat map is built autonomously — he does not rewatch everything sent. He is the appeals court: he corrects the map when it's wrong (as on 2026-10-03), or not at all. The protocol must work without him; it works better with him.

## 6. Beat-source tooling status (2026-10-03 audit)

What the pipeline can actually supply for each beat source today:

- **Transcript — automated, works on arbitrary inputs.** `scripts/transcribe.py` (thin wrapper over `python3 -m hva.transcribe`) accepts any audio or video file (`--wav in.mp4 --out transcript.json`): PyAV decodes it and resamples to 16k mono in code. Requires the pipeline venv (`workspace/.venv-pipeline`, faster-whisper installed); model weights live in `models/faster-whisper-base` (+ `models/faster-whisper-medium`). Also supports `--separate-vocals`. One command, no manual step.
- **Audio track — wired into the streaming runner (2026-10-03).**
  `run_stream.py` extracts the audio track by default (`--audio`,
  `--no-audio` to disable) and writes `audio_features.npy`
  (per-50 ms-moment RMS, spectral flux, spectral centroid),
  `audio_events.json` (spectral-flux onset events on the master
  timeline), and `av_binding.json` (audio<->visual transient
  coincidence pairs, ±250 ms). Attended transcription:
  `--transcribe-onsets K` transcribes ±4 s slices around the top-K
  onsets (vad=False; the VAD eats real speech on short slices).
  Score changes, silence, and dialogue density are now first-class
  beat sources, not manual steps. See STREAMING-REPORT.md (audio
  addendum) for the sealed-prediction validation.
- **Diegetic signage — human-read step today.** There is no OCR capability anywhere in the codebase. Reading in-world labels ("MAINTENANCE BAY NO ADMITTANCE", "LAUNCH") is done by the reviewer from the percept video or frame-indexed bundles (`scripts/build_frame_index.py`), with timestamps, and cited as visual evidence per §3. If sign-reading ever becomes automatable, it gets a beat source of its own here.

---

*Standing: applies to every Sensorium media review (Star Tours, FF3, and whatever comes next). The review format (header, review, what-the-eyes-caught, evidence, errata) is unchanged; this protocol governs how claims earn their place in it.*
