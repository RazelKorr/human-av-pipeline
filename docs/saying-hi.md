# Saying hi, and understanding — as-built (2026-09-30)

The pipeline can now hear its name, answer, and — short of a full
language model — comprehend: grounding words in its own perceptual
state.

## The "saying hi" loop

  hail audio -> RollingTranscriber -> TurnDetector -> ResponsePolicy
      -> tts speak -> response audio file

Demo: `python scripts/demo_say_hi.py hail_16k.wav --outdir /tmp/sayhi`

Verified 2026-09-30: a TTS-synthesized "Hi Wodehaus, can you hear me?"
(1.9 s) was transcribed, detected, answered, and spoken back as an
8.16 s MP3. Whisper heard the name as "I would house" — the fuzzy
name matcher caught it.

## Comprehension (`hva/understanding.py`)

Thesis: for this system, comprehension means grounding words in its
perceptual state — not generating text about seeing, but consulting
the map.

- **Intent** (rule-based): greeting, farewell, identity, see-question,
  hear-question, look-command, look-at-thing, thanks, yes/no, wh-,
  statement.
- **PerceptualState**: read-only view over a live OnlineLevel3 — gaze
  location, recent saccades, map peak, plain-language `describe()`.
- **Language → perception**: "look left" becomes a Gaussian bias blob
  on the 56×56 joint map via the `task_bias` channel (added to
  `JointPriorityMap.step()` and `OnlineLevel3.tick()`). The saccade
  controller reads the combined map, so language steers gaze through
  the same machinery the senses use. A command is a nudge, not a
  clamp — the blob decays with the map.
- **DialogueState**: turn history (last 10), last-mentioned region.
- **Honest limits** (updated 2026-09-30): "look at the red car" ->
  "I don't know what a red car looks like yet" when nothing matching
  was recognized; but "look at the X" now resolves X against the
  ObjectMemory fed by the foveal CLIP classifier -- directions AND
  recognized things both steer the bias channel. See
  `docs/recognition.md`.

Demo: `python scripts/demo_understand.py`

Verified 2026-09-30: with a salient blob on the right, gaze sat at
(174,114). "Wodehaus look left" moved it to (62,110) for the 3 s the
bias was held, then it returned. "What do you see?" answered from the
live loop: "I'm looking at the right of the frame... I've made 33
saccades so far."

## Components (`hva/conversation.py`)

**TurnDetector**: watches the rolling transcript. Fires when the
system's name appears in new words AND no word has ended within
`silence_s` (default 1.5 s). Returns the addressed span as a Turn.
Two evidence-based guards, no phrase blacklists: **acoustic
agreement** — when given a `speech(t0, t1)` callable (fraction of
the span with acoustic speech energy, wired from per-tick
`EnergyVAD.hot` history in the runner), the addressed span must
clear `min_speech_frac` (default 0.2), suppressing Whisper
hallucinations over silence as `no_speech`; and **re-fire
suppression** — a turn whose span overlaps the previous fired
turn's span (+`refire_margin_s`, default 1.0 s) is suppressed as
`refire`, defeating duplicate turns from overlapping transcription
windows re-emitting the same utterance. Suppressed turns are
recorded on `detector.suppressed` as `(reason, text)`.
Only reasons over transcribed words — it never claims to have heard
what the transcriber hasn't produced yet.

**ResponsePolicy** (v3): grounded via `hva.understanding`. Attach a
live loop with `policy.perceptual = PerceptualState(loop)`; "look"
biases land on `policy.pending_bias` for the tick loop to collect with
`take_bias()`. `generate(turn)` is the LLM seam: attach `ApiGenerator`
(Anthropic API, needs `ANTHROPIC_API_KEY`), `HFGenerator` (HuggingFace
Inference Providers, needs `HF_TOKEN`, free tier, default model
`Qwen/Qwen3-8B`), `PollinationsGenerator` (Pollinations.ai OpenAI-compatible endpoint:
keyless JSON POST to `https://text.pollinations.ai/openai` — no
account, no key, rate-limited by IP. Dialogue and perceptual
snapshots travel in the request body, but it is still a
third-party/public service with no SLA and no model guarantee, so
non-sensitive prototyping only), or `LocalGenerator` (llama-server
at `--llm-url`) via `--llm {none,api,local,hf,free,auto}`
(`--hf-model` / `--free-model` override the models). `auto` tries
local, then HF, then API, then the keyless free backend — free before
paid, your own credentials before a public gateway.
A trailing `LOOK: <direction>` line in the LLM reply is stripped before
speaking and converted to a task bias. No backend or API failure falls
back to the rule-based `understand()`.

**speak()**: synthesizes via `/opt/hatch/bin/tts speak` (default voice
avocado_v2:MAI_03). Writes MP3 to a caller-chosen path. `AsyncTTS`
synthesizes on a background thread so the perceptual tick loop never
blocks; if the user speaks during synthesis, the pending response is
dropped as stale.

## Known limits

- **No mic capture yet.** The loop eats files/URLs as live; genuine
  microphone or virtual-source input is desk work (VM has no `/dev/snd`).
- **No echo cancellation.** Live mic + live speaker will hear itself;
  needs handling or explicit routing assumptions.
- **No deep understanding.** Intents are patterns, not semantics. The
  module reports what it did ("I looked left because you said left"),
  it does not pretend to grasp meaning.
- **No output device on this VM** (`/dev/snd` doesn't exist). Output is
  a file. On a real host, play it through speakers or route it to the
  call.
- **The acoustic gate hears energy, not identity.** The phantom-turn
  defense rejects transcript claims over digital silence, but any
  speech-energy in the addressed span passes the gate — far-end call
  audio or a TV in the room counts as "acoustic speech". Speaker
  separation is still unsolved.

## Zoom plumbing (for a real machine)

Hearing the call:
  Zoom audio output -> virtual audio cable (VB-Cable, BlackHole, or
  PulseAudio null sink) -> pipeline's rolling buffer input.
  `hva/stream.py`'s StreamSource already reads from ffmpeg-readable
  sources; a PulseAudio monitor source works the same way.

Being heard:
  response MP3 -> play through a virtual microphone device
  (PulseAudio `module-virtual-source`, VB-Cable input) -> selected as
  Zoom's microphone.

Latency budget (measured 2026-09-30):
  - Perception (10 Hz ticks): real-time, 0.36x compute.
  - Transcription: 30 s window / 10 s step; the gate trails reality by
    ~10-30 s. A turn is detected ~1.5 s after the transcriber produces
    the final words, not after the human stops speaking.
  - TTS: ~2 s for a short response.
  - Realistic hail-to-reply: 15-35 s. Honest, not snappy.

## Tests

`tests/test_conversation.py`: 19 tests — name variants (including the
observed "would house" mishearing), turn firing on name+silence, no
double-fire, unaddressed speech ignored, greeting/question/silence
policy branches, plus 2 AsyncTTS tests (ticks continue during slow
synthesis; synthesis failure doesn't kill the loop), plus 3 LLM-path
tests: the LOOK backstop steers from classified intent when the model
drops the LOOK line, an explicit LOOK line still wins, and ordinary
replies leave the shared map alone. Phantom-turn defense: hallucination
over digital silence suppressed (`no_speech`), real acoustically
supported speech passes, window re-fires suppressed (`refire`) while a
genuine later address still fires, per-tick `EnergyVAD.hot` presence,
and a runner-level regression test proving the trailing audio chunk
reaches the loop (the old flush bug: 105 moments instead of ~138).
`tests/test_understanding.py`: 34 tests — intent classification,
direction extraction, bias targeting, perceptual description, dialogue
history, and the end-to-end language→map→peak proof. All pass.

## Live duplex + barge-in (2026-09-30)

`scripts/run_conversation.py` wires the pieces into one live loop:
stream -> transcribe -> turn-detect -> grounded respond -> TTS -> Speaker.
States: LISTENING -> (turn) -> RESPONDING -> (finished | barged in) -> LISTENING.

**Barge-in:** while the Speaker is playing, an EnergyVAD watches the
incoming mic at 100 ms resolution (adaptive noise floor, 300 ms hangover).
Speech onset during playback stops the response immediately and returns
to listening. The transcript is too slow for this (10-30 s lag); the VAD
is the fast path. Assumes playback doesn't leak into the mic (headphones /
virtual routing) -- echo cancellation is out of scope.

Verified 2026-09-30 on a constructed 15 s scenario:
hail ("Hi Wodehaus, can you hear me?") -> turn at 3.0 s -> response
speaking -> interruption ("Wodehaus stop talking, never mind.") at 4.0 s
-> **BARGE-IN at 4.3 s**, playback stopped -> interruption transcribed as
a new turn at 7.3 s -> new response. Log: 2 turns, 1 barge-in.

Known limits: TTS synthesis runs in a background thread (AsyncTTS) so the
10 Hz tick loop never stalls; the Speaker simulates playback
timing on this VM (no audio device) -- a real player plugs in via play_fn/
stop_fn hooks. If the user speaks while a response is still synthesizing,
the stale response is dropped instead of played.

**Phantom turns on long silence (observed 2026-09-30):** Whisper can
hallucinate speech -- prompt-colored toward the system's name -- on
several seconds of near-silence, and the turn detector will fire on the
phantom (name + 1.5 s silence both check out). The reply stays honest
because it is grounded in the perceptual snapshot, but a phantom look
command still steers the shared map. This was observed on a synthetic
file with 6 s of digital zeros; real room tone makes it rarer, but it is
a known model-level behavior, not a stitching bug (overlapping windows
dedupe by segment-start key). No heuristic filter yet -- the honest
grounding keeps the failure benign.
