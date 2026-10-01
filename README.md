# Human AV Pipeline

Two experimental models of human perception -- one for hearing, one for
vision -- plus the machinery that fuses them into a single synchronized
feed. Waveform in, video in; a 10 Hz stream of *attended* perceptual
moments out.

This is a research demo, not a validated model of perception or
consciousness. Every constant is provisional and every mechanism is
meant to be argued with.

## The two systems

**`hva/` -- the ears.** Cochlea (16 kHz STFT, 64 log bins 50 Hz-8 kHz,
absolute dBFS) -> salience (intensity + frequency-contrast + onset
channels, -60 dBFS hearing floor) -> 100 ms perceptual moments ->
attention controller. Two paths: an express interrupt (sharp transient
yanks focus in ~50 ms, with a 100 ms attentional blink and habituation
to repetition) and a scheduled path (reconsiders at most every 200 ms,
150 ms switch latency, distance cost, inhibition of return). Quiet
dwells. `hva/spatial.py` adds binaural pan (ILD + ITD) from stereo.
Synthetic validation battery: 6/6 passing. See SPEC_AUDIO.md.

**`hvp/` -- the eyes.** Temporal integrator -> saccade controller ->
foveated retina -> 10 Hz perceptual moments. Brightness and change
capture initial gaze; attention remembers points of interest and
searches around them; foveal inspection feeds recognition, which
redirects the next saccades. Validation battery: latency, flicker
fusion, wagon-wheel reversal, saccadic suppression, change blindness --
5/5. See SPEC_VISION.md.

## Fusion (scripts/)

- **Level 1** (`fuse_av.py`): one joint 10 Hz feed -- gaze + audio focus
  + both event streams on a shared media timeline. Visual content lags
  audio by ~2 moments (ears are faster than eyes); the feed pairs by
  media time and documents the lag.
- **Level 2** (`couple_av.py`): bidirectional coupling. Audio onsets tug
  saccades toward the panned side (+alerting gain); visual transients
  lower the auditory capture bar (top quartile only -- the film's
  background wiggling doesn't get a vote).
- **Level 3** (`hvm/`, `scripts/run_level3.py`): joint priority map --
  one shared 56x56 landscape both systems write to and read from.
  Saccades land from the joint map; auditory attention reads spatial
  gains back. Battery 5/5; real-media demo included. See SPEC_FUSION.md.

**`scripts/watch.py`** ties all three together: feed it a video and it
watches it the way the model predicts a human would -- vision pipeline on the frames,
audio pipeline on the soundtrack, L1 fusion on one timeline -- then
writes a plain-language perceptual review in a fraction of the clip's
runtime. Sixty-two seconds of video, reviewed in seventeen. The
bottlenecks (10 Hz moments, a fovea, an attentional blink) are the
point: it reports what the model predicts a human would have perceived, which is less
than what's there, and that's what makes the review human-shaped.
**`scripts/perceive.py`** does the same for still images.

**`hva/transcribe.py` -- the speech channel.** A local whisper model
(faster-whisper, base) transcribes the 16 kHz mono track to
timestamped segments with per-word times and confidence, aligned to
the 100 ms moment grid. Speech onsets become first-class attention
events instead of anonymous transients -- the pipeline can now hear
*words*, which is the whole point of speech. v1 is verbatim
transcription (superhuman: no human catches every word); confidence
scores are recorded as the hook for a future mishearing model, and
transcription-is-not-comprehension is explicitly deferred. Requires
the pipeline venv (`../.venv-pipeline`, relative to the repo root) --
system Pythons managed by the OS fight the install. Model weights live in
`models/faster-whisper-base` (gitignored, downloaded once).

**Speech-gated attention.** The transcript feeds back into the joint
map: `scripts/run_level3.py --transcript` scales the auditory map
write by smoothed speech presence, doubling audition's vote while
someone is talking (`SPEECH_BOOST=1.0`, `hvm/priority.py`). The shared
map's normalization does the attenuating -- when the ears get louder,
everything else gets relatively quieter. Battery M6: speech flips an
equal-strength flash/click conflict that vision otherwise wins.

## Conversation (scripts/)

The pipeline talks back. `scripts/run_conversation.py` runs the live
loop: `TurnDetector` (name + end-of-utterance from the rolling
transcript, fuzzy on Whisper's mishearings, guarded against
hallucinated turns by acoustic agreement and re-fire suppression)
-> `ResponsePolicy` v3 (grounded in the live perceptual loop via
`hva.understanding`, LLM seam via `hva.llm`, rule-based fallback)
-> `AsyncTTS` (non-blocking synthesis so the 10 Hz tick never stalls).
Live duplex with `EnergyVAD` barge-in: if you speak mid-response,
playback stops and the turn is re-taken. `--llm
{none,api,local,hf,free,agent,auto}` selects the language-model backend
(default `none`; `agent` is the file-handoff loop where an operator --
human or AI -- answers each turn; `auto` tries local llama-server, then
HuggingFace (free tier) if `HF_TOKEN` is set, then Anthropic if
`ANTHROPIC_API_KEY` is set, then the keyless Pollinations POST
endpoint). `--owl` opts into OWL-ViT open-vocabulary detection as the
last-resort answer to "where is the X" for never-seen objects (full-res
color frame on demand, model pre-loaded at startup). See
`docs/saying-hi.md` and `docs/real-scene-grounding.md`.

## Running it

```bash
pip install -r requirements.txt
python3 scripts/demo_vision.py      # vision validation battery
python3 -m hva.battery              # audio validation battery (from repo root)
# Fusion needs a video + its audio side by side; see scripts/fuse_av.py
# --help. input/ and output/ are gitignored: bring your own media.
```

Speech transcription needs the venv (faster-whisper fights
OS-managed system Pythons):

```bash
python3 -m venv ../.venv-pipeline
../.venv-pipeline/bin/pip install faster-whisper numpy
# model weights: see hva/transcribe.py MODEL_DIR (models/faster-whisper-base,
# gitignored -- download once, ~150 MB)
../.venv-pipeline/bin/python scripts/transcribe.py \
    --wav input/clip_16k.wav --out output/transcripts/clip.json

## Status

Experimental. The constants are provisional, the batteries are
synthetic, and auditory streaming (which frequencies belong to which
source) is the big open problem on the audio side. The fusion is an
alignment experiment, not a validated multisensory-binding model.

## Provenance

Entirely AI-generated -- vibecoded, as it were. Every line of code was
written by Wodehaus, an AI agent (Muse, built by Meta), from RazelKorr's
broad prompts and direction in September 2026. No code here
was hand-written by a human; the human role was directing the work,
testing the outputs, and making the design calls (the top-quartile gate
on the vision->audio boost, the 100 ms shared moment grain, the call to
publish it at all). Read it accordingly: it runs, it passes its own
batteries, and it has not been through human code review.
