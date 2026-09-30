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
- **Level 3**: joint priority map (planned).

**`scripts/watch.py`** ties all three together: feed it a video and it
watches it the way a human would -- vision pipeline on the frames,
audio pipeline on the soundtrack, L1 fusion on one timeline -- then
writes a plain-language perceptual review in a fraction of the clip's
runtime. Sixty-two seconds of video, reviewed in seventeen. The
bottlenecks (10 Hz moments, a fovea, an attentional blink) are the
point: it reports what a human would have perceived, which is less
than what's there, and that's what makes the review human-shaped.
**`scripts/perceive.py`** does the same for still images.

## Running it

```bash
pip install -r requirements.txt
python3 scripts/demo_vision.py      # vision validation battery
python3 -m hva.battery              # audio validation battery (from repo root)
# Fusion needs a video + its audio side by side; see scripts/fuse_av.py
# --help. input/ and output/ are gitignored: bring your own media.
```

## Status

Experimental. The constants are provisional, the batteries are
synthetic, and auditory streaming (which frequencies belong to which
source) is the big open problem on the audio side. The fusion is an
alignment experiment, not a validated multisensory-binding model.

## Provenance

Entirely AI-generated -- vibecoded, as it were. Every line of code was
written by Wodehaus, an AI agent (Muse, built by Meta), from RazelKorr
RazelKorr's broad prompts and direction in September 2026. No code here
was hand-written by a human; the human role was directing the work,
testing the outputs, and making the design calls (the top-quartile gate
on the vision->audio boost, the 100 ms shared moment grain, the call to
publish it at all). Read it accordingly: it runs, it passes its own
batteries, and it has not been through human code review.
