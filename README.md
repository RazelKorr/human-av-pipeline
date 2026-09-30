# Human Audio Pipeline

An experimental model of human auditory attention: waveform in, a 10 Hz
stream of *attended* perceptual moments out. Built as a companion to a
human-vision pipeline, with cross-modal (audio-visual) coupling.

This is a research demo, not a validated model of hearing or
consciousness. Every constant is provisional and every mechanism is
meant to be argued with.

## Architecture

```
waveform -> cochlea -> salience -> moments -> attention -> trace
                |                                  |
           spatial.py                         vis_boost (Level 2)
           (stereo pan)                        (from vision)
```

- **hva/cochlea.py** -- 16 kHz STFT (20 ms Hann, 5 ms hop), 64
  log-spaced bins 50 Hz-8 kHz, absolute dBFS. The cochlea does a lot of
  the work for free; the hard part is deciding what belongs to what.
- **hva/salience.py** -- intensity + frequency-contrast + temporal-
  contrast (onset) channels, fused into a salience map. A -60 dBFS
  hearing floor keeps near-silence from hallucinating onsets.
- **hva/moments.py** -- 100 ms perceptual moments (20 frames each):
  mean spectrum, mean salience, loudness, max short-term change.
  The 100 ms grain is shared with the vision pipeline.
- **hva/attention.py** -- the controller. Two paths:
  - *Express interrupt*: a sharp transient (above the 95th percentile
    of recent change, floored at 6 dB) yanks focus in ~50 ms, with a
    100 ms attentional blink afterward and habituation to repetition.
  - *Scheduled*: at most every 200 ms, focus reconsiders; switching
    costs 150 ms latency, with distance cost and inhibition of return.
  - In quiet it dwells rather than chasing noise.
- **hva/spatial.py** -- binaural cues from stereo: per-moment ILD pan
  plus ITD refinement around onsets. *Where* the sound is.
- **hva/battery.py** -- synthetic validation: abrupt-event latency,
  unattended vs attended change, cocktail-party streaming, onset
  capture, post-switch refractory. 6/6 passing.

## Levels of audio-visual fusion

- **Level 1** (`scripts/fuse_av.py`): synchronized 10 Hz joint feed --
  gaze + audio focus + both event streams on one media timeline.
  Visual content lags audio by ~2 moments (ears are faster than eyes);
  the feed pairs by media time and documents the lag.
- **Level 2** (`scripts/couple_av.py`): bidirectional coupling. Audio
  onsets tug saccades toward the panned side (+alerting gain); visual
  transients lower the auditory capture bar (top quartile only).
- **Level 3**: joint priority map (planned).

## Running it

```bash
pip install -r requirements.txt
# Put a 16 kHz mono wav at input/ and adapt scripts/run_star_tours.py,
# or run the synthetic battery:
python3 -m hva.battery
```

`input/` and `output/` are gitignored -- bring your own media
(the original runs used a Star Tours ride-film soundtrack).

## Status

Experimental. Constants are provisional, the battery is synthetic, and
auditory streaming (which frequencies belong to which source) is the
big open problem. See SPEC.md for the full design notes and caveats.
