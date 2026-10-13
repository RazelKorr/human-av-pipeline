"""Stem-stream experiment (Phase 5, 2026-10-12; audit item 14).

EXPERIMENTAL. Not wired into the flagship streaming path.

The idea: demucs splits a clip into stems (vocals / no_vocals);
each stem then runs through the pipeline's existing machinery —
hva/online.py moment features + onset picking, hva/interpret.py
two-tier transcription and CLAP labeling — and the script emits
one stream per stem: its onsets, its labels, its transcript.
That is a first auditory-object layer: instead of one mixed
stream in which engines and voices blur together, the pipeline
can say "the vocal stem said X while the other stem did Y."

Honesty constraints, same as everywhere in this project:
- Every transcript segment carries its avg_logprob (the two-tier
  machinery records them); a stem whose transcript is all
  low-logprob is reported as garbage-flagged, not quoted.
- Stems are demucs MODEL OUTPUT. Separation artifacts are real:
  a stem stream describes what demucs isolated, not ground truth
  about the original mix.
- Validation is against ground truth only where ground truth
  exists (the synthetic mix built by --make-mix, whose components
  are known exactly).

Usage:
  # build the deterministic synthetic mix + ground-truth record
  python scripts/stem_streams.py --make-mix \
      --speech separated/mdx_extra/star_tours_full_16k/vocals.wav \
      --speech-start 35 --duration 20 --out-mix input/stem_mix_synth.wav

  # analyze any media file (separates, then per-stem streams)
  python scripts/stem_streams.py --input input/stem_mix_synth.wav \
      --out output/stem_streams_synth.json

  # analyze pre-separated stems without re-running demucs
  python scripts/stem_streams.py --stems vocals.wav no_vocals.wav \
      --out output/stem_streams.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from hva.online import moment_features, pick_onsets, SPM, SR  # noqa: E402

TOP_ONSET_WINDOWS = 5     # CLAP-label this many strongest onsets
CLAP_WINDOW_S = 4.0       # seconds of audio per labeled window


# ---------------------------------------------------------------------
# Per-stem stream analysis
# ---------------------------------------------------------------------

def stream_features(audio: np.ndarray) -> tuple[list, np.ndarray]:
    """50 ms moment features + picked onsets for a 16 kHz mono stem.

    Returns (onsets, rms) where onsets = [(t_s, strength)].
    """
    n_mom = len(audio) // SPM
    if n_mom == 0:
        return [], np.zeros(0, dtype=np.float32)
    x = audio[:n_mom * SPM].reshape(n_mom, SPM)
    rms, flux, _cent, _last = moment_features(x)
    return pick_onsets(flux, rms), rms


def aggregate_labels(audio: np.ndarray, onsets: list,
                     top_k_windows: int = TOP_ONSET_WINDOWS) -> dict:
    """CLAP-label the windows around the strongest onsets and
    aggregate: mean score per label across windows, top label
    first. {} when there are no onsets or no audio."""
    from hva.interpret import label_window
    if not onsets or len(audio) == 0:
        return {}
    strongest = sorted(onsets, key=lambda o: -o[1])[:top_k_windows]
    half = int(CLAP_WINDOW_S * SR / 2)
    agg: dict[str, list[float]] = {}
    for t_s, _strength in strongest:
        c = int(t_s * SR)
        win = audio[max(0, c - half):c + half]
        for lab in label_window(win, top_k=5):
            agg.setdefault(lab["label"], []).append(lab["score"])
    return {label: round(float(np.mean(scores)), 4)
            for label, scores in
            sorted(agg.items(), key=lambda kv: -np.mean(kv[1]))}


def analyze_stem(name: str, audio: np.ndarray,
                 transcribe: bool = True) -> dict:
    """One stem -> its stream record."""
    from hva.interpret import transcribe_two_tier
    audio = np.asarray(audio, dtype=np.float32).ravel()
    onsets, _rms = stream_features(audio)
    tx = transcribe_two_tier(audio) if transcribe else None
    segments = tx["segments"] if tx else []
    n_low = sum(1 for s in segments
                if s["avg_logprob"] is not None
                and s["avg_logprob"] < -0.8)
    return {
        "stem": name,
        "duration_s": round(len(audio) / SR, 2),
        "n_onsets": len(onsets),
        "onsets": [{"t_s": round(t, 2), "strength": round(s, 1)}
                   for t, s in onsets],
        "labels_agg": aggregate_labels(audio, onsets),
        "transcript": {
            "n_segments": len(segments),
            "n_words": sum(len(s["text"].split()) for s in segments),
            "n_low_conf_segments": n_low,
            "garbage_flagged": bool(segments)
                               and n_low >= (len(segments) + 1) // 2,
            "segments": segments,
        },
    }


def analyze_mix(audio: np.ndarray) -> dict:
    """The un-separated stream, for sparsity comparison."""
    onsets, _rms = stream_features(np.asarray(audio, dtype=np.float32))
    return {"n_onsets": len(onsets),
            "onsets": [{"t_s": round(t, 2), "strength": round(s, 1)}
                       for t, s in onsets]}


# ---------------------------------------------------------------------
# Synthetic mix (deterministic ground truth)
# ---------------------------------------------------------------------

# Planted music events: (start_s, dur_s, freqs, kind)
TONE_EVENTS = [
    (2.0, 0.3, (880.0,), "burst"),
    (7.0, 0.3, (880.0,), "burst"),
    (9.0, 3.0, (220.0, 277.18, 329.63), "pad"),
    (13.0, 0.3, (660.0,), "burst"),
    (16.0, 0.5, (110.0,), "thump"),
]


def tone_sequence(duration_s: float, sr: int = SR) -> np.ndarray:
    """The planted music component: known events at known times."""
    n = int(duration_s * sr)
    out = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / sr
    for start, dur, freqs, _kind in TONE_EVENTS:
        i0, i1 = int(start * sr), int((start + dur) * sr)
        seg_t = t[i0:i1]
        env = np.ones(i1 - i0, dtype=np.float32)
        fade = int(0.01 * sr)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        wave = sum(np.sin(2 * np.pi * f * seg_t) for f in freqs)
        out[i0:i1] += (0.3 * wave * env).astype(np.float32)
    return out


def build_mix(speech: np.ndarray, duration_s: float,
              music_db: float = -6.0, sr: int = SR) -> tuple:
    """speech excerpt + tone sequence, music at music_db relative
    to the speech RMS. Returns (mix, music_component)."""
    n = int(duration_s * sr)
    speech = np.asarray(speech, dtype=np.float32)[:n]
    if len(speech) < n:
        speech = np.pad(speech, (0, n - len(speech)))
    music = tone_sequence(duration_s, sr)
    s_rms = float(np.sqrt((speech ** 2).mean())) or 1e-6
    m_rms = float(np.sqrt((music ** 2).mean())) or 1e-6
    music = music * (s_rms * 10 ** (music_db / 20.0) / m_rms)
    mix = np.clip(speech + music, -1.0, 1.0).astype(np.float32)
    return mix, music.astype(np.float32)


def _write_wav(path: str, audio: np.ndarray, sr: int = SR):
    import soundfile as sf
    sf.write(path, audio, sr)


def _load16k(path: str) -> np.ndarray:
    from hva.transcribe import load_audio
    return load_audio(path)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=
                                 argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-mix", action="store_true")
    ap.add_argument("--speech")
    ap.add_argument("--speech-start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--music-db", type=float, default=-6.0)
    ap.add_argument("--out-mix")
    ap.add_argument("--input")
    ap.add_argument("--stems", nargs=2,
                    metavar=("VOCALS", "NO_VOCALS"))
    ap.add_argument("--out")
    ap.add_argument("--no-transcribe", action="store_true")
    args = ap.parse_args()

    if args.make_mix:
        full = _load16k(args.speech)
        i0 = int(args.speech_start * SR)
        excerpt = full[i0:i0 + int(args.duration * SR)]
        mix, _music = build_mix(excerpt, args.duration, args.music_db)
        _write_wav(args.out_mix, mix)
        gt = {"mix": args.out_mix, "speech_source": args.speech,
              "speech_start_s": args.speech_start,
              "duration_s": args.duration, "music_db": args.music_db,
              "tone_events": [{"start_s": s, "dur_s": d,
                               "freqs": list(f), "kind": k}
                              for s, d, f, k in TONE_EVENTS]}
        gt_path = os.path.splitext(args.out_mix)[0] + "_groundtruth.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=1)
        print(f"mix -> {args.out_mix}; ground truth -> {gt_path}")
        return

    record: dict = {"experiment": "stem-streams-v1 (Phase 5, item 14)"}
    if args.input:
        from hva.transcribe import separate_vocals
        mix_audio = _load16k(args.input)
        record["input"] = args.input
        record["mix_stream"] = analyze_mix(mix_audio)
        vocals_path = separate_vocals(args.input)
        no_vocals_path = os.path.join(os.path.dirname(vocals_path),
                                      "no_vocals.wav")
        stem_paths = {"vocals": vocals_path,
                      "no_vocals": no_vocals_path}
    elif args.stems:
        stem_paths = {"vocals": args.stems[0],
                      "no_vocals": args.stems[1]}
        record["stems_only"] = True
    else:
        ap.error("one of --make-mix, --input, --stems is required")

    record["stem_streams"] = {}
    for name, path in stem_paths.items():
        print(f"analyzing stem {name}: {path}", flush=True)
        record["stem_streams"][name] = analyze_stem(
            name, _load16k(path),
            transcribe=not args.no_transcribe)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=1)
        print(f"-> {args.out}")
    else:
        print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
