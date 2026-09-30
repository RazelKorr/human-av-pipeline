"""Speech transcription channel for the audio pipeline.

v1: verbatim word/segment transcription via a local whisper model
(faster-whisper), aligned to the 10 Hz perceptual-moment grid. This
gives the audio attention system access to *linguistic* content --
speech onsets, word boundaries, speaker turns -- instead of anonymous
spectral transients. The whole point of speech, as the user put it.

Honest limits (v1):
- Verbatim transcription is superhuman: no human catches every word.
  Per-segment confidence (avg_logprob) and per-word probabilities are
  recorded as the hook for a future mishearing model.
- "Perfect" hearing still misinterprets: transcription is not
  comprehension. Word sense, sarcasm, irony, reference resolution are
  not modeled here. That is a later layer, explicitly deferred.
- No speaker diarization in v1. Speaker *turns* are approximated from
  segment boundaries plus inter-segment gaps, not voice identity.
  Two alternating voices will read as turns; one voice pausing will
  too. Documented, not hidden.

Public API:
    transcribe(wav_path, model_size="base") -> list of segments
    align_to_moments(segments, n_moments, moment_s=0.1) -> per-moment info
    speech_events(segments, min_gap_s=0.4) -> speech-onset event times

CLI:
    python3 -m hva.transcribe --wav in.wav --out transcript.json
"""

from __future__ import annotations

import argparse
import json
import os
import wave

import numpy as np

MOMENT_S = 0.1
DEFAULT_MODEL = "base"
# Local model dirs live here (gitignored; too large to commit). A model
# "size" that is a path to a directory is used directly, which also
# sidesteps environments where the HF download client can't parse the
# egress proxy config.
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "models")


def _load_model(model_size: str):
    from faster_whisper import WhisperModel
    if os.path.isdir(model_size):
        path = model_size
    else:
        local = os.path.join(MODEL_DIR, f"faster-whisper-{model_size}")
        path = local if os.path.isdir(local) else model_size
    # CPU is fine for clips; int8 keeps it light.
    return WhisperModel(path, device="cpu", compute_type="int8")


def load_wav_mono(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load a wav as float32 mono at target_sr.

    Pure stdlib + numpy (no av dependency). Handles 8/16/32-bit PCM and
    any channel count; resamples by linear interpolation when needed.
    """
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sampwidth == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sampwidth}")
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    if sr != target_sr:
        t_old = np.arange(len(audio)) / sr
        t_new = np.arange(int(len(audio) * target_sr / sr)) / target_sr
        audio = np.interp(t_new, t_old, audio).astype(np.float32)
    return audio


def transcribe(wav_path: str, model_size: str = DEFAULT_MODEL,
               vad: bool = True) -> list[dict]:
    """Transcribe a wav file.

    Returns segments: {start, end, text, avg_logprob, words}.
    words: [{start, end, word, prob}]. Times in seconds.
    """
    model = _load_model(model_size)
    audio = load_wav_mono(wav_path)
    segments, _info = model.transcribe(
        audio,
        vad_filter=vad,
        word_timestamps=True,
    )
    out = []
    for s in segments:
        words = []
        for w in (s.words or []):
            words.append({
                "start": round(w.start, 3),
                "end": round(w.end, 3),
                "word": w.word,
                "prob": round(float(w.probability), 3),
            })
        out.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": s.text.strip(),
            "avg_logprob": round(float(s.avg_logprob), 3),
            "words": words,
        })
    return out


def align_to_moments(segments: list[dict], n_moments: int,
                     moment_s: float = MOMENT_S) -> list[dict]:
    """Map segments onto the perceptual-moment grid.

    Per moment: {speech: bool, words: [str], text: str}.
    A word belongs to every moment its [start, end) span touches.
    """
    moms: list[dict] = [
        {"speech": False, "words": [], "text": ""} for _ in range(n_moments)
    ]
    for s in segments:
        for w in s["words"]:
            m0 = int(w["start"] / moment_s)
            m1 = int(max(w["end"] - 1e-6, w["start"]) / moment_s)
            for m in range(max(m0, 0), min(m1, n_moments - 1) + 1):
                moms[m]["speech"] = True
                moms[m]["words"].append(w["word"])
    for m in moms:
        if m["words"]:
            m["text"] = "".join(m["words"]).strip()
    return moms


def speech_presence(segments: list[dict], n_moments: int,
                    moment_s: float = MOMENT_S,
                    hangover_ms: float = 300.0) -> np.ndarray:
    """Per-moment speech presence in [0,1] for the attention system.

    Binary from word spans (via align_to_moments), then a hangover:
    presence decays exponentially after speech ends instead of
    chattering the auditory gain off between words. hangover_ms=300
    matches the map's TAU_MS -- the gain release and the map's memory
    run on the same clock, which is a guess, documented as one.
    """
    moms = align_to_moments(segments, n_moments, moment_s)
    binary = np.array([1.0 if m["speech"] else 0.0 for m in moms],
                      dtype=np.float32)
    decay = float(np.exp(-(moment_s * 1000.0) / hangover_ms))
    out = np.zeros_like(binary)
    carry = 0.0
    for i, b in enumerate(binary):
        carry = b if b > carry * decay else carry * decay
        out[i] = carry
    return out


def speech_events(segments: list[dict],
                  min_gap_s: float = 0.4) -> list[dict]:
    """Speech-onset events: a segment starting after >= min_gap_s of
    silence. These are the linguistic onsets the attention system can
    treat as events, alongside acoustic captures.

    Returns [{t, text, gap_before}].
    """
    events = []
    prev_end = 0.0
    for s in segments:
        gap = s["start"] - prev_end
        if gap >= min_gap_s or not events:
            events.append({
                "t": s["start"],
                "text": s["text"],
                "gap_before": round(gap, 2),
            })
        prev_end = max(prev_end, s["end"])
    return events


def _slice_wav(src: str, dst: str, seconds: float) -> None:
    with wave.open(src, "rb") as w:
        n = min(int(w.getframerate() * seconds), w.getnframes())
        params = w.getparams()
        frames = w.readframes(n)
    with wave.open(dst, "wb") as w:
        w.setparams(params)
        w.writeframes(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe speech to the "
                                 "perceptual-moment grid.")
    ap.add_argument("--wav", required=True, help="input wav (16k mono ideal)")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="whisper size: tiny/base/small")
    ap.add_argument("--seconds", type=float, default=0,
                    help="only transcribe the first N seconds (0 = all)")
    ap.add_argument("--moment", type=float, default=MOMENT_S)
    args = ap.parse_args()

    wav = args.wav
    if args.seconds and args.seconds > 0:
        wav = args.out + ".clip.wav"
        _slice_wav(args.wav, wav, args.seconds)

    segments = transcribe(wav, model_size=args.model)
    duration = max((s["end"] for s in segments), default=0.0)
    n_moments = int(duration / args.moment) + 1
    moments = align_to_moments(segments, n_moments, args.moment)
    events = speech_events(segments)

    payload = {
        "wav": args.wav,
        "model": args.model,
        "n_segments": len(segments),
        "n_speech_events": len(events),
        "frac_moments_with_speech": round(
            sum(1 for m in moments if m["speech"]) / max(len(moments), 1), 3),
        "segments": segments,
        "speech_events": events,
        # moments omitted from the json by default (large); rebuild with
        # align_to_moments() when needed.
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"transcribed {len(segments)} segments, "
          f"{len(events)} speech events -> {args.out}")


if __name__ == "__main__":
    main()
