"""Demo: the "saying hi" loop, end to end.

  1. Loads a hail audio file (16k mono WAV).
  2. Runs the real RollingTranscriber over it (windowed, like the stream).
  3. TurnDetector watches for the name + end-of-utterance.
  4. ResponsePolicy generates a reply.
  5. speak() synthesizes the reply via the tts CLI.

Usage:
  python scripts/demo_say_hi.py /tmp/sayhi/hail_16k.wav --outdir /tmp/sayhi

The output is the response audio file. On a machine with speakers (or a
virtual audio cable into Zoom), that file is what gets played.
"""
import argparse
import os
import wave

import numpy as np

from hva.stream import RollingTranscriber, AUDIO_SR
from hva.conversation import TurnDetector, ResponsePolicy, speak


def load_mono16k(path: str) -> np.ndarray:
    w = wave.open(path)
    assert w.getframerate() == 16000 and w.getnchannels() == 1, \
        "need 16k mono WAV"
    return (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            .astype(np.float32) / 32768.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hail_wav", help="16k mono WAV of someone hailing the system")
    ap.add_argument("--outdir", default="/tmp/sayhi")
    ap.add_argument("--model-size", default="base")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    audio = load_mono16k(args.hail_wav)
    dur_s = len(audio) / AUDIO_SR
    print(f"hail: {dur_s:.1f}s, {len(audio)} samples")

    # 1-2. Stream the audio through the rolling transcriber in 100 ms ticks,
    # exactly like the live loop does.
    tx = RollingTranscriber(model_size=args.model_size,
                            window_s=30.0, step_s=10.0)
    tick_n = int(AUDIO_SR * 0.1)
    detector = TurnDetector(silence_s=1.5)
    policy = ResponsePolicy()
    turns = []
    t_ms = 0.0
    for start in range(0, len(audio), tick_n):
        chunk = audio[start:start + tick_n]
        if len(chunk) < tick_n:
            chunk = np.pad(chunk, (0, tick_n - len(chunk)))
        tx.push(t_ms, chunk)
        t_ms += 100.0
        # The detector sees only what has been transcribed SO FAR.
        turns += detector.update(tx.transcript(), t_ms / 1000.0)

    # Finalize: force one last transcription of the tail, then let the
    # utterance end. (The live loop does this continuously; the demo's
    # clip is shorter than one window step.)
    tx._transcribe(dur_s)
    # Walk the detector forward in 100 ms steps until the silence rule fires.
    now = dur_s
    for _ in range(50):
        now += 0.1
        new = detector.update(tx.transcript(), now)
        turns += new
        if new:
            break

    print(f"transcript segments: {len(tx.transcript())}")
    for s in tx.transcript():
        print(f"  {s['start']:.1f}-{s['end']:.1f}s: {s['text']}")
    print(f"turns detected: {len(turns)}")
    for t in turns:
        print(f"  {t}")

    if not turns:
        print("no turn detected -- nobody hailed the system (or the name "
              "wasn't recognized).")
        return

    # 4-5. Respond and speak.
    for i, turn in enumerate(turns):
        reply = policy.generate(turn)
        print(f"response {i}: {reply}")
        if reply is None:
            continue
        out = os.path.join(args.outdir, f"response_{i}.mp3")
        speak(reply, out)
        print(f"  spoken -> {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
