"""Live duplex conversation: the stream watches, listens, and answers.

One OnlineLevel3 loop runs the joint priority map tick by tick. A
rolling transcriber feeds a TurnDetector; completed turns go to the
ResponsePolicy (grounded in the live loop via PerceptualState); replies
are synthesized with TTS and "played" through a Speaker.

Barge-in: while the Speaker is playing, a fast energy VAD watches the
incoming mic audio. Speech onset during playback stops the response
immediately and returns to listening -- the interruption is logged, and
whatever the user said gets transcribed and handled as a new turn.

States: LISTENING -> (turn) -> RESPONDING -> (finished | barged in)
        -> LISTENING.

Usage:
  python scripts/run_conversation.py --src input/foo.mp4 --seconds 60
  python scripts/run_conversation.py --src rtmp://host/live/key --realtime

This is file/URL-as-live. A real call swaps --src for the virtual
audio cable (see docs/saying-hi.md). TTS synthesis blocks the tick
loop in v1 -- noted as a limitation, not hidden.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from hvp import baseline as B
from hvm.online import OnlineLevel3
from hva.stream import (StreamSource, VisionFrontEnd, AudioFrontEnd,
                        RollingTranscriber)
from hva.conversation import (TurnDetector, ResponsePolicy, Speaker,
                              EnergyVAD, speak)
from hva.understanding import PerceptualState


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--outdir", default="output/conversation")
    ap.add_argument("--model-size", default="base")
    ap.add_argument("--tx-window", type=float, default=10.0,
                    help="transcription window s (smaller = more responsive)")
    ap.add_argument("--tx-step", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    source = StreamSource(args.src, duration=args.seconds,
                          realtime=args.realtime, speed=args.speed)
    dva = B.FIELD_WIDTH_DEG / 224.0
    vision_fe = VisionFrontEnd()
    audio_fe = AudioFrontEnd()
    tx = RollingTranscriber(model_size=args.model_size,
                            window_s=args.tx_window, step_s=args.tx_step)
    loop = OnlineLevel3(dva, t_end_ms=(args.seconds or 1e9) * 1000.0)

    detector = TurnDetector(silence_s=1.5)
    policy = ResponsePolicy()
    policy.perceptual = PerceptualState(loop)
    speaker = Speaker()
    vad = EnergyVAD()

    n_mom = 0
    n_turns = 0
    n_barges = 0
    log = []

    def note(msg):
        line = f"[t={n_mom / 10.0:6.1f}s] {msg}"
        print(line, flush=True)
        log.append(line)

    vis_queue = []
    for tick in source:
        vis_queue.append(vision_fe.push(tick.frame))
        for am in audio_fe.push(tick):
            t_ms = n_mom * 100.0
            now_s = t_ms / 1000.0
            vis = vis_queue.pop(0)
            aud = (am["Tprof_n"], am["pan"])
            mono = am["mono"]
            tx.push(t_ms, mono)
            sp = tx.speech_tick(t_ms)

            # --- barge-in: fast VAD on the raw mic, checked BEFORE the
            # --- turn detector (the transcript lags seconds behind).
            if speaker.is_playing(now_s) and vad.update(mono):
                speaker.stop()
                n_barges += 1
                note("BARGE-IN: user speech during playback -- "
                     "stopped response, listening")

            # --- perception tick (with any language bias from last turn)
            loop.tick(t_ms, vis, aud, sp, task_bias=policy.take_bias())

            # --- turn detection on transcribed-so-far words
            for turn in detector.update(tx.transcript(), now_s):
                n_turns += 1
                note(f"TURN: {turn.text!r}")
                t0 = time.time()
                reply = policy.generate(turn)
                if reply is None:
                    continue
                note(f"REPLY ({time.time() - t0:.1f}s to generate): "
                     f"{reply!r}")
                out = os.path.join(args.outdir, f"reply_{n_turns:02d}.mp3")
                t0 = time.time()
                speak(reply, out)  # v1: blocks the tick loop
                note(f"TTS ({time.time() - t0:.1f}s) -> {out}")
                speaker.play(out, now_s)
                note("SPEAKING...")

            if speaker.check_finished(now_s):
                note("done speaking -- listening")

            n_mom += 1

    note(f"stream ended: {n_mom} moments, {n_turns} turns, "
         f"{n_barges} barge-ins")
    with open(os.path.join(args.outdir, "conversation.log"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
