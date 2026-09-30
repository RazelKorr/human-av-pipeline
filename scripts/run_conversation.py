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
audio cable (see docs/saying-hi.md). TTS synthesis runs in a background
thread (AsyncTTS) so the 10 Hz tick loop never stalls while speaking.
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
                              EnergyVAD, AsyncTTS)
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
    ap.add_argument("--llm", default="none",
                    choices=["none", "api", "local", "hf", "free", "auto"],
                    help="language-model backend for ResponsePolicy: "
                         "none=rule-based only, api=Anthropic (needs "
                         "ANTHROPIC_API_KEY), local=llama-server at --llm-url, "
                         "hf=HuggingFace Inference (needs HF_TOKEN, free tier), "
                         "free=Pollinations.ai (no account, no key), "
                         "auto=local if reachable else hf if tokened else "
                         "api if keyed else free else none")
    ap.add_argument("--llm-url", default="http://localhost:8080",
                    help="base URL for the local llama-server backend")
    ap.add_argument("--api-model", default="claude-haiku-4-5-20251001",
                    help="Anthropic model id for --llm api/auto")
    ap.add_argument("--hf-model", default="Qwen/Qwen3-8B",
                    help="HuggingFace model id for --llm hf/auto")
    ap.add_argument("--free-model", default="openai",
                    help="Pollinations.ai model label for --llm free/auto")
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

    # --- LLM backend selection (the seam is real now) ---
    from hva.llm import select_llm_backend
    llm_backend, llm_desc = select_llm_backend(
        args.llm, args.llm_url, args.api_model, args.hf_model,
        args.free_model)
    policy.llm = llm_backend
    print(f"[llm] backend: {llm_desc}", flush=True)

    speaker = Speaker()
    vad = EnergyVAD()
    tts = AsyncTTS()

    # Per-tick acoustic speech presence for the turn detector's
    # hallucination guard: (t_s, hot). Pruned to the recent past;
    # the transcript lags the audio by up to the window length.
    vad_hist: list[tuple[float, bool]] = []
    VAD_HIST_S = 120.0

    def speech_fraction(t0: float, t1: float) -> float:
        ticks = [h for (t, h) in vad_hist if t + 0.1 > t0 and t < t1]
        if not ticks:
            return 0.0
        return sum(ticks) / len(ticks)

    n_mom = 0
    n_turns = 0
    n_barges = 0
    n_dropped = 0
    drop_stale = False  # user spoke while TTS was synthesizing
    log = []

    def note(msg):
        line = f"[t={n_mom / 10.0:6.1f}s] {msg}"
        print(line, flush=True)
        log.append(line)

    def process_measure(am, vis):
        """One 100 ms audio measure through the full live loop."""
        nonlocal n_mom, n_turns, n_barges, n_dropped, drop_stale
        t_ms = n_mom * 100.0
        now_s = t_ms / 1000.0
        aud = (am["Tprof_n"], am["pan"])
        mono = am["mono"]
        tx.push(t_ms, mono)
        sp = tx.speech_tick(t_ms)

        # --- barge-in: fast VAD on the raw mic, checked BEFORE the
        # --- turn detector (the transcript lags seconds behind).
        # --- VAD runs every tick now (not just during playback) so
        # --- we can also drop a response that is still synthesizing
        # --- when the user starts talking.
        onset = vad.update(mono)
        vad_hist.append((now_s, vad.hot))
        while vad_hist and vad_hist[0][0] < now_s - VAD_HIST_S:
            vad_hist.pop(0)
        if onset and speaker.is_playing(now_s):
            speaker.stop()
            n_barges += 1
            note("BARGE-IN: user speech during playback -- "
                 "stopped response, listening")
        elif onset and tts.has_pending():
            drop_stale = True
            note("user spoke during TTS synthesis -- "
                 "will drop stale response")

        # --- perception tick (with any language bias from last turn)
        # --- This never blocks on TTS: synthesis lives in AsyncTTS.
        loop.tick(t_ms, vis, aud, sp, task_bias=policy.take_bias())

        # --- collect finished syntheses and start playback
        for outpath, _text in tts.poll_ready():
            if drop_stale:
                drop_stale = False
                n_dropped += 1
                note(f"dropped stale response ({outpath}) -- "
                     "user spoke during synthesis")
                continue
            speaker.play(outpath, now_s)
            note(f"SPEAKING... ({outpath})")

        # --- turn detection on transcribed-so-far words, guarded by
        # --- the acoustic record (hallucinated speech over silence
        # --- is suppressed) and by re-fire suppression.
        for turn in detector.update(tx.transcript(), now_s,
                                    speech=speech_fraction):
            n_turns += 1
            note(f"TURN: {turn.text!r}")
            t0 = time.time()
            reply = policy.generate(turn)
            if reply is None:
                continue
            note(f"REPLY ({time.time() - t0:.1f}s to generate): "
                 f"{reply!r}")
            out = os.path.join(args.outdir, f"reply_{n_turns:02d}.mp3")
            tts.submit(reply, out)  # non-blocking: tick loop continues
            note(f"TTS queued -> {out}")

        if speaker.check_finished(now_s):
            note("done speaking -- listening")

        n_mom += 1

    vis_queue = []
    try:
        for tick in source:
            vis_queue.append(vision_fe.push(tick.frame))
            for am in audio_fe.push(tick):
                process_measure(am, vis_queue.pop(0))
        # Flush trailing audio: the frontend works in 10 s chunks, so a
        # partial final chunk never emits unless flushed. Without this,
        # up to ~10 s of trailing audio -- and any turns in it -- is lost.
        for am in audio_fe.flush():
            vis = vis_queue.pop(0) if vis_queue else None
            process_measure(am, vis)
    finally:
        tts.shutdown()

    note(f"stream ended: {n_mom} moments, {n_turns} turns, "
         f"{n_barges} barge-ins, {n_dropped} dropped (stale synthesis), "
         f"{len(detector.suppressed)} suppressed "
         f"({sum(1 for r, _ in detector.suppressed if r == 'no_speech')} "
         f"no_speech, "
         f"{sum(1 for r, _ in detector.suppressed if r == 'refire')} refire)")
    for reason, text in detector.suppressed:
        note(f"SUPPRESSED ({reason}): {text!r}")
    with open(os.path.join(args.outdir, "conversation.log"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
