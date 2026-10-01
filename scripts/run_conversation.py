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
from hvp.recognize import (FovealClassifier, GENERAL_VOCAB,
                           DARK_STREET_VOCAB, foveal_crop_pil)
from hvp.detect import pil_from_frame


class LatestFrame:
    """The runner's most recent frame, served as PIL RGB.

    process_measure() calls update() once per tick with the 224x224
    reflex frame and the current stream time; policy.frame_fn is this
    object's as_pil. The 10 Hz loop only stores the frame -- the
    detector itself runs solely inside the turn handler (generate ->
    understand's last-resort path), i.e. on demand, never per tick.

    as_pil prefers a full-res color frame decoded on demand from the
    source file at the latest stream timestamp (2026-10-01: the live
    --owl fire showed the detector starved on the 224px grayscale
    reflex frame -- borderline 0.10-0.18 scores where full-res color
    sees clearly). Falls back to the reflex frame for live/non-file
    sources, where no seekable file exists.
    """

    def __init__(self, src=None):
        self.frame = None  # 224x224 float32 in 0..1, or None
        self.t_s = 0.0     # stream time of the latest frame, seconds
        self.src = src     # source path; full-res grab needs a file

    def update(self, frame, t_s=None):
        if frame is not None:
            self.frame = frame
            if t_s is not None:
                self.t_s = t_s

    def as_pil(self):
        """PIL RGB of the latest frame, or None if no frame seen yet."""
        if self.src and os.path.isfile(self.src):
            img = grab_frame(self.src, self.t_s)
            if img is not None:
                return img
        if self.frame is None:
            return None
        return pil_from_frame(self.frame)


def grab_frame(src, t_s, max_w=960):
    """One full-res color frame from a local file at stream time t_s.

    Fast keyframe seek (not sample-accurate, but instant); capped at
    max_w wide so the PNG stays small. Returns a PIL RGB image, or
    None when ffmpeg can't produce one.
    """
    import io
    import subprocess
    from PIL import Image
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t_s):.1f}",
           "-i", src, "-frames:v", "1",
           "-vf", f"scale={max_w}:-1",
           "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if not p.stdout:
        return None
    try:
        return Image.open(io.BytesIO(p.stdout)).convert("RGB")
    except Exception:
        return None


def wire_owl_detection(policy, latest_frame, enabled,
                       detector_factory=None):
    """Opt-in OWL-ViT for the live conversation loop.

    Sets policy.detector / policy.frame_fn so "where is the X" / "find
    the X" with no memory track scans a fresh full-res frame on
    demand. The model loads EAGERLY at wire time (~350 MB, ~19 s on
    2 CPU cores) so the first live query answers in ~2 s instead of
    paying the cold start mid-conversation. Detection stays out of
    the 10 Hz reflex loop -- update() above is the only per-tick
    cost, a reference store. detector_factory is a seam for tests
    (defaults to hvp.detect.ObjectDetector).
    """
    if not enabled:
        return
    if detector_factory is None:
        from hvp.detect import ObjectDetector
        detector_factory = ObjectDetector
    policy.detector = detector_factory()
    policy.detector.warmup()
    policy.frame_fn = latest_frame.as_pil


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--outdir", default="output/conversation")
    ap.add_argument("--model-size", default="base")
    ap.add_argument("--foveal-vocab", default="none",
                    choices=["none", "general", "dark-street"],
                    help="foveal object recognition vocabulary feeding "
                         "policy.memory ('what do you see?' / 'look at "
                         "the X'): none=off, general=open-world labels, "
                         "dark-street=Star-Tours gate/corridor labels")
    ap.add_argument("--owl", action="store_true",
                    help="opt-in on-demand OWL-ViT detection: 'where is the "
                         "X' / 'find the X' with no memory track scans a "
                         "full-res color frame on demand (model pre-loads "
                         "at startup, ~19 s on 2 CPU cores; detection "
                         "never runs in the 10 Hz tick loop)")
    ap.add_argument("--tx-window", type=float, default=10.0,
                    help="transcription window s (smaller = more responsive)")
    ap.add_argument("--tx-step", type=float, default=3.0)
    ap.add_argument("--llm", default="none",
                    choices=["none", "api", "local", "hf", "free", "agent",
                             "auto"],
                    help="language-model backend for ResponsePolicy: "
                         "none=rule-based only, api=Anthropic (needs "
                         "ANTHROPIC_API_KEY), local=llama-server at --llm-url, "
                         "hf=HuggingFace Inference (needs HF_TOKEN, free tier), "
                         "free=Pollinations.ai (no account, no key), "
                         "agent=file handoff in handoff/ (an operator answers "
                         "each turn), "
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
    policy.perceptual = PerceptualState(loop, memory=policy.memory)

    # --- opt-in OWL-ViT: the runner keeps the latest frame; a turn
    # --- that asks for something never seen queries it on demand.
    # --- The model loads eagerly here (~19 s on 2 cores), so the
    # --- first live query answers fast.
    latest_frame = LatestFrame(src=args.src)
    wire_owl_detection(policy, latest_frame, args.owl)
    if args.owl:
        print("[owl] on-demand detection enabled (model pre-loaded at "
              "startup)", flush=True)

    # --- foveal recognition: lazily loaded; classifies the fixated crop
    # --- when gaze moves (throttled) and feeds policy.memory. Off unless
    # --- --foveal-vocab names a vocabulary.
    recognizer = None
    if args.foveal_vocab != "none":
        vocab = (list(DARK_STREET_VOCAB.items())
                 if args.foveal_vocab == "dark-street"
                 else [(n, n) for n in GENERAL_VOCAB])
        _clf = FovealClassifier()
        recognizer = lambda crop: _clf.classify(crop, vocab)
    last_gaze = (112.0, 112.0)
    last_recog_ms = -1e9
    RECOG_MOVE_DEG = 2.0   # re-classify only after gaze moves this far
    RECOG_MIN_GAP_MS = 400.0  # and at most ~2.5 classifications/second

    # --- LLM backend selection (the seam is real now) ---
    from hva.llm import select_llm_backend
    llm_backend, llm_desc = select_llm_backend(
        args.llm, args.llm_url, args.api_model, args.hf_model,
        args.free_model)
    policy.llm = llm_backend
    print(f"[llm] backend: {llm_desc}", flush=True)

    speaker = Speaker(player_cmd=os.environ.get("HVA_AUDIO_PLAYER"))
    if speaker.streaming:
        print(f"[audio] playback: streaming via "
              f"{' '.join(speaker.player_cmd)}", flush=True)
    else:
        print("[audio] playback: simulated (no audio device -- set "
              "HVA_AUDIO_PLAYER, e.g. \"ffplay -nodisp -autoexit -\")",
              flush=True)
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

    def process_measure(am, vis, frame):
        """One 100 ms audio measure through the full live loop."""
        nonlocal n_mom, n_turns, n_barges, n_dropped, drop_stale
        nonlocal last_gaze, last_recog_ms
        t_ms = n_mom * 100.0
        now_s = t_ms / 1000.0
        # Latest frame for opt-in on-demand detection (reference store
        # only -- the detector never runs here).
        latest_frame.update(frame, now_s)
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
        elif not speaker.streaming and onset and tts.has_pending():
            # File mode only: in streaming mode synthesis and playback
            # are one pipeline, so barge-in above already covers it.
            drop_stale = True
            note("user spoke during TTS synthesis -- "
                 "will drop stale response")

        # --- perception tick (with any language bias from last turn)
        # --- This never blocks on TTS: synthesis lives in AsyncTTS.
        bias = policy.take_bias()
        if bias is not None:
            note("perceptual bias applied (look command)")
        loop.tick(t_ms, vis, aud, sp, task_bias=bias)

        # --- foveal recognition: the label stream for "what do you see?"
        # --- and "look at the X". Classifies the fixated crop when gaze
        # --- has moved; sightings accumulate in policy.memory.
        if recognizer is not None and frame is not None:
            _, gx, gy = loop.scanpath[-1]
            moved_deg = (((gx - last_gaze[0]) ** 2
                          + (gy - last_gaze[1]) ** 2) ** 0.5) * dva
            if moved_deg > RECOG_MOVE_DEG and \
                    t_ms - last_recog_ms >= RECOG_MIN_GAP_MS:
                label, conf = recognizer(foveal_crop_pil(frame, gx, gy))
                policy.memory.add(label, gx / 4.0, gy / 4.0,
                                  t_ms=t_ms, conf=conf)
                last_gaze = (gx, gy)
                last_recog_ms = t_ms
                if label != "unknown":
                    note(f"RECOGNIZED '{label}' ({conf:.2f}) at "
                         f"({gx:.0f},{gy:.0f})")

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
        # --- is suppressed) and by re-fire suppression. Endpointing
        # --- uses the latest transcription window's time, not the tick
        # --- clock, so a window that truncates an utterance can't fire
        # --- on the truncated tail.
        for turn in detector.update(tx.transcript(), now_s,
                                    speech=speech_fraction,
                                    tx_time=tx.last_tx_time):
            n_turns += 1
            note(f"TURN: {turn.text!r}")
            t0 = time.time()
            reply = policy.generate(turn)
            if reply is None:
                continue
            note(f"REPLY ({time.time() - t0:.1f}s to generate): "
                 f"{reply!r}")
            out = os.path.join(args.outdir, f"reply_{n_turns:02d}.mp3")
            if speaker.streaming:
                # Streaming path: tts --stream straight into the
                # player (non-blocking); the record file is still
                # written. Barge-in above pipe-kills it.
                try:
                    speaker.play_stream(reply, out, now_s)
                    note(f"SPEAKING (streaming)... ({out})")
                except RuntimeError as e:
                    # One bad reply never kills the loop (mirrors the
                    # AsyncTTS poll_ready failure path in file mode).
                    note(f"TTS stream failed, skipping reply: {e}")
            else:
                tts.submit(reply, out)  # non-blocking: tick loop continues
                note(f"TTS queued -> {out}")

        if speaker.check_finished(now_s):
            note("done speaking -- listening")

        n_mom += 1

    vis_queue = []
    frame_queue = []
    try:
        for tick in source:
            vis_queue.append(vision_fe.push(tick.frame))
            frame_queue.append(tick.frame)
            for am in audio_fe.push(tick):
                process_measure(am, vis_queue.pop(0), frame_queue.pop(0))
        # Flush trailing audio: the frontend works in 10 s chunks, so a
        # partial final chunk never emits unless flushed. Without this,
        # up to ~10 s of trailing audio -- and any turns in it -- is lost.
        for am in audio_fe.flush():
            vis = vis_queue.pop(0) if vis_queue else None
            frame = frame_queue.pop(0) if frame_queue else None
            process_measure(am, vis, frame)
    finally:
        tts.shutdown()

    note(f"stream ended: {n_mom} moments, {n_turns} turns, "
         f"{n_barges} barge-ins, {n_dropped} dropped (stale synthesis), "
         f"{len(detector.suppressed)} suppressed "
         f"({sum(1 for r, _, _ in detector.suppressed if r == 'no_speech')} "
         f"no_speech, "
         f"{sum(1 for r, _, _ in detector.suppressed if r == 'refire')} refire)")
    for reason, text, t in detector.suppressed:
        # Suppressions print at stream end, so they carry the stream
        # time they actually happened at -- not the final timestamp.
        line = f"[t={t:6.1f}s] SUPPRESSED ({reason}): {text!r}"
        print(line, flush=True)
        log.append(line)
    with open(os.path.join(args.outdir, "conversation.log"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
