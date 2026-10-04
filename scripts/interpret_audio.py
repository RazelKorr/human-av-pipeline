"""Run the "what" pathway over an existing AV run's onsets.

Reads audio_events.json + av_binding.json from a finished run's
outdir, cuts windows around each onset from the source media, runs
hva.interpret (labels + two-tier transcription + music features),
and writes audio_interpretation.json. Never modifies the existing
artifacts -- the interpretation is a new file alongside them.

Usage:
    python3 scripts/interpret_audio.py \
        --outdir output/st_stream_av_full \
        --video <source media> \
        [--max-onsets N] [--window-s 8.0] [--offset N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from hva.interpret import interpret_window, is_music  # noqa: E402
from hva.transcribe import load_audio  # noqa: E402


def cut_window(video: str, t_s: float, window_s: float,
               tmpdir: str) -> str:
    """Cut a 16 kHz mono wav window around an onset. Returns path."""
    w0 = max(0.0, t_s - window_s / 2.0)
    wav = os.path.join(tmpdir, f"w_{t_s:.2f}.wav")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(w0),
         "-i", video, "-t", str(window_s), "-vn",
         "-ar", "16000", "-ac", "1", wav],
        check=True)
    return wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True,
                    help="finished AV run dir (has audio_events.json)")
    ap.add_argument("--video", required=True,
                    help="source media to cut onset windows from")
    ap.add_argument("--max-onsets", type=int, default=0,
                    help="only interpret the first N onsets (0 = all)")
    ap.add_argument("--window-s", type=float, default=8.0)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N onsets")
    ap.add_argument("--no-transcribe", action="store_true",
                    help="labels + music only (faster)")
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "audio_events.json")) as f:
        events = json.load(f)
    with open(os.path.join(args.outdir, "av_binding.json")) as f:
        binding = json.load(f)
    pair_by_t = {p["audio_t_s"]: p for p in binding["bound_pairs"]}

    sel = events[args.offset:]
    if args.max_onsets:
        sel = sel[:args.max_onsets]
    print(f"interpreting {len(sel)} onsets from {args.outdir}",
          flush=True)

    records = []
    n_music = 0
    n_escalated = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, o in enumerate(sel):
            wav = cut_window(args.video, o["t_s"], args.window_s,
                             tmpdir)
            audio = load_audio(wav)
            rec = interpret_window(
                audio, t_s=o["t_s"], strength=o["strength"],
                transcribe=not args.no_transcribe)
            rec["rms"] = o.get("rms")
            pair = pair_by_t.get(round(o["t_s"], 3))
            if pair is not None:
                rec["bound"] = True
                rec["visual_t_s"] = pair["visual_t_s"]
                rec["dt_s"] = pair["dt_s"]
            else:
                rec["bound"] = False
                rec["visual_t_s"] = None
                rec["dt_s"] = None
            if rec["music"] is not None:
                n_music += 1
            if rec["transcript"] is not None:
                n_escalated += rec["transcript"]["n_escalated"]
            records.append(rec)
            if (i + 1) % 10 == 0 or i + 1 == len(sel):
                print(f"  {i + 1}/{len(sel)} "
                      f"(music={n_music}, escalated={n_escalated})",
                      flush=True)

    out_dir = os.path.join(ROOT, "output", "st_av_interpret")
    os.makedirs(out_dir, exist_ok=True)
    sib = os.path.join(out_dir, "audio_interpretation.json")
    with open(sib, "w") as f:
        json.dump({"records": records,
                   "n_music_windows": n_music,
                   "n_escalated_segments": n_escalated}, f, indent=1)
    print(f"wrote {sib} ({len(records)} windows)", flush=True)


if __name__ == "__main__":
    main()
