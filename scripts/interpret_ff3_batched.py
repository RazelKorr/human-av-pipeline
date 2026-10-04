"""Batched audio interpretation driver for large runs (FF3 HD).

Phase 1: labels + music on ALL onsets (--no-transcribe), one process.
Phase 2: two-tier transcription on a prioritized sample
         (top-K loudest onsets + all bound onsets, deduped),
         in small batches with a fresh process per batch to avoid the
         CLAP+whisper-base+medium OOM seen on Star Tours (died at 50/238).
Transcripts are merged back into the phase-1 records by timestamp.

Usage:
    python3 scripts/interpret_ff3_batched.py \
        --outdir output/ff3_hd_av --video input/ff3_full_hd.mp4 \
        [--top-k 60] [--batch 25]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
SCRIPT = os.path.join(HERE, "interpret_audio.py")


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--top-k", type=int, default=60,
                    help="loudest onsets to transcribe")
    ap.add_argument("--batch", type=int, default=25,
                    help="onsets per transcription batch process")
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "audio_events.json")) as f:
        events = json.load(f)
    print(f"{len(events)} onsets in run", flush=True)

    # ---- Phase 1: labels + music on everything (no transcription) ----
    run([VENV_PY, SCRIPT, "--outdir", args.outdir, "--video", args.video,
         "--no-transcribe"])
    master_path = os.path.join(args.outdir, "audio_interpretation.json")
    with open(master_path) as f:
        master = json.load(f)
    records = master["records"]
    print(f"phase 1: {len(records)} records, "
          f"music={master['n_music_windows']}", flush=True)

    # ---- Phase 2: prioritized transcription sample ----
    with open(os.path.join(args.outdir, "av_binding.json")) as f:
        binding = json.load(f)
    bound_ts = {round(p["audio_t_s"], 3) for p in binding["bound_pairs"]}

    by_strength = sorted(events, key=lambda o: o.get("strength", 0),
                         reverse=True)
    sample = []
    seen = set()
    for o in by_strength[:args.top_k]:
        key = round(o["t_s"], 3)
        if key not in seen:
            seen.add(key)
            sample.append(o)
    for o in events:
        key = round(o["t_s"], 3)
        if key in bound_ts and key not in seen:
            seen.add(key)
            sample.append(o)
    sample.sort(key=lambda o: o["t_s"])
    print(f"phase 2: transcribing {len(sample)} sampled onsets "
          f"({args.top_k} loudest + {len(bound_ts)} bound, deduped)",
          flush=True)

    # Stage a scratch outdir with just the sample events.
    scratch = tempfile.mkdtemp(prefix="ff3_tx_")
    with open(os.path.join(scratch, "audio_events.json"), "w") as f:
        json.dump(sample, f)
    shutil.copy(os.path.join(args.outdir, "av_binding.json"), scratch)

    tx_by_t: dict[float, dict] = {}
    n_batches = (len(sample) + args.batch - 1) // args.batch
    for b in range(n_batches):
        off = b * args.batch
        run([VENV_PY, SCRIPT, "--outdir", scratch, "--video", args.video,
             "--offset", str(off), "--max-onsets", str(args.batch)])
        with open(os.path.join(scratch, "audio_interpretation.json")) as f:
            bout = json.load(f)
        for rec in bout["records"]:
            tx_by_t[round(rec["t_s"], 3)] = rec
        print(f"  batch {b + 1}/{n_batches}: "
              f"escalated_total={bout['n_escalated_segments']} "
              f"errors={bout['n_errors']}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)

    # ---- Merge transcripts into the master records ----
    n_merged = 0
    for rec in records:
        tx = tx_by_t.get(round(rec["t_s"], 3))
        if tx is not None and tx.get("transcript") is not None:
            rec["transcript"] = tx["transcript"]
            n_merged += 1
    n_esc = sum((r.get("transcript") or {}).get("n_escalated", 0)
                for r in records)
    master["records"] = records
    master["n_transcribed_windows"] = n_merged
    master["n_escalated_segments"] = n_esc
    master["transcription_sample"] = {
        "top_k_loudest": args.top_k,
        "n_bound": len(bound_ts),
        "n_sampled": len(sample),
        "batch": args.batch,
    }
    with open(master_path, "w") as f:
        json.dump(master, f, indent=1)
    print(f"merged {n_merged} transcripts "
          f"({n_esc} escalated segments) -> {master_path}", flush=True)


if __name__ == "__main__":
    main()
