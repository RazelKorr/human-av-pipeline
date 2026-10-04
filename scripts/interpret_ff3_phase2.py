"""Phase 2: two-tier transcription on a prioritized sample.

Sample = top-K loudest onsets + every Nth bound onset, restricted to
onsets already covered by the phase-1 sample (so transcripts merge cleanly
by t_s). Runs in small batches with a fresh process per batch to avoid the
CLAP+whisper-base+medium OOM. Merges transcripts into the run's
audio_interpretation.json.

Usage: python3 scripts/interpret_ff3_phase2.py \
           --outdir output/ff3_hd_av --video input/ff3_full_hd.mp4 \
           --sample-json /tmp/ff3_p1_sample.json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--sample-json", required=True,
                    help="phase-1 sample events (transcribe a subset)")
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--bound-every", type=int, default=6)
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()

    with open(args.sample_json) as f:
        sample = json.load(f)
    with open(os.path.join(args.outdir, "av_binding.json")) as f:
        binding = json.load(f)
    bound_ts = {round(p["audio_t_s"], 3) for p in binding["bound_pairs"]}

    by_strength = sorted(sample, key=lambda o: o.get("strength", 0),
                         reverse=True)
    tx = []
    seen = set()
    for o in by_strength[:args.top_k]:
        key = round(o["t_s"], 3)
        if key not in seen:
            seen.add(key)
            tx.append(o)
    bound_in_sample = [o for o in sample
                       if round(o["t_s"], 3) in bound_ts]
    for o in bound_in_sample[::args.bound_every]:
        key = round(o["t_s"], 3)
        if key not in seen:
            seen.add(key)
            tx.append(o)
    tx.sort(key=lambda o: o["t_s"])
    print(f"transcribing {len(tx)} windows "
          f"({args.top_k} loudest + {len(bound_in_sample[::args.bound_every])} "
          f"bound-sampled)", flush=True)

    scratch = tempfile.mkdtemp(prefix="ff3_p2_")
    with open(os.path.join(scratch, "audio_events.json"), "w") as f:
        json.dump(tx, f)
    shutil.copy(os.path.join(args.outdir, "av_binding.json"), scratch)

    tx_by_t: dict[float, dict] = {}
    n_batches = (len(tx) + args.batch - 1) // args.batch
    for b in range(n_batches):
        off = b * args.batch
        print(f"$ batch {b + 1}/{n_batches} [{off},{off + args.batch})",
              flush=True)
        subprocess.run(
            [VENV_PY, SCRIPT, "--outdir", scratch, "--video", args.video,
             "--offset", str(off), "--max-onsets", str(args.batch)],
            check=True)
        with open(os.path.join(scratch, "audio_interpretation.json")) as f:
            bout = json.load(f)
        for rec in bout["records"]:
            tx_by_t[round(rec["t_s"], 3)] = rec
        print(f"  done: escalated={bout['n_escalated_segments']} "
              f"errors={bout['n_errors']}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)

    master_path = os.path.join(args.outdir, "audio_interpretation.json")
    with open(master_path) as f:
        master = json.load(f)
    n_merged = 0
    for rec in master["records"]:
        t = tx_by_t.get(round(rec["t_s"], 3))
        if t is not None and t.get("transcript") is not None:
            rec["transcript"] = t["transcript"]
            n_merged += 1
    n_esc = sum((r.get("transcript") or {}).get("n_escalated", 0)
                for r in master["records"])
    master["n_transcribed_windows"] = n_merged
    master["n_escalated_segments"] = n_esc
    master["transcription_sample"] = {
        "top_k_loudest": args.top_k,
        "bound_every": args.bound_every,
        "n_sampled": len(tx),
        "batch": args.batch,
    }
    with open(master_path, "w") as f:
        json.dump(master, f, indent=1)
    print(f"merged {n_merged} transcripts ({n_esc} escalated) -> "
          f"{master_path}", flush=True)


if __name__ == "__main__":
    main()
