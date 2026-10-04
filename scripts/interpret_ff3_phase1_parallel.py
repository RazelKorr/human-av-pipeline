"""Parallel phase-1: labels + music on all onsets, 2 workers.

Each worker runs scripts/interpret_audio.py --no-transcribe over its half
of the onset list (via --offset/--max-onsets) in a scratch outdir, writing
its own audio_interpretation.json. Results are merged into the run's
audio_interpretation.json afterwards.

Usage: python3 scripts/interpret_ff3_phase1_parallel.py \
           --outdir output/ff3_hd_av --video input/ff3_full_hd.mp4
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
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "audio_events.json")) as f:
        events = json.load(f)
    n = len(events)
    print(f"{n} onsets, {args.workers} workers", flush=True)

    scratches = []
    procs = []
    chunk = (n + args.workers - 1) // args.workers
    for w in range(args.workers):
        off = w * chunk
        cnt = min(chunk, n - off)
        if cnt <= 0:
            break
        sd = tempfile.mkdtemp(prefix=f"ff3_p1_w{w}_")
        scratches.append(sd)
        shutil.copy(os.path.join(args.outdir, "audio_events.json"), sd)
        shutil.copy(os.path.join(args.outdir, "av_binding.json"), sd)
        log = open(os.path.join(sd, "worker.log"), "w")
        p = subprocess.Popen(
            [VENV_PY, SCRIPT, "--outdir", sd, "--video", args.video,
             "--offset", str(off), "--max-onsets", str(cnt),
             "--no-transcribe"],
            stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log, sd, off, cnt))
        print(f"worker {w}: onsets [{off},{off + cnt}) -> {sd}", flush=True)

    all_records: list[dict] = []
    n_music = n_err = 0
    for p, log, sd, off, cnt in procs:
        rc = p.wait()
        log.close()
        if rc != 0:
            print(f"WORKER FAILED rc={rc} (onsets [{off},{off + cnt})); "
                  f"see {sd}/worker.log", flush=True)
            sys.exit(1)
        with open(os.path.join(sd, "audio_interpretation.json")) as f:
            out = json.load(f)
        all_records.extend(out["records"])
        n_music += out["n_music_windows"]
        n_err += out["n_errors"]
        print(f"worker [{off},{off + cnt}): {len(out['records'])} records, "
              f"music={out['n_music_windows']} errors={out['n_errors']}",
              flush=True)
        shutil.rmtree(sd, ignore_errors=True)

    all_records.sort(key=lambda r: r["t_s"])
    master = {"records": all_records,
              "n_music_windows": n_music,
              "n_escalated_segments": 0,
              "n_errors": n_err}
    dest = os.path.join(args.outdir, "audio_interpretation.json")
    with open(dest, "w") as f:
        json.dump(master, f, indent=1)
    print(f"merged {len(all_records)} records -> {dest}", flush=True)


if __name__ == "__main__":
    main()
