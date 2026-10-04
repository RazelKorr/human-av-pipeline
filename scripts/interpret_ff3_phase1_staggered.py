"""Staggered parallel phase-1: labels + music on all onsets, 2 workers.

Worker 1 launches only after worker 0's log shows inference progress
(past CLAP model load), avoiding the dual-model-load OOM that killed a
simultaneous launch. Merges both audio_interpretation.json files at the
end into the run's audio_interpretation.json.

Usage: python3 scripts/interpret_ff3_phase1_staggered.py \
           --outdir output/ff3_hd_av --video input/ff3_full_hd.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENV_PY = os.path.join(ROOT, ".venv", "bin", "python")
SCRIPT = os.path.join(HERE, "interpret_audio.py")
PROG = re.compile(r"^ {2}\d+/\d+", re.M)


def launch(sd: str, args, off: int, cnt: int):
    log = open(os.path.join(sd, "worker.log"), "w")
    p = subprocess.Popen(
        [VENV_PY, SCRIPT, "--outdir", sd, "--video", args.video,
         "--offset", str(off), "--max-onsets", str(cnt),
         "--no-transcribe"],
        stdout=log, stderr=subprocess.STDOUT)
    return p, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    with open(os.path.join(args.outdir, "audio_events.json")) as f:
        events = json.load(f)
    n = len(events)
    nw = args.workers
    # Contiguous ranges, as even as possible.
    bounds = [0]
    for w in range(nw):
        bounds.append(bounds[-1] + (n - bounds[-1]) // (nw - w))
    ranges = [(bounds[w], bounds[w + 1]) for w in range(nw)]
    print(f"{n} onsets; {nw} workers: {ranges}", flush=True)

    scratches = []
    for w in range(nw):
        sd = tempfile.mkdtemp(prefix=f"ff3_p1s_w{w}_")
        scratches.append(sd)
        shutil.copy(os.path.join(args.outdir, "audio_events.json"), sd)
        shutil.copy(os.path.join(args.outdir, "av_binding.json"), sd)

    # Stagger: each worker launches only after the previous one shows
    # inference progress (past CLAP model load). This avoids the
    # simultaneous-load OOM.
    procs: list[tuple] = []
    for w in range(nw):
        off, end = ranges[w]
        cnt = end - off
        if w > 0:
            prev_sd = scratches[w - 1]
            print(f"waiting for worker {w - 1} inference progress...",
                  flush=True)
            for _ in range(120):
                time.sleep(10)
                if procs[w - 1][0].poll() is not None:
                    print(f"worker {w - 1} exited early!", flush=True)
                    sys.exit(1)
                with open(os.path.join(prev_sd, "worker.log")) as f:
                    if PROG.search(f.read()):
                        break
            else:
                print(f"worker {w - 1} never reached inference; aborting",
                      flush=True)
                sys.exit(1)
        p, log = launch(scratches[w], args, off, cnt)
        procs.append((p, log, scratches[w], off, cnt))
        print(f"worker {w} [{off},{off + cnt}) launched", flush=True)

    all_records: list[dict] = []
    n_music = n_err = 0
    for i, (p, log, sd, off, cnt) in enumerate(procs):
        rc = p.wait()
        log.close()
        if rc != 0:
            print(f"WORKER {i} FAILED rc={rc}; see {sd}/worker.log",
                  flush=True)
            sys.exit(1)
        with open(os.path.join(sd, "audio_interpretation.json")) as f:
            out = json.load(f)
        all_records.extend(out["records"])
        n_music += out["n_music_windows"]
        n_err += out["n_errors"]
        print(f"worker {i} [{off},{off + cnt}): {len(out['records'])} "
              f"records, music={out['n_music_windows']} "
              f"errors={out['n_errors']}", flush=True)
        shutil.rmtree(sd, ignore_errors=True)

    all_records.sort(key=lambda r: r["t_s"])
    dest = os.path.join(args.outdir, "audio_interpretation.json")
    with open(dest, "w") as f:
        json.dump({"records": all_records,
                   "n_music_windows": n_music,
                   "n_escalated_segments": 0,
                   "n_errors": n_err}, f, indent=1)
    print(f"merged {len(all_records)} records -> {dest}", flush=True)


if __name__ == "__main__":
    main()
