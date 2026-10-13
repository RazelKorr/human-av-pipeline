"""Backfill binding chance baselines for existing runs.

Phase 2 (2026-10-12): bind() now writes its Poisson chance
baseline, tight-window rates, and dt summary into av_binding.json
at run time. Runs made before that carry only the raw coincidence
numbers. This script reads a run dir's existing artifacts and
writes a NEW sidecar, av_binding_baseline.json, with the same
fields computed by streaming.bind_av.binding_summary from the
stored bound_pairs. Originals are never modified.

Duration comes from run_report.json's processed_seconds when
present (the same duration the report's hand baselines used);
otherwise it falls back to the latest event timestamp and says so
in chance_inputs.duration_source.

Usage:
  .venv/bin/python scripts/binding_baseline.py output/ff3_hd_av \
      [more run dirs...]
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from streaming.bind_av import binding_summary  # noqa: E402


def baseline_for_run(run_dir):
    with open(os.path.join(run_dir, "av_binding.json")) as f:
        binding = json.load(f)
    pairs = binding["bound_pairs"]
    n_audio = binding["n_audio_onsets"]
    n_visual = binding["n_visual_transients"]
    coincidence_ms = binding.get("coincidence_ms", 250.0)

    duration, source = None, "max_event_time_estimate"
    report_path = os.path.join(run_dir, "run_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        if report.get("processed_seconds"):
            duration = float(report["processed_seconds"])
            source = "run_report.processed_seconds"
    if duration is None:
        latest = max([p["audio_t_s"] for p in pairs]
                     + [o["t_s"] for o in binding.get("unbound_audio", [])]
                     + [o["t_s"] for o in binding.get("unbound_visual", [])],
                     default=0.0)
        duration = latest if latest > 0 else None

    summary = binding_summary(pairs, n_audio, n_visual,
                              duration_s=duration,
                              coincidence_ms=coincidence_ms,
                              duration_source=source)
    sidecar = {
        "sidecar_for": "av_binding.json",
        "generated": "2026-10-12",
        "generator": "scripts/binding_baseline.py",
        "estimator": "poisson: 1 - exp(-lambda_v * 2W), "
                     "lambda_v = n_visual / duration_s "
                     "(STREAMING-REPORT hand-baseline estimator)",
        "n_bound": binding["n_bound"],
        **summary,
    }
    out_path = os.path.join(run_dir, "av_binding_baseline.json")
    with open(out_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    return out_path, sidecar


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+")
    args = ap.parse_args()
    for run_dir in args.run_dirs:
        out_path, s = baseline_for_run(run_dir)
        tight50 = s["tight_windows"]["50"]["fraction_of_bound"]
        print(f"{run_dir}: bound {s['bound_rate']:.3f} vs chance "
              f"{s['chance_rate_poisson']:.3f} "
              f"(excess {s['excess_over_chance']:+.3f}); "
              f"|dt|<=50ms fraction of bound: {tight50:.3f}; "
              f"wrote {out_path}")


if __name__ == "__main__":
    main()
