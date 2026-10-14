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

2026-10-14 (defrag review Fix 5): the sidecar's "generated" stamp
is the actual run date (it was hardcoded to 2026-10-12), the
estimator string also covers the grid-corrected tight chance, and
the sidecar states explicitly that the shift-null context is not
computable from stored pairs alone (binding_summary already marks
it unavailable; the note here makes the limitation visible at the
sidecar's top level).
"""

import argparse
import datetime
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
        "generated": datetime.date.today().isoformat(),
        "generator": "scripts/binding_baseline.py",
        "estimator": "poisson: 1 - exp(-lambda_v * 2W), "
                     "lambda_v = n_visual / duration_s "
                     "(STREAMING-REPORT hand-baseline estimator; a "
                     "non-exclusive upper reference, not the "
                     "matcher's chance rate -- see "
                     "chance_rate_poisson_label). Tight windows "
                     "also carry the grid-corrected chance "
                     "1 - exp(-lambda_v * (2W + 50 ms)).",
        "shift_null_note": "shift_null is unavailable in this "
                           "sidecar: the null re-runs the matcher "
                           "on time-shifted event streams and "
                           "cannot be computed from stored pairs "
                           "alone. Runs made with bind() since "
                           "2026-10-14 carry it in av_binding.json.",
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
        tight50 = s["tight_windows"]["50"]
        print(f"{run_dir}: bound {s['bound_rate']:.3f} "
              f"(Poisson upper reference "
              f"{s['chance_rate_poisson']:.3f}); "
              f"|dt|<=50ms rate {tight50['bound_rate']:.3f} vs "
              f"grid-corrected chance "
              f"{tight50['chance_rate_grid_corrected']:.3f} "
              f"(excess {tight50['excess_over_grid_chance']:+.3f}); "
              f"shift null not computable from pairs; "
              f"wrote {out_path}")


if __name__ == "__main__":
    main()
