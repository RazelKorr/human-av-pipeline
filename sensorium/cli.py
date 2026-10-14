"""sensorium: one front door for the human-av-pipeline (W2,
commission 2, 2026-10-14).

Before this module, the pipeline had a dozen doors: a new
arrival had to know that streaming lives in streaming/, that
backfills live in scripts/, and which of the run_* scripts is
the flagship. The knowledge existed only as oral history in
the reviews. This CLI is a map, not a rewrite: every
subcommand delegates to the existing code, unchanged, through
that code's own argument parser. The old entry points keep
working; this is a second door into the same rooms.

    python -m sensorium run input/clip.mp4 --outdir output/run1
    python -m sensorium bind output/run1
    python -m sensorium baseline output/run1
    python -m sensorium audit
    python -m sensorium golden
    python -m sensorium fixtures --kind dark_busy
    python -m sensorium percept

Run with the repo venv: .venv/bin/python -m sensorium ...
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EPILOG = """\
the workflows, in the order a new arrival usually needs them:

  run       the flagship streaming pipeline: play a media file
            as if live; write the percept artifacts (fixations,
            frame energies, audio features, onsets, AV binding,
            run report, provenance) into --outdir.
  bind      recompute a stored run's AV binding from its own
            artifacts (audio_events.json + frame_energies.npy),
            e.g. after the binder changes. Writes av_binding.json
            back into the run dir (or --out PATH).
  baseline  backfill the binding chance-baseline sidecar
            (av_binding_baseline.json) for stored runs.
  audit     re-derive the headline corpus numbers from the
            stored artifacts and flag any repo doc whose cited
            figures disagree. Reports only; edits nothing.
  golden    re-run the golden-fixture regression pack and diff
            against the frozen expectations (also a test-suite
            test; this is the same check by hand).
  fixtures  (re)generate the synthetic fixtures (quiet room,
            dark-busy, bright-silent, fill-busy) into input/.
  percept   render the closed-loop Star Tours run as a percept
            video -- the visual input a review is written from.

Every subcommand except bind/audit/golden hands its remaining
arguments to the existing tool's own parser: `sensorium run
--help` is run_stream.py's help, unchanged.
"""


def _delegate(module_name, argv):
    """Import an existing tool module and call its main() with
    argv, exactly as if it had been invoked directly."""
    for p in (ROOT, os.path.join(ROOT, "scripts"),
              os.path.join(ROOT, "streaming")):
        if p not in sys.path:
            sys.path.insert(0, p)
    mod = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [module_name.replace(".", " ")] + list(argv)
    try:
        mod.main()
    finally:
        sys.argv = old_argv


def _cmd_bind(argv):
    """bind: run_stream's Phase 2 as a standalone door.

    Rebuilds the visual transient stream from a stored run's
    frame_energies.npy exactly as run_stream does (total energy
    = static + chroma + transient columns), binds the stored
    audio_events.json onsets with streaming.bind_av.bind, stamps
    provenance (W1), and writes the binding JSON.
    """
    import json

    import numpy as np

    ap = argparse.ArgumentParser(
        prog="sensorium bind",
        description="Recompute a stored run's AV binding from "
                    "its artifacts (see module docstring).")
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None,
                    help="output path (default: RUN_DIR/"
                         "av_binding.json)")
    args = ap.parse_args(argv)

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from streaming.bind_av import bind, visual_transients
    from streaming.provenance import provenance_record

    run_dir = args.run_dir
    with open(os.path.join(run_dir, "audio_events.json")) as f:
        audio_events = json.load(f)
    rows = np.load(os.path.join(run_dir, "frame_energies.npy"))
    e_tot = rows[:, 1] + rows[:, 2] + rows[:, 3]
    vtrans = visual_transients(e_tot, rows[:, 0].astype(np.float64))
    duration = None
    report_path = os.path.join(run_dir, "run_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            duration = json.load(f).get("processed_seconds")
    result = bind([(o["t_s"], o["strength"]) for o in audio_events],
                  vtrans, duration_s=duration)
    result["provenance"] = provenance_record(
        config={"workflow": "sensorium bind", "run_dir": run_dir},
        seeds={"shift_null": result.get("shift_null", {}).get("seed")})
    out = args.out or os.path.join(run_dir, "av_binding.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{run_dir}: {result['n_bound']}/{result['n_audio_onsets']} "
          f"onsets bound, {result['n_visual_transients']} visual "
          f"transients; wrote {out}")


def _help_parser():
    ap = argparse.ArgumentParser(
        prog="sensorium",
        description="The Sensorium (human-av-pipeline): a "
                    "perceptual front end that watches and hears "
                    "media the way its reviews describe -- "
                    "saccades, pursuits, cochlear onsets, and "
                    "the binding between them. This CLI is the "
                    "one front door; each subcommand delegates "
                    "to the existing tool, unchanged.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subcommand", nargs="?",
                    choices=["run", "bind", "baseline", "audit",
                             "golden", "fixtures", "percept"],
                    help="see the workflow map below")
    return ap


def main(argv=None):
    # Manual dispatch, deliberately NOT argparse subparsers:
    # everything after the subcommand name belongs to the
    # delegate's own parser, verbatim (argparse REMAINDER
    # intercepts leading --flags, which would make
    # `sensorium fixtures --kind dark_busy` fail -- a front
    # door should not need `--` escapes).
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _help_parser().print_help()
        return
    cmd, rest = argv[0], argv[1:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    if cmd == "run":
        _delegate("streaming.run_stream", rest)
    elif cmd == "bind":
        _cmd_bind(rest)
    elif cmd == "baseline":
        _delegate("scripts.binding_baseline", rest)
    elif cmd == "audit":
        _delegate("scripts.self_audit", rest)
    elif cmd == "golden":
        import importlib.util
        path = os.path.join(ROOT, "tests", "golden", "run_golden.py")
        spec = importlib.util.spec_from_file_location(
            "run_golden", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.main(rest)
        if rc:
            raise SystemExit(rc)
    elif cmd == "fixtures":
        _delegate("scripts.make_fixtures", rest)
    elif cmd == "percept":
        _delegate("scripts.render_full_percept", rest)
    else:
        _help_parser().print_help()
        raise SystemExit(f"sensorium: unknown subcommand {cmd!r}")





if __name__ == "__main__":
    main()
