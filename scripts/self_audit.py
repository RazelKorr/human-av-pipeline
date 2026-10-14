"""Self-audit: docs vs stored artifacts (W4, commission 2,
2026-10-14).

Why this exists: every correction in this project's history
(the Poisson mislabel, PX.1's filed onset counts, the binder's
representation dust) was found by a human re-deriving a filed
number from the stored artifacts and noticing the mismatch --
weeks later, by hand. This command mechanizes that spot-check.
It re-derives the headline corpus numbers from the artifacts
themselves and flags any repo document whose cited figures
disagree with the recomputation.

It REPORTS. It never edits a doc, never rewrites an artifact.
A FLAG means "a human should look", with the recomputed value
and the cited value side by side.

Checks:
  1. Binding consistency -- for each stored run whose
     av_binding.json carries summary fields, recompute
     binding_summary from the stored pairs and compare
     (bound_rate, tight-50 rate). For runs carrying only a
     baseline sidecar, compare the sidecar against the same
     recomputation.
  2. Recount registry -- the recount numbers filed by the
     2026-10-14 defrag + commission 2 (post-PX.1 ear, post-R1
     eye) are recomputed from stored features/energies, and
     each doc that cites them is checked for the citation.
     Adding a future filed number = adding a registry row.
  3. Correction trail -- the STREAMING-REPORT carries its
     dated 2026-10-14 addendum (the Poisson correction is
     part of the public record, not just the reviews).

Usage: .venv/bin/python scripts/self_audit.py [--root DIR]
Exit code 0 = all checks pass; 1 = at least one FLAG.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from hva.online import pick_onsets  # noqa: E402
from streaming.bind_av import binding_summary, visual_transients  # noqa: E402

FLAGS = []


def report(ok, label, detail=""):
    print(f"AUDIT {'PASS' if ok else 'FLAG'} {label}"
          + (f" -- {detail}" if detail else ""))
    if not ok:
        FLAGS.append(label)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _duration(run_dir, binding):
    rp = os.path.join(run_dir, "run_report.json")
    if os.path.exists(rp):
        d = _load_json(rp).get("processed_seconds")
        if d:
            return float(d)
    return None


def check_binding_consistency(root):
    runs = []
    out = os.path.join(root, "output")
    for name in sorted(os.listdir(out)):
        d = os.path.join(out, name)
        if os.path.isdir(d) and os.path.exists(
                os.path.join(d, "av_binding.json")):
            runs.append(d)
        exp = os.path.join(d, "")  # exp-* containers hold runs
        if name.startswith("exp-") or name.startswith("wild-"):
            for sub in sorted(os.listdir(d)):
                sd = os.path.join(d, sub)
                if os.path.isdir(sd) and os.path.exists(
                        os.path.join(sd, "av_binding.json")):
                    runs.append(sd)
    checked = 0
    for run_dir in runs:
        label = os.path.relpath(run_dir, root)
        try:
            binding = _load_json(os.path.join(run_dir,
                                              "av_binding.json"))
            pairs = binding.get("bound_pairs") or []
            dur = _duration(run_dir, binding)
            recomputed = binding_summary(
                pairs, binding["n_audio_onsets"],
                binding["n_visual_transients"], duration_s=dur,
                coincidence_ms=binding.get("coincidence_ms", 250.0))
            sidecar_path = os.path.join(run_dir,
                                        "av_binding_baseline.json")
            if "bound_rate" in binding:
                stored, src = binding, "av_binding.json"
            elif os.path.exists(sidecar_path):
                stored, src = _load_json(sidecar_path), "sidecar"
            else:
                continue  # raw pairs only; nothing filed to check
            ok = (stored.get("bound_rate") == recomputed["bound_rate"]
                  and stored.get("tight_windows", {}).get(
                      "50", {}).get("bound_rate")
                  == recomputed["tight_windows"]["50"]["bound_rate"])
            report(ok, f"binding consistency {label} ({src})",
                   f"stored bound_rate {stored.get('bound_rate')} vs "
                   f"recomputed {recomputed['bound_rate']}")
            checked += 1
        except Exception as e:
            report(False, f"binding consistency {label}",
                   f"unreadable artifacts: {e}")
    print(f"AUDIT INFO binding consistency: {checked} runs checked")


def _ear_recount(run_dir):
    afe = np.load(os.path.join(run_dir, "audio_features.npy"))
    return len(pick_onsets(afe[:, 1], afe[:, 0]))


def _eye_recount(run_dir):
    rows = np.load(os.path.join(run_dir, "frame_energies.npy"))
    e_tot = rows[:, 1] + rows[:, 2] + rows[:, 3]
    return len(visual_transients(e_tot, rows[:, 0].astype(np.float64)))


# (label, recompute, expected, [docs that must cite the string])
RECOUNT_REGISTRY = [
    ("FF3 onsets under the fixed ear",
     lambda root: _ear_recount(os.path.join(
         root, "output", "ff3_hd_av_2026-10-13")),
     2409, "2,409",
     ["improvements-2026-10/REVIEW-DEFRAG-2026-10-14.md"]),
    ("Star Tours onsets under the fixed ear",
     lambda root: _ear_recount(os.path.join(
         root, "output", "st_stream_av_full")),
     103, "103",
     ["improvements-2026-10/REVIEW-DEFRAG-2026-10-14.md"]),
    ("Dr Tran ep4 onsets under the fixed ear",
     lambda root: _ear_recount(os.path.join(
         root, "output", "drtran_ep4_av")),
     377, "377",
     ["improvements-2026-10/REVIEW-DEFRAG-2026-10-14.md"]),
    ("YTPMV onsets under the fixed ear",
     lambda root: _ear_recount(os.path.join(
         root, "output", "exp-2026-10-13", "ytpmv_music")),
     212, "212",
     ["improvements-2026-10/REVIEW-DEFRAG-2026-10-14.md"]),
    ("dark-busy transients under the fixed eye (R1)",
     lambda root: _eye_recount(os.path.join(
         root, "output", "exp-2026-10-13", "dark_busy")),
     53, "53",
     ["improvements-2026-10/REVIEW-DEFRAG-2026-10-14.md"]),
]


def check_recounts(root):
    for label, fn, expected, cite, docs in RECOUNT_REGISTRY:
        try:
            got = fn(root)
        except Exception as e:  # artifact missing/unreadable
            report(False, f"recount {label}", f"recompute failed: {e}")
            continue
        report(got == expected, f"recount {label}",
               f"recomputed {got}, filed {expected}")
        for doc in docs:
            path = os.path.join(root, doc)
            text = open(path).read() if os.path.exists(path) else ""
            report(cite in text, f"citation of {label} in {doc}",
                   f"looking for {cite!r}")


def check_correction_trail(root):
    path = os.path.join(root, "streaming", "STREAMING-REPORT.md")
    text = open(path).read() if os.path.exists(path) else ""
    report("Addendum 2026-10-14" in text,
           "STREAMING-REPORT carries the 2026-10-14 addendum")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    check_binding_consistency(root)
    check_recounts(root)
    check_correction_trail(root)
    if FLAGS:
        print(f"AUDIT RESULT: {len(FLAGS)} FLAG(S): {FLAGS}")
        return 1
    print("AUDIT RESULT: all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
