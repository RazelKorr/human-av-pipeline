"""Provenance records for run artifacts (W1, commission 2,
2026-10-14).

Why this exists: this project's recurring failure mode is a
number quoted from a run whose producing code state is no
longer identifiable -- onset counts filed under a picker bar
that was later rescaled (PX.1), binding rates filed under a
chance estimator that was later relabelled. The artifacts
survived; the code state that produced them had to be
reconstructed from commit archaeology. A provenance record
makes the reconstruction unnecessary: every run output carries
the git commit, a hash of the exact run configuration, and the
ear/eye/binder version constants in force, in machine-readable
form, at the moment it is written.

What it is not: a backfill. Artifacts written before this
module carry no record and cannot honestly acquire one after
the fact; their code state remains a matter for the reviews
that filed them.
"""

from __future__ import annotations

import datetime
import hashlib
import inspect
import json
import os
import subprocess

PROVENANCE_SCHEMA = "sensorium-provenance/1"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_git_cache: dict = {}


def _git(args, root=None):
    root = root or _ROOT
    key = (root, tuple(args))
    if key not in _git_cache:
        try:
            out = subprocess.run(
                ["git", "-C", root] + args, capture_output=True,
                text=True, timeout=10)
            _git_cache[key] = out.stdout.strip() if (
                out.returncode == 0) else None
        except (OSError, subprocess.SubprocessError):
            _git_cache[key] = None
    return _git_cache[key]


def git_commit(root=None):
    """HEAD commit hash of the repo, or None outside a checkout."""
    return _git(["rev-parse", "HEAD"], root)


def git_dirty(root=None):
    """True when the working tree has uncommitted changes."""
    out = _git(["status", "--porcelain"], root)
    return bool(out) if out is not None else None


def config_hash(config):
    """sha256 of the canonical JSON encoding of a config dict.
    Key order does not matter; values must be JSON-serializable.
    """
    canon = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def version_constants():
    """The ear/eye/binder constants that shape every number a
    run produces, read from the modules themselves (never
    restated here, so the record cannot drift from the code).
    """
    from hva.online import MAD_TO_SIGMA, MOMENT_MS as A_MOMENT_MS, SR
    from streaming import bind_av
    sig = inspect.signature(bind_av.shift_null_context)
    return {
        "ear": {"module": "hva.online",
                "mad_to_sigma": MAD_TO_SIGMA,
                "moment_ms": A_MOMENT_MS,
                "sample_rate": SR},
        "eye": {"module": "streaming.bind_av",
                "visual_mad_to_sigma": bind_av.VISUAL_MAD_TO_SIGMA,
                "moment_ms": bind_av.MOMENT_MS},
        "binder": {"module": "streaming.bind_av",
                   "coincidence_ms": bind_av.COINCIDENCE_MS,
                   "shift_null_seed_default": sig.parameters[
                       "seed"].default,
                   "shift_null_n_shifts_default": sig.parameters[
                       "n_shifts"].default},
    }


def default_seeds():
    """The seeds in force when a caller does not override them,
    read from the owning functions' signatures (never restated).
    """
    from streaming import bind_av
    sig = inspect.signature(bind_av.bind)
    return {"shift_null": sig.parameters["null_seed"].default}


def provenance_record(config=None, seeds=None, root=None):
    """One run's provenance record (JSON-serializable dict)."""
    return {
        "schema": PROVENANCE_SCHEMA,
        "generated": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "config": config,
        "config_hash": config_hash(config) if config else None,
        "versions": version_constants(),
        "seeds": dict(seeds or {}),
    }


def artifact_hashes(outdir, names):
    """{filename: sha256} for the named files present in outdir."""
    out = {}
    for name in names:
        path = os.path.join(outdir, name)
        if os.path.isfile(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            out[name] = h.hexdigest()
    return out


def write_manifest(outdir, record, artifact_names):
    """Write provenance.json into outdir: the record plus the
    sha256 of each artifact it covers (the manifest itself is
    excluded -- it cannot hash its own final bytes)."""
    manifest = dict(record)
    manifest["artifacts"] = artifact_hashes(outdir, artifact_names)
    path = os.path.join(outdir, "provenance.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path
