"""W1 (2026-10-14): provenance records for run artifacts."""
import hashlib
import json
import os

from streaming.provenance import (PROVENANCE_SCHEMA, config_hash,
                                  default_seeds, provenance_record,
                                  version_constants, write_manifest)


def test_record_structure():
    rec = provenance_record(config={"a": 1}, seeds={"s": 7})
    assert rec["schema"] == PROVENANCE_SCHEMA
    assert rec["git_commit"] and len(rec["git_commit"]) == 40
    assert rec["config"] == {"a": 1}
    assert rec["config_hash"] == config_hash({"a": 1})
    assert rec["seeds"] == {"s": 7}
    v = rec["versions"]
    assert v["ear"]["mad_to_sigma"] > 1.0
    assert v["eye"]["visual_mad_to_sigma"] > 1.0
    assert v["binder"]["coincidence_ms"] == 250.0


def test_config_hash_canonical():
    # key order must not matter; values must
    assert config_hash({"a": 1, "b": [1, 2]}) == \
        config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_version_constants_match_modules():
    from hva.online import MAD_TO_SIGMA
    from streaming.bind_av import VISUAL_MAD_TO_SIGMA
    v = version_constants()
    assert v["ear"]["mad_to_sigma"] == MAD_TO_SIGMA
    assert v["eye"]["visual_mad_to_sigma"] == VISUAL_MAD_TO_SIGMA
    assert default_seeds()["shift_null"] == 20261014


def test_write_manifest_hashes_artifacts(tmp_path):
    art = tmp_path / "av_binding.json"
    art.write_text('{"n_bound": 3}')
    rec = provenance_record(config={"x": 1})
    out = write_manifest(str(tmp_path), rec,
                         ["av_binding.json", "missing.npy"])
    man = json.loads(open(out).read())
    want = hashlib.sha256(b'{"n_bound": 3}').hexdigest()
    assert man["artifacts"] == {"av_binding.json": want}
    assert man["git_commit"] == rec["git_commit"]
    # the manifest never claims to hash itself
    assert "provenance.json" not in man["artifacts"]
    assert os.path.basename(out) == "provenance.json"
