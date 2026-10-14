"""W2 (2026-10-14): the sensorium front door -- help works
cold, and the bind subcommand reproduces run_stream's Phase 2
on synthetic artifacts with provenance stamped."""
import json
import os

import numpy as np

from sensorium.cli import _cmd_bind, main


def test_help_runs_cold(capsys):
    main([])
    out = capsys.readouterr().out
    assert "front door" in out
    assert "run" in out and "audit" in out


def test_bind_subcommand(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with open(run / "audio_events.json", "w") as f:
        json.dump([{"t_s": 1.0, "strength": 3.0},
                   {"t_s": 5.0, "strength": 2.0}], f)
    t = np.arange(0, 10000, 100)  # 100 frames at 100 ms
    rows = np.zeros((100, 4), dtype=np.float32)
    rows[:, 0] = t
    rows[10, 1] = 5.0  # energy jump into t=1.0 s
    np.save(run / "frame_energies.npy", rows)
    with open(run / "run_report.json", "w") as f:
        json.dump({"processed_seconds": 10.0}, f)
    _cmd_bind([str(run)])
    b = json.loads((run / "av_binding.json").read_text())
    assert b["n_audio_onsets"] == 2
    assert b["provenance"]["schema"] == "sensorium-provenance/1"
    assert b["provenance"]["git_commit"]
