"""W3 (2026-10-14): the golden-fixture regression pack, wired
into the suite. The pack itself lives in tests/golden/
(run_golden.py + frozen expectations); this test runs the
same check `sensorium golden` runs, so silent drift in the
flagship ear/eye/binder math fails here, loudly, in seconds.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "run_golden", os.path.join(_HERE, "golden", "run_golden.py"))
run_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_golden)


def test_golden_pack_matches():
    lines, diffs = run_golden.check()
    assert not diffs, "\n".join(lines + diffs)
