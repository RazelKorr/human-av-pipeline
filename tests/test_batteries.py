"""The three synthetic validation batteries, wrapped for pytest.

Until now the batteries (hvp T0-T4, hva A0-A4+A1b, hvm M1-M6) lived
outside the test suite: "full suite green" did not include the
headline validation claims the reviews lean on. These are thin
wrappers -- each calls the battery's own run_all() and asserts
all-pass, so the battery remains the single source of truth.
"""

from hva import battery as hva_battery
from hvp import battery as hvp_battery
from hvm import battery as hvm_battery


def test_vision_battery():
    results = hvp_battery.run_all()
    failed = [name for name, r in results.items() if not r.get("pass")]
    assert not failed, f"vision battery failures: {failed}"


def test_audio_battery():
    results = hva_battery.run_all()
    failed = [name for name, r in results.items() if not r.get("pass")]
    assert not failed, f"audio battery failures: {failed}"


def test_fusion_battery():
    assert hvm_battery.run_all(), "fusion battery (M1-M6) not all-pass"
