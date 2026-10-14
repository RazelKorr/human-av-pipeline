"""W6 (2026-10-14): determinism contracts.

Same seed + same code = byte-identical outputs on the paths
that consume randomness: the binder's shift null, and the
battery's noise stimuli (explicit-seed route).
"""
import json

import numpy as np

from hva.battery import click, room_tone
from streaming.bind_av import bind


def test_bind_double_run_byte_identical():
    onsets = [(1.0, 3.0), (2.05, 2.0), (4.5, 4.0), (7.0, 1.5)]
    vtrans = [(1.05, 2.0), (2.0, 1.0), (4.75, 3.0), (9.0, 1.0)]
    a = json.dumps(bind(onsets, vtrans, duration_s=30.0),
                   sort_keys=True)
    b = json.dumps(bind(onsets, vtrans, duration_s=30.0),
                   sort_keys=True)
    assert a == b  # shift null included: it is seeded


def test_battery_seeded_stimuli_reproducible():
    assert np.array_equal(click(seed=7), click(seed=7))
    assert np.array_equal(room_tone(0.5, seed=7),
                          room_tone(0.5, seed=7))
    assert not np.array_equal(click(seed=7), click(seed=8))
