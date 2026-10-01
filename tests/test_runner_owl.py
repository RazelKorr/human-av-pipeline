"""Runner-level tests for opt-in OWL-ViT in scripts/run_conversation.py.

These test the wiring the live loop actually uses (LatestFrame +
wire_owl_detection), with a stubbed detector standing in for the
~350 MB weights. The key claim under test: when a turn asks for
something never seen, the detector's live query receives the
runner's *latest* frame -- not a stale one, not a crop -- and the
query never fires outside that on-demand path.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from hva.conversation import ResponsePolicy, Turn  # noqa: E402
from hvp.detect import pil_from_frame  # noqa: E402
from run_conversation import (  # noqa: E402
    LatestFrame, wire_owl_detection)


class StubDetector:
    """Stands in for ObjectDetector: records the live query, returns
    one centered box for 'a window'."""

    def __init__(self):
        self.calls = []  # (image, queries) per live query

    def detect(self, image, queries, threshold=0.10):
        self.calls.append((image, list(queries)))
        if any("window" in q for q in queries):
            return [("a window", 100.0, 100.0, 140.0, 140.0, 0.50)]
        return []

    @staticmethod
    def box_center_map(det):
        _, x0, y0, x1, y1, _ = det
        return ((x0 + x1) / 2.0 / 4.0, (y0 + y1) / 2.0 / 4.0)


def _wire():
    """A policy wired exactly the way the runner wires it."""
    policy = ResponsePolicy()
    latest = LatestFrame()
    stub_holder = {}

    def factory():
        stub = StubDetector()
        stub_holder["stub"] = stub
        return stub

    wire_owl_detection(policy, latest, True, detector_factory=factory)
    return policy, latest, stub_holder


def test_latest_frame_none_before_first_frame():
    latest = LatestFrame()
    assert latest.as_pil() is None
    latest.update(None)  # blind moments don't clear the last good frame
    assert latest.as_pil() is None


def test_latest_frame_serves_most_recent():
    latest = LatestFrame()
    black = np.zeros((224, 224), np.float32)
    white = np.ones((224, 224), np.float32)
    latest.update(black)
    latest.update(white)
    img = latest.as_pil()
    assert img.size == (224, 224)
    assert img.mode == "RGB"
    # Latest frame was white, not the earlier black one.
    assert img.getpixel((0, 0)) == (255, 255, 255)
    assert img.getpixel((223, 223)) == (255, 255, 255)


def test_pil_from_frame_matches_detector_contract():
    gray = np.full((224, 224), 0.5, np.float32)
    img = pil_from_frame(gray)
    assert img.size == (224, 224) and img.mode == "RGB"
    assert img.getpixel((100, 100)) == (127, 127, 127)


def test_wire_owl_disabled_leaves_policy_untouched():
    policy = ResponsePolicy()
    latest = LatestFrame()
    wire_owl_detection(policy, latest, False,
                       detector_factory=StubDetector)
    assert policy.detector is None
    assert policy.frame_fn is None
    # And the honest-miss path still works with no detector wired.
    reply = policy.generate(Turn("Wodehaus, where is the window?", 10.0))
    assert reply.startswith("I don't know what a window looks like yet")


def test_latest_frame_reaches_live_query():
    policy, latest, holder = _wire()
    black = np.zeros((224, 224), np.float32)
    white = np.ones((224, 224), np.float32)
    # Two ticks, like the 10 Hz loop: the turn must see the latest.
    latest.update(black)
    latest.update(white)

    reply = policy.generate(Turn("Wodehaus, where is the window?", 10.0))

    stub = holder["stub"]
    assert len(stub.calls) == 1, "exactly one on-demand query"
    image, queries = stub.calls[0]
    assert queries == ["a window"], "noun-phrase wrapping preserved"
    # The query saw the latest frame's bytes, not the stale one.
    assert image.getpixel((0, 0)) == (255, 255, 255)
    assert image.getpixel((223, 223)) == (255, 255, 255)
    assert reply.startswith("Found the window --")


def test_detection_not_fired_for_other_intents():
    policy, latest, holder = _wire()
    latest.update(np.ones((224, 224), np.float32))
    policy.generate(Turn("Wodehaus, what do you see?", 10.0))
    policy.generate(Turn("Wodehaus, look to the left please", 10.0))
    assert holder["stub"].calls == [], \
        "no query outside the where-is-it path"


def test_memory_track_short_circuits_detector():
    policy, latest, holder = _wire()
    latest.update(np.ones((224, 224), np.float32))
    policy.memory.add("window", 28.0, 28.0, t_ms=0.0, conf=0.90)
    reply = policy.generate(Turn("Wodehaus, look at the window", 10.0))
    assert holder["stub"].calls == [], \
        "known track answers from memory, no live query"
    assert reply.startswith("Looking at the window --")
