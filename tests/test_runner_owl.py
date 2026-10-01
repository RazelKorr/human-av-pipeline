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
        self.warmed = False

    def warmup(self):
        self.warmed = True

    def detect(self, image, queries, threshold=0.10):
        self.calls.append((image, list(queries)))
        if any("window" in q for q in queries):
            return [("a window", 100.0, 100.0, 140.0, 140.0, 0.50)]
        return []


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
    assert reply.startswith("I don't see one right now")


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


def test_detector_warmed_eagerly_at_wire_time():
    _, _, holder = _wire()
    assert holder["stub"].warmed, \
        "model loads at wire time, not on the first live query"


def _make_test_clip(path):
    import subprocess
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
         "-pix_fmt", "yuv420p", str(path)],
        check=True)


def test_fullres_grab_prefers_source_file(tmp_path):
    from run_conversation import grab_frame
    clip = tmp_path / "clip.mp4"
    _make_test_clip(clip)
    # Direct grab: full-res color, capped at 960 wide.
    img = grab_frame(str(clip), 1.0)
    assert img is not None and img.mode == "RGB"
    assert img.size[0] == 960 and img.size[1] == 720
    # Through LatestFrame: the live query sees the full-res frame,
    # not the 224px reflex bytes.
    latest = LatestFrame(src=str(clip))
    latest.update(np.ones((224, 224), np.float32), t_s=1.0)
    served = latest.as_pil()
    assert served.size == (960, 720)


def test_fullres_falls_back_for_nonfile_src():
    # Live/URL sources have no seekable file: the 224px reflex frame
    # is served instead of failing.
    latest = LatestFrame(src="/nonexistent/stream-key")
    latest.update(np.ones((224, 224), np.float32), t_s=5.0)
    img = latest.as_pil()
    assert img.size == (224, 224) and img.mode == "RGB"
    assert img.getpixel((0, 0)) == (255, 255, 255)


def test_box_coords_scale_with_frame_size():
    # The policy maps detector boxes to 56-map by the frame's actual
    # size, not a hardcoded /4 (which assumed 224px input).
    policy, latest, holder = _wire()
    latest.update(np.ones((224, 224), np.float32))
    reply = policy.generate(Turn("Wodehaus, where is the window?", 10.0))
    # Stub box (100,100)-(140,140) in 224-space -> center (30,30) map.
    tracks = [tr for tr in policy.memory.tracks if tr.label == "a window"]
    assert tracks, "detection added a memory track"
    assert abs(tracks[0].x - 30.0) < 1e-6
    assert abs(tracks[0].y - 30.0) < 1e-6
    assert reply.startswith("Found the window --")
