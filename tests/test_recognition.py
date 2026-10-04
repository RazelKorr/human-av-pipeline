"""Tests for hvp/recognize.py. The softmax/threshold/normalize logic is
tested with a stubbed model so the suite never downloads CLIP weights;
the empirical check is scripts/audit_recognition.py."""
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvp.recognize import FovealClassifier, GENERAL_VOCAB, prompt_for


class StubModel:
    """Returns a fixed image embedding; configurable logit scale."""
    def __init__(self, vec, logit_scale=1.0):
        self._vec = torch.tensor([vec], dtype=torch.float32)
        self.logit_scale = torch.tensor([logit_scale])

    class _Out:
        def __init__(self, t):
            self.pooler_output = t

    def get_image_features(self, **kwargs):
        v = self._vec / self._vec.norm(dim=-1, keepdim=True)
        return StubModel._Out(v)


def make_clf(image_vec, text_vecs, logit_scale=1.0):
    clf = FovealClassifier()
    clf._ensure = lambda: None  # no download
    clf._processor = lambda **kwargs: {}
    clf._model = StubModel(image_vec, logit_scale)

    def fake_text(pairs):
        t = torch.tensor(text_vecs, dtype=torch.float32)
        return t / t.norm(dim=-1, keepdim=True)
    clf._text_features = fake_text
    return clf


def test_prompt_for():
    assert prompt_for("dog") == "a photo of dog"


def test_normalize_bare_and_pairs():
    clf = FovealClassifier()
    pairs = clf._normalize(["dog", ("cat", "a feline")])
    assert pairs == [("dog", "a photo of dog"), ("cat", "a feline")]


def test_general_vocab_nonempty():
    assert len(GENERAL_VOCAB) >= 10
    assert "person" in GENERAL_VOCAB


def test_classify_picks_best_match():
    # image vec aligns with first text vec
    clf = make_clf([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    name, conf = clf.classify(object(), ["aaa", "bbb"])
    assert name == "aaa"
    assert conf > 0.9


def test_unknown_below_threshold():
    # 8 evenly spread labels, soft logit scale -> near-uniform, top < 0.30
    import math as _m
    vecs = [[_m.cos(a), _m.sin(a)] for a in
            [i * _m.pi / 4 for i in range(8)]]
    clf = make_clf([1.0, 1.0], vecs, logit_scale=0.1)
    name, conf = clf.classify(object(), list("abcdefgh"))
    assert name == "unknown"
    assert conf < 0.30


def test_distribution_sums_to_one():
    clf = make_clf([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    dist = clf.distribution(object(), ["aaa", "bbb"])
    assert math.isclose(sum(p for _, p in dist), 1.0, rel_tol=1e-5)
    assert dist[0][1] >= dist[1][1]  # ranked


def test_text_cache_keyed_by_prompt():
    clf = FovealClassifier()
    clf._ensure = lambda: None
    clf._processor = lambda **kwargs: {}
    calls = []

    class CountingModel(StubModel):
        def get_text_features(self, **kwargs):
            calls.append(1)
            v = torch.tensor([[1.0, 0.0]])
            return StubModel._Out(v / v.norm(dim=-1, keepdim=True))

    clf._model = CountingModel([1.0, 0.0])
    crop = object()
    clf.distribution(crop, ["x"])
    clf.distribution(crop, ["x"])  # same prompts -> cache hit
    clf.distribution(crop, ["y"])  # new prompts -> recompute
    assert len(calls) == 2  # second ["x"] served from cache


def test_owlvit_detects_window_in_crop():
    # EMPIRICAL INTEGRATION TEST. Fixture regenerated 2026-10-03
    # (Mykal's call): tests/fixtures/foveal_crops.png is a 4x3 montage
    # (330px cells, title strip at each cell top, per extract_crops)
    # of hand-labeled crops from current pipeline-relevant footage
    # (Star Tours ride film + OWL-ViT probe clips). Labels live in
    # tests/fixtures/foveal_labels.json. crop[10] is hand-labeled as
    # containing a window -- the assertion checks exactly that.
    # NOTE (2026-10-01 live finding): the detector starves on 224px
    # grayscale reflex frames (0.10-0.18 scores) -- the montage crops
    # are full-res color, which is why this works.
    pytest.importorskip("transformers")
    from hvp.detect import ObjectDetector
    from scripts.audit_recognition import extract_crops
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    montage = os.path.join(repo, "tests", "fixtures", "foveal_crops.png")
    assert os.path.exists(montage), f"missing fixture {montage}"
    crops = extract_crops(montage)
    crop = crops[10].convert("RGB").resize((224, 224))  # hand-labeled
    det = ObjectDetector()                              # "windows"
    dets = det.detect(crop, ["a window", "a gate", "a sign"],
                      threshold=0.05)
    assert dets, "no detections at all"
    top = dets[0]
    assert top[0] == "a window", f"top detection was {top[0]}"
    assert top[5] > 0.10
    mx, my = ObjectDetector.box_center_map(top, 224, 224)
    assert 0 <= mx <= 56 and 0 <= my <= 56
