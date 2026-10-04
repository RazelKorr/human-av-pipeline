"""Tests for the streaming harness (streaming/).

These cover the harness only. The pipeline itself is untouched, so the
existing suite must keep passing unmodified.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "streaming"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from feeder import ChunkedFeeder
from online_driver import OnlineAttentionDriver
from online_render import OnlineVisionPipeline
from hvp.pipeline import VisionPipeline
from hvp import baseline as B


@pytest.fixture(scope="module")
def tiny_mp4(tmp_path_factory):
    """2 s, 30 fps, 128x96 test video."""
    p = str(tmp_path_factory.mktemp("vid") / "tiny.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
         "-i", "testsrc=size=128x96:rate=30:duration=2",
         "-pix_fmt", "yuv420p", p], check=True)
    return p


def test_feeder_frame_count_and_timestamps(tiny_mp4):
    from run_video_color import decode_color
    f = ChunkedFeeder(tiny_mp4, 2.0, 30.0, attn_wh=(224, 224),
                      work_wh=(64, 48), chunk_s=0.5)
    got = []
    for ch in f:
        got.extend(ch.frames)
    assert len(got) == 60, f"expected 60 frames, got {len(got)}"
    assert f.total_frames == 60
    # timestamps match the batch decoder exactly
    ref = [t for t, _ in decode_color(tiny_mp4, 2.0, 30.0, 224, 224)]
    assert len(ref) == 60
    for (t_ms, _fa, _fw, _fm), t_ref in zip(got, ref):
        assert abs(t_ms - t_ref) < 1e-9
    # chunking covers the timeline without gaps
    chunks = list(ChunkedFeeder(tiny_mp4, 2.0, 30.0, chunk_s=0.5))
    assert chunks[0].t0_s == 0.0
    assert abs(chunks[-1].t1_s - 60 / 30.0) < 1e-9


def test_feeder_pipes_are_lockstep(tiny_mp4):
    f = ChunkedFeeder(tiny_mp4, 2.0, 30.0, work_wh=(64, 48))
    for ch in f:
        for (t_ms, fa, fw, _fm) in ch.frames:
            assert fa.shape == (224, 224, 3)
            assert fw.shape == (48, 64, 3)


def test_driver_is_deterministic():
    rng = np.random.default_rng(7)
    frames = [(i * 100.0 / 3,
               rng.random((224, 224, 3), dtype=np.float32))
              for i in range(30)]  # 10 s at 3 fps (cheap)
    outs = []
    for _ in range(2):
        d = OnlineAttentionDriver(3.0, 10_000.0)
        sacc = []
        for t, fr in frames:
            sacc.extend(d.push(t, fr))
        outs.append((sacc, np.array(d.energies), d.n_saccades))
    assert outs[0][2] == outs[1][2]
    assert np.array_equal(outs[0][1], outs[1][1])
    assert outs[0][0] == outs[1][0]


def _synth_frames(n=40, fps=30.0, w=64, h=48, seed=3):
    rng = np.random.default_rng(seed)
    return [(i * 1000.0 / fps,
             rng.random((h, w, 3), dtype=np.float32)) for i in range(n)]


def test_online_render_matches_batch_bit_exact():
    """Incremental push/pull == batch push-all/pull-all, bit for bit."""
    frames = _synth_frames()
    t_last = frames[-1][0]
    w, h = 64, 48
    dva = B.FIELD_WIDTH_DEG / w
    script = [(0, w / 2.0, h / 2.0)]
    batch = VisionPipeline((h, w, 3), dva, script, moment_ms=50.0)
    for t, f in frames:
        batch.push(f, t)
    b_moms = list(batch.moments(t_last))

    online = OnlineVisionPipeline((h, w, 3), dva, script, moment_ms=50.0)
    o_moms = []
    for t, f in frames:
        online.push(f, t)
        o_moms.extend(online.pull(t, t_last))
    o_moms.extend(online.pull(t_last, t_last))

    assert len(o_moms) == len(b_moms), \
        f"{len(o_moms)} vs {len(b_moms)} moments"
    for (tb, pb, mb), (to, po, mo) in zip(b_moms, o_moms):
        assert tb == to
        assert np.array_equal(pb, po), f"percept differs at t={tb}"
        assert mb["suppressed"] == mo["suppressed"]
        assert mb["fixation"] == mo["fixation"]
        assert mb["supp_frac"] == mo["supp_frac"]


def test_bounded_buffer_stays_exact_and_bounded():
    """Aggressive eviction must not change moments, and must bound memory."""
    frames = _synth_frames(n=90, fps=30.0)  # 3 s
    t_last = frames[-1][0]
    w, h = 64, 48
    dva = B.FIELD_WIDTH_DEG / w
    script = [(0, w / 2.0, h / 2.0)]
    batch = VisionPipeline((h, w, 3), dva, script, moment_ms=50.0)
    for t, f in frames:
        batch.push(f, t)
    b_moms = [(t, p) for t, p, _ in batch.moments(t_last)]

    online = OnlineVisionPipeline((h, w, 3), dva, script, moment_ms=50.0,
                                  keep_margin_ms=0.0)
    o_moms = []
    max_buf = 0
    for t, f in frames:
        online.push(f, t)
        max_buf = max(max_buf, len(online._times))
        o_moms.extend(online.pull(t, t_last))
    assert max_buf < len(frames), \
        f"buffer never evicted ({max_buf} frames retained)"
    assert online.n_evicted > 0
    assert len(o_moms) == len(b_moms)
    for (tb, pb), (to, po, _mo) in zip(b_moms, o_moms):
        assert tb == to and np.array_equal(pb, po)


def test_pull_t_end_cap_no_extra_tail_moment():
    """pull() must not emit past t_end (the batch grid cap)."""
    frames = _synth_frames(n=30, fps=30.0)
    t_last = frames[-1][0]
    w, h = 64, 48
    dva = B.FIELD_WIDTH_DEG / w
    script = [(0, w / 2.0, h / 2.0)]
    online = OnlineVisionPipeline((h, w, 3), dva, script, moment_ms=50.0)
    moms = []
    for t, f in frames:
        online.push(f, t)
        moms.extend(online.pull(t, t_last))
    moms.extend(online.pull(t_last, t_last))
    assert all(m[0] <= t_last + 1e-9 for m in moms)
    # and the count matches the batch grid exactly
    batch = VisionPipeline((h, w, 3), dva, script, moment_ms=50.0)
    for t, f in frames:
        batch.push(f, t)
    assert len(moms) == len(list(batch.moments(t_last)))


def _collect_feeder(video, seconds, fps, **kw):
    f = ChunkedFeeder(video, seconds, fps, chunk_s=seconds, **kw)
    frames = []
    for ch in f:
        frames.extend(ch.frames)
    return frames


def test_single_split_matches_dual_decode_bit_identical(tiny_mp4):
    """single_mode='split' must reproduce dual_decode frames exactly."""
    dual = _collect_feeder(tiny_mp4, 2.0, 30.0, dual_decode=True)
    split = _collect_feeder(tiny_mp4, 2.0, 30.0, single_mode="split")
    assert len(dual) == len(split) == 60
    for (t1, fa1, fw1, _), (t2, fa2, fw2, _) in zip(dual, split):
        assert abs(t1 - t2) < 1e-9
        assert np.array_equal(fa1, fa2), "attn frames differ"
        assert np.array_equal(fw1, fw2), "work frames differ"


def test_single_pil_frame_count_timestamp_parity(tiny_mp4):
    """single_mode='pil': frame count + timestamps match dual_decode;
    work frames identical (same pipe); attn frames close."""
    dual = _collect_feeder(tiny_mp4, 2.0, 30.0, dual_decode=True)
    pil = _collect_feeder(tiny_mp4, 2.0, 30.0, single_mode="pil")
    assert len(dual) == len(pil) == 60
    for (t1, fa1, fw1, _), (t2, fa2, fw2, _) in zip(dual, pil):
        assert abs(t1 - t2) < 1e-9
        assert np.array_equal(fw1, fw2), "work frames differ"
    max_attn = max(np.abs(a[1] - b[1]).max() for a, b in zip(dual, pil))
    mean_attn = float(np.mean([np.abs(a[1] - b[1]).mean()
                               for a, b in zip(dual, pil)]))
    # sanity bound only: synthetic testsrc edges are the worst case
    # for resampler agreement (real footage measured max 0.11);
    # tight agreement is validated on real footage, not here.
    assert mean_attn < 0.05, f"attn frames diverged too far: {mean_attn}"
    assert max_attn < 0.5, f"attn frames diverged too far: {max_attn}"


def test_single_split_is_deterministic(tiny_mp4):
    a = _collect_feeder(tiny_mp4, 2.0, 30.0, single_mode="split")
    b = _collect_feeder(tiny_mp4, 2.0, 30.0, single_mode="split")
    assert len(a) == len(b)
    for (t1, fa1, fw1, _), (t2, fa2, fw2, _) in zip(a, b):
        assert t1 == t2 and np.array_equal(fa1, fa2) \
            and np.array_equal(fw1, fw2)


def test_feeder_rejects_bad_single_mode(tiny_mp4):
    import pytest
    with pytest.raises(ValueError):
        ChunkedFeeder(tiny_mp4, 2.0, 30.0, single_mode="bogus")


def test_split_three_outputs_frame_parity(tiny_mp4):
    """decode_split with w3/h3: three lockstep outputs, timestamps
    identical, shapes as requested, no deadlock on a 60-frame clip."""
    from streaming.feeder import decode_split
    n = 0
    for (t1, f1), (t2, f2), tm in decode_split(
            tiny_mp4, 2.0, 30.0, 64, 48, 32, 32, 96, 54):
        assert tm is not None
        t3, f3 = tm
        assert t1 == t2 == t3
        assert f1.shape == (48, 64, 3)
        assert f2.shape == (32, 32, 3)
        assert f3.shape == (54, 96, 3)
        n += 1
    assert n == 60


def test_split_two_outputs_unchanged(tiny_mp4):
    """decode_split without w3/h3 keeps the two-output contract."""
    from streaming.feeder import decode_split
    n = 0
    for (t1, f1), (t2, f2), tm in decode_split(
            tiny_mp4, 2.0, 30.0, 64, 48, 32, 32):
        assert tm is None
        assert t1 == t2
        n += 1
    assert n == 60


def test_feeder_thumb_none_by_default(tiny_mp4):
    """chunk.frames carry None thumbs unless motion_thumb_wh is set."""
    frames = _collect_feeder(tiny_mp4, 2.0, 30.0, single_mode="split")
    assert all(fm is None for (_, _, _, fm) in frames)
    frames = _collect_feeder(tiny_mp4, 2.0, 30.0, single_mode="split",
                             motion_thumb_wh=(96, 54))
    assert len(frames) == 60
    for (t, fa, fw, fm) in frames:
        assert fm is not None and fm.shape == (54, 96, 3)
        assert fa.shape == (224, 224, 3)
    # and the attn/work outputs are untouched by the third branch
    plain = _collect_feeder(tiny_mp4, 2.0, 30.0, single_mode="split")
    for (t1, fa1, fw1, fm1), (t2, fa2, fw2, _) in zip(frames, plain):
        assert t1 == t2
        assert np.array_equal(fa1, fa2), "attn changed by thumb branch"
        assert np.array_equal(fw1, fw2), "work changed by thumb branch"


def test_feeder_rejects_bad_thumb_size(tiny_mp4):
    import pytest
    with pytest.raises(ValueError):
        ChunkedFeeder(tiny_mp4, 2.0, 30.0,
                      motion_thumb_wh=(0, 54))


def _fake_args(**kw):
    import argparse
    ns = argparse.Namespace()
    ns.speed = 1.0
    ns.motion_weight = None
    ns.pursuit = False
    ns.motion_thumb = None
    ns.dual_decode = False
    ns.chunk_s = 2.0
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_validate_args_rejects_bad_combos():
    import run_stream
    with __import__("pytest").raises(SystemExit):
        run_stream._validate_args(_fake_args(speed=0.0))
    with __import__("pytest").raises(SystemExit):
        run_stream._validate_args(_fake_args(motion_weight=-1.0))
    # sane combos pass silently
    run_stream._validate_args(_fake_args())
    run_stream._validate_args(_fake_args(motion_weight=0.5, pursuit=True))
    run_stream._validate_args(_fake_args(dual_decode=True))


def test_feeder_rejects_nonpositive_speed(tiny_mp4):
    import pytest
    with pytest.raises(ValueError):
        ChunkedFeeder(tiny_mp4, 2.0, 30.0, speed=0.0)


def test_driver_rejects_bad_motion_thumb():
    import pytest
    with pytest.raises(ValueError):
        OnlineAttentionDriver(30.0, 2000.0, motion_weight=1.0,
                              motion_thumb_wh=(0, 54))


def test_pursuit_silently_off_without_motion():
    d = OnlineAttentionDriver(30.0, 2000.0, pursuit=True)
    assert d.motion is None
    assert d.pursuit is False
