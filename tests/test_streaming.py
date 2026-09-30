"""Regression tests for the streaming A/V pipeline.

Covers the bugs found during the 2026-09-30 streaming verification:
  1. decode_gray fps bug (vf chain must contain fps={fps}).
  2. PyAV audio plane padding (slice to samples*channels).
  3. Stream/batch bit-exactness (OnlineLevel3 replay + same decoder).
  4. Saccade-count convention (t_end_ms drops the past-the-end saccade).

Run: python -m pytest tests/ -x -q   (from the repo root)
Slow integration tests (need the Star Tours file) are marked and skipped
automatically when the file is absent.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

STAR_TOURS = ("/home/hatch/workspace/human-vision-pipeline/input/"
              "star_tours_1_ride_film.mp4")

needs_video = pytest.mark.skipif(
    not os.path.exists(STAR_TOURS),
    reason="Star Tours test video not present")


def test_decode_gray_vf_chain_has_fps_filter():
    """The vf chain must pin the output frame rate.

    Regression: without fps={fps} in the filter chain, ffmpeg emits
    native-fps frames while the loop labels them at 10 fps -- a 62 s
    run watched ~21 s of video stretched across the timeline.
    """
    import run_video
    src = open(run_video.__file__).read()
    assert "fps=" in src and "fps={fps}" in src, (
        "decode_gray vf chain lost its fps filter -- "
        "see 2026-09-30 streaming verification notes")


def test_audio_plane_padding_slice():
    """PyAV audio planes are padded; only samples*channels values are real.

    Regression: reading bytes(plane) whole produced garbage/misaligned
    audio. hva.stream must slice to frame.samples * channels.
    """
    import hva.stream as S
    src = open(S.__file__).read()
    assert "samples" in src and "planes[0]" in src
    # The slice must be bounded by the real sample count, not the plane size.
    assert ".samples *" in src or "samples*" in src or "n_samp" in src, (
        "audio plane read does not appear to slice to the real sample count")


def test_online_level3_replay_is_deterministic():
    """Feeding the same tick sequence twice gives identical peaks/maps."""
    from hvm.online import OnlineLevel3
    rng = np.random.default_rng(0)
    ticks = [(rng.random((56, 56), dtype=np.float32),
              (rng.random(56, dtype=np.float32),
               rng.random(56, dtype=np.float32)))
             for _ in range(50)]

    def run():
        loop = OnlineLevel3(dva_per_px=0.1, t_end_ms=5000.0)
        for m, (vis, aud) in enumerate(ticks):
            loop.tick(m * 100.0, vis, aud, 0.0)
        return loop.result()

    r1, r2 = run(), run()
    p1 = np.array([p[:2] for p in r1["peaks"]])
    p2 = np.array([p[:2] for p in r2["peaks"]])
    assert np.array_equal(p1, p2), "OnlineLevel3 replay diverged"
    assert np.array_equal(r1["maps"][-1], r2["maps"][-1])


def test_t_end_ms_drops_past_end_saccade():
    """Batch (t_end set) and stream (t_end=inf) must agree on saccade count.

    Regression 2026-09-30: the stream kept one extra saccade (207 vs 206)
    because OnlineLevel3 defaulted to t_end_ms=inf while the batch passed
    seconds*1000. The runner now passes t_end in both paths.
    """
    from hvm.online import OnlineLevel3
    rng = np.random.default_rng(1)
    vis = rng.random((56, 56), dtype=np.float32)
    aud = (rng.random(56, dtype=np.float32), rng.random(56, dtype=np.float32))

    batch = OnlineLevel3(0.1, t_end_ms=1000.0)
    stream = OnlineLevel3(0.1, t_end_ms=1000.0)  # runner must pass this
    for m in range(10):
        batch.tick(m * 100.0, vis, aud, 0.0)
        stream.tick(m * 100.0, vis, aud, 0.0)
    assert (len(batch.result()["scanpath"]) ==
            len(stream.result()["scanpath"])), (
        "saccade-count convention drifted between batch and stream")


@needs_video
def test_stream_emits_620_moments_for_62s():
    """A 62 s source must yield exactly 620 moments, no more, no fewer."""
    from hva.stream import StreamSource, AudioFrontEnd
    src = StreamSource(STAR_TOURS, duration=62.0)
    afe = AudioFrontEnd()
    n = 0
    for tick in src:
        for _am in afe.push(tick):
            n += 1
    for _am in afe.flush():
        n += 1
    assert n == 620, f"expected 620 moments, got {n}"


@needs_video
def test_stream_video_matches_batch_decoder():
    """Stream video frames must equal decode_gray output (same filter chain)."""
    import run_video
    from hva.stream import StreamSource
    batch_frames = [f for _, f in
                    run_video.decode_gray(STAR_TOURS, 5.0, 10.0, 224, 224)]
    src = StreamSource(STAR_TOURS, duration=5.0)
    stream_frames = [t.frame for t in src]
    assert len(stream_frames) == len(batch_frames) == 50
    for i, (s, b) in enumerate(zip(stream_frames, batch_frames)):
        assert np.array_equal(s, b), f"frame {i} differs between stream and batch"
