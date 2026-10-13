"""Fixture-regime guards: the corpus is a system, not an accident.

The historical lesson (audit-2026-10-12, Lane 4): dialogue-heavy
fixtures with ~16x less motion energy than Star Tours hid the
motion pathway for weeks, because nothing in the suite measured
the fixtures themselves. These tests measure the fixtures:

  (a) regime ordering -- on a fast frame-diff proxy, the
      motion-rich fixture (Star Tours) must sit clearly above
      the dialogue fixture (Dr Tran ep4). If a future fixture
      swap inverts this, the suite says so instead of quietly
      re-hiding a pathway.
  (b) the generated quiet/near-silent fixture
      (scripts/make_fixtures.py) must actually be quiet: motion
      proxy in the quiet band, audio floor below the streaming
      ear's RMS gate, and the one planted chime detectable where
      it was planted.
  (c) the motion pathway's unit-level exercise lives in
      tests/test_motion.py (Phase 3, sealed predictions P8.5:
      driver-level legacy-vs-residual pursuit on synthetic
      streams). It is referenced here, not duplicated -- this
      test only asserts those guards still exist, so the
      pathway cannot silently leave the rotation.

Media are gitignored; every media test skips when its file is
absent. The corpus table (measured properties, provenance) is
tests/fixtures/FIXTURES.md.

Run: python -m pytest tests/test_fixture_regimes.py -q
"""
import functools
import importlib.util
import os
import shutil
import subprocess

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(REPO, "input")

STAR_TOURS = os.path.join(INPUT, "star_tours_1_ride_film.mp4")
DRTRAN_EP4 = os.path.join(INPUT, "drtran_ep4_toy_cack.mp4")
QUIET_ROOM = os.path.join(INPUT, "quiet_room_60s.mp4")

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


@functools.lru_cache(maxsize=None)
def motion_proxy(path, seconds=120.0):
    """Mean abs frame difference of a bounded sample: frames at
    2 fps, 320x180 gray, from the start of the file. The same
    proxy is tabulated in tests/fixtures/FIXTURES.md. Not the
    pipeline's motion energy -- a cheap, stable stand-in whose
    only job is regime ordering."""
    w, h, fps = 320, 180, 2
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(seconds),
           "-vf", f"fps={fps},scale={w}:{h},format=gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    assert n >= 10, f"proxy decoded only {n} frames from {path}"
    frames = (np.frombuffer(raw[:n * w * h], dtype=np.uint8)
              .reshape(n, w * h).astype(np.float32) / 255.0)
    return float(np.abs(np.diff(frames, axis=0)).mean())


@needs_ffmpeg
@pytest.mark.skipif(not os.path.exists(STAR_TOURS),
                    reason="Star Tours fixture not present")
@pytest.mark.skipif(not os.path.exists(DRTRAN_EP4),
                    reason="Dr Tran ep4 fixture not present")
def test_motion_regime_ordering_star_tours_above_dialogue():
    st = motion_proxy(STAR_TOURS)
    ep4 = motion_proxy(DRTRAN_EP4)
    assert st > ep4, (
        f"regime inversion: Star Tours proxy {st:.5f} <= Dr Tran "
        f"ep4 {ep4:.5f} -- the motion-rich/dialogue fixture roles "
        "have swapped; see tests/fixtures/FIXTURES.md")
    assert st >= 1.25 * ep4, (
        f"regime separation too thin: Star Tours {st:.5f} vs Dr "
        f"Tran ep4 {ep4:.5f} (measured 1.83x at systematization, "
        "2026-10-12); a thin margin means fixture choice can "
        "again hide the motion pathway")


@needs_ffmpeg
@pytest.mark.skipif(not os.path.exists(QUIET_ROOM),
                    reason="quiet fixture not generated "
                           "(python3 scripts/make_fixtures.py)")
def test_quiet_fixture_motion_in_quiet_band():
    q = motion_proxy(QUIET_ROOM, seconds=60.0)
    assert q < 0.005, (
        f"quiet fixture motion proxy {q:.5f} outside the quiet "
        "band (< 0.005; measured 0.00122 at generation) -- the "
        "fixture is no longer quiet; regenerate or re-seal")


@needs_ffmpeg
@pytest.mark.skipif(not os.path.exists(QUIET_ROOM),
                    reason="quiet fixture not generated "
                           "(python3 scripts/make_fixtures.py)")
def test_quiet_fixture_audio_floor_and_planted_chime(tmp_path):
    import soundfile as sf
    wav = str(tmp_path / "quiet.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", QUIET_ROOM,
                    "-ac", "1", "-ar", "16000", wav], check=True)
    x, sr = sf.read(wav)
    floor = x[int(2 * sr):int(28 * sr)]
    floor_rms = float(np.sqrt((floor ** 2).mean()))
    assert floor_rms < 8e-4, (
        f"quiet floor RMS {floor_rms:.6f} too high: it must sit "
        "below the streaming ear's 1e-3 RMS gate (measured "
        "0.00050 at generation)")
    # loudest 100 ms window must be the planted chime at t=30 s
    w = int(0.1 * sr)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="same"))
    t_peak = float(np.argmax(rms)) / sr
    assert 29.5 <= t_peak <= 31.0, (
        f"loudest window at {t_peak:.2f} s, not the planted "
        "chime (t = 30.0 s)")
    assert rms.max() > 5 * floor_rms, (
        f"planted chime barely above floor: peak window RMS "
        f"{rms.max():.5f} vs floor {floor_rms:.6f}")


def test_motion_pathway_guarded_by_phase3_driver_tests():
    """Reference, not duplication: Phase 3's driver-level pursuit
    tests (synthetic streams; residual vs legacy) are the motion
    pathway's unit-level exercise. This asserts they still exist
    under their sealed-prediction names."""
    path = os.path.join(REPO, "tests", "test_motion.py")
    spec = importlib.util.spec_from_file_location("test_motion_ref",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("test_driver_egomotion_pursuit_legacy_vs_residual",
                 "test_driver_residual_pursues_local_mover_static_bg",
                 "test_driver_residual_pursues_mover_on_moving_bg"):
        assert callable(getattr(mod, name, None)), (
            f"Phase 3 motion-pathway guard {name} missing from "
            "tests/test_motion.py -- the motion pathway must stay "
            "exercised by the suite (audit Lane 4 lesson)")
