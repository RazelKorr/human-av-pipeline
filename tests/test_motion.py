"""Tests for the magno channel (motion energy) and smooth pursuit."""

import numpy as np
import pytest

from hvp import attention as A
from hvp import baseline as B
from hvp import motion as M
from hvp.saccades import SaccadeController
from streaming.online_driver import OnlineAttentionDriver


def _disk_frame(cx, cy, r=20, size=224, bg=0.05, fg=0.9):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.hypot(xx - cx, yy - cy)
    lum = np.full((size, size), bg, dtype=np.float32)
    lum[d < r] = fg
    return np.stack([lum, lum, lum], axis=-1)


def test_motion_channel_global_flood_is_gated():
    mc = M.MotionChannel(56, 56)
    dark = np.zeros((56, 56), dtype=np.float32)
    white = np.ones((56, 56), dtype=np.float32)
    mc.push(dark, 33.3)
    e = mc.push(white, 33.3)
    # full-frame flash: raw floods, coherence gate must zero it
    assert e.shape == (56, 56)
    assert e.dtype == np.float32
    assert float(e.max()) < 1e-6, f"flood leaked: max={e.max()}"


def test_motion_channel_local_blob_survives():
    mc = M.MotionChannel(56, 56)
    a = np.zeros((56, 56), dtype=np.float32)
    b = np.zeros((56, 56), dtype=np.float32)
    yy, xx = np.mgrid[0:56, 0:56].astype(np.float32)
    b[np.hypot(xx - 30, yy - 28) < 5] = 0.8
    mc.push(a, 33.3)
    e = mc.push(b, 33.3)
    assert float(e[28, 30]) > 0.1, "local motion blob was gated away"
    assert float(e[5, 5]) == 0.0, "energy leaked far from the blob"


def test_motion_channel_threshold_kills_noise():
    mc = M.MotionChannel(56, 56)
    rng = np.random.default_rng(0)
    a = rng.random((56, 56), dtype=np.float32) * 0.01
    b = rng.random((56, 56), dtype=np.float32) * 0.01
    mc.push(a, 33.3)
    e = mc.push(b, 33.3)
    assert float(e.max()) == 0.0


def test_motion_channel_first_push_is_zero():
    mc = M.MotionChannel(56, 56)
    e = mc.push(np.ones((56, 56), dtype=np.float32), 33.3)
    assert float(e.max()) == 0.0


def test_velocity_at_constant_motion():
    mc = M.MotionChannel(56, 56)
    # disk gliding right at 0.02 map-px/ms
    for k in range(14):
        f = np.zeros((56, 56), dtype=np.float32)
        yy, xx = np.mgrid[0:56, 0:56].astype(np.float32)
        f[np.hypot(xx - (10 + 0.02 * k * 33.3), yy - 28) < 4] = 0.8
        mc.push(f, 33.3, t_ms=k * 33.3)
    v = mc.velocity_at(20.0, 28.0)
    assert v is not None, "no velocity for clean constant motion"
    assert abs(v[0] - 0.02) < 0.008, f"vx wrong: {v}"
    assert abs(v[1]) < 0.008, f"vy should be ~0: {v}"


def test_velocity_at_reversing_motion_is_none():
    mc = M.MotionChannel(56, 56)
    for k in range(14):
        f = np.zeros((56, 56), dtype=np.float32)
        yy, xx = np.mgrid[0:56, 0:56].astype(np.float32)
        cx = 10 + (0.02 * k * 33.3 if k < 7 else 0.02 * (14 - k) * 33.3)
        f[np.hypot(xx - cx, yy - 28) < 4] = 0.8
        mc.push(f, 33.3, t_ms=k * 33.3)
    assert mc.velocity_at(20.0, 28.0) is None


def test_velocity_at_static_is_none():
    mc = M.MotionChannel(56, 56)
    f = np.zeros((56, 56), dtype=np.float32)
    f[25:30, 25:30] = 0.8
    for k in range(14):
        mc.push(f, 33.3, t_ms=k * 33.3)
    assert mc.velocity_at(27.0, 27.0) is None


def test_salience_map_motion_off_is_identical():
    rng = np.random.default_rng(1)
    small = rng.random((56, 56), dtype=np.float32)
    trans = rng.random((56, 56), dtype=np.float32) * 0.1
    rgb = rng.random((56, 56, 3), dtype=np.float32)
    dva = B.FIELD_WIDTH_DEG / 224
    ctl = SaccadeController([(0, 112.0, 112.0)], dva)
    m = rng.random((56, 56), dtype=np.float32)
    base = A.salience_map(small, trans, 500.0, ctl, [])
    assert np.array_equal(
        base, A.salience_map(small, trans, 500.0, ctl, [],
                             motion_map=m, motion_weight=None))
    assert np.array_equal(
        base, A.salience_map(small, trans, 500.0, ctl, [],
                             motion_map=m, motion_weight=0.0))
    boosted = A.salience_map(small, trans, 500.0, ctl, [],
                             motion_map=m, motion_weight=1.0,
                             small_rgb=rgb)
    plain = A.salience_map(small, trans, 500.0, ctl, [], small_rgb=rgb)
    assert (boosted >= plain).all(), "motion must only add salience"


def test_pursuit_glides_linearly_no_suppression():
    ctl = SaccadeController([(0, 0.0, 0.0)], 0.1)
    ctl.add_pursuit(100.0, 1100.0, 0.05, -0.02)
    for dt in (0.0, 250.0, 500.0, 999.0):
        fx, fy, supp = ctl.state_at(100.0 + dt)
        assert not supp, "pursuit must never suppress"
        assert abs(fx - 0.05 * dt) < 1e-6
        assert abs(fy - (-0.02 * dt)) < 1e-6
    # after the glide ends, fixation rests at the end position
    fx, fy, supp = ctl.state_at(1500.0)
    assert not supp
    assert abs(fx - 0.05 * 1000.0) < 1e-6
    assert abs(fy - (-0.02 * 1000.0)) < 1e-6


def test_saccade_interrupts_pursuit():
    ctl = SaccadeController([(0, 0.0, 0.0)], 0.1)
    ctl.add_pursuit(100.0, 1100.0, 0.05, 0.0)
    # saccade at t=600 from the glide position (25.0, 0.0)
    ctl.add_saccade(600.0, 100.0, 0.0)
    fx, fy, supp = ctl.state_at(620.0)
    assert supp, "mid-saccade must suppress"
    assert abs(fx - 25.0) < 1e-6, "saccade must launch from glide position"
    # pursuit was truncated: after the saccade lands, no more gliding
    fx2, fy2, _ = ctl.state_at(5000.0)
    assert abs(fx2 - 100.0) < 1e-6 and abs(fy2) < 1e-6


def test_pursuit_then_saccade_sequence():
    ctl = SaccadeController([(0, 50.0, 50.0)], 0.1786)
    ctl.add_pursuit(200.0, 700.0, 0.1, 0.0)
    fx, fy, _ = ctl.state_at(500.0)
    assert abs(fx - (50.0 + 0.1 * 300.0)) < 1e-6
    ctl.add_saccade(800.0, 10.0, 10.0)
    fx, fy, supp = ctl.state_at(810.0)
    assert supp
    fx, fy, _ = ctl.state_at(5000.0)
    assert abs(fx - 10.0) < 1e-6 and abs(fy - 10.0) < 1e-6


def test_add_pursuit_rejects_bad_window():
    ctl = SaccadeController([(0, 0.0, 0.0)], 0.1)
    with pytest.raises(ValueError):
        ctl.add_pursuit(500.0, 500.0, 0.1, 0.1)
    with pytest.raises(ValueError):
        ctl.add_pursuit(600.0, 500.0, 0.1, 0.1)


def test_driver_motion_off_matches_baseline():
    rng = np.random.default_rng(7)
    frames = [(i * 100.0 / 3, rng.random((224, 224, 3), dtype=np.float32))
              for i in range(30)]
    outs = []
    for kw in ({}, {"motion_weight": None}, {"motion_weight": 0.0}):
        d = OnlineAttentionDriver(3.0, 10_000.0, **kw)
        sacc = []
        for t, fr in frames:
            sacc.extend(d.push(t, fr))
        outs.append(sacc)
    assert outs[0] == outs[1] == outs[2]
    # pursuit=True without motion must not engage anything
    d = OnlineAttentionDriver(3.0, 10_000.0, pursuit=True)
    assert d.pursuit is False
    for t, fr in frames:
        d.push(t, fr)
    assert d.n_pursuits == 0


def test_driver_motion_sees_moving_disk_not_static():
    fps = 30.0
    d_move = OnlineAttentionDriver(fps, 20_000.0, motion_weight=1.0)
    d_static = OnlineAttentionDriver(fps, 20_000.0, motion_weight=1.0)
    for i in range(90):  # 3 s
        t = i * 1000.0 / fps
        cx = 40.0 + i * 1.5  # 45 px/s rightward
        d_move.push(t, _disk_frame(cx, 112.0))
        d_static.push(t, _disk_frame(112.0, 112.0))
    me_move = np.array([e for _, e in d_move.motion_energies])
    me_static = np.array([e for _, e in d_static.motion_energies])
    assert me_move.mean() > 5 * (me_static.mean() + 1e-9), \
        "motion channel must fire on moving disk, not static"


def test_driver_no_pursuit_on_global_flash():
    fps = 30.0
    d = OnlineAttentionDriver(fps, 20_000.0, motion_weight=1.0, pursuit=True)
    for i in range(120):  # 4 s of full-frame flashing
        t = i * 1000.0 / fps
        v = 0.9 if i % 2 == 0 else 0.05
        f = np.full((224, 224, 3), v, dtype=np.float32)
        d.push(t, f)
    assert d.n_pursuits == 0, "global flash must never lock pursuit"


def test_driver_pursues_constant_velocity_target():
    fps = 30.0
    d = OnlineAttentionDriver(fps, 30_000.0, motion_weight=1.0, pursuit=True)
    # bright disk gliding right at 45 px/s for 8 s; eyes should find
    # it (bright on dark) and then pursue it
    for i in range(240):
        t = i * 1000.0 / fps
        d.push(t, _disk_frame(30.0 + i * 1.5, 112.0))
    assert d.n_pursuits > 0, "pursuit never engaged on clean linear motion"
    # no suppression during any pursuit segment
    for (t0, t1, vx, vy) in d.pursuit_log:
        for tt in np.arange(t0 + 1, t1, 50.0):
            _, _, supp = d.controller.state_at(tt)
            assert not supp, f"suppressed during pursuit at t={tt}"
    # pursuit velocities point rightward, roughly the stimulus speed
    vxs = [vx for (_, _, vx, _) in d.pursuit_log]
    assert sum(vxs) / len(vxs) > 0, "pursuit should go rightward"


# --- Thumb parameterization & 16:9 geometry (2026-10-03) ---

def test_motion_channel_default_is_16x9():
    mc = M.MotionChannel()
    assert mc.thumb_w / mc.thumb_h == pytest.approx(16 / 9, rel=0.02)
    e = mc.push(np.zeros((mc.thumb_h, mc.thumb_w), dtype=np.float32), 33.3)
    assert e.shape == (mc.thumb_h, mc.thumb_w)


def test_motion_channel_thumb_shape_and_rejection():
    mc = M.MotionChannel(160, 90)
    e = mc.push(np.zeros((90, 160), dtype=np.float32), 33.3)
    assert e.shape == (90, 160)
    assert e.dtype == np.float32
    with pytest.raises(ValueError):
        mc.push(np.zeros((56, 56), dtype=np.float32), 33.3)


def test_motion_channel_flood_gated_at_all_sizes():
    for tw, th in [(96, 54), (160, 90), (480, 270)]:
        mc = M.MotionChannel(tw, th)
        mc.push(np.zeros((th, tw), dtype=np.float32), 33.3)
        e = mc.push(np.ones((th, tw), dtype=np.float32), 33.3)
        assert float(e.max()) < 1e-6, f"flood leaked at {tw}x{th}"


def test_motion_channel_local_blob_survives_16x9():
    mc = M.MotionChannel(160, 90)
    a = np.zeros((90, 160), dtype=np.float32)
    b = np.zeros((90, 160), dtype=np.float32)
    yy, xx = np.mgrid[0:90, 0:160].astype(np.float32)
    b[np.hypot(xx - 80, yy - 45) < 8] = 0.8
    mc.push(a, 33.3)
    e = mc.push(b, 33.3)
    assert float(e[45, 80]) > 0.1, "local motion blob was gated away"
    assert float(e[5, 5]) == 0.0, "energy leaked far from the blob"


def test_velocity_scales_with_thumb_width():
    # same world motion (deg/s) at two thumb widths: measured
    # thumb-px/ms must scale with width, and land inside each
    # channel's own calibrated [vel_min, vel_max]
    def run(tw, th, speed_pxs):
        mc = M.MotionChannel(tw, th)
        for k in range(14):
            f = np.zeros((th, tw), dtype=np.float32)
            yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
            cx = tw * 0.2 + speed_pxs * k * 33.3
            f[np.hypot(xx - cx, yy - th / 2) < th * 0.08] = 0.8
            mc.push(f, 33.3, t_ms=k * 33.3)
        return mc.velocity_at(tw * 0.35, th / 2)
    v56 = run(56, 56, 0.02)
    v112 = run(112, 63, 0.04)  # same world speed, 2x the thumb px
    assert v56 is not None and v112 is not None
    assert abs(v56[0] - 0.02) < 0.008
    assert abs(v112[0] - 0.04) < 0.016
    assert abs(v112[0] / v56[0] - 2.0) < 0.4, "velocity must scale with width"


def test_downsampled_gate_matches_direct_gate():
    mc = M.MotionChannel(160, 90)
    rng = np.random.default_rng(3)
    raw = rng.random((90, 160), dtype=np.float32)
    g_direct = mc._gate(raw, direct=True)
    g_down = mc._gate(raw, direct=False)
    assert g_direct.shape == g_down.shape == (90, 160)
    assert float(np.abs(g_direct - g_down).max()) < 0.03, \
        "downsampled gate diverged from direct gate"


def test_driver_thumb_from_work_frame():
    from streaming.online_driver import _downsample_thumb
    rng = np.random.default_rng(5)
    work = rng.random((360, 640, 3), dtype=np.float32)
    thumb = _downsample_thumb(work, 160, 90)
    assert thumb.shape == (90, 160)
    assert thumb.dtype == np.float32
    assert 0.0 <= float(thumb.min()) and float(thumb.max()) <= 1.0


def test_driver_motion_thumb_source_work_and_fallback():
    fps = 30.0
    rng = np.random.default_rng(9)
    d = OnlineAttentionDriver(fps, 20_000.0, motion_weight=1.0,
                              motion_thumb_wh=(96, 54))
    for i in range(10):
        t = i * 1000.0 / fps
        f224 = rng.random((224, 224, 3), dtype=np.float32)
        fwork = rng.random((360, 640, 3), dtype=np.float32)
        d.push(t, f224, fwork)   # work-frame source
        d.push(t, f224)          # legacy fallback, must not crash
    assert d.motion.energy.shape == (54, 96)
    assert len(d.motion_energies) == 20


def test_driver_legacy_attn_source_runs():
    fps = 30.0
    d = OnlineAttentionDriver(fps, 20_000.0, motion_weight=1.0,
                              motion_thumb_wh=(56, 56),
                              motion_thumb_source="attn")
    for i in range(90):
        t = i * 1000.0 / fps
        d.push(t, _disk_frame(40.0 + i * 1.5, 112.0))
    me = np.array([e for _, e in d.motion_energies])
    assert len(me) == 90
    assert d.motion.energy.shape == (56, 56)


# --- input validation & NaN hardening (2026-10-03 audit) ---------------------

def test_motion_channel_nan_input_does_not_poison():
    mc = M.MotionChannel(56, 56)
    mc.push(np.zeros((56, 56), dtype=np.float32), 33.3)
    bad = np.zeros((56, 56), dtype=np.float32)
    bad[10, 10] = np.nan
    bad[20, 20] = np.inf
    e = mc.push(bad, 33.3)
    assert not np.isnan(e).any(), "NaN input poisoned the channel"
    assert not np.isinf(e).any(), "inf input poisoned the channel"
    # channel recovers on the next clean frame
    e2 = mc.push(np.zeros((56, 56), dtype=np.float32), 33.3)
    assert not np.isnan(e2).any()
    assert np.isfinite(mc.flood_frac)


def test_motion_channel_rejects_bad_tau():
    with pytest.raises(ValueError):
        M.MotionChannel(56, 56, tau_ms=0.0)
    with pytest.raises(ValueError):
        M.MotionChannel(56, 56, tau_ms=-5.0)


def test_motion_channel_rejects_bad_thumb():
    with pytest.raises(ValueError):
        M.MotionChannel(0, 56)
    with pytest.raises(ValueError):
        M.MotionChannel(56, 0)


def test_pursuit_boundary_exactness():
    ctl = SaccadeController([(0, 0.0, 0.0)], 0.1)
    ctl.add_pursuit(100.0, 1100.0, 0.05, -0.02)
    # before the glide: initial fixation
    fx, fy, supp = ctl.state_at(50.0)
    assert (fx, fy, supp) == (0.0, 0.0, False)
    # exactly at onset: glide position, never suppressed
    fx, fy, supp = ctl.state_at(100.0)
    assert abs(fx) < 1e-9 and abs(fy) < 1e-9 and not supp
    # exactly at end: rest at the end position
    fx, fy, supp = ctl.state_at(1100.0)
    assert abs(fx - 50.0) < 1e-9 and abs(fy + 20.0) < 1e-9 and not supp


def test_pursuit_supersede_is_continuous():
    ctl = SaccadeController([(0, 0.0, 0.0)], 0.1)
    ctl.add_pursuit(100.0, 1100.0, 0.05, 0.0)
    ctl.add_pursuit(500.0, 900.0, -0.1, 0.0)
    # old glide is gone; the new one starts where the eye was at t=500
    assert len(ctl.pursuits) == 1
    fx, fy, _ = ctl.state_at(500.0)
    assert abs(fx - 20.0) < 1e-9 and abs(fy) < 1e-9
    # mid new-glide: linear from the handoff point
    fx, fy, _ = ctl.state_at(700.0)
    assert abs(fx - (20.0 - 0.1 * 200.0)) < 1e-6
    # end of the new glide rests at its end position
    fx, fy, _ = ctl.state_at(900.0)
    assert abs(fx - (20.0 - 0.1 * 400.0)) < 1e-6


def test_saccade_from_mid_pursuit_uses_glide_position():
    ctl = SaccadeController([(0, 0.0, 0.0)], 0.1)
    ctl.add_pursuit(0.0, 1000.0, 0.1, 0.0)
    ctl.add_saccade(500.0, 200.0, 0.0)
    t_on, dur, frm, to = ctl.events[-1]
    assert abs(frm[0] - 50.0) < 1e-9 and abs(frm[1]) < 1e-9
    # the truncated glide ends exactly at the saccade onset
    assert ctl.pursuits == [(0.0, 500.0, (0.0, 0.0), (0.1, 0.0))]


def test_salience_constants_hold_values():
    # The 2026-10-03 magic-number naming refactor must not drift the
    # model values; behavior is pinned by the trial/color test suites.
    assert A.TRANSIENT_WEIGHT == 1.5
    assert A.INHIB_WINDOW_MS == 1500.0
    assert A.INHIB_TAU_MS == 800.0
    assert A.INHIB_SIGMA == 5.0
    assert A.FIX_DISK_SIGMA == 3.5
    assert A.CS_SIGMA == 6.0
