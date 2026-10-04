"""OnlineAttentionDriver: the batch v2 pass-1 attention math, stateful.

This is scripts/run_video_saccades_color.py's pass-1 loop refactored
into a push-based object, with the math reproduced EXACTLY (same
constants, same operation order) so stream-vs-batch comparison is
meaningful. Everything here is causal: transient decay, inhibition of
return (1.5 s), POI memory (12 s), and the saccade decision clock are
all functions of past frames only.

The one file-level fact the driver needs is t_end_ms (the media
duration, from the container header -- not lookahead): saccades
scheduled past the end are dropped, matching the batch convention.
"""

from __future__ import annotations

import numpy as np
from PIL import Image as _Im
from scipy.ndimage import gaussian_filter

from hvp import attention as A
from hvp import baseline as B
from hvp import motion as M
from hvp.saccades import SaccadeController

VSIZE = 224  # pass-1 decode size; matches hvp.attention.SIZE

# BT.601 luma weights, shared by the thumbnail derivations below.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Default motion thumbnail: 16:9, derived from the work-resolution
# frame (never the anamorphic 224x224 attention frame). Set by the
# 2026-10-03 thumb-size benchmark; override with motion_thumb_wh.
DEFAULT_MOTION_THUMB_WH = (96, 54)


def _downsample_thumb(frame, tw, th):
    """Any-size HxWx3 float32 frame -> (th, tw) grayscale thumbnail."""
    img = _Im.fromarray(
        (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8))
    rgb = np.asarray(img.resize((tw, th), _Im.BILINEAR),
                     dtype=np.float32) / 255.0
    return rgb @ _LUMA


class OnlineAttentionDriver:
    """Push 224x224 RGB frames; pull saccade decisions.

    push(t_ms, frame224) -> list of (t_on, tx, ty) saccades decided
    on this frame, in 224px coords. Usually empty; at most one per
    frame (the decision clock runs at ~3.5 Hz).
    """

    def __init__(self, fps, t_end_ms, chroma_weight=None, motion_weight=None,
                 pursuit=False, motion_thumb_wh=None,
                 motion_thumb_source="work"):
        self.dt = 1000.0 / float(fps)
        self.t_end = float(t_end_ms)
        self.chroma_weight = chroma_weight
        # Magno channel: None/0 = off (v2-identical); a positive weight
        # enables motion-energy salience at that weight.
        self.motion_weight = motion_weight
        # 16:9 thumbnail from the work frame (true aspect). "attn" is
        # the legacy anamorphic fallback (56x56 attention thumb
        # stretched); kept for validation comparisons only.
        if motion_thumb_wh is not None:
            tw_, th_ = int(motion_thumb_wh[0]), int(motion_thumb_wh[1])
            if tw_ <= 0 or th_ <= 0:
                raise ValueError("motion_thumb_wh must be positive WxH, "
                                 f"got {motion_thumb_wh!r}")
            self.motion_thumb_wh = (tw_, th_)
        else:
            self.motion_thumb_wh = DEFAULT_MOTION_THUMB_WH
        if motion_thumb_source not in ("work", "attn"):
            raise ValueError("motion_thumb_source must be 'work' or 'attn'")
        self.motion_thumb_source = motion_thumb_source
        tw, th = self.motion_thumb_wh
        self.motion = (M.MotionChannel(tw, th)
                       if motion_weight else None)
        # Smooth pursuit: only meaningful with the motion channel on;
        # default off (v2-identical).
        self.pursuit = bool(pursuit) and self.motion is not None
        self.dva1 = B.FIELD_WIDTH_DEG / VSIZE
        cx = cy = VSIZE / 2.0
        self.controller = SaccadeController([(0, cx, cy)], self.dva1)
        self.inhib = []            # (ix, iy, t) inhibition of return
        self.scanpath = [(0.0, cx, cy)]
        self.pois = []             # (px, py, t, strength), 12 s memory
        self.trans = np.zeros((A.SMALL, A.SMALL), dtype=np.float32)
        self.prev_small = None
        self.decay = np.exp(-self.dt / A.TRANS_TAU_MS)
        self.next_decision = 100.0
        self.energies = []         # (t_ms, lum_e, chroma_e, trans_e)
        self.motion_energies = []  # (t_ms, motion_e) -- only when enabled
        self.pursuit_log = []      # (t0, t1, vx, vy) in 224px coords
        self.n_frames = 0

    def push(self, t_ms, frame224, frame_work=None, frame_thumb=None):
        """frame224: 224x224x3 float32 RGB in 0..1; frame_work: the
        work-resolution frame (16:9), used for the motion thumbnail;
        frame_thumb: feeder-supplied motion thumbnail (HxWx3 RGB
        float32) -- when present, the magno channel consumes it
        directly (luma only) instead of PIL-downsampling frame_work.

        Returns a list of decided eye movements this frame, each a
        tagged tuple: ("sac", t_on, tx, ty) or ("pur", t0, t1, vx, vy),
        in 224px coords. Usually empty; at most one per frame (the
        decision clock runs at ~3.5 Hz).
        """
        t = float(t_ms)
        small_rgb = A._downsample_color(frame224)
        # luma thumbnail for the luminance static + transient channels
        lum = (small_rgb @ _LUMA)
        small = lum
        if self.prev_small is not None:
            self.trans = np.maximum(self.trans * self.decay,
                                    np.abs(small - self.prev_small))
        self.prev_small = small
        # magno channel: motion energy on a 16:9 thumbnail derived
        # from the work frame (true aspect -- the 224px attention
        # frame is anamorphic). The salience map stays 56x56
        # anamorphic, so the energy map is resampled to it below.
        motion_map = None
        if self.motion is not None:
            tw, th = self.motion_thumb_wh
            if (frame_thumb is not None
                    and self.motion_thumb_source == "work"):
                # Feeder-supplied 16:9 RGB thumbnail (third split
                # output): take luma directly -- no in-process
                # downsample. libswscale produced it during decode.
                assert frame_thumb.shape == (th, tw, 3), \
                    (f"feeder thumb shape {frame_thumb.shape} != "
                     f"motion thumb {(th, tw, 3)}")
                thumb = (frame_thumb @ _LUMA).astype(np.float32)
            elif (frame_work is not None
                    and self.motion_thumb_source == "work"):
                thumb = _downsample_thumb(frame_work, tw, th)
            else:
                # legacy/validation fallback: stretch the 56x56
                # attention thumb (anamorphic geometry)
                thumb = np.asarray(
                    _Im.fromarray(small.astype(np.float32), mode="F")
                       .resize((tw, th), _Im.BILINEAR), dtype=np.float32)
            motion_map = self.motion.push(thumb, self.dt, t_ms=t)
            self.motion_energies.append((t, float(motion_map.mean())))
        static_e = float(np.abs(small - gaussian_filter(small, 6)).mean())
        chroma_map = A.chroma_salience(small_rgb)
        self.energies.append((t, static_e, float(chroma_map.mean()),
                              float(self.trans.mean())))
        self.n_frames += 1

        decided = []
        if t >= self.next_decision:
            motion_for_sal = None
            if motion_map is not None:
                # the salience grid is 56x56 anamorphic; resample the
                # 16:9 energy map into it (identity when already 56x56)
                if motion_map.shape == (A.SMALL, A.SMALL):
                    motion_for_sal = motion_map
                else:
                    motion_for_sal = np.asarray(
                        _Im.fromarray(motion_map.astype(np.float32),
                                      mode="F").resize(
                                          (A.SMALL, A.SMALL),
                                          _Im.BILINEAR), dtype=np.float32)
            sal_base = A.salience_map(small, self.trans, t, self.controller,
                                      self.inhib, small_rgb=small_rgb,
                                      chroma_weight=self.chroma_weight,
                                      motion_map=motion_for_sal,
                                      motion_weight=self.motion_weight)
            pursued = self._maybe_pursue(t) if self.pursuit else False
            if not pursued:
                fx0, fy0, _ = self.controller.state_at(t)
                yy, xx = np.mgrid[0:A.SMALL, 0:A.SMALL].astype(np.float32)
                dist_deg = (np.hypot(xx - fx0 / A.SCALE, yy - fy0 / A.SCALE)
                            * A.SCALE * self.dva1)
                sal = sal_base + 1.5 * np.exp(-((dist_deg - 8.0) / 7.0) ** 2)
                for (px, py, pt, ps) in self.pois:
                    age = t - pt
                    if age < 12000.0:
                        sal = (sal + ps * np.exp(-age / 6000.0)
                               * A._blob((A.SMALL, A.SMALL), px, py, 12.0))
                iy, ix = np.unravel_index(int(np.argmax(sal)), sal.shape)
                tx, ty = (ix + 0.5) * A.SCALE, (iy + 0.5) * A.SCALE
                t_on = t + B.SACCADE_LATENCY_MS
                if t_on < self.t_end:
                    self.controller.add_saccade(t_on, tx, ty)
                    self.scanpath.append((t_on, tx, ty))
                    # A saccade interrupts any glide it overlaps: keep
                    # the pursuit log truthful about effective segments.
                    if self.pursuit_log:
                        self.pursuit_log = [
                            (a, min(b, t_on), vx, vy)
                            for (a, b, vx, vy) in self.pursuit_log]
                    strength = float(np.clip(sal_base[iy, ix] / 2.0, 0.2, 1.5))
                    self.pois.append((float(ix), float(iy), t, strength))
                    self.pois = [(x, y, it, s)
                                 for (x, y, it, s) in self.pois
                                 if t - it < 12000.0][-10:]
                    decided.append(("sac", t_on, tx, ty))
            self.inhib = [(x, y, it) for (x, y, it) in self.inhib
                          if t - it < 1500.0]
            if not pursued:
                # No inhibition of return while pursuing: the whole
                # point is to stay with the target.
                self.inhib.append((ix, iy, t))
            self.next_decision = t + 1000.0 / B.SACCADE_RATE_HZ
        return decided

    def _maybe_pursue(self, t):
        """Try to start a smooth-pursuit glide at decision time t.

        Pursuit engages when the eyes are already on coherent local
        motion with a consistent velocity: the magno channel's
        centroid near fixation has held direction and a trackable
        speed. Returns True if a pursuit segment was scheduled (the
        caller then skips the saccade decision and inhibition).
        """
        mc = self.motion
        fx, fy, supp = self.controller.state_at(t)
        if supp:
            return False
        # fixation (224px anamorphic) -> motion-thumb coords; thumb is
        # true 16:9, so x and y scale independently
        tw, th = self.motion_thumb_wh
        fx_t, fy_t = fx * tw / VSIZE, fy * th / VSIZE
        ixt = int(np.clip(fx_t, 0, tw - 1))
        iyt = int(np.clip(fy_t, 0, th - 1))
        if mc.energy[iyt, ixt] < M.PURSUIT_MIN_ENERGY:
            return False
        v = mc.velocity_at(fx_t, fy_t)
        if v is None:
            return False
        # thumb-px/ms -> 224px/ms (back into the controller's space)
        vx = v[0] * (VSIZE / tw) * B.PURSUIT_GAIN
        vy = v[1] * (VSIZE / th) * B.PURSUIT_GAIN
        t0 = t + B.PURSUIT_LATENCY_MS
        if t0 >= self.t_end:
            return False
        # Never launch a glide into a saccade already in flight.
        _, _, supp0 = self.controller.state_at(t0)
        if supp0:
            return False
        t1 = min(t0 + B.PURSUIT_SEGMENT_MS, self.t_end)
        self.controller.add_pursuit(t0, t1, vx, vy)
        # A new glide supersedes an overlapping older one: keep the
        # log's segments non-overlapping like the controller's.
        if self.pursuit_log and self.pursuit_log[-1][1] > t0:
            a, b, vx0, vy0 = self.pursuit_log[-1]
            self.pursuit_log[-1] = (a, t0, vx0, vy0)
        self.pursuit_log.append((t0, t1, float(vx), float(vy)))
        return True

    @property
    def n_saccades(self):
        return len(self.scanpath) - 1

    @property
    def n_pursuits(self):
        return len(self.pursuit_log)
