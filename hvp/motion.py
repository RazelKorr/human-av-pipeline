"""Magno channel: fast, coarse, colorblind motion energy.

Biology is the spec: primate vision runs a dedicated magnocellular
pathway -- fast, low-resolution, colorblind, motion-sensitive --
alongside the slow detailed parvocellular path. This module is that
pathway: frame differencing on a coarse grayscale thumbnail, with a
local-coherence gate so global flicker (flashes, cuts, whiteouts)
does not read as motion.

The gate is the thing to get right (2026-10-03): raw frame difference
floods on any full-field change. Subtracting a heavily blurred copy
-- center-surround on the motion map itself -- leaves only *local*
motion contrast (a moving craft against background or background
flow) while uniform floods cancel to ~zero. No separate flood flag:
the gate IS the coherence filter, plus a diagnostic flood fraction.

Thumbnail geometry (2026-10-03): the thumb is 16:9, derived from the
work-resolution frame -- never the anamorphic 224x224 attention
frame. A square thumb on 16:9 footage squishes the motion field
horizontally (~11.4x vs ~6.4x downsampling), so pursuit velocities
and motion centroids were systematically wrong in x. All map-px
constants scale from the 56-reference: velocities are calibrated in
deg/s (3.6..107) and converted per thumb width.
"""

import numpy as np
from PIL import Image as _Im
from scipy.ndimage import gaussian_filter

from . import baseline as B

MAP_REF = 56          # reference size all map-px constants were tuned at
THRESH = 0.05         # luminance-difference floor (kills encode noise)
COHERENCE_SIGMA = 12  # blur radius for the local-coherence gate (ref px)
TAU_MS = 120.0        # magno persistence: fast and brief (vs 200 ms transient)
VEL_WINDOW_MS = 400.0  # centroid history window for velocity estimates
VEL_MIN_SAMPLES = 3    # minimum centroid samples for a velocity estimate
VEL_MIN_SPAN_MS = 150.0  # minimum track duration for a velocity estimate
VEL_MIN_DEGS = 3.6     # min trackable speed, deg/s (was 0.005 map-px/ms @56)
VEL_MAX_DEGS = 107.0   # max trackable speed, deg/s (was 0.15 map-px/ms @56)
VEL_COS_MIN = 0.5      # direction-consistency floor (mean segment cosine)
FOCUS_RADIUS_FRAC = 0.25  # map-width fraction around fixation for pursuit
PURSUIT_MIN_ENERGY = 0.03  # min local motion energy at fixation to pursue
# The coherence gate is O(N*sigma) with scipy's gaussian; above this
# thumb width the blur is computed on a <=64px working copy and
# upsampled -- same effective blur (same width fraction), constant
# cost. At or below it the blur is direct (bit-identical to the
# validated 56-reference behavior).
GATE_DIRECT_MAX_W = 128
GATE_WORK_W = 64


class MotionChannel:
    """Coarse motion-energy sensor on a 16:9 grayscale thumbnail.

    push(small_luma, dt_ms, t_ms) -> (thumb_h, thumb_w) float32
    motion-energy map. small_luma must be (thumb_h, thumb_w) float32
    in 0..1 -- the driver derives it from the work-resolution frame,
    so the marginal cost is one PIL downsample, one abs-diff, one
    (possibly downsampled) gaussian, and a threshold.
    """

    def __init__(self, thumb_w=96, thumb_h=54, tau_ms=TAU_MS):
        tw, th = int(thumb_w), int(thumb_h)
        if tw < 1 or th < 1:
            raise ValueError(
                f"thumb size must be >= 1 px, got {(thumb_w, thumb_h)!r}")
        if not float(tau_ms) > 0:
            # tau_ms=0 used to die later in push() with a bare
            # ZeroDivisionError; NaN tau would silently never decay.
            raise ValueError(f"tau_ms must be > 0, got {tau_ms!r}")
        self.thumb_w = tw
        self.thumb_h = th
        tw, th = self.thumb_w, self.thumb_h
        self.energy = np.zeros((th, tw), dtype=np.float32)
        self.prev = None
        self.tau_ms = float(tau_ms)
        self.t_ms = 0.0
        self.hist = []          # (t_ms, energy_map) for velocity estimates
        self.flood_frac = 0.0   # diagnostic: last frame's raw flood fraction
        # Size-relative calibration (56-reference):
        self.kx = tw / MAP_REF
        # deg/s -> thumb-px/ms: (degs/1000) * (tw / FIELD_WIDTH_DEG)
        self.vel_min = VEL_MIN_DEGS * tw / (B.FIELD_WIDTH_DEG * 1000.0)
        self.vel_max = VEL_MAX_DEGS * tw / (B.FIELD_WIDTH_DEG * 1000.0)
        self.focus_radius = FOCUS_RADIUS_FRAC * tw
        yy, xx = np.mgrid[0:th, 0:tw].astype(np.float32)
        self._xx = xx
        self._yy = yy

    def _gate(self, raw, direct=None):
        """Local-coherence gate: raw minus its heavily blurred copy.

        direct=None selects by thumb width (direct <= 128px, else the
        constant-cost downsampled path). direct=True/False forces a
        path -- used by tests to check they agree.
        """
        tw, th = self.thumb_w, self.thumb_h
        use_direct = (tw <= GATE_DIRECT_MAX_W) if direct is None else direct
        if use_direct:
            return gaussian_filter(raw, COHERENCE_SIGMA * self.kx)
        gw = GATE_WORK_W
        gh = max(1, round(GATE_WORK_W * th / tw))
        small_raw = np.asarray(
            _Im.fromarray(raw.astype(np.float32), mode="F")
               .resize((gw, gh), _Im.BILINEAR), dtype=np.float32)
        blurred = gaussian_filter(small_raw, COHERENCE_SIGMA * (gw / MAP_REF))
        return np.asarray(
            _Im.fromarray(blurred.astype(np.float32), mode="F")
               .resize((tw, th), _Im.BILINEAR), dtype=np.float32)

    def push(self, small, dt_ms, t_ms=None):
        small = np.asarray(small, dtype=np.float32)
        if small.shape != (self.thumb_h, self.thumb_w):
            raise ValueError(
                f"MotionChannel thumb is {(self.thumb_h, self.thumb_w)}, "
                f"got {small.shape}")
        # A corrupt input frame (NaN/inf from a dropped decode) must not
        # poison the channel: NaN survives the coherence gate and
        # np.maximum then propagates it into energy forever, silently
        # killing pursuit (velocity_at -> None) and salting the salience
        # map. Clamp to the documented 0..1 input range; clean inputs
        # pass through bit-identical.
        small = np.nan_to_num(small, nan=0.0, posinf=1.0, neginf=0.0)
        if t_ms is not None:
            self.t_ms = float(t_ms)
        else:
            self.t_ms += float(dt_ms)
        decay = float(np.exp(-float(dt_ms) / self.tau_ms))
        if self.prev is not None:
            raw = np.abs(small - self.prev).astype(np.float32)
            self.flood_frac = float((raw > THRESH).mean())
            # Local-coherence gate: uniform floods cancel, local
            # motion contrast survives.
            local = raw - self._gate(raw)
            local = np.clip(local, 0.0, None)
            local[local < THRESH] = 0.0
            self.energy = np.maximum(self.energy * decay,
                                     local).astype(np.float32)
            self.hist.append((self.t_ms, self.energy.copy()))
            cutoff = self.t_ms - VEL_WINDOW_MS
            while self.hist and self.hist[0][0] < cutoff:
                self.hist.pop(0)
        self.prev = small.copy()
        return self.energy

    def _centroid_in_disk(self, emap, cx, cy, radius):
        disk = ((self._xx - cx) ** 2 + (self._yy - cy) ** 2) <= radius ** 2
        w = emap * disk
        tot = float(w.sum())
        if tot < 1e-9:
            return None
        return (float((w * self._xx).sum() / tot),
                float((w * self._yy).sum() / tot))

    def velocity_at(self, fx, fy, radius=None):
        """Velocity of the motion centroid near (fx, fy), in thumb-px/ms.

        fx/fy are thumb-pixel coordinates. Returns (vx, vy) or None
        when the track is too short, too slow, too fast, or
        direction-inconsistent. This is the pursuit drive: the magno
        channel's answer to "which way is the thing under the fovea
        going, and is it sure?"
        """
        if radius is None:
            radius = self.focus_radius
        track = []
        for (t, em) in self.hist:
            c = self._centroid_in_disk(em, fx, fy, radius)
            if c is not None:
                track.append((t, c[0], c[1]))
        if len(track) < VEL_MIN_SAMPLES:
            return None
        if track[-1][0] - track[0][0] < VEL_MIN_SPAN_MS:
            return None
        segs = []
        for (ta, xa, ya), (tb, xb, yb) in zip(track[:-1], track[1:]):
            dt = tb - ta
            if dt > 0:
                segs.append(((xb - xa) / dt, (yb - ya) / dt))
        if not segs:
            return None
        vx = sum(s[0] for s in segs) / len(segs)
        vy = sum(s[1] for s in segs) / len(segs)
        speed = (vx ** 2 + vy ** 2) ** 0.5
        if not (self.vel_min <= speed <= self.vel_max):
            return None
        cosines = []
        for (sx_, sy_) in segs:
            sp = (sx_ ** 2 + sy_ ** 2) ** 0.5
            if sp > 1e-9:
                cosines.append((sx_ * vx + sy_ * vy) / (sp * speed))
        if cosines and sum(cosines) / len(cosines) < VEL_COS_MIN:
            return None
        return (vx, vy)
