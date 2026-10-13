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

Residual mode (2026-10-12, audit item 8): the coherence gate above
removes *uniform* floods, but egomotion (camera walking through a
static world) is not uniform in the diff map -- textured ground,
walls and furniture all produce strong local diffs everywhere, and
the pursuit logic happily hop-sustains on the flow field (FF3:
1,127 of 1,127 pursuits at exactly the minimum duration). Residual
mode estimates the global translation (phase correlation on the
thumbnail, subpixel via peak-neighborhood centroid) and computes
the frame difference AFTER motion-compensating the previous frame
by that shift (Fourier shift theorem, so fractional shifts align
exactly). Pure camera motion cancels to ~zero; what survives is
residual motion -- things moving differently from the world. The
compensated diff then goes through the same local-coherence gate
as before: the gate was not replaced, it is fed a better input.
Compensation is confidence-gated: when the correlation peak is
diffuse (PC_CONF_MIN) there is no decisive global flow -- a static
scene with one small mover, say -- and the plain diff is used, so
a meaningless shift estimate can never eat real content.
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
# Residual mode: minimum phase-correlation peak height for the
# global-flow estimate to be trusted. A genuine full-field
# translation of textured content peaks decisively (>>0.1); a
# static scene or a single small mover gives a diffuse surface.
PC_CONF_MIN = 0.10
# Below this fraction of structurally non-flat prev-frame pixels,
# global flow is unidentifiable (too little shared structure to
# separate camera motion from the one moving thing): never
# compensate.
FLOW_MIN_COVERAGE = 0.03


class MotionChannel:
    """Coarse motion-energy sensor on a 16:9 grayscale thumbnail.

    push(small_luma, dt_ms, t_ms) -> (thumb_h, thumb_w) float32
    motion-energy map. small_luma must be (thumb_h, thumb_w) float32
    in 0..1 -- the driver derives it from the work-resolution frame,
    so the marginal cost is one PIL downsample, one abs-diff, one
    (possibly downsampled) gaussian, and a threshold.
    """

    def __init__(self, thumb_w=96, thumb_h=54, tau_ms=TAU_MS,
                 residual_mode=False):
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
        # Residual (egomotion-compensated) differencing; see module
        # docstring. Legacy mode (False) is the pre-2026-10-12
        # behavior, bit-identical.
        self.residual_mode = bool(residual_mode)
        self.global_flow = (0.0, 0.0)  # last estimate (dx, dy), thumb-px/frame
        self.flow_conf = 0.0           # its correlation-peak height
        win = (np.hanning(th)[:, None] * np.hanning(tw)[None, :])
        self._pc_window = win.astype(np.float32)
        fy = np.fft.fftfreq(th)[:, None]
        fx = np.fft.fftfreq(tw)[None, :]
        self._pc_fy = fy
        self._pc_fx = fx
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

    def _estimate_global_shift(self, prev, cur):
        """Global translation of cur relative to prev, in thumb-px
        (dx, dy, peak): windowed phase correlation, subpixel by a
        centroid over the peak's 5x5 neighborhood (weights above
        half the peak height). Parabolic refinement was tried first
        and is biased on resampled content (it read 2.10/3.00 px
        for a true 2.66 px thumb shift); the centroid recovers the
        true shift and cuts the compensated residual ~7x vs the
        raw diff. Shifting prev by (dx, dy) (Fourier shift theorem)
        aligns it onto cur."""
        w = self._pc_window
        F = np.fft.fft2(cur * w)
        G = np.fft.fft2(prev * w)
        R = F * np.conj(G)
        R /= (np.abs(R) + 1e-12)
        r = np.fft.ifft2(R).real
        th, tw = prev.shape
        py, px = np.unravel_index(int(np.argmax(r)), r.shape)
        peak = float(r[py, px])
        dx = px - tw if px > tw // 2 else px
        dy = py - th if py > th // 2 else py
        numx = numy = den = 0.0
        for oy in range(-2, 3):
            for ox in range(-2, 3):
                v = float(r[(py + oy) % th, (px + ox) % tw]) - 0.5 * peak
                if v > 0.0:
                    numx += v * ox
                    numy += v * oy
                    den += v
        if den > 0.0:
            dx += numx / den
            dy += numy / den
        return float(dx), float(dy), peak

    def _shift_frame(self, frame, dx, dy):
        """Translate frame by (dx, dy) px (content moves right/down
        for positive values), Fourier shift theorem; wraps at the
        edges like np.roll."""
        F = np.fft.fft2(frame)
        phase = np.exp(-2j * np.pi * (self._pc_fx * dx
                                      + self._pc_fy * dy))
        return np.real(np.fft.ifft2(F * phase)).astype(np.float32)

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
            if self.residual_mode:
                # Egomotion compensation: estimate the global shift,
                # align prev onto small, and difference against the
                # aligned copy. flood_frac above stays the honest
                # uncompensated diagnostic. Compensation applies
                # only on a decisive correlation peak; otherwise the
                # plain diff stands (no confident global flow). The
                # Fourier-shift wrap strip (content wrapped from the
                # far edge) is zeroed -- it is not real motion, it
                # is the torus artifact.
                dx, dy, conf = self._estimate_global_shift(self.prev,
                                                           small)
                self.global_flow = (dx, dy)
                self.flow_conf = conf
                # Identifiability guard: global flow is only
                # identifiable when enough of the frame carries
                # shared structure. On a near-featureless frame the
                # one moving object IS the correlation peak, and
                # "compensating" for it would cancel the mover
                # itself; phase correlation still returns a sharp-
                # looking peak there, and its wrap-strip zeroing
                # would eat real content. Below the coverage floor
                # the plain diff stands.
                med = float(np.median(self.prev))
                coverage = float((np.abs(self.prev - med)
                                  > 0.05).mean())
                if coverage < FLOW_MIN_COVERAGE:
                    dx = dy = 0.0
                if conf >= PC_CONF_MIN and (dx or dy):
                    aligned = self._shift_frame(self.prev, dx, dy)
                    raw = np.abs(small - aligned).astype(np.float32)
                    by = int(np.ceil(abs(dy)))
                    bx = int(np.ceil(abs(dx)))
                    if by > 0:
                        if dy > 0:
                            raw[:by, :] = 0.0
                        else:
                            raw[-by:, :] = 0.0
                    if bx > 0:
                        if dx > 0:
                            raw[:, :bx] = 0.0
                        else:
                            raw[:, -bx:] = 0.0
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
