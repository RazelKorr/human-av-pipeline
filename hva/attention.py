"""Auditory attention: the cocktail-party focus.

The honest difference from vision: you can't move your ears. The
"fovea" here is a frequency band -- about an octave wide -- that gets
high-resolution processing, and the "saccade" is a switch of that band
from one stream to another. Switches cost: ~150ms to land, and
sensitivity dips for ~100ms after (the analog of saccadic suppression;
in the literature this is the attentional blink / switch cost).

Mechanics per 100ms moment:
  - salience vector over 64 bins, minus inhibition-of-return bumps at
    recently attended bands, minus a distance cost (far switches are
    expensive, like long saccades).
  - silence gate (RazelKorr's darkness rule, ported 2026-09-29): in
    near-silence the focus dwells -- it does NOT hop around on noise.
  - onset interrupt: a sharp transient can yank attention mid-stream
    with ~50ms latency (audition orients faster than vision; this is
    the one place the ears beat the eyes).
"""

import numpy as np
from . import cochlea as C

BAND_HALF = 5          # attended half-bandwidth in bins (~1 octave)
SWITCH_MIN_MS = 200.0  # the decision clock: at most one scheduled
                       # reconsideration per 200ms (the ears' analog
                       # of the ~3Hz saccade clock)
SWITCH_LATENCY_MS = 150.0
SWITCH_SUPPRESS_MS = 100.0
SWITCH_THRESH_BINS = 1.5
DIST_W = 0.004         # per-bin distance cost
IOR_SIGMA = 6.0
IOR_DECAY_S = 6.0
ONSET_PCT = 95.0       # interrupt fires above this percentile of
                       # recent Tmax (the most transient 5% of moments)
ABS_FLOOR_DB = 6.0     # ignore sub-6dB wiggles: the perceptual floor
SUPPRESS_ONSET_X = 3.0 # right after a switch, onsets need 3x the evidence
                       # (the attentional blink -- the ears' analog of
                       # saccadic suppression)
ONSET_LATENCY_MS = 50.0
SILENCE_LOUD = 0.15    # below this loudness: dwell, don't switch
ATT_GAIN = 1.5         # attentional gain on the attended band: the
                       # fovea's resolution advantage. Attended channels
                       # get enhanced processing (Fritz et al. 2003),
                       # which is also what makes an attended change
                       # noticeable and an unattended one missable.


class AuditoryAttention:
    def __init__(self, cf_init=32.0):
        self.cf = float(cf_init)
        self.pending = None        # (target_cf, land_t)
        self.suppress_until = -1.0
        self.ior = []              # (cf, t)
        self.tmax_hist = []
        self.hab = {}              # coarse band -> [t] of recent captures
        self.last_dec = -1.0       # last scheduled decision time

    def _ior_bumps(self, t):
        b = np.zeros(C.N_BINS, np.float32)
        bins = np.arange(C.N_BINS)
        for cf0, t0 in self.ior:
            age = t - t0
            if age < IOR_DECAY_S:
                b += np.exp(-age / (IOR_DECAY_S / 3.0)) * np.exp(
                    -((bins - cf0) ** 2) / (2 * IOR_SIGMA ** 2))
        return np.clip(b, 0, 1) * 0.5

    def run(self, moms, Tprof, vis_boost=None):
        """moms: list of moment dicts. Tprof: per-moment temporal-
        contrast profile, (n_moments, 64). vis_boost: optional per-moment
        array in [0,1] of visual transient strength (Level 2 cross-modal
        coupling) -- a flash makes you listen harder. Returns trace: one
        dict per moment with the attended band and what happened."""
        trace = []
        vb_shaped = None
        if vis_boost is not None:
            # Reshape: only the top quartile of visual transients gets a
            # vote. The film is always wiggling a little; only the
            # genuinely busy moments lower the auditory bar.
            vb_arr = np.asarray(vis_boost, dtype=np.float64)
            q75 = float(np.percentile(vb_arr, 75.0))
            vb_shaped = np.clip((vb_arr - q75) / max(1.0 - q75, 1e-9),
                                0.0, 1.0)
        for k, m in enumerate(moms):
            t0, t1 = m["t0"], m["t1"]
            # land pending switches
            if self.pending and t0 >= self.pending[1]:
                self.cf, _ = self.pending
                self.pending = None
                self.suppress_until = t0 + SWITCH_SUPPRESS_MS / 1000.0
                self.ior.append((self.cf, t0))
                self.ior = [(c, tt) for c, tt in self.ior
                            if t0 - tt < IOR_DECAY_S]
            sal = m["sal"].astype(np.float32).copy()
            # attentional gain: the attended band gets a processing
            # advantage -- this is what holds a stream at a cocktail
            # party, and what makes attended changes noticeable.
            bins = np.arange(C.N_BINS)
            sal *= 1.0 + (ATT_GAIN - 1.0) * np.exp(
                -((bins - self.cf) ** 2) / (2 * BAND_HALF ** 2))
            sal -= self._ior_bumps(t0)
            sal -= DIST_W * np.abs(bins - self.cf)
            sal = np.clip(sal, 0, None)

            event = None
            # onset interrupt: sharp transient yanks focus (express).
            # Threshold is the 95th percentile of recent Tmax -- it fires
            # on the most transient 5% of moments, adapting to busy vs.
            # quiet scenes. (A 4x-median rule was tried; in busy real
            # audio the median itself is high and nothing ever fired.)
            # In true quiet the percentile is ~0, so the threshold floors
            # at ABS_FLOOR_DB -- otherwise a click in silence could never
            # capture, which is exactly backwards. Right after a switch
            # the bar is higher (attentional blink).
            self.tmax_hist.append(m["Tmax"])
            recent = self.tmax_hist[-200:]
            thresh = max(float(np.percentile(recent, ONSET_PCT)),
                         ABS_FLOOR_DB)
            if vb_shaped is not None:
                # Level 2, vision->audio: a strong visual transient lowers
                # the auditory capture bar (up to 36% off). The flash
                # makes the bang easier to notice.
                thresh *= (1.0 - 0.36 * float(vb_shaped[k]))
            if t0 < self.suppress_until:
                thresh *= SUPPRESS_ONSET_X
            if m["Tmax"] > thresh:
                # (Guard: argmax of an all-zero profile is 0, which is a
                # real bin -- never capture to a phantom target.)
                if Tprof[k].max() > 0:
                    tgt = float(np.argmax(Tprof[k]))
                    if abs(tgt - self.cf) > SWITCH_THRESH_BINS:
                        # stimulus-specific adaptation: the third
                        # identical onset within a second is expected,
                        # not news. The auditory nerve adapts to
                        # repetition; so does this.
                        key = int(round(tgt / 2.0))
                        recent = [tt for tt in self.hab.get(key, [])
                                  if t0 - tt < 1.0]
                        if len(recent) < 2:
                            self.pending = (tgt,
                                            t0 + ONSET_LATENCY_MS / 1000.0)
                            event = "onset-capture"
                            self.hab[key] = recent + [t0]

            # scheduled switch at moment boundary (unless silent:
            # RazelKorr's rule -- in quiet, dwell, don't hop on noise).
            # Rate-limited like the saccade clock: at most one
            # reconsideration per SWITCH_MIN_MS. And refractory right
            # after a switch: the just-landed band gets 100ms to
            # establish itself before anything can dislodge it. (Scaling
            # the whole salience map was tried -- it changes nothing,
            # because argmax is scale-invariant. The refractory is the
            # honest mechanism.)
            if event is None and not self.pending:
                if t0 - self.last_dec >= SWITCH_MIN_MS / 1000.0:
                    self.last_dec = t0
                    if t0 < self.suppress_until:
                        event = "refractory"
                    elif m["loud"] >= SILENCE_LOUD:
                        # argmax of an all-zero field is meaningless --
                        # nothing worth switching to.
                        if sal.max() > 0:
                            cand = float(np.argmax(sal))
                            if abs(cand - self.cf) > SWITCH_THRESH_BINS:
                                self.pending = (
                                    cand,
                                    t0 + SWITCH_LATENCY_MS / 1000.0)
                                event = "switch"
                    else:
                        event = "dwell-quiet"

            lo = max(0, int(self.cf) - BAND_HALF)
            hi = min(C.N_BINS, int(self.cf) + BAND_HALF + 1)
            trace.append({
                "t0": t0, "t1": t1,
                "cf_bin": self.cf, "cf_hz": C.bin_to_hz(
                    np.clip(int(round(self.cf)), 0, C.N_BINS - 1)),
                "band": (lo, hi), "loud": m["loud"], "event": event,
            })
        return trace
