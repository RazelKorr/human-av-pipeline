"""Level 3: the joint priority map.

The superior-colliculus analog: one shared spatial priority landscape
that both systems write to and read from.

Writers (per 100 ms moment):
  - vision writes retinotopic salience (static + transient), normalized.
  - audition writes each frequency bin's transient salience, projected
    into space by its estimated pan (interaural level difference).
    No elevation cue exists, so the splat sits at midline -- azimuth only,
    which is honest about what ILD can tell you. Mono sources (pan 0)
    land center: a central alerting blob, not a spatial peak.

Readers:
  - the saccade controller picks targets from the map (inhibition of
    return and the oculomotor amplitude prior apply at read time --
    they are not sensory, so they are not written to the map).
  - auditory attention reads per-bin spatial gains: bins whose pan
    projects near a map peak get their transients amplified
    ("listen to the frequencies at the attended location").

This replaces Level 2's cross-biases. Instead of system A's output
nudging system B's input, both systems meet at the map.

Provisional constants -- audit before citing:
  W_VIS=1.0, W_AUD=0.7 : vision dominates spatial orienting
      (the ventriloquism direction). The audio weight is a guess.
  SPEECH_BOOST=1.0 : when speech is present, audition's map vote
      doubles (0.7 -> 1.4), enough to outvote vision in an
      equal-strength conflict. The voice captures the map -- the
      cocktail-party direction. A round-number guess; the battery's
      M6 pins the behavioral requirement (speech flips a conflict
      audition otherwise loses), not the value.
  TAU_MS=300 : map persistence. Between the transient channel (200 ms)
      and inhibition of return (1500 ms). A guess.
  AUD_SPREAD_PX=6.0 : the auditory splat is broad on purpose -- hearing
      localizes worse than vision, and the map should know that.
  READ_GAIN_K=1.5 : how strongly a map peak boosts bins at its azimuth.

Speech gating is stream-level, not bin-level: `speech` is a scalar per
moment in [0,1] (smoothed presence from the transcript channel). Every
bin's write is scaled equally, so a loud non-speech transient during
speech gets boosted too -- honest v1 limitation. The real fix is stream
separation (identifying *which* bins carry the voice); until then the
boost is "something is being said, so weight the ears more," not
"the voice is at this azimuth." No validated human multisensory binding
is claimed; this is gain control, not binding.
"""

import numpy as np

from hvp.attention import _blob

SIZE = 56            # map resolution; matches visual salience
W_VIS = 1.0
W_AUD = 0.7
SPEECH_BOOST = 1.0
TAU_MS = 300.0
AUD_SPREAD_PX = 6.0
READ_GAIN_K = 1.5
PAN_FULL_DB = 12.0  # matches hva.spatial: |ILD| >= 12 dB is full lateral


def pan_to_x(pan, size=SIZE):
    """Azimuth -> map column. pan in [-1,1]; -1 = hard left."""
    return (0.5 + 0.4 * np.clip(pan, -1.0, 1.0)) * size


class JointPriorityMap:
    def __init__(self, size=SIZE, w_vis=W_VIS, w_aud=W_AUD, tau_ms=TAU_MS):
        self.size = size
        self.w_vis = w_vis
        self.w_aud = w_aud
        self.tau_ms = tau_ms
        self.map = np.zeros((size, size), dtype=np.float32)

    def step(self, dt_ms, vis_sal=None, aud=None, speech=0.0):
        """Integrate one tick.

        vis_sal: (56,56) array in ~[0,1], or None.
        aud: (bin_sal_64, bin_pan_64), both in ~[0,1] / [-1,1], or None.
        speech: scalar in [0,1] -- smoothed speech presence this moment.
            Scales the auditory write by (1 + SPEECH_BOOST*speech).
            The shared map's normalization does the attenuating: when
            the ears get louder, everything else gets relatively
            quieter. No separate visual-suppression knob -- the Dr Tran
            data shows vision keeps working during dense narration
            (title-card dwells), so global visual attenuation during
            speech would be wrong.
        Returns the map.
        """
        self.map *= np.exp(-dt_ms / self.tau_ms)
        if vis_sal is not None:
            self.map += self.w_vis * vis_sal
        if aud is not None:
            bin_sal, bin_pan = aud
            # Last-ditch input sanitation: a NaN pan used to poison the
            # whole map (NaN blob -> argmax (0,0)). NaN pan -> center,
            # NaN salience -> silence. The caller should still warn.
            bin_sal = np.nan_to_num(np.asarray(bin_sal, dtype=np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0)
            bin_pan = np.clip(np.nan_to_num(
                np.asarray(bin_pan, dtype=np.float32), nan=0.0), -1.0, 1.0)
            # Mean over bins, not sum: a broadband crash drives the map
            # harder than a narrow click (loudness-like), and the audio
            # total stays bounded ~[0, w_aud] so one loud moment cannot
            # permanently evict vision from its own map. The per-bin
            # spatial structure is preserved -- bins still splat at
            # their own pans -- only the total is normalized.
            w_aud_eff = self.w_aud * (1.0 + SPEECH_BOOST * speech)
            acc = 0.0
            for s, p in zip(bin_sal, bin_pan):
                if s > 1e-6:
                    x0 = pan_to_x(p, self.size)
                    acc = acc + s * _blob((self.size, self.size),
                                         x0, self.size / 2.0, AUD_SPREAD_PX)
            self.map += w_aud_eff * acc / max(len(bin_sal), 1)
        return self.map

    def peak(self):
        """(x, y, value) of the current maximum, in map pixels."""
        iy, ix = np.unravel_index(int(np.argmax(self.map)), self.map.shape)
        return float(ix), float(iy), float(self.map[iy, ix])

    def gains_for_pans(self, pans, k=READ_GAIN_K):
        """Per-bin spatial gain from the current map.

        A bin whose azimuth projects onto a salient map column gets its
        transients amplified. Sampling is column-max: auditory spatial
        attention is hemifield-ish, not pinpoint.
        """
        xs = pan_to_x(np.asarray(pans, dtype=np.float32), self.size)
        xs = np.nan_to_num(xs, nan=self.size / 2.0)  # NaN pan -> center
        ix = np.clip(np.round(xs - 0.5).astype(int), 0, self.size - 1)
        colmax = self.map.max(axis=0)
        return 1.0 + k * colmax[ix]
