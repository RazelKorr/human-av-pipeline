"""Validation battery: does the pipeline trip over the same illusions a human does?

T0 latency            -- step input; you live ~200 ms in the past
T1 flicker fusion     -- square-wave patch; fusion by ~60 Hz
T2 wagon-wheel        -- 8-spoke wheel; 9 Hz spoke-pass aliases backward
T3 saccadic suppression -- flash mid-saccade is not seen
T4 change blindness   -- flicker paradigm; the mask hides the change
"""

import numpy as np

from . import baseline as B
from .pipeline import VisionPipeline
from . import stimuli as S
from . import attention as A


def _dva_per_px(width):
    return B.FIELD_WIDTH_DEG / width


def _run(script, gen, size):
    pipe = VisionPipeline((size, size), _dva_per_px(size), script)
    t_end = 0.0
    for t, frame in gen:
        pipe.push(frame, t)
        t_end = t
    return list(pipe.moments(t_end))


# ---------------------------------------------------------------- T0

def test_latency():
    """Step 0->1 at t=1000 ms. First moment with mean > 0.5 -> latency."""
    size, fps = 64, 500
    cx = size / 2
    moments = _run([(0, cx, cx)], S.gen_step(1.6, fps, size), size)
    for t, p, _ in moments:
        if p.mean() > 0.5:
            latency = t - 1000.0
            return {"latency_ms": latency,
                    "pass": 100.0 <= latency <= 250.0,
                    "detail": f"step->percept {latency:.0f} ms "
                              f"(100 ms pipeline + 100 ms integration window)"}
    return {"latency_ms": None, "pass": False, "detail": "no crossing found"}


# ---------------------------------------------------------------- T1

def test_flicker(freqs=(5, 15, 30, 60, 120)):
    """Temporal variance of the percept stream across flicker frequencies."""
    size, fps = 64, 1000
    cx = size / 2
    variances = {}
    for f in freqs:
        moments = _run([(0, cx, cx)], S.gen_flicker(2.0, f, fps, size), size)
        means = np.array([p.mean() for _, p, _ in moments[3:]])  # settled
        variances[f] = float(means.var())
    ratio = variances[60] / max(variances[5], 1e-12)
    return {"variances": variances,
            "fusion_ratio_60_over_5": ratio,
            "pass": bool(ratio < 0.10),
            "detail": f"var(60Hz)/var(5Hz) = {ratio:.4f} "
                      f"(fused below 0.10); note the null near 30 Hz from "
                      f"the 100 ms boxcar integrator"}


# ---------------------------------------------------------------- T2

def _angular_profile(frame, cx, cy, r0, r1, n_bins=720):
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - cx, yy - cy)
    theta = np.arctan2(yy - cy, xx - cx)
    mask = (r >= r0) & (r < r1)
    b = ((theta[mask] + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    prof = np.bincount(b, weights=frame[mask], minlength=n_bins)
    cnt = np.bincount(b, minlength=n_bins)
    return prof / np.maximum(cnt, 1)


def _perceived_rev_per_s(moment_frames, size):
    """Track the 8th angular harmonic's phase across perceptual moments.

    The wheel's 8-fold spoke pattern lives in the 8th angular harmonic;
    its phase (mod spoke-spacing) is the wheel angle. Cross-correlation
    fails here because the time-averaged static component dominates at
    zero shift -- phase tracking isolates the moving residual.
    """
    cx = cy = size / 2.0
    r0, r1 = 0.25 * size, 0.40 * size
    n_bins = 720
    phases = []
    for f in moment_frames:
        p = _angular_profile(f, cx, cy, r0, r1, n_bins)
        # rfft[8] ~ exp(-i*8*theta0) for peaks at theta0 (image y-axis down)
        phases.append(-np.angle(np.fft.rfft(p - p.mean())[8]) / 8.0)
    unwrapped = np.unwrap(np.array(phases), period=np.pi / 4)
    dt_s = B.PERCEPTUAL_MOMENT_MS / 1000.0
    slope_rad_per_moment = np.polyfit(np.arange(len(unwrapped)), unwrapped, 1)[0]
    return float(slope_rad_per_moment / (2 * np.pi) / dt_s)


def test_wagon_wheel():
    """9 Hz spoke-pass against the 10 Hz moment clock must read backward."""
    size, fps = 256, 500
    cx = size / 2
    results = {}
    for rev_per_s, tag in [(1.125, "spoke_pass_9Hz"), (0.25, "spoke_pass_2Hz")]:
        moments = _run([(0, cx, cx)],
                       S.gen_wheel(4.0, fps, size, n_spokes=8,
                                   rev_per_s=rev_per_s), size)
        frames = [p for _, p, _ in moments[2:]]  # settled
        perceived = _perceived_rev_per_s(frames, size)
        results[tag] = {"true_rev_per_s": rev_per_s, "perceived": perceived,
                        "frames": frames}
    aliased = results["spoke_pass_9Hz"]["perceived"]
    control = results["spoke_pass_2Hz"]["perceived"]
    ok = bool(aliased < 0.0 and control > 0.0)
    return {"aliased_9Hz": aliased, "control_2Hz": control, "pass": ok,
            "frames_aliased": results["spoke_pass_9Hz"]["frames"],
            "detail": f"9 Hz spoke-pass reads {aliased:+.3f} rev/s "
                      f"(expect ~-0.125, reversed); 2 Hz control reads "
                      f"{control:+.3f} rev/s (expect ~+0.25, veridical)"}


# ---------------------------------------------------------------- T3

def test_suppression():
    """20 ms flash during a scripted saccade vs. during steady fixation."""
    size, fps = 64, 1000
    cx = size / 2
    dva = _dva_per_px(size)
    # 13-degree saccade: onset 990 ms, ~50 ms duration, flash at 1000-1020 ms
    amp_px = 13.0 / dva
    script_sac = [(0, cx, cx), (990.0, cx + amp_px, cx)]
    script_fix = [(0, cx, cx)]

    def max_dev(script):
        moments = _run(script, S.gen_flash(1.6, fps, size, flash_at_ms=1000.0,
                                           flash_dur_ms=20.0), size)
        # skip unsettled moments while the pipeline primes (t < 500 ms)
        return max(abs(p.mean() - 0.3) for t, p, _ in moments if t >= 500.0)

    dev_fix = max_dev(script_fix)
    dev_sac = max_dev(script_sac)
    ratio = dev_sac / max(dev_fix, 1e-12)
    return {"dev_fixation": dev_fix, "dev_saccade": dev_sac,
            "ratio": ratio, "pass": bool(ratio < 0.25),
            "detail": f"saccade-case deviation {dev_sac:.4f} vs "
                      f"fixation-case {dev_fix:.4f} (ratio {ratio:.3f} < 0.25)"}


def run_all():
    return {
        "T0_latency": test_latency(),
        "T1_flicker": test_flicker(),
        "T2_wagon_wheel": test_wagon_wheel(),
        "T3_suppression": test_suppression(),
        "T4_change_blindness": test_change_blindness(),
    }


# ---------------------------------------------------------------- T4

def test_change_blindness(n_seeds=6, max_cycles=25):
    """Flicker paradigm: one bar of 64 flips orientation between A and B.

    Without the mask the flip's local transient captures the saccade
    system in ~2 cycles; with the mask the transient is swamped and the
    change is found only by serial search (many cycles, often never).
    Masked runs that never find it are censored at max_cycles -- the
    human pattern for hard changes (Rensink et al.).
    """
    res = {}
    example = None
    for use_mask in (False, True):
        cyc = []
        for seed in range(n_seeds):
            sa, sb, ch = S.make_change_scene(seed=seed)
            r = A.run_trial(sa, sb, ch, use_mask=use_mask,
                            max_cycles=max_cycles)
            cyc.append(r["cycles_to_detect"])
            if use_mask and example is None and not r["found"]:
                example = r
        res["mask" if use_mask else "no_mask"] = cyc
    m_mask = float(np.mean(res["mask"]))
    m_no = float(np.mean(res["no_mask"]))
    ok = bool(m_mask >= 4 * m_no and m_no <= 3.0)
    return {"cycles_mask": res["mask"], "cycles_no_mask": res["no_mask"],
            "mean_mask": m_mask, "mean_no_mask": m_no,
            "example_masked_run": example, "pass": ok,
            "detail": f"no-mask {m_no:.1f} cycles vs masked {m_mask:.1f} "
                      f"(censored at {max_cycles}); expect ~2 vs >>8"}
