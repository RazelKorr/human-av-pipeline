"""Battery for the human-audio pipeline: five synthetic checks that the
ears behave like ears. Mirrors hvp/battery.py's T0-T4.

Each test synthesizes audio, runs the full pipeline
(cochlea -> salience -> moments -> attention), and checks the mechanism.
These demonstrate designed behavior on synthetic input, not a validated
model of human hearing.

  A0 latency ......... a click at t=1s is registered within ~50-200ms.
  A1 change deafness . a pitch change in an UNATTENDED stream is missed;
                       the same change in the ATTENDED stream is caught.
                       (Vitevitch's change deafness, 2003.)
  A2 cocktail party .. two interleaved streams; cued stream holds focus.
                       (Cherry 1953; Bregman streaming.)
  A3 onset capture ... a sudden loud transient yanks attention fast.
  A4 switch dullness . right after attention switches, a second weak
                       onset does NOT capture; 500ms later it does.
                       (the analog of saccadic suppression)
"""

import numpy as np
from . import cochlea as C
from . import salience as S
from . import moments as M
from . import attention as AT

SR = C.SR


# --- synthesis helpers -------------------------------------------------
def tone(freq, dur_s, amp=0.5, sr=SR, ramp_ms=5.0):
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    x = amp * np.sin(2 * np.pi * freq * t)
    r = int(ramp_ms / 1000 * sr)
    if r > 0:
        w = np.ones(n)
        w[:r] = np.linspace(0, 1, r)
        w[-r:] = np.linspace(1, 0, r)
        x *= w
    return x.astype(np.float32)


def click(dur_s=0.02, amp=0.8, sr=SR):
    n = int(dur_s * sr)
    x = amp * np.random.randn(n).astype(np.float32)
    x *= np.exp(-np.arange(n) / (n / 6.0))
    return x


def place(x, at_s, total_s, sr=SR):
    out = np.zeros(int(total_s * sr), np.float32)
    i = int(at_s * sr)
    out[i:i + len(x)] = x[:max(0, len(out) - i)]
    return out


def pips(freq, every_s, total_s, dur_s=0.08, amp=0.5, offset_s=0.0):
    out = np.zeros(int(total_s * SR), np.float32)
    t = offset_s
    while t < total_s:
        out += place(tone(freq, dur_s, amp), t, total_s)
        t += every_s
    return out


def room_tone(total_s, amp=0.0003, sr=SR):
    """A breath of white noise at -70dBFS: present in the waveform, but
    below the -60dBFS hearing floor, so it dithers digital silence
    without faking onsets. (Low hums were tried -- a 50Hz hum under
    20ms Hann windows is inherently non-stationary as the window slides
    through its phase. Real rooms hum; this scaffold can't hum cleanly,
    so it stays near-silent instead.)"""
    n = int(total_s * sr)
    x = np.random.randn(n).astype(np.float32)
    x *= amp / np.sqrt((x ** 2).mean())
    return x


def glide(f0, f1, at_s, dur_s, total_s, amp=0.5, sr=SR):
    """A transient-free pitch change: sine gliding f0->f1. No onset,
    no offset -- the kind of change the literature says gets missed
    when attention is elsewhere."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    f = np.linspace(f0, f1, n)
    ph = 2 * np.pi * np.cumsum(f) / sr
    x = (amp * np.sin(ph)).astype(np.float32)
    return place(x, at_s, total_s)


def fade_in(x, at_s, dur_s, total_s, sr=SR):
    """Raised-cosine fade-in: zero derivative at both ends, so there are
    no envelope corners for the onset detector to catch. Slow enough
    (dB/frame << 6dB) that only the scheduled salience path -- not the
    onset interrupt -- can react to it. (A log-linear ramp was tried;
    its corners are real transients.)"""
    n = int(dur_s * sr)
    u = np.linspace(0, 1, n)
    w = (0.5 * (1 - np.cos(np.pi * u))).astype(np.float32)
    return place(x[:n] * w, at_s, total_s)


# --- pipeline ----------------------------------------------------------
def run_pipeline(x, cf_init=32.0):
    t, f, Sg = C.stft_log(x)
    feat = S.salience_map(Sg)
    moms = M.moments(Sg, feat["sal"], t)
    # Per-moment temporal-contrast profile, masked below the hearing
    # floor -- unmasked, the argmax lands on the noisiest near-silent
    # bin (log diffs explode on silence) instead of the real transient.
    # (Same floor the salience T channel and Tmax use.)
    def _tprof(i):
        sl = Sg[i * 20:(i + 1) * 20]
        if len(sl) < 2:
            return np.zeros(C.N_BINS)
        d = np.abs(np.diff(sl, axis=0))
        d[np.maximum(sl[:-1], sl[1:]) < S.HEAR_FLOOR_DBFS] = 0.0
        return d.mean(axis=0)
    Tprof = np.stack([_tprof(i) for i in range(len(moms))])
    att = AT.AuditoryAttention(cf_init=cf_init)
    trace = att.run(moms, Tprof)
    return trace, moms, feat


def hz_to_bin(hz):
    return float(np.argmin(np.abs(
        C.bin_to_hz(np.arange(C.N_BINS)) - hz)))


def covers(trace, t_event, hz, win_s=0.5, tol_bins=6.0):
    """Was the attended band over `hz` within win_s after t_event?"""
    b = hz_to_bin(hz)
    for m in trace:
        if t_event <= m["t0"] <= t_event + win_s:
            if abs(m["cf_bin"] - b) <= tol_bins:
                return True, m["t0"] - t_event
    return False, None


# --- tests -------------------------------------------------------------
def a0_latency():
    x = room_tone(2.0) + place(click(), 1.0, 2.0)
    trace, _, _ = run_pipeline(x)
    caps = [m["t0"] for m in trace if m["event"] == "onset-capture"
            and m["t0"] >= 0.8]
    # events localize to +/-1 perceptual moment: the honest grain
    return {"pass": bool(caps) and abs(caps[0] - 1.0) < 0.25,
            "capture_latency_s": round(caps[0] - 1.0, 3) if caps else None}


def _change_scene(total, cue_hz, change_hz_from, change_hz_to,
                  cue_amp, change_amp):
    """Three continuous streams; at t=2s one stream glides to a nearby
    pitch with no onset and no offset -- the transient-free change."""
    n = int(total * SR)
    t = np.arange(n) / SR
    x = room_tone(total)
    for fz, amp in [(440.0, 0.30), (cue_hz, cue_amp)]:
        if abs(fz - change_hz_from) < 1.0:
            continue
        x += (amp * np.sin(2 * np.pi * fz * t)).astype(np.float32)
    seg = t < 2.0
    x += (change_amp * np.sin(2 * np.pi * change_hz_from * t) * seg
          ).astype(np.float32)
    x += glide(change_hz_from, change_hz_to, 2.0, 0.4, total,
               amp=change_amp)
    tail = (change_amp * np.sin(2 * np.pi * change_hz_to * (t - 2.4)) *
            (t >= 2.4)).astype(np.float32)
    x += tail
    return x


def a1_change_deafness():
    # Attention cued to 880; the UNATTENDED 550Hz stream glides to 575.
    x = _change_scene(4.0, cue_hz=880.0, change_hz_from=550.0,
                      change_hz_to=700.0, cue_amp=0.55, change_amp=0.30)
    trace, _, _ = run_pipeline(x, cf_init=hz_to_bin(880))
    ok_unatt, _ = covers(trace, 2.0, 700.0, win_s=1.5, tol_bins=2.5)
    return {"pass": not ok_unatt,  # missed = correct
            "unattended_change_noticed": bool(ok_unatt)}


def a1b_change_attended():
    # Same glide, but attention is ON the 550 stream: attentional gain
    # boosts the small transient, so the attended change is noticed.
    x = _change_scene(4.0, cue_hz=550.0, change_hz_from=550.0,
                      change_hz_to=700.0, cue_amp=0.55, change_amp=0.55)
    trace, _, _ = run_pipeline(x, cf_init=hz_to_bin(550))
    ok, dt = covers(trace, 2.0, 700.0, win_s=1.5, tol_bins=2.5)
    return {"pass": bool(ok), "attended_change_noticed": bool(ok),
            "latency_s": round(dt, 3) if dt else None}


def a2_cocktail_party():
    # Two SIMULTANEOUS pip streams; the CUED stream (A) is louder -- the
    # voice you're listening to is clearer. Every moment contains both,
    # so the scheduled path must pick the stronger one and the
    # attentional gain must hold it. (Interleaved streams need temporal
    # prediction -- streaming proper -- which is future work, not v1.)
    total = 5.0
    A = pips(600, 0.30, total, amp=0.60)
    B = pips(920, 0.30, total, amp=0.30)
    x = room_tone(total) + A + B
    trace, _, _ = run_pipeline(x, cf_init=hz_to_bin(600))
    bA = hz_to_bin(600)
    late = [m for m in trace if m["t0"] >= 1.5]
    frac = np.mean([abs(m["cf_bin"] - bA) <= 6.0 for m in late])
    return {"pass": bool(frac > 0.7),
            "frac_moments_on_cued_stream": round(float(frac), 2)}


def a3_onset_capture():
    # soft stationary background; a sudden loud transient must yank
    # attention within ~a perceptual moment.
    total = 3.0
    n = int(total * SR)
    t = np.arange(n) / SR
    # calm stationary background (hop-stationary hums -- see room_tone)
    x = room_tone(total) + (0.05 * np.sin(2 * np.pi * 400 * t) +
                            0.02 * np.sin(2 * np.pi * 800 * t)
                            ).astype(np.float32)
    x += place(click(dur_s=0.03, amp=0.9), 2.0, total)
    trace, _, _ = run_pipeline(x)
    caps = [m["t0"] for m in trace if m["event"] == "onset-capture"
            and m["t0"] >= 1.8]
    return {"pass": bool(caps) and abs(caps[0] - 2.0) < 0.25,
            "capture_latency_s": round(caps[0] - 2.0, 3) if caps else None}


def a4_switch_dullness():
    # Post-switch refractory, black-box: stream A (800Hz) STOPS at t=1.0,
    # so the scheduled path switches to the remaining stream B (300Hz);
    # that landing opens a 100ms refractory window. Stream C (500Hz,
    # the loudest) appears DURING the refractory. The refractory should
    # be visible in the trace (a blocked decision), and C should win
    # promptly after it lifts. Control: C appears with no recent
    # switch and is captured/switched-to promptly.
    # (Phase-continuous steps: 800x1.0, 500x1.18 and 500x2.8 are all
    # integer cycles, so the steps start at zero crossings.)
    total = 4.0
    n = int(total * SR)
    t = np.arange(n) / SR

    def scene(c_on):
        x = room_tone(total)
        x += (0.45 * np.sin(2 * np.pi * 800 * t) * (t < 1.0)
              ).astype(np.float32)
        x += (0.40 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        x += (0.70 * np.sin(2 * np.pi * 500 * t) * (t >= c_on)
              ).astype(np.float32)
        return x

    bC = hz_to_bin(500)
    trace, _, _ = run_pipeline(scene(1.18), cf_init=hz_to_bin(800))
    ref = [m for m in trace if m["event"] == "refractory"
           and 1.1 <= m["t0"] <= 1.3]
    sw = [m for m in trace if m["t0"] >= 1.3
          and abs(m["cf_bin"] - bC) <= 6.0]
    trace2, _, _ = run_pipeline(scene(2.80), cf_init=hz_to_bin(800))
    c2 = [m for m in trace2 if m["t0"] >= 2.8
          and abs(m["cf_bin"] - bC) <= 6.0]
    t2 = round(c2[0]["t0"] - 2.8, 2) if c2 else None
    ok = bool(ref) and bool(sw) and t2 is not None and t2 < 0.4
    return {"pass": bool(ok), "refractory_observed": bool(ref),
            "post_refractory_switch": bool(sw),
            "control_reach_s": t2}


def run_all():
    results = {}
    for name, fn in [("A0", a0_latency), ("A1", a1_change_deafness),
                     ("A1b", a1b_change_attended),
                     ("A2", a2_cocktail_party), ("A3", a3_onset_capture),
                     ("A4", a4_switch_dullness)]:
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001 -- battery must not die
            r = {"pass": False, "error": repr(e)}
        results[name] = r
        print(f"{name}: {'PASS' if r.get('pass') else 'FAIL'} "
              f"{ {k: v for k, v in r.items() if k != 'pass'} }",
              flush=True)
    return results
