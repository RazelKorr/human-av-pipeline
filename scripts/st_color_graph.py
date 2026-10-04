#!/usr/bin/env python3
"""Fancy graph: what the color pipeline saw during the Star Tours ride.

Decodes the color percept video (every moment, no sampling) and computes
per-frame metrics:
  - luma: mean brightness
  - sat: mean saturation (colorfulness)
  - diff: mean abs frame-to-frame change (luma)
  - huebar: per-frame mean color strip (the ride's color journey)

Detects events from the diff signal (same bridging rule as event_map.py)
and renders a multi-panel figure:
  1. hue barcode — the film as a strip of color over time
  2. brightness + saturation curves (twin axes)
  3. change signal with event markers

Usage:
  python3 scripts/st_color_graph.py --percept output/st_color/video_percept.mp4 \
      --out output/st_color/st_graph.png
"""
import argparse
import json
import subprocess

import numpy as np

W, H = 160, 90


def decode(percept, fps):
    cmd = ["ffmpeg", "-v", "error", "-i", percept,
           "-vf", f"scale={W}:{H},format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fb = W * H * 3
    frames = []
    while True:
        raw = proc.stdout.read(fb)
        if len(raw) < fb:
            break
        frames.append(np.frombuffer(raw, dtype=np.uint8)
                      .reshape(H, W, 3).astype(np.float32) / 255.0)
    proc.wait()
    return np.stack(frames)


def metrics(frames):
    n = len(frames)
    luma = np.zeros(n, dtype=np.float32)
    sat = np.zeros(n, dtype=np.float32)
    diff = np.zeros(n, dtype=np.float32)
    huebar = np.zeros((n, 3), dtype=np.float32)
    prev_l = None
    for i, f in enumerate(frames):
        l = f @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        mx = f.max(axis=-1)
        mn = f.min(axis=-1)
        s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        luma[i] = l.mean()
        sat[i] = s.mean()
        huebar[i] = f.reshape(-1, 3).mean(axis=0)
        if prev_l is not None:
            diff[i] = np.abs(l - prev_l).mean()
        prev_l = l
    return luma, sat, diff, huebar


def detect_events(diff, fps, thresh=0.08):
    # Frame-to-frame diffs shrink as the moment grid gets finer
    # (consecutive moments overlap more), so scale the threshold with
    # the sampling interval to keep event semantics comparable.
    thresh = thresh * 10.0 / fps
    bridge = max(2, int(round(5 * fps / 10.0)))
    events = []
    in_event, start = False, 0
    n = len(diff)
    for i in range(n):
        active = diff[i] > thresh
        if active and not in_event:
            in_event, start = True, i
        elif not active and in_event:
            j, quiet = i, 0
            while j < n and diff[j] <= thresh and quiet < bridge:
                quiet += 1
                j += 1
            if quiet >= bridge:
                events.append((start / fps, (i - 1) / fps,
                               round(float(diff[start:i].max()), 2)))
                in_event = False
    if in_event:
        events.append((start / fps, (n - 1) / fps,
                       round(float(diff[start:].max()), 2)))
    return events


def smooth(x, k=15):
    k = int(k)
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percept", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio_rms", default=None,
                    help="npy file with per-moment (10Hz) audio RMS envelope")
    ap.add_argument("--fixations", default=None,
                    help="npy file with per-moment (t_ms, fx, fy, suppressed)")
    ap.add_argument("--fix_w", type=float, default=640.0)
    ap.add_argument("--fix_h", type=float, default=360.0)
    ap.add_argument("--fps", type=float, default=10.0,
                    help="percept-video frame rate (= moment Hz)")
    ap.add_argument("--title", default="Star Tours — what the color pipeline saw")
    args = ap.parse_args()
    fps = args.fps
    # smoothing kernels are in samples; keep their *time* constants fixed
    kscale = fps / 10.0

    print("decoding percept video…", flush=True)
    frames = decode(args.percept, fps)
    n = len(frames)
    dur = n / fps
    print(f"{n} moments, {dur:.1f}s", flush=True)
    luma, sat, diff, huebar = metrics(frames)
    events = detect_events(diff, fps)
    print(f"{len(events)} events detected", flush=True)

    audio = None
    if args.audio_rms:
        audio = np.load(args.audio_rms).astype(np.float32)
        if len(audio) != n:
            # resample to moment count
            xi = np.linspace(0, len(audio) - 1, n)
            audio = np.interp(xi, np.arange(len(audio)), audio).astype(np.float32)
        print(f"audio envelope: {len(audio)} samples", flush=True)

    fix = None
    if args.fixations:
        fix = np.load(args.fixations).astype(np.float32)
        # columns: t_ms, fx, fy, suppressed
        print(f"fixations: {len(fix)} moments, "
              f"{int(fix[:, 3].sum())} suppressed", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(n) / fps
    panels = ["barcode", "bs", "audio", "track", "change", "events"]
    if audio is None:
        panels.remove("audio")
    if fix is None:
        panels.remove("track")
    ratios = {"barcode": 1, "bs": 2.2, "audio": 2.2, "track": 2.2,
              "change": 2.2, "events": 2.2}
    fig = plt.figure(figsize=(16, 3 + 2.4 * (len(panels) - 1)))
    gs = fig.add_gridspec(len(panels), 1,
                          height_ratios=[ratios[p] for p in panels],
                          hspace=0.35)
    fig.suptitle(args.title, fontsize=15, fontweight="bold")
    pi = 0

    # 1. hue barcode
    ax0 = fig.add_subplot(gs[pi]); pi += 1
    ax0.imshow(huebar[np.newaxis, :, :], aspect="auto",
               extent=[0, dur, 0, 1])
    ax0.set_xlim(0, dur)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel("mean\ncolor", fontsize=9)

    # 2. brightness + saturation
    ax1 = fig.add_subplot(gs[pi], sharex=ax0); pi += 1
    ax1.plot(t, smooth(luma, int(15*kscale)), color="#f5c542", lw=1.4, label="brightness")
    ax1.set_ylabel("brightness", color="#f5c542", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#f5c542")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(t, smooth(sat, int(25*kscale)), color="#c44df0", lw=1.4, label="saturation")
    ax2.set_ylabel("saturation", color="#c44df0", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="#c44df0")
    ax2.set_ylim(0, 1)
    ax1.grid(alpha=0.2)

    gi = pi
    if audio is not None:
        # 3. audio loudness (what the pipeline hears)
        axa = fig.add_subplot(gs[gi], sharex=ax0)
        gi += 1
        axa.fill_between(t, smooth(audio, int(9*kscale)), color="#3ddc84", alpha=0.55)
        axa.plot(t, smooth(audio, int(9*kscale)), color="#1d7a46", lw=1.0)
        axa.set_ylabel("audio\nloudness", fontsize=10)
        axa.set_ylim(0, max(audio.max() * 1.15, 1e-6))
        axa.grid(alpha=0.2)

    if fix is not None:
        # 4. visual tracking: where the eyes were (normalized), saccades marked
        axt = fig.add_subplot(gs[gi], sharex=ax0)
        gi += 1
        ft = fix[:, 0] / 1000.0
        fx = fix[:, 1] / args.fix_w
        fy = fix[:, 2] / args.fix_h
        supp = fix[:, 3] > 0.5
        # saccade flights: gaze jumps between consecutive moments (the
        # the pipeline's binary suppressed flag only fires when a whole
        # integration window falls mid-saccade, which undercounts flights
        # at coarse moment spacings
        jumps = np.zeros(len(fx), dtype=bool)
        if len(fx) > 1:
            d = np.hypot(np.diff(fx), np.diff(fy))
            jumps[1:] = d > 0.05
        axt.plot(ft, fx, color="#ff7a3d", lw=0.8, label="gaze x")
        axt.plot(ft, 1.0 - fy, color="#3dc4ff", lw=0.8, label="gaze y (up+)")
        axt.scatter(ft[jumps], np.full(jumps.sum(), 1.02), s=6, color="#d33",
                    marker="|", label="saccade")
        axt.set_ylabel("visual\ntracking", fontsize=10)
        axt.set_ylim(-0.05, 1.1)
        axt.legend(fontsize=8, loc="upper right", ncol=3)
        axt.grid(alpha=0.2)
        print(f"saccades detected from gaze jumps: {int(jumps.sum())}",
              flush=True)

    # 5. change signal + events
    ax3 = fig.add_subplot(gs[gi], sharex=ax0)
    gi += 1
    ax3.plot(t, smooth(diff, int(5*kscale)), color="#4da6f0", lw=1.0, label="frame change")
    for s, e, pk in events:
        ax3.axvspan(s, e, color="#4da6f0", alpha=0.18)
    ax3.set_ylabel("change", fontsize=10)
    ax3.grid(alpha=0.2)

    # 5. event timeline
    ax4 = fig.add_subplot(gs[gi], sharex=ax0)
    ax4.set_ylim(0, 1)
    for i, (s, e, pk) in enumerate(events):
        ax4.barh(0.5, e - s, left=s, height=0.6, color="#4da6f0", alpha=0.7)
        if e - s > 4:
            ax4.text(s + 0.3, 0.5, f"{s:.0f}s", va="center", ha="left",
                     fontsize=7, color="white")
    ax4.set_yticks([])
    ax4.set_ylabel("events", fontsize=10)
    ax4.set_xlabel("seconds", fontsize=10)

    for ax in (ax1, ax3, ax4):
        ax.set_xlim(0, dur)

    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"wrote {args.out}", flush=True)

    meta = {"n_moments": n, "duration_s": round(dur, 1),
            "n_events": len(events),
            "mean_brightness": round(float(luma.mean()), 3),
            "mean_saturation": round(float(sat.mean()), 3),
            "events": [{"start_s": round(s, 1), "end_s": round(e, 1),
                        "peak_diff": pk} for s, e, pk in events]}
    with open(args.out.replace(".png", "_metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote metrics json", flush=True)


if __name__ == "__main__":
    main()
