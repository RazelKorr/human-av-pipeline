#!/usr/bin/env python3
"""Build the frame-indexed sensorium bundle (frame_index.json).

The review addressing scheme from ideas/frame-indexed-bundles.md: the
primary timeline is the *film's own frames*, in sequence, and each frame
carries a bundle of the sensorium output associated with it — so the review
still sees the full movie while the perceptual data rides along as
annotation instead of requiring a linear re-watch of every moment.

Per source frame:
  t_ms        -- frame timestamp
  moments     -- moment t_ms values whose integration window included it
  fixation    -- [fx, fy] in work-px, from the nearest moment
  suppressed  -- nearest moment's saccadic-suppression flag
  audio_rms   -- RMS loudness in a frame-period window around t
  lum_e / chroma_e / trans_e -- salience-channel energies from pass 1
  event       -- index into the event list, or null

The frame->moment mapping is analytic (no stored cross-ref needed):
moment t integrates source frames in [t - latency - window, t - latency).

Usage:
  python3 scripts/build_frame_index.py --outdir output/st_full_v2 \
      --audio-wav /tmp/st_audio.wav \
      --events output/st_full_v2/st_sensorium_metrics.json
"""
import argparse
import json
import os
import subprocess

import numpy as np

LATENCY_MS = 100.0  # must match hvp.baseline.PIPELINE_LATENCY_MS


def load_wav_mono(path, target_sr=16000):
    """Decode any audio to mono float32 at target_sr via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1",
           "-ar", str(target_sr), "-f", "f32le", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32), target_sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--audio-wav", required=True)
    ap.add_argument("--events", required=True,
                    help="*_metrics.json with an 'events' list")
    args = ap.parse_args()

    report = json.load(open(os.path.join(args.outdir, "run_report.json")))
    moment_ms = float(report["moment_ms"])
    window_ms = float(report.get("integration_window_ms", 100.0))
    fps = float(report["source_meta"]["fps"])
    w, h = report["working_res"]

    fix = np.load(os.path.join(args.outdir, "fixations.npy"))  # t,fx,fy,supp
    en = np.load(os.path.join(args.outdir, "frame_energies.npy"))  # t,l,c,tr
    ev = json.load(open(args.events))["events"]

    n_frames = len(en)
    frame_t = en[:, 0]  # pass-1 frame timestamps, ms
    moment_t = fix[:, 0]

    print(f"{n_frames} frames @ {fps:.2f} fps, {len(moment_t)} moments @ "
          f"{1000.0/moment_ms:.0f} Hz", flush=True)

    # audio RMS per frame period
    wav, sr = load_wav_mono(args.audio_wav)
    frame_period_s = 1.0 / fps
    audio_rms = np.zeros(n_frames, dtype=np.float32)
    for i, t_ms in enumerate(frame_t):
        c = int(t_ms / 1000.0 * sr)
        hw = int(frame_period_s * sr / 2)
        seg = wav[max(0, c - hw):c + hw]
        audio_rms[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0

    # event lookup: frame t (s) -> event index
    ev_ranges = [(e["start_s"], e["end_s"]) for e in ev]

    def event_at(t_s):
        for idx, (s, e) in enumerate(ev_ranges):
            if s <= t_s <= e:
                return idx
        return None

    frames = []
    for i in range(n_frames):
        t = float(frame_t[i])
        # moments whose integration window contained this frame
        a = moment_t - LATENCY_MS - window_ms
        b = moment_t - LATENCY_MS
        in_win = [float(mt) for mt, lo, hi in zip(moment_t, a, b)
                  if lo <= t < hi]
        # nearest moment for fixation / suppression
        j = int(np.argmin(np.abs(moment_t - t)))
        frames.append({
            "t_ms": round(t, 2),
            "moments": [round(m, 2) for m in in_win],
            "fixation": [round(float(fix[j, 1]), 1),
                         round(float(fix[j, 2]), 1)],
            "suppressed": bool(fix[j, 3] > 0.5),
            "audio_rms": round(float(audio_rms[i]), 4),
            "lum_e": round(float(en[i, 1]), 4),
            "chroma_e": round(float(en[i, 2]), 4),
            "trans_e": round(float(en[i, 3]), 4),
            "event": event_at(t / 1000.0),
        })

    out = {
        "video": report["video"],
        "moment_ms": moment_ms,
        "integration_window_ms": window_ms,
        "source_fps": fps,
        "working_res": [w, h],
        "n_frames": n_frames,
        "n_moments": len(moment_t),
        "n_events": len(ev),
        "frames": frames,
    }
    path = os.path.join(args.outdir, "frame_index.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
