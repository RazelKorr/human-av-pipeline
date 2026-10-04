"""compare.py: batch post-hoc output vs streaming harness output.

Loads fixations.npy, frame_energies.npy, run_report.json, and
video_percept.mp4 from a batch run directory and a stream run
directory and reports how close the stream got:

  - moment counts and timestamp grids (must match exactly)
  - per-moment fixation displacement (work-px and degrees)
  - suppressed-flag agreement
  - salience-channel energy deltas
  - saccade counts
  - percept-video frame deltas + event lists via the same
    detect_events() the sensorium graph uses

Usage:
  python3 streaming/compare.py output/st_batch_62s output/st_stream_62s \\
      --out output/st_stream_62s/compare_report.json

Writes compare_report.json and prints the summary. Exit code is 0
regardless; strictness is a reporting decision, not a crash.
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from st_color_graph import detect_events
from hvp import baseline as B


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    if name.endswith(".npy"):
        return np.load(p)
    with open(p) as f:
        return json.load(f)


def decode_diff_signal(path, w=160, h=90, batch=400):
    """Luma frame-difference signal, decoded in batches (low memory)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path,
           "-vf", f"scale={w}:{h},format=rgb24",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    fb = w * h * 3
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sig = []
    prev_l = None
    n = 0
    first = True
    while True:
        raw = proc.stdout.read(fb * batch)
        if not raw:
            break
        m = len(raw) // fb
        v = (np.frombuffer(raw[:m * fb], dtype=np.uint8)
             .reshape(m, h, w, 3).astype(np.float32) / 255.0)
        l = v @ weights
        d = np.abs(l[1:] - l[:-1]).mean(axis=(1, 2))
        if first:
            sig.append(0.0)  # graph convention: dd[0] = 0
            first = False
        elif prev_l is not None:
            sig.append(float(np.abs(l[0] - prev_l).mean()))
        sig.extend(float(x) for x in d)
        prev_l = l[-1]
        n += m
    proc.wait()
    return np.array(sig, dtype=np.float32), n


def decode_frame_stats(path_a, path_b, w=160, h=90, batch=400):
    """Mean/max abs frame diff between two videos, batched (low memory)."""
    def pipe(p):
        cmd = ["ffmpeg", "-v", "error", "-i", p,
               "-vf", f"scale={w}:{h},format=rgb24",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE)
    pa, pb = pipe(path_a), pipe(path_b)
    fb = w * h * 3
    tot = mx = cnt = 0.0
    n = 0
    while True:
        ra, rb = pa.stdout.read(fb * batch), pb.stdout.read(fb * batch)
        if not ra or not rb:
            break
        m = min(len(ra), len(rb)) // fb
        if m == 0:
            break
        va = np.frombuffer(ra[:m * fb], dtype=np.uint8).astype(np.float32)
        vb = np.frombuffer(rb[:m * fb], dtype=np.uint8).astype(np.float32)
        d = np.abs(va - vb) / 255.0
        tot += float(d.sum())
        mx = max(mx, float(d.max()))
        n += m
        cnt += d.size
    pa.wait()
    pb.wait()
    return n, (tot / cnt if cnt else 0.0), mx


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir")
    ap.add_argument("stream_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fix-w", type=float, default=640.0)
    args = ap.parse_args()

    rep = {"batch_dir": args.batch_dir, "stream_dir": args.stream_dir}
    lines = []

    def say(s):
        lines.append(s)
        print(s, flush=True)

    # ---- moments / fixations ----
    fb = load(args.batch_dir, "fixations.npy")
    fs = load(args.stream_dir, "fixations.npy")
    if fb is None or fs is None:
        say("MISSING fixations.npy in one directory -- cannot compare")
    else:
        rep["n_moments_batch"] = int(len(fb))
        rep["n_moments_stream"] = int(len(fs))
        say(f"moments: batch={len(fb)} stream={len(fs)} "
            f"(delta={len(fs) - len(fb)})")
        n = min(len(fb), len(fs))
        tdiff = float(np.abs(fb[:n, 0] - fs[:n, 0]).max()) if n else 0.0
        rep["timestamp_grid_max_diff_ms"] = tdiff
        say(f"timestamp grid max |dt|: {tdiff:.6f} ms")
        if n:
            dva = B.FIELD_WIDTH_DEG / args.fix_w
            disp_px = np.hypot(fb[:n, 1] - fs[:n, 1],
                               fb[:n, 2] - fs[:n, 2])
            rep["fixation_disp_px"] = {
                "mean": float(disp_px.mean()),
                "max": float(disp_px.max()),
                "n_nonzero": int((disp_px > 1e-9).sum()),
            }
            rep["fixation_disp_deg"] = {
                "mean": float((disp_px * dva).mean()),
                "max": float((disp_px * dva).max()),
            }
            say(f"fixation displacement: mean={disp_px.mean():.4f}px "
                f"({disp_px.mean() * dva:.4f} deg), "
                f"max={disp_px.max():.4f}px, "
                f"nonzero={int((disp_px > 1e-9).sum())}/{n}")
            agree = float((fb[:n, 3] == fs[:n, 3]).mean())
            rep["suppressed_flag_agreement"] = agree
            say(f"suppressed-flag agreement: {100 * agree:.2f}%")
            sf = np.abs(fb[:n, 3].astype(float) - fs[:n, 3].astype(float))
            rep["suppressed_flag_mismatches"] = int((sf > 0).sum())

    # ---- energies ----
    eb = load(args.batch_dir, "frame_energies.npy")
    es = load(args.stream_dir, "frame_energies.npy")
    if eb is None or es is None:
        say("MISSING frame_energies.npy in one directory")
    else:
        n = min(len(eb), len(es))
        d = np.abs(eb[:n] - es[:n])
        rep["energies"] = {
            "n_frames_batch": int(len(eb)),
            "n_frames_stream": int(len(es)),
            "max_abs_diff": float(d.max()) if n else 0.0,
            "per_channel_max": [float(x) for x in d.max(axis=0)] if n else [],
        }
        say(f"energies: frames batch={len(eb)} stream={len(es)}, "
            f"max|d|={rep['energies']['max_abs_diff']:.3e} "
            f"(per-channel {rep['energies']['per_channel_max']})")

    # ---- reports ----
    rb = load(args.batch_dir, "run_report.json")
    rs = load(args.stream_dir, "run_report.json")
    if rb and rs:
        rep["n_saccades_batch"] = rb.get("n_saccades")
        rep["n_saccades_stream"] = rs.get("n_saccades")
        say(f"saccades: batch={rb.get('n_saccades')} "
            f"stream={rs.get('n_saccades')}")
        say(f"wall: batch={rb.get('wall_clock_s')}s "
            f"stream={rs.get('wall_clock_s')}s")
        st = rs.get("streaming", {})
        if st:
            say(f"stream: {st.get('moments_per_wall_s')} moments/wall-s, "
                f"realtime_factor={st.get('realtime_factor')}x, "
                f"deadline_misses={st.get('chunk_deadline_misses')}, "
                f"frames_evicted={st.get('frames_evicted')}")

    # ---- percept video + events ----
    pb = os.path.join(args.batch_dir, "video_percept.mp4")
    ps = os.path.join(args.stream_dir, "video_percept.mp4")
    if os.path.exists(pb) and os.path.exists(ps):
        fps = (rs or {}).get("moment_hz") or (rb or {}).get("moment_hz") \
            or 20.0
        nb, mean_d, max_d = decode_frame_stats(pb, ps)
        rep["percept_frames_compared"] = int(nb)
        rep["percept_frame_diff"] = {"mean": mean_d, "max": max_d}
        say(f"percept frames: batch={nb} stream={nb}, "
            f"mean|d|={mean_d:.3e}, max|d|={max_d:.3e}")
        # events via the graph's own detector (luma diff signal)
        sigb, _ = decode_diff_signal(pb)
        sigs, _ = decode_diff_signal(ps)
        n = min(len(sigb), len(sigs))
        evb = detect_events(sigb[:n], fps)
        evs = detect_events(sigs[:n], fps)
        rep["events_batch"] = evb
        rep["events_stream"] = evs
        same = (len(evb) == len(evs)
                and all(abs(a[0] - b[0]) < 1e-9
                        and abs(a[1] - b[1]) < 1e-9
                        for a, b in zip(evb, evs)))
        rep["events_identical"] = bool(same)
        say(f"events: batch={len(evb)} stream={len(evs)} "
            f"identical={same}")
    else:
        say("percept mp4 missing in one directory -- skipping event check")

    out = args.out or os.path.join(args.stream_dir, "compare_report.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    say(f"wrote {out}")


if __name__ == "__main__":
    main()
