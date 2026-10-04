"""run_stream.py: the v2 color-saccades pipeline as a live stream.

  feeder (file played as if live) -> OnlineAttentionDriver (saccade
  decisions, 200 ms latency) -> OnlineVisionPipeline (bounded buffer,
  20 Hz moments) -> video_percept.mp4 + fixations.npy + frame_energies.npy

The driver and the render run in ONE online loop (the batch driver's
two passes merged): saccade decisions are mirrored into the render
controller as they are made. This is valid because a render query at
t_c = t_moment - latency - window/2 only ever sees saccades with
t_on <= t_c, and every such saccade was decided at t_on - 200 ms,
strictly before the stream time at which the query runs. No lookahead.

Usage:
  python3 streaming/run_stream.py input/star_tours_1_ride_film.mp4 \\
      --seconds 62 --outdir output/st_stream_62s
  python3 streaming/run_stream.py input/star_tours_1_ride_film.mp4 \\
      --seconds 62 --outdir output/st_stream_rt --realtime --speed 1.0

Writes into --outdir: video_percept.mp4, fixations.npy,
frame_energies.npy, scanpath.npy, run_report.json -- the same
artifacts as scripts/run_video_saccades_color.py, for direct
comparison.
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_video
from hvp import baseline as B
from feeder import ChunkedFeeder
from online_driver import OnlineAttentionDriver, DEFAULT_MOTION_THUMB_WH
from online_render import OnlineVisionPipeline


def _validate_args(args):
    """Reject silently-degenerate flag combos before any work starts.

    Pure argument checking -- no pipeline math touched.
    """
    if args.speed is not None and args.speed <= 0:
        raise SystemExit("--speed must be positive")
    if args.motion_weight is not None and args.motion_weight <= 0:
        raise SystemExit("--motion-weight must be positive "
                         "(None = off, v2-identical)")
    if args.pursuit and not args.motion_weight:
        print("WARNING: --pursuit needs --motion-weight; the magno "
              "channel is off, so no pursuit segments will be "
              "scheduled (v2-identical saccades only).", flush=True)
    if args.motion_thumb is not None and args.dual_decode:
        # The feeder's third split-graph output only exists in
        # single-decode split mode; with dual_decode the driver
        # derives the thumbnail in-process from the work frame.
        # Behavior is correct either way -- math unchanged.
        print("NOTE: --motion-thumb is honored in-process with "
              "--dual-decode (no feeder third output).", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--chunk-s", type=float, default=2.0,
                    help="feeder chunk size in seconds (default 2)")
    ap.add_argument("--realtime", action="store_true",
                    help="pace frame emission to the wall clock")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="realtime speed multiplier")
    ap.add_argument("--moment-ms", type=float, default=50.0)
    ap.add_argument("--chroma-weight", type=float, default=None)
    ap.add_argument("--motion-weight", type=float, default=None,
                    help="magno-channel weight in salience (None = off, "
                         "v2-identical; positive = motion attracts saccades)")
    ap.add_argument("--pursuit", action="store_true",
                    help="enable smooth-pursuit tracking of coherently "
                         "moving targets (needs --motion-weight; off = "
                         "v2-identical)")
    ap.add_argument("--motion-thumb", default=None, metavar="WxH",
                    help="magno-channel thumbnail size, e.g. 160x90 "
                         "(default: 96x54 16:9; the benchmarked winner "
                         "becomes the default)")
    ap.add_argument("--motion-thumb-source", default="work",
                    choices=["work", "attn"],
                    help="'work' = 16:9 thumbnail from the work frame "
                         "(default, true aspect); 'attn' = legacy "
                         "anamorphic fallback from the 224px frame")
    ap.add_argument("--fovea", type=float, default=None)
    ap.add_argument("--chroma-mode", default="luma_ratio",
                    choices=["foveated", "passthrough", "luma_ratio"],
                    help="'foveated': validated v2 math, chroma "
                         "desaturates with eccentricity (explicit opt-in "
                         "for bit-agreement with the v2 reference). "
                         "'passthrough': foveate luminance only, chroma "
                         "unfolded. 'luma_ratio' (default): foveate luma, "
                         "rescale RGB by the foveated-luma ratio, no "
                         "YCbCr round-trip (fastest)")
    ap.add_argument("--fovea-levels", type=int, default=3,
                    help="blur-pyramid depth (3 = default fast combo; "
                         "5 = v2; fewer = coarser eccentricity "
                         "quantization, changes the math)")
    ap.add_argument("--mask-cache", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="memoize eccentricity-band masks per fixation "
                         "(bit-identical to uncached; default on; "
                         "--no-mask-cache for explicit v2 mode)")
    ap.add_argument("--box-sigma-threshold", type=float, default=None,
                    help="above this per-band sigma use the stacked "
                         "box-blur approximation (not bit-identical)")
    ap.add_argument("--dual-decode", action="store_true",
                    help="decode the video twice (224px + work res) like "
                         "the batch v2 driver; default is a single decode "
                         "with in-process downsampling")
    ap.add_argument("--single-mode", default="split", choices=["pil", "split"],
                    help="single-decode downsample method: 'split' = one "
                         "ffmpeg process with a split filter graph (224px "
                         "output bit-identical to dual-decode; default); "
                         "'pil' = in-process PIL bilinear (faster, but "
                         "224px frames differ slightly -- see report)")
    args = ap.parse_args()
    _validate_args(args)
    moment_ms = args.moment_ms
    moment_hz = 1000.0 / moment_ms
    w, h = args.width, args.height
    if args.motion_thumb is not None:
        try:
            _tw, _th = args.motion_thumb.lower().split("x")
            motion_thumb_wh = (int(_tw), int(_th))
        except ValueError:
            raise SystemExit("--motion-thumb must look like WxH, e.g. 160x90")
    else:
        motion_thumb_wh = None

    os.makedirs(args.outdir, exist_ok=True)
    meta = run_video.probe(args.video)
    fps = meta["fps"]
    total = args.seconds or meta["duration"]
    total = min(total, meta["duration"])
    t_end_ms = total * 1000.0
    print(f"video: {meta['w']}x{meta['h']} @ {fps:.2f} fps, "
          f"{meta['duration']:.1f}s; streaming {total:.1f}s "
          f"({'realtime x%.1f' % args.speed if args.realtime else 'as fast as possible'})",
          flush=True)

    feeder_thumb_wh = None
    eff_thumb_wh = motion_thumb_wh or DEFAULT_MOTION_THUMB_WH
    if (args.motion_weight and args.motion_thumb_source == "work"
            and not args.dual_decode and args.single_mode == "split"):
        # Third split-graph output at the motion thumb size: the
        # driver consumes it directly, skipping the ~2.9 ms/frame PIL
        # work-frame downsample. Any other decode configuration falls
        # back to the driver deriving the thumbnail in-process.
        feeder_thumb_wh = eff_thumb_wh
    feeder = ChunkedFeeder(args.video, total, fps,
                           attn_wh=(224, 224), work_wh=(w, h),
                           chunk_s=args.chunk_s,
                           realtime=args.realtime, speed=args.speed,
                           dual_decode=args.dual_decode,
                           single_mode=args.single_mode,
                           motion_thumb_wh=feeder_thumb_wh)
    driver = OnlineAttentionDriver(fps, t_end_ms,
                                   chroma_weight=args.chroma_weight,
                                   motion_weight=args.motion_weight,
                                   pursuit=args.pursuit,
                                   motion_thumb_wh=motion_thumb_wh,
                                   motion_thumb_source=args.motion_thumb_source)
    sx, sy = w / 224.0, h / 224.0
    dva2 = B.FIELD_WIDTH_DEG / w
    render = OnlineVisionPipeline((h, w, 3), dva2, [(0, w / 2.0, h / 2.0)],
                                  fovea_radius_deg=args.fovea,
                                  moment_ms=moment_ms,
                                  chroma_mode=args.chroma_mode,
                                  fovea_levels=args.fovea_levels,
                                  mask_cache=args.mask_cache,
                                  box_sigma_threshold=args.box_sigma_threshold)

    out_mp4 = os.path.join(args.outdir, "video_percept.mp4")
    # NOTE: ultrafast preset -- this mp4 is a visualization artifact, not
    # a validation input (all agreement metrics come from the .npy files).
    # The default medium preset spends ~10-70 ms/frame on encoding, which
    # would dominate the realtime measurement of the pipeline itself.
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{moment_hz:.1f}",
         "-i", "pipe:0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         out_mp4],
        stdin=subprocess.PIPE)

    t0_wall = time.time()
    n_frames = n_moments = 0
    fix_log = []
    frame_proc = []       # per-frame processing seconds (driver+render+write)
    chunk_proc = []       # per-chunk processing seconds
    deadline_misses = 0
    t_last = 0.0

    for chunk in feeder:
        c0 = time.time()
        media_dur = chunk.t1_s - chunk.t0_s
        for (t_ms, f_attn, f_work, f_thumb) in chunk.frames:
            f0 = time.perf_counter()
            # 1. attention driver decides eye movements online (224px coords)
            for d in driver.push(t_ms, f_attn, f_work, f_thumb):
                # 2. mirror into the render controller (work-px coords).
                # Rescale is linear and dva rescales consistently, so
                # saccade durations match the batch pass-2 controller.
                if d[0] == "sac":
                    _, t_on, tx, ty = d
                    render.add_saccade(t_on, tx * sx, ty * sy)
                else:
                    _, t0, t1, vx, vy = d
                    render.add_pursuit(t0, t1, vx * sx, vy * sy)
            # 3. push work-res frame, pull ready moments
            render.push(f_work, t_ms)
            for (mt, p, m) in render.pull(t_ms, t_end_ms):
                ff.stdin.write(
                    (np.clip(p, 0, 1) * 255).astype(np.uint8).tobytes())
                fx, fy = m["fixation"]
                fix_log.append((mt, float(fx), float(fy),
                                bool(m["suppressed"])))
                n_moments += 1
            frame_proc.append(time.perf_counter() - f0)
            n_frames += 1
            t_last = t_ms
        # Stream time ends at the last pushed frame; flush any moments
        # whose window completed with it (capped at the media duration).
        for (mt, p, m) in render.pull(t_last, t_end_ms):
            ff.stdin.write(
                (np.clip(p, 0, 1) * 255).astype(np.uint8).tobytes())
            fx, fy = m["fixation"]
            fix_log.append((mt, float(fx), float(fy), bool(m["suppressed"])))
            n_moments += 1
        proc_wall = time.time() - c0
        chunk_proc.append(proc_wall)
        # Deadlines only exist when paced: in fast mode the consumer
        # pulls as fast as it can and every chunk trivially "misses".
        miss = (args.realtime
                and proc_wall > media_dur / args.speed + 1e-9)
        deadline_misses += int(miss)
        print(f"chunk [{chunk.t0_s:.0f}-{chunk.t1_s:.0f}s]: "
              f"frames={len(chunk.frames)} moments={n_moments} "
              f"proc={proc_wall:.2f}s vs media {media_dur:.2f}s"
              f"{' MISSED' if miss else ''} "
              f"buf={render.buffer_ms:.0f}ms evicted={render.n_evicted}",
              flush=True)

    ff.stdin.close()
    ff.wait()
    wall_total = time.time() - t0_wall
    print(f"pushed {n_frames} frames, pulled {n_moments} moments, "
          f"wall {wall_total:.1f}s", flush=True)

    fix_arr = np.array(fix_log, dtype=np.float32)
    np.save(os.path.join(args.outdir, "fixations.npy"), fix_arr)
    print(f"wrote fixations.npy ({len(fix_arr)} moments)", flush=True)
    energies_arr = np.array(driver.energies, dtype=np.float32)
    np.save(os.path.join(args.outdir, "frame_energies.npy"), energies_arr)
    print(f"wrote frame_energies.npy ({len(energies_arr)} frames)",
          flush=True)
    scan = np.array([(t, x * sx, y * sy) for (t, x, y) in driver.scanpath],
                    dtype=np.float32)
    np.save(os.path.join(args.outdir, "scanpath.npy"), scan)
    if driver.motion is not None:
        me_arr = np.array(driver.motion_energies, dtype=np.float32)
        np.save(os.path.join(args.outdir, "motion_energies.npy"), me_arr)
        print(f"wrote motion_energies.npy ({len(me_arr)} frames)", flush=True)
    if driver.pursuit_log:
        pl = np.array([(t0, t1, vx * sx, vy * sy)
                       for (t0, t1, vx, vy) in driver.pursuit_log],
                      dtype=np.float32)
        np.save(os.path.join(args.outdir, "pursuit_log.npy"), pl)
        print(f"wrote pursuit_log.npy ({len(pl)} segments)", flush=True)

    fp = np.array(frame_proc)
    report = {
        "video": os.path.basename(args.video),
        "source_meta": meta,
        "processed_seconds": total,
        "working_res": [w, h],
        "n_saccades": driver.n_saccades,
        "n_pursuits": driver.n_pursuits,
        "motion_weight": args.motion_weight,
        "pursuit": bool(args.pursuit),
        "pursuit_effective": bool(driver.pursuit),
        "frames_pushed": n_frames,
        "moments_pulled": n_moments,
        "wall_clock_s": round(wall_total, 1),
        "color": True,
        "chroma_mode": args.chroma_mode,
        "fovea_levels": args.fovea_levels,
        "mask_cache": bool(args.mask_cache),
        "box_sigma_threshold": args.box_sigma_threshold,
        "dual_decode": bool(args.dual_decode),
        "single_mode": args.single_mode,
        "saccades": True,
        "color_attention": True,
        "moment_ms": moment_ms,
        "moment_hz": moment_hz,
        "chroma_weight": args.chroma_weight,
        "motion_thumb": (f"{motion_thumb_wh[0]}x{motion_thumb_wh[1]}"
                         if motion_thumb_wh else "default"),
        "motion_thumb_source": args.motion_thumb_source,
        "motion_thumb_direct": (f"{feeder_thumb_wh[0]}x{feeder_thumb_wh[1]}"
                                if feeder_thumb_wh else None),
        "integration_window_ms": B.INTEGRATION_WINDOW_MS,
        "streaming": {
            "realtime": args.realtime,
            "speed": args.speed,
            "chunk_s": args.chunk_s,
            "frames_evicted": render.n_evicted,
            "final_buffer_ms": round(render.buffer_ms, 1),
            "frame_proc_s": {
                "mean": round(float(fp.mean()), 4),
                "p50": round(float(np.median(fp)), 4),
                "p99": round(float(np.percentile(fp, 99)), 4),
                "max": round(float(fp.max()), 4),
                "frame_interval_s": round(1.0 / fps, 4),
            },
            "chunk_deadline_misses": deadline_misses,
            "n_chunks": len(chunk_proc),
            "moments_per_wall_s": round(n_moments / wall_total, 2),
            "realtime_factor": round(total / wall_total, 2),
        },
    }
    with open(os.path.join(args.outdir, "run_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote run_report.json", flush=True)


if __name__ == "__main__":
    main()
