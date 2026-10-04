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
from hva.online import (moment_features, pick_onsets, SR as AUDIO_SR,
                        MOMENT_MS as AUDIO_MOMENT_MS, SPM as AUDIO_SPM)
from feeder import ChunkedFeeder, decode_audio_track
from bind_av import visual_transients, bind as bind_av
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
    ap.add_argument("--audio", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="extract the audio track and compute per-moment "
                         "cochlear features + onset events (passive: it "
                         "observes, never steers; default on; --no-audio "
                         "for a video-only run)")
    ap.add_argument("--transcribe-onsets", type=int, default=0, metavar="K",
                    help="attended transcription: transcribe ±4 s wav "
                         "slices around the top-K audio onsets by strength "
                         "(0 = off; the transcript is a spotlight, not a "
                         "floodlight)")
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

    # --- Audio (Phase 1: cochlea + onsets). Pre-decoded once: 269 s
    # @16 kHz mono is ~8.6 MB and decodes in ~2 s; slicing downstream
    # by sample count makes moment alignment exact by construction.
    # Passive by design: features observe the master grid, never steer
    # the driver or render, so the video path is bit-identical on/off.
    audio_on = bool(args.audio)
    audio_track, audio_prev_mag, audio_feat_log, audio_proc = \
        None, None, [], []
    if audio_on:
        try:
            a0 = time.time()
            audio_track, asr = decode_audio_track(args.video, total,
                                                 sr=AUDIO_SR)
            print(f"audio: decoded {len(audio_track)/asr:.1f}s @ {asr} Hz "
                  f"in {time.time()-a0:.1f}s", flush=True)
        except RuntimeError as e:
            print(f"WARNING: {e} -- continuing audio-off", flush=True)
            audio_on = False

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
    frame_t_log = []      # per-frame t_ms, for audio-visual binding
    frame_proc = []       # per-frame processing seconds (driver+render+write)
    chunk_proc = []       # per-chunk processing seconds
    deadline_misses = 0
    t_last = 0.0

    for chunk in feeder:
        c0 = time.time()
        media_dur = chunk.t1_s - chunk.t0_s
        # --- audio: per-chunk cochlear features on the master grid
        if audio_on:
            a0 = time.perf_counter()
            s0 = int(round(chunk.t0_s * AUDIO_SR))
            s1 = int(round(chunk.t1_s * AUDIO_SR))
            seg = audio_track[s0:s1]
            n_mom = int(round((chunk.t1_s - chunk.t0_s) /
                              (AUDIO_MOMENT_MS / 1000.0)))
            need = n_mom * AUDIO_SPM
            if len(seg) < need:
                seg = np.pad(seg, (0, need - len(seg)))
            blocks = seg[:need].reshape(n_mom, AUDIO_SPM)
            rms, flux, cent, audio_prev_mag = moment_features(
                blocks, prev_mag=audio_prev_mag)
            audio_feat_log.append(np.stack([rms, flux, cent], axis=1))
            audio_proc.append(time.perf_counter() - a0)
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
            frame_t_log.append(t_ms)
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

    # --- Audio phases 1-3 (passive: never steers the video path)
    audio_events, binding, onset_transcripts = [], None, []
    audio_ms_per_moment = None
    if audio_on and audio_feat_log:
        afe = np.concatenate(audio_feat_log, axis=0).astype(np.float32)
        if len(afe) != n_moments:
            print(f"NOTE: audio moments {len(afe)} vs video moments "
                  f"{n_moments} -- trimming/padding to the video grid",
                  flush=True)
            if len(afe) > n_moments:
                afe = afe[:n_moments]
            else:
                pad = np.zeros((n_moments - len(afe), 3), dtype=np.float32)
                afe = np.concatenate([afe, pad], axis=0)
        np.save(os.path.join(args.outdir, "audio_features.npy"), afe)
        print(f"wrote audio_features.npy ({len(afe)} moments x "
              f"[rms, flux, centroid])", flush=True)
        audio_ms_per_moment = (float(np.sum(audio_proc)) / max(n_moments, 1)
                               * 1000.0)
        # Phase 1: onset events on the master timeline
        for t_s, strength in pick_onsets(afe[:, 1], afe[:, 0]):
            mi = min(int(round(t_s / 0.05)), len(afe) - 1)
            audio_events.append({"t_s": round(t_s, 3),
                                 "strength": round(strength, 4),
                                 "rms": round(float(afe[mi, 0]), 4)})
        with open(os.path.join(args.outdir, "audio_events.json"), "w") as f:
            json.dump(audio_events, f, indent=2)
        print(f"wrote audio_events.json ({len(audio_events)} onsets)",
              flush=True)
        # Phase 2: cross-modal binding ("that made that")
        # driver.energies rows are (t, static_e, chroma_e, trans_e);
        # the transient signal is the total energy change.
        e_tot = [r[1] + r[2] + r[3] for r in driver.energies]
        vtrans = visual_transients(e_tot, frame_t_log)
        binding = bind_av([(o["t_s"], o["strength"]) for o in audio_events],
                          vtrans)
        with open(os.path.join(args.outdir, "av_binding.json"), "w") as f:
            json.dump(binding, f, indent=2)
        print(f"wrote av_binding.json ({binding['n_bound']}/"
              f"{binding['n_audio_onsets']} onsets bound, "
              f"{binding['n_visual_transients']} visual transients)",
              flush=True)
        # Phase 3: attended transcription -- spotlight, not floodlight
        if args.transcribe_onsets > 0 and audio_events:
            from hva.transcribe import transcribe as transcribe_wav
            top = sorted(audio_events, key=lambda o: -o["strength"]
                         )[:args.transcribe_onsets]
            for o in top:
                w0 = max(0.0, o["t_s"] - 4.0)
                wav = os.path.join(
                    args.outdir, f"onset_{o['t_s']:.1f}s.wav")
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-ss", str(w0),
                     "-i", args.video, "-t", "8", "-vn",
                     "-ar", "16000", "-ac", "1", wav], check=True)
                segs = transcribe_wav(wav, model_size="base", vad=False)
                onset_transcripts.append({
                    "t_s": o["t_s"], "strength": o["strength"],
                    "window_s": [round(w0, 1), round(w0 + 8.0, 1)],
                    "segments": [{"start": round(s["start"], 2),
                                  "end": round(s["end"], 2),
                                  "text": s["text"].strip(),
                                  "avg_logprob": round(
                                      float(s["avg_logprob"]), 3)}
                                 for s in segs]})
            with open(os.path.join(args.outdir, "onset_transcripts.json"),
                      "w") as f:
                json.dump(onset_transcripts, f, indent=2)
            print(f"wrote onset_transcripts.json "
                  f"({len(onset_transcripts)} windows)", flush=True)

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
        "audio": bool(audio_on),
        "n_audio_onsets": len(audio_events),
        "audio_ms_per_moment": (round(audio_ms_per_moment, 4)
                                if audio_ms_per_moment is not None else None),
        "n_av_bound": binding["n_bound"] if binding else 0,
        "transcribe_onsets": int(args.transcribe_onsets),
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
