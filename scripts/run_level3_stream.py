"""Level 3 on a live stream: the joint priority map watches streaming media.

Instead of precomputing every moment's input (batch mode), this script
pulls 100 ms ticks from a StreamSource -- a file played as if live, or
any ffmpeg-readable URL (RTMP/HLS) -- and steps three OnlineLevel3 loops
(joint, vision-only, speech-gated) tick by tick, the way a real-time
perceptual system would.

The sensory front-ends are the online versions of the batch code:
  vision: incremental salience (prev frame + decaying transient)
  audio:  10 s chunks with 1 s overlap running the exact batch
          STFT/transient/ILD DSP; central 10 s emitted
  speech: windowed faster-whisper (30 s window, 10 s step), stitched;
          the gate uses only already-transcribed words (honest lag)

Usage:
  python3 scripts/run_level3_stream.py --src input/foo.mp4 --seconds 62
  python3 scripts/run_level3_stream.py --src rtmp://host/live/key --realtime
  python3 scripts/run_level3_stream.py --src input/foo.mp4 --seconds 62 \\
      --verify output/level3/star_tours_full_v4/joint_maps.npz \\
      --inject-ceiling 12.34

--verify compares the streamed joint/visonly peaks against a batch run's
joint_maps.npz and reports the max deviation. --inject-ceiling supplies
the batch run's percentile-99 Tprof ceiling (printed by the batch
script... currently it isn't; compute from the npz's Tprof) so the
comparison isolates the streaming machinery from the online ceiling
estimator.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from hvp import baseline as B
from hvm.online import OnlineLevel3
from hvm.priority import JointPriorityMap
from hva import attention as AT
from hva import moments as M
from hva import salience as S
from hva.stream import (StreamSource, VisionFrontEnd, AudioFrontEnd,
                        RollingTranscriber)


DEFAULT_SRC = os.path.join(ROOT, "input", "star_tours_1_ride_film.mp4")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC,
                    help="file path or ffmpeg-readable stream URL")
    ap.add_argument("--seconds", type=float, default=62.0)
    ap.add_argument("--realtime", action="store_true",
                    help="pace ticks to the wall clock (default: as fast "
                         "as possible)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="realtime speed multiplier")
    ap.add_argument("--transcribe", action="store_true",
                    help="enable the rolling transcriber + speech gate")
    ap.add_argument("--model-size", default="base",
                    help="whisper model for the live path (base keeps up "
                         "on CPU; medium lags)")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "output",
                                                     "level3_stream"))
    ap.add_argument("--verify", default=None,
                    help="batch joint_maps.npz to compare peaks against")
    ap.add_argument("--inject-ceiling", type=float, default=None,
                    help="batch percentile-99 Tprof ceiling for exact "
                         "verification (else the online estimator runs)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    source = StreamSource(args.src, realtime=args.realtime,
                          speed=args.speed, duration=args.seconds)
    vision_fe = VisionFrontEnd()
    audio_fe = AudioFrontEnd(ceiling=args.inject_ceiling)
    tx = (RollingTranscriber(model_size=args.model_size,
                             prompt=args.prompt)
          if args.transcribe else None)

    dva = B.FIELD_WIDTH_DEG / 224.0
    # Match the batch convention: saccades landing past t_end are dropped.
    # Without this the stream keeps one extra saccade (206 vs 207).
    t_end = args.seconds * 1000.0 if args.seconds else float("inf")
    joint = OnlineLevel3(dva, t_end_ms=t_end)
    visonly = OnlineLevel3(dva, t_end_ms=t_end)
    gated = OnlineLevel3(dva, t_end_ms=t_end)

    vis_queue: list = []       # vision outputs waiting for their audio twin
    Tprof_list, pan_list = [], []
    n_mom = 0
    for tick in source:
        vis_queue.append(vision_fe.push(tick.frame))
        for am in audio_fe.push(tick):
            t_ms = n_mom * 100.0
            vis = vis_queue.pop(0)
            aud = (am["Tprof_n"], am["pan"])
            if tx is not None:
                tx.push(t_ms, am["mono"])   # moment order: no future words
                sp = tx.speech_tick(t_ms)
            else:
                sp = 0.0
            joint.tick(t_ms, vis, aud, sp)
            visonly.tick(t_ms, vis, None, 0.0)
            gated.tick(t_ms, vis, aud, sp)
            Tprof_list.append(am["Tprof"])
            pan_list.append(am["pan"])
            n_mom += 1
    # flush trailing audio (< 1 chunk of overlap may remain)
    for am in audio_fe.flush():
        t_ms = n_mom * 100.0
        vis = vis_queue.pop(0) if vis_queue else None
        aud = (am["Tprof_n"], am["pan"])
        if tx is not None:
            tx.push(t_ms, am["mono"])
            sp = tx.speech_tick(t_ms)
        else:
            sp = 0.0
        joint.tick(t_ms, vis, aud, sp)
        visonly.tick(t_ms, vis, None, 0.0)
        gated.tick(t_ms, vis, aud, sp)
        Tprof_list.append(am["Tprof"])
        pan_list.append(am["pan"])
        n_mom += 1
    if vis_queue:
        print(f"WARNING: {len(vis_queue)} vision outputs unpaired "
              f"(audio shorter than video?)", flush=True)

    Tprof = np.stack(Tprof_list)
    pan_bin = np.stack(pan_list)
    print(f"streamed {n_mom} moments ({n_mom / 10.0:.1f}s)", flush=True)
    if tx is not None:
        print(f"  transcriber: {tx.n_transcriptions} windows, "
              f"{len(tx.transcript())} stitched segments", flush=True)

    jr = joint.result()
    vr = visonly.result()
    gr = gated.result()
    jp = np.array([p[:2] for p in jr["peaks"]])
    vp = np.array([p[:2] for p in vr["peaks"]])
    disp = np.hypot(jp[:, 0] - vp[:, 0], jp[:, 1] - vp[:, 1]) * (224.0 / 56)
    moved = disp > 8.0
    print(f"  joint saccades: {len(jr['scanpath']) - 1}; "
          f"vision-only saccades: {len(vr['scanpath']) - 1}", flush=True)
    print(f"  audio moved the peak >8px in {moved.sum()} moments "
          f"({100 * moved.mean():.1f}%)", flush=True)
    print(f"  mean displacement: {disp.mean():.1f}px; "
          f"max {disp.max():.1f}px", flush=True)
    if tx is not None:
        gp = np.array([p[:2] for p in gr["peaks"]])
        gdisp = (np.hypot(gp[:, 0] - jp[:, 0], gp[:, 1] - jp[:, 1])
                 * (224.0 / 56))
        print(f"  speech gating moved the peak >8px in {(gdisp > 8).sum()} "
              f"moments ({100 * (gdisp > 8).mean():.1f}%)", flush=True)

    # ---- read path: map-gained auditory attention over banked moments --
    # (same construction as the batch script; duplicated, not shared --
    # the batch path is verified and stays untouched)
    Sg = audio_fe.spectrogram()
    n_use = min(len(jr["maps"]), Sg.shape[0] // 20)
    Sg = Sg[:n_use * 20]
    times = np.arange(Sg.shape[0]) * 0.005
    feat = S.salience_map(Sg)
    moms = M.moments(Sg, feat["sal"], times)
    moms = moms[:n_use]
    Tprof_u = Tprof[:n_use]
    pan_u = pan_bin[:n_use]
    probe = JointPriorityMap()
    gains = np.ones_like(Tprof_u)
    for mm in range(n_use):
        probe.map = gr["maps"][mm]
        gains[mm] = probe.gains_for_pans(pan_u[mm])
    moms_gain = []
    for k, mm_ in enumerate(moms):
        g = dict(mm_)
        g["sal"] = (mm_["sal"] * gains[k]).astype(np.float32)
        g["Tmax"] = float((Tprof_u[k] * gains[k]).max())
        moms_gain.append(g)
    att = AT.AuditoryAttention(cf_init=32.0)
    tr_base = att.run(moms, Tprof_u)
    att2 = AT.AuditoryAttention(cf_init=32.0)
    tr_gain = att2.run(moms_gain, Tprof_u * gains)
    cap = lambda tr: sum(1 for e in tr if e["event"] == "onset-capture")
    print(f"  audio captures: baseline {cap(tr_base)}, "
          f"map-gained {cap(tr_gain)}", flush=True)

    np.savez(os.path.join(args.outdir, "stream_maps.npz"),
             maps_joint=np.array(jr["maps"]),
             maps_visonly=np.array(vr["maps"]),
             peaks_joint=np.array(jr["peaks"]),
             peaks_visonly=np.array(vr["peaks"]),
             peaks_gated=np.array(gr["peaks"]),
             disp=disp, Tprof=Tprof, pan_bin=pan_bin)
    if tx is not None:
        import json as _json
        with open(os.path.join(args.outdir,
                               "stream_transcript.json"), "w") as f:
            _json.dump(tx.transcript(), f, indent=1)
    print(f"  saved {args.outdir}/", flush=True)

    if args.verify:
        z = np.load(args.verify)
        bj = np.array(z["peaks_v"])[:, :2]      # batch joint peaks
        bv = np.array(z["peaks_visonly"])[:, :2]
        n = min(len(bj), len(jp))
        dj = np.abs(jp[:n] - bj[:n]).max()
        dv = np.abs(vp[:n] - bv[:n]).max()
        print()
        print(f"verify vs {args.verify} ({n} moments):")
        print(f"  max |stream - batch| joint peak:  {dj:.4f} map-px")
        print(f"  max |stream - batch| visonly peak: {dv:.4f} map-px")
        # displacement-stat comparison
        bdisp = z["disp"][:n]
        print(f"  batch : moved>8px {(bdisp > 8).sum()} "
              f"({100 * (bdisp > 8).mean():.1f}%), mean {bdisp.mean():.1f}px")
        print(f"  stream: moved>8px {(disp[:n] > 8).sum()} "
              f"({100 * (disp[:n] > 8).mean():.1f}%), "
              f"mean {disp[:n].mean():.1f}px")


if __name__ == "__main__":
    main()
