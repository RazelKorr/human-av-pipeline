"""ChunkedFeeder: a test video played as if live.

Reads a video as if it were a live sensor and emits frames grouped
into wall-clock-paced chunks.

Decode modes (default: single decode via split filter graph):
- dual_decode=True: TWO ffmpeg pipes in lockstep -- the attention
  resolution (224x224 RGB) and the work resolution (WxH RGB). This
  mirrors the batch v2 driver's two decode processes exactly (same
  filter chains: fps={fps},scale=WxH,format=rgb24), so stream-vs-batch
  differences isolate the streaming machinery, not the decoder.
- dual_decode=False, single_mode="split" (default): ONE ffmpeg
  process decodes once and a split filter graph produces both
  resolutions (same per-branch filter chains as dual_decode). The
  224px frames are bit-identical to dual_decode; only the decode is
  deduplicated. Validated 2026-10-03: full 30 s streaming agreement
  vs dual_decode is bit-identical on every metric (see
  STREAMING-REPORT.md addendum). With motion_thumb_wh set, the graph
  gains a third output at that size for the motion channel, so no
  in-process thumbnail downsample is needed.
- dual_decode=False, single_mode="pil": ONE ffmpeg decode at work
  resolution; the attention thumbnail is downsampled in-process with
  PIL bilinear (uint8, the same convention as hvp.attention's own
  _downsample_color). Slightly faster, but the 224px frames differ
  slightly from dual_decode (~6% of saccades flip to adjacent
  saliency cells; measured 2026-10-03).

A real sensor would be single-resolution with in-process
downsampling; the single-decode modes simulate that while the
dual-decode mode keeps the original validation configuration
reproducible.

Pacing: realtime=True paces frame emission to the wall clock
(media_time / speed). realtime=False yields as fast as the consumer
pulls. Pacing never touches timestamps, so the math is identical
either way -- only the wall-clock timing changes. Deadline
accounting (did processing keep up?) is the runner's job; the
feeder only reports what it emitted.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_video_color import decode_color


@dataclass
class Chunk:
    """One paced group of frames."""
    t0_s: float          # media start (s)
    t1_s: float          # media end (s)
    frames: list         # [(t_ms, frame_attn_224, frame_work, frame_thumb)]
    # frame_thumb is the feeder-supplied motion thumbnail (HxWx3 RGB,
    # float32) when the split graph was asked for one, else None --
    # in which case the driver derives the thumbnail in-process.


def downsample_pil(frame, wh):
    """HxWx3 float32 RGB -> wh float32 RGB thumbnail (PIL bilinear).

    Same convention as hvp.attention._downsample_color: quantize to
    uint8, PIL BILINEAR, back to float32.
    """
    w, h = int(wh[0]), int(wh[1])
    u8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    small = Image.fromarray(u8).resize((w, h), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255.0


def decode_split(path, seconds, fps, w1, h1, w2, h2, w3=None, h3=None,
                 t0=0.0):
    """One decode, two or three scaled outputs via a split filter graph.

    Yields ((t_ms, H1xW1x3 frame), (t_ms, H2xW2x3 frame)) in lockstep,
    plus a third ((t_ms, H3xW3x3 frame)) when w3/h3 are given (used
    for the motion thumbnail: libswscale produces it during the
    decode, so no in-process downsample is needed downstream).
    Branch filter chains are the same fps/scale/format chain as
    decode_color, so each output matches its single-pipe equivalent
    bit-for-bit; only the decode is deduplicated.
    """
    import subprocess
    n = int(seconds * fps)
    three = w3 is not None and h3 is not None
    if three:
        vf = (f"fps={fps},split=3[full][small][mot];"
              f"[full]scale={w1}:{h1},format=rgb24[fo];"
              f"[small]scale={w2}:{h2},format=rgb24[so];"
              f"[mot]scale={w3}:{h3},format=rgb24[mo]")
    else:
        vf = (f"fps={fps},split=2[full][small];"
              f"[full]scale={w1}:{h1},format=rgb24[fo];"
              f"[small]scale={w2}:{h2},format=rgb24[so]")
    r_extra, w_extra = os.pipe()
    # -frames:v n on EVERY output: -t only binds the first output
    # file, so without this the pass_fds outputs run to input EOF
    # (found 2026-10-03: the writer then wedges in the fps filter
    # at the -t boundary -- alive, no output, never exiting -- and
    # teardown either spams EPIPE or hangs draining it). With the
    # per-output frame cap each pipe EOFs right after frame n and
    # ffmpeg exits 0; the short-read break below still covers
    # inputs shorter than `seconds`.
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", path,
           "-t", str(seconds), "-filter_complex", vf,
           "-map", "[fo]", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
           "-map", "[so]", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           f"pipe:{w_extra}"]
    pass_fds = [w_extra]
    r_mot = None
    if three:
        # Third output rides a second pass_fds pipe. Its frames are
        # tiny (thumb-sized, far under the 64 KiB pipe buffer), so it
        # can never block the writer; the two big outputs keep the
        # same relative read order as the validated two-pipe scheme.
        r_mot, w_mot = os.pipe()
        cmd += ["-map", "[mo]", "-frames:v", str(n),
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                f"pipe:{w_mot}"]
        pass_fds.append(w_mot)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            pass_fds=tuple(pass_fds))
    os.close(w_extra)  # parent keeps only the read ends
    if three:
        os.close(w_mot)
    f2 = os.fdopen(r_extra, "rb")
    f3 = os.fdopen(r_mot, "rb") if three else None
    b1, b2 = w1 * h1 * 3, w2 * h2 * 3
    b3 = w3 * h3 * 3 if three else 0
    base = t0 * 1000.0
    try:
        for i in range(n):
            # Strictly alternate reads in map order: each big frame
            # exceeds the pipe buffer, so reading one side dry would
            # deadlock ffmpeg.
            raw1 = proc.stdout.read(b1)
            raw2 = f2.read(b2)
            raw3 = f3.read(b3) if three else b""
            if (len(raw1) < b1 or len(raw2) < b2
                    or (three and len(raw3) < b3)):
                break
            t = base + i * 1000.0 / fps
            fr1 = (np.frombuffer(raw1, dtype=np.uint8)
                   .reshape(h1, w1, 3).astype(np.float32) / 255.0)
            fr2 = (np.frombuffer(raw2, dtype=np.uint8)
                   .reshape(h2, w2, 3).astype(np.float32) / 255.0)
            tm = None
            if three:
                fr3 = (np.frombuffer(raw3, dtype=np.uint8)
                       .reshape(h3, w3, 3).astype(np.float32) / 255.0)
                tm = (t, fr3)
            yield (t, fr1), (t, fr2), tm
    finally:
        # Every output is capped at n frames (-frames:v), so all
        # three pipes EOF right after the last frame and the writer
        # exits 0 -- no drain needed, no EPIPE spam.
        proc.stdout.close()
        f2.close()
        if f3 is not None:
            f3.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # never seen; belt and braces
            proc.kill()
            proc.wait()


def decode_audio_track(path, seconds, sr=16000, t0=0.0):
    """Decode the audio track once, whole, to mono float32 @sr.

    Pre-decoded per run (a few seconds for a feature film; ~8.6 MB
    for 269 s @16 kHz) and sliced downstream by sample count, so
    moment alignment is exact by construction -- no pipe interleave,
    no deadlock risk, zero disturbance to the validated video pipes.
    Raises RuntimeError if the source has no audio stream.
    """
    import subprocess
    n_expected = int(seconds * sr)
    cmd = ["ffmpeg", "-v", "error", "-ss", str(t0), "-i", path,
           "-t", str(seconds), "-vn", "-ar", str(sr), "-ac", "1",
           "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    raw = proc.stdout
    if proc.returncode != 0 or len(raw) < sr:  # <1 s of audio: no track
        raise RuntimeError(
            f"no usable audio track in {path} "
            f"(rc={proc.returncode}, {len(raw)} bytes)")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if abs(len(x) - n_expected) > sr:  # >1 s skew vs video: bug
        raise RuntimeError(
            f"audio/video length skew: {len(x)/sr:.2f}s audio vs "
            f"{seconds:.2f}s requested")
    if len(x) < n_expected:
        x = np.pad(x, (0, n_expected - len(x)))
    return x[:n_expected].astype(np.float32), int(sr)


class ChunkedFeeder:
    def __init__(self, video, seconds, fps, attn_wh=(224, 224),
                 work_wh=(640, 360), chunk_s=2.0, realtime=False,
                 speed=1.0, dual_decode=False, single_mode="split",
                 motion_thumb_wh=None):
        self.video = video
        self.seconds = float(seconds)
        self.fps = float(fps)
        self.attn_wh = tuple(attn_wh)
        self.work_wh = tuple(work_wh)
        self.chunk_s = float(chunk_s)
        self.realtime = bool(realtime)
        if float(speed) <= 0:
            raise ValueError(f"speed must be positive, got {speed!r}")
        self.speed = float(speed)
        self.dual_decode = bool(dual_decode)
        if single_mode not in ("pil", "split"):
            raise ValueError(f"single_mode must be 'pil' or 'split', "
                             f"got {single_mode!r}")
        # "split" is the default: one decode, and the 224px branch uses
        # the same libswscale chain as dual_decode, so its frames are
        # bit-identical to the dual-decode baseline. "pil" (in-process
        # PIL bilinear) is faster but its 224px frames differ slightly,
        # flipping ~6% of saccade decisions to adjacent saliency cells
        # (measured 2026-10-03, see STREAMING-REPORT.md).
        self.single_mode = single_mode
        # Optional third split-graph output: the motion thumbnail at
        # (w, h). Only honored in split mode; lets the motion channel
        # consume libswscale-produced thumbs straight from the feeder
        # instead of PIL-downsampling the work frame in-process.
        # None = no third output (driver derives the thumbnail itself).
        if motion_thumb_wh is not None:
            mw, mh = motion_thumb_wh
            if int(mw) <= 0 or int(mh) <= 0:
                raise ValueError("motion_thumb_wh must be positive WxH, "
                                 f"got {motion_thumb_wh!r}")
            self.motion_thumb_wh = (int(mw), int(mh))
        else:
            self.motion_thumb_wh = None
        self.n_frames_expected = int(self.seconds * self.fps)
        self.total_frames = 0

    def _frame_pairs(self):
        """Yield (t_ms, frame_attn, frame_work, frame_thumb) per decode
        mode. frame_thumb is None unless split mode was given a
        motion_thumb_wh."""
        aw, ah = self.attn_wh
        ww, wh = self.work_wh
        if self.dual_decode:
            gen_a = decode_color(self.video, self.seconds, self.fps, aw, ah)
            gen_w = decode_color(self.video, self.seconds, self.fps, ww, wh)
            for (t_a, fa), (t_w, fw) in zip(gen_a, gen_w):
                yield t_a, t_w, None, fa, fw, None
        elif self.single_mode == "split":
            mw, mh = self.motion_thumb_wh or (None, None)
            gen = decode_split(self.video, self.seconds, self.fps,
                               ww, wh, aw, ah, mw, mh)
            for (t_w, fw), (t_a, fa), tm in gen:
                t_m, fm = tm if tm is not None else (None, None)
                yield t_a, t_w, t_m, fa, fw, fm
        else:  # "pil": one decode at work res, in-process downsample
            gen_w = decode_color(self.video, self.seconds, self.fps, ww, wh)
            for t_w, fw in gen_w:
                yield t_w, t_w, None, downsample_pil(fw, (aw, ah)), fw, None

    def __iter__(self):
        start_wall = None
        chunk_frames: list = []
        chunk_t0 = 0.0
        n = 0

        for t_a, t_w, t_m, fa, fw, fm in self._frame_pairs():
            # Pipes (or branches) decode the same timeline; any skew
            # is a bug.
            assert abs(t_a - t_w) < 1e-6, \
                f"pipe skew: attn t={t_a} work t={t_w}"
            if t_m is not None:
                assert abs(t_a - t_m) < 1e-6, \
                    f"pipe skew: thumb t={t_m} attn t={t_a}"
            if start_wall is None:
                start_wall = time.monotonic()
            if self.realtime:
                target = start_wall + (t_a / 1000.0) / self.speed
                dt = target - time.monotonic()
                if dt > 0:
                    time.sleep(dt)
            t_s = t_a / 1000.0
            if chunk_frames and t_s - chunk_t0 >= self.chunk_s:
                yield Chunk(t0_s=chunk_t0, t1_s=t_s, frames=chunk_frames)
                chunk_frames = []
                chunk_t0 = t_s
            chunk_frames.append((t_a, fa, fw, fm))
            n += 1
        if chunk_frames:
            # Stream time ends at the frame grid, not at `seconds`:
            # decode yields n=int(seconds*fps) frames starting at t=0.
            yield Chunk(t0_s=chunk_t0, t1_s=n / self.fps,
                        frames=chunk_frames)
        self.total_frames = n
