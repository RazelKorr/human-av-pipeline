"""Streaming input for the Human A/V pipeline.

Batch mode precomputes everything, then loops. A stream -- a file played
as if live, an RTMP/HLS URL, eventually a microphone -- only moves
forward. This module is the online sensory front-end:

  StreamSource       file or ffmpeg-readable URL -> 10 Hz Tick stream.
                     Video is decoded by an ffmpeg rawvideo pipe running
                     the SAME filter chain as the batch decoder
                     (fps=10,scale=224:224,format=gray) -- bit-exact with
                     decode_gray by construction. (The batch decode_gray
                     bug of 2026-09-30 -- reading native-fps frames while
                     labeling them 10 fps -- is NOT repeated here; and a
                     2026-09-30 PyAV reformat path that differed by ~2 LSB
                     from ffmpeg's scaler was replaced for the same
                     reason.) Audio is demuxed via PyAV and binned by
                     sample count; moments pair video frame m with audio
                     [m*1600,(m+1)*1600).
  VisionFrontEnd     incremental retinotopic salience; the exact batch
                     math (prev frame + decaying transient channel),
                     just stateful instead of a precomputed array.
  AudioFrontEnd      chunked DSP: 10 s chunks with 1 s overlap run the
                     exact batch STFT/transient/ILD code; only the central
                     10 s of moments are emitted, so chunk edges never
                     touch the output. The Tprof ceiling is a running
                     percentile-99 over history (batch uses the whole run;
                     inject the batch ceiling for exact verification).
  RollingTranscriber windowed faster-whisper (30 s window, 10 s step),
                     stitched by start time; speech presence per tick uses
                     only already-transcribed words -- the gate honestly
                     trails reality by the transcription lag.

What streaming does NOT do yet: Demucs vocal separation is non-causal
(it needs the whole file), so the live path transcribes the raw mix.
Perception lags speech by ~10-30 s here; that is the documented cost of
causality, not a bug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from hvp import attention as A
from hva import cochlea as C
from hva import salience as S
from hva import transcribe as TR

MOMENT_MS = 100.0
AUDIO_SR = 16000
AUDIO_PER_MOMENT = 1600          # 16kHz * 0.1 s
FRAME_W = FRAME_H = 224


@dataclass
class Tick:
    t_ms: float
    frame: np.ndarray | None      # 224x224 float32 gray in 0..1, or None
    mono: np.ndarray              # (1600,) float32 16kHz mono
    stereo: np.ndarray            # (1600, 2) float32 16kHz stereo


class StreamSource:
    """Yield Tick per 100 ms from a file or stream URL.

    Video comes from an ffmpeg rawvideo pipe using the SAME filter chain
    as the batch decoder (fps=10,scale=224:224,format=gray) -- bit-exact
    with decode_gray by construction (2026-09-30: the PyAV reformat path
    differed by ~2 LSB from ffmpeg's scaler, which the salience
    normalization amplified on dark frames). Audio is demuxed via PyAV
    and binned by sample count. Moments are paired by index (frame m and
    audio [m*1600,(m+1)*1600) both start at t=0). realtime=True paces
    ticks to the wall clock (times speed).
    """

    def __init__(self, src: str, realtime: bool = False, speed: float = 1.0,
                 duration: float | None = None):
        self.src = src
        self.realtime = realtime
        self.speed = speed
        self.duration = duration

    def __iter__(self):
        import av
        import subprocess
        # ---- video: ffmpeg pipe, same filters as batch decode_gray ----
        vcmd = ["ffmpeg", "-v", "error", "-i", self.src]
        if self.duration:
            vcmd += ["-t", str(self.duration)]
        vcmd += ["-vf", f"fps=10,scale={FRAME_W}:{FRAME_H},format=gray",
                 "-f", "rawvideo", "pipe:1"]
        vproc = subprocess.Popen(vcmd, stdout=subprocess.PIPE)
        vframe_bytes = FRAME_W * FRAME_H
        vframes: list = []      # uint8 frames decoded so far
        veof = False

        def video_frame(mm):
            """Return uint8 frame mm, reading the pipe as needed.
            None only after EOF."""
            nonlocal veof
            while not veof and len(vframes) <= mm:
                raw = vproc.stdout.read(vframe_bytes)
                if len(raw) < vframe_bytes:
                    veof = True
                    vproc.wait()
                    break
                vframes.append(
                    np.frombuffer(raw, dtype=np.uint8)
                    .reshape(FRAME_H, FRAME_W))
            return vframes[mm] if mm < len(vframes) else None

        # ---- audio: PyAV demux + resample ----
        container = av.open(self.src)
        astream = next((s for s in container.streams
                        if s.type == "audio"), None)
        resampler = None
        if astream:
            resampler = av.AudioResampler(format="s16", layout="stereo",
                                          rate=AUDIO_SR)

        abuf: list[np.ndarray] = []  # (n, 2) float32 stereo chunks
        abuf_start = 0               # absolute sample index of abuf[0][0]
        n_audio = 0                  # total samples seen
        m = 0
        start_wall = time.monotonic()
        max_m = int(self.duration * 10) if self.duration else None

        def audio_window(mm):
            """Samples [mm*1600, (mm+1)*1600) as (1600, 2) float32."""
            nonlocal abuf_start
            s0, s1 = mm * AUDIO_PER_MOMENT, (mm + 1) * AUDIO_PER_MOMENT
            while abuf and abuf_start + len(abuf[0]) <= s0:
                abuf_start += len(abuf[0])
                abuf.pop(0)
            parts, pos = [], abuf_start
            for ch in abuf:
                if pos >= s1:
                    break
                if pos + len(ch) > s0:
                    a = max(0, s0 - pos)
                    b = min(len(ch), s1 - pos)
                    parts.append(ch[a:b])
                pos += len(ch)
            w = (np.concatenate(parts, axis=0) if parts
                 else np.zeros((0, 2), np.float32))
            if len(w) < AUDIO_PER_MOMENT:
                w = np.concatenate(
                    [w, np.zeros((AUDIO_PER_MOMENT - len(w), 2),
                                 np.float32)])
            return w

        def pace(mm):
            if self.realtime:
                target = start_wall + (mm + 1) * 0.1 / self.speed
                dt = target - time.monotonic()
                if dt > 0:
                    time.sleep(dt)

        def emit(mm):
            vf = video_frame(mm)
            w = audio_window(mm)
            tick = Tick(mm * MOMENT_MS,
                        (vf.astype(np.float32) / 255.0
                         if vf is not None else None),
                        w.mean(axis=1), w)
            pace(mm)
            return tick

        for packet in container.demux():
            if astream is not None and packet.stream is astream:
                for frame in packet.decode():
                    for rf in resampler.resample(frame):
                        # packed s16 stereo: interleaved LRLR... in plane 0.
                        # NOTE: the plane buffer is padded -- only the
                        # first rf.samples are valid (2026-09-30: reading
                        # the whole buffer produced garbage audio).
                        raw = np.frombuffer(
                            bytes(rf.planes[0]),
                            dtype=np.int16)[:rf.samples * 2]
                        s = (raw.reshape(-1, 2).astype(np.float32)
                             / 32768.0)
                        abuf.append(s)
                        n_audio += s.shape[0]
            # Emit every moment with complete audio. Video frames are read
            # on demand (for files ffmpeg runs ahead; for live sources the
            # read blocks -- a stalled video stalls emission, documented).
            # A video EOF before audio EOF ends the stream: the batch also
            # stops at min(video, audio) moments.
            while (n_audio >= (m + 1) * AUDIO_PER_MOMENT
                   and (max_m is None or m < max_m)):
                if veof and m >= len(vframes):
                    break
                yield emit(m)
                m += 1
            if max_m is not None and m >= max_m:
                break
            if veof and m >= len(vframes):
                break
        # EOF: flush remaining audio while video frames last.
        if max_m is None:
            total_m = n_audio // AUDIO_PER_MOMENT
        else:
            total_m = min(n_audio // AUDIO_PER_MOMENT, max_m)
        while m < total_m:
            if veof and m >= len(vframes):
                break
            yield emit(m)
            m += 1
        container.close()
        if not veof:
            vproc.terminate()


class VisionFrontEnd:
    """Incremental version of the batch vision salience computation in
    scripts/run_level3.py. Same math, stateful: prev frame and the
    decaying transient channel persist across push() calls."""

    def __init__(self):
        self.prev = None
        self.trans = np.zeros_like(
            A._downsample(np.zeros((224, 224), np.float32)))
        self.decay = np.exp(-100.0 / A.TRANS_TAU_MS)

    def push(self, frame224: np.ndarray | None):
        """frame224: 224x224 float32 in 0..1, or None (blind moment).
        Returns (56,56) salience, or None."""
        if frame224 is None:
            return None
        small = A._downsample(frame224)
        if self.prev is not None:
            self.trans = np.maximum(self.trans * self.decay,
                                    np.abs(small - self.prev))
        self.prev = small
        static = A._norm(np.abs(small - gaussian_filter(small, 6)))
        return A._norm(static + 1.5 * A._norm(self.trans))


def _tprof(Sg20):
    """Per-bin transient profile for one 20-STFT-frame (100 ms) slice.
    Exact copy of the batch helper in scripts/run_level3.py."""
    sl = Sg20
    if len(sl) < 2:
        return np.zeros(C.N_BINS, dtype=np.float32)
    d = np.abs(np.diff(sl, axis=0))
    d[np.maximum(sl[:-1], sl[1:]) < S.HEAR_FLOOR_DBFS] = 0.0
    return d.mean(axis=0).astype(np.float32)


def _ild_pans(S_l, S_r, n_mom):
    """Per-bin pan from stereo ILD, with the NaN/empty-slice guards from
    the 2026-09-30 batch fix. Exact copy of the batch logic."""
    pan_bin = np.zeros((n_mom, C.N_BINS), dtype=np.float32)
    for m in range(n_mom):
        sl = slice(m * 20, (m + 1) * 20)
        seg_l, seg_r = S_l[sl], S_r[sl]
        if len(seg_l) < 2 or len(seg_r) < 2:
            continue  # stays centered
        ild = (seg_l - seg_r).mean(axis=0)  # dB; + = left louder
        ild = np.nan_to_num(ild, nan=0.0, posinf=0.0, neginf=0.0)
        pan_bin[m] = -np.clip(ild / 12.0, -1.0, 1.0)  # -1 = left
    return pan_bin


class AudioFrontEnd:
    """Chunked online version of the batch audio DSP.

    10 s chunks with 1 s overlap; only the central 10 s of moments are
    emitted (the first chunk emits from 0, since the batch STFT's
    boundary="zeros" first frame matches). Chunk boundaries sit on exact
    10 s marks -- multiples of the 5 ms STFT hop -- so the frame grid
    aligns with the batch full-clip STFT bit-for-bit on interior frames.

    The Tprof ceiling: batch uses percentile-99 over the whole run.
    Online that doesn't exist yet, so a running percentile-99 over all
    emitted history is used (warmup: whatever history exists). Pass
    ceiling=<float> to inject the batch value for exact verification.
    """

    CHUNK_S = 10.0
    OVERLAP_S = 1.0

    def __init__(self, ceiling: float | None = None):
        self.mono_hist: list[np.ndarray] = []
        self.stereo_hist: list[np.ndarray] = []
        self.n_hist = 0
        self.consumed = 0          # absolute samples consumed
        self.fixed_ceiling = ceiling
        self.tprof_hist: list[np.ndarray] = []
        self.pending: list[dict] = []
        self.sg_pieces: list[np.ndarray] = []   # interior STFT (read path)

    def push(self, tick: Tick):
        self.mono_hist.append(tick.mono)
        self.stereo_hist.append(tick.stereo)
        self.n_hist += len(tick.mono)
        need = int((self.CHUNK_S + self.OVERLAP_S) * AUDIO_SR)
        while self.n_hist - self.consumed >= need:
            self._process_chunk()
        return self.drain()

    def _dsp(self, mono, stereo, n_mom):
        t, f, Sg = C.stft_log(mono)
        _, _, S_l = C.stft_log(stereo[:, 0])
        _, _, S_r = C.stft_log(stereo[:, 1])
        Tprof = np.stack([_tprof(Sg[i * 20:(i + 1) * 20])
                          for i in range(n_mom)])
        pan_bin = _ild_pans(S_l, S_r, n_mom)
        self.tprof_hist.append(Tprof)
        if self.fixed_ceiling is not None:
            ceil = self.fixed_ceiling
        else:
            ceil = float(np.percentile(
                np.concatenate(self.tprof_hist, axis=0), 99))
        Tprof_n = np.clip(Tprof / max(ceil, 1e-9), 0, 1)
        return Sg, Tprof, Tprof_n, pan_bin

    def _process_chunk(self):
        full_m = np.concatenate(self.mono_hist)
        full_s = np.concatenate(self.stereo_hist)
        n = int((self.CHUNK_S + self.OVERLAP_S) * AUDIO_SR)
        mono = full_m[self.consumed:self.consumed + n]
        stereo = full_s[self.consumed:self.consumed + n]
        n_dsp = int((self.CHUNK_S + self.OVERLAP_S) * 10)  # 110 local moms
        n_emit = int(self.CHUNK_S * 10)                     # 100 central moms
        Sg, Tprof, Tprof_n, pan_bin = self._dsp(mono, stereo, n_dsp)
        ov = int(self.OVERLAP_S * 10 // 2)      # 5 moments each side
        first = 0 if self.consumed == 0 else ov
        # NOTE (2026-09-30 bugfix): the emit window is [first, n_emit+ov):
        # the DSP ran on CHUNK+OVERLAP seconds, and only one overlap flank
        # is discarded per side. Writing n_emit-ov here silently dropped
        # 10 moments per chunk.
        for i in range(first, n_emit + ov):
            self.pending.append({"Tprof_n": Tprof_n[i], "pan": pan_bin[i],
                                 "Tprof": Tprof[i],
                                 "mono": mono[i * AUDIO_PER_MOMENT:
                                              (i + 1) * AUDIO_PER_MOMENT]})
        self.sg_pieces.append(Sg[first * 20:(n_emit + ov) * 20])
        self.consumed += int(self.CHUNK_S * AUDIO_SR)

    def drain(self):
        out, self.pending = self.pending, []
        return out

    def flush(self):
        """End of stream: emit moments for remaining audio.

        The first 0.5 s were already emitted as the last chunk's trailing
        overlap; they serve as left context and are skipped."""
        out = self.drain()
        rest = self.n_hist - self.consumed
        if rest >= 2 * AUDIO_SR:
            full_m = np.concatenate(self.mono_hist)
            full_s = np.concatenate(self.stereo_hist)
            n_mom = rest // AUDIO_PER_MOMENT
            mono = full_m[self.consumed:self.consumed
                          + n_mom * AUDIO_PER_MOMENT]
            stereo = full_s[self.consumed:self.consumed
                            + n_mom * AUDIO_PER_MOMENT]
            Sg, Tprof, Tprof_n, pan_bin = self._dsp(mono, stereo, n_mom)
            # Skip the leading overlap only if a chunk already emitted it;
            # for streams shorter than one chunk, emit from moment 0.
            ov = int(self.OVERLAP_S * 10 // 2) if self.consumed else 0
            for i in range(ov, n_mom):
                out.append({"Tprof_n": Tprof_n[i], "pan": pan_bin[i],
                            "Tprof": Tprof[i],
                            "mono": mono[i * AUDIO_PER_MOMENT:
                                         (i + 1) * AUDIO_PER_MOMENT]})
            self.sg_pieces.append(Sg[ov * 20:n_mom * 20])
        self.mono_hist, self.stereo_hist = [], []
        self.n_hist = self.consumed = 0
        return out

    def spectrogram(self):
        """Banked interior STFT pieces concatenated (for the read path)."""
        if not self.sg_pieces:
            return np.zeros((0, C.N_BINS), np.float32)
        return np.concatenate(self.sg_pieces, axis=0)


class RollingTranscriber:
    """Windowed online transcription: every step_s of stream time,
    transcribe the last window_s of mono audio and stitch segments by
    start time (later windows overwrite earlier ones on overlap).

    speech_tick(t_ms) returns smoothed speech presence for that moment
    using only words transcribed SO FAR -- the gate trails reality by
    the window/step lag, honestly. Same hangover math as
    TR.speech_presence (300 ms exponential release), computed
    incrementally."""

    def __init__(self, model_size: str = "base", window_s: float = 30.0,
                 step_s: float = 10.0, prompt: str | None = None):
        self.model_size = model_size
        self.window_s = window_s
        self.step_s = step_s
        self.prompt = prompt
        self.buf: list[np.ndarray] = []
        self.n_buf = 0
        self.next_tx = step_s
        self.segments: dict[float, dict] = {}
        self.words: list[tuple[float, float]] = []   # (start, end)
        self._carry = 0.0
        self._decay = float(np.exp(-100.0 / 300.0))
        self.n_transcriptions = 0

    def push(self, t_ms: float, mono: np.ndarray):
        self.buf.append(mono)
        self.n_buf += len(mono)
        maxn = int(self.window_s * AUDIO_SR)
        if self.n_buf > maxn:      # cap the reservoir at the window
            full = np.concatenate(self.buf)
            self.buf = [full[self.n_buf - maxn:]]
            self.n_buf = maxn
        t_s = t_ms / 1000.0
        if t_s >= self.next_tx:
            self._transcribe(t_s)
            while self.next_tx <= t_s:   # catch up after a slow window
                self.next_tx += self.step_s

    def _transcribe(self, t_s: float):
        audio = np.concatenate(self.buf)
        t0 = t_s - len(audio) / AUDIO_SR   # reservoir start in stream time
        segs = TR.transcribe_audio(audio, model_size=self.model_size,
                                   prompt=self.prompt)
        for s in segs:
            s2 = dict(s)
            s2["start"] = s["start"] + t0
            s2["end"] = s["end"] + t0
            s2["words"] = [dict(w, start=w["start"] + t0,
                                end=w["end"] + t0)
                           for w in s["words"]]
            self.segments[round(s2["start"], 3)] = s2
        self.words = sorted(
            (w["start"], w["end"])
            for s in self.segments.values() for w in s["words"])
        self.n_transcriptions += 1

    def speech_tick(self, t_ms: float) -> float:
        """Smoothed speech presence for the moment starting at t_ms."""
        t = t_ms / 1000.0
        m0, m1 = t, t + 0.1
        b = 1.0 if any(ws < m1 and we > m0 for ws, we in self.words) \
            else 0.0
        c = self._carry
        self._carry = b if b > c * self._decay else c * self._decay
        return float(self._carry)

    def transcript(self):
        return [self.segments[k] for k in sorted(self.segments)]
