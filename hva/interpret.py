"""Audio interpretation: the ear's "what" pathway.

Everything here runs OUTSIDE the 50 ms hot loop (hva.online's
moment_features is untouched -- the invariant). Inputs are onset
windows: seconds of 16 kHz mono audio cut around detected onsets.
Three jobs:

1. label_window: sound-event labels per window (CLAP zero-shot).
   Answers "what was that?" for onsets transcription can't touch --
   explosions, laughter, whooshes, music beds.
2. transcribe_two_tier: whisper base first pass with VAD-derived
   segment boundaries; segments below the sealed confidence
   threshold escalate to whisper medium.
3. music_features: chroma + tempo/downbeat for windows the labeler
   flags as music.

Standing conventions:
- Every claim carries its score. A label at 0.51 is reported as
  0.51, not as fact.
- Transcription claims carry logprobs or don't ship (unchanged).
- Empty/silent windows return empty results, never exceptions.
- Missing optional dependencies raise ImportError naming the pip
  package, never a bare ModuleNotFoundError from deep in a call.
"""

from __future__ import annotations

import numpy as np

# laion-clap import-time code tries to fetch a BERT tokenizer from
# HuggingFace, whose download client cannot parse the egress proxy
# config (httpx InvalidURL). All weights are local (models/), so
# forbid network access outright -- a download attempt here is always
# a bug, never a need.
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SR = 16000          # analysis sample rate, matches hva.cochlea
CLAP_SR = 48000     # laion-clap's native rate

# Sealed 2026-10-03 (see output/st_av_interpret/sealed-predictions.md):
# below this avg_logprob, a base-tier segment escalates to medium.
LOW_CONF_THRESHOLD = -0.8

# Softmax temperature for label scores. Fixed before any measurement;
# not tuned to outcomes. Lower = more decisive; 0.1 means a label must
# clearly beat the runner-up to clear 0.5.
LABEL_TEMPERATURE = 0.1

# Fixed zero-shot vocabulary (documented; change it and old scores
# are not comparable). Prompt template: "the sound of {label}".
LABEL_VOCABULARY = [
    "speech",
    "shouting",
    "laughter",
    "applause",
    "cheering",
    "explosion",
    "gunfire",
    "whoosh",
    "rumble",
    "thud",
    "music",
    "singing",
    "siren",
    "alarm",
    "engine",
    "door slam",
    "footsteps",
    "glass breaking",
    "wind",
    "silence",
]

MUSIC_LABELS = {"music", "singing"}


def _sanitize(audio: np.ndarray) -> np.ndarray:
    """float32 mono in -1..1, non-finite values zeroed.

    The motion channel's NaN-poisoning fix (2026-10-03 audit) applies
    here too: one bad sample must not nuke a window's labels.
    """
    a = np.asarray(audio, dtype=np.float32).ravel()
    if not np.all(np.isfinite(a)):
        a = np.where(np.isfinite(a), a, 0.0).astype(np.float32)
    return np.clip(a, -1.0, 1.0)


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    n_out = int(round(len(audio) * sr_out / sr_in))
    t_old = np.arange(len(audio), dtype=np.float64) / sr_in
    t_new = np.arange(n_out, dtype=np.float64) / sr_out
    return np.interp(t_new, t_old, audio).astype(np.float32)


_LABELER = None
_LABEL_TEXT_EMB = None


def _require_clap():
    try:
        from laion_clap import CLAP_Module  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "audio labeling needs laion-clap in the venv "
            "(pip install laion-clap)") from e


def get_labeler():
    """Lazy singleton CLAP module + precomputed text embeddings.

    Weights live at models/laion-clap-630k-audioset-best.pt (gitignored;
    too large to commit) because the HF download client can't parse
    the egress proxy config -- same reason hva.transcribe keeps local
    model dirs. Raises ImportError with the pip package name when
    laion-clap is missing, FileNotFoundError naming the expected
    checkpoint path when the weights aren't downloaded yet.
    """
    global _LABELER, _LABEL_TEXT_EMB
    _require_clap()
    if _LABELER is None:
        import torch
        from laion_clap import CLAP_Module
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "models",
                            "laion-clap-630k-audioset-best.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"CLAP weights not found at {ckpt}; download "
                "'https://huggingface.co/lukewys/laion_clap/resolve/main/"
                "630k-audioset-best.pt' there (curl with the egress "
                "proxy CA if needed)")
        _LABELER = CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny")  # 768-dim: matches 630k-audioset-best.pt
        _LABELER.load_ckpt(ckpt=ckpt)
        prompts = [f"the sound of {label}" for label in LABEL_VOCABULARY]
        with torch.no_grad():
            _LABEL_TEXT_EMB = _LABELER.get_text_embedding(prompts,
                                                          use_tensor=True)
    return _LABELER, _LABEL_TEXT_EMB


def label_window(audio: np.ndarray, sr: int = SR,
                 top_k: int = 5) -> list[dict]:
    """Zero-shot sound-event labels for one onset window.

    Returns [{label, score}] sorted by descending score, where score
    is the softmax probability over LABEL_VOCABULARY at
    LABEL_TEMPERATURE. Empty audio -> []. Near-silence still gets
    labels (usually "silence" on top); the scores say how much to
    trust them.
    """
    audio = _sanitize(audio)
    if len(audio) == 0:
        return []
    model, text_emb = get_labeler()
    import torch
    wav48 = _resample(audio, sr, CLAP_SR)
    # CLAP wants >= ~1 s; pad short windows with zeros (documented:
    # padding dilutes, it doesn't invent).
    if len(wav48) < CLAP_SR:
        wav48 = np.pad(wav48, (0, CLAP_SR - len(wav48)))
    # laion-clap randomly truncates inputs over 10 s (rand_trunc) --
    # non-deterministic. Center-truncate deterministically first so
    # the same window always yields the same labels.
    max_len = 10 * CLAP_SR
    if len(wav48) > max_len:
        start = (len(wav48) - max_len) // 2
        wav48 = wav48[start:start + max_len]
    with torch.no_grad():
        # Default (use_tensor=False) path: accepts numpy, applies the
        # training-time int16 quantization before feature extraction.
        a_emb = model.get_audio_embedding_from_data(wav48[np.newaxis, :])
        a_emb = torch.from_numpy(np.asarray(a_emb)).float()
        sims = (a_emb @ text_emb.T).squeeze(0).float()
        probs = torch.softmax(sims / LABEL_TEMPERATURE, dim=0)
    order = torch.argsort(probs, descending=True)[:top_k]
    return [{"label": LABEL_VOCABULARY[i],
             "score": round(float(probs[i]), 4)}
            for i in order.tolist()]


def is_music(labels: list[dict], threshold: float = 0.3) -> bool:
    """True when the label set says this window is music."""
    return any(l["label"] in MUSIC_LABELS and l["score"] >= threshold
               for l in labels)


def transcribe_two_tier(audio: np.ndarray, sr: int = SR,
                        base: str = "base", second: str = "medium",
                        low_conf_threshold: float = LOW_CONF_THRESHOLD,
                        offset_s: float = 0.0) -> dict:
    """Two-tier attended transcription of one onset window.

    Tier 1: whisper `base` with VAD segmentation (faster-whisper's
    vad_filter) -- the VAD sets segment boundaries, which fixes the
    fixed-window edge fuzz the v1 reviews complained about.
    Tier 2: any segment with avg_logprob below `low_conf_threshold`
    is re-run through whisper `second` on its own slice.

    Returns {"segments": [{start, end, text, avg_logprob, tier,
    words}], "n_escalated": int, "tiers": {base, second, threshold}}.
    Times are absolute (offset_s added). Empty audio -> no segments.
    A tier-2 pass that returns nothing falls back to the tier-1
    segment, marked tier "base" -- the fallback is data, not silence.
    """
    from hva.transcribe import _load_model
    audio = _sanitize(audio)
    if sr != SR:
        audio = _resample(audio, sr, SR)
    out: list[dict] = []
    n_escalated = 0
    if len(audio) == 0:
        return {"segments": out, "n_escalated": 0,
                "tiers": {"base": base, "second": second,
                          "low_conf_threshold": low_conf_threshold}}
    m1 = _load_model(base)
    segs, _info = m1.transcribe(audio, vad_filter=True,
                                word_timestamps=True, beam_size=5)
    m2 = None
    for s in segs:
        words = [{"start": round(w.start + offset_s, 3),
                  "end": round(w.end + offset_s, 3),
                  "word": w.word,
                  "prob": round(float(w.probability), 3)}
                 for w in (s.words or [])]
        # avg_logprob should always be a float; None means "unknown" --
        # keep the segment on base rather than crashing or escalating
        # blind.
        lp = s.avg_logprob if s.avg_logprob is not None else 0.0
        seg = {"start": round(s.start + offset_s, 3),
               "end": round(s.end + offset_s, 3),
               "text": s.text.strip(),
               "avg_logprob": round(float(lp), 3),
               "tier": "base",
               "words": words}
        if float(lp) < low_conf_threshold and s.text.strip():
            if m2 is None:
                m2 = _load_model(second)
            a0 = max(0, int(s.start * SR))
            a1 = min(len(audio), int(s.end * SR))
            if a1 > a0:
                rsegs, _ = m2.transcribe(audio[a0:a1],
                                         vad_filter=False,
                                         word_timestamps=True,
                                         beam_size=5)
                if rsegs:
                    n_escalated += 1
                    for r in rsegs:
                        rwords = [{
                            "start": round(w.start + s.start + offset_s, 3),
                            "end": round(w.end + s.start + offset_s, 3),
                            "word": w.word,
                            "prob": round(float(w.probability), 3)}
                            for w in (r.words or [])]
                        out.append({
                            "start": round(r.start + s.start + offset_s, 3),
                            "end": round(r.end + s.start + offset_s, 3),
                            "text": r.text.strip(),
                            "avg_logprob": round(float(r.avg_logprob), 3),
                            "tier": "medium",
                            "words": rwords})
                    continue
        out.append(seg)
    return {"segments": out, "n_escalated": n_escalated,
            "tiers": {"base": base, "second": second,
                      "low_conf_threshold": low_conf_threshold}}


def music_features(audio: np.ndarray, sr: int = SR) -> dict | None:
    """Chroma + tempo for one window. Returns None for windows too
    short (< 1 s) or too quiet (rms < 1e-4) to analyze -- None means
    "not analyzable," not "not music."

    chroma_mean/std: 12 pitch-class vectors (C..B) over the window.
    tempo_bpm: librosa beat-track estimate (None when no beat found).
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError(
            "music analysis needs librosa in the venv "
            "(pip install librosa)") from e
    audio = _sanitize(audio)
    if sr != SR:
        audio = _resample(audio, sr, SR)
        sr = SR
    if len(audio) < SR or float(np.sqrt((audio ** 2).mean())) < 1e-4:
        return None
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    tempo_out = librosa.beat.beat_track(y=audio, sr=sr)
    tempo = tempo_out[0]
    tempo_bpm = float(np.atleast_1d(tempo)[0]) if tempo is not None else None
    return {
        "chroma_mean": [round(float(v), 4) for v in chroma.mean(axis=1)],
        "chroma_std": [round(float(v), 4) for v in chroma.std(axis=1)],
        "tempo_bpm": round(tempo_bpm, 1) if tempo_bpm else None,
        "n_chroma_frames": int(chroma.shape[1]),
    }


def interpret_window(audio: np.ndarray, sr: int = SR,
                     t_s: float = 0.0, strength: float = 0.0,
                     label_top_k: int = 5,
                     transcribe: bool = True,
                     window_start_s: float | None = None) -> dict:
    """One onset window through the full "what" pathway.

    Returns {t_s, strength, window_start_s, labels, transcript, music}
    where transcript is the two-tier result (or None when
    transcribe=False) and music is music_features output, or None when
    the window isn't music-flagged or isn't analyzable.

    Transcript segment times are relative to the window start; add
    window_start_s for absolute media time. (The script that cuts the
    window passes its start here.)
    """
    audio = _sanitize(audio)
    labels = label_window(audio, sr=sr, top_k=label_top_k)
    tx = transcribe_two_tier(audio, sr=sr, offset_s=0.0) \
        if transcribe else None
    music = music_features(audio, sr=sr) \
        if is_music(labels) else None
    return {
        "t_s": round(float(t_s), 3),
        "strength": round(float(strength), 4),
        "window_start_s": (round(float(window_start_s), 3)
                           if window_start_s is not None else None),
        "labels": labels,
        "transcript": tx,
        "music": music,
    }
