"""OCR for diegetic signage: reading the text inside the frame.

EXPERIMENTAL (Phase 5, 2026-10-12; audit item 17). Not wired
into the flagship streaming path.

REVIEW-PROTOCOL §6 has always sourced signage by human reading —
the pipeline had no way to read text in the frame, so beat maps
cite signs a person read. This module is the machine seam for
that job: frames in, [{t, text, confidence, bbox}] out, every
reading carrying the engine's own confidence, filtered at a
floor. A reading is a claim with a score, like every other
channel in this project — below the floor it does not ship.

Engine situation (2026-10-12): tesseract 5.3.4 is installed on
this machine (/usr/bin/tesseract, an OS-level binary like ffmpeg
— VM-ephemeral, see SETUP.md) and pytesseract 0.3.13 is in the
venv. The engine interface is pluggable (any object with
.read_words(image) -> [{text, confidence, bbox}]) so tests run
against a stub and a future engine (easyocr, a VLM pass) can
slot in without touching callers. If pytesseract is missing the
tesseract engine falls back to the `tesseract` CLI's TSV output;
if the binary itself is missing, TesseractEngine raises
FileNotFoundError naming the install step.

Preprocessing: grayscale, and a 2x upscale when the frame is
narrower than 1000 px (tesseract's accuracy falls off a cliff
on small text; upscaling does not invent detail, it stops the
engine from discarding what is there). Both steps are noted in
the returned records' "engine" tag via the engine name only —
callers that need the raw frame's limits should OCR the raw
frame too and compare, not trust one pass.
"""

from __future__ import annotations

import subprocess
import tempfile
import os

import numpy as np

DEFAULT_MIN_CONF = 40.0   # tesseract word confidence floor (0-100)
ENGINE_NAME = "tesseract"


class StubEngine:
    """Test double: returns canned word records verbatim."""

    name = "stub"

    def __init__(self, words: list[dict]):
        self._words = words

    def read_words(self, image) -> list[dict]:
        return [dict(w) for w in self._words]


class TesseractEngine:
    """tesseract via pytesseract, falling back to the CLI's TSV."""

    name = ENGINE_NAME

    def __init__(self, lang: str = "eng", psm: int | None = None):
        self.lang = lang
        self.psm = psm

    def _config(self) -> str:
        return f"--psm {self.psm}" if self.psm else ""

    def read_words(self, image) -> list[dict]:
        try:
            import pytesseract
            from pytesseract import Output
            data = pytesseract.image_to_data(
                image, lang=self.lang, config=self._config(),
                output_type=Output.DICT)
            out = []
            for i, txt in enumerate(data["text"]):
                out.append({
                    "text": txt,
                    "confidence": float(data["conf"][i]),
                    "bbox": {"x": int(data["left"][i]),
                             "y": int(data["top"][i]),
                             "w": int(data["width"][i]),
                             "h": int(data["height"][i])},
                })
            return out
        except ImportError:
            return self._read_words_cli(image)

    def _read_words_cli(self, image) -> list[dict]:
        exe = _tesseract_exe()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "frame.png")
            image.save(path)
            cmd = [exe, path, "stdout", "--psm",
                   str(self.psm or 3), "tsv"]
            proc = subprocess.run(cmd, capture_output=True,
                                  text=True, check=True)
        rows = proc.stdout.splitlines()
        if not rows:
            return []
        header = rows[0].split("\t")
        idx = {name: i for i, name in enumerate(header)}
        out = []
        for line in rows[1:]:
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            try:
                conf = float(parts[idx["conf"]])
            except ValueError:
                continue
            out.append({
                "text": parts[idx["text"]],
                "confidence": conf,
                "bbox": {"x": int(parts[idx["left"]]),
                         "y": int(parts[idx["top"]]),
                         "w": int(parts[idx["width"]]),
                         "h": int(parts[idx["height"]])},
            })
        return out


def _tesseract_exe() -> str:
    from shutil import which
    exe = which("tesseract")
    if exe is None:
        raise FileNotFoundError(
            "tesseract binary not found on PATH; install it at the "
            "OS level (apt-get install tesseract-ocr) — see SETUP.md "
            "system prerequisites")
    return exe


def get_engine(engine=None):
    """Resolve an engine argument: None -> TesseractEngine; an
    object with read_words passes through; the string 'stub'
    is rejected (tests construct StubEngine explicitly)."""
    if engine is None:
        _tesseract_exe()
        return TesseractEngine()
    if hasattr(engine, "read_words"):
        return engine
    raise TypeError(f"unknown OCR engine: {engine!r}")


def _prepare(frame) -> "object":
    """numpy frame (H,W,3|4) or PIL image -> grayscale PIL image,
    2x-upscaled when narrower than 1000 px."""
    from PIL import Image
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(frame)
    else:
        img = frame
    img = img.convert("L")
    if img.width < 1000:
        img = img.resize((img.width * 2, img.height * 2),
                         Image.LANCZOS)
    return img


def ocr_frame(frame, t_s: float = 0.0, engine=None,
              min_conf: float = DEFAULT_MIN_CONF) -> list[dict]:
    """OCR one frame. Returns word records [{t, text, confidence,
    bbox}] with confidence >= min_conf and non-empty text, in
    engine reading order. bbox is in PREPARED-image coordinates
    (after the possible 2x upscale) — noted, not hidden."""
    eng = get_engine(engine)
    img = _prepare(frame)
    scale = img.width / (frame.shape[1] if isinstance(frame,
                        np.ndarray) else frame.width)
    out = []
    for w in eng.read_words(img):
        text = (w.get("text") or "").strip()
        conf = float(w.get("confidence", -1))
        if not text or conf < min_conf:
            continue
        bbox = dict(w["bbox"])
        if scale != 1.0:
            bbox = {k: int(round(v / scale)) for k, v in bbox.items()}
        out.append({"t": round(float(t_s), 3), "text": text,
                    "confidence": round(conf, 1), "bbox": bbox})
    return out


def ocr_frames(frames, engine=None,
               min_conf: float = DEFAULT_MIN_CONF) -> list[dict]:
    """frames: iterable of (t_s, frame). Flat word records."""
    eng = get_engine(engine)
    out = []
    for t_s, frame in frames:
        out.extend(ocr_frame(frame, t_s=t_s, engine=eng,
                             min_conf=min_conf))
    return out


def extract_frame(video_path: str, t_s: float):
    """One frame at t_s via the system ffmpeg (same dependency
    class as streaming/feeder.py). Returns an (H, W, 3) uint8
    array."""
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "frame.png")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{t_s}", "-i", video_path,
             "-frames:v", "1", path],
            capture_output=True, check=True)
        return np.asarray(Image.open(path).convert("RGB"))


def ocr_video(video_path: str, timestamps, engine=None,
              min_conf: float = DEFAULT_MIN_CONF) -> list[dict]:
    """OCR a video at the given timestamps (seconds)."""
    eng = get_engine(engine)
    out = []
    for t_s in timestamps:
        frame = extract_frame(video_path, float(t_s))
        out.extend(ocr_frame(frame, t_s=float(t_s), engine=eng,
                             min_conf=min_conf))
    return out
