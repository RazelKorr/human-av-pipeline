#!/usr/bin/env python3
"""Transcribe speech in a wav to the perceptual-moment grid.

Thin wrapper around hva.transcribe. Requires the pipeline venv
(workspace/.venv-pipeline) for faster-whisper; the base model lives in
models/faster-whisper-base (gitignored, downloaded once).

Example:
    scripts/transcribe.py --wav input/drtran_ep17_16k.wav \
        --out output/transcripts/ep17.json
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.argv[0] = "hva.transcribe"
runpy.run_module("hva.transcribe", run_name="__main__")
