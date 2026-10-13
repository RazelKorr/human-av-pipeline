# Setup — rebuilding the working environment from scratch

The venv in this repo was built by accretion; this file is the recipe
that reproduces it. Followed in order, it gets you from a bare clone to
a tree where the test suite and the flagship runners work.

## 0. What is NOT in the repo

`input/`, `output/`, `models/`, `separated/`, and `.venv/` are all
gitignored. The repo alone cannot reproduce a past review run: the
media, the run artifacts, and the model weights live only in the local
workspace (size + copyright). A fresh clone gets the code, the specs,
and the synthetic batteries — bring your own media for real runs.
Two test files also depend on local media and skip when it is absent
(see step 5).

## 1. System prerequisite: ffmpeg

The streaming feeder (`streaming/feeder.py`) and `hva/stream.py` shell
out to a system `ffmpeg` binary. It is not a pip dependency and is not
in the venv. On the dev VM it is ffmpeg 8.x at `/usr/bin/ffmpeg`,
installed at the OS level — note the VM is ephemeral (only `~`
survives a replacement), so after a VM rebuild this must be
reinstalled before anything streaming works:

```bash
ffmpeg -version   # must succeed before proceeding
```

## 2. Create the venv and install Python deps

The venv lives **inside** the repo at `./.venv` (older docs said
`../.venv-pipeline`; that path is retired).

```bash
cd human-av-pipeline
python3 -m venv .venv

# torch first, from the CPU index -- plain PyPI torch either fails or
# pulls a multi-GB CUDA build on a GPU-less machine:
.venv/bin/pip install torch==2.14.0+cpu --index-url https://download.pytorch.org/whl/cpu

.venv/bin/pip install -r requirements.txt

# laion-clap separately: its metadata caps numpy<2 but it runs fine on
# the pinned numpy 2.5.3, so it must bypass the resolver (its real
# deps are already pinned in requirements.txt):
.venv/bin/pip install --no-deps laion-clap==1.1.7
```

`requirements.txt` is pinned to the versions the current working venv
runs (numpy 2.5.3 / scipy 1.18.1 era). If a pin and this file disagree,
the venv's `pip list` is the ground truth — update both.
`pip check` on a correct rebuild reports exactly one conflict — the
laion-clap numpy cap above. That one is known and expected.

## 3. Egress-proxy CA step (required for yt-dlp / HF downloads)

Network egress on this machine goes through a TLS-intercepting proxy.
The venv's stock certifi bundle does not trust it, so HTTPS fetches
from inside the venv (yt-dlp downloads, HuggingFace model fetches)
fail with certificate errors until the proxy CA is appended:

```bash
cat /run/hatch/egress-tls/ca-bundle.pem >> \
  .venv/lib/python3*/site-packages/certifi/cacert.pem
```

The current venv's bundle has ~487 certificates (stock certifi has far
fewer) — that count is the tell that the step was done. **A rebuilt
venv silently loses this.** If yt-dlp breaks right after a rebuild,
this is why. This step was previously documented only outside the repo
(workspace TOOLS.md); it lives here now.

## 4. Model weights (`models/`, gitignored, ~3.3 GB)

Place under `models/` at the repo root:

| Directory / file | Used by | Notes |
|---|---|---|
| `faster-whisper-base/` | `hva/transcribe.py` (MODEL_DIR) | Needs `vocabulary.json` + `preprocessor_config.json` (feature_size 80) for ctranslate2 ≥4 — both present in the working copy. |
| `faster-whisper-medium/` | `hva/interpret.py` two-tier escalation | Same `vocabulary.json` / `preprocessor_config.json` requirement; fixed 2026-10-12. `preprocessor_config.json` copied from base, but `vocabulary.json` was generated from medium's **own** `vocabulary.txt` — do NOT copy base's list: base carries a newer tokenizer revision (extra `<|yue|>` / `<|nospeech|>` tokens shift every id ≥ 50358). Verified by before/after medium transcription smoke test: identical segments and word probs. |
| `laion-clap-630k-audioset-best.pt` | `hva/interpret.py` (CLAP labels) | ~1.9 GB single checkpoint file. |

The faster-whisper dirs are local copies of the Systran repos with the
two config fixes above; ctranslate2 ≥4 refuses the stock
`vocabulary.txt`-only layout.

## 5. Verify

```bash
.venv/bin/python -m pytest tests/ -q     # full suite
.venv/bin/python -m hva.battery          # audio battery: 6/6 passed
.venv/bin/python -m hvp.battery          # vision battery: 5/5 passed
.venv/bin/python -m hvm.battery          # fusion battery: 6/6 passed
```

Media-dependent tests (`test_streaming.py` needs
`input/star_tours_1_ride_film.mp4`; parts of `test_interpret.py` need
`output/drtran_ep4_av/onset_59.1s.wav`) **skip** when the media is
absent — a fresh-clone run shows skips, not errors.

## 6. Optional: Deno for yt-dlp

yt-dlp works without a JS runtime but warns about it. The dev machine
keeps Deno 2.4.5 at `~/workspace/bin/deno` (outside the repo, persists
across VM replacements). Add it to PATH or pass
`--js-runtimes deno:/path/to/deno` if the warnings bother you.

## 7. Vocal separation (demucs)

`hva.transcribe --separate-vocals` shells out to `python -m demucs`
(two-stems vocals). demucs is pinned in `requirements.txt`; stems land
in `separated/` (gitignored). First run downloads the separation model
weights (~a few hundred MB) — which needs step 3 done first.
