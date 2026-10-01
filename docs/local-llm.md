# Local LLM staple

How to put real weights behind `ResponsePolicy.generate()` on the
desktop (no API, no network, fully local).

## The machine (desktop, verified 2026-09-16)

- Intel Core i7-7700K @ 4.20 GHz
- 32 GB RAM
- Radeon RX 580 8 GB (no NVIDIA -- CUDA is out; llama.cpp via Vulkan)
- LM Studio installed (corrected 2026-09-30; earlier note said LocalLlama)

## Model rec (researched 2026-09-30)

Primary: **Qwen3-8B-Instruct, Q4_K_M GGUF** (~5 GB)
- `Qwen/Qwen3-8B-GGUF:Q4_K_M` or `lmstudio-community/Qwen3-8B-GGUF`
- The safe, mature pick: strong instruction following, thinking and
  non-thinking modes, fits the 8 GB VRAM with KV headroom.

Fast alternative: **LFM2.5-8B-A1B** (~5.2 GB)
- Hybrid MoE, only ~1.5B active per token -- near-3B speed at 8B
  quality. Day-one llama.cpp support.

Newest: **Qwen3.5-4B** (~2.7 GB)
- Strongest benchmarks per GB right now, but needs a very recent
  llama.cpp (Gated DeltaNet + MTP support). Check the HF page for
  the minimum build.

Q4_K_M is the sweet spot. Q3 and below degrade instruction following;
Q5/Q6 are nicer if RAM allows.

## Install (Windows, RX 580)

1. Grab the latest release from https://github.com/ggml-org/llama.cpp/releases — the zip with `win-vulkan-x64` in the name — and extract it somewhere simple like `C:\llama`. Make sure your AMD Adrenalin drivers are current; Vulkan comes with them.
2. Open a terminal in that folder and run:

```bash
# downloads the model (~5 GB) on first run, then serves it
llama-server -hf unsloth/Qwen3-8B-GGUF:Q4_K_M --port 8080
```

Leave that window open. In another shell:

```bash
git clone https://github.com/RazelKorr/human-av-pipeline
cd human-av-pipeline
```

Then run the conversation loop with `--llm local` (it defaults to `http://localhost:8080`; `--llm-url` overrides).

## Staple it (one line)

```python
from hva.llm import LocalGenerator
policy.llm = LocalGenerator()  # http://localhost:8080
```

`LocalGenerator.available` probes `/health`, so if the server isn't
up the policy silently falls back to the rule-based replies. Same
payload, same `LOOK:` line convention as the API backend.

## API alternative

`policy.llm = ApiGenerator()` reads `ANTHROPIC_API_KEY` from the
environment. Same seam, over the wire instead of local.
