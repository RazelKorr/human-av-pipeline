# Sensorium (human-av-pipeline) — Deep Audit

**Date:** 2026-10-12 (PDT) · **Scope:** read-only audit commissioned by Mykal: current state, spec-vs-code drift, test health, run-data verification, improvement opportunities. No pipeline code, specs, or outputs were modified; this report is the only file written.

**Evidence discipline:** every claim below names its source (file, line-level location, command, or run artifact). Claims I could not establish from code or run data are marked **UNVERIFIED**. Doc claims are labeled as doc claims, never as verified behavior.

**Headline state:** The project is real, coherent, and heavily self-documented. The code runs: the full test suite passes 259/259 and all three synthetic validation batteries pass when actually invoked. The two flagship runs' headline numbers reproduce exactly from the run artifacts on disk. The weaknesses are not "it doesn't work" — they are (1) two architecture generations coexisting while the specs mostly describe the older one, (2) the binding metric shipping without its chance baseline, (3) the pursuit channel degenerating on egomotion footage, (4) a stale `requirements.txt` that would not rebuild the current venv, and (5) a README battery command that silently does nothing.

---

## Lane 1 — Codebase & architecture

### Inventory (verified: `find`, `wc -l`, `git ls-files`)

108 tracked files; working tree clean at HEAD `9d6eb3d` (2026-10-04 08:15 UTC, "Streaming report: FF3 1080p full Sensorium watch"). ~16,600 lines of Python across packages, scripts, streaming, and tests. Nothing in the repo has been modified since 2026-10-04 except `input/` and `output/` (the Pink Panther run, 2026-10-06).

| Package | Files | What it actually does (from module docstrings + imports) |
|---|---|---|
| `hva/` (ears, batch gen) | 14 py | `cochlea.py` (16 kHz STFT, 64 log bins 50 Hz–8 kHz, absolute dBFS), `salience.py` (intensity/freq-contrast/temporal-contrast, weights 0.4/0.3/0.6, −60 dBFS floor — matches SPEC_AUDIO exactly), `moments.py` (100 ms moments, linear-power loudness), `attention.py` (frequency-band controller: 95th-percentile onset interrupt, 200 ms decision clock, 150 ms switch latency, blink, IOR, silence-dwell), `spatial.py` (ILD/ITD pan), `transcribe.py` (faster-whisper, per-word probs + `avg_logprob` recorded at lines 198/204), `interpret.py` (CLAP zero-shot labels, two-tier base→medium transcription with logprob escalation threshold, music features), `online.py` (**the streaming ear**: per-50 ms RMS/spectral-flux/centroid + onset picking), `stream.py` (batch-parity streaming front end), `understanding.py` + `conversation.py` + `llm.py` (grounded Q&A / turn-taking / LLM seam for the conversation loop), `battery.py` (synthetic A-battery). |
| `hvp/` (eyes, batch gen) | 11 py | `pipeline.py` (VisionPipeline: temporal integrator → saccade controller → foveated retina → 10 Hz moments), `retina.py` (grayscale foveation + color foveation incl. `luma_ratio` fast path), `saccades.py` (main-sequence durations, pursuit segments, darkness gate: `DARK_LUM=0.03`, up to 3× dwell), `attention.py` (56×56 salience: center-surround + transient + chroma opponency, IOR), `motion.py` (MotionChannel: frame-diff energy, local-coherence gate, τ=120 ms persistence, centroid velocity tracking, pursuit drive), `recognize.py` (CLIP foveal classifier), `detect.py` (OWL-ViT open-vocab), `battery.py` (T0–T4). |
| `hvm/` (fusion L3) | 5 py | `priority.py` (JointPriorityMap, 56×56; constants at lines 54–61 match SPEC_FUSION verbatim: W_VIS 1.0, W_AUD 0.7, TAU 300 ms, READ_GAIN_K 1.5, SPEECH_BOOST 1.0), `driver.py` (closed loop), `battery.py` (M1–M6), `online.py`. |
| `streaming/` | 7 py + report | `feeder.py` (ffmpeg split-decode, chunked, bounded), `online_driver.py` (causal attention driver + magno/pursuit), `online_render.py` (bounded-buffer VisionPipeline subclass), `run_stream.py` (**the flagship runner**), `bind_av.py` (onset↔transient coincidence binding, ±250 ms, greedy by descending audio strength), `compare.py` (batch-vs-stream comparator), `STREAMING-REPORT.md` (74 KB build diary + run addenda). |
| `scripts/` | 34 py | Entry points: `run_video_saccades_color.py` (batch v2 color path the streaming harness mirrors), `run_level3.py`, `fuse_av.py` (L1), `couple_av.py` (L2), `watch.py`, `perceive.py`, `run_conversation.py`, FF3 interpretation drivers (parallel/staggered/batched/phase-2), plus a dozen one-off `audit_*.py` probes. |
| `tests/` | 14 files | 259 test functions (counted via `grep -c "def test"` per file; largest: `test_understanding.py` 68, `test_motion.py` 32, `test_conversation.py` 29). |

### The central architectural finding: two generations, one set of specs

The flagship review runs (`streaming/run_stream.py`) do **not** use the SPEC_AUDIO attention stack. Verified from `run_stream.py` imports (lines 41–47, 384): vision comes from `online_driver`/`online_render` (the v2 color-saccades path), audio from `hva/online.py` (coarse 50 ms RMS/flux/centroid features + `pick_onsets`), binding from `streaming/bind_av.py`, transcription from `hva/transcribe.py` on the top-K onsets only. `hva/cochlea.py`, `hva/salience.py`, `hva/attention.py` — the entire cocktail-party controller that SPEC_AUDIO describes — are exercised by the L1/L2/L3 scripts and `hva/stream.py`, but **not** by any flagship review run. Likewise the moment grain differs by generation: batch specs and README describe 100 ms / 10 Hz moments; flagship runs are 50 ms / 20 Hz (`run_report.json`: `"moment_ms": 50.0, "moment_hz": 20.0` in the Pink Panther run; FF3/ST `audio_features.npy` rows = 2× processed seconds).

This is not hidden — `hva/online.py`'s docstring calls itself "the ear's magno channel" and STREAMING-REPORT documents the harness thoroughly — but a reader of README + the three SPECs alone would form a materially wrong picture of what produced the FF3 and Star Tours reviews.

### Drift list (promised-not-implemented / implemented-not-documented / stale)

1. **SPEC_VISION.md never mentions motion.** `grep -i motion SPEC_VISION.md` = 0 hits. The MotionChannel (`hvp/motion.py`, 195 lines) and smooth pursuit — headline behaviors of every flagship run — have no spec. SPEC_VISION §4 also lists "No color opponency" as a deliberate non-goal, while `hvp/attention.py` implements chroma opponency salience and `hvp/retina.py` implements full color foveation; color is the flagship v2 path. The spec describes the v1 temporal baseline only.
2. **SPEC_AUDIO.md does not document `hva/online.py`** (the 50 ms streaming front end that actually feeds the flagship runs) or `hva/interpret.py` (CLAP + two-tier transcription + music analysis; the FF3 interpretation pathway). Both are documented only in STREAMING-REPORT addenda and module docstrings.
3. **README Level-3 battery count is wrong.** README: "Battery 5/5". SPEC_FUSION: 6/6 (M1–M6). Audit run of `hvm.battery.run_all()`: **6/6 PASS**, with M-values matching SPEC_FUSION's text exactly (M1 ratio 1.59, M2 peak x=10.0, M3 x=42.0, M5 gains 1.38/1.00, M6 flip 42.0→10.0). The spec is right; the README is stale.
4. **README's transcription setup is stale.** README says transcription "Requires the pipeline venv (`../.venv-pipeline`, relative to the repo root)". Actual: the working venv is `./.venv` inside the repo, and `hva/transcribe.py` `MODEL_DIR` (line 45–46) resolves to repo-local `models/`. Nobody following the README would find the working setup.
5. **`hva/battery.py` docstring says "five synthetic checks"**; it implements six (A0, A1, A1b, A2, A3, A4 — all PASS, Lane 2). SPEC_AUDIO and README say six. Docstring drift only.
6. **README's `watch.py` claim** ("Sixty-two seconds of video, reviewed in seventeen") is a legacy L1-era line; no current run artifact verifies it. **UNVERIFIED** against present code.
7. **What is NOT drift:** SPEC_AUDIO's cochlea/salience constants, SPEC_FUSION's joint-map constants, and SPEC_VISION's timing budget all match the code at the lines cited above. The older generation is honestly specced; it's the newer generation that's under-specced.

---

## Lane 2 — Test health

### Result (verified, controlled rerun)

```
cd ~/workspace/human-av-pipeline
.venv/bin/python -m pytest tests/ -q   →   259 passed, 1 warning in 127.99s (exit 0)
```

259 collected = 259 test functions counted in source. The 1 warning is a torch `meshgrid` deprecation notice in `test_interpret.py::test_label_window_schema_and_scores` — cosmetic.

**Process note (corrected after the fact):** the full suite was run twice during this audit and **both runs passed 259/259, exit 0** — the original run ("259 passed, 1 warning in 199.77s") and a controlled rerun with output captured to a file ("259 passed, 1 warning in 127.99s"). Mid-audit, an unreliable automated summary of the first run reported a failure (a test name that does not exist in the tree, counts that did not sum to 259); that account was wrong, and this report's first draft repeated it. The run's actual terminal output, delivered afterward, is unambiguous. Supporting checks: `tests/test_recognition.py` standalone **8/8 passed** (25 s); model-heavy subset (`test_interpret.py`, `test_runner_owl.py`, `test_llm.py`) **37/37 passed**. One residual observation: `.pytest_cache/v/cache/lastfailed` contained an entry for `test_recognition.py::test_owlvit_detects_gate_in_crop`, but the cache directory predates this audit (Sep 30) and the test passes standalone and in both full runs — a stale cache entry, not a live failure. No flake was observed.

### Batteries (verified by direct invocation, 2026-10-12)

| Battery | Result | How verified |
|---|---|---|
| `hvp.battery` T0–T4 | **5/5 PASS** | `run_all()` — T0 200 ms; T1 fused (var ratio 0.0000); T2 −0.125 rev/s (reversed, as expected); T3 ratio 0.000 < 0.25; T4 2.0 vs 25.0 cycles |
| `hva.battery` A0–A4+A1b | **6/6 PASS** | `run_all()` — A1 unattended glide missed, A1b attended glide noticed (0.8 s), A2 100% dwell on cued stream, A4 refractory observed. Minor anomaly: A0/A3 report `capture_latency_s: -0.1` — captures land one moment *before* the click on the moment grid; a grid-alignment quirk worth a comment, not a behavior failure |
| `hvm.battery` M1–M6 | **6/6 PASS** | `python -m hvm.battery` — values match SPEC_FUSION text exactly (see Lane 1) |

### Finding: the documented battery command is a silent no-op

README's "Running it" section says `python3 -m hva.battery`. `hva/battery.py` defines `run_all()` (line 270) and a print loop, but has **no `if __name__ == "__main__"` guard** — verified: the command exits 0 printing nothing. `hvp/battery.py` has the same defect. Only `hvm/battery.py` (line 165) has the guard. Anyone "verifying" the audio battery per the README in the last weeks has verified nothing.

### What the tests do NOT cover (verified by grep over `tests/`)

- **The three batteries are not in pytest at all** (no test file imports any `battery` module). "Full suite green" — the phrase every flagship run's review leans on — does not include the headline T/A/M validation batteries.
- **No chance-baseline test or code for binding** (Lane 3).
- **No direct tests of the L1/L2/L3 real-media scripts** (`fuse_av.py`, `couple_av.py`, `run_level3.py`, `watch.py`, `perceive.py`). `JointPriorityMap` is touched only indirectly via `test_understanding.py`; `bind_av` *is* unit-tested (`test_audio.py`: greediness, shapes, decode).
- **No end-to-end review-production test** — nothing exercises REVIEW-PROTOCOL compliance (beat map precedence, sealed predictions) mechanically; it's human/agent discipline only.
- **Fixture dependence:** `test_streaming.py` requires `input/star_tours_1_ride_film.mp4` (gitignored media) and `test_interpret.py` requires `output/drtran_ep4_av/onset_59.1s.wav` — the suite is not runnable from a fresh clone without the local media/output tree.

---

## Lane 3 — Pathways vs run data

All numbers below were read from the run artifacts on disk during this audit (`run_report.json`, `av_binding.json`, `audio_events.json`, `audio_interpretation.json`, `*.npy` shapes), not from README, reviews, or memory.

### Flagship run 1 — FF3 1080p (`output/ff3_hd_av/`, 2026-10-04)

| Metric | Measured |
|---|---|
| Duration / moments | 2,701.0 s processed; **54,019 moments** (`audio_features.npy` 54019×3) at 50 ms grid |
| Saccades / pursuits | **7,867** saccades (`scanpath.npy` 7868 rows incl. initial), **1,127** pursuits (`pursuit_log.npy` 1127×4) |
| Audio onsets | **3,362** (`audio_events.json` length = `av_binding.json` `n_audio_onsets`) at 1.24/s |
| Visual transients | 2,735 |
| Bound | **1,208 → 35.9%** of onsets; 2,154 unbound audio, 1,527 unbound visual |
| Interpretation | 918 windows sampled (every 4th onset + top-100 loudest, per `audio_interpretation.json` `sampling`); CLAP top-1: **speech 308/918 = 33.6%** (verified by recount), footsteps 222 (24.2%), thud 78, engine 60, music 58; two-tier transcription 110 windows, **47 escalated** to medium, **0 errors** |
| Cost | wall 3,346.5 s → **0.81× realtime**; 0/1350 chunk deadline misses; frame proc mean 31.3 ms vs 33.4 ms interval |

### Flagship run 2 — Star Tours AV (`output/st_stream_av_full/`, 2026-10-04)

| Metric | Measured |
|---|---|
| Duration / moments | 269.0 s; **5,380 moments** |
| Saccades / pursuits | **735 / 161** |
| Onsets / visual transients | **238 / 328** |
| Bound | **91 → 38.2%** of onsets |
| Interpretation | all 238 onsets as records; 66 music windows; 0 escalations, 0 errors |
| Cost | **0.86× realtime**; 0/135 deadline misses; audio cost ~0.039 ms/moment (report figure, consistent with run_report telemetry) |

### Newest run — Pink Panther / Cato Attacks (`output/pink-panther-cato-attacks/`, 2026-10-06)

4,001 moments over 200.1 s; 685 saccades; **motion channel OFF** (`motion_weight: null`, `pursuit: false`, 0 pursuits — the only flagship-class run without the magno path); 170 onsets, 260 visual transients, **89 bound (52.4%)**; full-house review in protocol format including a filed protocol-deviation note. This run appears **nowhere in STREAMING-REPORT.md** (Lane 6).

### Pathway-by-pathway verdicts

- **Vision — saccades:** Real and measured in every run. But the rate is suspiciously content-independent: FF3 review reports 2.9/s flat across all beats (lowest 5-min bin at 94% of mean); Pink Panther review reports ~3.3/s "dead flat" across a quiet bedroom and a brawl. The decision clock behaves like a metronome with salience steering *where*, not *whether/when*. That is a model property worth naming as a finding, not a bug per se — human scan rates do vary with scene demand, and this one largely doesn't.
- **Vision — pursuit/motion:** The MotionChannel fires (motion energies present in all motion-on runs), but on FF3 it **degenerates**: all 1,127 pursuit segments are exactly the 0.40 s minimum duration, chained through walking footage (427.3 s of gliding, 15.8% of the film) — the channel locks onto the egomotion flow field and hop-sustains instead of tracking objects. Star Tours shows the healthy contrast (161 segments, 60.2 s, variable). Dr Tran ep4 shows the starved contrast (51 segments, all minimum-duration, 4.6%). One channel, three regimes, only one of them is "pursuit" as a human would mean it.
- **Color:** Present and load-bearing in the flagship path (`chroma_mode: luma_ratio`, `color: true` in run reports; `foveate_color_luma_ratio` in `hvp/retina.py`). Not in SPEC_VISION (Lane 1 drift #1).
- **Audio (flagship):** The 50 ms features are exactly what `hva/online.py` promises — RMS/flux/centroid per moment, absolute flux with median+MAD adaptive onset threshold and an RMS floor. Onset counts are real (JSON artifacts above). The batch cochlea/attention stack is separately real (battery-verified) but is not this pathway.
- **Fusion/binding — the audit's most important measurement finding:** `streaming/bind_av.py` is a ±250 ms coincidence counter with greedy nearest matching. It computes **no chance baseline**. The project's own red-team addenda (STREAMING-REPORT) supply the baselines by hand and they change the meaning of the headline number in two of three cases: Dr Tran ep4's 50.7% **equals** its Poisson chance rate (0.507 vs 0.50, flat dt distribution); FF3's 35.9% is **below** its 39.7% chance rate (~159 zero-dt pairs are the genuine candidates); only Star Tours' 38.2% shows dt clustering consistent with causal binding. **The bound percentage, as currently produced, is not comparable across runs and does not by itself measure binding.** The pipeline's reviews know this; the pipeline's code doesn't — `av_binding.json` ships the raw rate with no baseline field.
- **Interpretation/transcription:** The logprob discipline is implemented, not just preached: `hva/transcribe.py` records per-word probabilities and per-segment `avg_logprob`; `hva/interpret.py` escalates below-threshold base segments to medium; onset transcripts in both flagship runs carry `avg_logprob` on every segment (verified in `onset_transcripts.json`). The two-tier mechanism demonstrably did its job on FF3 (the "We're in Marvel!" base-tier hallucination at 1649.7 s resolved to screams by the medium tier — present in the run's interpretation records and transcripts). CLAP labeling runs at scale (918 windows, 0 errors). Remaining gap, documented in SPEC_AUDIO and still true: no mishearing model — confidence is recorded and used for escalation, but low-confidence text is still verbatim text, gated only by reviewer discipline.

---

## Lane 4 — Fixtures & inputs

### Inputs (`input/`, 1.6 GB, 27 files, all gitignored)

Star Tours dominates: the full ride film + 62 s cut + seven derived WAVs (mono/stereo/vocals stems at multiple rates). Dr Tran: episodes 1, 17, 4 (ep4 added 2026-10-04 specifically as the motion-fixture corrective) + derived WAVs. FF3: 640×480 full, tail, audio stems, and the 727 MB 1080p `ff3_full_hd.mp4`. Plus `agent-test.mp4`, three `owl-test*.mp4`, a YTPMV clip, and `sensorium-yt-Tu1RZaFnkKs.mp4` (the Pink Panther clip, newest, 2026-10-06).

### Fixture balance — the known issue, re-measured

The historical failure (dialogue-heavy fixtures ~16× lower motion energy than Star Tours hid the motion pathway for weeks) is **substantially mitigated but not systematized**:

- A dedicated motion-rich-vs-dialogue contrast now exists in the run corpus, and I measured the actual motion-energy column (`motion_energies.npy` col 1, mean): Star Tours 0.0227, FF3 0.0186, Dr Tran ep4 0.0125. On this metric the modern corpus spans only ~1.8×, not 16× — but raw frame-difference energy was never the whole story: the *pursuit* outcomes still separate cleanly by fixture type (ST variable-length tracking; ep4 all-minimum starved; FF3 all-minimum egomotion-locked), which is the fixture lesson with numbers on it (STREAMING-REPORT, Dr Tran addendum).
- Unit-test fixtures remain nearly nonexistent as media: `tests/fixtures/` holds exactly two files (`foveal_crops.png`, `foveal_labels.json`, for recognition). Everything else is synthetic scenes generated in-test (good for determinism) or depends on the gitignored `input/`/`output/` trees (bad for portability).
- There is still **no quiet/near-silent video fixture** in the run rotation — the regime where the audio dwell rule and the vision darkness gate are supposed to matter, and where SPEC_AUDIO's own Star Tours A/B showed the dwell rule never triggers on ride footage.

### Models (`models/`, 3.3 GB, gitignored)

| Model | Size | Referenced by |
|---|---|---|
| `faster-whisper-base` | 145 MB | `hva/transcribe.py` MODEL_DIR; has the corrected `vocabulary.json` + `preprocessor_config.json` (feature_size 80) per SPEC_AUDIO's setup notes — verified present |
| `faster-whisper-medium` | 1.53 GB | `hva/interpret.py` two-tier escalation — verified working in FF3 run (47 escalations, 0 errors). **Fragility:** this dir still ships `vocabulary.txt` (not `.json`) and no `preprocessor_config.json`; SPEC_AUDIO documents ctranslate2 ≥4 demanding `vocabulary.json` for the Systran repos. It loads under the installed ctranslate2 4.8.2 today; a version bump could break the escalation tier silently into per-window errors |
| `laion-clap-630k-audioset-best.pt` | 1.86 GB | `hva/interpret.py` (path at line 129, with a helpful missing-weights error) — the single largest model and the least documented in the specs |

No orphan models: everything in `models/` is referenced by code.

---

## Lane 5 — Environment & reproducibility

**Verdict: the current venv works; `requirements.txt` would not rebuild it.** Verified by comparing `requirements.txt` against `.venv/bin/python -m pip list` (104 packages):

- **Version drift on pinned core deps:** requirements pins numpy 1.26.4 / scipy 1.11.4 / matplotlib 3.6.3; the venv actually runs **numpy 2.5.3 / scipy 1.18.1 / matplotlib 3.8.4**. Pillow 10.2.0, faster-whisper 1.2.1, transformers 5.17.0, huggingface_hub 1.33.0 match. torch is installed as 2.14.0+cpu; requirements lists plain `torch==2.14.0` with a comment about the CPU index (the comment is the only thing standing between a rebuild and a multi-GB CUDA pull or a failure).
- **Undeclared dependencies the code imports:** PyAV (`av` 19.0.0 — imported by `hva/transcribe.py` and `hva/stream.py`), `laion_clap` 1.1.7 (+ `librosa`, `soundfile`, `torchlibrosa`), `yt-dlp` (2026.8.19, in-venv), and `pytest` 9.1.1 itself. None appear in `requirements.txt`. A fresh `pip install -r requirements.txt` produces an environment that cannot transcribe, cannot interpret, and cannot run the tests.
- **Currently broken path:** `demucs` is **not installed** (`import demucs` → ModuleNotFoundError). `hva/transcribe.py --separate-vocals` therefore fails today (it does raise an informative error, line ~158). The `separated/` stems on disk (99 MB) date from when it was installed. REVIEW-PROTOCOL §6 still describes `--separate-vocals` as a working one-command step.
- **System ffmpeg is a hard dependency and is VM-ephemeral:** the feeder and `hva/stream.py` shell out to `ffmpeg` (system 8.1.2 at `/usr/bin/ffmpeg`). It is not mentioned as a prerequisite in README. On a VM replacement it must be reinstalled; only `~` survives.
- **Egress-proxy CA quirk is live and correctly applied here:** the venv's certifi bundle (`.../.venv/.../certifi/cacert.pem`) contains 487 certificates — far above stock — consistent with the documented append of `/run/hatch/egress-tls/ca-bundle.pem` (which exists). yt-dlp 2026.8.19 is installed in this venv. A rebuilt venv loses the append and yt-dlp/HF fetches break until it's redone; the recipe lives in the workspace TOOLS.md, **not in the repo** — the repo itself contains no record of this requirement.
- **Deno** 2.4.5 present at `~/workspace/bin/deno` (persists, outside the repo).
- **Git hygiene:** tree clean, last commit 2026-10-04. `AGENTS.md` records three `git reset --hard` incidents wiping uncommitted work — the "commit early" rule is written down; the 8-day commit gap since Oct 4 is fine only because code hasn't changed (the Oct 6 Pink Panther run produced gitignored outputs only).
- **Run reproducibility caveat:** `input/`, `output/`, `models/`, `.venv/` are all gitignored. The entire evidentiary basis of the reviews (run artifacts, media, weights) exists only in this workspace. That is a deliberate local-only design (copyright + size), but it means "reproduce the FF3 run" is impossible from the repo alone.

---

## Lane 6 — Outputs & docs currency

### Outputs (`output/`, 992 MB, 48 entries)

- **Flagship tier is well-formed:** `ff3_hd_av/`, `st_stream_av_full/`, `drtran_ep4_av/`, `pink-panther-cato-attacks/` each carry run_report + binding + features + beat-map + review (+ sealed predictions for FF3/ST/Dr Tran). Both flagship reviews follow the house format (header/how-watched, review, what-the-ears-caught, what-the-eyes-caught, evidence, errata) — verified from section headings.
- **Superseded runs are not marked as superseded in place:** `output/ff3/` (Oct 2, 640×480 notes/transcript) and `output/ff3_full/` (Oct 2, the superseded draft review) contain no marker that `ff3_hd_av/` replaced them; same for the pre-streaming Star Tours generations (`st_full/`, `st_full_v2/`, `st_batch_*`, `st_stream_rt_*`, `st_stream_magno_*`). The lineage is reconstructible from STREAMING-REPORT, but anyone browsing `output/` cold will find at least three FF3 "reviews" with no ranking. (Known open item from prior housekeeping; still open.)
- **Scratch tier:** `owl-test` ×7, `conversation_free_test` ×3, `agent-test` ×2, `ytpmv-tea-party` — conversation/OWL experiment logs with no summaries. Harmless, unlabeled.
- **Validation twins are the good example:** `st_stream_av_full_audio_off_twin/` and kin are named for exactly what they are.
- `output/REPORT.md` is the old HVP battery report (T0–T4 PASS) — accurate but undated in content.

### Docs currency

- **STREAMING-REPORT.md** is the project's real living document (1,241 lines; every flagship number I checked against run artifacts matched it, including the self-critical binding-chance addenda). Two currency defects: its header says "Last updated 2026-10-03" while its newest addendum is 2026-10-04 (FF3), and it omits the 2026-10-06 Pink Panther run entirely. Header also claims "seven addenda"; the changelog contains more than a dozen dated sections.
- **REVIEW-PROTOCOL.md vs practice:** the protocol was followed in the flagship reviews — beat maps exist and are falsifiable (ST's beat map carries the corrected LEFT turn as beat 4), sealed predictions precede runs, errata sections contain genuine red-team reversals (whiteout gaze, hyperspace gate 1.5-for-2, R5 rescoring), and claims cite logprobs. One precedence caveat: FF3's `beat-map.md` mtime is the same minute as `review.md` (2026-10-04 08:15), after run completion (07:13) — consistent with the documented pre-review B1 correction, but mtimes alone cannot prove the map preceded the drafting. ST's map (03:52) cleanly precedes its run outputs (03:56). Protocol §6's tooling claims are accurate except `--separate-vocals` (Lane 5, demucs missing).
- **docs/** (5 files: streaming, recognition, saying-hi, real-scene-grounding, local-llm) are design docs from Sep 30–Oct 4; nothing in them contradicted code I sampled, and `docs/streaming.md` is largely superseded by STREAMING-REPORT (it says so implicitly by age, not explicitly).

---

## Lane 7 — Improvement opportunities (ranked; NOT implemented — audit only)

### (a) Safe / small

1. **Fix the battery entry points and README commands.** Add `__main__` guards to `hva/battery.py` and `hvp/battery.py`; correct README (L3 battery 6/6; venv path `./.venv`; the 10 Hz headline → note the two moment grains). *Finding: Lane 2 no-op command; Lane 1 drift #3/#4.* Impact: anyone can actually verify the headline claims. Effort: minutes.
2. **Refresh `requirements.txt` to reality + add a rebuild recipe.** Pin the installed numpy/scipy/matplotlib; add `av`, `laion-clap`, `librosa`, `soundfile`, `yt-dlp`, `pytest`; note ffmpeg as a system prerequisite and the certifi CA-append step in a repo-local SETUP note. *Finding: Lane 5.* Impact: a VM replacement stops being an archaeology expedition.
3. **Restore or retire `--separate-vocals`.** Either reinstall demucs (and pin it) or mark the path deprecated in REVIEW-PROTOCOL §6 and `hva/transcribe.py` help. *Finding: Lane 5.*
4. **Mark superseded outputs.** One-line `SUPERSEDED.md` (or a top-level `output/INDEX.md`) in `ff3/`, `ff3_full/`, and the pre-AV Star Tours dirs pointing at the canonical runs; add the Pink Panther addendum + fix the "Last updated" header in STREAMING-REPORT. *Finding: Lane 6.*
5. **Ship the chance baseline with the binding.** Extend `streaming/bind_av.py` output with the Poisson chance rate for the run's transient densities + a dt histogram summary, so every future `av_binding.json` carries its own context. Small, pure-additive. *Finding: Lane 3.*
6. **Whisper-medium model hygiene.** Drop large-v3's `vocabulary.json` (+ a `preprocessor_config.json` with feature_size 80) into `models/faster-whisper-medium/`, mirroring the base fix, before a ctranslate2 bump breaks the escalation tier. *Finding: Lane 4.*
7. **Wrap the batteries in pytest.** Thin test files that call each `run_all()` and assert all-pass — makes "full suite green" cover T/A/M claims; also add a skip-aware marker for media-dependent tests so a fresh-clone run degrades to skips, not errors. *Finding: Lane 2.*

### (b) Substantive

8. **Pursuit-channel egomotion fix.** FF3: 1,127/1,127 pursuits at exactly minimum duration is a degenerate mode, not tracking. Options: discount coherent full-field flow (the coherence gate exists in `hvp/motion.py` but the streaming driver's pursuit logic admits the flow field), require residual (non-global) motion for pursuit initiation, or raise minimum-duration chaining cost. Validate with sealed predictions on FF3 walking segments vs Star Tours, using the existing three-regime corpus as the test set. *Finding: Lane 3.*
9. **Binding that means something.** Move from coincidence counting toward the dt-clustered subset as the reported metric (e.g., report bound-within-±50 ms separately from the ±250 ms coincidence rate — FF3's genuine candidates were ~159 zero-dt pairs), and consider strength-weighted or causal (audio-after-visual asymmetry) scoring. The reviews already reason this way; the code should produce it. *Finding: Lane 3.*
10. **Content-sensitive saccade clock.** Investigate why the streaming driver's decision rate is near-flat across content regimes (FF3 2.9/s across beats; Pink Panther ~3.3/s bedroom-vs-brawl) when the batch path has a working darkness/dwell modulation. A luminance/activity-modulated decision interval in `online_driver` (port of `darkness_gain`) is the obvious candidate mechanism. *Finding: Lane 3.*
11. **Fixture systematization.** A small fixture table (committed, with measured motion energy + onset density + regime) spanning: motion-rich (Star Tours ✓), dialogue (Dr Tran ✓), egomotion (FF3 ✓), and the missing **quiet/near-silent** regime — plus one short motion-rich clip added to the *test* rotation so the motion pathway can never again hide for weeks. *Finding: Lane 4.*
12. **Full-suite flake hunt — WITHDRAWN.** This item's premise (a one-off, irreproducible failure in the audit's first suite run) was disproven when that run's actual output arrived: both full runs passed 259/259 (see Lane 2's corrected process note). No flake was observed; nothing to hunt. Kept in place, marked withdrawn, rather than silently deleted — the correction is part of the record.
13. **Causal Level-2 gate.** SPEC_FUSION's own recorded open work: replace the noncausal full-sequence 75th-percentile visual-transient gate in `couple_av.py` with a rolling estimator. *Finding: spec-acknowledged.*

### (c) Needs Mykal's decision

14. **Auditory streaming / object formation.** The standing v1 gap named in both SPEC_AUDIO and SPEC_FUSION: the pipeline cannot tell engines from score. Any real step (stem-separation-based streams, learned source models) is a research direction, not a patch — and it interacts with the currently-broken demucs path. His call on whether this is the next mountain.
15. **Mishearing model.** The logprob hooks exist (`hva/transcribe.py`); turning low-confidence verbatim into modeled mishearings changes the epistemics of every review. Design decision, his standing "transcription claims carry logprobs or they don't ship" rule governs.
16. **Live sensors / webcam embodiment.** Parked by his ruling (SPEC_VISION §5) and by hardware terms-first. The streaming harness + `streaming/docs/external-stream-front-door.md` are the seam when he reopens it. No action recommended.
17. **OCR for diegetic signage.** REVIEW-PROTOCOL §6 records signage as a human-read beat source because no OCR exists in the codebase. Adding one is a new capability dependency — his call.

---

## Appendix — verification commands used (selection)

- `git status --short --branch; git log -5` — clean tree, HEAD 9d6eb3d (2026-10-04).
- `.venv/bin/python -m pytest tests/ -q` — run twice, **both 259 passed, 1 warning, exit 0** (199.77 s and 127.99 s test time; see Lane 2's corrected process note on the unreliable interim summary).
- `.venv/bin/python -m hvm.battery` → 6/6; `run_all()` for `hva.battery` → 6/6, `hvp.battery` → 5/5; `.venv/bin/python -m hva.battery` → silent exit 0 (no `__main__`).
- Python reads of `output/ff3_hd_av/{run_report,av_binding,audio_events,audio_interpretation}.json` + `*.npy` shapes/means; same for `st_stream_av_full`, `drtran_ep4_av`, `pink-panther-cato-attacks`.
- `pip list` diffed against `requirements.txt`; `import demucs` (fails); certifi bundle certificate count (487) vs `/run/hatch/egress-tls/ca-bundle.pem` presence.
- Greps: `motion` in SPEC_VISION (0), battery imports in tests (0), `poisson|chance` in all pipeline `.py` (0 — baselines exist only in report prose).
