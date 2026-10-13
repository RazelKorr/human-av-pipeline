# Sealed predictions — Sensorium improvements, 2026-10

Written before the measurement/backfill runs they describe, per the
repo rule (sealed predictions before any sim/measurement run). Each
phase's section is signed and dated when sealed; scoring is appended
after the run, never edited into the predictions themselves.

---

## Phase 2 — binding integrity (sealed 2026-10-12)

Estimator (the report's own, verified by hand against its two
published values before sealing): for a Poisson visual-transient
stream of density λ_v = n_visual / duration, the chance that at
least one transient falls within ±W of a given onset is
1 − exp(−λ_v · 2W). FF3: 2735/2701.0 s → 39.7% (report: 39.7% ✓).
Dr Tran ep4: 601/436.1 s → 49.8% (report: ≈0.50 ✓).

Predictions for the backfill (`scripts/binding_baseline.py` over the
four canonical runs), chance rates ±2 pts:

| Run | n_visual / duration | Predicted chance (±250 ms) | Actual bound | Prediction: bound vs chance |
|---|---|---|---|---|
| FF3 1080p (`ff3_hd_av`) | 2735 / 2701.0 s | **39.7%** | 35.9% | Bound **below** chance (report already says this) |
| Star Tours AV (`st_stream_av_full`) | 328 / 269.0 s | **45.6%** | 38.2% | Bound **below** chance by rate — the causal signal is in the dt distribution, not the rate |
| Dr Tran ep4 (`drtran_ep4_av`) | 601 / 436.1 s | **49.8%** | 50.7% | Bound **≈ chance** (within ~1 pt; report: "equals") |
| Pink Panther (`pink-panther-cato-attacks`) | 260 / 200.1 s | **47.8%** | 52.4% | Bound **exceeds** chance by ~4–5 pts — the only rate-excess of the four |

Tight-dt predictions (fraction of bound pairs with |dt| ≤ 50 ms;
uniform-dt coincidence over a ±250 ms window expects ≈20%):

- **Star Tours shows the strongest tight-dt concentration of the
  four runs**, ≥35% of its bound pairs within |dt| ≤ 50 ms (the
  report's "~zero-dt concentration" / "tight where the film is
  multimodal," made numeric).
- FF3: |dt| ≤ 50 ms fraction ≈ 25–28% — dt is quantized to the
  50 ms moment grid and reported "roughly uniform," so the
  {-50, 0, +50} grid points should hold ≈3/11 of pairs; the
  report's ~159 exactly-zero-dt pairs are 13.2% of 1,208.
- Dr Tran ep4: |dt| ≤ 50 ms fraction ≈ 13% (the report's ~39 pairs
  of 298), i.e. *below* the 20% uniform expectation despite the
  50.7% headline rate — flat dt, density-driven.
- Pink Panther: |dt| ≤ 50 ms fraction ≈ 25% (modest excess over the
  20% uniform expectation, below Star Tours).
- Signed asymmetry: no strong directional prediction except that
  Star Tours' tight pairs skew audio-after-visual (real-world
  sound follows the visible event); FF3/Dr Tran near-symmetric.

Sealed: Wodehaus, 2026-10-12 (before `binding_baseline.py` was run
on any canonical run).
