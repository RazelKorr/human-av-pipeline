# Review — commission 2: the repairs, then six builds (2026-10-14)

Commission: run the defrag review's actionable left-open
repairs (R1–R3), then implement six discretionary builds
(W1–W6) chosen in advance by the parent agent, full chain:
baseline → repairs → builds → validation → this review →
commit/push. Predictions for the repairs were sealed before
any run (`logs/wild-2026-10-14/SEALED-PREDICTIONS.md`); every
number below is copied from a captured file under
`logs/wild-2026-10-14/` (the repo's `logs/` gitignore leaves
them untracked, the standing convention — they are on disk at
the cited paths).

**Baseline** (`pytest_baseline.log`, collected before any
change): **307 passed, 1 skipped** of 308 collected. The
defrag's final was 308 passed; the delta is one
environment-conditional skip, not a regression — the final
log names it: `tests/test_ocr.py`'s tesseract condition
(this VM has no tesseract binary installed; the OCR seam's
test skips by design when the binary is absent, and the
baseline's single skip is the same test by count and
condition).
**Final** (`pytest_final.log`): **318 passed, 1 skipped** —
319 collected, exactly the baseline's 308 plus this
commission's 11 new tests (R1 1, W1 4, W2 2, W3 1, W5 1,
W6 2), no pre-existing test modified or lost.

---

## Part A — the repairs

### R1 — visual_transients: indicted by its sealed fixture, fixed

*Why:* the defrag left the visual picker untouched "on
suspicion alone" — its global median+MAD bar has the same
unscaled-MAD shape that made the ear false-alarm (PX.1), but
suspicion is not evidence. This item bought the evidence.

*What/evidence:* a sealed synthetic fixture (stationary noisy
energy bed, σ=0.05, 10 planted transients, 2,400 moments).
Pre-fix: **41 picks, 31 false alarms, planted 10/10**
(`r1_prefix.log`) — the PX.1 signature exactly: a fixed
tail-exceedance count set by the bed's duration. PR1.1 sealed
≥20 total / ≥10 false: **indicted**. The fix is the ear's own:
the MAD is scaled by a named constant
`VISUAL_MAD_TO_SIGMA = 1.4826`, so k=4.0 means robust sigmas
as documented. Post-fix: **11 picks, 1 false alarm, planted
10/10** (sealed ≤13 / ≤3 / 10/10 — PR1.2 hit). Corpus check:
the dark-busy run of record reproduces its filed 56
transients exactly under the old bar and reads **53** under
the fixed one (sealed: down, ≥20 — PR1.3 hit; the real motion
survives, the bed noise doesn't). A regression test pins the
fixture in the suite. Transient counts across this boundary
are not comparable, the same caveat class as PX.1, and the
docstring says so.

### R2 — Star Tours / Dr Tran recounts under the fixed ear: filed

*Why:* the defrag recounted FF3 and YTPMV but left the other
two corpus verdicts standing on counts produced by the
indicted 2σ bar. If the corpus numbers are ever republished,
they should all be from the same ear.

*What/evidence:* identical method to the defrag recounts
(stored features → current picker → stored visual stream →
bind). The script first reproduces FF3's filed 2,409 from
stored features exactly (methods check: MATCH) before its
own output is trusted. Results (`r2_recount.log`,
filed alongside the earlier recounts as an append-only
addendum to the defrag review): Star Tours onsets
**238 → 103**, bound 91 → 37, shift-null tight percentile
**81.0**; Dr Tran ep4 onsets **588 → 377**, bound 298 → 199,
tight percentile **87.0**. Sealed scoring: PR2.1 **partial**
— both counts dropped as sealed, Dr Tran landed in its band
(350–520), Star Tours dropped past its band (103 < 140): the
fixed ear is stricter on Star Tours' feature stream than the
seal allowed, recorded as a band miss, not hidden. PR2.2
**hit** — both verdicts unchanged: at chance, as filed under
both earlier bars. The recount changes the counts; it does
not change a single corpus conclusion, which is itself the
finding worth filing.

### R3 — adaptive-clock adjudication: adaptive_td retired

*Why:* two experiments had failed to adjudicate the top-
decile clock because Pink Panther's regimes differ by only
~1.7x under any whole-frame statistic. The named remedy was
fixtures whose motion differs by an order of magnitude.

*What/evidence:* quiet room (motion-energy mean 0.000000)
vs a new generated fixture, `fill_busy` (seeded random-dot
field scrolling diagonally — motion at every pixel), plus
dark_busy as the named pair. Predictions sealed with a
disposition rule: hit → graduate the statistic; miss →
retire it. Results (`r3_scores.log`): fixed controls flat
(3.317 vs 3.317/s, divergence 0.000 — PR3.2 hit).
adaptive_td: quiet 1.567/s, fill 2.300/s (divergence
**+0.733**, sealed ≥ +2.0 — PR3.1 **miss**); the named pair
went backwards (dark 1.433/s vs quiet 1.567/s, **−0.133** —
PR3.3 **miss**); td_dark's 1.433/s also dips a hair under the
clock's nominal 1.5/s floor on frame-quantized decision
times (PR3.4 narrow miss, mechanism named in the addendum).

Two honest complications, both filed in the EXPERIMENTS
addendum. First, `fill_busy` failed its own sealed
calibration: its driver motion-energy mean is **0.009252**,
statistically dark_busy's 0.009238, not the sealed ≥0.05 —
a uniform scroll is global translation, which is exactly
what the magno channel's coherence gate exists to remove.
Second, the pattern across all four regime pairs is
consistent and mechanistic: the decile tracks the motion
map's frame *coverage* (fill's gate-leakage is frame-wide;
dark_busy's stronger motion covers 3% of the map and never
reaches the 90th percentile). Per the sealed rule,
**adaptive_td is retired**: marked deprecated in the driver
and the run help, kept runnable so the sealed runs stay
reproducible, default stays fixed. The mean-statistic
`adaptive` clock is untouched — its Exp (ii) separations
stand.

---

## Part B — the six builds

### W1 — Provenance stamping

*Why:* every forensic episode in this project (PX.1's counts,
the Poisson label) began the same way: a filed number whose
producing code state had to be reconstructed from commit
archaeology. Provenance makes the reconstruction unnecessary
by writing the code state into the artifact at birth.

*What:* `streaming/provenance.py` — a record carrying the
schema tag, UTC timestamp, git commit + dirty flag, the run
config and its sha256, the ear/eye/binder version constants
(read from the modules' own constants and signatures, never
restated, so the record cannot drift from the code), and the
seeds in force. run_stream stamps it into `av_binding.json`
and `run_report.json` and writes a `provenance.json`
manifest with a sha256 per artifact; `binding_baseline`
sidecars and `sensorium bind` output carry it too.

*Validation* (`w1_validation.log`): a live 10 s run stamps
commit `20e1092` (HEAD at the time), honestly reports
`dirty: true` (W1 itself was uncommitted during its own
validation), and all 9 artifact hashes verify against the
files on disk; the report's embedded record and the manifest
agree. 4 unit tests. *Not* backfilled: old artifacts cannot
honestly acquire a record after the fact, and none did.

### W2 — One front door: `python -m sensorium`

*Why:* the pipeline had a dozen doors and the map between
them was oral history — flagship in streaming/, backfills in
scripts/, and no sign anywhere. A front door is a
streamlining of *arrival*, not of code: every subcommand
delegates to the existing tool through its own parser,
unchanged, and the old entry points keep working.

*What:* `sensorium` package; subcommands run / bind /
baseline / audit / golden / fixtures / percept, with
top-level help written for a human arriving cold (what the
Sensorium is, the workflow order, where each door leads).
`bind` is the one new surface: run_stream's Phase 2 as a
standalone recompute over a stored run's artifacts, with
provenance stamped. Dispatch is manual, deliberately — the
first cut used argparse REMAINDER subparsers, which eat a
delegate's leading `--flags` (`sensorium fixtures --kind`
failed); a front door should not need `--` escapes, and now
it doesn't.

*Validation:* every subcommand smoke-tested live — help,
golden (match), audit (pass), bind (rebind of the quiet-room
run with provenance), baseline through the door (sidecar
carries W1 provenance), fixtures through the door (mp4
written), run --help (the delegate's own help) — plus 2 CLI
tests including a synthetic bind.

### W3 — Golden-fixture regression pack

*Why:* PX.1's lesson was that the flagship numbers can move
silently when a shared constant changes, and the discovery
cost is a forensic investigation months later. A frozen,
tiny, fast pack moves that discovery to the moment of the
change, in the suite, in under a second.

*What:* `tests/golden/` — three seeded canonical cases
(ear: waveform → features → picks; eye: the R1 bed →
transients; binder: literal streams → full bind() output,
shift null included) with frozen expected JSON;
`run_golden.py` checks or (`--bless`) re-freezes — re-
blessing is a deliberate act that belongs in the commit
message of whatever changed the math. Wired into the suite
(`tests/test_golden.py`) and behind `sensorium golden`.

*Validation:* blessed at creation, then checks green
(`w3_test.log`: 1 passed in 0.48 s; all three cases MATCH).
Note the ordering honestly: the pack was created after R1,
so its eye case freezes the post-R1 bar — it did not catch
R1 (nothing in this commission needed catching; R1 was the
change, made openly). What it changes is the NEXT bar
change: that one will fail this test in the suite run where
it lands, and re-blessing it will require saying so in a
commit message.

### W4 — Self-audit command

*Why:* every correction in the corpus's history was found by
a human re-deriving a filed number from stored artifacts,
by hand, weeks late. The spot-check that kept saving the
project deserved to be a command.

*What:* `scripts/self_audit.py` (behind `sensorium audit`):
recomputes binding summaries from stored pairs for every
stored run and compares against the filed fields (21 runs);
recomputes the recount registry (FF3 2,409 / Star Tours 103
/ Dr Tran 377 / YTPMV 212 / dark-busy 53) from the stored
features themselves and checks the citing docs; verifies the
STREAMING-REPORT carries its 2026-10-14 correction
addendum. It reports and edits nothing; a FLAG is a human's
cue, with both values printed side by side.

*Validation* (`w4_audit.log`): first full run — **all checks
pass**, exit 0. Two limitations stated plainly: citation
checks are substring presence (the recompute beside them is
the substantive layer), and runs with zero onsets pass the
consistency check vacuously (bound_rate None on both sides,
which is at least verified to be None on both sides).

### W5 — Doubt track

*Why:* the mishearing model existed, worked, and was wired
into nothing — its annotations lived in a side file a review
might never open, in a project whose founding review
failure was a claim that outran its evidence. Doubt that
doesn't travel with the claim isn't doubt, it's decoration.

*What:* `hva.mishear.build_doubt_track` packages the
modeled annotations as one labeled track object (model tag
and disclaimer at its head), entries parallel to the
segments by index, verbatim byte-identical by construction.
run_stream's `onset_transcripts.json` now carries the track
per transcription window, inside the standard artifact — and
the word probabilities the model needs travel with the
segments now, too. REVIEW-PROTOCOL.md gains the doubt-track
pass: no claim rests on a flagged word without the flag
being stated.

*Validation* (`w5_tests.log`, `w5_validation.log`): 9
mishear tests including the track's parallelism/labeling
invariants; a live Star Tours run wrote 2 windows whose
doubt tracks are parallel to their segments — and both
windows' base-tier segments were in fact modeled (2/2),
which on ride audio is the expected, honest outcome.

### W6 — Determinism pass

*Why:* sealed predictions are only falsifiable if a re-run
can reproduce the run they describe. Every stochastic path
therefore takes an explicit seed, recorded where W1's
provenance can see it.

*What:* the inventory found most paths already seeded (the
shift null — seed recorded in its output and, since W1, in
provenance; the fixture generators' lavfi/RNG seeds; the
golden cases). The gap was `hva.battery`'s noise stimuli,
which drew from the global RNG and were reproducible only
by arriving through `run_all`'s seeding: `click` and
`room_tone` now take `seed=` (None preserves the historical
global-RNG behavior exactly, so filed battery
characterizations do not move).

*Validation* (`w6_double_run.log`): double-runs are
byte-identical across every class — bind() on FF3's stored
streams (shift null included), the seeded battery stimuli,
all three golden cases, and a full media double-build of
`fill_busy_60s.mp4` (same sha256 for the encoded file).
Plus 2 determinism tests in the suite.

---

## Left open

1. **The mean clock on a true order-of-magnitude fixture.**
   R3 could not ask its sharpest question: no fixture yet
   produces a DRIVER-statistic motion energy ≥10x the quiet
   room's, because global motion is gated out and local
   movers cover too little frame. Such a fixture needs
   incoherent local motion at frame scale (confetti, crowds,
   rain). If built, test the mean `adaptive` clock on it
   first — the decile's question is closed.
2. **Self-audit citation depth.** Recounts are recomputed
   (substantive); citations are substring checks (coarse). A
   machine-readable citation registry in the docs would let
   W4 verify figures in context, not just presence.
3. **Provenance coverage.** New runs only, by design and by
   honesty. The filed corpus stays traceable through its
   reviews, which is the gap W1 exists to close going
   forward.
4. **Model-internal stochasticity.** Whisper decoding and
   CLAP's random truncation are not seedable from this
   codebase; the determinism contract covers the pipeline's
   own paths and names this boundary rather than hiding it.
5. **Doubt-track coverage.** Windows whose transcription
   path supplies no word probabilities yield unmodeled
   entries by construction (no logprob, no claim — the
   standing rule). The track is only as wide as the
   transcriber's word timestamps.
6. **Fill_busy's table row.** FIXTURES.md records its
   measured motion energy (0.009252) with the proxy column
   unmeasured; the regime table's lesson — the two motion
   columns measure different things — now has its sharpest
   example, and the addendum in EXPERIMENTS-2026-10-13
   carries the mechanism.
