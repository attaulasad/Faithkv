# Plan and status

## Current status: B0 method-pivot gate ran and FAILED — METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET

**Phase B0 (2026-07-19, CHANGELOG.md) ran the method-pivot specification
and adversarial method-novelty gate** on three candidate methods targeting
a prospective causal-false-negative failure mode (`docs/METHOD_PIVOT_SPEC.md`
§5 — an untested hypothesis, not a finding): M1 residual causal-utility
protection, M2 interaction-aware dynamic rescue, M3
faithfulness-constrained memory allocation. Against a 2026-07-19-cutoff
adversarial search (`docs/METHOD_NOVELTY_MATRIX.md`,
`docs/B0_SEARCH_LOG.md`, `docs/method_novelty_matrix.json`):

- **M1: PARTIAL — INSUFFICIENT METHOD NOVELTY** (ablation supervision →
  ArborKV arXiv:2605.22106; residual-correction architecture → IntentKV
  arXiv:2606.09916; counterfactual-ablation-to-policy pipeline → ThinKV
  arXiv:2510.01290v2; protected partition → CASK/VaSE/arXiv:2605.18053);
- **M2: KILLED** (cache-state-conditional sequential eviction is
  ForesightKV arXiv:2602.03203's MDP and Neural Garbage Collection
  arXiv:2604.18002; R-KV itself already rescores per compaction);
- **M3: PARTIAL — INSUFFICIENT METHOD NOVELTY** (allocation machinery →
  ReasonAlloc arXiv:2606.11164 and the Ada-KV/LKV lineage; a new
  constraint metric is insufficient by the predeclared standard).

**METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET. B1 is not
permitted; no GPU, no MATH-500 work, no method implementation is
authorized.** The only permitted next activity remains method design: a
future candidate must contain a cache operation absent as an operation
class from `docs/METHOD_NOVELTY_MATRIX.md` §5, then re-clear a fresh
B0-style gate.

## Prior status: A3 diagnostic novelty kill-check DOES NOT SURVIVE; PHASE B BLOCKED

**Phase A3 (2026-07-19, CHANGELOG.md) found that CASK (arXiv:2604.10900),
released 2026-04-13, independently implements the fixed-generated-trace /
teacher-forced / cache-policy-varying replay diagnostic this repository's
narrower novelty claim (N1) rested on — confirmed by direct inspection of
CASK's official evaluation code, not just its abstract
(`docs/RELATED_WORK_MATRIX.md` §6.1, §8).** Early answering itself is
independently non-novel since Lanham et al. 2023 (arXiv:2307.13702). No
paper was found that combines both (KV-cache-policy replay + early-
answering/omitted-suffix intervention) under an accuracy-neutral gate with
realized-memory matching and held-out per-example mechanism classification —
that specific empirical intersection remains open, but per the project's
predefined rule it is an application of known ingredients, not a standalone
method contribution, so the overall verdict is still negative:

**DIAGNOSTIC SURVIVAL VERDICT: DOES NOT SURVIVE — PHASE B: BLOCKED —
DIAGNOSTIC NOT NOVEL.** Full matrix: `docs/RELATED_WORK_MATRIX.md`; search
log: `docs/A3_SEARCH_LOG.md`; machine-readable: `docs/related_work_matrix.json`.

This is layered on top of, and does not reverse, the pre-existing GSM8K
b128 status below — the operating point was already retired on independent
(accuracy) grounds before this literature check ran.

## Prior status (2026-07-19): protocol-v3 GSM8K b128 gate FAILED; GSM8K b128 retired; hypothesis `not_tested`

The implementation is complete and its GPU correctness gates have passed, but
the pilot has **not** reached the §1 research question. The protocol-v3
natural R-KV accuracy gate ran on the full 50-pair GSM8K calibration manifest
and failed: FullKV answered 33/50 (66%) correctly, natural R-KV b128 13/50
(26%) — a 40pp drop past the 0.10 pilot ceiling
(`results/decisions/early_gap_v3_b128_accuracy_gate.json`: `gate_passed:
false`). The fixed-trace analysis path exited before computing any PSS/CPSS,
so **no protocol-v3 PSS/CPSS decision exists and `hypothesis_status` remains
`not_tested`** — the research hypothesis is neither supported nor refuted; it
has not been tested.

The GSM8K + `DeepSeek-R1-Distill-Qwen-1.5B` + b128 operating point is
**retired** as structurally unviable — FullKV traces on this manifest run
276–847 generated tokens (median ~440), leaving no fixed budget that is both
accuracy-plausible and meaningfully compressing. No further GSM8K b128/b160
runs are planned.

Full detail and provenance live in the docs updated alongside this entry:
`README.md`, `CHANGELOG.md` (2026-07-19), `docs/EXPERIMENT.md` §11, and
`docs/GPU_VALIDATION_PLAN.md` (2026-07-19 note). This file is the roadmap
summary; those are the source of truth for the numbers.

## Development model

This repository is developed and maintained on a CPU-only, no-GPU machine
(`pytest -m "not gpu" tests/`, `--dry-run`). GPU-dependent work runs on a
rented host and is synced back as committed artifacts. GPU code *has* now been
executed on such a host — the correctness gates, the protocol-v2 fixed-trace
screen (returned `screen_valid=false`), and the failed protocol-v3 natural
accuracy gate above — so the earlier "no GPU code has been executed" status is
obsolete.

## What's done

- Upstream audit (`docs/UPSTREAM_AUDIT.md`): H1-H8 confirmed with
  file:line citations, plus grounding for `mix_lambda`/`retain_ratio`/
  `retain_direction`, plus the `retain_ratio` inertness finding.
- Full package (`src/kvcot/`): schemas, config, data/manifest freezing
  (real GSM8K + MATH-500 data actually downloaded), answer extraction,
  think-span parsing, metrics/stats (sign convention, Pratt zeros, f=0/f=1
  exclusion), generation engine (state reset, sampling, policies, decode,
  provenance, replay), analysis (summaries, plots), CLI, runtime.
- CPU test suite: passes in full (see the build report for the exact
  count).
- GPU correctness gates: **passed** on a rented host — `test_replay_gpu.py`
  (all seven cases), `test_patched_noop_parity_gpu.py`,
  `test_no_state_leak_gpu.py`, determinism and compaction
  (`logs/gpu_validation/*.log`). The §10 f=1 probe-stability control
  (`test_probe_stability_gpu.py`) is the exception — it remains **UNRESOLVED**
  under the corrected validity definition (`docs/GPU_VALIDATION_PLAN.md`).
- Pilot screens run on GPU: protocol-v2 fixed-trace screen
  (`screen_valid=false`, `hypothesis_status=not_tested` — a valid negative
  screening outcome) and the protocol-v3 natural accuracy gate (FAILED, above).
- **Phase A2 — failure atlas (2026-07-19, CHANGELOG.md).** Deterministic,
  tested, CPU-only atlas over the 50 committed protocol-v3 gate pairs
  (`kvcot failure-atlas`, `src/kvcot/failure_atlas.py`):
  `results/tables/gsm8k_v3_b128_failure_atlas.{csv,md}`,
  `results/decisions/gsm8k_v3_b128_failure_atlas_summary.json`. Headline
  recomputation matches the prior manual analysis exactly (0/50 diverge
  before first compaction; 9/50 identical through `</think>`, 3 of those
  flip correct→wrong: rows 30, 271, 1115) and adds a new finding: 41/50
  pairs first diverge *inside* the reasoning span itself, so the
  identical-through-think flip is the minority pattern, not the typical
  one, at this retired operating point. Still `post_hoc_diagnostic` /
  `hypothesis_status: not_tested` — this does not test the §1 hypothesis.
- **Phase A3 — adversarial literature matrix and diagnostic novelty
  kill-check (2026-07-19, CHANGELOG.md).** `docs/RELATED_WORK_MATRIX.md`,
  `docs/A3_SEARCH_LOG.md`, `docs/related_work_matrix.json` (20 papers,
  schema-validated). CASK (arXiv:2604.10900) independently implements this
  repository's core fixed-trace/teacher-forced replay diagnostic primitive
  (confirmed against its official evaluation code); Lanham et al.
  (arXiv:2307.13702) independently established early answering. **DIAGNOSTIC
  SURVIVAL VERDICT: DOES NOT SURVIVE — PHASE B: BLOCKED — DIAGNOSTIC NOT
  NOVEL.** A specific empirical intersection (KV-cache replay + early
  answering + accuracy gate + held-out per-example classification) remains
  unstudied but is not, by itself, a new method.
- **Phase B0 — method pivot specification and adversarial method-novelty
  gate (2026-07-19, CHANGELOG.md).** `docs/METHOD_PIVOT_SPEC.md`,
  `docs/METHOD_NOVELTY_MATRIX.md`, `docs/B0_SEARCH_LOG.md`,
  `docs/method_novelty_matrix.json` (25 records, schema-validated).
  Documentation-only; no code/config/test/schema/result touched, no GPU,
  no inference. Outcome: M1 PARTIAL, M2 KILLED, M3 PARTIAL — **METHOD
  PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET** (see Current status).
- All docs: `UPSTREAM_AUDIT.md`, `REPLAY_DESIGN.md`, `EXPERIMENT.md`,
  `PROBE_PROTOCOL.md` (real tokenizer output), `SCHEMA.md`,
  `REPRODUCIBILITY.md`, `GPU_VALIDATION_PLAN.md`.

## What's next (CPU-only; no new GPU rental; Phase B/MATH-500 blocked)

1. ~~**Failure atlas** over the existing 50 gate pairs~~ — **done, 2026-07-19**
   (Phase A2 above).
2. ~~**Literature matrix** situating this negative pilot result against prior
   faithfulness / KV-compression work.~~ — **done, 2026-07-19** (Phase A3
   above): **DOES NOT SURVIVE**, Phase B blocked under the diagnostic's old
   novelty story.

**MATH-500 implementation and any other Phase B work remain BLOCKED** —
not merely "not yet started" — until a genuinely new technique is designed
and approved; the current diagnostic combination (fixed-trace replay +
early answering + KV compression) is not, by itself, that new technique
(`docs/RELATED_WORK_MATRIX.md` §16).

That design phase has now run once: **Phase B0 (2026-07-19) evaluated
three candidate methods and returned BLOCKED — NO NOVEL METHOD YET** (see
Current status above). **B1 is not permitted, even in principle, under
this B0 result** — it would have required at least one SURVIVES
PROVISIONALLY candidate, and there is none. B1 has not started. The next
activity remains further method design (CPU/paper-only), targeting a cache
operation absent as a class from `docs/METHOD_NOVELTY_MATRIX.md` §5,
followed by a fresh novelty gate. **No GPU experiment is authorized**, and
no MATH-500 manifest, config, evaluator, or script may be created until a
redesigned candidate passes such a gate and is separately approved. The §10
f=1 stability control remains UNRESOLVED and was not a B0 task; the GSM8K
b128 operating point remains retired.

3. **Phase C — GPU rental.** No new GPU host is rented until a redesigned,
   non-retired, genuinely-novel experiment is specified and approved. The
   retired GSM8K b128 operating point is not re-run, and Phase C does not
   begin before a design phase addressing the A3 verdict is complete.

## Open decisions needing human input

- **License.** Not chosen. See `README.md`.
- **Whether to pursue MATH-500 at all, and under what redesigned method.**
  The old "Stage 1A decides GSM8K vs MATH-500" decision was already moot
  (GSM8K b128 retired on accuracy grounds); Phase A3 adds a second, deeper
  reason a MATH-500 rerun of the SAME diagnostic would not be worth GPU
  spend even if GSM8K accuracy had passed — the diagnostic combination
  itself does not clear the novelty bar. Any MATH-500 work needs both the
  fresh feasibility design AND a design response to the A3 gap, not the
  current frozen configuration.
- **§10 f=1 stability control.** UNRESOLVED, and a separate Stage-0
  prerequisite that any future non-retired stage must clear on its own terms
  (`docs/GPU_VALIDATION_PLAN.md`, 2026-07-19).
- **What the genuinely new technique should be.** Still open. Phase B0
  (2026-07-19) tested three candidates against the literature and none
  survived (M1 PARTIAL, M2 KILLED, M3 PARTIAL —
  `docs/METHOD_PIVOT_SPEC.md`). This remains the single open question
  blocking any Phase B successor; the B0 matrix's operation tables
  (`docs/METHOD_NOVELTY_MATRIX.md` §5–§7) are the map of what is already
  taken.

## Changes to frozen settings

None. Retiring an operating point changes no frozen §1/§4/§8/§9 value in
`configs/lock.yaml`; the retirement is recorded in `CHANGELOG.md`
(2026-07-19) as a documentation-only status update. Any future change to
frozen values still requires a dated `CHANGELOG.md` entry first, per the
build brief.
