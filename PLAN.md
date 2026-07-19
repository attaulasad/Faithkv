# Plan and status

## Current status: protocol-v3 GSM8K b128 gate FAILED; GSM8K b128 retired; hypothesis `not_tested`

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
- All docs: `UPSTREAM_AUDIT.md`, `REPLAY_DESIGN.md`, `EXPERIMENT.md`,
  `PROBE_PROTOCOL.md` (real tokenizer output), `SCHEMA.md`,
  `REPRODUCIBILITY.md`, `GPU_VALIDATION_PLAN.md`.

## What's next (CPU-only; no new GPU rental until Phase C)

The immediate next work is **CPU-only** and uses only the committed gate
artifacts — no new GPU generation:

1. **Failure atlas** over the existing 50 gate pairs (per-pair divergence,
   compaction, retention, and correct→wrong flips), built from the committed
   `results/gate_artifacts/early_gap_v3_b128_*.jsonl.gz`.
2. **Literature matrix** situating this negative pilot result against prior
   faithfulness / KV-compression work.

Only after those, and still on paper / CPU:

3. A MATH-500 longer-trace feasibility **design**, with separate calibration
   and held-out manifests — a redesign, not the current frozen config re-run.

4. **Phase C — GPU rental.** No new GPU host is rented until a redesigned,
   non-retired experiment is specified and approved. The retired GSM8K b128
   operating point is not re-run, and Phase C does not begin before steps 1–3
   are complete.

## Open decisions needing human input

- **License.** Not chosen. See `README.md`.
- **Whether to pursue MATH-500 at all.** The old "Stage 1A decides GSM8K vs
  MATH-500" decision is now moot — the failed natural gate retired GSM8K b128
  outright. MATH-500, if pursued, needs the fresh feasibility design in step 3
  above, not the current frozen configuration.
- **§10 f=1 stability control.** UNRESOLVED, and a separate Stage-0
  prerequisite that any future non-retired stage must clear on its own terms
  (`docs/GPU_VALIDATION_PLAN.md`, 2026-07-19).

## Changes to frozen settings

None. Retiring an operating point changes no frozen §1/§4/§8/§9 value in
`configs/lock.yaml`; the retirement is recorded in `CHANGELOG.md`
(2026-07-19) as a documentation-only status update. Any future change to
frozen values still requires a dated `CHANGELOG.md` entry first, per the
build brief.
