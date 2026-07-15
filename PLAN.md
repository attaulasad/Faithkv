# Plan and status

## Current status: implementation complete; GPU validation pending

This repository was built end-to-end on a CPU-only, no-GPU machine per the
original build brief. Every module, test, config, and doc listed in that
brief's repository layout exists and (where CPU-runnable) passes. No GPU
code has been executed; no model weights were downloaded.

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
- GPU test suite (`tests/integration/*_gpu.py`): implemented in full,
  marked `@pytest.mark.gpu`, verified to auto-skip cleanly on this machine.
  Never run.
- All docs: `UPSTREAM_AUDIT.md`, `REPLAY_DESIGN.md`, `EXPERIMENT.md`,
  `PROBE_PROTOCOL.md` (real tokenizer output), `SCHEMA.md`,
  `REPRODUCIBILITY.md`, `GPU_VALIDATION_PLAN.md`.
- `--dry-run` exercised for every stage config against every relevant
  condition — the only end-to-end check possible without a GPU.

## What's next (in order — see docs/GPU_VALIDATION_PLAN.md for exact commands)

1. Rent a GPU host, run `scripts/setup_vast.sh` + `verify_environment.sh`.
2. Run the four `@pytest.mark.gpu` test files, in the order listed in
   `docs/GPU_VALIDATION_PLAN.md`. Do not skip ahead of replay identity
   (`test_replay_gpu.py`) — nothing downstream is meaningful until it
   passes, per `docs/REPLAY_DESIGN.md` §5's assumption list.
3. Stage 0 (smoke) → read its throughput extrapolation before continuing.
4. Stage 1A (measurability) → read `recommendation` before continuing.
5. Stage 1B (calibration) → manually review and fill in
   `configs/selected_operating_point.yaml` from the real decision JSON.
6. Stage 2 (main pilot, n=200, 3 seeds) → the actual result.

## Open decisions needing human input

- **License.** Not chosen. See `README.md`.
- **Stage 1A outcome** determines whether Stage 2 runs on GSM8K or the
  frozen MATH-500 backup — this cannot be known before Stage 1A actually
  runs on GPU.
- **Stage 1B outcome** determines the Stage 2 operating point, or may
  determine that no accuracy-plausible, compression-active budget exists
  for this pilot on GSM8K at all — in which case Stage 2 does not run as
  currently configured, and that itself is a reportable finding.

## Changes to frozen settings

None yet. Any future change to `configs/lock.yaml`'s frozen values requires
a dated entry in `CHANGELOG.md` first, per the build brief.
