# B2A-R3 Production Selected-Row Freezer Audit Repair

Date: 2026-07-24

Failed audit report:
`/workspace/faithkv-b2a-r3-freezer-implementation-independent-audit.md`

Failed audit SHA-256:
`dccd528c8bb10586f260962dba8a38b24750b5b9ee7dc3cc2e717af2f84cff7d`

Target implementation SHA:
`56d257236874d8947d5f127bf2074824cff62395`

Repair authorization SHA:
`e5fbee6a1b8da9d7c36bf1caaf39128477083eb3`

Authorization document:
`docs/B2A_R3_PRODUCTION_SELECTED_ROW_FREEZER_AUDIT_REPAIR_AUTHORIZATION_2026-07-24.md`

## Defects And Repairs

1. Full-history CI:
   `.github/workflows/cpu-tests.yml` now checks out with `fetch-depth: 0`
   and explicitly proves the repository is non-shallow and that the
   required ancestor commits exist. The shallow-checkout skip helper was
   removed from `tests/unit/test_cli_b2a_r3.py`; the three real-evidence
   dry-run tests now execute and pass.

2. Tokenizer identity:
   `construct_production_freeze_plan` no longer accepts
   `tokenizer_repository`, `tokenizer_requested_revision`,
   `tokenizer_resolved_revision`, `tokenizer_local_path`, or
   `tokenizer_renderer`. The public API derives tokenizer identity
   internally through the production resolver boundary and binds it with a
   module-private verification token before constructing a publishable
   plan.

3. No-Torch tokenizer boundary:
   `src/kvcot/discovery/b2a_r3_tokenizer_worker.py` renders prompts in a
   fresh subprocess under `USE_TORCH=0`, `USE_TF=0`, `USE_FLAX=0`,
   `CUDA_VISIBLE_DEVICES=`, `HF_HUB_OFFLINE=1`, and
   `TRANSFORMERS_OFFLINE=1`. The child installs an import guard that
   refuses `torch` imports before importing Transformers, verifies no
   `torch` module exists at start or exit, and returns validated prompt
   identity to the parent.

4. No model-weight access:
   The tokenizer worker guards `builtins.open`, `io.open`, and `os.open`
   against model-weight and model-index names, including `*.safetensors`,
   `*.bin`, `*.pt`, `*.pth`, `model.safetensors.index.json`, and
   `pytorch_model.bin.index.json`. Existing model files in the shared
   repository snapshot are not a failure by presence; opening them during
   tokenizer rendering is.

5. Crash recovery and concurrency:
   `publish_production_freeze` now holds an exclusive `fcntl.flock` lock
   under `.git` across temporary-file reconciliation, Git/worktree safety,
   state classification, publication, reload, and final provenance
   verification. Recognized freezer-owned temporary files are reconciled
   only when they are regular non-symlink files in the exact output
   directory, match the expected payload bytes, and the surrounding final
   output state is recognized. Provenance temp aliases left after
   `os.link` must be the same hard-link inode/device as the final
   provenance target. Unknown names, unknown bytes, symlinks, multiple
   leftovers, and conflicts refuse without deletion.

## Local Evidence

Local validation:

- `python -m compileall -q src tests`
- `python -m pytest --collect-only -q`: `1918 tests collected`
- `python -m pytest -q tests/unit/discovery/test_b2a_r3_freeze.py -q`
- `python -m pytest -q tests/unit/test_cli_b2a_r3.py -q`
- Required real-evidence trio:
  `3 passed in 3.69s`, zero skipped.
- Focused hostile/no-Torch/crash/concurrency set:
  `15 passed`, zero skipped.
- Full CPU-safe suite:
  `1904 passed, 14 deselected, 0 skipped, 5 warnings in 140.60s`.
- `git diff --check`: clean.
- `kvcot verify-b2a-r3-candidates`: passed.
- `kvcot verify-b2a-r3-qualification`: passed.
- `kvcot freeze-b2a-r3-selected-row --dry-run`: passed with
  `would_freeze=True`, selected ordinal `1`, selected unique ID
  `test/number_theory/631.json`, no tokenizer execution, and no writes.

The real no-Torch subprocess test rendered the accepted selected row and
matched the qualification prompt hash while reporting:

- no `torch` modules at child start;
- no `torch` modules at child exit;
- empty `CUDA_VISIBLE_DEVICES`;
- no model-weight file open attempts.

## Production Output State

The production freezer was not executed. The production selected manifest
still contains the historical row `test/number_theory/820.json`, and
`results/decisions/b2a_r3_selection_provenance.json` remains absent.
Stage C, FullKV, R-KV, and B2B remain blocked.
