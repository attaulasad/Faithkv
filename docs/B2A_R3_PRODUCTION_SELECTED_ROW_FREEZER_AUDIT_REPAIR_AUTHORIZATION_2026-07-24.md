# B2A-R3 Production Selected-Row Freezer Audit Repair Authorization

Date: 2026-07-24

Repository: `asad073-ui/Faithkv`
Branch: `research/b2a-r3-runtime-qualified-calibration`
Starting SHA: `56d257236874d8947d5f127bf2074824cff62395`
Target implementation SHA under audit: `56d257236874d8947d5f127bf2074824cff62395`
Failed audit report: `/workspace/faithkv-b2a-r3-freezer-implementation-independent-audit.md`
Failed audit SHA-256:
`dccd528c8bb10586f260962dba8a38b24750b5b9ee7dc3cc2e717af2f84cff7d`

This document authorizes a bounded CPU-only repair of independently
audited defects in the B2A-R3 production selected-row freezer. It
supersedes nothing scientific and authorizes no production freezer
execution.

## Blocking Findings

The failed independent audit identified four required repairs:

1. Exact-SHA CI was green, but the three real-evidence freezer dry-run
   tests skipped because GitHub Actions used a shallow checkout.
2. `construct_production_freeze_plan` accepted caller-supplied production
   tokenizer repository, requested revision, resolved revision, and local
   path metadata, then produced a construction-token-bearing plan.
3. The production tokenizer rendering boundary was not clean on this host:
   rendering through `AutoTokenizer` caused `torch`/`torch.cuda` modules to
   enter `sys.modules`, and the local snapshot contains model-weight files
   that require a proof they are not opened during tokenizer rendering.
4. A crash after no-clobber provenance hard-link publication and before
   temporary-file removal could leave a recognized freezer temporary file
   that blocks the next invocation at the worktree safety guard.

## Authorized Repairs

This repair authorizes exactly:

1. Full-history CPU CI for the real-evidence tests.
2. Removal of the shallow-checkout skip behavior.
3. Internal derivation of production tokenizer identity from a verified
   resolver result.
4. A tokenizer-rendering boundary that imports no Torch and initializes no
   CUDA.
5. Proof that no model-weight file is opened during tokenizer rendering.
6. Safe reconciliation of recognized crash-leftover temporary files.
7. Process locking for concurrent freezer publication.
8. Corresponding CPU tests and documentation.

Permitted source and test paths are limited to:

- `.github/workflows/cpu.yml`
- `src/kvcot/discovery/b2a_r3_production_tokenizer.py`
- a new tokenizer worker module if required
- `src/kvcot/discovery/b2a_r3_freeze.py`
- `src/kvcot/cli.py` only if required
- `tests/unit/discovery/test_b2a_r3_freeze.py`
- `tests/unit/test_cli_b2a_r3.py`
- strictly necessary new test helpers
- `CLAUDE.md`
- `CHANGELOG.md`
- `PLAN.md`
- `README.md`
- the implementation/repair documentation

## Prohibitions

This authorization explicitly prohibits:

- `--execute`
- selected-manifest modification
- selection-provenance creation
- Stage C
- FullKV
- R-KV
- B2B
- model loading
- CUDA
- dataset access
- scientific-setting changes
- accepted-evidence changes
- R-KV pin changes
- branch merge

The production selected manifest must remain the historical row
`test/number_theory/820.json`; the selected production row authorized for
future freezing remains candidate ordinal 1,
`test/number_theory/631.json`; and
`results/decisions/b2a_r3_selection_provenance.json` must remain absent
through this repair.
