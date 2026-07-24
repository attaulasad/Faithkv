# B2A-R3 Stage-B evidence-acceptance test-assumption repair — 2026-07-24

## Context

The Stage-B evidence-acceptance commit
(`be59ca97fe72ec8fe44b37495b9f13f849a2bcba`, "Accept audited B2A-R3
Stage-B qualification evidence") committed the audited, independently
verified consumed claim
(`results/decisions/b2a_r3_authorization_claims/stage-b-2026-07-24-r2-final.json`)
and qualification artifact (`results/decisions/b2a_r3_qualification.json`)
to their real production paths, as explicitly required.

Exact-SHA GitHub Actions CPU CI on that commit (run `30075578177`) failed
with 2 of 1850 non-GPU tests failing:

- `tests/unit/discovery/test_b2a_r3_authorization.py::test_no_production_claims_directory_touched_by_dry_run`
- `tests/unit/discovery/test_b2a_r3_qualification_coordinator.py::test_no_production_files_written`

## Root cause

Both tests were written when no Stage-B claim or qualification artifact
had ever been persisted to the repository, and each encoded that
transient fact as a hard, non-negotiable assertion (`assert after is
False, "the real authorization-claims directory must not exist in this
repository"` and `assert not Path(QUALIFICATION_ARTIFACT_PATH).exists()`)
rather than testing the actual invariant under test: that dry-run
planning and the pure in-memory qualification coordinator never touch
those filesystem paths. Committing the accepted evidence to its real
production location -- an explicit requirement of the evidence-acceptance
step -- falsified the stale "must never exist" assumption while leaving
the real "must not be touched by this code path" invariant untouched and
still true.

This is a test-environment/test-assumption defect exposed by an intended,
authorized repository state change, not a production code defect.
`src/kvcot/discovery/b2a_r3_authorization.py`,
`src/kvcot/discovery/b2a_r3_qualification_coordinator.py`, and every other
file under `src/`, `configs/`, `third_party/R-KV/`, and `results/` are
unchanged by this repair.

```text
B2A-R3 STAGE-B EVIDENCE-ACCEPTANCE TEST-ASSUMPTION REPAIR AUTHORIZED — TEST FILES ONLY

AUTHORIZED:
NARROW REPAIR OF TWO TESTS THAT ASSERTED PRODUCTION EVIDENCE PATHS
MUST NEVER EXIST, REPLACED WITH BEFORE/AFTER NO-TOUCH ASSERTIONS

PROHIBITED:
PRODUCTION SOURCE CHANGES
SCIENTIFIC CONFIGURATION CHANGES
EVIDENCE / CLAIM / QUALIFICATION-ARTIFACT CHANGES
MODEL INFERENCE
FULLKV/R-KV EXECUTION
STAGE C
```

## Repair

- `test_no_production_claims_directory_touched_by_dry_run`: kept the
  existing `before == after` existence-equality assertion (the real
  invariant -- dry-run planning must not touch the directory either way)
  and removed the now-false `assert after is False` over-assertion.
- `test_no_production_files_written`: replaced the flat
  `assert not Path(QUALIFICATION_ARTIFACT_PATH).exists()` with a
  before/after snapshot of both existence and byte content, asserting the
  coordinator neither creates, deletes, nor modifies whatever is (or is
  not) already at that path. This is a strictly stronger test than the
  one it replaces: it now also catches the coordinator silently
  overwriting a pre-existing production artifact, which the original
  assertion could never detect once such an artifact existed.
- No other test, fixture, or assertion changed. No file under `src/`,
  `configs/`, `third_party/R-KV/`, or `results/` changed.

## Local validation

- Both repaired tests pass individually.
- `python -m compileall -q src tests` clean.
- Full non-GPU suite: `python -m pytest -m "not gpu" -q` → 1850 passed, 14
  deselected, 0 failed (same pass count as before the evidence-acceptance
  commit, now including the two repaired assertions).
- `git diff --check` clean.

## Status

```text
B2A-R3 STAGE-B EVIDENCE-ACCEPTANCE TEST-ASSUMPTION REPAIR COMPLETE —
LOCAL VALIDATION GREEN; PUSHED FOR EXACT-SHA CI;

READY TO CONFIRM EXACT-SHA CI GREEN FOR PHASE 1 COMPLETION;
STAGE C REMAINS BLOCKED
```
