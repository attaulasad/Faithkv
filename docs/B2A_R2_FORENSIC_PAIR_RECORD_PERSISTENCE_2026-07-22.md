# B2A-R2 forensic pair-record persistence repair (dated 2026-07-22)

CPU-only durable-artifact repair. No GPU use, no model inference, no
dataset access. Does not rerun, reinterpret, or weaken B2A-R2's frozen
runtime verdict. Does not authorize B2A-R3 or B2B. Implements no
compression policy, no learned eviction policy, no FaithKV method.

## 1. What B2A-R2 established mechanically

`docs/B2A_R2_RESULT_2026-07-22.md` (unmodified by this repair, aside from
the dated clarification appended in its own §8) records that the single
authorized B2A-R2 attempt (`fb6f5081d47f45f4b4f9258c25e6883d`, against the
qualified row `test/number_theory/820.json`, `example_index=365`) ran to
completion: 27 of 28 legacy-gate conditions passed, including every
trajectory/parity/identity/hash-match condition, exact event/pair counts
(12/12 real pairs, 1/1 no-op), meaningful compression (observed retention
ratio 0.2797), and peak tracked VRAM within the 22 GiB limit. The frozen
runtime failure stands, unchanged:

```text
B2A-R2 FINAL VERDICT: FAIL -- B2B BLOCKED
```

Projected runtime 5.01 GPU-hours for a full 12-example/144-real-pair pilot,
25% over the 4.00-hour limit -- see this document's §5 for the corrected
component breakdown (a documentation-arithmetic fix only, not a re-run;
the verdict is identical either way).

## 2. The post-run persistence defect

A post-run audit of the preserved B2A-R2 archive found that "full per-pair
evidence" in the existing documentation was too broad. What actually
survived to disk was execution accounting -- pair counts, pair identities,
semantic-mutation/parity reports, per-pair timings -- **not** the twelve
real interventions' scientific outcomes:

- `swap_gain`
- `baseline_per_token_nll` (48 values)
- `swapped_per_token_nll` (48 values)

The no-op control's NLL information *did* survive (`kvcot.discovery
.b2a_evidence.build_no_op_evidence` derives it from the in-memory
`example_result.pair_records` at worker-result-construction time and folds
it into `rkv/no_op_evidence`'s already-persisted dict) -- but the twelve
real-pair `SwapPairRecord` objects themselves were never serialized onto
any field `RKVWorkerResult` exposed, so they existed only for the lifetime
of the worker process and were discarded when it exited.

**Consequence:** B2A-R2 remains a fully valid mechanical and resource
calibration. No causal-mismatch conclusion can be drawn from B2A-R2's
per-pair science, because that science was never durably recorded. The
missing twelve swap gains cannot be reconstructed after the GPU instance
was destroyed. **No values have been estimated, fabricated, inferred from
pair identities, or backfilled anywhere in this repair.**

## 3. Exact loss path (verified against the actual code, this session)

1. `kvcot.discovery.schemas.SwapPairRecord` already carried the complete
   scientific record -- `swap_gain`, 48-length `baseline_per_token_nll`,
   48-length `swapped_per_token_nll`, recomputed scores
   (`score_e`/`score_r`/`score_margin_e_minus_r`), and full event/layer/
   head/token identity -- with its own cross-field validators
   (`_swap_gain_consistency`, `_noop_invariants`,
   `_nll_finiteness_for_valid_records`) already enforcing internal
   consistency. This module needed no change.
2. `kvcot.discovery.orchestrator.ExampleResult.pair_records:
   tuple[SwapPairRecord, ...]` already held every successfully-built
   record in memory, for the lifetime of `run_example`'s caller.
3. `kvcot.discovery.b2a_workers.RKVWorkerResult` (the Pydantic model a
   worker process serializes to `rkv/result.json` before exiting) had no
   field carrying `pair_records` -- confirmed by direct inspection: its
   ~110 fields cover identities, timings, memory, parity/trajectory
   conditions, and dedicated dict-shaped evidence blocks, but never the
   typed record population itself.
4. `run_rkv_worker`'s `return RKVWorkerResult(...)` construction (the
   function's only return statement) passed `attempted_pair_identities`,
   `completed_pair_identities`, `failed_pair_identities`,
   `semantic_mutation_reports`, and `no_op_evidence` (all identity- or
   summary-shaped) -- never `example_result.pair_records` itself.
5. `run_both_workers_via_subprocess`'s attempt-preservation block wrote
   `rkv/pair_identities.json` (identities only) and
   `rkv/semantic_swaps.json` (mutation reports only) -- no
   `rkv/pair_records.json` existed as a concept anywhere in the codebase.
6. Consequence: the moment `run_rkv_worker`'s process exited, every
   successfully-completed real pair's NLL arrays and swap gain were
   unreachable -- not merely unwritten to the specific attempt directory
   that was later archived, but structurally unreachable from ANY future
   attempt run under the pre-repair code, forever.

No part of this finding contradicted the actual code; every claim above
was confirmed by direct inspection before any line was changed.

## 4. Why no causal conclusion is permitted, and why nothing was reconstructed

The 12 real-pair `SwapPairRecord` objects for the B2A-R2 attempt existed
only inside the worker process's memory for the ~465.6s of Pass 2. The GPU
instance that ran them has since been released. There is no cached
tensor, no checkpoint, no partial dump, and no derivable proxy (the
persisted `no_op_evidence` block covers only the mandatory no-op control,
which by construction has zero swap gain and cannot stand in for a real
intervention) from which the 12 real swap gains could be recomputed
without re-running R-KV against the exact frozen row, seed, and revision
-- which is a new GPU attempt, not a forensic repair, and is not
authorized by this document. Estimating, interpolating, or inferring
those values from anything else committed to this repository would be
fabrication. None was performed.

## 5. Version-aware repair

`kvcot.discovery.b2a_workers.RKVWorkerResult` is split into two explicit,
structurally-versioned Pydantic models:

- **`RKVWorkerResultV1`** -- the original (pre-repair) shape, unmodified,
  field-for-field. Kept so an already-archived V1 result blob (or any
  future payload lacking a `pair_records` key) remains parseable for
  historical integrity checking. Never edited to add a field it never
  had; never fabricates one.
- **`RKVWorkerResultV2(RKVWorkerResultV1)`** -- adds
  `pair_records: list[SwapPairRecord]` as a **required** field (no
  default) and a `schema_version: Literal["rkv_worker_result.v2"]`
  discriminator. A payload that omits `pair_records` fails Pydantic
  validation outright -- a future successful worker can no longer
  silently regress to discarding this evidence the way the original code
  did.

`RKVWorkerResult` (the name every existing import site already uses) is
now an alias for `RKVWorkerResultV2` -- every current and future
production construction of an R-KV worker result requires
`pair_records` explicitly. `run_rkv_worker`'s single return statement now
passes `pair_records=list(example_result.pair_records)` -- read directly
from the same in-memory population every other identity/summary field on
that return statement already reads from, never reconstructed from
identities or summaries.

Version dispatch is **structural**, never attempt-ID- or SHA-based:
`parse_rkv_worker_result(raw)` returns an `RKVWorkerResultV2` when
`"pair_records" in raw`, otherwise an `RKVWorkerResultV1`.
`classify_pair_record_availability(result)` reports
`scientific_pair_records_available` / `scientific_pair_artifacts_verified`
/ `legacy_pair_record_schema` from `isinstance(result, RKVWorkerResultV2)`
alone -- never from which attempt or archive the payload came from.

## 6. New durable artifacts and invariants

For every future successful V2 R-KV worker run that preserves an attempt
directory (`kvcot.discovery.b2a_workers.run_both_workers_via_subprocess`,
gate-passed or gate-failed -- writing is unconditional, exactly like the
pre-existing `rkv/pair_identities.json`/`rkv/semantic_swaps.json`, never
gated on the scientific outcome), two new files are written atomically:

- **`rkv/pair_records.json`** -- the canonical serialized representation
  of the complete `SwapPairRecord` population (never a reduced,
  identity-only representation).
- **`rkv/scientific_summary.json`** -- a pure-CPU summary
  (`kvcot.discovery.scientific_summary.build_scientific_summary`) derived
  exclusively from the validated pair records: `real_pair_count`,
  `no_op_pair_count`, `positive_gain_count`, `gain_above_0_01_count`,
  `median`/`mean`/`minimum`/`maximum_swap_gain` (each `null`, never a
  fabricated `0.0`, when zero valid real records exist), and
  `spearman_score_margin_vs_swap_gain` (a dependency-free, tie-aware
  Spearman correlation between `score_margin_e_minus_r` and `swap_gain`
  over valid real records -- `null` below two valid pairs or under zero
  variance in either series, never a fabricated correlation). Bound to
  `rkv/pair_records.json` via `pair_records_sha256`.

Both files are included in the live coordinator's final reference
manifest (`kvcot.discovery.attempt_artifacts.build_attempt_references`,
which walks the whole attempt directory and now recognizes both
filenames' semantic role) and hash-verified by
`verify_final_reference_manifest` exactly like every other attempt
artifact.

**Partial-failure path** (`kvcot.discovery.worker_partial_evidence
.PartialWorkerEvidence`): gained a `pair_records` field populated from
`example_result.pair_records` at the moment of failure -- exactly the
records that genuinely completed, never padded toward 12 or 13, never
fabricated for a pair that never finished. A failure before any pair
evaluation began honestly reports an empty list. Pre-existing
zero-compaction-event and other failure paths continue to fail cleanly
(`gate_failed`, never an unrelated schema exception).

## 7. Verification

`kvcot.discovery.attempt_verification.verify_pair_record_artifacts` is a
**dedicated, standalone** verifier -- deliberately kept separate from
`verify_attempt_artifacts`/`REQUIRED_ATTEMPT_FILES` (that shared,
pre-existing function is exercised by
`tests/unit/discovery/test_attempt_verification.py` against minimal,
non-B2A-shaped fake pair identities unrelated to this repair; folding a
hard pair-record population/identity requirement into it would fail those
unrelated tests for reasons that have nothing to do with what they test).

For a genuine V2 result (`"pair_records" in rkv_result`), verification
fails unless: both files exist and parse; every record validates as a
typed `SwapPairRecord` (reusing its own validators -- never a second,
independently-written validity check); the population is exactly 12 real
+ 1 no-op; every identity is unique; the population's identities exactly
match `completed_pair_identities`; no identity in `failed_pair_identities`
also appears as completed; `rkv/pair_records.json` matches
`rkv/result.json`'s own `pair_records` field exactly; and
`rkv/scientific_summary.json` recomputes byte-for-byte from
`rkv/pair_records.json` (reusing `build_scientific_summary` -- never a
second statistics formula). A legacy (V1) result returns `(True, ())`
unconditionally -- it never fabricates a requirement a pre-repair result
never had.

**§10 (audit repair round 2) supersedes this section's original wiring**:
`verify_pair_record_artifacts`'s result is now MANDATORY for a V2 attempt's
success, not merely recorded evidence -- see §10.

## 8. Test evidence

New: `tests/unit/discovery/test_pair_record_persistence.py` -- V1/V2
serialization and round-trip, structural version dispatch, tie-aware
Spearman (perfect positive/negative correlation, ties, `<2` pairs, zero
variance, non-finite inputs), scientific-summary recomputation (including
the empty-population and mixed-valid/invalid cases), coordinator
persistence via a faked subprocess runner (never a real subprocess/model),
`verify_pair_record_artifacts` acceptance and every listed failure mode,
and the partial-failure/pre-pair-failure paths. Every new field addition
to a pre-existing shared model
(`kvcot.discovery.worker_partial_evidence.PartialWorkerEvidence`) and
every pre-existing test fixture that now requires an explicit
`pair_records` key
(`tests/unit/discovery/test_b2a_workers.py`,
`tests/unit/discovery/test_b2a_execute_coordinator.py`) was updated in
place, never worked around.

No GPU was used to produce or verify any of the above -- every test
constructs its own synthetic `SwapPairRecord`/worker-result payloads.

## 9. Status

B2A-R2's frozen verdict (`FAIL -- B2B BLOCKED`) is unchanged. B2A-R3 is
**not** authorized by this document -- it requires its own separate,
future, dated authorization, exactly as prior CLAUDE.md sections already
require. B2B remains blocked. No FaithKV method exists. This repair adds
durable persistence and verification infrastructure only; it establishes
no new scientific finding about B2A-R2 and reverses none of its existing
ones.

## 10. Audit repair round 2 (dated 2026-07-22): pair-artifact verification made mandatory

An independent audit found that round 1's `verify_pair_record_artifacts`
wiring (§7, original text preserved above) was **never-fatal**: its result
was recorded under `payload["pair_record_verification"]` but did not
affect `overall_passed`, `exit_code`, or `completion.json`'s outcome. A
future V2 attempt could therefore still be reported as a completed success
while `rkv/pair_records.json`/`rkv/scientific_summary.json` was missing,
incomplete, duplicated, mismatched, or corrupt -- defeating this repair's
own purpose. This section documents the fix, CPU-only, no GPU/inference/
re-run, no change to §6's underlying artifacts or the frozen B2A-R2
verdict.

**Mandatory gate wiring** (`kvcot.discovery.b2a_execute.run_b2a_calibration`):
`verify_pair_record_artifacts` now runs BEFORE `overall_passed` is
computed. A new derived condition,

```python
scientific_pair_artifacts_verified = (
    isinstance(rkv, RKVWorkerResultV2) and pair_record_verified
)
```

is combined into the outer coordinator's success determination as a third,
independent, ANDed factor:

```python
overall_passed = bool(
    gate_result.passed and final_gate_result.passed and scientific_pair_artifacts_verified
)
```

exactly the same pattern already used to combine the legacy 28-condition
gate and the final 31-condition gate at this outer layer -- **never folded
into `FINAL_MANDATORY_GATE_CONDITIONS` itself**, which remains completely
unmodified (CLAUDE.md's frozen-intervention-protocol constraint forbids
touching that tuple's identity; this repair achieves the audit's required
behavior -- `overall_passed=false`, `exit_code=2`,
`outcome="gate_failed"` whenever pair-artifact verification fails --
without doing so). Both the derived boolean and every reason are preserved
in `final.json` (`payload["scientific_pair_artifacts_verified"]`,
`payload["pair_record_verification"]["reasons"]`).

This applies only to attempts this live coordinator creates going forward
-- `rkv` is always `RKVWorkerResultV2` on that path, so the `isinstance`
guard is a documentation-level safeguard, not a live branch. It is **not**
applied retroactively: B2A-R2's own already-recorded, frozen
`completion.json`/`final.json` are historical, immutable artifacts and are
untouched by this section.

**`parse_rkv_worker_result` schema-version dispatch fixed.** Previously,
version dispatch was purely structural (`"pair_records" in raw`) -- a
payload explicitly labeled `schema_version="rkv_worker_result.v2"` but
missing `pair_records` (e.g. a corrupted or hand-edited historical blob)
would fail the structural check and silently be accepted as a legitimate
V1 (legacy) result, with its `schema_version` marker dropped by V1's
default field-ignoring behavior -- a broken V2 payload masquerading as an
honest legacy one. Fixed: `schema_version` is now checked FIRST. Present
and equal to `"rkv_worker_result.v2"` -> must validate as
`RKVWorkerResultV2` or raise (never falls back to V1).  Present and equal
to anything else -> raises `UnknownRKVWorkerResultSchemaVersion` (an
unrecognized version is rejected outright, never guessed at). Absent
entirely -> the original structural fallback (the correct behavior for
every result committed before this repair, none of which ever had a
`schema_version` field). This function is a standalone historical-blob
parser, not part of the live coordinator path (which already used strict
`RKVWorkerResult.model_validate_json`, i.e. V2-only, and is unaffected).

**Test evidence** (`tests/unit/discovery/test_b2a_execute_coordinator.py`,
`tests/unit/discovery/test_pair_record_persistence.py`): ten new tests,
asserting the COORDINATOR's outcome (`overall_passed`,
`completion.json["exit_code"]`, `completion.json["outcome"]`), never only
the standalone verifier's own return value -- a control case (valid
pair-record artifacts do not spuriously fail the new gate) plus one test
per corruption mode: missing `pair_records.json`, missing
`scientific_summary.json`, an incomplete population (11 of 12 real
records, deliberately chosen so the pre-existing legacy gate -- which
never inspects `pair_records` at all -- cannot see it), a duplicate
identity, a `completed_pair_identities` mismatch (mutating `layer_index`
rather than `compaction_event_id`, so the pre-existing legacy identity
conditions remain genuinely passing and the test isolates the new
cross-artifact check specifically), a corrupted `scientific_summary.json`,
and a `schema_version="rkv_worker_result.v2"` payload missing
`pair_records` (both at the coordinator level, where the pre-existing,
unmodified `RKVWorkerResult.model_validate_json` already rejects it before
`completion.json` is ever written, and directly against
`parse_rkv_worker_result`). Every corruption test uses the REAL
`run_both_workers_via_subprocess` to write genuinely correct artifacts
first, then corrupts them -- proving production code's own writer output
is what gets caught, not a hand-built bad fixture. 231 tests across every
directly and indirectly affected test file pass locally (torch-independent
subset); full-suite confirmation is GitHub Actions CPU CI (this
repository's local Windows environment cannot import real torch -- see
CLAUDE.md's session notes).

```text
B2A-R2 FORENSIC CLOSURE VERDICT:
PAIR-RECORD PERSISTENCE REPAIRED -- READY FOR INDEPENDENT REVIEW; B2A-R3/B2B REMAIN BLOCKED
```

```text
B2A-R2 FORENSIC CLOSURE VERDICT:
PAIR-RECORD PERSISTENCE REPAIRED -- READY FOR INDEPENDENT REVIEW; B2A-R3/B2B REMAIN BLOCKED
```

## 11. Audit repair round 3 (dated 2026-07-22): one authoritative result across every public surface

An independent audit of round 2 found that `overall_passed` (correctly
computed for `completion.json`) never reached three other surfaces:
`payload["passed"]` (embedded in `final.json`, built earlier in
`run_b2a_calibration` from only the legacy and final gates and never
updated), `B2ACalibrationArtifact` (which exposed no pair-artifact
outcome at all), and `kvcot.cli.cmd_b2a_calibrate` (which independently
recomputed a two-gate `overall_passed` of its own, ignoring pair-artifact
verification entirely -- meaning the CLI could print `passed=True` and
return exit code `0` while `completion.json` said `gate_failed`/`2`). This
section documents the fix, CPU-only, no GPU/inference/re-run, no change to
the frozen B2A-R2 verdict, `FINAL_MANDATORY_GATE_CONDITIONS`, or any prior
section's underlying artifacts.

**One computation, before `payload` is even built.**
`run_b2a_calibration` now computes `pair_record_verified`/
`scientific_pair_artifacts_verified`/`overall_passed` ONCE, immediately
after `final_gate_result`, before `payload = {...}` is constructed.
`payload["passed"]` is set directly from this `overall_passed` (never a
separately recomputed expression), and `payload["scientific_pair_artifacts_verified"]`/
`payload["pair_record_verification"]` are populated at the same point.
`completion.json` (written later) uses the SAME already-computed
`overall_passed` -- it was never wrong, but now shares its source value
with `final.json` by construction rather than by coincidence.

**`attempt_directory is None` is never silently "verified".**
`kvcot.discovery.attempt_verification.verify_pair_record_artifacts` was
split: the on-disk-file checks stay there, and the population/identity
checks (typed validity, 12 real + 1 no-op, unique identities, agreement
with `completed_pair_identities`, no failed identity double-counted) moved
into a new, standalone `verify_pair_record_population(rkv_result)` that
needs no attempt directory at all. Production `--execute` always supplies
an attempt directory (`kvcot.cli.cmd_b2a_calibrate` calls
`create_attempt_directory()` unconditionally under `--execute`); the
`attempt_directory is None` path exists only for helper/test callers. That
path now calls `verify_pair_record_population` directly -- corrupting the
in-memory pair-record population (with no file involved at all) still
correctly flips `overall_passed` to `False`, proven by
`test_no_attempt_directory_still_genuinely_verifies_pair_records_not_silently_true`.

**`B2ACalibrationArtifact` gained three required-in-practice fields:**
`overall_passed`, `scientific_pair_artifacts_verified`,
`pair_record_verification_reasons` -- populated once, at the single
`return B2ACalibrationArtifact(...)` call site, from the same values
`payload`/`completion.json` already used. No caller reconstructs them.

**The CLI now relays, never recomputes.**
`kvcot.cli.cmd_b2a_calibrate` replaced its own two-gate
`artifact.gate_result.passed and final_passed` computation with
`overall_passed = artifact.overall_passed` -- a straight read, not a
second definition of success. On failure it now also prints
`pair_artifact_verification_reasons` when pair-artifact verification is
why the attempt failed. The pre-existing `completion.json` fallback
(written only `if not (attempt.path / "completion.json").exists()`) is
unchanged -- the CLI still never overwrites the coordinator-authored
record.

**Test evidence.** `tests/unit/discovery/test_b2a_execute_coordinator.py`:
the shared failure-assertion helper now also asserts
`artifact.overall_passed is False`, `payload["passed"] is False`, and that
`payload["passed"] == completion["gate_passed"] == artifact.overall_passed`
across all six existing corruption scenarios (missing
`pair_records.json`, missing `scientific_summary.json`, incomplete
population, duplicate identity, identity mismatch, corrupt summary); the
control case gained the matching positive assertions; one new test proves
the no-attempt-directory path. `tests/unit/test_cli_b2a_calibrate.py`:
`test_cli_returns_2_for_an_isolated_pair_artifact_failure` (legacy gate
and final gate both `True`, pair-artifact verification the sole failing
factor -- asserts exit code `2`, `passed=False` printed, the specific
reason printed, and `completion.json` agreement),
`test_cli_returns_0_when_all_three_factors_pass`, and a
parametrized `test_cli_return_code_and_printed_passed_value_never_disagree_with_coordinator`
proving the CLI's return code and printed value track
`artifact.overall_passed` exactly in both directions. The two pre-existing
CLI tests that fake the coordinator's return value
(`test_execute_writes_real_device_preflight_evidence_and_threads_it_to_coordinator`,
`test_execute_writes_completion_record_on_gate_failure`) were updated to
supply the three new fields their fakes now require. 260 tests across
every directly and indirectly affected file pass locally (torch-independent
subset); full-suite confirmation is GitHub Actions CPU CI.

```text
B2A-R2 FORENSIC CLOSURE VERDICT:
PAIR-RECORD PERSISTENCE REPAIRED -- READY FOR INDEPENDENT REVIEW; B2A-R3/B2B REMAIN BLOCKED
```

## 12. Merge closure (2026-07-22)

The round-3 repair above was independently audited and merged into `main`:

```text
PR: #20
Merged at: 2026-07-22 15:48:27 UTC
Merge commit: 9e78bc5edda0f0086d9e9aaea98896ac24caa7b0
Audited repair head: dc2681158ed4d68fd8fe08f313033015a54ec942
```

This merge closes the durable pair-record persistence and
outcome-consistency defects documented in §§1-11 above. It changes no
historical B2A-R2 measurement or conclusion: B2A-R2 remains a consumed
experimental attempt, scientifically unsuccessful because projected
runtime was approximately 5.01 GPU-hours, above the 4.00-hour limit
(`B2A-R2 FINAL VERDICT: FAIL -- B2B BLOCKED`, unchanged), and B2A-R2 is
prohibited from rerun. This closure is not an authorization for B2B and
not an authorization for FaithKV method implementation.

The only next authorized activity is B2A-R3 protocol design and CPU-only
implementation on `research/b2a-r3-runtime-qualified-calibration`:

Design and implement the B2A-R3 runtime-qualified protocol entirely on CPU,
using deterministic outcome-blind candidate selection and a conservative
3.60-GPU-hour qualification target, while retaining the hard 4.00-GPU-hour
execution gate.

GPU rental, FullKV qualification, R-KV execution, pair evaluation, and
B2A-R3 execution remain unauthorized.

```text
B2A-R2 FORENSIC CLOSURE VERDICT:
PAIR-RECORD PERSISTENCE REPAIRED AND MERGED
```
