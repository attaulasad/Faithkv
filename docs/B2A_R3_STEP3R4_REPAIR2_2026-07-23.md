# B2A-R3 Step 3R4 Independent Re-Audit Repair, Round 2 (dated 2026-07-23)

```text
STEP 3R4 INDEPENDENT RE-AUDIT REPAIR ROUND 2 IMPLEMENTED —
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED
```

This record covers a second, bounded, CPU-only repair round against Step
3R4 starting SHA `187236426b5fb48321c18a91556cf0d560918494` on
`research/b2a-r3-runtime-qualified-calibration`
(`docs/B2A_R3_STAGE_B_READINESS_REPAIR_2026-07-23.md`, the round this one
supersedes for the seven findings below). An independent re-audit of that
SHA returned **FAIL**, with seven blocking findings plus one report/
documentation inconsistency. It is an implementation and regression-test
record, not an independent approval of its own work — a separate,
genuinely independent re-audit of the final SHA in this round remains the
next required action before Stage A (§1e/§1j) can proceed.

## Authorization boundary

This round used pure Python, strict pydantic schemas, injected fakes (a
real, deterministic, torch-CPU-tensor model/tokenizer/cuda backend for the
new worker path; synthetic evidence elsewhere), and temporary-directory
filesystem tests. It did not initialize real CUDA, load real model
weights, run a real FullKV or R-KV worker, create a real qualification
artifact, replace the real selected manifest, create a real authorization
document or claim, execute B2A-R3 or B2B, or implement any FaithKV method.

The pinned R-KV gitlink remains `45eaa7d69d20b7388321f077020a610d9afb65bd`
and was never touched. The committed B2A-R3 candidate manifest
(`configs/discovery/b2a_r3_candidate_manifest.json`) is unchanged; its
`canonical_sha256` is unchanged:

```text
b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42
```

These production paths remain exactly as they were (none exist in this
repository, and none are created by this round):

```text
results/decisions/b2a_r3_qualification.json
configs/discovery/b2a_one_example_manifest.json
results/decisions/b2a_r3_selection_provenance.json
results/decisions/b2a_r3_authorization_claims/
results/decisions/b2a_r3_attempt_*/
```

## Root-cause and repair ledger

### Blocking Finding 1 — no real FullKV worker path produced R3 evidence

- **Root cause:** `kvcot.discovery.b2a_r3_worker_adapter
  .adapt_fullkv_worker_result_to_r3_evidence` only ever converted an
  already-strict `FullKVWorkerResultR3` — every existing test hand-built
  that object with a helper. The canonical, already-frozen
  `kvcot.discovery.b2a_workers.run_fullkv_worker` (used unmodified by
  historical B2A-R1/R2) returns the historical `FullKVWorkerResult` shape,
  which has no thinking-span fields, no expected/observed prompt-hash
  pair, no R3 candidate hashes, and no R3 worker-schema version. There was
  no production path from a candidate row all the way to qualification
  evidence.
- **Repair:** a new module,
  `kvcot.discovery.b2a_r3_qualification_worker`, adds
  `run_fullkv_r3_qualification_worker(config, candidate, **injection_seams)`.
  It renders the candidate's prompt via a new, factored-out canonical
  renderer helper, `kvcot.discovery.manifest_prepare
  .render_with_loaded_tokenizer` (extracted from `_render_and_tokenize`
  with byte-identical behavior for its two existing callers), reuses
  `run_fullkv_worker` completely unmodified (never a second,
  independently-written FullKV loop), threads the SAME tokenizer instance
  into both the renderer and the worker's own `_load_tokenizer` seam, calls
  `kvcot.probes.early_answering.find_think_span` with that identical
  tokenizer to derive the think-span fields, reconstructs the
  qualification-comparison-shaped generation-config payload from
  `config.generation` directly (the worker's own `runtime_generation` dict
  has a different, call-shape-focused field set and cannot reproduce
  `GENERATION_CONFIG_SHA256`), and assembles/validates a strict
  `FullKVWorkerResultR3`.
- **Source:** `kvcot.discovery.b2a_r3_qualification_worker`,
  `kvcot.discovery.manifest_prepare`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_qualification_worker.py` —
  exercises the REAL, injected `run_fullkv_worker` body end to end (the
  same fakes `tests/unit/discovery/test_b2a_workers_real_bodies.py`
  already uses for `run_fullkv_worker` directly), then feeds that REAL
  output through the existing, already-tested
  `adapt_fullkv_worker_result_to_r3_evidence` adapter — never a hand-built
  `FullKVWorkerResultR3` — and separately proves the reconstructed
  generation-config payload reproduces `GENERATION_CONFIG_SHA256` exactly
  for the real frozen generation config.
- **Status:** repaired.

### Blocking Findings 2/3 — timing/memory validators rejected real worker output

- **Root cause:** `fullkv_qualification_timing_complete`'s allowed phase
  vocabulary aliased `FULLKV_REQUIRED_TIMING_PHASES` directly, on the
  claim that it was "the same canonical, already-frozen 10-phase
  vocabulary" the real worker emits. That claim was false: the real
  `run_fullkv_worker` body's `measured()` helper times AND memory-samples
  `before_model_load`/`post_load_baseline` in one call, so both genuinely
  appear in the TIMING stream (not only memory evidence), and
  `answer_verification` is genuinely nested BEFORE (never after)
  `fullkv_complete_natural_generation` (called from inside the
  `run_natural_pass1` call that phase wraps). Separately,
  `fullkv_qualification_memory_complete` required exactly the historical
  5-phase `FULLKV_REQUIRED_MEMORY_PHASES` set, but the real worker's
  `measured()` helper also memory-samples `tokenizer_load` and
  `post_load_validation` — real memory evidence has 7 phases, not 5. A
  real worker result failed both validators outright.
- **Repair:** `kvcot.discovery.final_contract` now derives
  `_FULLKV_QUALIFICATION_PHASE_ORDER` as its own frozen 12-phase tuple
  matching the real worker's actual emitted order exactly (`before_model_load,
  fullkv_worker_startup, snapshot_tokenizer_resolution, tokenizer_load,
  model_load, post_load_validation, post_load_baseline, fullkv_prefill,
  fullkv_decode, answer_verification, fullkv_complete_natural_generation,
  fullkv_complete_worker`), independently reproduced against a REAL
  (non-fake) `run_fullkv_worker` invocation in
  `tests/unit/discovery/test_b2a_workers_real_bodies.py`. A new
  `_FULLKV_QUALIFICATION_MEMORY_PHASES` (7 phases) replaces the historical
  5-phase set for `fullkv_qualification_memory_complete` only.
  `FULLKV_REQUIRED_TIMING_PHASES`/`FULLKV_REQUIRED_MEMORY_PHASES` and the
  historical two-worker `timing_contract_satisfied`/
  `memory_contract_satisfied` gates are byte-for-byte unchanged — this
  repair adds a second, qualification-only vocabulary rather than touching
  the historical B2A-R1/R2 gate. The shared `_timing()`/`_memory()` test
  fixtures (used across the whole qualification test suite) are corrected
  to the real order/vocabulary.
- **Source:** `kvcot.discovery.final_contract`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_timing_memory_contract.py`
  (frozen-vocabulary assertions, missing-phase coverage for all 12 timing
  phases and all 7 memory phases, duplicate-singleton rejection for
  `before_model_load`/`post_load_baseline`, unknown-phase rejection),
  `tests/unit/discovery/test_b2a_r3_qualification_worker.py` (the real
  worker's own timing/memory evidence passes both validators end to end
  through the adapter).
- **Status:** repaired.

### Blocking Finding 4 — phase-wide authorization time was not a hard limit

- **Root cause:** `run_b2a_r3_qualification_coordinator` checked elapsed
  phase-wide wall time before launching each candidate, but always passed
  the FULL frozen `PER_CANDIDATE_WORKER_TIMEOUT_SECONDS` (7200s) to the
  worker regardless of how little authorized phase time actually remained,
  and never re-checked elapsed time after a worker returned. A candidate
  started near the end of an authorized phase window could run for up to
  two hours despite a much smaller remaining authorization, and a run that
  blew its phase deadline on its LAST authorized candidate would exit the
  loop normally and mis-report `stopped_reason="all_authorized_candidates_exhausted"`.
- **Repair:** the coordinator now computes
  `effective_worker_timeout = min(per_candidate_timeout_seconds, remaining_phase_seconds)`
  before every launch, and re-checks elapsed phase-wide time immediately
  after every worker completes (not only before the next launch), setting
  `stopped_reason="phase_wall_time_exhausted"` in either direction.
- **Source:** `kvcot.discovery.b2a_r3_qualification_coordinator`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_qualification_coordinator.py`
  (`test_effective_worker_timeout_capped_at_remaining_phase_time`,
  `test_effective_worker_timeout_never_exceeds_the_frozen_per_candidate_cap`,
  `test_post_worker_check_relabels_stop_reason_on_last_candidate`); the
  shared fake runner's timeout assertion is relaxed from exact equality to
  `<=` the frozen cap, matching the new (correct) capping behavior.
- **Status:** repaired.

### Blocking Finding 5 — atomic writer performed only shallow verification

- **Root cause:** `write_qualification_artifact_atomic` checked only the
  canonical self-hash and top-level pydantic schema — both transport
  integrity checks. A canonically-rehashed artifact with internally
  self-consistent but semantically fabricated outcome fields (never
  actually reproducible from raw evidence) would pass.
- **Repair:** `write_qualification_artifact_atomic` now requires
  `candidate_manifest`/`expected_config_sha256` as parameters and calls the
  one authoritative semantic verifier, `verify_qualification_artifact`
  (which independently re-derives every attempted outcome's 27 conditions
  and replays first-pass selection), both BEFORE writing and again on the
  bytes actually read back from disk.
- **Source:** `kvcot.discovery.b2a_r3_artifacts`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_qualification_coordinator.py`
  (`test_atomic_writer_refuses_semantically_fabricated_but_canonically_rehashed_artifact`,
  plus every existing writer test updated to pass the now-required
  parameters).
- **Status:** repaired.

### Blocking Finding 6 — stop reasons were not independently verified

- **Root cause:** `QualificationArtifactR3.qualification_stopped_reason`
  had no validator requiring membership in the four frozen reasons, and
  `build_qualification_artifact`/the artifact schema had no record of the
  authorization's actual `maximum_candidates` — so
  `qualification_stopped_reason="all_authorized_candidates_exhausted"`
  could never be checked against how many candidates were actually
  authorized. A canonically-rehashed artifact with an arbitrary
  `qualification_stopped_reason` string, or a fabricated exhaustion claim
  against too few attempted candidates, would pass.
- **Repair:** `QUALIFICATION_ARTIFACT_SCHEMA_VERSION` is bumped v2 to v3.
  `QualificationArtifactR3` gains a required `authorized_maximum_candidates`
  field (validated `1..QUALIFICATION_CANDIDATE_LIMIT`) and a
  `qualification_stopped_reason` field validator requiring membership in
  `ALLOWED_QUALIFICATION_STOPPED_REASONS`. A cross-field invariant requires
  `attempted_candidate_count <= authorized_maximum_candidates` always, and
  exact equality when the reason is
  `all_authorized_candidates_exhausted`. `build_qualification_artifact`
  takes the same required parameter (sourced by the coordinator from
  `VerifiedAuthorizationContext.maximum_candidates`, never a caller
  default) and enforces the identical checks before construction, so a
  caller gets a clear `QualificationArtifactBuildRefused` rather than a raw
  pydantic error.
- **Source:** `kvcot.discovery.b2a_r3_contract`,
  `kvcot.discovery.b2a_r3_artifacts`,
  `kvcot.discovery.b2a_r3_qualification_coordinator`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_qualification_coordinator.py`
  (`test_builder_rejects_unknown_authorized_maximum_candidates`,
  `test_builder_rejects_exhausted_reason_when_count_disagrees_with_authorization`),
  every existing builder/writer/coordinator test updated to supply
  `authorized_maximum_candidates`, `tests/unit/discovery/test_b2a_r3_qualification.py`
  (`test_qualification_artifact_versions_are_v2` updated to v3).
- **Status:** repaired.

### Blocking Finding 7 — claim path not bound to the Git provider's repository root

- **Root cause:** `verify_authorization_preconditions`/`claim_authorization`
  accepted `git_state` and `repository_root` as two independent parameters
  with no check that they referred to the same filesystem root — in
  principle, Git state could be verified against one repository while the
  claim/document I/O happened under a different one, letting the same
  `authorization_id` be consumed under more than one root.
- **Repair:** `GitStateProvider` gains a required `repository_root`
  property (already present as a plain attribute on the production
  `SubprocessGitStateProvider`). A new
  `kvcot.discovery.b2a_r3_provenance.verify_git_state_bound_to_repository_root`
  compares `Path(git_state.repository_root).resolve()` against
  `Path(repository_root).resolve()` and raises
  `GitStateRepositoryRootMismatch` on any disagreement.
  `verify_authorization_preconditions` and `claim_authorization` both call
  it before doing anything else with either argument.
- **Source:** `kvcot.discovery.b2a_r3_provenance`,
  `kvcot.discovery.b2a_r3_authorization`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_authorization.py`
  (`test_git_state_repository_root_mismatch_rejected_by_preconditions`,
  `test_git_state_repository_root_mismatch_rejected_by_claim_authorization`);
  every existing `FakeGitState` construction that participates in
  precondition/claim verification now carries a matching
  `repository_root`.
- **Status:** repaired.

### Report inconsistency — v2 outcome field count

The independent re-audit report stated the v2 per-candidate qualification
outcome has 37 fields; it actually has exactly 53
(`6 + 8 + 3 + 4 + 9 + 8 + 4 + 7 + 1 + 3 = 53`), matching
`kvcot.discovery.b2a_r3_contract.QUALIFICATION_OUTCOME_V2_FIELD_NAMES`
exactly (already enforced by
`test_qualification_outcome_v2_field_set_matches_contract`). No source
file changed for this item — it was a report/documentation error, not a
repository defect. Recorded here as the authoritative correction so a
future audit does not repeat it.

## Validation status

All local validation used `CUDA_VISIBLE_DEVICES=""` and this repository's
existing `pytest` configuration.

- Focused Blocking-Finding 1-7 test files (new + touched): all green,
  individually and together.
- `python -m pytest tests/unit -k "b2a_r3 or cli_b2a" -q`: 504 passed.
- `python -m pytest tests/unit -q` (full CPU unit suite): 1835 passed, 1
  skipped (the skip is the pre-existing, unmodified GPU-only
  `test_capture.py` case — no CUDA device on this machine).

## Authorization boundary (restated)

```text
STAGE B FULLKV QUALIFICATION: NOT AUTHORIZED
STAGE C B2A-R3 EXECUTION: NOT AUTHORIZED
GPU RENTAL: PROHIBITED UNTIL INDEPENDENT RE-AUDIT PASSES
R-KV DURING QUALIFICATION: PROHIBITED
B2B: BLOCKED
FAITHKV METHOD IMPLEMENTATION: BLOCKED
NEXT REQUIRED ACTION: INDEPENDENT RE-AUDIT OF THE FINAL REPAIR SHA
BRANCH MERGED: NO
```
