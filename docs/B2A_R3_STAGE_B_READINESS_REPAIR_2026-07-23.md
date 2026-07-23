# B2A-R3 Step 3R4 Stage-B Readiness Repair (dated 2026-07-23)

```text
STEP 3R4 CPU PROTOCOL ALIGNMENT AND STAGE-B READINESS REPAIRS IMPLEMENTED —
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED
```

This record covers the bounded, CPU-only Step 3R4 repair of six findings
identified against Step 3 Stage-A implementation starting SHA
`7062f3cb8a6f555d3b67cf9e9be3bd6710e78120` on
`research/b2a-r3-runtime-qualified-calibration`, whose original ancestor
is `382de26fe68206ac195ba85a56cc7f5f7696be11`. It is an implementation and
regression-test record, not an independent approval of its own work. The
frozen B2A-R3 protocol
(`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md`) retains
precedence except for the narrow qualification-outcome schema/version
fields the new dated amendment
(`docs/B2A_R3_STAGE_A_PROTOCOL_ALIGNMENT_AMENDMENT_2026-07-23.md`)
supersedes. Stage B remains blocked until a separate independent re-audit
accepts the final Step 3R4 repair SHA.

## Authorization boundary

This round used pure Python, strict pydantic schemas, synthetic/injected
evidence, temporary-directory filesystem tests, and CPU-only verification
paths. It did not initialize CUDA, load model weights, load a tokenizer
for execution, run FullKV or R-KV, create a real qualification artifact,
replace the real selected manifest, write real selection provenance,
create a real authorization document, create a real authorization claim
or attempt directory, execute B2A-R3 or B2B, or implement any FaithKV
method.

The pinned R-KV gitlink remains `45eaa7d69d20b7388321f077020a610d9afb65bd`
and was never touched. Historical B2A-R1/R2 artifacts, timing/provenance
behavior, and the committed B2A-R3 candidate manifest were not modified:

```text
configs/discovery/b2a_r3_candidate_manifest.json canonical_sha256
(unchanged, before and after this repair):
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

### Finding 1 — qualification schema/protocol mismatch

- **Root cause:** the v1 per-candidate outcome record
  (`CandidateQualificationOutcomeR3`) duplicated four artifact-level
  identity fields onto every attempted outcome
  (`candidate_manifest_canonical_sha256`, `config_sha256`, `budget`,
  `divide_length`) and persisted a redundant nested `runtime_prediction`
  object alongside its own already-flattened runtime fields.
  `generation_config_sha256` was reused, ambiguously, both as "the
  worker's observed hash" (on the outcome) and "the frozen expected hash"
  (on the artifact).
- **Repair:** a dated protocol amendment
  (`docs/B2A_R3_STAGE_A_PROTOCOL_ALIGNMENT_AMENDMENT_2026-07-23.md`)
  freezes the exact v2 field table; `QUALIFICATION_ARTIFACT_SCHEMA_VERSION`/
  `QUALIFICATION_PROTOCOL_VERSION` are bumped v1->v2
  (`kvcot.discovery.b2a_r3_contract`). `CandidateQualificationOutcomeR3`
  drops the four redundant fields and the nested runtime object, and
  renames its generation-config field to `worker_generation_config_sha256`.
  `rederive_and_verify_qualification_outcome` now reruns
  `predict_runtime(total_processed_tokens)` fresh and requires exact
  field-by-field equality, rather than trusting a stored nested object. A
  new frozen `QUALIFICATION_OUTCOME_V2_FIELD_NAMES` contract constant, and
  a matching drift test, prevent the schema and the contract from
  silently diverging again.
- **Source:** `kvcot.discovery.b2a_r3_contract`,
  `kvcot.discovery.b2a_r3_qualification`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_qualification.py`
  (`test_qualification_outcome_v2_field_set_matches_contract`,
  `test_qualification_outcome_v2_rejects_v1_redundant_fields`,
  `test_qualification_artifact_versions_are_v2`), plus the existing
  qualification/artifact/freeze suites re-run green against the realigned
  schema.
- **Status:** repaired.

### Finding 2 — incorrect FullKV timing vocabulary

- **Root cause:** `fullkv_qualification_timing_complete` used a hand-rolled
  12-phase vocabulary that spliced two MEMORY phases
  (`before_model_load`, `post_load_baseline`) into the TIMING-only phase
  list, and placed `answer_verification` before
  `fullkv_complete_natural_generation` — backwards relative to the real,
  already-frozen `FULLKV_REQUIRED_TIMING_PHASES` vocabulary every other
  FullKV timing consumer in this repository (`timing_contract_satisfied`)
  already uses.
- **Repair:** `fullkv_qualification_timing_complete` now derives its
  singleton phase set and expected order directly from
  `FULLKV_REQUIRED_TIMING_PHASES` (10 phases, `fullkv_decode` variable
  multiplicity >= 1, every other phase exactly once, canonical worker
  order). A new `fullkv_qualification_memory_complete` validates FullKV
  memory-phase evidence separately, via the existing (unmodified)
  `FULLKV_REQUIRED_MEMORY_PHASES`/`FULLKV_MEMORY_EXACT_MULTIPLICITY`
  contract, and `peak_cuda_bytes_from_qualification_memory_evidence`
  extracts peak CUDA bytes from the `fullkv_complete_worker` memory
  record. The shared `_timing()` test fixture (used across the whole
  qualification test suite) is corrected to the canonical order; a
  companion `_memory()` fixture is added.
- **Source:** `kvcot.discovery.final_contract`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_timing_memory_contract.py`
  (canonical historical timing/memory pass; memory phase spliced into
  timing fails; timing phase spliced into memory fails; every required
  phase individually missing fails; duplicate singleton fails; multiple
  `fullkv_decode` records still pass; wrong order fails; negative/NaN
  duration fails; wrong wall-time binding fails; peak-byte extraction from
  the correct phase), plus `tests/unit/discovery/test_final_contract.py`
  unaffected (`timing_contract_satisfied` untouched).
- **Status:** repaired.

### Finding 3 — circular claim-derived authorization policy

- **Root cause:** `verify_authorization_preconditions` treated the dated
  Stage B/C authorization Markdown document as an opaque byte blob — only
  its whole-file `sha256_file` hash was checked, never its content — while
  the caller (the CLI command) built the enforced
  `AttemptProvenancePolicy` directly from the CLAIM's own fields
  (`typed.authorized_branch`, `typed.authorized_commit_sha`, ...). A claim
  could therefore assert any branch/commit/ancestor/R-KV identity it
  wanted, fully decoupled from the actual committed document — confirmed
  directly against the pre-repair test fixture, which wrote the document
  as the literal string `"synthetic Stage B authorization"` and still
  passed full verification.
- **Repair:** a new module,
  `kvcot.discovery.b2a_r3_authorization_document`, requires exactly one
  `<!-- BEGIN/END B2A-R3 AUTHORIZATION JSON -->` marker pair containing
  exactly one ` ```json ` fence, with duplicate-key detection, unknown/
  missing-field rejection, and Stage-B-only
  (`maximum_candidates`/`phase_wall_time_limit_seconds`) vs. Stage-C-only
  (`qualification_artifact_canonical_sha256`/`selected_manifest_sha256`/
  `selected_manifest_hash_algorithm`) field-exclusivity enforcement.
  `policy_from_authorization_document` builds the `AttemptProvenancePolicy`
  entirely from the parsed document.
  `verify_authorization_preconditions` now parses the document FIRST,
  derives the policy from it, and requires the claim's fields to equal
  the document's fields (never the reverse) — the `policy` parameter is
  removed from its signature entirely, closing the loophole structurally,
  not just by convention. `VerifiedAuthorizationContext` now also exposes
  `maximum_candidates`/`phase_wall_time_limit_seconds`, sourced from the
  document, for the coordinator (Finding 6) to read.
- **Source:** `kvcot.discovery.b2a_r3_authorization_document`,
  `kvcot.discovery.b2a_r3_authorization`, `kvcot.cli`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_authorization_document.py`
  (45 cases: valid Stage B/C documents parse; the historical placeholder
  text now fails; missing/duplicate markers; wrong fence language; stray
  text inside the markers but outside the fence; two fenced blocks;
  invalid JSON; duplicate JSON keys; every required field individually
  missing; noncanonical stage; invalid authorization ID; Stage-B/C field
  exclusivity in both directions; wrong repository;
  `authorization_document_sha256` forbidden inside the block itself), plus
  `tests/unit/discovery/test_b2a_r3_authorization.py` rewired end to end
  (its `_verified_stage_b`/`_stage_c_inputs` fixtures now write real
  embedded-JSON documents and no longer pass an externally-constructed
  `policy`).
- **Status:** repaired.

### Finding 4 — non-global claim path

- **Root cause:** `claim_authorization`'s public signature accepted an
  arbitrary `claims_root: str | Path` parameter — any caller (in
  principle, a future real Stage-B entry point) could direct consumption
  at any directory, not the one deterministic
  `results/decisions/b2a_r3_authorization_claims/<authorization_id>.json`
  path protocol §14.4.1 requires.
- **Repair:** the public API is now `claim_authorization(payload, *,
  repository_root, verified_context, git_state)`. The exact claim path is
  derived internally as `repository_root /
  global_claim_path(authorization_id)`; both the payload's and the
  verified context's own `global_claim_path` are required to agree with
  it. Immediately before the exclusive-create call, Git/worktree state is
  reverified against the context's policy (narrowing the window between
  the earlier `verify_authorization_preconditions` call and actual
  consumption). The private, low-level `_create_authorization_claim` now
  takes an explicit `claim_path` for direct race testing, never a root a
  caller could redirect.
- **Source:** `kvcot.discovery.b2a_r3_authorization`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_authorization.py`
  (`test_claim_authorization_uses_exact_global_path_under_repository_root`,
  `test_claim_authorization_no_longer_accepts_claims_root_parameter`,
  `test_claim_authorization_reverifies_worktree_immediately_before_claim`),
  the existing 20-trial threaded and 5-trial multiprocess exclusivity
  races re-run green against the new `claim_path`-based low-level helper.
- **Status:** repaired.

### Finding 5 — missing canonical FullKV R3 adapter

- **Root cause:** no schema-validated, canonical path existed from a real
  (or realistically synthetic) FullKV worker result into
  `B2AR3FullKVQualificationEvidence` — every existing test hand-assembled
  evidence fields directly, and a future Stage B integration would have
  had no better option.
- **Repair:** a new module, `kvcot.discovery.b2a_r3_worker_adapter`, adds
  `FullKVWorkerResultR3` (a new, versioned, backward-compatible schema —
  the historical B2A-R1/R2 `FullKVWorkerResult` is untouched) and
  `adapt_fullkv_worker_result_to_r3_evidence`, the one authoritative
  conversion function. It rejects a legacy worker result or an arbitrary
  dict outright; binds the result to an exact candidate ordinal (source
  index, unique ID, three row hashes); rejects a dataset/model/tokenizer
  identity that disagrees with the frozen contract; validates timing and
  memory evidence separately via the Finding-2 contract before extracting
  peak CUDA bytes from memory evidence alone (never a caller-supplied
  value); and recomputes the static R-KV compaction schedule and runtime
  prediction fresh via the existing canonical functions, never trusting
  worker-supplied schedule/runtime data. `B2AR3FullKVQualificationEvidence`
  itself has no `thinking_span_valid`/`trace_complete` fields at all —
  those are only ever derived downstream by the qualification evaluator,
  so a "manually fabricated thinking result" has no field to be smuggled
  through in the first place.
- **Source:** `kvcot.discovery.b2a_r3_worker_adapter`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_worker_adapter.py` (24
  cases: canonical conversion succeeds; legacy `FullKVWorkerResult`
  refused; an arbitrary dict refused; wrong candidate ordinal/source
  index/unique ID/row hash refused; wrong dataset/model/tokenizer
  identity refused (4 cases); wrong generated-token hash and wrong
  generation-config hash rejected at schema construction; boolean token ID
  rejected; wrong role/answer-status/schema-version rejected; unknown
  field rejected; a memory phase spliced into timing evidence refused; a
  missing memory phase refused; peak bytes are extracted from memory
  evidence, never caller-supplied; the adapter's own signature accepts no
  gate-map parameter; a subprocess-isolated import-safety test proving no
  R-KV/torch/transformers import).
- **Status:** repaired.

### Finding 6 — missing qualification coordinator/writer

- **Root cause:** no CPU-testable path existed from a sequence of
  attempted outcomes to a complete, hash-verified qualification artifact,
  and no sequential coordinator existed to drive first-pass qualification
  against a verified Stage-B authorization context.
- **Repair:** `kvcot.discovery.b2a_r3_artifacts` gains
  `build_qualification_artifact` (independently semantically re-derives
  every attempted outcome; enforces contiguous ordinals starting at zero
  and the protocol's 8-candidate ceiling; replays first-pass selection via
  the same `select_first_qualified_r3` the artifact schema itself already
  uses; validates `stopped_reason` against one of four frozen values that
  must agree with whether a candidate actually qualified; populates every
  top-level identity field from the frozen contract and the independently-
  verified candidate manifest, never from a caller parameter) and
  `write_qualification_artifact_atomic` (full strict re-validation before
  write; refuses to overwrite an existing artifact — write-once
  immutability; reads back and re-verifies the written bytes). A new
  module, `kvcot.discovery.b2a_r3_qualification_coordinator`, adds
  `run_b2a_r3_qualification_coordinator`: requires a genuinely verified
  Stage B `VerifiedAuthorizationContext` (rejecting a hand-constructed
  fake one via the same private verification-token sentinel
  `claim_authorization` already checks); reads
  `maximum_candidates`/`phase_wall_time_limit_seconds` only from that
  context, never a CLI argument or hard-coded default; iterates candidates
  in exact ordinal order with a hard cap at
  `min(authorized_maximum, protocol_maximum=8, len(manifest.candidates))`;
  checks phase-wide remaining wall time before every worker launch;
  applies the one frozen per-candidate timeout
  (`PER_CANDIDATE_WORKER_TIMEOUT_SECONDS=7200`, rejecting any other
  value); converts every worker result through the Finding-5 adapter;
  stops immediately at the first passing candidate, never evaluating a
  later ordinal; and fails closed (`QualificationCoordinatorRefused`) on
  malformed worker evidence, never silently recording it as an ordinary
  scientific rejection. The coordinator never writes to any filesystem
  path itself — a future, separately-authorized Stage-B entry point would
  decide whether/where to call the atomic writer.
- **Source:** `kvcot.discovery.b2a_r3_artifacts`,
  `kvcot.discovery.b2a_r3_qualification_coordinator`.
- **Tests:** `tests/unit/discovery/test_b2a_r3_qualification_coordinator.py`
  (26 cases covering the builder, the atomic writer, and the coordinator:
  candidate 0 passes with exactly one worker call; candidate 0 fails,
  candidate 1 passes with exactly two calls; all eight fail with exactly
  eight calls; an authorization-limited maximum of three never calls a
  fourth; a pass at ordinal 2 never evaluates ordinals 3-7; a worker
  returning evidence bound to the wrong candidate is refused; a legacy
  worker result is refused; a per-candidate timeout (both the module's own
  `CandidateWorkerTimeout` and the stdlib `TimeoutError`) stops cleanly
  with exactly the right attempted count; phase-wide wall-time exhaustion
  before the next candidate makes no additional call; an unverified fake
  context is refused; a swapped (structurally valid but differently-
  hashed) candidate manifest is refused; every returned artifact passes
  full independent semantic verification across three scenarios; no
  production file is ever written; a wrong `per_candidate_timeout_seconds`
  is refused; a subprocess-isolated import-safety test).
- **Status:** repaired.

## Candidate manifest

`configs/discovery/b2a_r3_candidate_manifest.json` content is unchanged;
its `canonical_sha256` is unchanged
(`b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42`); it is
reverified under the (unmodified) strict verifier throughout this round's
new tests.

## Validation status

All local validation used `CUDA_VISIBLE_DEVICES=""` and `PYTHONPATH=src`.

- `python -m compileall -q src tests`: pass.
- Focused Finding 1-6 test files (new + touched): all green, individually
  and together, at every intermediate commit in this round.
- `python -m pytest tests/unit/discovery/ -k "b2a_r3" tests/unit/test_cli_b2a_r3.py -q`:
  466 passed, 847 deselected.
- Full CPU suite (`python -m pytest -m "not gpu" -q`): 1822 passed, 14
  deselected (the 14 deselected are the pre-existing, unmodified GPU-only
  tests -- the same count as every prior Stage-A validation pass).
- `git diff --check`: pass.
- `git submodule status third_party/R-KV`: unchanged
  (`45eaa7d69d20b7388321f077020a610d9afb65bd`).

## Authorization boundary (restated)

```text
STAGE B FULLKV QUALIFICATION: NOT AUTHORIZED
STAGE C B2A-R3 EXECUTION: NOT AUTHORIZED
GPU RENTAL: PROHIBITED UNTIL INDEPENDENT RE-AUDIT PASSES
R-KV DURING QUALIFICATION: PROHIBITED
B2B: BLOCKED
FAITHKV METHOD IMPLEMENTATION: BLOCKED
NEXT REQUIRED ACTION: INDEPENDENT RE-AUDIT OF THE FINAL STEP-3R4 SHA
BRANCH MERGED: NO
```
