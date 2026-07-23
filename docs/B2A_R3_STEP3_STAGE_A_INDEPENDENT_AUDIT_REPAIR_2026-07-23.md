# B2A-R3 Step 3 Stage-A independent-audit repair (2026-07-23)

```text
STEP 3 STAGE-A AUDIT REPAIRS IMPLEMENTED —
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED
```

This record covers the bounded, CPU-only repair of the independent audit
findings against starting SHA
`6778fd119c33f5025f328321a33eb7c62ad86d20` on
`research/b2a-r3-runtime-qualified-calibration`. It is an implementation
and regression-test record, not an independent approval of its own work.
The frozen B2A-R3 protocol retains precedence, and Stage B remains blocked
until a separate independent re-audit accepts the final repair SHA.

## Authorization boundary

The round used pure Python, strict schemas, synthetic evidence,
temporary-directory filesystem tests, spawned CPU processes, Git
inspection and CPU-only verification commands. It did not initialize
CUDA, load model weights, load a tokenizer for execution, run FullKV or
R-KV, create a real qualification artifact, replace the real selected
manifest, create a real authorization claim or attempt directory, execute
B2A-R3 or B2B, or implement a FaithKV method.

The pinned R-KV gitlink remains
`45eaa7d69d20b7388321f077020a610d9afb65bd`. Historical B2A-R1/R2
artifacts and the committed B2A-R3 candidate manifest were not modified or
regenerated.

## Repair ledger

### P0-1 — qualification booleans

- Root cause: `CandidateQualificationOutcomeR3.conditions` and its
  `qualified`/`failed_conditions` summaries were accepted without replay
  from raw evidence.
- Repair: persisted every raw input needed by the frozen gates and added
  `rederive_and_verify_qualification_outcome()`. It locates the exact
  candidate row, reconstructs strict immutable evidence, invokes the same
  authoritative 27-condition evaluator, and requires exact condition,
  qualification and tuple-ordered failure equality.
- Source: `b2a_r3_qualification.rederive_and_verify_qualification_outcome`;
  `b2a_r3_artifacts.verify_qualification_artifact`.
- Regression: canonically rehashed answer-status, cap, batch-size, memory,
  prompt-hash and runtime-projection contradictions are rejected.
- Status: repaired.

### P0-2 — schedule derivation

- Root cause: qualification trusted supplied positions, counts and eligible
  indices.
- Repair: evidence now carries frozen budget/divide length and exact prompt
  and generated counts. Construction and replay call
  `kvcot.analysis.rkv_schedule.predicted_compaction_event_positions` and
  `kvcot.discovery.pass1.eligible_event_positions`; stored fields are only
  checked echoes.
- Source: `b2a_r3_qualification.evaluate_b2a_r3_qualification_conditions`;
  `build_qualification_outcome`; `rederive_and_verify_qualification_outcome`.
- Regression: changed/reordered positions, altered count, fabricated
  eligible index, ineligible future event and alternate length-consistent
  schedules are rejected after canonical rehashing.
- Status: repaired.

### P0-3 — timing verification

- Root cause: a nonempty timing list with completed rows could satisfy the
  R3 helper without the historical FullKV phase contract.
- Repair: factored pure `fullkv_qualification_timing_complete()` around the
  existing phase vocabulary and exact phase order/multiplicity. It checks
  strict finite non-boolean timestamps/durations, nonnegativity, exact
  duration arithmetic, completion/failure fields, synchronization flags
  and wall-time consistency. The historical
  `timing_contract_satisfied()` behavior was not changed.
- Source: `final_contract.fullkv_qualification_timing_complete`;
  `b2a_r3_qualification._fullkv_timing_complete`.
- Regression: generic, missing, duplicate, negative, NaN, infinity,
  backwards, duration-mismatched, incomplete, failed, unsynchronized,
  reordered and wall-time-mismatched evidence is rejected; historical
  timing suites remain green.
- Status: repaired.

### P0-4 — frozen candidate identity

- Root cause: candidate manifests could redefine protocol identity fields
  and choose their own config path/hash.
- Repair: strict schemas bind dataset, model, tokenizer, revisions, budget,
  config path, generation/exclusion hashes, counts and level mixture to
  contract constants. `config_sha256` is lowercase hexadecimal and the
  production CLI supplies the hash of the frozen committed config.
  Dataset reproduction fetches the frozen contract identity, not manifest
  echoes.
- Source: `b2a_r3_candidates.CandidateManifestR3`;
  `verify_candidate_manifest_structure`; CLI candidate handlers.
- Regression: wrong repository/revisions/model/tokenizer/budget/config
  path/hash, uppercase hash, numeric-string budget and self-consistent
  alternate dataset identity are rejected.
- Status: repaired.

### P0-5 — fake candidate manifest

- Root cause: a self-hashed partial dictionary was treated as sufficient
  candidate evidence.
- Repair: evaluator, outcome builder, artifact verifier and freezer require
  the complete strict manifest and bind ordinal, source index, unique ID
  and all row hashes to its exact candidate.
- Source: `verify_candidate_manifest_structure` and all qualification/
  freeze entry points.
- Regression: minimal self-hashed objects, missing candidates, extra
  fields, wrong schema, absent ordinal and candidate identity/hash
  substitutions are rejected.
- Status: repaired.

### P0-6 — generated-token identity

- Root cause: the stored generated-token hash was not reproduced from the
  natural token array.
- Repair: token IDs are strict integers (never booleans), count must equal
  array length and `sha256_int_ids()` must reproduce the stored hash.
  Disagreement is malformed provenance and raises rather than becoming an
  ordinary scientific failure.
- Source: `b2a_r3_qualification.FullKVQualificationEvidenceR3` and
  `CandidateQualificationOutcomeR3` validators/replay.
- Regression: changed ID/hash/count/order, empty substitution and boolean
  IDs are rejected.
- Status: repaired.

### P0-7 — authorization/provenance coupling

- Root cause: public claim consumption only checked claim schema/self-hash.
- Repair: `verify_authorization_preconditions()` verifies the exact dated
  document path, committed file and byte hash; claim/policy/observed Git
  repository, branch, commit, ancestors and R-KV identity; clean pre-claim
  worktree; strict candidate manifest; and the complete stage-specific
  artifact chain. It returns a privately marked
  `VerifiedAuthorizationContext`; `claim_authorization()` refuses any
  other context or payload.
- Source: `b2a_r3_authorization.verify_authorization_preconditions`;
  `claim_authorization`.
- Regression: missing/tampered/uncommitted documents, repository/branch/
  commit/R-KV/ancestor disagreements, manifest mismatch, Stage-C artifact
  mismatches and direct unverified consumption are rejected.
- Status: repaired.

### P0-8 — post-claim allowlist

- Root cause: callers could nominate arbitrary dirty paths, including
  configuration files.
- Repair: removed the generic allowlist. `ActiveAuthorizationPaths` is
  derived from a verified claim, validates deterministic claim and attempt
  paths, and carries a private derivation marker. Only the exact claim path
  and the active attempt root/children represented by Git status are
  accepted.
- Source: `b2a_r3_provenance.ActiveAuthorizationPaths`;
  `verify_attempt_provenance`.
- Regression: config/source paths, different claims/attempts, siblings,
  traversal, common-prefix paths and forged typed contexts are rejected.
- Status: repaired.

### P0-9 — authorization CLI verification

- Root cause: `verify-b2a-r3-authorization` performed schema/self-hash
  validation only.
- Repair: the command now constructs the CPU-safe Git provider and calls
  the same semantic precondition verifier with document, candidate and all
  required Stage-C artifacts. It never consumes a claim.
- Source: `kvcot.cli` authorization parser/handler;
  `b2a_r3_provenance.SubprocessGitStateProvider`.
- Regression: end-to-end CLI semantic verification and tampered-claim
  refusal; guarded subprocess import tests.
- Status: repaired.

### P1-1 — multiprocess race

- Root cause: atomic exclusivity was exercised only with threads.
- Repair: retained exclusive `os.open(O_CREAT|O_EXCL)` creation and added
  five two-process races using `multiprocessing.get_context("spawn")` and
  a process-safe start event.
- Source: private `b2a_r3_authorization._create_authorization_claim`.
- Regression: every trial has exactly one success, one
  `AuthorizationAlreadyConsumed`, one nonempty filesystem entry and no
  overwrite; the 20 thread trials remain.
- Status: repaired.

### P1-2 — renderer validation

- Root cause: the freezer trusted an unvalidated renderer dataclass.
- Repair: `PromptRenderingResult` is a strict, frozen, extra-forbidden
  schema. It verifies lowercase hashes, strict token IDs, count/hash
  reproduction, tokenizer revision binding and the exact frozen rendering
  convention.
- Source: `b2a_r3_freeze.PromptRenderingResult`;
  `_verify_rendering_binding`.
- Regression: wrong hash/count/token/revision/roles/generation flag/
  tokenize flag/special-token note, boolean token and uppercase hash are
  rejected.
- Status: repaired.

### P1-3 — selection provenance replay

- Root cause: selection verification compared object hashes without
  replaying the semantic freeze decision and every selected-row field.
- Repair: `verify_selection_provenance()` first runs the strict candidate
  and qualification verifiers plus `verify_freeze_chain()`, then binds
  selected ordinal/ID, dataset/index/row/gold identity, reproducible prompt
  tokens, qualification prompt identity, tokenizer revision, external
  selected-manifest hash and all provenance links.
- Source: `b2a_r3_freeze.verify_freeze_chain`;
  `verify_selection_provenance`.
- Regression: canonically rehashed provenance/selected-manifest field
  changes, valid qualification/candidate-chain replacement and introduction
  of an earlier qualifying row are rejected.
- Status: repaired.

## Candidate manifest

`configs/discovery/b2a_r3_candidate_manifest.json` content is unchanged;
its `canonical_sha256` is unchanged; it is reverified under the repaired
strict verifier and the frozen committed config hash.

## Validation status

All local validation used `CUDA_VISIBLE_DEVICES=""` and `PYTHONPATH=src`.

- `python -m compileall -q src tests`: pass.
- Independent-audit attack file: 49 passed.
- Aggregate R3 suite: 330 passed.
- Historical qualification/contract/timing/schedule/worker/MATH regression
  set: 191 passed.
- Subprocess import guards and source scan: 6 passed, 9 deselected.
- `python -m pytest -m "not gpu" -q`: 1686 passed, 14 deselected.
- `git diff --check`: pass.

The first aggregate attempts exposed an intermittent historical test issue:
exact answer identity unnecessarily launched the isolated symbolic verifier
and could time out under suite load, making a controlled worker exit before
its injected failure point. The exact-identity fast path now records a
typed equivalent result without a subprocess; every non-identical
comparison retains the frozen symbolic verifier and timeout. The focused
worker/MATH regressions and subsequent complete CPU run are green.

The first pushed repair SHA's Linux CI run `29988322217` exposed a
platform-specific exact-byte config identity: Windows checked out the
frozen config with CRLF (the candidate manifest's recorded `de8ac65…`
hash), while Linux checked it out with LF (`98d958…`). The protocol's
required `sha256_file` algorithm and candidate manifest were preserved;
`.gitattributes` now freezes CRLF for that one config on every platform,
with a focused checkout-byte/hash regression. The replacement remote CI
result is recorded in the final audit report after the new final SHA is
pushed. Until then: `REMOTE CPU CI UNCONFIRMED`.

## Final state

```text
STAGE B FULLKV QUALIFICATION: NOT AUTHORIZED
STAGE C B2A-R3 EXECUTION: NOT AUTHORIZED
GPU RENTAL: PROHIBITED
B2B: BLOCKED
FAITHKV METHOD IMPLEMENTATION: BLOCKED
NEXT REQUIRED ACTION: INDEPENDENT RE-AUDIT OF THE FINAL REPAIR SHA
BRANCH MERGED: NO
```
