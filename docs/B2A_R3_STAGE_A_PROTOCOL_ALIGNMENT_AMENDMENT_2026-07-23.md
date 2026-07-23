# B2A-R3 Stage-A Protocol Alignment Amendment (dated 2026-07-23)

```text
STEP 3R4 CPU PROTOCOL ALIGNMENT AND STAGE-B READINESS REPAIRS IMPLEMENTED —
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED
```

This is a dated amendment to
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` (the frozen,
independently-audited B2A-R3 protocol, §22 disposition: "INDEPENDENT STEP
2B RE-AUDIT: PASS", `docs/B2A_R3_STEP2B_INDEPENDENT_REAUDIT_2026-07-23.md`).
It repairs six findings identified against Step 3 Stage-A implementation
SHA `7062f3cb8a6f555d3b67cf9e9be3bd6710e78120`. It supersedes nothing in
§1-§21 of the base protocol except the narrow qualification-outcome
schema/version fields this document names explicitly below.

## 1. Why the qualification artifact and protocol may be safely version-bumped

**No real B2A-R3 qualification artifact has yet been produced.** Every
qualification artifact that exists anywhere in this repository's tests is
synthetic, built under `tmp_path` or from injected fixtures, and no
production code path (CLI or otherwise) has ever written a real file at
`results/decisions/b2a_r3_qualification.json` (protocol §14.1 forbids this
under Stage A regardless of audit outcome). Because zero real qualification
outcomes exist to reinterpret, the qualification-outcome schema and its two
version identifiers can be corrected now, before any real execution, with
no risk of silently reinterpreting a historical artifact — exactly the
same reasoning the base protocol itself already applied when versioning
new B2A-R3 artifacts distinctly from B2A-R1/R2 (§12.9).

**Unchanged by this amendment:**

- Candidate ordering, the candidate manifest, and its canonical hash
  (`configs/discovery/b2a_r3_candidate_manifest.json`,
  `canonical_sha256 = b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42`)
  — verified unchanged by this repair (§9 below).
- Historical B2A-R1 and B2A-R2 schemas, artifacts, and verification code —
  untouched.
- Every threshold, budget, model, dataset, revision, or gate frozen by
  `CLAUDE.md` §1/§1a-§1i/§4/§4a-§4c or the base protocol's §4/§7/§8/§9/§10.
- `candidate_order_protocol_version`, `runtime_predictor_version`,
  `selection_protocol_version`, and the authorization-claim artifact
  schema version (`faithkv-b2a-r3-authorization-claim-v1`) — none of these
  are bumped by this amendment.
- The candidate-manifest schema version
  (`faithkv-b2a-r3-candidate-manifest-v1`) is unchanged.

**Superseded by this amendment:** only the B2A-R3 qualification-outcome
and qualification-artifact portions of the base protocol necessary for
independent semantic replay and eventual Stage-B execution readiness —
namely the per-candidate outcome field set (§5 below), the FullKV
timing/memory validation contract (§6), the authorization-document format
(§7), the global authorization-claim path binding (§8), the canonical
FullKV worker-result adapter (§9), and the qualification
artifact/coordinator architecture (§10). Nothing else in the base protocol
is reopened.

## 2. Version bumps

```text
QUALIFICATION_ARTIFACT_SCHEMA_VERSION:
    old: "faithkv-b2a-r3-qualification-artifact-v1"
    new: "faithkv-b2a-r3-qualification-artifact-v2"

QUALIFICATION_PROTOCOL_VERSION:
    old: "faithkv-b2a-r3-qualification-v1"
    new: "faithkv-b2a-r3-qualification-v2"
```

Not changed by this amendment (repeated here only for cross-reference,
per §12.9 of the base protocol):

```text
candidate_order_protocol_version   = "faithkv-b2a-r3-row-order-v1"
runtime_predictor_version          = "faithkv-b2a-r3-runtime-predictor-v1"
selection_protocol_version         = "faithkv-b2a-r3-selection-v1"
authorization claim schema version = "faithkv-b2a-r3-authorization-claim-v1"
candidate manifest schema version  = "faithkv-b2a-r3-candidate-manifest-v1"
```

## 3. Finding 1 — qualification schema v1→v2 field set

### 3.1 Root cause

The v1 per-candidate outcome schema (`CandidateQualificationOutcomeR3`)
duplicated four artifact-level identity fields
(`candidate_manifest_canonical_sha256`, `config_sha256`, `budget`,
`divide_length`) onto every one of up to eight attempted outcomes, and
additionally persisted a nested `runtime_prediction` object alongside the
already-flattened runtime fields it echoes — a second source of truth for
the same seven numbers. `generation_config_sha256` was used, ambiguously,
as both "the worker's observed hash" (on the outcome) and "the frozen
expected hash" (on the artifact), with no field name distinguishing which
meaning a given occurrence carried.

### 3.2 Exact v2 per-candidate qualification-outcome field table

Every field below is **required** unless marked nullable. Unknown fields
are rejected (`extra="forbid"`). A missing field is rejected. This is the
one frozen table Step 3R4's implementation and its drift test
(`test_qualification_outcome_v2_field_set_matches_contract`) must match
exactly.

**Candidate identity**

| Field | Type |
|---|---|
| `candidate_ordinal` | int |
| `source_example_index` | int |
| `unique_id` | str |
| `raw_row_sha256` | str (64-hex) |
| `problem_sha256` | str (64-hex) |
| `gold_answer_sha256` | str (64-hex) |

**Worker identity**

| Field | Type |
|---|---|
| `worker_dataset_repo` | str |
| `worker_dataset_config` | str |
| `worker_dataset_split` | str |
| `worker_dataset_revision` | str |
| `worker_model_name` | str |
| `worker_model_revision` | str |
| `worker_tokenizer_name` | str |
| `worker_tokenizer_revision` | str |

**Prompt identity**

| Field | Type |
|---|---|
| `expected_prompt_token_ids_sha256` | str (64-hex) |
| `observed_prompt_token_ids_sha256` | str (64-hex) |
| `prompt_token_count` | int |

**Generated-trace evidence**

| Field | Type |
|---|---|
| `natural_generated_token_ids` | list[int] (strict, no bool) |
| `generated_token_count` | int |
| `generated_token_ids_sha256` | str (64-hex) |
| `total_processed_tokens` | int |

**Answer and thinking evidence**

| Field | Type |
|---|---|
| `cap_hit` | bool |
| `extracted_answer` | str, nullable |
| `answer_verification_status` | str (`correct`/`incorrect`/`unverifiable`) |
| `think_parse_status` | str |
| `think_start_index` | int, nullable |
| `think_end_index` | int, nullable |
| `generation_prompt_preopened_think` | bool |
| `thinking_span_valid` | bool |
| `trace_complete` | bool |

**FullKV execution evidence**

| Field | Type |
|---|---|
| `fullkv_wall_seconds` | float |
| `fullkv_timing_evidence` | list[object] (protocol §6 timing vocabulary only) |
| `requested_device` | str |
| `parameter_placement_evidence` | object |
| `actual_batch_size` | int |
| `peak_cuda_allocated_bytes` | int |
| `peak_cuda_reserved_bytes` | int |
| `peak_cuda_tracked_bytes` | int |

**Static schedule evidence**

| Field | Type |
|---|---|
| `predicted_compaction_event_positions` | list[int] |
| `predicted_event_count` | int |
| `eligible_event_indices` | list[int] |
| `eligible_event_count` | int |

**Runtime prediction evidence (flattened only — no nested object)**

| Field | Type |
|---|---|
| `reference_seconds_per_token` | float |
| `predicted_example_seconds` | float |
| `predicted_pair_seconds` | float |
| `projected_total_seconds` | float |
| `projected_gpu_hours` | float |
| `safety_multiplier` | float |
| `runtime_predictor_version` | str |

**Generation configuration**

| Field | Type |
|---|---|
| `worker_generation_config_sha256` | str (64-hex) — the worker-observed value only |

**Qualification result**

| Field | Type |
|---|---|
| `conditions` | object (27-key bool map) |
| `qualified` | bool |
| `failed_conditions` | list[str] |

### 3.3 Removed from the per-outcome record (v1→v2)

```text
candidate_manifest_canonical_sha256   -- artifact-level only (already present there)
config_sha256                        -- artifact-level only (already present there)
generation_config_sha256             -- renamed worker_generation_config_sha256;
                                         the ambiguous "expected" meaning stays
                                         artifact-level only, under the SAME
                                         unchanged field name there
runtime_prediction (nested object)   -- redundant with the flattened fields above
budget                               -- frozen protocol constant (BUDGET),
                                         imported from b2a_r3_contract, never
                                         persisted per-outcome
divide_length                        -- frozen protocol constant (DIVIDE_LENGTH),
                                         same treatment
```

These identities are not lost: `budget`/`divide_length`/`config_sha256`/
`candidate_manifest_canonical_sha256`/`generation_config_sha256` (expected)
all remain exactly once, at the qualification artifact's top level
(protocol §12.6, unchanged by this amendment), which every outcome in
`attempted` is already read alongside.

### 3.4 Verifier rerun requirement

The verifier (`rederive_and_verify_qualification_outcome`) recomputes the
complete runtime prediction fresh via
`kvcot.discovery.b2a_r3_runtime.predict_runtime(total_processed_tokens)`
and requires exact equality, field by field, against the seven persisted
flattened runtime fields above — never trusting a stored nested object
(there is none in v2) and never trusting the flattened fields as
self-certifying.

## 4. Finding 2 — FullKV timing/memory contract

The Step 3 Stage-A `fullkv_qualification_timing_complete` helper
constructed its own ad hoc 12-phase vocabulary
(`_FULLKV_QUALIFICATION_PHASE_ORDER`) that spliced two MEMORY phases
(`before_model_load`, `post_load_baseline`) into the TIMING phase list,
and placed `answer_verification` before
`fullkv_complete_natural_generation` — backwards relative to the real,
already-frozen `FULLKV_REQUIRED_TIMING_PHASES` vocabulary
(`src/kvcot/discovery/final_contract.py`), which every other FullKV
timing consumer in this repository (`timing_contract_satisfied`, the real
worker's own phase emission order) already uses. This amendment corrects
`fullkv_qualification_timing_complete` to validate against
`FULLKV_REQUIRED_TIMING_PHASES` exactly (10 phases, `fullkv_decode`
variable-multiplicity `>= 1`, every other phase exactly once, in the
canonical worker order), and adds a separate
`fullkv_qualification_memory_complete` validating FullKV memory-phase
evidence against the existing `FULLKV_REQUIRED_MEMORY_PHASES`/
`FULLKV_MEMORY_EXACT_MULTIPLICITY` contract. Memory phases must never
appear inside `fullkv_timing_evidence`.

## 5. Finding 3 — machine-readable authorization document

Prior to this amendment, `verify_authorization_preconditions` treated the
dated Markdown authorization document as an opaque byte blob: only its
`sha256_file` hash was checked, never its content. The claim itself
supplied every enforceable field (`authorized_branch`, `authorized_commit_sha`,
`required_ancestor_shas`, `required_rkv_sha`, ...), which
`verify_authorization_preconditions`'s caller then wrapped into an
`AttemptProvenancePolicy` and handed back in — a circular design in which
the claim defined its own policy. This amendment freezes an exact embedded
JSON block format for the authorization document
(`src/kvcot/discovery/b2a_r3_authorization_document.py`) and requires the
policy to be constructed from the PARSED DOCUMENT, never from the claim;
the claim's fields are then required to equal the parsed document's
fields, never the reverse.

## 6. Finding 4 — global authorization-claim path

`claim_authorization`'s public signature accepted an arbitrary
`claims_root` parameter, letting a caller direct claim consumption at any
directory. This amendment binds the public API to
`repository_root`/`verified_context`/`git_state` only; the exact
repository-relative claim path is derived internally via
`kvcot.discovery.b2a_r3_contract.global_claim_path`, and Git/worktree
state is reverified immediately before the exclusive-create call.

## 7. Finding 5 — canonical FullKV worker-result adapter

No canonical, schema-validated path existed from a real (or realistically
synthetic) `FullKVWorkerResult`-shaped payload into
`B2AR3FullKVQualificationEvidence` — a future Stage B integration would
have had no choice but to hand-assemble evidence fields, in the same way
this repair's own audit-fixture helpers do today for tests. This amendment
adds a versioned `FullKVWorkerResultR3` schema and one authoritative
adapter (`kvcot.discovery.b2a_r3_worker_adapter
.adapt_fullkv_worker_result_to_r3_evidence`).

## 8. Finding 6 — qualification artifact builder, atomic writer, coordinator

No CPU-testable path existed from a sequence of attempted outcomes to a
complete, hash-verified qualification artifact, nor a sequential
first-pass coordinator wired to a verified Stage-B authorization context.
This amendment adds `build_qualification_artifact`,
`write_qualification_artifact_atomic`, and
`run_b2a_r3_qualification_coordinator`
(`kvcot.discovery.b2a_r3_qualification_coordinator`).

## 9. Candidate manifest unchanged status

```text
configs/discovery/b2a_r3_candidate_manifest.json canonical_sha256:
b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42
```

This amendment does not regenerate, reorder, or otherwise touch the
candidate manifest. Its canonical hash is reverified unchanged as part of
this repair's validation (§15 of
`docs/B2A_R3_STAGE_B_READINESS_REPAIR_2026-07-23.md`).

## 10. Authorization boundary (restated, not expanded)

```text
STAGE B FULLKV QUALIFICATION: NOT AUTHORIZED
STAGE C B2A-R3 EXECUTION: NOT AUTHORIZED
GPU / CUDA / MODEL INFERENCE: PROHIBITED
R-KV DURING QUALIFICATION: PROHIBITED
B2B: BLOCKED
FAITHKV METHOD IMPLEMENTATION: BLOCKED
```

This amendment authorizes exactly the same scope §1e/§1h of `CLAUDE.md`
already authorized for Step 3 Stage-A: CPU-only implementation, CPU-only
tests, dependency-injected/synthetic evidence, and CPU-only planning/
verification commands. It does not itself authorize Stage B, and it does
not self-certify — a separate, genuinely independent re-audit of the
Step 3R4 repair SHA is the next required action, exactly as it was for
Step 3 Stage-A itself (`docs/B2A_R3_STEP2B_INDEPENDENT_REAUDIT_2026-07-23.md`).
