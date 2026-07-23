# B2A-R3 Stage-B Binding Repair (dated 2026-07-23)

This document records the repair for the independent re-audit that followed
commit `6828e2f263da64bf552e605fb39a140af5be0c07`.

## Authority Boundary

This is a CPU-only code and governance repair. It does not authorize a real
Stage-B FullKV qualification run, any CUDA initialization, any model or
tokenizer weight load, Stage-C B2A-R3 execution, B2B execution, or any
FaithKV method implementation.

Stage B remains blocked until both conditions are true:

1. A genuinely independent re-audit accepts the final repair SHA.
2. Remote CI is green for that exact final SHA.

The production Stage-B command contract may exist in source so it can be
audited, but it must not be treated as runtime authorization.

## Repairs

### Persisted Stage-B Binding

`verify_persisted_stage_b_authorization_binding()` no longer reuses the
pre-claim/current-HEAD provenance gate. That gate intentionally requires
the current checkout to equal the authorized commit and be clean before
claim consumption, which is the wrong lifecycle state for downstream
verification after Stage-B outputs exist or after HEAD advances for
Stage-C authorization.

The persisted binding path now verifies:

- the claim file is at the deterministic global claim path;
- the claim is canonically self-hashed and schema-valid;
- the Stage-B authorization document path matches the dated Stage-B
  pattern;
- the Stage-B authorization document is tracked in the current checkout
  and its byte hash matches the claim;
- the document's machine-readable fields match the claim;
- the historical Stage-B authorized commit exists;
- required ancestors verify against the historical authorized commit;
- the R-KV gitlink at the historical authorized commit matches the
  claim/document pin;
- the supplied candidate manifest and config identity match the claim.

Stage-C authorization verification still independently verifies the
current Stage-C checkout against the Stage-C claim and document before it
uses the persisted Stage-B binding.

### Subprocess Determinism

The R3 Stage-B worker subprocess now reuses the canonical worker
environment constructor:

```text
kvcot.discovery.b2a_workers._worker_subprocess_env(CONFIG_PATH)
```

The child process receives:

```text
PYTHONHASHSEED=13
TOKENIZERS_PARALLELISM=false
```

This preserves the frozen determinism boundary before Python interpreter
startup.

### Qualification Artifact Schema v4

`QUALIFICATION_ARTIFACT_SCHEMA_VERSION` is now:

```text
faithkv-b2a-r3-qualification-artifact-v4
```

Schema v4 adds required persisted Stage-B authorization binding fields:

- `authorized_phase_wall_time_limit_seconds`
- `stage_b_authorization_id`
- `authorization_document_sha256`
- `authorization_claim_canonical_sha256`

No real qualification artifact has been produced under v2 or v3, so this
does not reinterpret historical output.

### Production Stage-B Command Contract

The audited command surface is:

```text
kvcot run-b2a-r3-stage-b-qualification --claim <claim-json>
```

The public command does not expose candidate order, maximum candidates,
phase wall-time limit, per-candidate timeout, output path, claims root,
config path, candidate-manifest path, or repository root as operator
overrides. Those values are frozen by the B2A-R3 contract and the parsed
authorization document.

## Verification

CPU tests include a real temporary-Git lifecycle test:

```text
commit authorized historical tree
commit Stage-B authorization document
create Stage-B claim and qualification artifact
verify persisted Stage-B binding while outputs are untracked
commit Stage-B outputs
commit Stage-C authorization document
verify persisted Stage-B binding after HEAD advances
```

Remote CI remains required on the exact final repair SHA before any
runtime authorization can be considered.

```text
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED;
REMOTE CI REQUIRED ON FINAL SHA
```
