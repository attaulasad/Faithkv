# B2A-R3 GPU-Host-Neutral Stage-B Preflight Test Repair (dated 2026-07-24)

## 1. Status

```text
B2A-R3 GPU-HOST-NEUTRAL PREFLIGHT TEST REPAIR AUTHORIZED — CPU TESTS ONLY

OLD CLAIM:
UNCONSUMED; SUPERSEDED WHEN THE REPAIR BRANCH ADVANCES

AUTHORIZED:
DETERMINISTIC REPAIR OF THREE HOST-DEPENDENT NO-CUDA TESTS

PROHIBITED:
PRODUCTION SOURCE CHANGES
SCIENTIFIC CONFIGURATION CHANGES
MODEL INFERENCE
FULLKV/R-KV EXECUTION
CLAIM CONSUMPTION
NEW STAGE-B AUTHORIZATION
```

## 2. Background

A Stage-B FullKV qualification preflight attempt was made on a rented
Vast.ai RTX 3090 host against the authorization document
`docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-07-23.md`
(`authorization_id=stage-b-2026-07-23-final`, execution commit
`4d559070df95def18fe5b649e2a7523d32bdba95`). Every precondition passed
except the mandatory CPU test suite, which reported 3 failures. Stage B
was correctly **blocked before claim consumption** as a result:

- No FullKV or R-KV inference started.
- No model weights were loaded for execution.
- No qualification artifact was produced
  (`results/decisions/b2a_r3_qualification.json` does not exist).
- The external claim at `/tmp/faithkv-stage-b-claim.json`
  (`authorization_id=stage-b-2026-07-23-final`,
  canonical_sha256=`992d7ebf68efcce14aca4bec49a932f8ba7d23517c9c7f0a1e5d11f5e46f5ec1`)
  was never consumed and remains byte-identical to its original copy.

## 3. Root cause

Three CPU tests depended incorrectly on the physical host's real CUDA
availability instead of controlling it deterministically:

```text
tests/unit/discovery/test_b2a_workers_real_bodies.py
    test_run_fullkv_worker_requires_cuda_when_no_fake_backend_injected
    test_run_rkv_worker_requires_cuda_when_no_fake_backend_injected

tests/unit/discovery/test_final_audit_repairs.py
    test_cuda_clean_refusal_is_not_wrapped_when_no_fake_backend_injected
```

Each calls `run_fullkv_worker`/`run_rkv_worker` with the production call
shape (`_cuda`, `_load_model`, `_load_tokenizer` all omitted), so the
worker selects real `torch.cuda` (`cuda = _cuda if _cuda is not None else
torch.cuda`, `src/kvcot/discovery/b2a_workers.py`) and calls the real
`torch.cuda.is_available()`. On the CPU-only machine this suite was
frozen on, that call always returns `False`, so the tests observed the
intended clean `WorkerFailedError("... requires CUDA ...")` refusal. On a
real GPU-visible host it returns `True`, so the workers instead proceed
into real (fake-model-name) snapshot resolution and fail on an unrelated
`SnapshotBoundaryError`.

This is a **test-environment determinism defect, not a production worker
defect** — independently re-confirmed against this exact repository
checkout:

1. All three tests intend to simulate a clean no-CUDA environment.
2. All three omit `_cuda`, `_load_model`, and `_load_tokenizer` to
   preserve the production worker call shape.
3. Both workers choose real `torch.cuda` when `_cuda` is omitted.
4. Both workers call `torch.cuda.is_available()` directly.
5. A clean `False` result must, and still does, produce a plain
   `WorkerFailedError` matching `"requires CUDA"`
   (`b2a_workers.py` lines ~971 and ~1307).
6. An exception raised *by* `is_available()` itself is already covered by
   a separate, distinct partial-evidence test
   (`test_cuda_availability_check_failure_produces_partial_evidence_fullkv`/
   `_rkv`, using an injected `_ExplodingCuda` passed via `_cuda=`) and
   remains wrapped in `WorkerBodyFailure` — unaffected by this repair.
7. The three tests require only CPU-importable `torch`; no real CUDA
   kernel, model weights, or GPU are required to exercise them correctly.
8. No production source repair is necessary — the defect is entirely in
   how the tests failed to control a host-dependent input.

## 4. Repair scope (frozen by this document)

**Authorized:** deterministic repair of exactly the three tests named
above, using `monkeypatch.setattr(torch.cuda, "is_available", lambda:
False)` to force the clean-refusal branch regardless of physical
hardware, while continuing to omit `_cuda`/`_load_model`/
`_load_tokenizer`/`_fresh_cache_factory`/`_device` so the production
default-selection call shape (`cuda = _cuda if _cuda is not None else
torch.cuda`) is exercised unchanged. Narrowly-scoped host-neutrality
assertions (an explicit `AssertionError` if snapshot/model resolution is
unexpectedly reached) may be added where they do not add unnecessary
module coupling.

**Prohibited by this document, even after this repair:**

- Any change under `src/` (production worker behavior is frozen).
- Any change under `configs/`, `third_party/R-KV/`, or `results/`.
- Any change to the model, tokenizer, dataset, cache budget, runtime
  threshold, VRAM limit, R-KV revision, or any of the 27 qualification
  conditions.
- Marking any of the three tests `@pytest.mark.gpu`, skipping them,
  xfailing them, weakening their `WorkerFailedError`/`"requires CUDA"`
  assertions, or catching/accepting `SnapshotBoundaryError` in their
  place.
- Modifying `docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-07-23.md`
  in any way.
- Regenerating, editing, or consuming the existing external claim.
- Creating any new Stage-B (or Stage-C) authorization document or claim.
- Running `kvcot run-b2a-r3-stage-b-qualification`, FullKV, R-KV, Stage C,
  the selected-row freezer, B2B, or any FaithKV method implementation.

## 5. Old claim disposition

The existing external claim
(`authorization_id=stage-b-2026-07-23-final`,
`authorized_code_commit_sha=d6bf377c3f694feb7cd012f9f3522615740bcddd`,
`observed_execution_commit_sha=4d559070df95def18fe5b649e2a7523d32bdba95`,
`canonical_sha256=992d7ebf68efcce14aca4bec49a932f8ba7d23517c9c7f0a1e5d11f5e46f5ec1`)
remains **unconsumed**. Once this repair advances branch HEAD past
`4d559070df95def18fe5b649e2a7523d32bdba95`, the claim's bound
`observed_execution_commit_sha` no longer equals current clean `HEAD`, so
the authorization verifier will correctly reject any future attempt to
use it — the claim becomes **superseded** and must never be executed. A
byte-identical copy is preserved for the record at
`/workspace/faithkv-superseded-claims/faithkv-stage-b-claim-unconsumed-2026-07-23.json`
(SHA-256 recorded alongside it) without modifying the original.

## 6. Next required action

```text
Exact-SHA GitHub Actions CPU CI on the final repair commit.
Independent re-audit of that exact final repair SHA.
```

Stage B remains blocked until a new, separate, dated Stage-B authorization
is produced against a newly audited code commit — this document does not
create one.
