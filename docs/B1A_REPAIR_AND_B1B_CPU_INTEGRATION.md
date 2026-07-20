# B1A repair and B1B CPU harness integration

Phase B1B-R1 artifact (2026-07-20). Branch
`research/b1b-cpu-harness-and-b1a-repairs`, cut from
`research/b1a-cpu-prerequisites-r2-2` at commit
`887cd0fe89486e44db973fdd1f1133d75244fb24` ("Merge pull request #16 from
attaulasad/research/b1a-cpu-prerequisites-r2-2"). This document does two
things in one consolidated pass: (1) repairs six defects found during
independent review of the B1A CPU prerequisites merged in PR #16, and (2)
integrates the B1B CPU harness **architecture** — Pass 1/Pass 2
orchestration, branch construction/evaluation, attrition accounting, a
dry-run planner, and a future B2A one-example contract — using
dependency-injected synthetic/deterministic components, exercised only in
CPU tests, never against a real model.

**B1A repairs and B1B CPU harness integration are implemented for review.
GPU, B2A, and Vast.ai remain blocked. No discovery result exists. No
method exists.**

PR #16 was merged before independent review completed. That merge is not
undone or rewritten here — this branch contains forward-only corrective
commits, per instruction.

## 1. Authorization

B1B (Pass 1/Pass 2 orchestration and everything built on it) was, before
this pass, explicitly **not authorized** by any document in this
repository — `CLAUDE.md` §1a stated a separate, explicit, future
authorization was required, and every `docs/B0_5_R2*` protocol document
repeated that B1B remains blocked. This pass adds that authorization as a
new, dated, narrowly-scoped `CLAUDE.md` §1b/§4b exception, mirroring
exactly the pattern §1a/§4a already used to authorize B1A:

- Authorizes **CPU-side B1B harness architecture only** — Pass 1/Pass 2
  orchestration, branch construction/evaluation, attrition accounting,
  `kvcot plan-discovery --dry-run`, and a documentation/validation-only
  future B2A contract — built and exercised exclusively against
  dependency-injected synthetic/deterministic components in CPU tests.
- Grants **no model inference or GPU use of any kind.** No path exercised
  by this exception's tests ever loads a real model, a real tokenizer, or
  a real dataset.
- Does **not** authorize B2A (GPU calibration) or B2B (the bounded
  discovery pilot) execution, or any Vast.ai activity — each still
  requires its own separate, future, dated authorization.
- Implements **no method** — no learned eviction policy, no
  faithfulness-aware compression policy.

See `CLAUDE.md` §1b/§4b for the exact text.

## 2. Six B1A blocker repairs

### Blocker 1 — unconditional no-offload assertion

**Defect:** `FullKVPolicy.load` never called
`assert_no_offloaded_parameters`; `_PatchedPolicyBase.load` called it only
under `if model.device.type == "cuda":` — a conditional that lets a
partially-offloaded (`device_map="auto"` placing some parameters on
`cpu`/`disk`/`meta`) model skip the check entirely, since `model.device` is
a single reported property, not a real per-parameter walk.

**Correction:** `assert_no_offloaded_parameters(model)` is now called
unconditionally, with no `model.device` guard, in both `FullKVPolicy.load`
and `_PatchedPolicyBase.load`. The assertion itself
(`kvcot.discovery.no_offload`) now also inspects `model.hf_device_map` when
present and rejects any entry assigned to `cpu`/`disk`/`meta`, and rejects
a model with zero named parameters (no vacuous pass).

**Files:** `src/kvcot/discovery/no_offload.py`,
`src/kvcot/generation/policies.py`.

**Tests:** `tests/unit/discovery/test_no_offload.py` (12 tests: all-cuda
pass, single/multiple offloaded parameters, empty-model failure, misleading
`model.device` never bypasses the real check, device-map `disk`/`cpu`/
`meta` rejection), `tests/unit/discovery/test_no_offload_policy_integration.py`
(11 policy-level mock tests: FullKV/RKV both call the assertion, a mixed
cuda/cpu parameter set fails despite `model.device` reporting `cpu`, a
later-parameter offload is caught, `hf_device_map` disk entries fail both
policies, no real network request or model load occurs anywhere in the
file), plus `tests/unit/test_policies_architecture_dispatch.py` updated so
every fake model used there exposes a real (all-cuda)
`named_parameters()` iterator.

### Blocker 2 — absolute survivor parity at every compaction event

**Defect:** `capture.py` only checked observed-vs-recomputed survivor
identity, via SET equality of pre-storage physical indices, at the FIRST
compaction event of a run (where pre-storage index trivially equals
absolute position) — every later event returned `None` (non-evaluable).
The active experiment excludes both the first and last probed events, so
this checked nothing that matters.

**Correction:** `capture_update_kv` now accepts a
`pre_event_position_map_fn` thunk, invoked fresh immediately before every
real `update_kv` call, returning the absolute source-token position
occupying each physical cache slot at that exact moment (sourced from
`kvcot.generation.provenance.LayerProvenance.positions`, never a shadow
FullKV reconstruction). At EVERY compaction event, the wrapper recomputes
the kept physical indices (top-k over the non-recent pool, then the
protected recent window in the real returned-cache order), gathers their
absolute positions from the supplied map, and compares the resulting
ORDERED tensor against `kv_cluster.kept_token_indices[-1]` via exact shape
equality and `torch.equal` — never set equality. Whenever survivor
bookkeeping is available on `kv_cluster`, parity is now ALWAYS a concrete
`True`/`False` for a compaction call — never `None` — and a missing/
malformed position map is itself a hard parity failure.

**Files:** `src/kvcot/discovery/capture.py`.

**Tests:** `tests/unit/discovery/test_capture.py` (19 tests, including a
real two-compaction-event scenario built on the genuine
`kvcot.generation.provenance.LayerProvenance` adapter — its second event's
pre-event map is provably non-identity — ordered-vs-set-equality tests,
missing-map and shape-mismatch hard-failure tests, and clone-not-alias
tests for both the position map and the observed positions).

### Blocker 3 — schema scientific closure

**Defect:** the active schema (`kvcot.discovery.schemas.SwapPairRecord`)
accepted records whose fields contradicted one another: an entropy/
logit-margin source value could be silently missing without recording
why, `parity_check_passed` and `valid_flag` could disagree, and
`score_margin_e_minus_r`/`swap_gain` were never checked against the values
they claim to be derived from.

**Correction:** four new `*_missing_reason` fields (one per uncertainty
source value), enforced exactly-one-of (value, reason); `entropy_diff`/
`logit_margin_diff` are `None` unless both sources exist; a
`parity_check_passed`/`parity_failure_reason` biconditional, plus
`valid_flag=True` is now incompatible with a failed parity check, and a
failed parity check requires `invalid_reason` to name it;
`score_margin_e_minus_r == score_e - score_r` and
`swap_gain == mean(baseline_per_token_nll) - mean(swapped_per_token_nll)`
are now validated through ONE canonical tolerance helper (`_close`,
`1e-9`) and ONE canonical mean-NLL function
(`kvcot.discovery.nll.mean_nll`, used by both the schema validator and the
`kvcot.discovery.branch_eval` producer); every mandatory numeric field on a
valid record must be finite; the no-op control now additionally requires
element-by-element-exact (never `allclose`) NLL-array equality,
`swap_gain == 0.0`, `net_physical_bytes_changed == 0`,
`parity_check_passed == True`, and `valid_flag == True`.

**Files:** `src/kvcot/discovery/schemas.py` (new),
`src/kvcot/discovery/nll.py` (new), `src/kvcot/discovery/branch_eval.py`
(now uses the canonical `mean_nll`).

**Tests:** `tests/unit/discovery/test_schemas.py` (50 tests).

### Blocker 4 — swap.py dtype/device/storage-overlap hardening

**Defect:** `apply_within_head_swap` checked shapes but a plain tensor
assignment can silently cast dtype/upcast/downcast, and the aliasing guard
compared only `tensor.data_ptr()` on the tensors themselves — missing an
offset VIEW into the same underlying storage (e.g. a candidate that is
itself a slice of the cache being written into).

**Correction:** explicit dtype/device equality checks for key/value cache
tensors and both candidates against their target cache tensors, an
explicit batch-size-1 check, a required-contiguous check on both
candidates, and a conservative storage-overlap guard
(`tensor.untyped_storage().data_ptr()` identity, checked pairwise across
all four tensors in play) that catches offset views, not just matching
starting addresses.

**Files:** `src/kvcot/discovery/swap.py`.

**Tests:** `tests/unit/discovery/test_swap.py` (26 tests, including
float64-into-float32, a genuine offset view sharing storage with the
target cache, a candidate that is itself a view into the cache, a
non-contiguous candidate, and a per-slice bit-exactness check over the
entire untouched cache).

### Blocker 5 — uncertainty tensor-shape contracts

**Defect:** `compute_entropy_nats`/`compute_logit_margin` never validated
`raw_logits.ndim`, so a malformed-rank input (a batch dimension, an extra
singleton dimension, a bare scalar) would silently be summed/broadcast over
by the underlying `torch` ops rather than being rejected as a caller bug.

**Correction:** both functions now require `raw_logits.ndim == 1` and raise
`ValueError` immediately otherwise — never flattening or reducing on the
caller's behalf.

**Files:** `src/kvcot/discovery/uncertainty.py`.

**Tests:** `tests/unit/discovery/test_uncertainty.py` (30 tests, including
rejection of shape `(2,2)`, `(1,2,3)`, and a 0-D scalar tensor for both
functions, one-element-vocabulary entropy/margin behavior, non-finite
results, and full source/missing-reason propagation into pair signals).

### Blocker 6 — frozen Llama-8B discovery-track config

**Defect:** no discovery-track config file existed; the Llama-8B revision
had never been resolved and frozen in an executable, validated config.

**Correction:** `configs/discovery/llama8b_math500_b1024.yaml`, validated
by a new Pydantic schema (`kvcot.discovery.discovery_config`). The
revision was resolved via the HF metadata API (no weights downloaded):

```python
from huggingface_hub import HfApi
info = HfApi().model_info("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
print(info.sha)
```

Resolved SHA: `6a6f4aa4197940add57724a7707d069478df56b1` — matching the
value this pass was required to freeze exactly (verified 2026-07-20). The
dataset revision is deliberately left `null` (`dataset.revision`): no
dataset repository ID or revision has been independently verified from an
authoritative source in this pass, and this schema makes that gap
machine-checkable (`DiscoveryDatasetLock.revision_is_frozen`) rather than
silently guessed. `configs/lock.yaml` (the frozen Qwen-1.5B primary
pipeline) is unchanged.

**Files:** `src/kvcot/discovery/discovery_config.py` (new),
`configs/discovery/llama8b_math500_b1024.yaml` (new).

**Tests:** `tests/unit/discovery/test_discovery_config.py` (18 tests,
including rejection of `"main"`, `"latest"`, `null`, and short hashes for
every revision field, and rejection of an `rkv.upstream_revision` that
does not match the pinned R-KV submodule commit used everywhere else in
this repository).

## 3. B1B CPU harness integration

New modules under `src/kvcot/discovery/`, each independently testable:

| Module | Responsibility |
|---|---|
| `harness_types.py` | `NaturalStepFn`/`NaturalStepResult`/`LayerStepObservation` — the one dependency-injection seam a future real-model integration would plug into. |
| `constants.py` | Shared, torch-free constants (`SCORED_HORIZON=48`, `MINIMUM_FUTURE_TOKENS_AFTER_EVENT=49`) — read by both the torch-importing runtime modules and the CLI's torch-free dry-run path. |
| `pass1.py` | Natural-run bookkeeping (`run_natural_pass1`), eligibility (`eligible_event_ids`), and deterministic outcome-blind selection (`build_pass1_plan`, built entirely on the already-tested `kvcot.discovery.sampling` draws). |
| `pass2.py` | Token-identical replay and targeted capture (`run_pass2_capture`) — fresh state, exact token-identity check, per-position capture-record lookup (never assumes sink-list-index equals absolute position, since `update_kv` is not called every step), cross-pass survivor-identity check. |
| `pipeline.py` | Branch construction/evaluation and `SwapPairRecord` assembly (`build_swap_pair_record`) — candidate/donor physical-index resolution, the fixed-shape swap, baseline/swapped branch evaluation, uncertainty lookup. |
| `attrition.py` | `AttritionCounters` — a denominator-consistency-checked funnel, used at both the whole-example and the per-pair level. |
| `orchestrator.py` | `run_example` — wires Pass 1, Pass 2, and per-pair construction/evaluation together end to end with attrition accounting. |
| `b2a_contract.py` | Future one-example B2A contract — documentation/validation only, never executed. |

Also added: `kvcot plan-discovery --dry-run` (`src/kvcot/cli.py`).

### Pass 1

Freezes the complete token trace (batch-1, token-by-token, greedy — no
sampling concept of its own) BEFORE any eligibility decision. Eligible
events exclude the first and last compaction event of the run and require
≥49 real future tokens after the event. Selection (exactly 3 events,
independently-permuted depth strata, layer/kv-head/candidate/donor) is
built entirely on the already-tested `kvcot.discovery.sampling` functions,
none of which accept a branch-gain/NLL argument — selection cannot be
influenced by any downstream outcome, by construction of those functions'
signatures. The donor pool is restricted to the topk-selected (non-recent)
survivors — the protected recent window never competes on score and is
excluded from donor sampling (a documented simplification of this
harness's architecture).

### Pass 2

Resets all mutable state (a fresh per-layer `LayerProvenance`, a fresh
synthetic-model state object), replays EXACTLY Pass 1's frozen token
sequence (teacher-forced, checked position-by-position before any capture
instrumentation is attached), instruments ONLY the 3 selected (layer,
kv_head) pairs' `kv_cluster`s via `kvcot.discovery.capture.capture_update_kv`,
and invalidates the WHOLE example on any of: a token mismatch, a missing/
mismatched compaction-event position at a targeted layer, a within-Pass-2
parity failure (`UpdateKvCaptureRecord.parity_check_passed`), or a
cross-pass survivor-identity mismatch against Pass 1's own recorded
positions for the same event/layer/head. Never reconstructs evicted states
from FullKV. Never modifies the pinned R-KV submodule.

### Branch construction and evaluation

For each selected event, `2 evicted × 2 donors = 4` real pair branches plus
one mandatory no-op control (`evicted == donor`, handled by the SAME code
path — every derived quantity comes out exactly zero/identical by
construction, not a special case). Candidate/donor absolute positions are
resolved to physical pre-event slots via the capture record's own
`pre_event_absolute_position_map` and `window_size` (stored on the record
explicitly — `recomputed_topk_indices.shape[-1]` is the number SELECTED,
`budget - window_size`, which is NOT the pool size except in the
degenerate case where the pre-event cache length equals the budget; an
earlier draft of this pipeline conflated the two and silently dropped most
candidate/donor pairs during integration testing, caught and fixed before
this pass was finalized). The fixed-shape swap
(`kvcot.discovery.swap.apply_within_head_swap`) produces the swapped
branch's initial cache; both branches are evaluated via
`kvcot.discovery.branch_eval.evaluate_swap_branches` against a
dependency-injected `BranchStepFn` — one unscored bridge token, exactly 48
teacher-forced scored targets, canonical `swap_gain` via
`kvcot.discovery.nll.mean_nll`. The resulting `SwapPairRecord` is
constructed and validated through the real, unmodified Blocker-3-repaired
schema.

### Attrition

Two independent `AttritionCounters` populations (`docs/.../§ Attrition`):
one per example attempted (a natural-run/Pass-1/Pass-2-level failure
invalidates the whole example, before any pair is built) and one per
(event, candidate, donor) pair attempted (`3 × 5 = 15` per valid example —
a pair-level failure never invalidates sibling pairs). Every stage listed
in the task brief is tracked; `AttritionCounters.assert_consistent`
structurally prevents a denominator from silently shrinking
(`total_entered == passed_all + sum(dropped_at.values())`, enforced, not
just documented).

### Dry-run planner

`kvcot plan-discovery --config configs/discovery/llama8b_math500_b1024.yaml
--dry-run` validates the config/revisions and prints the planned model,
revisions, dataset label, R-KV budget/revision, `bridge_tokens=1`,
`scored_horizon=48`, `minimum_future_tokens_after_event=49`, `3 events, 2
candidates, 2 donors, 4 pair branches per event`, and the 12-example pilot's
total of 144 pair branches as planning information only — creating no
result files. Every import in its code path is CPU-only, pure-Python
(deliberately NOT `kvcot.discovery.branch_eval`/`kvcot.discovery.pass1`,
which import torch at module scope for their runtime use — the shared
`constants.py` module exists specifically so the CLI's dry-run path never
needs to import torch, matching every other `--dry-run` command in this
CLI).

### Future B2A contract

`kvcot.discovery.b2a_contract` is documentation and validation code only —
it defines the 14 measurements a future one-example B2A run must report
and a pure-Python evaluator for the frozen hard-stop conditions (>4.00
projected GPU-hours, >22 GiB peak allocated memory, any parameter not on
CUDA, no meaningful compression, insufficient eligible events; trajectory
mismatch and capture/gather/absolute-position parity failure are already
structural pass/fail outcomes from `kvcot.discovery.pass2.run_pass2_capture`,
reused rather than re-evaluated independently). It never executes anything.
Because no dataset manifest has been downloaded in this pass, actual B2A
execution remains blocked until the one-example manifest identity and
dataset revision are independently frozen — this contract does not and
cannot resolve that gap.

## 4. Test evidence

```text
python -m compileall -q src tests   -> exit 0
pytest -q                            -> see final report for exact counts
git diff --check                     -> see final report
```

## 5. Prohibited-work confirmation

No GPU. No Vast.ai. No model inference. No model weights downloaded. No
dataset downloaded (the HF metadata-only `model_info` call for Blocker 6
does not download weights). No manifest or result directory created. No
method implemented. `third_party/R-KV` pinned commit unchanged. Nothing
merged into `main`.

## 6. Remaining blockers

Actual B2A remains blocked until: (1) this branch passes independent
review; (2) the exact dataset revision and one-example manifest identity
are frozen from an authoritative source; (3) GPU use is separately
authorized. B2B and any Vast.ai activity remain blocked pending all of the
above plus their own separate authorization.
