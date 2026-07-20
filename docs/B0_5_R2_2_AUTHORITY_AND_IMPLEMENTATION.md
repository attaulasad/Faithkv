# B0.5-R2.2 — Authority reconciliation and B1A CPU prerequisite implementation

> **Superseded in part, 2026-07-20:** `docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`
> repairs six defects found in this pass's B1A implementation during
> independent review, and adds a new, dated `CLAUDE.md` §1b/§4b exception
> that narrowly authorizes bounded B1B CPU-harness **architecture** work
> (still no GPU, no inference, no method). Everything below this notice
> remains the accurate historical record of what THIS pass (B0.5-R2.2)
> implemented and authorized — it is not rewritten. Statements below that
> B1B "is not authorized"/"remains blocked" describe the state as of
> 2026-07-19, before the 2026-07-20 §1b/§4b exception narrowly authorized
> the CPU-harness-architecture subset of B1B; B2A, B2B, and Vast.ai remain
> unauthorized by both documents.

Phase B0.5-R2.2 artifact (2026-07-19). Branch
`research/b1a-cpu-prerequisites-r2-2`, cut from
`research/b0-5-r2-dense-cache-repair` at commit
`ac3e7d545d13e4b663fe575430ba13e6e4b9bdc5` ("Finalize B0.5 timing sampling
and discovery controls"). This document does two things in one
consolidated pass: (1) reconciles `CLAUDE.md`'s frozen-project statements
with the already-authorized bounded discovery track (Llama-8B/MATH-500),
which a strict reading of `CLAUDE.md`'s original blanket "no 7B support"
language technically contradicted; and (2) implements every CPU-side B1A
prerequisite `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §11/§14 named as
outstanding, plus four additional, independently-identified defects (a
remaining position-field naming error, a layer-depth/event-time confound,
and undefined entropy/logit-margin operational semantics).

**No inference ran. No GPU was used. No model weights or datasets were
downloaded. No MATH-500 manifest, evaluator result, or inference result
directory was created. `third_party/R-KV` was not modified — its pinned
commit (`45eaa7d69d20b7388321f077020a610d9afb65bd`) is unchanged.**

## 1. Status

**B0.5-R2.2 authority reconciliation complete; B1A CPU prerequisites
implemented and CPU-validated. B1B/B2A/B2B/GPU/Vast.ai remain
unauthorized.**

This does **not** claim that the B0.5 discovery hypothesis passed — no
inference result exists anywhere in this repository. It does not claim
accuracy preservation, causal faithfulness, or a successful method. It
authorizes exactly what is listed in §3 below and nothing else.

## 2. Authority reconciliation (Part I of the task brief)

### 2.1 The contradiction

`CLAUDE.md` §1 (original text): *"Do not implement faithfulness-aware
eviction, KIVI, mistake insertion, 7B support, vLLM, SGLang, multi-GPU, an
LLM judge, or a benchmark suite."* Read literally and in isolation, this
blanket-prohibits any architecture beyond the frozen Qwen-1.5B pipeline.
But `docs/b0_5_decision.json`'s `selected_operating_point` (Candidate A,
chosen in Phase B0.5) is `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` +
MATH-500 — already selected, and every B0.5-R/-R2/-R2.1 revision already
scoped a CPU-side B1A prerequisite list (architecture-aware dispatch,
MATH-500 verifier) that only makes sense for that model. The frozen-project
document and the already-authorized discovery track were not literally
consistent with each other; this section resolves that in `CLAUDE.md`
itself rather than leaving it as an implicit, undocumented exception.

### 2.2 The resolution

`CLAUDE.md` now contains two new, dated, clearly-labeled subsections —
**§1a** (research-question/scope level) and **§4a** (frozen-settings-table
level) — added by this document, that:

- state explicitly that the original Qwen-1.5B/GSM8K pipeline is
  unchanged (§1's research question, claim boundary, and the original §4
  frozen-settings table are not edited, only appended to);
- authorize exactly two things for a bounded discovery track: architecture
  support for `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, and dataset
  support for MATH-500 — both CPU-side infrastructure only;
- state explicitly that this is infrastructure/failure-discovery support,
  never a method implementation, and that no GPU/inference authorization is
  granted by the amendment;
- state explicitly that B1B, B2A, B2B, and any Vast.ai activity each
  require their own separate, future, explicit authorization;
- correct the "7B support" blanket line so it narrowly excepts the
  authorized Llama-8B discovery track without weakening any other
  prohibition (vLLM, SGLang, multi-GPU, an LLM judge, a benchmark suite, or
  any method implementation are all still fully prohibited);
- state explicitly that this narrow exception must never be described as
  general benchmark-suite expansion.

`configs/lock.yaml` is **not** changed — the executable source of truth for
the frozen Qwen-1.5B/GSM8K pipeline is untouched. No discovery-track GPU
configuration exists anywhere in this repository.

## 3. What this document authorizes — and what it does not

**Authorizes:**

- CPU-side architecture-aware R-KV monkeypatch dispatch supporting
  `qwen2`, `llama`, and `qwen3` model types (verified against the pinned
  `third_party/R-KV/HuggingFace/rkv/monkeypatch.py`).
- A CPU-side MATH-500 symbolic-equivalence verifier
  (`kvcot.utils.math_verifier`), isolated per-comparison in a child OS
  process with a frozen 5.0-second hard timeout.
- The corrected, active discovery pairwise-provenance Pydantic schema
  (`kvcot.discovery.schemas.SwapPairRecord`, `schema_version =
  "b0_5_r2_2.v1"`).
- Deterministic sampling utilities (event selection, independently
  permuted depth-stratum assignment, layer/KV-head/candidate/donor
  sampling) with golden-vector tests.
- A per-instance, read-only capture-wrapper prerequisite around
  `R1KV.update_kv`, with independent score recomputation and parity
  checks, tested against synthetic CPU tensors and a fake/minimal
  R1KV-like instance.
- A fixed-shape within-head swap primitive and a strengthened, complete
  branch-output no-op control.
- CPU unit/integration tests for all of the above.

**Does NOT authorize:**

- B1B (the intervention harness itself — Pass 1/Pass 2 orchestration,
  wiring the above prerequisites into an end-to-end pipeline).
- B2A (one-example GPU calibration) or B2B (the bounded discovery pilot).
- Any GPU use, model inference, or Vast.ai activity of any kind.
- Any model-weight or dataset download (including MATH-500 itself — no
  manifest, no evaluator result, no result directory was created).
- Any final compression method, faithfulness-aware eviction policy, or
  accuracy/faithfulness claim.
- Any change to the frozen Qwen-1.5B primary pipeline (`configs/lock.yaml`,
  `CLAUDE.md` §1/§4's original tables).

## 4. Four blocker closures (Part II-IV of the task brief)

### 4.1 Blocker 1 — frozen-project contradiction

**Defect:** `CLAUDE.md`'s blanket "no 7B support" line technically
contradicted the already-selected Llama-8B discovery operating point.
**Correction:** dated `CLAUDE.md` §1a/§4a exception (§2 above).
**Files changed:** `CLAUDE.md`.
**Tests:** `tests/unit/test_b1a_scope_guards.py::test_claude_md_states_no_gpu_or_method_authorization_in_discovery_exception`.

### 4.2 Blocker 2 — position-field naming error

**Defect:** the active-schema field `first_affected_logit_absolute_position`
(as specified in `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §3.4/§10, never
actually coded in `src/` before this pass) stored `t+1` — an **input**
position — under a name that reads as a **logit target** position, which
is actually `t+2`. No prior committed record ever used this field (it was
documentation-only until this pass), so there was no migration to perform.

**Correction:** the active schema (`kvcot.discovery.schemas.SwapPairRecord`,
freshly implemented in this pass) uses five explicit fields instead:

```text
event_token_absolute_position                    = t
bridge_token_absolute_position                    = t + 1
first_affected_forward_input_absolute_position    = t + 1
first_affected_logit_target_absolute_position     = t + 2
first_scored_absolute_position                    = t + 2
```

with a `model_validator` enforcing all four required invariants
(`bridge == event + 1`; `bridge == first_affected_forward_input`;
`first_affected_logit_target == first_scored`;
`first_affected_logit_target == first_affected_forward_input + 1`) at
construction time — a malformed record is rejected, never repaired.
`schema_version` is renamed to `"b0_5_r2_2.v1"`.

**Files changed:** `src/kvcot/discovery/schemas.py` (new),
`docs/B0_5_R2_1_FINAL_PROTOCOL.md` (superseded banner added, §3.4/§10 not
deleted), `docs/b0_5_decision.json` (`superseded_by_r2_2` block added).

**Tests:** `tests/unit/discovery/test_schemas.py` (23 tests, including the
4 timing-invariant-violation parametrized cases and the frozen-version
check); `tests/unit/test_b1a_scope_guards.py::test_no_active_use_of_ambiguous_position_field_name`,
`::test_active_schema_uses_corrected_split_timing_fields`.

### 4.3 Blocker 3 — layer-depth/event-time confound

**Defect:** `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §5 assigned each selected
event's depth stratum directly from its **chronological draw order**
(`ordinal = {event_id: k for k, event_id in enumerate(selected_sorted)}`,
then `k` fed straight into the layer-range formula) — silently confounding
*when* a compaction event happened with *how deep* its sampled layer was,
since the two were never independently randomized.

**Correction:** `kvcot.discovery.sampling.assign_depth_strata`
independently permutes `[0, 1, 2]` via its own SHA-256-seeded
`random.Random` stream (suffix `"b05r22_depth_permutation"`, entirely
separate from event selection's `"b05r21_event"` stream) and assigns depth
strata to events by chronological ordinal only as a *lookup index into that
permutation* — never as the stratum value itself. `chronological_event_ordinal`
and `depth_stratum` are stored as two separate schema fields, never merged
into one ambiguous `event_ordinal`.

**Files changed:** `src/kvcot/discovery/sampling.py` (new),
`src/kvcot/discovery/schemas.py` (new — `chronological_event_ordinal` +
`depth_stratum` fields).

**Tests:** `tests/unit/discovery/test_sampling.py` (20 tests, including
exactly-three-without-replacement, every-example-gets-all-three-strata,
golden-vector reproduction, seed-dependent chronology-to-depth remapping,
set/list-order independence, and sub-three-eligible-events ineligibility);
`tests/unit/test_b1a_scope_guards.py::test_no_direct_chronology_to_depth_mapping_in_sampling_module`.

### 4.4 Blocker 4 — undefined entropy/logit-margin signals

**Defect:** `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §8.2 classified entropy and
logit margin as **mandatory** deployable signals but never defined,
computationally, what "entropy at capture" or "logit margin at capture"
actually meant — which logits, what timing, what units, float32 vs.
default precision, natural log vs. log2, or how a per-candidate value
(rather than one shared per-event value) is obtained.

**Correction:** `kvcot.discovery.uncertainty` freezes the exact
computation — `entropy = -(p * log_p).sum()` in float32 natural-log nats
over `log_softmax(raw_logits)`; `logit_margin = top2[0] - top2[1]` over raw
float32 logits, before temperature/top-p/sampling; both computed from the
**raw next-token-prediction logits at the moment the token was originally
predicted during the natural run** (`UNCERTAINTY_SIGNAL_SOURCE =
"raw_next_token_logits_at_token_prediction_time"`), never from the
compaction-event logits and never after the synthetic swap. Position 0 is
explicitly marked unavailable, never invented. Every candidate `e` and
donor `r` gets its own independently-looked-up value (never one shared
value per eviction event) — enforced structurally by
`resolve_prediction_logit_source(absolute_position, prompt_length)`, which
resolves a *specific* forward-call output index per token position. Both
source values and the difference are always stored together
(`entropy_e`/`entropy_r`/`entropy_diff`,
`logit_margin_e`/`logit_margin_r`/`logit_margin_diff`) — never a diff-only,
unauditable field.

**Files changed:** `src/kvcot/discovery/uncertainty.py` (new),
`src/kvcot/discovery/schemas.py` (new — six uncertainty fields plus
`uncertainty_signal_source` provenance field, with a validator requiring
`entropy_diff == entropy_e - entropy_r`/`logit_margin_diff ==
logit_margin_e - logit_margin_r` whenever both source values exist).

**Tests:** `tests/unit/discovery/test_uncertainty.py` (17 tests, including
hand-computed uniform/peaked-distribution entropy, hand-computed
logit-margin, vocabulary-size-independence, position-zero unavailability,
and prefill/decode call-source resolution); `tests/unit/discovery/test_schemas.py::test_entropy_diff_must_equal_e_minus_r_when_both_present`,
`::test_logit_margin_diff_must_equal_e_minus_r_when_both_present`;
`tests/unit/test_b1a_scope_guards.py::test_entropy_and_logit_margin_are_pair_specific_not_event_shared`.

## 5. B1A deliverables (Parts V-IX of the task brief)

| Deliverable | Module | Key tests |
|---|---|---|
| Architecture dispatch | `kvcot.discovery.dispatch` (+ `kvcot.generation.policies._PatchedPolicyBase.load` rewiring) | `tests/unit/discovery/test_dispatch.py`, `tests/unit/test_policies_architecture_dispatch.py` |
| Generalized mutable-state reset | `kvcot.generation.state.reset_patched_state` (docstring/comments generalized; logic was already architecture-generic by construction — verified, not assumed) | `tests/unit/test_generation_state_reset.py` |
| No-offload assertion | `kvcot.discovery.no_offload` (wired into `_PatchedPolicyBase.load`, inert on this CPU-only build) | `tests/unit/discovery/test_no_offload.py` |
| MATH-500 verifier | `kvcot.utils.math_verifier` + `kvcot.utils._math_verify_worker` | `tests/unit/test_math_verifier.py` (28 tests) |
| Active discovery schema | `kvcot.discovery.schemas` | `tests/unit/discovery/test_schemas.py` |
| Deterministic sampling | `kvcot.discovery.sampling` | `tests/unit/discovery/test_sampling.py` |
| Read-only capture wrapper | `kvcot.discovery.capture` | `tests/unit/discovery/test_capture.py` |
| Fixed-shape swap primitive | `kvcot.discovery.swap` | `tests/unit/discovery/test_swap.py` |
| Complete no-op branch evaluator | `kvcot.discovery.branch_eval` | `tests/unit/discovery/test_branch_eval.py` |

### 5.1 On the MATH verifier's timeout mechanism

`math_verify==0.9.0`'s own `parsing_timeout`/`timeout_seconds` mechanism
uses `multiprocessing`, which was found — empirically, on this Windows
development host, not hypothetically — to raise
`OSError: [WinError 6] The handle is invalid` when invoked from this
environment. `kvcot.utils.math_verifier` therefore disables that
mechanism (`parsing_timeout=None`, `timeout_seconds=None` inside the
worker) and enforces the frozen 5.0-second timeout itself, at the parent
level, via `subprocess.run(..., timeout=5.0)` against a freshly launched
`python -m kvcot.utils._math_verify_worker` child process — which
genuinely kills the process (not a thread) on expiry. This is the
"otherwise isolate verification in a child process and terminate it on
timeout" fallback the task brief anticipates.

### 5.2 On the `-oo` test-matrix deviation

The task brief's example test matrix included `(-\infty, 3) == (-oo, 3)`
as an "accepted" case. Direct inspection of the installed
`math-verify==0.9.0` found this false: its expression grammar parses bare
`oo` as an implicit product of two symbols named `o` (`-o*o`), not sympy's
infinity — `\infty` is required. `tests/unit/test_math_verifier.py::test_negative_infinity_bare_oo_is_not_treated_as_infinity_by_this_library`
records this finding explicitly (asserting the *actual*, verified
behavior) rather than asserting a false positive; a substitute
interval-equivalence case using the real `\infty` spelling
(`test_interval_open_notation_equivalence_with_real_infinity_spelling`)
covers the same underlying capability (interval-notation symbolic
equivalence).

## 6. Test evidence

```text
python -m compileall -q src tests   -> exit 0
pytest -q                            -> see final report for exact counts
git diff --check                     -> see final report
```

## 7. Prohibited-work confirmation

No Vast.ai access. No CUDA execution. No GPU-marked test executed (GPU
tests remain collected and auto-skipped by the pre-existing
`tests/conftest.py` mechanism, unchanged). No Llama-8B or Qwen model
weights downloaded. No MATH-500 dataset downloaded. No MATH-500 manifest
or result directory created. No model generation run. The pinned R-KV
submodule commit is unchanged. Torch/Transformers/FlashAttention pins in
`requirements.txt`/`requirements-lock.txt` are unchanged (only
`math-verify[antlr4_13_2]==0.9.0` and its two pure-Python dependencies,
`latex2sympy2_extended==1.11.0` and `antlr4-python3-runtime==4.13.2`, were
added). No final compression method was implemented. No B1B/B2A/B2B code
was implemented. Nothing was merged into `main`.

## 8. Remaining blockers

B1B (the intervention harness itself), B2A (GPU calibration), B2B (the
bounded discovery pilot), and any Vast.ai activity all remain blocked
pending independent review of this branch. This document does not claim
scientific discovery, accuracy preservation, causal faithfulness, or a
successful method — no inference result exists anywhere in this
repository.
