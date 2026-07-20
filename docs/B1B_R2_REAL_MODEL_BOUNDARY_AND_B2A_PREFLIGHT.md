# B1B-R2: real-model boundary and B2A preflight

Phase B1B-R2 artifact (2026-07-20). Branch
`research/b1b-r2-gpu-boundary-repair`, cut from `main` at commit
`eadee9a183024a51f3030117c2faae928d6ee162` ("Merge pull request #17 from
attaulasad/research/b1b-cpu-harness-and-b1b-repairs" — the B1B-R1 CPU
harness architecture pass). This document repairs eight defects found
during independent review of B1B-R1 and implements the CPU-side
**real-model boundary and B2A preflight path** the review requested:
absolute-position device/dtype parity, target-only bounded capture,
branching from a complete `ModelStateSnapshot`, exact prefill/decode call
boundaries, a fully frozen and hashed discovery configuration, a frozen
MATH-500 one-example manifest, corrected 144-branch B2B accounting, and a
mandatory-field B2A gate plus a one-example-only `b2a-calibrate` CLI
command.

**B1B-R2 implements the real-model boundary and B2A preflight path. No GPU
execution has occurred. B2A remains unaudited and blocked pending
independent review. B2B and method design remain blocked. No scientific
result or new method is claimed.**

## 1. Authorization

This pass does not introduce a new capability beyond what `CLAUDE.md`
§1b/§4b already authorized ("CPU-side harness architecture only... a
documentation/validation-only future B2A contract... does not authorize
B2A... or B2B... execution"). It is an engineering and protocol-correction
pass under that same authorization: every defect fixed and every module
added here (device/dtype normalization, target-only capture, complete-
snapshot branching, prefill/decode adapters, frozen config/manifest, the
B2A gate, and `b2a-calibrate --dry-run`/`--execute`) is CPU-side harness
architecture, dependency-injected synthetic components in CPU tests, or
pure-Python configuration/validation code. No line item here requires a new
`CLAUDE.md` exception, and none is added. In particular:

- No model inference or GPU use is authorized or has occurred.
- `b2a-calibrate --execute`'s code path exists (`kvcot.discovery
  .b2a_execute`, `kvcot.discovery.real_model_adapter`) but is never invoked
  in this pass — every precondition it enforces (CUDA required; the frozen
  manifest's prompt-token identity resolved) currently fails closed on this
  CPU-only, no-download build, and would fail closed on any machine until a
  future, separately-authorized run resolves that gap for real.
- B2B (12-example pilot) execution remains unauthorized and unimplemented;
  no B2B command exists.

## 2. Eight review-defect repairs

### Defect 1 — absolute-position device/dtype parity

**Defect:** `kvcot.discovery.capture._recomputed_kept_physical_indices`
built `recent_window` via a bare `torch.arange(...)` (silently CPU) and
concatenated it with `recomputed_topk_indices` (which, on a real CUDA
compaction, lives on CUDA) before gathering against the provenance map
(itself CPU, `kvcot.generation.provenance.LayerProvenance` is
`dtype=long`, on CPU by construction). A real GPU run would raise a
device-mismatch error the CPU-only synthetic tests could never surface.

**Fix:** `_recomputed_kept_physical_indices` now takes explicit
`device`/`dtype` keyword arguments — the PROVENANCE MAP's own device/dtype,
never assumed to match `recomputed_topk_indices`. Both the top-k indices
and the freshly-built `recent_window` are normalized to that device/dtype
BEFORE concatenation; shape and ordering are unchanged.
`_check_observed_kept_absolute_position_parity` passes
`pre_event_position_map.device`/`.dtype` in.

- **Files:** `src/kvcot/discovery/capture.py`.
- **Tests:** `tests/unit/discovery/test_capture.py` —
  `test_recomputed_kept_physical_indices_normalizes_dtype_and_device`,
  `test_recomputed_kept_physical_indices_preserves_shape_and_ordering_when_already_matching`,
  and a `@pytest.mark.gpu`-marked
  `test_recomputed_kept_physical_indices_cuda_topk_cpu_provenance_matches_cpu_reference`
  (top-k indices on CUDA, provenance map on CPU, result compared against a
  pure-CPU reference; skips cleanly via `tests/conftest.py`'s existing `gpu`
  marker convention when CUDA is unavailable — never executed on this
  build machine).

### Defect 2 — capture was neither target-only nor memory-bounded

**Defect:** `capture_update_kv` cloned and stored complete pre-call
K/Q/V and returned tensors for EVERY real `update_kv` call, and Pass 2
wrapped every target layer for the complete replay — memory bounded by the
total number of decode/compaction calls in a run, not by the 3 preselected
targets.

**Fix:** `capture_update_kv` gained `layer_idx`/`current_position_fn`/
`should_capture` keyword arguments (all optional, default `None` — the
default behavior, "capture everything", is unchanged, so every pre-existing
caller/test keeps working). When `should_capture(position, layer_idx)`
returns `False` (or no position is available), the wrapper calls the
ORIGINAL `update_kv` directly and returns unchanged: no clone, no stored
tensor, no capture record. `kvcot.discovery.pass2.run_pass2_capture` wires
`should_capture` to match exactly the 3 preselected `(absolute_position,
layer_index)` pairs Pass 1 selected — retained capture state is bounded by
the target count, never by total call volume.

- **Files:** `src/kvcot/discovery/capture.py`, `src/kvcot/discovery/pass2.py`.
- **Tests:** `tests/unit/discovery/test_capture.py` —
  `test_should_capture_false_skips_clone_and_capture_entirely` (spies on
  `torch.Tensor.clone` to prove zero clone calls on a non-target call),
  `test_should_capture_selects_exactly_the_target_event_layer_pairs` (15
  calls across 2 layers, only the 2 designated targets captured),
  `test_repeated_non_target_calls_do_not_grow_retained_state`,
  `test_should_capture_none_preserves_capture_everything_default`
  (backward compatibility).

### Defect 3 — branch construction used one layer's K/V as the whole model's state

**Defect:** `kvcot.discovery.pipeline.build_swap_pair_record` built
`baseline_state`/`swapped_state` from `(record.returned_key_states,
record.returned_value_states)` — one (layer, all-heads) tensor pair
standing in for the complete causal-LM continuation state. Not a valid
branch state for any real model.

**Fix:** `kvcot.generation.state.ModelStateSnapshot` (the PRIMARY
pipeline's own existing, already-tested complete-state snapshot — every
layer's K/V, query cache, compression flags, model/cache length, R-KV
`kv_cluster` bookkeeping, compaction history, provenance) gained a
`.clone()` method: an independent deep copy, no tensor aliasing, filling
the one gap the pre-existing `capture_snapshot`/`restore_snapshot` pair
didn't cover (a snapshot-to-snapshot clone with no live model in the loop).
`kvcot.discovery.pass2.run_pass2_capture` now takes an injected
`snapshot_fn` and captures ONE pristine, complete snapshot immediately
after each selected target event; `TargetCapture` carries it. `pipeline
.build_swap_pair_record` now clones the pristine snapshot twice
(`baseline_snapshot`, `swapped_snapshot`), mutates only the swapped copy's
targeted (layer, kv_head, slot) via the pre-existing, already-hardened
`apply_within_head_swap`, and passes both COMPLETE snapshots into
`evaluate_swap_branches`. No new state-snapshot abstraction was created —
the existing one was extended and reused.

- **Files:** `src/kvcot/generation/state.py` (`.clone()`),
  `src/kvcot/discovery/pass2.py`, `src/kvcot/discovery/pipeline.py`,
  `src/kvcot/discovery/branch_eval.py` (docstring only — the type shape was
  already generic).
- **Tests:** `tests/unit/test_model_state_snapshot_clone.py` (round-trip of
  every field, no storage aliasing, mutating one clone never affects
  another, `provenance=None` handled); `tests/unit/discovery
  /test_b1b_integration.py` — `test_multi_event_plan_has_non_identity
  _pre_event_map_on_a_later_event` (asserts every `pristine_snapshot`
  covers all `NUM_LAYERS` layers), `test_branch_evaluation_order_does_not
  _change_results` (baseline-first vs. swap-first give identical
  per-branch outputs), `test_incomplete_one_layer_snapshot_is_rejected`
  (a truncated one-layer snapshot is rejected via `apply_within_head_swap`'s
  existing out-of-range check, surfaced as a branch-evaluation failure).

### Defect 4 — no explicit prefill/decode call boundary

**Defect:** Pass 1/Pass 2 fed EVERY position — prompt included — through
one generic per-token `step_fn`, unable to distinguish "this is the
one-shot prefill call" from "this is decode call #1". R-KV can legitimately
initialize/behave differently on prefill vs. decode; a real integration
plugging into the old seam could not honor the real, frozen batch-1
prefill-then-single-token-decode call shape (CLAUDE.md §4,
`docs/REPLAY_DESIGN.md` §2) at all.

**Fix:** `kvcot.discovery.harness_types` gained `PrefillFn` (`(state,
prompt_token_ids) -> PrefillStepResult`, called EXACTLY ONCE per pass, with
the complete prompt) and kept the one-token shape as `DecodeOneFn` (alias
of the prior `NaturalStepFn`). `run_natural_pass1`/`run_pass2_capture` now
take `prefill_fn`/`decode_one_fn` explicitly: one prefill call produces
per-prompt-position logits/uncertainty and layer observations; every
continuation token uses its own `decode_one_fn` call. Prompt uncertainty is
read directly from the prefill's own `per_position_logits`, never from a
repeated decode call. `eligible_event_ids` now additionally excludes any
compaction event whose `absolute_event_position < prompt_length` — a real
one-shot prefill call is architecturally opaque from the outside (no valid
"immediately after this specific mid-prefill position" snapshot boundary
exists), so a prefill-phase event can never be a valid branch-construction
target; this restriction is what makes Defect 3's per-target snapshot
capture well-defined at all.

- **Files:** `src/kvcot/discovery/harness_types.py`,
  `src/kvcot/discovery/pass1.py`, `src/kvcot/discovery/pass2.py`,
  `src/kvcot/discovery/orchestrator.py`,
  `tests/unit/discovery/_synthetic_harness.py`,
  `tests/unit/discovery/_synthetic_harness_variants.py`.
- **Tests:** `tests/unit/discovery/test_b1b_integration.py` —
  `test_prefill_called_exactly_once_with_complete_prompt`,
  `test_every_continuation_token_uses_exactly_one_decode_one_call`,
  `test_pass1_and_pass2_have_identical_call_boundary_traces`,
  `test_uncertainty_derived_from_prefill_logits_not_repeated_decode_calls`,
  `test_replay_token_mismatch_causes_hard_failure_not_silent_repair`,
  `test_prefill_phase_compaction_events_are_never_eligible`.

### Defect 5 — discovery configuration was not fully frozen or hashed

**Defect:** `DiscoveryConfig` covered only model/dataset-name/rkv-budget;
generation mode, attention backend, cache implementation, batch size,
framework seed, the full R-KV hyperparameter set, and the prompt template
were not represented anywhere as frozen, hashable configuration.

**Fix:** `DiscoveryRkvLock` gained `window_size`/`mix_lambda`/
`retain_ratio`/`retain_direction`/`divide_method`/`divide_length`/
`compression_content`/`kernel_size` (reused verbatim from `configs/lock
.yaml`'s own frozen R-KV row, CLAUDE.md §4 — never independently guessed).
A new `DiscoveryGenerationLock` freezes `generation_mode="greedy"`,
`do_sample=False`, `temperature=None`, `top_p=None`, `batch_size=1`,
`max_new_tokens=6144` (the SAME cap CLAUDE.md §4 freezes for the primary
pipeline), `framework_seed=13` (reused from the existing
`cmd_freeze_manifests` precedent — the same seed already used to freeze
this repository's GSM8K/MATH-500 manifests — never a newly-invented seed),
`attention_backend="flash_attention_2"` (verified supported by the pinned
R-KV submodule's generic `ALL_ATTENTION_FUNCTIONS[self.config
._attn_implementation]` dispatch, `third_party/R-KV/HuggingFace/rkv
/modeling.py`), `cache_implementation="DynamicCache"`, and
`no_offload_required=True`. `generation_config_hash`/`rkv_config_hash`/
`prompt_template_hash`/`canonical_config_hash` (pure `sha256_json`/
`sha256_text` over these frozen fields, reusing `kvcot.utils.hashing`, no
independent hashing scheme) are new pure-Python functions. Nothing is
`"auto"` or a hidden library default anywhere in this schema.

- **Files:** `src/kvcot/discovery/discovery_config.py`,
  `configs/discovery/llama8b_math500_b1024.yaml`.
- **Tests:** `tests/unit/discovery/test_discovery_config.py` (generation
  lock defaults, hash stability/sensitivity for generation/R-KV/canonical
  hashes, prompt template hash).

### Defect 6 — MATH-500 dataset revision and one-example manifest were unresolved

**Defect:** `configs/discovery/llama8b_math500_b1024.yaml`'s
`dataset.revision` was `null` (deliberately, per B1B-R1's own honest
non-fabrication policy) and no one-example manifest existed at all.

**Fix:** The MATH-500 dataset revision was resolved directly against the
Hugging Face Hub API (`GET https://huggingface.co/api/datasets
/HuggingFaceH4/MATH-500`, the `sha` field), verified via two independent
lookups against the live source on 2026-07-20:
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`. This is now frozen in
`configs/discovery/llama8b_math500_b1024.yaml`'s `dataset.revision` (config
+ split also frozen: `default`/`test`). A one-example manifest,
`configs/discovery/b2a_one_example_manifest.json`, freezes: dataset repo/
config/split/revision, `example_index=0`, the dataset row's own
`unique_id` (`test/precalculus/807.json`, fetched via the HF dataset viewer
API), and a `sha256_json` hash of the raw row content
(`0fea4410b42518afa81f3cc6458db017a4091d480b93367ba57794d194c0a991`). Two
fields are honestly `null`: `prompt_token_ids_sha256` and
`tokenizer_revision_used_for_prompt_hash` — resolving them requires
actually running the live tokenizer, which this CPU-only, no-model-download
pass cannot and must not do. `kvcot.discovery.manifest
.B2AOneExampleManifest.prompt_identity_is_resolved` reports this gap
explicitly; `load_b2a_one_example_manifest` refuses to load a manifest
whose `dataset_repo`/`dataset_revision` disagrees with the frozen discovery
config.

- **Files:** `src/kvcot/discovery/manifest.py`,
  `configs/discovery/b2a_one_example_manifest.json`,
  `src/kvcot/discovery/discovery_config.py` (dataset revision/config/split
  fields, `MATH500_DATASET_REPO`/`MATH500_DATASET_REVISION` constants),
  `configs/discovery/llama8b_math500_b1024.yaml`.
- **Tests:** `tests/unit/discovery/test_manifest.py` (load/validate, hash
  stability, unresolved-prompt-identity state, rejects a manifest
  disagreeing with the frozen dataset repo/revision).

### Defect 7 — branch-count accounting language

**Defect:** the review flagged a risk of "5 branches per event" / 180-total
assumptions contaminating planning paths (4 real cross-product swaps + 1
mandatory no-op, if miscounted as "5 real branches", inflates
`12 x 3 x 5 = 180`).

**Finding:** `kvcot.cli.cmd_plan_discovery` already printed the CORRECT
`12 x 3 x 4 = 144` accounting (verified by inspection and by
`tests/unit/test_cli_plan_discovery.py`'s pre-existing assertions) — no
contradictory 180-branch total existed in any executable path. This pass
(a) moved the accounting constants into `kvcot.discovery.constants`
(`EVENTS_SELECTED_PER_EXAMPLE`, `CANDIDATES_PER_EVENT`, `DONORS_PER_EVENT`,
`PAIR_BRANCHES_PER_EVENT`, `B2B_PILOT_EXAMPLE_COUNT`,
`B2B_PILOT_TOTAL_REAL_BRANCHES`) as the single source of truth both
`cmd_plan_discovery` and `cmd_b2a_calibrate` read from (previously
`cmd_plan_discovery` had these as local literals — no drift existed yet,
but nothing prevented one), and (b) made the B2B-vs-no-op separation
explicit in both commands' output: "144 real branches (the mandatory no-op
control is excluded from this total)". `kvcot.discovery.orchestrator
.run_example` still builds `4 cross-product pairs + 1 mandatory no-op = 5`
pair-construction ATTEMPTS per event through the same code path (Part
IX.20's own requirement: the no-op is not a special case), but the no-op is
never counted toward the "144 real branches" total anywhere it is reported.

- **Files:** `src/kvcot/discovery/constants.py`, `src/kvcot/cli.py`.
- **Tests:** `tests/unit/test_cli_plan_discovery.py`,
  `tests/unit/test_cli_b2a_calibrate.py` (both assert the exact "144 real
  branches" / no-op-excluded wording).

### Defect 8 — B2A gate could pass without evidence

**Defect:** `B2AGateResult` was `(passed: bool, failed_conditions: tuple)`
— trivially constructible as `B2AGateResult(passed=True,
failed_conditions=())` with zero evidence behind that claim, and none of
the identity/parity/trajectory conditions the task brief requires
(`token_identical_replay`, `absolute_position_parity`,
`dataset_revision_match`, `manifest_hash_match`, ... 19 total) were
represented as fields anywhere.

**Fix:** `B2AGateEvidence` (pydantic, every field REQUIRED — no
`Optional`, no default) now carries all 19 mandatory conditions from the
task brief plus the 2 pre-existing measurement-based conditions
(`meaningful_compression_observed`, `sufficient_eligible_events`) this
repository already froze, plus raw `runtime_gpu_hours`/`peak_vram_gib`
measurements. `B2AGateResult` mirrors every one of those as an explicit
field (never collapsed into `passed` alone) and its `__post_init__`
RE-DERIVES `passed`/`failed_conditions` from the mandatory fields
themselves — hand-constructing an internally-inconsistent result (e.g.
`passed=True` while any field is `False`) raises immediately.
`evaluate_b2a_gate` is the only intended constructor: `runtime_within_limit`/
`peak_vram_within_limit` are always DERIVED from the raw measurements
against the frozen thresholds (`<= 4.00 GPU-hours`, `<= 22 GiB`), never
accepted as an independent caller-supplied claim. `build_gate_evidence_from
_measurement` derives `no_offload_verified` from the measurement's own
`every_parameter_on_cuda` for the same reason (never a second,
independently-assertable copy that could disagree with what was actually
measured).

- **Files:** `src/kvcot/discovery/b2a_contract.py`.
- **Tests:** `tests/unit/discovery/test_b2a_contract.py` — parametrized
  "every single mandatory condition being False fails the gate" test,
  missing-field rejection, cannot hand-construct a passing result with a
  false field (and vice versa), runtime/VRAM threshold tests.

## 3. One-example B2A executable path (new: `kvcot b2a-calibrate`)

`kvcot plan-discovery` remained planning-only, by design (CLAUDE.md
§1a/§1b). This pass adds `kvcot b2a-calibrate`, the one-example-only
calibration command the task brief requires:

- **`--dry-run`** (the only mode exercised in this pass): requires no CUDA,
  downloads no model weights, runs no inference (enforced structurally —
  `tests/unit/test_cli_b2a_calibrate.py` intercepts `builtins.__import__`
  for `transformers`/`datasets`/`huggingface_hub`/`rkv` and asserts a CUDA
  API is never called). Validates the frozen configuration and one-example
  manifest, prints the exact call plan, the selected example's identity,
  the 144-branch B2B cost model and the B2A-only no-op calibration plan
  SEPARATELY, every configuration/manifest hash, and fails non-zero
  (`exit code 2`) whenever an unresolved/inconsistent field is detected —
  in this build, that is unconditional, since the manifest's prompt-token
  identity remains unresolved (Defect 6).
- **`--execute`** (explicit GPU execution mode; code path implemented, never
  invoked in this pass): requires the `--execute` flag explicitly (never a
  silent default), rejects `--problem-index`/`--limit` overrides (exactly
  one manifest example, never a range or unrestricted iteration), requires
  CUDA, and refuses on any blocker BEFORE ever calling a CUDA API
  (`tests/unit/test_cli_b2a_calibrate.py::test_execute_refuses_on
  _unresolved_manifest_before_touching_cuda` proves the ordering). The real
  run itself (`kvcot.discovery.b2a_execute.run_b2a_calibration`) reuses the
  primary pipeline's own model-loading (`kvcot.generation.policies
  .RKVPolicy`, including its existing `assert_no_offloaded_parameters`
  call and correct patch-before-construct dispatch ordering), the new
  `kvcot.discovery.real_model_adapter` `PrefillFn`/`DecodeOneFn`/
  `SnapshotFn`/`BranchStepFn` builders (themselves built from
  `kvcot.generation.decode`/`kvcot.generation.replay`/`kvcot.generation
  .provenance` primitives, never a second independent implementation), and
  `kvcot.discovery.orchestrator.run_example` — collects every mandatory
  B2A gate field via `kvcot.discovery.b2a_contract.evaluate_b2a_gate`, and
  writes an immutable JSON artifact under `results/decisions/` before
  stopping. It never starts a B2B pilot, and there is no B2B command in
  this repository.

The command's help text states: "B2A is a one-example engineering
calibration. It does not authorize the 12-example pilot."

## 4. Validation performed (CPU-only)

```
python -m compileall .
pytest --collect-only
pytest -m "not gpu"
```

See the accompanying report for exact collected/passed/failed/skipped
counts and the `b2a-calibrate --dry-run` output. No CUDA test was executed;
no Llama-8B weights were downloaded; no Hugging Face inference ran; no
Vast.ai activity occurred; B2A/B2B were not run.

## 5. Remaining blockers (unchanged in kind, restated for this pass)

- No real Llama-8B/R-KV CUDA execution has occurred under this pass.
- RTX 3090 memory and runtime remain unmeasured.
- The manifest's prompt-token identity (`prompt_token_ids_sha256`/
  `tokenizer_revision_used_for_prompt_hash`) remains unresolved — resolving
  it requires running the live tokenizer once, which is out of scope here.
- B2A remains blocked pending independent review of this pass.
- B2B (the 12-example, 144-branch pilot) remains blocked; no B2B command
  exists.
- No FaithKV method exists in this repository.
