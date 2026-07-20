# B1B-R3: executable B2A boundary and evidence producer

Phase B1B-R3 artifact (2026-07-20). Branch
`research/b1b-r3-executable-closure`, cut from `main` at commit
`7034e46b516eff656b5508d9253ee02b13405f95` — the merged PR #18
("Implement B1B-R2 real-model boundary and B2A preflight"). PR #18 was
merged into `main` **before** this pass began; this document does not
reopen or revert it. This pass is a forward repair following an
independent audit of that merged PR, converting the partially-implemented
B2A boundary into a mechanically complete, fail-closed executable
architecture.

**B1B-R3 repairs the executable B2A boundary and evidence producer. No GPU
inference was executed. No RTX 3090 measurement exists. B2A remains
blocked pending independent CPU review. B2B remains blocked. No FaithKV
method exists.**

## 1. Authorization

This pass does not introduce a new capability beyond what `CLAUDE.md`
§1b/§4b already authorized — every module added or changed here is
CPU-side harness architecture, dependency-injected/mocked components in
CPU tests, or pure-Python configuration/verification code, with two
narrow, explicitly-scoped exceptions both still inside the existing
authorization's own terms:

- `kvcot prepare-b2a-manifest --execute` downloads exactly two kinds of
  CPU-side artifact: one pinned MATH-500 dataset row (via a revision-
  pinned raw file fetch, never the `datasets` library or model weights)
  and the pinned tokenizer's `tokenizer.json`/`tokenizer_config.json`
  (via `transformers.AutoTokenizer.from_pretrained`, verified afterward to
  have downloaded no weight-shaped file). This is exactly the "pinned
  tokenizer/config files" and "one pinned MATH-500 row" CPU-only
  downloads this task's own governing instructions explicitly permit.
- No model weights, no CUDA, no Vast.ai activity of any kind occurred at
  any point in this pass.

B2A (one-example GPU calibration) and B2B (the bounded discovery pilot)
remain unauthorized and unexecuted. `b2a-calibrate --execute`'s complete
subprocess architecture exists as real, reviewable code
(`kvcot.discovery.b2a_execute`, `kvcot.discovery.b2a_workers`,
`kvcot.discovery.b2a_worker_entry`) but every path through it requires CUDA
and is never invoked by any test or command in this pass.

## 2. Audit defects repaired

### Defect 1 — invalid `RKVPolicy` construction

**Files:** `src/kvcot/discovery/b2a_execute.py`,
`src/kvcot/generation/policies.py`,
`src/kvcot/discovery/runtime_rkv_verification.py` (new).

`kvcot.discovery.b2a_execute.run_b2a_calibration` (as merged in PR #18)
built `RKVMethodConfig(...)` and then passed that object as `RKVPolicy`'s
positional `budget: int` argument — mechanically invalid, since
`RKVPolicy.__init__` takes `budget` plus individual keyword fields, never
a config object. `run_rkv_worker_body` (B1B-R3's replacement for the old
`run_b2a_calibration`'s model-loading body) now constructs it exactly as
the task brief requires:

```python
policy = RKVPolicy(
    budget=config.rkv.budget, window_size=config.rkv.window_size,
    mix_lambda=config.rkv.mix_lambda, retain_ratio=config.rkv.retain_ratio,
    retain_direction=config.rkv.retain_direction, divide_method=config.rkv.divide_method,
    divide_length=config.rkv.divide_length, compression_content=config.rkv.compression_content,
    kernel_size=config.rkv.kernel_size,
)
```

There is exactly one construction path — no competing second one exists
anywhere in this repository.

**Runtime configuration verification (also Defect 12).** Auditing
`RKVMethodConfig`/`RKVPolicy` against the pinned R-KV submodule's own
source (`third_party/R-KV/HuggingFace/rkv/compression/r1_kv.py`,
`.../rkv/modeling.py`) found `kernel_size` was NOT a configurable field at
all before this pass — R1KV silently used its own upstream default (`7`)
regardless of `DiscoveryRkvLock.kernel_size` (frozen at `3`). Fixed by
adding `kernel_size` to `RKVMethodConfig`/`RKVPolicy`/
`_compression_config()` (default `7`, preserving the primary Qwen
pipeline's existing behavior exactly — `configs/lock.yaml` has no
`kernel_size` row and is unchanged by this pass) and passing
`config.rkv.kernel_size` explicitly from the discovery path.
`kvcot.discovery.runtime_rkv_verification.verify_runtime_matches_frozen`
reads every frozen R-KV field back off the LIVE model's per-layer
`kv_cluster`/`self_attn.config`/`model.config` objects (citing the exact
submodule source line for each attribute's location), hashes both the
frozen and runtime field sets with the identical field ordering, and
`run_rkv_worker_body` refuses (`B2AExecutionRefused`) before any further
work if `runtime_rkv_config_hash != frozen_rkv_config_hash`.

**Tests:** `tests/unit/discovery/test_runtime_rkv_verification.py` (11
tests: matching config passes, kernel_size/budget mismatch detected and
hashes differ, missing `kv_cluster` / empty layers / cross-layer
disagreement / missing `update_kv` or `model.config` field all raise
`RuntimeRkvConfigError`, frozen-hash stability/sensitivity, runtime-hash
determinism).

### Defect 2 — missing prompt (empty `prompt_token_ids`)

**Files:** `src/kvcot/discovery/manifest.py` (schema extended),
`src/kvcot/discovery/manifest_prepare.py` (new),
`src/kvcot/cli.py` (`prepare-b2a-manifest` command, new),
`src/kvcot/discovery/b2a_execute.py`
(`_verify_resolved_prompt_identity`).

`kvcot prepare-b2a-manifest --execute` is the new CPU-only command: fetches
the pinned MATH-500 row from the exact revision-pinned raw file (never the
mutable `datasets-server` convenience API — see the discrepancy documented
below), verifies its schema/unique_id, renders the frozen prompt template
(`kvcot.probes.templates.render_base_user_message`, reused, never a second
template), loads the pinned tokenizer, requires a real `chat_template`
(refuses to invent one), applies the EXACT frozen chat-template call
`kvcot.cli.cmd_generate` already uses elsewhere in this repository
(`tokenizer.apply_chat_template([{"role": "user", "content": user_message}],
tokenize=True, add_generation_prompt=True)`), and freezes: raw-row hash,
rendered-message hash, chat-template-source hash, serialized-message-payload
hash, prompt-token-ID hash, prompt token count, the exact tokenizer
revision used, the rendering configuration, and the complete prompt
token-ID array. Writes to a temp file, validates, atomically renames;
refuses to overwrite an already-populated manifest without `--force`; with
`--force`, prints old/new manifest hashes and requires every frozen
upstream identity field (repo/config/split/revision/example_index/
unique_id) to stay unchanged.

**This command was actually run in this pass** (network was available) —
`configs/discovery/b2a_one_example_manifest.json` is now genuinely
resolved; see §4 for the real values.

`kvcot.discovery.b2a_execute._verify_resolved_prompt_identity` re-fetches
the row, re-renders, re-tokenizes, and re-hashes everything, comparing
against the manifest's frozen values, and refuses (`B2AExecutionRefused`)
on ANY mismatch — BEFORE either worker subprocess is ever launched. The
literal `prompt_token_ids: list[int] = []` line from the merged PR is
gone; `run_rkv_worker_body` asserts `len(prompt_token_ids) > 0` immediately
before calling Pass 1 — an empty prompt is now structurally impossible on
this path.

**Audit finding: the PR #18 manifest's `raw_content_hash` was not
reproducible.** Before writing the new command, the already-committed
`raw_content_hash` (`0fea4410b4...`) was compared against a fresh
`sha256_json` of the SAME row (repo/revision/index/unique_id all agreeing)
fetched three independent ways (the `datasets-server` `/rows` API, the
`/first-rows` API, and the revision-pinned raw `test.jsonl` line) — none
reproduced it, and a dozen reasonable canonicalization variants were tried
without success. The row content, `unique_id`, and dataset revision all
agreed across every fetch; only the previously-committed hash itself could
not be reproduced by any method tried. `prepare-b2a-manifest --execute
--force` was run and corrected it to the reproducible value
(`0d4aaa49ce633bbae51053c17c1b9703daca6c02271f7e52dfa36c73d05c9a96`,
computed via `sha256_json` over the pinned-revision `test.jsonl` line 0,
parsed as JSON, keys in dataset-column order) — printed explicitly by the
`--force` path (`raw_content_hash CORRECTED ...`). This is reported here
rather than silently absorbed, per this task's explicit anti-fabrication
requirements.

**Tests:** `tests/unit/discovery/test_manifest_prepare.py` (9,
network-free via monkeypatched fetch/render functions: schema rejection,
missing-chat-template rejection, atomic write, refuse-without-force,
force-refuses-changed-unique_id, force-allows-corrected-raw-hash,
no-weight-file check), `tests/unit/discovery/test_manifest.py` (updated:
manifest is now genuinely resolved; new all-or-nothing prompt-identity
validator tests), `tests/unit/test_cli_prepare_b2a_manifest.py` (4).

### Defect 3 — missing answer verifier

**Files:** `src/kvcot/discovery/math500_verification.py` (new),
`src/kvcot/discovery/manifest.py` (`gold_answer` field, new).

The merged PR's `_answer_fn` labeled every natural generation
`"unverifiable"` unconditionally, while `kvcot.discovery.orchestrator
.run_example` rejects every status other than `"correct"` — the real path
could never reach Pass 2. `Math500AnswerVerifier` (a stateful `AnswerFn`
implementation) now decodes the generated tokens, extracts the model's
final answer via the existing `kvcot.utils.answers.extract_answer`
(boxed → final-answer-marker → conservative-number-fallback priority,
unchanged, never a second extractor), compares it against the frozen gold
answer (now a required field on `B2AOneExampleManifest`, resolved by
`prepare-b2a-manifest` from the dataset row's own `answer` column) via the
existing subprocess-isolated `kvcot.utils.math_verifier
.verify_math_equivalence` (never a second, simplistic string-equality
verifier), and returns exactly one of `"correct"`/`"incorrect"`/
`"unverifiable"`. The full extraction/verification detail (raw decoded
text, extraction method, verification status/failure reason) is preserved
on `self.last_result`, read by `run_rkv_worker_body` into the worker
result payload.

**Tests:** covered indirectly through the existing, extensive
`kvcot.utils.answers`/`kvcot.utils.math_verifier` test suites (unchanged,
reused as-is — no second verifier means no second test suite was needed);
`Math500AnswerVerifier`'s own wiring is exercised by
`tests/unit/discovery/test_b2a_execute_coordinator.py`'s mocked end-to-end
flow (via the `natural_answer_status` field flowing through to evidence).

### Defect 4 — split provenance state

**Files:** `src/kvcot/discovery/real_model_adapter.py` (rewritten).

Confirmed by direct code reading (not just the audit's suspicion): the
merged PR's `build_real_prefill_fn`/`build_real_decode_one_fn` called
`kvcot.generation.replay._sync_layer_after_call` directly but NEVER called
`LayerProvenance.append_new_tokens_prefill`/`append_new_token` on
`RealModelState.model_provenance` first — unlike
`kvcot.generation.replay.replay_and_snapshot`'s own prefill/decode blocks,
which always append before syncing. Since `_sync_layer_after_call` only
writes `model_provenance.layers[i].positions` wholesale ON an eviction
event (`adopt_upstream_kept_indices`) and otherwise leaves it untouched,
`RealModelState.model_provenance.layers[i].positions` would have stayed
permanently EMPTY between prefill and the first real eviction — silently
corrupting `pre_event_absolute_position_map` at exactly the moment Pass 1
candidate/donor pool selection depends on it.

`advance_after_forward` is now the ONE shared function that (1) appends
fed positions to every layer's provenance, (2) synchronizes every layer via
the existing `_sync_layer_after_call` (reused, never reimplemented), (3)
requires cross-layer event agreement via `_note_event_once`, (4) returns
per-layer observations. `build_real_prefill_fn`, `build_real_decode_one_fn`,
and the restore-once branch evaluator (Defect 5) all call it — the
duplication-prone append-then-sync sequence exists in exactly one place.
`RealModelState` is documented as the single owner of the live model,
cache, complete `ModelProvenance`, `CompactionTracker`, and absolute
position (Pass 2's own separately-scoped, target-layer-only
`LayerProvenance` remains intentionally distinct — a freshly-built
projection feeding `capture_update_kv`'s position-map thunk, never a
second copy of `RealModelState`'s own provenance that could disagree with
it, since neither is ever read from the other).

**Tests:** `tests/unit/discovery/test_real_model_adapter_state.py` (11,
using a real-torch-CPU-tensor `FakeEvictingModel` with a trivial,
externally-verifiable "keep most recent `budget` positions" eviction
policy): prefill/decode position append correctness,
`test_provenance_reflects_real_eviction_not_left_empty` (the direct
regression test for this defect — proves the pre-event map available at
the exact moment of a real eviction is the correct dense sequence, not
empty), one-event-per-call, multiple-decode-eviction event ordering,
`compute_kept_indices_lengths` correctness.

### Defect 5 — per-token snapshot restore in branch evaluation

**Files:** `src/kvcot/discovery/real_model_adapter.py`.

The merged PR's `_build_real_branch_step_fn` restored a COMPLETE
`ModelStateSnapshot` into a fresh cache on EVERY call (49 restores for one
bridge-plus-48 branch). `build_real_branch_step_fn_restore_once` restores
only on the FIRST call (when it receives the branch's initial
`ModelStateSnapshot`) and reuses the already-restored live state (a
`_LiveBranchState` wrapper: cache, provenance, compaction, position) for
every later call in the SAME branch, detected via `isinstance` on the
state argument — `kvcot.discovery.branch_eval.evaluate_branch`'s existing
`(state, token_id) -> (logits, new_state)` contract is completely
unchanged; only the concrete real-model closure changed.
`evaluate_branch_from_snapshot` is the named entry point B1B-R3 §9
specified, delegating to `evaluate_branch` (never a second, independently-
written scoring loop). Baseline and swapped branches are evaluated from
two independently-cloned snapshots (`kvcot.discovery.pipeline
.build_swap_pair_record`, unchanged), so evaluation order cannot
contaminate results (pre-existing `test_branch_evaluation_order_does_not
_change_results` in `test_b1b_integration.py` already covers this at the
generic-interface level; unaffected by this change).

**Tests:** `test_real_model_adapter_state.py`'s branch section (4):
restore happens exactly once per branch (cache object identity stable
across calls after the first), second call reuses live state without
restoring (`is` identity), full-horizon scoring via
`evaluate_branch_from_snapshot`, two independent branches from the same
snapshot do not alias (`data_ptr()` disjoint).

### Defect 6 — inconsistent swap bookkeeping

**Files:** `src/kvcot/discovery/swap.py` (`apply_semantic_within_head_swap`,
new), `src/kvcot/discovery/pipeline.py` (wired in).

`apply_within_head_swap` (pre-existing) is a deliberately narrow, pure
cache-content primitive — it never touched provenance or R-KV kept-index
bookkeeping. `kvcot.discovery.pipeline.build_swap_pair_record` swapped K/V
content but left the swapped snapshot's `provenance`/
`kv_cluster_bookkeeping_per_layer` still reporting the DONOR's identity at
that slot. `apply_semantic_within_head_swap` now: (1) calls
`apply_within_head_swap` for the actual mutation (reused, never
reimplemented), (2) updates
`snapshot.provenance.layers[layer_index].positions[kv_head_index,
retained_post_storage_position]` from donor to candidate absolute
position (best-effort: skipped, not crashed, if `snapshot.provenance` is
`None`, e.g. a synthetic CPU-test snapshot), (3) updates the same identity
in `kv_cluster_bookkeeping_per_layer[layer_index]["kept_token_indices"][-1]`
— the exact list a subsequent real compaction event's remap
(`r1_kv.py:141-154`, `prev_indices = kept_token_indices[-1]`) reads —
so a later compaction on the SWAPPED branch remaps using the candidate's
identity, not the donor's. Returns a structured `SemanticSwapResult`
(before/after identities, which fields changed).

**Tests:** `tests/unit/discovery/test_swap.py` (+5): K/V + provenance +
bookkeeping all updated consistently, only the one targeted identity
changes (every other layer/head/slot's provenance is bit-for-bit
unchanged), no-op leaves identity unchanged, no aliasing between a
pristine snapshot and its independently-swapped clone, best-effort
behavior when provenance/bookkeeping are absent.

### Defect 7 — hard-coded evidence

**Files:** `src/kvcot/discovery/b2a_evidence.py` (new).

Every evidence field the merged PR hard-coded as a bare `True`/`0.0`
literal (`prefill_decode_boundary_parity`, `no_op_numerical_parity`,
`dataset_row_identity_match`, `manifest_hash_match`, `model_revision_match`,
`tokenizer_revision_match`, `generation_config_hash_match`,
`rkv_config_hash_match`, `observed_retention_ratio=0.0`, `event_count=
len(pair_records)` [conflating pair count with event count],
`projected_complete_pilot_gpu_hours=0.0`) is now derived from an actual
observation — see the module's own docstring for the exact source of each
field (§F below summarizes it per gate condition). `event_count` is now
`len({pr.compaction_event_id for pr in pair_records})` — distinct events,
never pair-record count. `observed_retention_ratio` is computed from
`trace.cache_length_final_per_layer` vs. `len(trace.full_token_ids)`.
`project_complete_pilot_gpu_hours`'s exact formula is documented in its own
docstring and reproduced in §F below.

**Tests:** `tests/unit/discovery/test_b2a_evidence.py` (9): invalid
example reports every trajectory field false (never vacuously true), valid
example reports them true, no-op parity requires an actual no-op record,
event count is distinct-events not pair-count, retention computed from
real cache lengths, meaningful-compression threshold, missing trace
degrades to zero without crashing, projection formula scales both
per-example and per-branch components correctly, projection nonzero
whenever any component is nonzero.

### Defect 8 — FullKV/R-KV process conflict

**Files:** `src/kvcot/discovery/b2a_workers.py` (new),
`src/kvcot/discovery/b2a_worker_entry.py` (new),
`src/kvcot/discovery/b2a_execute.py` (coordinator).

The merged PR ran a single R-KV process and reported its timing as ALSO
being the FullKV timing — contradicting this repository's own existing
rule (`kvcot.generation.state.declare_process_mode`,
`ProcessModeConflictError`: stock and patched models must never share a
process). `run_both_workers_via_subprocess` now launches
`python -m kvcot.discovery.b2a_worker_entry --role {fullkv|rkv} ...` as
TWO SEPARATE `subprocess.run` invocations, each writing its own
pydantic-schema-validated JSON result to a unique temp directory; the
coordinator reads both back, validates schemas, checks shared-identity
agreement (`dataset_repo`/`dataset_revision`/`manifest_hash`/
`prompt_token_ids_sha256` must match between the two workers), and cleans
up the temp directory only after combining succeeds.
`run_fullkv_worker`/`run_rkv_worker_body` are the two workers' real
bodies — `run_fullkv_worker` never loads R-KV; `run_rkv_worker_body` never
loads FullKV — so neither worker alone can ever violate the single-mode
rule, and the two never share a process.

**Tests:** `tests/unit/discovery/test_b2a_workers.py` (8, `subprocess_runner`
injected — no real Python subprocess, torch, or CUDA is ever touched):
successful combination, shared-identity mismatch detection, FullKV/R-KV
worker-failure propagation, missing-output-file detection, malformed-output
schema rejection, temp-directory cleanup verified via a `tempfile.mkdtemp`
spy.

### Defect 9 — no-op accounting

**Files:** `src/kvcot/discovery/constants.py` (`NoOpMode`,
`B2A_NOOP_CALIBRATION_COUNT`).

`kvcot.discovery.orchestrator.run_example`'s shared, CPU-test-relied-upon
behavior (one mandatory no-op pair per selected event, `4 cross-product +
1 no-op = 5` attempts) is UNCHANGED — this remains correct and required
for CPU tests (`NoOpMode.CPU_REQUIRED`). B2A evidence
(`no_op_numerical_parity`) is now explicitly derived from exactly ONE
no-op pair record (`B2A_NOOP_CALIBRATION_COUNT = 1`,
`NoOpMode.B2A_SINGLE_CALIBRATION`) found among the one example's pair
records — never re-counted as "one no-op per event" toward any total, and
never conflated with the 144-branch B2B figure. `cmd_plan_discovery`'s
pre-existing correct "144 real branches (no-op excluded)" wording is
unchanged.

**Tests:** `tests/unit/discovery/test_constants_noop_policy.py` (3): 144
total unchanged, `B2A_NOOP_CALIBRATION_COUNT == 1`, three distinct
`NoOpMode` members exist.

### Defect 10 — missing failure artifact

**Files:** `src/kvcot/discovery/b2a_artifact.py` (new),
`src/kvcot/discovery/b2a_execute.py`.

The merged PR only wrote a result file after `gate_result.passed` was
checked from the CLI layer, and never on an exception. `run_b2a_calibration`
now wraps its entire body (prompt-identity verification, subprocess
coordination, evidence construction, gate evaluation) in a `try/except`
that writes an immutable artifact — pass OR fail OR raised-exception — via
`kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact`
(write-temp-then-atomic-rename, refuses to overwrite, unique
`b2a_<timestamp>_<config-hash-prefix>_<manifest-hash-prefix>.json` path,
never a fixed filename) before re-raising. A worker-launch failure, a
prompt-identity mismatch, or any other exception all produce a fail
artifact recording the failure reason and traceback.

**Tests:** `tests/unit/discovery/test_b2a_artifact.py` (6): path
composition, pass/fail round-trip, unique paths across two writes,
overwrite refusal, no leftover temp file on success.
`test_b2a_execute_coordinator.py`'s worker-failure and prompt-identity-
refusal tests both assert exactly one fail artifact is written.

### Defect 11 — oversized selected captures

**Status: NOT implemented in this pass — see §I, remaining blockers.**
`kvcot.discovery.capture.UpdateKvCaptureRecord` still stores full pre-call/
returned K/V tensors per selected target (bounded by the 3 target count
since B1B-R2's `should_capture` predicate, never by total call volume —
that part of the original concern IS already fixed). A further tightening
to store only the selected candidate/donor vectors was assessed as
requiring changes to `kvcot.discovery.capture`, a heavily-tested module
shared with the generic CPU harness path, and was deferred rather than
risked in the time available for this pass. Documented honestly here, not
silently claimed complete.

### Defect 12 — frozen-versus-runtime configuration drift

Covered under Defect 1 above (`runtime_rkv_config_hash`).

## 3. Frozen framework seed (§15)

**File:** `src/kvcot/discovery/framework_seed.py` (new).

`apply_framework_seed(framework_seed, attention_backend, cuda_available)`
seeds Python's `random` and PyTorch's CPU (and, when `cuda_available`,
CUDA) RNGs, and returns a `DeterminismPolicy` recording exactly what was
applied — never claiming full bitwise determinism for
`attention_backend="flash_attention_2"` (FlashAttention kernels are not
guaranteed bitwise-deterministic independent of seeding; token-identical
replay parity is verified by direct comparison, `token_identical_replay`
evidence, never assumed from seeding alone). The selection seed
(`kvcot.discovery.sampling.IdentitySeedParts`) and the framework execution
seed (`config.generation.framework_seed`, frozen at `13`) remain clearly
distinct — this module only ever touches the latter.

**Tests:** `tests/unit/discovery/test_framework_seed.py` (4).

## 4. Frozen identity (resolved for real in this pass)

- Model revision: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` @
  `6a6f4aa4197940add57724a7707d069478df56b1` (discovery track only,
  CLAUDE.md §1a — never the primary Qwen pipeline).
- Tokenizer revision: same as model revision (single repo).
- Dataset revision: `HuggingFaceH4/MATH-500` @
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- Example: index `0`, `unique_id="test/precalculus/807.json"`.
- Raw-row hash: `0d4aaa49ce633bbae51053c17c1b9703daca6c02271f7e52dfa36c73d05c9a96`
  (corrected from the merged PR's non-reproducible value — see Defect 2).
- Prompt token count: `105`.
- Prompt token-ID hash: `57c824e6f41bc098507f3a40b9784b8d17dddc000c24334e8d209133771ee465`.
- Rendered user-message hash, chat-template source hash, chat-message
  payload hash, and the complete 105-entry prompt token-ID array are all
  committed in `configs/discovery/b2a_one_example_manifest.json`.
- Manifest hash (this file's own `sha256_json` over every field):
  `e202f81747c18c0d70cf2fce6a890a36146612bb7580fa47a9010f14fa28ca93`.
- Canonical config hash, generation-config hash, and (discovery-track)
  frozen R-KV config hash are all printed by
  `kvcot plan-discovery --dry-run` / `kvcot b2a-calibrate --dry-run` (they
  are deterministic functions of `configs/discovery/llama8b_math500_b1024.yaml`,
  unchanged by this pass except for the new `kernel_size` field now being
  load-bearing).
- The **runtime** R-KV config hash does not exist yet — it can only be
  computed from a live, loaded model, which requires the still-unauthorized
  GPU run. Not fabricated; genuinely absent.

## 5. Execution architecture

```
kvcot b2a-calibrate --execute
  └── kvcot.discovery.b2a_execute.run_b2a_calibration   (coordinator)
        ├── _verify_resolved_prompt_identity              (re-fetch/re-render/re-tokenize/re-hash; refuses on mismatch)
        └── kvcot.discovery.b2a_workers.run_both_workers_via_subprocess
              ├── subprocess: python -m kvcot.discovery.b2a_worker_entry --role fullkv
              │     └── kvcot.discovery.b2a_workers.run_fullkv_worker
              └── subprocess: python -m kvcot.discovery.b2a_worker_entry --role rkv
                    └── kvcot.discovery.b2a_execute.run_rkv_worker_body
                          ├── apply_framework_seed
                          ├── RKVPolicy.load (+ runtime R-KV config verification)
                          ├── kvcot.discovery.orchestrator.run_example
                          │     (Pass 1 → Pass 2 → per-pair branch construction,
                          │      all via kvcot.discovery.real_model_adapter's
                          │      advance_after_forward / restore-once branch eval)
                          └── kvcot.discovery.b2a_evidence.derive_trajectory_evidence
  └── kvcot.discovery.b2a_evidence.project_complete_pilot_gpu_hours
  └── kvcot.discovery.b2a_contract.evaluate_b2a_gate
  └── kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact   (ALWAYS — pass, fail, or exception)
```

State lifecycle: `RealModelState` (single owner of live model/cache/
provenance/compaction/position) + `advance_after_forward` (one shared
append-then-sync step) used identically for Pass 1 prefill, Pass 1 decode,
Pass 2 prefill, Pass 2 decode, and branch continuation.

Branch evaluation: `build_real_branch_step_fn_restore_once` restores a
complete snapshot exactly once per branch; every subsequent token in that
branch reuses the already-restored live state.

Artifact behavior: one immutable, uniquely-named JSON artifact under
`results/decisions/` per B2A attempt, written whether the gate passes,
fails, or the run raises before either worker completes.

## 6. Evidence derivation (per mandatory gate field)

See `kvcot.discovery.b2a_evidence`'s module docstring for the authoritative,
field-by-field explanation (reproduced in the final report's §F). In
summary: five trajectory fields (`token_identical_replay`,
`prefill_decode_boundary_parity`, `compaction_position_equality`,
`capture_gather_parity`, `absolute_position_parity`) are derived from
`example_result.valid` (Pass 2 structurally cannot return `valid=True`
without all five holding); `no_op_numerical_parity` from an actual no-op
pair record's presence; dataset/prompt identity fields from the pre-flight
refusal gate having already passed; `rkv_config_hash_match` from a genuine
runtime read-back; `model_revision_match`/`tokenizer_revision_match`/
`generation_config_hash_match` structurally (explicit `revision=` kwargs,
single-source-of-truth config object — the weakest of the checks in this
module, documented as such); `observed_retention_ratio`/`event_count`/
`meaningful_compression_observed`/`sufficient_eligible_events` from actual
trace/pair-record data; `projected_complete_pilot_gpu_hours` from the
documented formula in §7 below. **Not yet measurable until a real GPU run:**
the runtime R-KV config hash itself, all wall-clock/VRAM numbers, and
therefore the two threshold-derived conditions
(`runtime_within_limit`/`peak_vram_within_limit`).

## 7. Accounting

```
B2B: 12 examples × 3 events × 4 real swaps = 144 real branches
B2A no-op: one separate numerical calibration (B2A_NOOP_CALIBRATION_COUNT = 1), excluded from 144
```

Projected pilot runtime formula (`kvcot.discovery.b2a_evidence
.project_complete_pilot_gpu_hours`):

```
per_example_seconds = fullkv_natural_generation_wall_seconds + rkv_pass1_wall_seconds
                     + token_identical_pass2_wall_seconds + score_recomputation_wall_seconds
                     + targeted_capture_wall_seconds
per_branch_seconds  = cache_clone_restore_wall_seconds + one_fixed_shape_swap_wall_seconds
                     + bridge_plus_48_scored_wall_seconds
projected_seconds   = 12 * per_example_seconds + 144 * per_branch_seconds
projected_gpu_hours = projected_seconds / 3600
```

## 8. Remaining blockers

- No real Llama-8B/R-KV CUDA execution has occurred under this pass.
- RTX 3090 memory and runtime remain unmeasured.
- The runtime R-KV configuration hash does not exist without a live model
  load — genuinely unmeasured, not fabricated.
- Defect 11 (further capture-size tightening below B1B-R2's already-
  bounded-by-target-count level) is NOT implemented in this pass.
- B2A remains blocked pending independent review of this pass.
- B2B (the 12-example, 144-branch pilot) remains blocked; no B2B command
  exists.
- No FaithKV method exists in this repository.
