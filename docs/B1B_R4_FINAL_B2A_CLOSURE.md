# B1B-R4 — final executable, measurement, and worker-evidence closure

Run on branch `research/b1b-r4-final-b2a-closure`, cut from `main` at
commit `fa117046bea2a2c492e17cd91276b2e3c6d59f7f` (the merged PR #19,
B1B-R3). PR #19 was merged as that exact SHA — this document's corrections
are forward fixes on top of it, never a rewrite of that history.

**Status: B1B-R4 IMPLEMENTATION — READY FOR INDEPENDENT CPU AUDIT. B2A and
all GPU/model-inference activity remain blocked pending a separate,
explicit, future authorization exactly as CLAUDE.md §1a/§1b already
require.** This document does not claim "ready for B2A" anywhere, and does
not claim completeness beyond what is explicitly listed below.

## 1. What this pass repairs

An independent audit of merged PR #19 found the one-example B2A
implementation still contained several defects the B1B-R3 closure document
had not fully closed. This pass repairs, with real code and CPU tests
(never GPU-executed):

1. **Execution-count semantics** (`kvcot.discovery.constants`): "branches"
   renamed to "pair evaluations" throughout (`REAL_PAIR_EVALUATIONS_PER_EVENT`,
   `B2A_REAL_PAIR_EVALUATIONS_TOTAL`, `B2A_NOOP_PAIR_EVALUATIONS_TOTAL`,
   `B2B_PILOT_TOTAL_REAL_PAIR_EVALUATIONS`), old names kept as deprecated
   aliases. B2A is now defined, everywhere, as exactly 3 selected events x
   4 real pair evaluations = 12 real pair evaluations, PLUS exactly 1
   no-op pair evaluation (separate, never folded in). B2B is 12 examples x
   3 events x 4 real = 144 real pair evaluations, 0 GPU no-op evaluations.

2. **FullKV replaced with exact greedy generation**
   (`kvcot.discovery.b2a_workers.run_fullkv_worker`): no longer calls
   `generate_base(..., temperature=0.0, top_p=1.0, generator=None)` (a
   sampling routine given degenerate parameters). Now reuses
   `kvcot.discovery.pass1.run_natural_pass1` together with
   `kvcot.discovery.real_model_adapter`'s real `PrefillFn`/`DecodeOneFn` --
   the IDENTICAL greedy/argmax, one-prefill-then-one-decode-per-token loop
   R-KV's own Pass 1 uses. No sampling function is ever called; EOS is
   never appended or fed.

3. **Framework determinism applied and recorded independently in both
   workers** (`kvcot.discovery.framework_seed.apply_framework_seed`, now
   called inside `run_fullkv_worker` AND `run_rkv_worker`, each in its own
   OS process). Both workers report the complete `DeterminismPolicy`;
   neither claims bitwise determinism under `flash_attention_2`.

4. **No-op mode is now an actual execution control**
   (`kvcot.discovery.orchestrator.PairExecutionPolicy`): `NoOpMode` is
   threaded into `run_example`'s pair-construction loop, not merely
   documented. `CPU_REQUIRED` (default, unchanged CPU-harness behavior: one
   no-op per selected event), `B2A_SINGLE_CALIBRATION` (exactly one no-op,
   for the first selected event only), `DISABLED` (zero no-op, for future
   B2B). `run_example` fails closed (`ValueError`) on an unrecognized mode.

5. **Independent trajectory/parity/identity/runtime-config evidence**
   (`kvcot.discovery.b2a_evidence`, `kvcot.discovery.call_trace`,
   `kvcot.discovery.runtime_evidence`): `token_identical_replay`,
   `prefill_decode_boundary_parity`, `compaction_position_equality`,
   `capture_gather_parity`, and `absolute_position_parity` are now five
   independently-derived booleans (from `Pass2Result.invalid_reason`, an
   INDEPENDENT Pass-1-vs-Pass-2 call-boundary trace comparison, and each
   selected target's own capture-parity flags) -- never all five read off
   one `example_valid` boolean. Resolved-vs-requested model/tokenizer
   revision is read back via `transformers`' own `config._commit_hash`/
   `tokenizer._commit_hash` (a genuine runtime read-back, `kvcot.discovery
   .runtime_evidence.read_resolved_model_revision`/
   `read_resolved_tokenizer_revision`) -- an unresolved readback is `False`,
   never silently `True`. `generation_config_hash_match` is a real
   comparison of the two workers' independently-computed runtime generation
   hashes, never a literal `True`.

6. **Batch size, parameter placement, and one-example scope are derived**
   (`kvcot.discovery.runtime_evidence.derive_parameter_placement`/
   `derive_batch_size_from_input_ids`): every parameter's `.device.type` is
   walked plus `hf_device_map` inspected for cpu/disk/meta entries (the gate's
   `no_offload_verified` uses this STRONGER check, not the weaker top-level
   `every_parameter_on_cuda` alone -- caught and fixed during this pass's own
   adversarial self-review, §5 below). `batch_size` is read off the real
   input tensor's shape. `one_example_only` is derived from both workers'
   observed `prompt_token_count` matching the manifest's own.

7. **Timing redesigned, non-overlapping, no aggregate-times-144**
   (`kvcot.discovery.orchestrator.run_example`'s injectable `clock_fn`,
   `kvcot.discovery.b2a_workers._RkvHarnessInstrumentation`,
   `kvcot.discovery.b2a_evidence.per_real_pair_projection_seconds`): one
   wall-clock duration per COMPLETED real pair evaluation and per no-op
   evaluation (never an aggregate bucket), Pass 1 vs Pass 2 attributed via
   the structural first-vs-second-prefill-call rule, snapshot/targeted-
   capture time folded into Pass 2's total exactly once (a genuine
   double-counting bug this pass's own self-review found and fixed, §5).
   Projection: `12 * per_example_total + 144 * max(the 12 real pair
   durations)` -- never sums the 12 and multiplies by 144.

8. **Branch compaction history restored, not reset**
   (`kvcot.discovery.real_model_adapter.restore_compaction_tracker_from_snapshot`):
   a branch restored from a `ModelStateSnapshot` now reconstructs its
   `CompactionTracker` from that snapshot's own `compaction_event_steps`/
   `tokens_since_last_compaction`, instead of `CompactionTracker()` (an
   always-empty tracker that silently discarded prior history).

9. **VRAM gated on max(allocated, reserved)**
   (`kvcot.discovery.b2a_contract.B2AOneExampleMeasurement.peak_vram_gib`,
   `MAX_PEAK_TRACKED_MEMORY_GIB`): both workers reset peak memory stats at
   the identical point (after model/tokenizer load, before measured
   inference) and report allocated-before/reserved-before/peak-allocated/
   peak-reserved; the gate compares the maximum of the two peaks.

10. **Weight-cache guard restricted to manifest preparation**
    (`kvcot.discovery.manifest_prepare`): the guard is no longer inside the
    shared `_render_and_tokenize` (used by BOTH manifest preparation and
    generic B2A prompt-identity re-verification) -- it now wraps only
    `resolve_prompt_identity`'s own call, as a before/after snapshot diff
    (`_snapshot_weight_shaped_files`/`_assert_no_new_weight_files_introduced`)
    that fails only on a genuinely NEW weight-shaped file, never a
    pre-existing one.

11. **Partial worker evidence preserved; durable worker envelopes**
    (`kvcot.discovery.b2a_workers.WorkerFailedError.partial_fullkv_result`,
    `kvcot.discovery.worker_envelope`): if R-KV fails after FullKV
    succeeds, the FullKV result is folded into the fail artifact rather
    than discarded. Every worker subprocess launch uses `capture_output=
    True, text=True, timeout=B2A_WORKER_TIMEOUT_SECONDS (7200), check=
    False`; a hung worker raises `WorkerFailedError` from
    `subprocess.TimeoutExpired`, never blocks forever. Each worker attempt
    always writes a `WorkerEnvelope` (`<output>.envelope.json`), success or
    failure -- attempt id, identities, partial measurements, determinism
    policy, error type/message/traceback.

12. **Collision-resistant artifact naming**
    (`kvcot.discovery.b2a_artifact.build_artifact_path`): microsecond-
    resolution UTC timestamp plus a random UUID4 hex suffix, never
    second-resolution-only.

13. **Selected captures minimized** (`kvcot.discovery.capture_minimize`):
    `MinimizedTargetEvidence` retains only the selected candidate/donor K/V
    row vectors and scalar scores for one (layer, kv_head) target --
    bounded at `(CANDIDATES_PER_EVENT + DONORS_PER_EVENT) * 2 * head_dim`
    scalars regardless of `num_layers`/`num_kv_heads`/cache length
    (asserted by `assert_minimized_bound`, tested against a synthetically
    20,000-token cache). Built inside `run_example`, immediately after
    Pass 2, before the object this function returns is ever handed to a
    caller -- no full-layer/full-cache tensor is reachable from
    `ExampleResult`.

14. **One canonical worker API**
    (`kvcot.discovery.b2a_workers.run_fullkv_worker`/`run_rkv_worker`): the
    B1B-R3 split (a `NotImplementedError` stub in `b2a_workers.run_rkv_worker`
    while the real body lived in `kvcot.discovery.b2a_execute
    .run_rkv_worker_body`, called directly by the worker entry point) is
    gone. `kvcot.discovery.b2a_worker_entry` calls exactly these two
    functions, with no other keyword arguments, for both roles.

15. **CPU tests execute the real worker-control bodies**
    (`tests/unit/discovery/test_b2a_workers_real_bodies.py`): both
    functions accept internal, underscore-prefixed dependency-injection
    seams (`_load_model`/`_load_tokenizer`/`_fresh_cache_factory`/`_cuda`/
    `_device`) never exposed by the production CLI/subprocess entry point
    (verified by source inspection in
    `test_production_call_shape_never_passes_injection_kwargs`) -- CPU
    tests inject small, real-torch-CPU-tensor fakes and exercise the ENTIRE
    function body (seed application, model/tokenizer loading seam, the
    real greedy loop, runtime R-KV verification against a real fake
    `kv_cluster`, worker-result construction) from real, not preconstructed,
    values. Scope note: the fake model used never naturally emits EOS, so
    these tests reach Pass 1's `cap_hit` path, not the full 12-real+1-no-op
    success path -- documented, not silently claimed complete (§6 below).

16. **Gate schema strengthened**
    (`kvcot.discovery.b2a_contract.MANDATORY_GATE_CONDITIONS`): four new
    required conditions -- `selected_event_count_exact`,
    `real_pair_count_exact`, `no_op_count_exact`,
    `all_required_pair_evaluations_completed` -- each derived from the
    R-KV worker's own independently-countable selection/pair-completion
    fields (`kvcot.discovery.b2a_evidence.derive_pair_completion_evidence`),
    never a single umbrella validity boolean.

17. **Measurement naming/event counts corrected**: `observed_total_
    compaction_events`, `eligible_compaction_events`,
    `selected_compaction_events`, and `events_with_all_four_real_pairs_
    completed` are all reported separately -- `selected_compaction_events`
    (not `len(pair_records)`, which conflates real and no-op pairs) is what
    feeds `meaningful_compression_observed`/the measurement's `event_count`.

## 2. Execution accounting (frozen)

```
B2A:
  3 selected events
  12 real swap-pair evaluations (12 baseline + 12 swapped continuations)
  1 no-op pair evaluation

B2B (not executed by this pass):
  12 examples x 3 events x 4 real swap-pair evaluations = 144
  0 GPU no-op evaluations
```

## 3. Git state

- Base `main` SHA at the start of this pass: `fa117046bea2a2c492e17cd91276b2e3c6d59f7f`
  (confirmed via `git merge-base --is-ancestor`, trivially true since `main`
  was exactly at that commit).
- Branch: `research/b1b-r4-final-b2a-closure`.
- No reset, rebase, revert, squash, or merge of PR #19's history occurred.

## 4. What remains genuinely incomplete (stated plainly, not hidden)

- **Full 7-8-phase per-branch timing decomposition** (§12's optional finer
  breakdown: clone/restore-only, semantic-swap-mutation-only,
  bridge-plus-scored-only, separately for baseline and swapped) was NOT
  built. This pass's timing repair stops the two most severe defects
  (conflating restore with bridge-forward inside one bucket, and
  multiplying an aggregate 12-pair bucket by 144) by measuring one
  wall-clock duration per WHOLE pair evaluation instead -- a real, tested,
  non-overlapping improvement, but coarser than the fully-decomposed
  design the brief describes as "recommended."
- **Worker-envelope hash is not yet threaded into the coordinator's final
  artifact.** Each worker attempt writes a durable envelope file
  (`kvcot.discovery.worker_envelope`), but `kvcot.discovery.b2a_workers
  .run_both_workers_via_subprocess` does not currently read it back or
  include its hash in `WorkerCoordinationResult`/the final artifact
  payload. The envelope exists on disk for post-mortem debugging; it is
  not yet cross-referenced from the pass/fail artifact.
- **The full 12-real+1-no-op success path is not exercised by any CPU
  test.** `test_b2a_workers_real_bodies.py`'s fake model deterministically
  never emits EOS, so Pass 1 always hits `cap_hit` first. The orchestrator-
  level success path (12 real pairs + 1 no-op, all evidence fields
  `True`/exact) IS exercised, extensively, by
  `test_orchestrator_pair_execution_policy.py` and
  `test_b1b_integration.py` against the synthetic Pass1/Pass2 harness --
  but not by a `run_rkv_worker`-level test with a real-adapter-shaped fake
  model reaching that far.
- These are the same kind of honestly-stated scope limitation this
  repository's own prior closure documents use (e.g. B1B-R3's
  `score_recomputation_wall_seconds` deferral) -- not silently omitted.

## 5. Adversarial self-review findings (§24)

A hostile second pass over this diff, searching explicitly for the
forbidden literal patterns and re-deriving every mandatory gate condition's
raw source, found and fixed three real defects before this branch was
committed:

1. **`no_offload_verified` used the weaker check.** The gate's
   `no_offload_verified` condition was derived from
   `measurement.every_parameter_on_cuda`, itself built from each worker's
   top-level `every_parameter_on_cuda` field -- a per-parameter `.device
   .type` walk that CANNOT detect a `device_map="auto"` load with an
   offloaded entry still present in `hf_device_map` (proven by
   `test_device_map_offload_entry_fails_no_offload_even_if_params_report_cuda`
   in `test_runtime_evidence.py`). Fixed: the coordinator now uses
   `parameter_placement["no_offload_verified"]` (the stronger check) from
   both workers. Regression test:
   `test_no_offload_gate_uses_the_stronger_device_map_check_not_just_every_parameter_on_cuda`.

2. **Identity conditions only checked worker-vs-worker, not worker-vs-
   manifest.** `dataset_revision_match`/`dataset_row_identity_match`/
   `manifest_hash_match`/`prompt_token_hash_match` were all derived solely
   from `coordination.shared_identity_ok` (FullKV agrees with R-KV) --
   two workers that agreed with each other on a WRONG value (e.g. both
   reporting the wrong `manifest_hash`) would have passed. Fixed: each
   condition now requires (1) FullKV matches the coordinator's own
   manifest-derived expected value, (2) R-KV matches it too, (3) the two
   workers agree with each other -- all three. Regression test:
   `test_both_workers_agreeing_with_each_other_but_not_the_manifest_still_fails_the_gate`.

3. **Snapshot/targeted-capture time was measured but never included in
   Pass 2's total**, silently UNDER-counting `wall_seconds_pass2` (and
   therefore `per_example_total_wall_seconds` and the projection) by
   however long the snapshot calls actually took. Fixed:
   `_RkvHarnessInstrumentation.snapshot` now adds its elapsed time into
   `pass2_wall_seconds` as well as the pre-existing diagnostic
   `targeted_capture_wall_seconds` breakdown -- added exactly once, per
   B1B-R4 §12's own "not added again if already contained in Pass 2 total"
   rule. Regression test:
   `test_snapshot_time_is_folded_into_pass2_wall_seconds_exactly_once`.

No other occurrence of the forbidden literal patterns
(`generation_config_hash_match=True`, `one_example_only=True`,
`every_parameter_on_cuda=True`, `batch_size=1` as a hard-coded assignment,
`token_identical_replay=example_valid` and its four siblings,
`swap_seconds=0.0`, `score_recomputation_wall_seconds=0.0`,
`CompactionTracker()` used to discard branch history,
`pairs_to_build = cross_product + noop` unconditionally,
`temperature=0.0`/`generator=None` in a sampling call) was found in
production code -- every remaining textual match is either a docstring
describing the repaired defect, or a legitimate fresh-state construction
at the START of Pass 1 (never mid-branch-restoration).

## 6. Test evidence

```
python -m compileall src tests -q          # clean, no output
python -m pytest --collect-only -q         # 945 tests collected, 0 errors
python -m pytest -m "not gpu" -q           # 931 passed, 14 deselected (gpu-marked)
git diff --check                           # no whitespace errors (CRLF-conversion notices only)
```

`python -m kvcot prepare-b2a-manifest --config configs/discovery/llama8b_math500_b1024.yaml --dry-run`
and
`python -m kvcot b2a-calibrate --config configs/discovery/llama8b_math500_b1024.yaml --dry-run`
both run to completion: no model loaded, no CUDA required, no result files
created. The b2a-calibrate dry-run prints the resolved manifest/config
identity, the exact `12 real pair evaluations + 1 no-op pair evaluation`
B2A accounting and `144 real pair evaluations, 0 GPU no-op` B2B accounting
(both now using the frozen §4 vocabulary), and every config/manifest hash.
`b2a-calibrate --execute` was NOT run.
