# B1B-R4.1 — focused completion amendment on B1B-R4 (CPU-only, no GPU)

Run on branch `research/b1b-r4-final-b2a-closure`, on top of commit
`4d7971b7b09c004c4670bfde3939416ab550ea71` ("Complete B1B-R4 final B2A
closure", already pushed). This document does not undo, reset, or rewrite
that commit — it is a forward completion pass fixing real defects an
audit found in the merged B1B-R4 state, plus new defects this pass's own
hostile self-review found in its own diff.

**Status: B1 FINAL CLOSURE VERDICT: INCOMPLETE — B2A/GPU REMAIN BLOCKED.**
This document does not claim full closure of every item in the originating
task brief (41 sections). It closes seven concrete, verified defects plus
one gate-condition addition, all with new CPU tests, and states plainly,
in §5 below, everything from the brief that remains unimplemented. No GPU
inference occurred. No model weights were downloaded. No RTX 3090
measurement exists. `configs/lock.yaml` is untouched. `third_party/R-KV`'s
pinned commit is untouched. B2A and B2B remain unauthorized to execute.

## 1. Audit method

Before writing any code, an Explore-agent audit was run against the actual
current state of `src/kvcot/discovery/` (timing, memory, gate conditions,
schemas, selected-event accounting, pair-failure evidence, semantic-swap
evidence, no-op evidence, artifact/envelope atomicity, determinism policy,
capture-memory bounds, branch-memory sequencing) — ground truth, not the
task brief's assumed defect list. Several assumed defects turned out to
already be fixed in the merged B1B-R4 state (Pydantic schemas already used
pervasively; most gate conditions already existed under slightly different
names; artifact writes were already atomic). The seven repairs below are
the CONFIRMED gaps that audit found, each verified by reading the actual
code before changing it.

## 2. Repairs (each with new CPU tests, full suite green after every one)

1. **One authoritative Pass-1/Pass-2 provenance state**
   (`kvcot.discovery.real_model_adapter.RealModelState`). Previously,
   `kvcot.discovery.pass2.run_pass2_capture` maintained its own mutable
   `dict[int, LayerProvenance]`, hand-kept in lockstep with
   `advance_after_forward`'s appends by two independently-written call
   sites — a real risk of silent divergence. `RealModelState` now owns
   `pending_fed_absolute_positions`/`pending_call_kind`
   (`register_pending_fed_positions`/`clear_pending`) and a
   `projected_pre_event_position_map` method that derives the pre-event map
   on demand from the one authoritative `model_provenance`, via a
   disposable `LayerProvenance.clone()` mutated with the SAME
   `append_new_token`/`append_new_tokens_prefill` methods
   `advance_after_forward` itself uses — never a second, independently
   written position-arithmetic implementation. `_pending_positions_scope`
   registers before every real forward call
   (`build_real_prefill_fn`/`build_real_decode_one_fn`/the branch step
   function) and clears on exception before `advance_after_forward` is ever
   attempted, so a failed forward call never leaves a partial commit.
   `pass2.py` detects (`hasattr(state, "projected_pre_event_position_map")`)
   whether it owns its own authoritative projection and, when it does,
   builds NO parallel `LayerProvenance` track at all — the CPU synthetic
   harness path (no such method) is untouched. Tests:
   `tests/unit/discovery/test_real_model_adapter_state.py` (9 new tests:
   projection with/without pending, mid-call visibility via a class-level
   monkeypatch spy, exception clearing with no partial commit for both
   `RealModelState` and `_LiveBranchState`, and a full Pass-1→Pass-2
   integration proving `LayerProvenance.empty` is never called on the real
   path and multi-compaction remapping is correct). This pass's own hostile
   review of that integration test found and documented (not silently
   papered over) a genuine, pre-existing edge case in the ported R-KV
   bookkeeping formula (`_fake_rkv_fixtures.FakeR1KV`): when an eviction
   event reorders storage without actually shrinking the cache
   (`kv_cache_len == budget` exactly), `evicted_token_num` stays at 0, so
   the NEXT event's remap incorrectly skips `prev_indices` translation —
   orthogonal to this repair (this repository's own provenance tracking
   reports the objectively correct positions throughout; it is R-KV's own
   bookkeeping that becomes ambiguous at that boundary), avoided in the new
   test by choosing parameters where the first eviction genuinely exceeds
   budget.

2. **Selected-event count derived from the frozen plan, not from surviving
   pair records** (`kvcot.discovery.orchestrator.ExampleResult
   .selected_event_ids`, `kvcot.discovery.b2a_evidence
   .derive_pair_completion_evidence`). Previously, `selected_compaction_
   events` (which feeds the `selected_event_count_exact` gate condition)
   counted distinct event IDs appearing in `pair_records` — an event every
   one of whose pairs failed attrition would silently vanish from the
   count instead of being reported as selected-but-failed. `run_example`
   now populates `selected_event_ids` directly from `plan.events` right
   after `build_pass1_plan` succeeds; `PairCompletionEvidence` gained a
   separately-named `events_with_at_least_one_completed_real_pair` field so
   the two quantities (planned vs. at-least-partially-completed) are never
   conflated again. Tests: `tests/unit/discovery/test_b2a_evidence.py`
   (`test_selected_event_count_comes_from_the_frozen_plan_not_surviving_
   pair_records` reproduces the exact bug scenario: 3 planned events, only
   2 with any surviving pair, asserting the count still reports 3).

3. **Structured per-pair failure evidence, actually populated**
   (`kvcot.discovery.attrition.PairFailureDetail`,
   `kvcot.discovery.orchestrator.run_example`). Previously,
   `pair_failure_details` was always an empty tuple in the production R-KV
   worker path: `derive_pair_completion_evidence`'s
   `pair_attrition_dropped_stages` parameter existed but
   `kvcot.discovery.b2a_workers.run_rkv_worker` never actually passed it.
   `run_example` now builds one `PairFailureDetail` (event/layer/head/
   candidate/donor/kind/stage/detail/elapsed-time) per failed pair attempt,
   live, at the exact point of failure, and `ExampleResult.pair_failure_
   details` carries them through; `derive_pair_completion_evidence` reads
   `example_result.pair_failure_details` directly (the misleading unused
   parameter was removed rather than left silently ineffective).
   `RKVWorkerResult.pair_failure_details` is now `list[PairFailureDetail]`
   (Pydantic natively validates a plain dataclass field). Test:
   `tests/unit/discovery/test_b1b_integration.py
   ::test_pair_failure_details_records_exactly_the_failed_pairs_not_an_
   empty_placeholder` forces exactly one of 15 real pair attempts to fail
   via a patched `build_swap_pair_record` and asserts the resulting detail
   names exactly that pair, while the other 14 still succeed.

4. **The capture-minimization bound is enforced in production, not only in
   its own test file** (`kvcot.discovery.orchestrator.run_example`).
   `kvcot.discovery.capture_minimize.assert_minimized_bound` existed but
   had zero call sites outside `test_capture_minimize.py`. `run_example`
   now calls it on every `MinimizedTargetEvidence` it builds, immediately.
   Test: `test_run_example_enforces_the_minimized_capture_bound_not_just_
   the_test_suite` monkeypatches an oversized evidence object and confirms
   `run_example` now raises `CaptureMinimizationError`.

5. **Baseline and swapped branch snapshots are never live simultaneously**
   (`kvcot.discovery.pipeline.build_swap_pair_record`). Previously both
   `pristine.clone()`s (baseline, swapped) were created up front and both
   held as local variables for the whole pair-construction call (via
   `evaluate_swap_branches`'s combined signature) — on a real model this
   doubles peak per-pair memory for no reason. Baseline is now cloned,
   evaluated (`kvcot.discovery.branch_eval.evaluate_branch`, called
   directly instead of the combined wrapper), and explicitly `del`eted
   (in a `finally`, covering the exception path too) BEFORE the swapped
   clone is even created. Test:
   `test_baseline_snapshot_clone_is_released_before_swapped_clone_is_
   created` uses a `weakref` to the exact `ModelStateSnapshot.clone()`
   result and a custom step function that never hands back the same
   object (mirroring the real adapter's restore-once branch stepping) to
   prove the baseline clone is actually garbage-collected — not merely
   "eventually released" — before the pair finishes.

6. **Semantic-swap parity and byte-delta evidence derived from the real
   mutation report, not hard-coded** (`kvcot.discovery.pipeline
   .build_swap_pair_record`). Previously `parity_check_passed=True`/
   `net_physical_bytes_changed=0` were literal values — a direct instance
   of the forbidden pattern this whole exercise exists to eliminate,
   despite `apply_semantic_within_head_swap` already returning a real
   `SemanticSwapResult` (`provenance_updated`/`kept_index_bookkeeping_
   updated`/`is_noop`) that was computed but never consumed.
   `parity_check_passed` is now derived from three independent checks
   (capture-record parity, provenance update — mandatory whenever the
   pristine snapshot carries provenance, kept-index bookkeeping update —
   mandatory whenever that bookkeeping is present for the target layer,
   each gated on its OWN presence signal, never a shared proxy, since the
   real-model adapter's snapshots carry both together but the CPU synthetic
   harness's snapshots deliberately carry bookkeeping without provenance).
   `net_physical_bytes_changed` is now `sum(numel*element_size)` before vs.
   after the swap across every layer's K/V (always computes to 0 for a
   successful swap, since `apply_within_head_swap` already rejects any
   shape change — the point is that it is COMPUTED, so a future shape
   regression would be caught). `valid_flag`/`invalid_reason` now equal the
   derived parity result instead of a separate hard-coded `True`/`None`,
   consistent with `build_swap_pair_record`'s own "same code path, never a
   special case" design. Tests:
   `test_semantic_swap_parity_is_derived_and_catches_a_missing_provenance_
   update` (monkeypatches `apply_semantic_within_head_swap` to report a
   missed update on a snapshot carrying a real, correctly-shaped dummy
   provenance object, confirms the resulting record is schema-valid but
   `parity_check_passed=False`/`valid_flag=False` with a reason naming the
   defect) and `test_semantic_swap_parity_passes_and_reports_zero_bytes_
   changed_on_the_real_happy_path`.

7. **`PYTHONHASHSEED` set on the subprocess environment before launch, not
   `random.seed()` inside the already-running worker**
   (`kvcot.discovery.b2a_workers._worker_subprocess_env`, `_launch_worker`).
   `random.seed()` does not and cannot control Python's hash-randomization
   seed — that is fixed once, at interpreter startup, from the
   `PYTHONHASHSEED` environment variable, before any of this repository's
   code runs; `kvcot.discovery.framework_seed.apply_framework_seed` was
   calling `random.seed()` inside the worker process, which cannot affect
   that same process's already-fixed hash seed. `_launch_worker` now builds
   the child environment explicitly, reading `framework_seed` off the same
   frozen config the worker itself will load (falling back to
   `DiscoveryGenerationLock`'s own schema default only when the config path
   cannot be loaded at all — e.g. coordinator-level CPU tests using
   intentionally-fake paths; the worker's own load moments later remains
   the authoritative check and still fails loudly on a genuinely malformed
   config) — never an independently-invented value. Also sets
   `TOKENIZERS_PARALLELISM=false`. `DeterminismPolicy` gained a genuine
   runtime observation field, `pythonhashseed_env_value` (read via
   `os.environ.get`, never a claim the function itself makes true). Tests:
   `tests/unit/discovery/test_b2a_workers.py` (3 new tests: real-config
   read, fallback-on-unloadable-path, and `_launch_worker` env threading
   including confirming `PATH` is still inherited, not stripped).

## 3. Additional gate hardening (§30 of the task brief, partial)

`semantic_swap_parity` was added as its own named `MANDATORY_GATE_
CONDITIONS` entry (`kvcot.discovery.b2a_contract`), derived in
`kvcot.discovery.b2a_execute.run_b2a_calibration` from whether the R-KV
worker reported any `pair_failure_details` entry with stage
`semantic_swap_parity_failure` — distinct from the five Pass-2-level
trajectory/parity conditions (which are evaluated before any pair is ever
attempted) and from the coarser `all_required_pair_evaluations_completed`.
Test: `tests/unit/discovery/test_b2a_execute_coordinator.py
::test_gate_fails_on_a_reported_semantic_swap_parity_failure_even_with_
every_count_exact` proves this condition fails independently, with every
count-based condition still passing, so it is not merely riding on an
already-failing coarser check.

The REST of §30's requested gate conditions — `git_clean_verified`,
`rkv_submodule_match`, `single_rtx3090_verified`, `prompt_identity_
verified` as its own named field, `events_with_four_unique_pairs_exact`,
`unique_real_pair_count_exact`, `all_required_timings_present`, and
per-worker/per-expected generation-config match granularity beyond the
existing collapsed `generation_config_hash_match` — were NOT implemented
this pass. See §5.

## 4. Hostile self-audit (§32/§33 of the task brief)

A forbidden-pattern grep was run against every file this pass touched, and
separately against only the newly-ADDED lines (`git diff | grep '^+'`).
Two hits in newly-added code:

- `# never an always-empty placeholder.` — a comment describing the repaired
  defect (§2.3 above), not an actual placeholder.
- `except Exception:` in `_framework_seed_for_env` — a deliberately broad,
  explicitly-commented fallback for an AUXILIARY env-var value only; the
  worker subprocess's own config load moments later remains the
  authoritative, fail-loud check. Documented at the call site.

No other occurrence of `parity_check_passed=True`, `net_physical_bytes_
changed=0`, `device_map="auto"`, a bare `except:`, or `NotImplementedError`
was introduced. Every remaining pre-existing occurrence of `time.monotonic`
(unsynchronized timing, §5/§6 of the brief) and `device_map="auto"` (§8)
predates this pass and remains an open, documented gap — see §5.

Call-graph trace performed for every field this pass added or changed
(`selected_event_ids`, `pair_failure_details`, `semantic_swap_parity`,
`pythonhashseed_env_value`, `events_with_at_least_one_completed_real_
pair`): each traced from its raw source (a live observation inside
`run_example`/`build_swap_pair_record`/`_worker_subprocess_env`) through
its one deriving function to its one consuming gate condition or test
assertion, with a negative test proving it can be `False`/non-empty when
the underlying condition is genuinely not met (never a field that can only
ever read as passing).

## 5. What remains genuinely INCOMPLETE (stated plainly)

This pass does not claim to close the full 41-section brief it was given.
The following are real, confirmed-by-audit gaps this pass did NOT fix:

- **§5/§6 — no CUDA-synchronized timing.** `kvcot.discovery.b2a_workers`
  still uses bare `time.monotonic()` with no `torch.cuda.synchronize()`
  immediately before/after each measured region; model-load time is still
  excluded from `project_complete_pilot_gpu_hours`; the full 7-8-phase
  per-branch timing decomposition (clone/restore-only, swap-only,
  bridge-plus-scored-only, separately for baseline/swapped) does not exist
  — this was already an explicitly-acknowledged B1B-R4 deferral, still
  open.
- **§7 — VRAM has one reset point, no distinct model-load-phase peak, no
  pre-branch memory guard.** Both workers still reset peak stats once,
  after load/before inference; there is no separate load-phase checkpoint
  and no guard estimating the next branch's live-cache requirement before
  attempting it.
- **§8 — no discovery-only strict single-GPU load path.**
  `kvcot.generation.policies` still uses `device_map="auto"` in the exact
  path B2A/discovery would exercise on real hardware; no explicit
  single-device map, no GPU-count/hardware-identity preflight exists.
- **§9/§35 — no Hub snapshot resolver; no real-tokenizer validation run
  this pass.** `kvcot.discovery.runtime_evidence` still relies solely on
  `_commit_hash`; no `huggingface_hub` snapshot-path resolution/shard
  completeness check was built, and the real pinned-tokenizer network/cache
  validation this task's §35 requires was not executed this session.
- **§11 — batch size still derived from an independently-constructed
  tensor**, not the tensor actually observed inside the real prefill/decode
  call.
- **§12/§13 partial — row-identity and raw call/compaction evidence.**
  Existing coverage (dataset/manifest/prompt hashes, call-boundary traces,
  compaction-position lists) was not re-audited or extended this pass
  beyond what §2's repairs touched.
- **§19 — no NLL/cache-state hashes for the no-op control.** Exact
  bit-equality is already enforced (pre-existing, in
  `kvcot.discovery.schemas.SwapPairRecord._noop_invariants`), but no
  separate before/after hash fields were added.
- **§20 — schemas were already Pydantic** (contrary to the brief's
  assumption); this pass added one new nested dataclass field type
  (`PairFailureDetail`) but did not perform a full strict-schema audit of
  every remaining nested `dict[str, Any]` field (e.g.
  `pass1_call_boundary`/`pass2_call_boundary`).
- **§21 — the full 12-real+1-no-op `run_rkv_worker`-level CPU success path
  is still not exercised.** This was already an explicit, documented B1B-R4
  deferral (the fake model used in `test_b2a_workers_real_bodies.py` never
  naturally emits EOS); still open — would require a materially larger
  fake-model fixture.
- **§22-§26 — no immutable attempt-directory architecture.**
  `kvcot.discovery.worker_envelope.write_worker_envelope` is still
  non-atomic (direct `write_text`, no temp+rename); `kvcot.discovery
  .b2a_execute` still never reads back or hashes the envelope into the
  final artifact (the exact B1B-R4-documented gap); there is no
  `invocation.json`/`preflight.json`/per-role subdirectory/`final.json`
  layout — each worker still writes one loose result file plus one loose
  envelope sibling.
- **§27 — full Git/software/hardware/invocation provenance export** was
  not built as a dedicated new subsystem this pass.
- **§29 — no automated contract/schema/dry-run name-consistency test.**
- **§30 — remaining gate conditions**, listed in §3 above.
- **§31 — no memory-safe local-snapshot preflight** (shard completeness,
  disk space, SHA verification) was built.

None of the above was silently claimed complete. Each is a genuine,
bounded piece of remaining work, most on the same scale as one of the
seven repairs in §2 (i.e., each would reasonably be its own focused pass).

## 6. Test evidence

```
python -m compileall src tests -q          # clean, no output
python -m pytest --collect-only -q         # 964 tests collected, 0 errors
python -m pytest -m "not gpu" -q           # 950 passed, 14 deselected (gpu-marked)
git diff --check                           # no whitespace errors (CRLF-conversion notices only)
```

`kvcot prepare-b2a-manifest --config configs/discovery/llama8b_math500_b1024.yaml --dry-run`
and `kvcot b2a-calibrate --config configs/discovery/llama8b_math500_b1024.yaml --dry-run`
both still run to completion after every change in this pass: no model
loaded, no CUDA required, no result files created; the printed plan is
unchanged in content and matches B1B-R4's own dry-run output
(3 selected events, 12 real pair evaluations + 1 no-op, all frozen config
hashes). `b2a-calibrate --execute` was NOT run. No real-tokenizer network
validation (§35) was run this pass — see §5.

## 7. Git state

- Prior commit: `4d7971b7b09c004c4670bfde3939416ab550ea71` ("Complete
  B1B-R4 final B2A closure"), confirmed an ancestor of `HEAD` at session
  start, working tree clean before any edit.
- This pass adds one new forward commit on the SAME branch,
  `research/b1b-r4-final-b2a-closure` — no reset, rebase, revert, amend, or
  force-push. The branch remains unmerged.
