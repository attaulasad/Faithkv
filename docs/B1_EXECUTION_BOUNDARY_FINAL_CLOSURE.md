# B1 execution-boundary closure — focused completion pass on B1B-R4.1 (CPU-only, no GPU)

Run on branch `research/b1b-r4-final-b2a-closure`, on top of commit
`4e45beac1912a0a7852a034420732a10d0d703e7` ("Finish B1 final GPU boundary
closure", already pushed). This document does not undo, reset, or rewrite
that commit — it is a forward completion pass. That prior commit's own
closure document (`docs/B1B_R4_1_FINAL_CLOSURE.md`) explicitly recorded
itself as **INCOMPLETE** and itemized what remained; this pass closes five
of those confirmed items and states plainly, in §3 below, everything that
still remains.

**Status: B1 EXECUTION-BOUNDARY CLOSURE VERDICT: INCOMPLETE — B2A/GPU
REMAIN BLOCKED.** No GPU inference occurred. No model weights were
downloaded. No RTX 3090 timing or memory measurement exists. B2A remains
blocked pending independent CPU audit. B2B remains blocked. No FaithKV
method exists. The frozen row may still fail eligibility on real hardware
— CPU closure cannot guarantee the absence of CUDA-only defects.
`configs/lock.yaml` is untouched. `third_party/R-KV`'s pinned commit is
untouched.

## 1. Audit method

Before writing any code, an Explore-agent audit re-verified the actual
current state of the affected call graph (`torch.cuda.reset_peak_memory_
stats` usage, timing, memory phases, branch-memory lifetime, capture-tensor
retention, semantic-swap clone count, the worker-level CPU success test's
documented scope limit, the semantic-swap gate's derivation, pair-identity
accounting, no-op evidence fields, batch-size derivation, `device_map`
usage, Hub-snapshot resolution, and the attempt-directory/envelope
architecture) against the actual current code — 16 numbered claims, all 16
CONFIRMED true by direct inspection before any repair began.

## 2. Repairs (each with new CPU tests, full suite green after every one)

1. **`reset_patched_state` no longer resets CUDA peak-memory stats itself**
   (`kvcot.generation.state.reset_patched_state`). It previously called
   `torch.cuda.reset_peak_memory_stats()` as a bundled side effect —
   `kvcot.discovery.b2a_workers.run_rkv_worker` calls it once to build Pass
   1's initial state and again (via `pass2_initial_state_factory`, invoked
   internally by `run_example` AFTER Pass 1 has already run) to build Pass
   2's, so the second call silently wiped whatever peak Pass 1 had already
   accumulated — the final worker-level peak reading excluded Pass 1's
   memory entirely. Peak-memory reset is now owned exclusively by each
   caller's own measurement boundary: `kvcot.discovery.b2a_workers`'s
   existing single, correctly-scoped worker-level reset (before Pass 1,
   spanning through Pass 2 and branch evaluation) is now the sole reset for
   that path — no code change was needed there, since removing the hidden
   reset makes it automatically authoritative. The primary pipeline
   (`kvcot.cli.cmd_generate`, `kvcot.generation.replay.replay_and_snapshot`)
   gained an explicit, equivalent reset immediately adjacent to their own
   `reset_patched_state` call, preserving their exact prior behavior (never
   silently changed). Tests: `tests/unit/test_generation_state_reset.py`
   (a monkeypatched `torch.cuda` proving the function body itself never
   calls the reset, with `is_available` forced `True` so the proof does
   not depend on this machine lacking a GPU) and two new assertions in
   `tests/unit/discovery/test_b2a_workers_real_bodies.py` (a fake CUDA
   facade counting reset calls, proving the worker-level reset stays at
   exactly 1 even though `reset_patched_state` is called during state
   construction).

2. **Baseline and swapped branch snapshot clones are never live at the
   same time, extended to their full live evaluation result**
   (`kvcot.discovery.pipeline.build_swap_pair_record`,
   `kvcot.discovery.branch_eval.CompactBranchScore`/`compact_branch_score`).
   The prior pass (`docs/B1B_R4_1_FINAL_CLOSURE.md` §2 item 5) released the
   initial `ModelStateSnapshot` clone but not `evaluate_branch`'s returned
   `BranchEvalResult` — its `final_cache_state` (a real-model
   `_LiveBranchState` holding a COMPLETE live multi-layer cache, distinct
   from the snapshot that seeded it) and `per_token_logits` (48
   full-vocabulary tensors) stayed reachable as an un-deleted local
   variable through the swapped branch's entire construction, even though
   neither field is ever read downstream (`SwapPairRecord` only consumes
   per-token NLL and the derived swap gain). Each branch now extracts a new
   `CompactBranchScore` (per-token NLL, mean, a canonical-JSON hash) the
   instant its evaluation finishes and `del`s the full result before
   proceeding. Test:
   `tests/unit/discovery/test_b1b_integration.py
   ::test_baseline_branch_eval_result_is_released_before_swapped_snapshot_is_cloned`
   wraps `pipeline.evaluate_branch` to capture a `weakref` to the exact
   `final_cache_state` object baseline evaluation returns, using a custom
   step function that never hands back the same object (mirroring the real
   adapter's restore-once branch stepping), and confirms it is
   garbage-collected before the swapped branch is even cloned — not merely
   released by the time the pair finishes.

3. **No redundant full-cache clone during the semantic swap**
   (`kvcot.discovery.swap.apply_within_head_swap_owned`,
   `apply_semantic_within_head_swap(..., owned=True)`). Validation logic
   was factored into a shared `_validate_swap_inputs` helper (identical
   checks, never duplicated) reused by both the original cloning primitive
   (kept, unchanged default, for callers that do not already own an
   independent clone) and a new owned-mutation variant that writes directly
   into caller-owned tensors — no second clone stacked on top of the
   snapshot clone `kvcot.discovery.pipeline.build_swap_pair_record` already
   makes. `apply_semantic_within_head_swap` gained an `owned: bool = False`
   parameter (default preserves every pre-existing caller's exact
   behavior); the pipeline's own call site passes `owned=True`, since it
   already owns an independent `swapped_snapshot` clone at that point.
   Tests: `tests/unit/discovery/test_swap.py` (6 new tests: in-place
   mutation returns the SAME objects with unchanged storage pointers, every
   validation still enforced, no-op detection matches the cloning variant,
   and `apply_semantic_within_head_swap`'s `owned=True`/`owned=False`
   dispatch is proven via a monkeypatch that makes the cloning primitive
   raise if called when it shouldn't be).

4. **Positive semantic-swap-check evidence, not absence-of-failure**
   (`kvcot.discovery.pipeline.PairBuildResult.semantic_swap_check_attempted`/
   `.semantic_swap_check_passed`, `kvcot.discovery.b2a_evidence
   .SemanticSwapCheckEvidence`/`derive_semantic_swap_check_evidence`). The
   prior pass's `semantic_swap_parity` gate condition (added in
   `docs/B1B_R4_1_FINAL_CLOSURE.md` §3) was derived as `not any(pair
   failure with this stage)` — vacuously `True` for a worker that never
   actually reached the semantic-swap check for any pair at all (e.g.
   every real pair failed earlier, at candidate/donor pool lookup). The
   parity/byte-delta derivation in `build_swap_pair_record` (which depends
   only on `pristine`/`ev`/`record`/`semantic_swap`, never on branch
   evaluation) was moved to run immediately after the swap succeeds, so its
   result is known — and reportable as positive evidence — at every
   subsequent return point, success or failure. `ExampleResult`/
   `RKVWorkerResult` gained `semantic_swap_checks_required` (frozen at 12)/
   `_attempted`/`_passed`/`_failed`, summed across every REAL pair attempt;
   the gate now requires `attempted == passed == required` AND `failed ==
   0`. Tests: 3 new tests in `test_b1b_integration.py` (attempted+passed on
   the happy path, attempted-but-not-passed on a provenance-update
   failure, not-attempted when the pool lookup fails before the swap is
   ever called), 3 in `test_b2a_evidence.py`, and 2 coordinator-level tests
   in `test_b2a_execute_coordinator.py` (the original swap-failure
   scenario, now asserting the POSITIVE counts, plus a new test proving a
   worker with zero attempted checks and an EMPTY `pair_failure_details`
   — the case the old derivation would have vacuously passed — now fails
   the gate).

5. **Exact, duplicate-detecting pair-identity accounting, not a bare count**
   (`kvcot.discovery.b2a_evidence.PairIdentityEvidence`/
   `derive_pair_identity_evidence`). `events_with_all_four_real_pairs_
   completed` (and the underlying `count >= 4` per event) could not tell
   four genuinely distinct `(compaction_event_id, layer_index,
   kv_head_index, evicted_absolute_position, donor_absolute_position)`
   identities apart from the same identity recorded more than once.
   Derived entirely from `example_result.pair_records` (no new per-pair
   state threading needed) into three new gate conditions:
   `unique_real_pair_count_exact` (12 distinct identities),
   `events_with_four_unique_pairs_exact` (exactly 3 events with exactly 4
   unique identities each), `no_duplicate_pair_identity` (real AND no-op,
   checked independently). Tests: 4 new tests in `test_b2a_evidence.py`
   (twelve distinct pairs passes; a duplicate recorded as a 4th "pair" is
   caught even though the bare count would read `4 >= 4`; a 5-pair event
   is not exactly 4; a duplicated no-op identity is detected independently
   of the real-pair check) and a coordinator-level test in
   `test_b2a_execute_coordinator.py` proving the gate fails on a reported
   duplicate while the bare-count condition (`real_pair_count_exact`) it
   augments stays untouched.

## 3. What remains genuinely INCOMPLETE (stated plainly)

This pass does not claim to close the full scope it was given. The
following, all confirmed by the same audit in §1, were NOT fixed this
pass:

- **No CUDA-synchronized timing abstraction.** `kvcot.discovery.b2a_workers`
  still uses bare `time.monotonic()` with no `torch.cuda.synchronize()`
  immediately before/after each measured region (prefill/decode/snapshot
  in `_RkvHarnessInstrumentation`, the FullKV natural-generation loop).
  Model-load time is still excluded from `project_complete_pilot_gpu_hours`.
  The full per-phase decomposition (worker init, tokenizer resolution,
  post-load verification, Pass-1/Pass-2 prefill vs. decode separately, and
  the 8-way per-pair breakdown: baseline clone/restore/bridge-plus-scored/
  release, swapped clone/mutation/restore/bridge-plus-scored) does not
  exist — still one coarse per-example and one coarse per-pair duration
  each, an already-acknowledged deferral from `docs/B1B_R4_FINAL_B2A_
  CLOSURE.md` that remains open.
- **VRAM still has one reset/read spanning the whole worker**, not
  distinct model-load-phase vs. inference-phase checkpoints, and no
  pre-branch memory guard (estimating the next branch's live-cache
  requirement from actual tensor shapes/dtypes before attempting it, and
  refusing before evaluation if the frozen 22 GiB boundary is already
  unreachable) exists anywhere.
- **`Pass2Result.target_captures` still holds the full `UpdateKvCaptureRecord`
  (complete pre-call/returned K/V tensors, full-layer score tensors)
  through all 12+1 pair evaluations.** `MinimizedTargetEvidence`
  (`kvcot.discovery.capture_minimize`) is built and bound-asserted for
  EVIDENCE/reporting purposes only — `orchestrator.py`'s pair loop still
  passes the FULL `target_capture` into `build_swap_pair_record`. A
  `CompactBranchTarget` conversion (built immediately after Pass-2 target
  validation, retaining only the selected candidate/donor row vectors and
  scalar scores the four real pairs and one no-op actually need, with the
  full capture record released before the pair loop starts) was not built.
  This is the largest of the confirmed-but-unfixed items — comparable in
  scope to everything fixed in §2 combined — and the most consequential
  for real GPU memory, since it is the ONE piece of state that stays alive
  across all 13 pair evaluations, not just two.
- **The full 12-real+1-no-op `run_rkv_worker`-level CPU success test still
  does not exist** (`tests/unit/discovery/test_b2a_workers_real_bodies.py`
  is still explicitly documented as stopping at `cap_hit`) — an
  already-acknowledged deferral, still open; would need a materially
  larger fake-model fixture that emits a controlled, delayed EOS after
  enough compaction events.
- **No positive no-op raw-hash evidence** (baseline/no-op NLL array
  hashes, mean-NLL difference, max per-token difference, cache-state
  before/after hashes) beyond the pre-existing exact-equality invariant in
  `kvcot.discovery.schemas.SwapPairRecord._noop_invariants`.
- **Batch size is still derived from an independently-constructed tensor**
  (`torch.tensor([manifest.prompt_token_ids])`), not the tensor actually
  observed inside the real prefill/decode call.
- **No raw Pass-1/Pass-2 token/call/compaction evidence export** beyond
  what already existed (call-boundary traces, `compaction_position_
  equality`) — first-mismatch indices, complete compaction-position lists
  as their own exported field, and per-call shape evidence were not added.
- **`device_map="auto"` is still used in the exact path B2A/discovery
  exercises** (`kvcot.generation.policies.FullKVPolicy.load`/
  `_PatchedPolicyBase.load`) — no discovery-only strict single-device load
  path or GPU-identity preflight (device count, RTX 3090 name match,
  compute capability, driver/CUDA/cuDNN versions) exists.
- **No Hub snapshot resolver.** `kvcot.discovery.runtime_evidence` still
  relies solely on `_commit_hash` reads off already-loaded objects; no
  `huggingface_hub`-based local snapshot-path resolution, shard-completeness
  verification, or `local_files_only=True` load path exists. No
  tokenizer-only real network/cache validation was run this pass.
- **No expected-vs-observed runtime-generation record** beyond the
  existing collapsed `generation_config_hash_match` (worker-vs-worker and
  worker-vs-manifest, from the B1B-R4 adversarial self-review) — a
  three-way `fullkv_generation_matches_expected`/`rkv_generation_matches_
  expected`/`workers_generation_match` split against one canonical
  expected record was not built.
- **No coordinator-side row-identity refetch/reconstruction** beyond the
  existing dataset/manifest/prompt-hash matching already in place.
- **No immutable attempt-directory architecture.**
  `kvcot.discovery.worker_envelope.write_worker_envelope` is still
  non-atomic; `kvcot.discovery.b2a_execute` still never reads back or
  hashes the envelope into the final artifact (the same gap
  `docs/B1B_R4_FINAL_B2A_CLOSURE.md` originally documented); there is no
  `invocation.json`/`preflight.json`/per-role subdirectory/`final.json`
  layout, no progress journal.
- **No expanded Git/software/hardware/invocation provenance export**
  beyond what already exists.
- **No contract-consistency test** verifying required-measurement names,
  schema fields, evidence-builder inputs, and dry-run output stay in sync.
- **No memory-safe local-snapshot preflight** (shard completeness, disk
  space, SHA verification before load).

None of the above is silently claimed complete, and none of the B2A
dry-run's printed plan claims any of it — the dry-run output is unchanged
from the prior pass specifically because none of the timing/memory/
attempt-directory/gate-list display work was done.

## 4. Hostile self-review (§24 of the task brief)

A forbidden-pattern grep was run against every file this pass touched,
restricted to only the newly-ADDED lines (`git diff | grep '^+'`). Every
hit was individually classified:

- Two `torch.cuda.reset_peak_memory_stats()` calls (in `kvcot.cli.
  cmd_generate` and `kvcot.generation.replay.replay_and_snapshot`) — both
  the INTENDED explicit replacements for the removed hidden reset (§2 item
  1), both guarded by `torch.cuda.is_available()`, both placed at the
  exact position the old implicit reset fired.
- Two mentions of `count >= 4` — both inside comments/docstrings
  describing the DEFECT this pass repairs (§2 item 5), never a literal
  occurrence of that comparison in executable code (the actual code now
  compares against `REAL_PAIR_EVALUATIONS_PER_EVENT` via exact identity
  sets).
- One prose mention of `torch.cuda.reset_peak_memory_stats()` inside
  `reset_patched_state`'s own docstring, explaining what was removed and
  why.
- `except Exception as exc:` in `pipeline.py`'s branch-evaluation failure
  handler — pre-existing from the prior pass, unchanged in kind, now
  wrapping slightly more code (the new `compact_branch_score` extraction)
  as a direct consequence of §2 item 2's `del` placement.

No new instance of a hard-coded parity/count literal, `device_map="auto"`,
a bare `except:`, or `NotImplementedError` was introduced. A full
line-by-line re-read of the reordered `build_swap_pair_record` (§2 items 2
and 4 both touch its control flow) found no leftover duplicate variable
definitions or dead code from the reordering.

## 5. Test evidence

```
python -m compileall src tests -q          # clean, no output
python -m pytest --collect-only -q         # 984 tests collected, 0 errors
python -m pytest -m "not gpu" -q           # 970 passed, 14 deselected (gpu-marked)
git diff --check                           # no whitespace errors (CRLF-conversion notices only)
```

`kvcot prepare-b2a-manifest --config configs/discovery/llama8b_math500_b1024.yaml --dry-run`
and `kvcot b2a-calibrate --config configs/discovery/llama8b_math500_b1024.yaml --dry-run`
both still run to completion after every change in this pass: no model
loaded, no CUDA required, no result files created; printed plan content is
byte-identical to the prior pass's (unchanged, since none of the §3 gaps
that would alter it were closed). `b2a-calibrate --execute` was NOT run.
No real-tokenizer network validation was run this pass.

## 6. Git state

- Prior commit: `4e45beac1912a0a7852a034420732a10d0d703e7` ("Finish B1
  final GPU boundary closure"), confirmed an ancestor of `HEAD` at session
  start, working tree clean before any edit.
- This pass adds one new forward commit on the SAME branch,
  `research/b1b-r4-final-b2a-closure` — no reset, rebase, revert, amend, or
  force-push. The branch remains unmerged.
