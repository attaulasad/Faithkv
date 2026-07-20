# Changelog

Frozen settings (`configs/lock.yaml`, and Sections 1/4/8/9 mirrored into
`CLAUDE.md`) may only change via a dated entry here, added **before** the
run that depends on the change (per the build brief). Entries are ordered
newest first.

## 2026-07-20 — Phase B1 execution-boundary closure: focused completion pass on B1B-R4.1 (no GPU used, no model inference, no model weights downloaded, no Vast.ai activity; `third_party/R-KV` pinned commit unchanged; `configs/lock.yaml` unchanged; prior commit `4e45beac...` not reset/rebased/amended)

Run on branch `research/b1b-r4-final-b2a-closure`, a forward completion
commit on top of `4e45beac1912a0a7852a034420732a10d0d703e7` ("Finish B1
final GPU boundary closure", already pushed and self-recorded INCOMPLETE).
Full detail: `docs/B1_EXECUTION_BOUNDARY_FINAL_CLOSURE.md`.

**Authorization.** No new `CLAUDE.md` exception — stays inside the
CPU-side harness architecture already authorized by §1b/§4b. No model
weights, no CUDA, no Vast.ai activity of any kind.

An evidence-based audit (16 numbered claims, all 16 confirmed against the
actual current code before any repair began) found and this pass repaired
**five confirmed defects**, each with new CPU tests: (1)
`kvcot.generation.state.reset_patched_state` no longer calls
`torch.cuda.reset_peak_memory_stats()` itself — the R-KV worker's Pass-2
state construction was silently wiping Pass 1's already-accumulated peak
partway through measurement; peak-memory reset is now owned exclusively by
each caller's own measurement boundary, with the primary pipeline
(`kvcot.cli.cmd_generate`, `kvcot.generation.replay.replay_and_snapshot`)
gaining an explicit, behavior-preserving replacement reset. (2) Baseline
and swapped branch evaluation are now released down to a compact
`kvcot.discovery.branch_eval.CompactBranchScore` (per-token NLL, mean,
hash) the instant each branch's evaluation finishes — the prior pass
released the snapshot clone but left `evaluate_branch`'s full returned
result (including a real-model `_LiveBranchState`'s complete live cache)
reachable through the swapped branch's entire construction; proven via a
`weakref` to the exact live-cache object. (3) The semantic swap on an
already-owned snapshot clone no longer clones the cache a second time
(`kvcot.discovery.swap.apply_within_head_swap_owned`, dispatched via a new
`owned=True` parameter defaulting to the prior cloning behavior for every
other caller). (4) `semantic_swap_parity`'s gate derivation changed from
absence-of-a-failure-record (vacuously true for a worker that never
reached the check for any pair at all) to positive
`checks_attempted == checks_passed == checks_required(12)` counts,
threaded through `PairBuildResult`/`ExampleResult`/`RKVWorkerResult`. (5)
Three new gate conditions (`unique_real_pair_count_exact`,
`events_with_four_unique_pairs_exact`, `no_duplicate_pair_identity`)
replace a bare per-event count (`count >= 4`) that could not distinguish
four genuinely distinct `(event, layer, head, candidate, donor)` identities
from the same identity recorded more than once.

Hostile self-audit (`docs/B1_EXECUTION_BOUNDARY_FINAL_CLOSURE.md` §4): a
forbidden-pattern grep over only the newly-ADDED lines found every hit
already accounted for (the two intentional replacement CUDA resets, two
`count >= 4` mentions inside comments describing the repaired defect, and
one pre-existing broad `except` unchanged in kind). A full re-read of the
reordered `build_swap_pair_record` found no leftover duplicate code from
the reordering.

`python -m pytest -m "not gpu" -q`: 970 passed, 0 failed, 14 deselected
(gpu-marked); 984 tests collected total. Both `kvcot prepare-b2a-manifest
--dry-run` and `kvcot b2a-calibrate --dry-run` still run to completion
unchanged (no model loaded, no CUDA required; printed plan is
byte-identical to the prior pass's, since none of the still-open gaps that
would alter it were closed this pass).

**This pass explicitly did NOT close everything it was given** —
`docs/B1_EXECUTION_BOUNDARY_FINAL_CLOSURE.md` §3 lists every remaining gap
plainly, largest first: `Pass2Result.target_captures` still retains the
FULL capture record (complete pre-call/returned K/V tensors) through all
12+1 pair evaluations (`MinimizedTargetEvidence` is evidence-only, never
substituted into the actual pair-building loop) — the single largest
remaining memory-safety gap, comparable in scope to everything fixed this
pass combined; no CUDA-synchronized timing abstraction and no
model-load-inclusive runtime projection; VRAM still has one reset spanning
the whole worker, no load-vs-inference phase split, no pre-branch memory
guard; the full 12-real+1-no-op `run_rkv_worker`-level CPU success test
still does not exist (an already-acknowledged deferral, still open); no
Hub snapshot resolver and no real-tokenizer network validation was run;
batch size still derived from an independently-constructed tensor;
`device_map="auto"` still used in the exercised path, no strict
single-GPU load path; no immutable attempt-directory architecture, worker
envelope still non-atomic and still never cross-referenced into the final
artifact; no expanded provenance export; no contract-consistency test.
**Status: B1 EXECUTION-BOUNDARY CLOSURE VERDICT: INCOMPLETE — B2A/GPU
REMAIN BLOCKED.** No discovery result exists. No method exists. GPU, B2A,
and B2B remain fully unauthorized by this entry.

## 2026-07-20 — Phase B1B-R4.1: focused completion amendment on B1B-R4 (no GPU used, no model inference, no model weights downloaded, no Vast.ai activity; `third_party/R-KV` pinned commit unchanged; `configs/lock.yaml` unchanged; prior commit `4d7971b7...` not reset/rebased/amended)

Run on branch `research/b1b-r4-final-b2a-closure`, a forward completion
commit on top of `4d7971b7b09c004c4670bfde3939416ab550ea71` ("Complete
B1B-R4 final B2A closure", already pushed). Full detail:
`docs/B1B_R4_1_FINAL_CLOSURE.md`.

**Authorization.** No new `CLAUDE.md` exception — stays inside the
CPU-side harness architecture already authorized by §1b/§4b. No model
weights, no CUDA, no Vast.ai activity of any kind.

An evidence-based audit of the ACTUAL current code (not the assumed defect
list of the originating task brief — several assumed defects, e.g.
non-Pydantic worker-result schemas, turned out already fixed) found and
this pass repaired **seven confirmed defects**, each with a new CPU test:
(1) `kvcot.discovery.pass2.run_pass2_capture` no longer maintains its own
mutable `LayerProvenance` shadow track for the real-model path —
`kvcot.discovery.real_model_adapter.RealModelState` is now the sole
authoritative provenance owner, exposing a `projected_pre_event_position_
map` derived via a disposable `LayerProvenance.clone()` (never
re-implementing position arithmetic) from pending-position registration
(`register_pending_fed_positions`/`clear_pending`, cleared on any forward
exception before a partial commit); the CPU synthetic-harness path is
unchanged. This pass's own hostile review of its new integration test
found (and documented, not silently papered over) a genuine pre-existing
boundary-condition ambiguity in the ported R-KV bookkeeping formula
(`evicted_token_num` staying 0 across a reorder-without-shrink eviction),
orthogonal to the repair. (2) `selected_compaction_events` (feeding the
`selected_event_count_exact` gate) is now derived from the frozen Pass-1
plan (`ExampleResult.selected_event_ids`, populated once, right after
`build_pass1_plan` succeeds) instead of counting distinct event IDs across
surviving pair records — the prior derivation silently under-counted
whenever every pair for a selected event failed attrition; a new
`events_with_at_least_one_completed_real_pair` field keeps the weaker,
completion-based quantity separately named. (3) `pair_failure_details` was
always an empty tuple in the production R-KV worker path (a parameter
existed but was never actually threaded through) — `kvcot.discovery
.orchestrator.run_example` now builds one structured
`kvcot.discovery.attrition.PairFailureDetail` (event/layer/head/candidate/
donor/kind/stage/detail/elapsed-time) per failed pair, live. (4)
`kvcot.discovery.capture_minimize.assert_minimized_bound` had zero call
sites outside its own test file — `run_example` now calls it on every
target it builds. (5) Baseline and swapped branch snapshot clones are now
released sequentially (baseline cloned, evaluated, explicitly `del`eted in
a `finally` — covering the exception path — BEFORE the swapped clone is
even created) instead of both being held live for the whole pair-build
call; proven via a `weakref` to the actual clone object. (6)
`kvcot.discovery.pipeline.build_swap_pair_record`'s `parity_check_passed`/
`net_physical_bytes_changed` were literal `True`/`0` despite
`apply_semantic_within_head_swap` already returning a real, unused
`SemanticSwapResult` — a direct instance of this project's own forbidden-
pattern list; both are now derived from that report (provenance/kept-index
bookkeeping updates mandatory whenever present, gated independently since
the CPU synthetic harness's snapshots carry one without the other; byte
delta genuinely computed, not asserted). (7) `PYTHONHASHSEED` is now set on
the worker subprocess's environment BEFORE launch
(`kvcot.discovery.b2a_workers._worker_subprocess_env`/`_launch_worker`,
reading `framework_seed` off the same frozen config the worker itself
loads) — `random.seed()` inside the already-running worker process cannot
retroactively change that process's already-fixed hash seed.

**One gate-condition addition:** `semantic_swap_parity`, a new named
`MANDATORY_GATE_CONDITIONS` entry, derived from whether the R-KV worker
reported any `pair_failure_details` entry with the new
`STAGE_SEMANTIC_SWAP_PARITY_FAILURE` stage — proven (in
`test_b2a_execute_coordinator.py`) to fail the gate independently, with
every count-based condition still passing.

Hostile self-audit (`docs/B1B_R4_1_FINAL_CLOSURE.md` §4): a forbidden-
pattern grep over only the newly-ADDED lines found two hits, both
justified and documented at their call sites (a descriptive comment, and
one deliberately narrow `except Exception:` for an auxiliary env-var
fallback whose authoritative check still happens moments later inside the
worker itself). No new instance of a hard-coded parity/count literal,
`device_map="auto"`, or a bare `except:` was introduced.

`python -m pytest -m "not gpu" -q`: 950 passed, 0 failed, 14 deselected
(gpu-marked); 964 tests collected total. Both `kvcot prepare-b2a-manifest
--dry-run` and `kvcot b2a-calibrate --dry-run` still run to completion
unchanged (no model loaded, no CUDA required).

**This pass explicitly did NOT close the full 41-section task brief it was
given** — `docs/B1B_R4_1_FINAL_CLOSURE.md` §5 lists every remaining gap
plainly: no CUDA-synchronized timing or model-load-inclusive projection
(§5/§6 of that brief), no distinct model-load-phase VRAM checkpoint or
pre-branch memory guard (§7), no discovery-only strict single-GPU load path
(§8, `device_map="auto"` still in the exercised path), no Hub snapshot
identity resolver and no real-tokenizer network validation was run this
pass (§9/§35), batch size still derived from an independently-constructed
tensor rather than the actually-observed call (§11), no NLL/cache-state
hashes for the no-op control (§19), the full 12-real+1-no-op
`run_rkv_worker`-level CPU success path is still not exercised (§21, an
already-existing B1B-R4 deferral), no immutable attempt-directory
architecture and the worker envelope is still non-atomic and still never
cross-referenced into the final artifact (§22-§26), and most of the
remaining requested gate conditions (§30: `git_clean_verified`,
`rkv_submodule_match`, `single_rtx3090_verified`, and others) were not
added. **Status: B1 FINAL CLOSURE VERDICT: INCOMPLETE — B2A/GPU REMAIN
BLOCKED.** No discovery result exists. No method exists. GPU, B2A, and B2B
remain fully unauthorized by this entry.

## 2026-07-20 — Phase B1B-R4: final executable, measurement, and worker-evidence closure (no GPU used, no model inference, no model weights downloaded, no Vast.ai activity; `third_party/R-KV` pinned commit unchanged; `configs/lock.yaml` unchanged; PR #19 merge not undone)

Run on branch `research/b1b-r4-final-b2a-closure`, cut from `main` at
commit `fa117046bea2a2c492e17cd91276b2e3c6d59f7f` (the merged PR #19,
B1B-R3). Full detail: `docs/B1B_R4_FINAL_B2A_CLOSURE.md`.

**Authorization.** No new `CLAUDE.md` exception — stays inside the
CPU-side harness architecture already authorized by §1b/§4b. No model
weights, no CUDA, no Vast.ai activity of any kind.

**Seventeen repairs** (independent audit of merged PR #19), plus three
further defects found and fixed by this pass's own adversarial
self-review: (1) execution-count vocabulary frozen ("pair evaluations",
never "branches"); (2) FullKV now uses exact greedy generation
(`kvcot.discovery.pass1.run_natural_pass1` + the real adapter), replacing a
sampling call given `temperature=0.0`/`generator=None`; (3) framework
determinism applied and recorded independently in both worker processes;
(4) `NoOpMode` now actually controls pair construction
(`kvcot.discovery.orchestrator.PairExecutionPolicy`) instead of only
documenting an intended interpretation; (5) five trajectory/parity
conditions (`token_identical_replay`, `prefill_decode_boundary_parity`,
`compaction_position_equality`, `capture_gather_parity`,
`absolute_position_parity`) are now derived independently
(`kvcot.discovery.b2a_evidence`, `kvcot.discovery.call_trace`) instead of
all five copied from one `example_valid` boolean; (6) resolved-vs-requested
model/tokenizer revision read back via `transformers`' own `_commit_hash`
attributes (`kvcot.discovery.runtime_evidence`); (7) batch size, parameter
placement, and one-example scope derived from real observations, never
hard-coded; (8) per-real-pair-evaluation timing measured individually via
an injectable clock (`kvcot.discovery.orchestrator.run_example`'s
`clock_fn`), never an aggregate bucket multiplied by 144; (9)
branch-restored `CompactionTracker` state reconstructed from the snapshot
(`restore_compaction_tracker_from_snapshot`) instead of reset to empty;
(10) VRAM gate uses `max(peak_allocated, peak_reserved)` across both
workers; (11) the weight-cache safety guard scoped to
`prepare-b2a-manifest`'s own call site only (before/after snapshot diff),
no longer rejecting generic prompt verification on a host with
pre-existing model weights; (12) partial FullKV evidence preserved when
R-KV fails after FullKV succeeds; (13) every worker subprocess launch uses
`capture_output=True, timeout=7200, check=False`, and every attempt writes
a durable `WorkerEnvelope`; (14) artifact names include microseconds and a
random UUID4 suffix; (15) selected-capture evidence minimized to a bounded
per-target record (`kvcot.discovery.capture_minimize`), finally closing the
B1B-R3 Defect-11 deferral; (16) one canonical FullKV/R-KV worker API
(`kvcot.discovery.b2a_workers.run_fullkv_worker`/`run_rkv_worker`),
removing the B1B-R3 split with a `NotImplementedError` stub; (17) both
worker-control bodies are now exercised by CPU tests against injected fake
backends, never a preconstructed result. Adversarial self-review (§5,
`docs/B1B_R4_FINAL_B2A_CLOSURE.md`) additionally found and fixed: the
`no_offload_verified` gate condition using a weaker check than available;
the four dataset/manifest identity conditions checking only worker-vs-
worker agreement, never worker-vs-manifest; and Pass 2 snapshot/capture
time being measured but never folded into `wall_seconds_pass2`,
under-counting the projection. `python -m pytest -m "not gpu" -q`: 931
passed, 14 deselected, 0 failed. **Status: B1B-R4 implemented,
ready for independent CPU audit. GPU, B2A, and B2B remain blocked.**

## 2026-07-20 — Phase B1B-R3: executable B2A boundary and evidence producer (no GPU used, no model inference, no model weights downloaded; one pinned MATH-500 row and the pinned tokenizer's config-only files downloaded via `kvcot prepare-b2a-manifest --execute`; `third_party/R-KV` pinned commit unchanged; `configs/lock.yaml` unchanged; PR #18 merge not undone)

Run on branch `research/b1b-r3-executable-closure`, cut from `main` at
commit `7034e46b516eff656b5508d9253ee02b13405f95` (the merged PR #18,
B1B-R2). Full detail: `docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md`.

**Authorization.** No new `CLAUDE.md` exception — stays inside the
CPU-side harness architecture already authorized by §1b/§4b, plus the
already-permitted "pinned tokenizer/config files" and "one pinned MATH-500
row" CPU-only downloads. No model weights, no CUDA, no Vast.ai activity of
any kind.

**Twelve audit-defect repairs** (independent audit of merged PR #18):
(1) `RKVPolicy` was being constructed with a config object as its
positional `budget` argument — fixed to explicit keywords, and the
previously-unwired `kernel_size` field is now threaded through and
runtime-verified (`kvcot.discovery.runtime_rkv_verification`); (2) the
one-example manifest's prompt identity is now genuinely resolved
(`kvcot prepare-b2a-manifest`, new CPU-only command) — this also caught
and corrected a non-reproducible `raw_content_hash` left over from B1B-R2;
(3) a real MATH-500 answer verifier
(`kvcot.discovery.math500_verification`) replaces a stub that always
returned `"unverifiable"`, unblocking Pass 2; (4) `RealModelState`'s
provenance was silently never populated between forward calls (missing
position-append step) — unified into one shared `advance_after_forward`
helper (`kvcot.discovery.real_model_adapter`), used by Pass 1, Pass 2, and
branch continuation alike; (5) branch evaluation now restores a snapshot
exactly once per branch (`build_real_branch_step_fn_restore_once`) instead
of once per scored token; (6) swap bookkeeping
(`apply_semantic_within_head_swap`, `kvcot.discovery.swap`) now keeps
provenance and R-KV kept-index history consistent with the K/V content it
swaps; (7) every B2A gate-evidence field is now derived from an actual
observation (`kvcot.discovery.b2a_evidence`) instead of a hard-coded
literal; (8) FullKV and R-KV now run in separate OS subprocesses
(`kvcot.discovery.b2a_workers`, `kvcot.discovery.b2a_worker_entry`),
coordinated by `kvcot.discovery.b2a_execute.run_b2a_calibration`; (9)
no-op accounting now has an explicit `NoOpMode` policy separating the
CPU-mandatory per-event no-op from B2A's single numerical calibration;
(10) every B2A attempt (pass, fail, or exception) now writes an immutable
artifact (`kvcot.discovery.b2a_artifact`); (11) selected-capture size
tightening beyond B1B-R2's existing target-count bound was assessed and
deliberately deferred — documented, not silently claimed complete; (12)
frozen-vs-runtime R-KV configuration drift is covered by (1)'s runtime
verification.

Also new: a CPU GitHub Actions workflow
(`.github/workflows/cpu-tests.yml`) and a frozen-framework-seed
application/recording module (`kvcot.discovery.framework_seed`).

**Validation performed (CPU-only):** `python -m compileall src tests`;
`pytest --collect-only`; `pytest -m "not gpu" -q` (843 passed, 14
deselected); `kvcot prepare-b2a-manifest --dry-run` and `--execute`
(network available, real resolution performed); `kvcot b2a-calibrate
--dry-run` (exit 0, no blockers). No CUDA test was executed; no Llama-8B
weights were downloaded; B2A/B2B were not run.

## 2026-07-20 — Phase B1B-R2: real-model boundary and B2A preflight (no GPU used, no model inference, no model weights downloaded; MATH-500 dataset revision independently verified against the live Hugging Face Hub API — no weights or rows downloaded; `third_party/R-KV` pinned commit unchanged; `configs/lock.yaml` unchanged; PR #17 merge not undone)

Run on branch `research/b1b-r2-gpu-boundary-repair`, cut from `main` at
commit `eadee9a183024a51f3030117c2faae928d6ee162` (the merged B1B-R1 PR
#17). Full detail:
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md`.

**Authorization.** No new `CLAUDE.md` exception — this pass repairs and
completes the CPU-side B1B harness architecture already authorized by
§1b/§4b; no line item requires or claims broader scope.

**Eight B1B-R1 review-defect repairs:** (1) absolute-position
device/dtype normalization in `kvcot.discovery.capture
._recomputed_kept_physical_indices` (provenance-map device/dtype is now
authoritative; a CUDA-marked mechanical test proves CUDA-topk/CPU-provenance
gather correctness, skipped cleanly on this CPU-only build); (2)
`capture_update_kv` gained an opt-in `should_capture` predicate so Pass 2
captures ONLY the 3 preselected (position, layer) targets — every other
call passes through with zero clone/storage, bounding retained state by
target count, not call volume; (3) `kvcot.generation.state
.ModelStateSnapshot` gained `.clone()` and Pass 2/`kvcot.discovery.pipeline`
now branch from a complete, independently-cloned post-event snapshot
(every layer's K/V and bookkeeping), never one layer's returned K/V tensors;
(4) Pass 1/Pass 2 now take explicit `PrefillFn`/`DecodeOneFn` adapters
(`kvcot.discovery.harness_types`) — exactly one opaque prefill call per
pass, then one call per continuation token — and `eligible_event_ids`
excludes prefill-phase compaction events (no valid mid-prefill snapshot
boundary exists); (5) `kvcot.discovery.discovery_config` now freezes and
hashes the complete generation/R-KV/prompt-template configuration
(`DiscoveryGenerationLock`, expanded `DiscoveryRkvLock`,
`canonical_config_hash`); (6) the MATH-500 dataset revision
(`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, verified directly against the
HF Hub API) and a one-example manifest
(`configs/discovery/b2a_one_example_manifest.json`,
`kvcot.discovery.manifest`) are now frozen — the tokenized-prompt hash is
honestly left unresolved (requires a live tokenizer, out of scope); (7)
branch-count accounting (`kvcot.discovery.constants`) is now a single
source of truth read by both `plan-discovery` and the new `b2a-calibrate`:
`12 x 3 x 4 = 144` real branches, no-op excluded, matching what
`plan-discovery` already printed correctly; (8) `B2AGateResult`
(`kvcot.discovery.b2a_contract`) now has 21 mandatory, non-optional fields
(19 from the task brief plus the 2 pre-existing measurement-based
conditions) and cannot be hand-constructed as passing with any field
`False` (`__post_init__` re-derives `passed`/`failed_conditions`).

**New:** `kvcot b2a-calibrate` (`--dry-run`, fully exercised; `--execute`,
code path implemented via `kvcot.discovery.b2a_execute`/
`kvcot.discovery.real_model_adapter` but never invoked — every precondition
fails closed on this build). Help text: "B2A is a one-example engineering
calibration. It does not authorize the 12-example pilot."

## 2026-07-20 — Phase B1B-R1: B1A defect repairs and B1B CPU harness architecture integration (no GPU used, no model inference, no model weights or datasets downloaded, no manifest/result directory created; `third_party/R-KV` pinned commit unchanged; `configs/lock.yaml` unchanged; PR #16 merge not undone)

Run on branch `research/b1b-cpu-harness-and-b1a-repairs`, cut from
`research/b1a-cpu-prerequisites-r2-2` at commit
`887cd0fe89486e44db973fdd1f1133d75244fb24`. Full detail:
`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`.

**Authorization.** New, dated `CLAUDE.md` §1b/§4b exception (mirroring the
§1a/§4a pattern already used for B1A) narrowly authorizes CPU-side B1B
**harness architecture** — Pass 1/Pass 2 orchestration, branch
construction/evaluation, attrition accounting, `plan-discovery --dry-run`,
and a documentation-only future B2A contract — built and exercised
exclusively against dependency-injected synthetic/deterministic components
in CPU tests. Grants no model inference or GPU use; does not authorize B2A/
B2B execution or any Vast.ai activity, each still requiring its own
separate, future, dated authorization; implements no method.

**Six B1A blocker repairs** (found during independent review of PR #16):
(1) the no-offload assertion is now called unconditionally in both
`FullKVPolicy.load` and `_PatchedPolicyBase.load`, never gated behind
`model.device.type == "cuda"`, and now also inspects `hf_device_map` for
`cpu`/`disk`/`meta` entries and rejects a zero-parameter model; (2)
absolute survivor parity in `kvcot.discovery.capture` now runs at EVERY
compaction event (not just the first) via a caller-supplied pre-event
absolute-position-map thunk, compared with exact shape equality and
`torch.equal`, never set equality; (3) the active discovery schema
(`kvcot.discovery.schemas.SwapPairRecord`) now closes every internal-
consistency gap named in review — missing-reason fields for each
uncertainty source, a parity-consistency biconditional, canonical
score-margin/swap-gain derived-value validation
(`kvcot.discovery.nll.mean_nll`, one tolerance, one helper, reused by both
producer and validator), and a fully-strengthened no-op control; (4)
`kvcot.discovery.swap.apply_within_head_swap` now rejects dtype/device
mismatches before any write and detects storage overlap via
`untyped_storage().data_ptr()` identity (catching offset views a
starting-address-only check would miss); (5)
`kvcot.discovery.uncertainty`'s two scalar functions now require
`raw_logits.ndim == 1`, raising rather than silently flattening/reducing a
malformed-rank input; (6)
`configs/discovery/llama8b_math500_b1024.yaml` freezes the Llama-8B
revision resolved via the HF metadata API
(`6a6f4aa4197940add57724a7707d069478df56b1`, verified against the value
this pass was required to freeze exactly), with the dataset revision
deliberately left unfrozen and machine-checkably so
(`kvcot.discovery.discovery_config`).

**B1B CPU harness architecture integration:** `kvcot.discovery.pass1`
(natural-run bookkeeping, outcome-blind eligibility/selection built
entirely on the already-tested `kvcot.discovery.sampling` draws),
`kvcot.discovery.pass2` (token-identical replay, targeted capture,
cross-pass survivor-identity checking), `kvcot.discovery.pipeline` (branch
construction/evaluation and `SwapPairRecord` assembly),
`kvcot.discovery.attrition` (denominator-consistency-checked funnel
accounting), `kvcot.discovery.orchestrator` (end-to-end wiring),
`kvcot.discovery.b2a_contract` (future one-example B2A contract,
documentation/validation only), and `kvcot plan-discovery --dry-run`. 10
synthetic CPU integration test scenarios (`tests/unit/discovery/
test_b1b_integration.py`) exercise the complete injected orchestration
end to end, including a real multi-event non-identity absolute-position
map, deliberately-diverged replay step functions proving trajectory/
survivor mismatches are actually detected, the mandatory no-op control,
and byte-identical repeated-run determinism. No discovery hypothesis
result exists — none is claimed. B2A, B2B, and any Vast.ai activity remain
unauthorized.

## 2026-07-19 — Phase B0.5-R2.2: authority reconciliation and B1A CPU prerequisite implementation (no GPU used, no model inference, no model weights or datasets downloaded, no MATH-500 manifest/result directory created; `third_party/R-KV` pinned commit unchanged; `configs/lock.yaml` unchanged)

Run on branch `research/b1a-cpu-prerequisites-r2-2`, cut from
`research/b0-5-r2-dense-cache-repair` at commit
`ac3e7d545d13e4b663fe575430ba13e6e4b9bdc5` ("Finalize B0.5 timing sampling
and discovery controls"). Two things in one consolidated pass:

**Authority reconciliation.** `CLAUDE.md` §1's original "no 7B support"
blanket line technically contradicted the already-selected
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` discovery operating point
(`docs/b0_5_decision.json`'s `selected_operating_point`, chosen in Phase
B0.5). Resolved with two new, dated `CLAUDE.md` subsections — §1a and §4a —
that narrowly authorize CPU-side infrastructure only (architecture
dispatch, state-reset generalization, construction-parity tests, the
MATH-500 verifier) for `DeepSeek-R1-Distill-Llama-8B` + MATH-500, state
explicitly that this is not a method implementation and not a GPU/inference
authorization, and require a separate future authorization before B1B,
B2A, B2B, or any Vast.ai activity. `CLAUDE.md` §1/§4's original Qwen-1.5B
tables and `configs/lock.yaml` are unchanged.

**Four further defects found and repaired this session:**

- **Position-field naming error.** The `first_affected_logit_absolute_position`
  field (documentation-only until this pass — never actually coded in
  `src/`) conflated an *input* position (`t+1`) with the *logit-target*
  position it produces (`t+2`). The freshly-implemented active schema
  (`kvcot.discovery.schemas.SwapPairRecord`, `schema_version=
  "b0_5_r2_2.v1"`) uses two explicit fields instead
  (`first_affected_forward_input_absolute_position`,
  `first_affected_logit_target_absolute_position`), with a validator
  enforcing all four timing invariants at construction time.
- **Layer-depth/event-time confound.** The frozen layer-selection rule
  (`docs/B0_5_R2_1_FINAL_PROTOCOL.md` §5) assigned each selected event's
  depth stratum directly from its chronological draw ordinal — silently
  confounding *when* a compaction event happened with *how deep* its
  sampled layer was. `kvcot.discovery.sampling.assign_depth_strata` now
  independently permutes `{0, 1, 2}` via its own SHA-256-seeded
  `random.Random` stream (suffix `"b05r22_depth_permutation"`), storing
  `chronological_event_ordinal` and `depth_stratum` as two separate schema
  fields, never one ambiguous `event_ordinal`.
- **Undefined entropy/logit-margin signals.** Classified "mandatory" by
  §8.2 of the prior protocol document but never operationally defined.
  `kvcot.discovery.uncertainty` now freezes the exact computation (float32
  natural-log-nats Shannon entropy over `log_softmax`; top-1-minus-top-2
  raw-logit margin), sourced from the raw next-token-prediction logits at
  the moment each token was originally predicted during the natural run,
  per-candidate (never one shared value per eviction event), with source
  values and differences both always stored.
- **All previously-identified-but-unimplemented B1A prerequisites are now
  actually implemented and CPU-tested**, closing the gap
  `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §11 and
  `docs/b0_5_decision.json`'s `b0_5_r2_1_prerequisites_repaired_not_implemented`
  had left open: architecture-aware R-KV monkeypatch dispatch
  (`kvcot.discovery.dispatch`, verified against the pinned
  `third_party/R-KV/HuggingFace/rkv/monkeypatch.py`'s exact three exported
  patchers — `replace_qwen2`/`replace_llama`/`replace_qwen3` — wired into
  `kvcot.generation.policies._PatchedPolicyBase.load` in the required
  AutoConfig-before-AutoModelForCausalLM order), a no-offload hard
  assertion (`kvcot.discovery.no_offload`), a MATH-500 symbolic-equivalence
  verifier isolated per-comparison in a child OS process with a frozen
  5.0-second timeout (`kvcot.utils.math_verifier` +
  `kvcot.utils._math_verify_worker` — `math_verify`'s own
  `multiprocessing`-based timeout mechanism was found to raise a real
  `OSError: [WinError 6]` on this Windows host, so the parent enforces the
  timeout itself via `subprocess.run(..., timeout=5.0)` against a fresh
  child process instead), deterministic sampling utilities with
  golden-vector tests, a per-instance read-only capture-wrapper
  prerequisite around `R1KV.update_kv` with independent score
  recomputation and gather-parity checks (`kvcot.discovery.capture`), a
  fixed-shape within-head swap primitive (`kvcot.discovery.swap`), and a
  strengthened, complete-branch-output no-op control comparing full
  per-token logit/NLL sequences and final cache states, not just a single
  self-assignment tensor (`kvcot.discovery.branch_eval`).

New dependency: `math-verify[antlr4_13_2]==0.9.0` (plus its pure-Python
dependencies `latex2sympy2_extended==1.11.0`,
`antlr4-python3-runtime==4.13.2`), added to `pyproject.toml`'s `cpu-tools`
extra, `requirements.txt`, and `requirements-lock.txt`. No other dependency
pin changed — Torch/Transformers/FlashAttention are untouched.

Full detail, verdict, and per-blocker file/test mapping:
`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`. Superseded passages in
`docs/B0_5_R2_1_FINAL_PROTOCOL.md` are marked inline with a top-of-file
banner, not deleted; `docs/b0_5_decision.json` retains every prior field
and adds a `superseded_by_r2_2`/`b0_5_r2_2_*` block.

**Status: B0.5-R2.2 authority reconciliation complete; B1A CPU
prerequisites implemented and CPU-validated. B1B/B2A/B2B/GPU/Vast.ai
remain unauthorized.** No inference ran, no GPU was used, no model weights
or datasets were downloaded, and no discovery hypothesis result exists —
none is claimed by this entry.

## 2026-07-19 — Phase B0.5-R2.1: final timing, sampling and discovery-control correction (documentation-only; no method or harness implemented; no GPU used, no model inference, no model weights or datasets downloaded; no MATH-500 manifest/config/evaluator/result directory created; no code under `src/`, `tests/`, `configs/`, `scripts/`, `results/`, `schemas/`, or `third_party/` touched; no frozen §1/§4/§8/§9 value changed)

Run on branch `research/b0-5-r2-dense-cache-repair`, HEAD
`9d04ecd7268656894815fedb7d080f0d27c7fad3` (the B0.5-R2 commit) confirmed
at session start. Purpose: this is the final B0.5 protocol correction,
fixing an off-by-one timing defect and an under-specified sampling rule in
B0.5-R2 rather than trusting its already-committed READY verdict. Three
defects were found and repaired:

- **Off-by-one branch-timing error.** B0.5-R2 §14 scored the reference
  continuation starting immediately after the swap. But the forward call
  that consumes the event token `x_t` (the call *during which* compaction
  fires) already produces the logits predicting `x_{t+1}` as an ordinary
  side effect, **before** the swap is applied to that call's cache
  output — the swap cannot change those logits. Repaired: `x_{t+1}` (the
  real, already-generated next token) is fed identically into both the
  baseline and swapped branches as one unscored "bridge" token first; the
  48-token scored window starts one token later, at `x_{t+2}` — the first
  position whose logits are actually computed by reading from the
  diverged (baseline-vs-swapped) cache. Frozen constants:
  `bridge_tokens=1`, `scored_horizon=48`,
  `minimum_future_tokens_after_event=49`. Branch evaluation is now named
  precisely — teacher-forced NLL evaluation of fixed reference tokens,
  never "greedy decoding" (the prior, ambiguous phrasing).
- **Under-specified event/layer/head/candidate/donor sampling.** B0.5-R2
  §10's layer/head rule was an unrestricted `SHA256(...) %
  num_hidden_layers` hash, independently per event — three independent
  uniform draws do **not** actually guarantee one early/middle/late-third
  layer per example, despite being described that way. B0.5-R2 §9.2's
  candidate/donor selection used a plain ascending-position tie-break,
  systematically biased toward the lowest absolute positions in each pool.
  Repaired: one canonical SHA-256-seed helper (pipe-joined UTF-8 parts,
  first 8 digest bytes as a big-endian unsigned integer) feeds four
  independent `random.Random` draws — event selection (`rng.sample` of 3
  from the sorted eligible event list), layer selection restricted to a
  per-selected-event-ordinal depth third (`lo=floor(k*L/3)`,
  `hi=floor((k+1)*L/3)`, guaranteeing real coverage, not a probabilistic
  tendency), KV-head selection over the full range, and separate
  evicted-candidate/donor sampling with two independent seed streams.
- **Gate 10's pooled Spearman statistic was the wrong granularity.**
  B0.5-R2 §16(b) computed one Spearman correlation pooled across all
  examples' pairs — capable of showing a strong apparent association
  driven entirely by between-example variance with no real within-example
  effect, or the reverse. Repaired: Spearman rho is computed separately
  within each example, absolute value taken, and the **median** across
  examples is the decision statistic (`< 0.30`, strict), for each of eight
  named mandatory deployable signals (the six B0.5-R2 §16 controls plus
  entropy and logit margin, both classified mandatory per
  `docs/METHOD_PIVOT_SPEC.md` §5a's existing confound-control precedent).
  An explicit floor (`>= 8` evaluable examples per signal) and a mandatory
  no-op control (replacing a donor with its own captured K/V must produce
  exactly zero change in deterministic CPU tests) are added; the outcome
  set expands from a two-way pass/fail to three outcomes —
  DISCOVERY-SUPPORTING / NOT DISCOVERY-SUPPORTING / **NOT ADJUDICABLE** —
  so a data-thinness or mechanism-bug problem is never misreported as a
  negative finding.

Full correction: `docs/B0_5_R2_1_FINAL_PROTOCOL.md`. Superseded passages
in `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` (§10, §11, §14, §15, §16, §21) and
top-of-file banners in `docs/B0_5_PROTOCOL_REPAIR.md`,
`docs/B0_5_DISCOVERY_PROTOCOL.md`, and `docs/B0_5_FEASIBILITY_AUDIT.md`
are marked inline, not deleted; `docs/b0_5_decision.json` retains every
original field and adds `superseded_by_r2_1`/`b0_5_r2_1_*` fields. The
fixed-shape within-head swap design, the capture-strategy wrapper, and the
aggregation hierarchy from B0.5-R2 are unaffected and not reopened.

**B0.5-R2.1 VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION** —
supersedes B0.5-R2's verdict below. Authorizes only CPU-side B1A
prerequisite implementation (MATH-500 verifier, architecture-aware R-KV
dispatch, the repaired pairwise provenance schema with timing fields, the
repaired per-instance read-only capture wrapper, the frozen deterministic
sampling algorithms, the mandatory no-op control's CPU unit test, CPU
tests generally). Does not authorize B1B, GPU use, model inference, or any
method implementation. The `CLAUDE.md` §4 model-freeze amendment remains
required before any GPU run of a later phase and is not granted by this
record.

## 2026-07-19 — Phase B0.5-R2: dense-cache representability and capture-strategy repair (documentation-only; no method or harness implemented; no GPU used, no model inference, no model weights or datasets downloaded; no MATH-500 manifest/config/evaluator/result directory created; no code under `src/`, `tests/`, `configs/`, `scripts/`, `results/`, `schemas/`, or `third_party/` touched; no frozen §1/§4/§8/§9 value changed)

Run on branch `research/b0-5-r2-dense-cache-repair`, cut from `main` at
`d472f0514cd1396774b557dc27ec19900a11c1eb` (tip of `origin/main`, containing
the B0.5-R merge). Purpose: verify B0.5-R's selected intervention and
capture-strategy claim directly against the pinned R-KV source
(`third_party/R-KV` @ `45eaa7d69d20b7388321f077020a610d9afb65bd`) and the
installed `transformers==4.55.4` cache implementation, rather than trust
B0.5-R's already-committed READY verdict. Two load-bearing assumptions were
found false:

- **B0.5-R §7-§8's "equal-byte add-back" / "retained-only physical
  ablation" intervention is not representable.** `transformers.DynamicLayer`
  stores K/V as one dense `(batch_size, num_heads, seq_len, head_dim)`
  tensor per layer (`cache_utils.py:68-104`, read directly from the
  installed package); R-KV's own `topk(budget - window_size, dim=-1)`
  (`r1_kv.py:82`) always selects the *same count* per head, so every head
  always has exactly `budget` slots after compaction. A slot cannot be
  added or removed "at one (layer, kv_head) pair only" while leaving every
  other head at that layer unchanged — the tensor has one shared `seq_len`
  dimension across every head. Repaired intervention: a fixed-shape
  **within-head swap** — `key_cache[L][0,h,r_slot,:] = captured_key_e`,
  `value_cache[L][0,h,r_slot,:] = captured_value_e` — net physical cache
  bytes always exactly 0, no dimension ever resized.
- **B0.5-R §6's capture-hook claim does not hold.** No supported Python
  wrapper can read a function's internal local variables
  (`final_score`, `indices`) at an arbitrary line inside `R1KV.update_kv`
  from outside the function. Repaired to a per-instance before/after
  wrapper (bound to each layer's own `R1KV` instance, never a class-level
  or global patch) that clones pre-call inputs, calls the original
  unmodified `update_kv`, and independently recomputes the real windowed
  score formula (`r1_kv.py:49-77`) — verified for parity against R-KV's
  own real `kept_token_indices` bookkeeping and a bit-exact
  gather-reproduction check, never fed back into R-KV.

Also discovered and repaired: R-KV's own persisted `kept_final_scores`
bookkeeping (`r1_kv.py:88-165`) is computed by a **different, unwindowed
formula** than the one that actually drives the real eviction decision
(`r1_kv.py:49-77`) — a schema field naively sourced from
`kept_final_scores` would silently log the wrong quantity. Also repaired:
a mandatory two-pass capture plan (event eligibility depends on the
complete natural-generation trajectory, so it can only be known after
Pass 1 finishes — a second, token-identical instrumented replay pass is
required to capture at preselected targets, a real cost previously absent
from the B0.5-R/B0.5 cost models); gate 10 (previously did not require any
actual positive ranking reversal to exist — repaired to require a
predeclared fraction of examples showing a reversal above a fixed noise
floor); and an explicit, deterministic (layer, KV-head) sampling rule
(previously left implicit).

Full repair: `docs/B0_5_R2_DENSE_CACHE_REPAIR.md`. Superseded passages in
`docs/B0_5_PROTOCOL_REPAIR.md`, `docs/B0_5_DISCOVERY_PROTOCOL.md`, and
`docs/B0_5_FEASIBILITY_AUDIT.md` are marked inline, not deleted;
`docs/b0_5_decision.json` retains every original field and adds
`superseded_by_r2`/`b0_5_r2_*` fields pointing to the repair. B0.5-R's
corrected decision unit (§4), B1A-1/B1A-2 prerequisite findings, and B0's
method-pivot verdict are unaffected and not reopened.

**B0.5-R2 VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION** —
supersedes B0.5-R's verdict below. Authorizes only CPU-side B1A
prerequisite implementation (MATH-500 verifier, architecture-aware R-KV
dispatch, the repaired pairwise provenance schema, the repaired
per-instance read-only capture wrapper, CPU tests). Does not authorize
B1B, GPU use, model inference, or any method implementation. The
`CLAUDE.md` §4 model-freeze amendment remains required before any GPU run
of a later phase and is not granted by this record.

## 2026-07-19 — Phase B0.5-R: causal discovery protocol repair (documentation-only; no method or harness implemented; no GPU used, no model inference, no model weights or datasets downloaded; no MATH-500 manifest/config/evaluator/result directory created; no code under `src/`, `tests/`, `configs/`, `scripts/`, `results/`, `schemas/`, or `third_party/` touched; no frozen §1/§4/§8/§9 value changed)

Run on branch `research/b0-5-protocol-repair`, cut from `main` at
`48fdf7f5020cbac6501fa45a7d1ffc38e668d4f1`. Purpose: verify Phase B0.5's
technical design directly against the pinned R-KV source
(`third_party/R-KV` @ `45eaa7d69d20b7388321f077020a610d9afb65bd`) rather
than trust its already-committed READY verdict. Two load-bearing
assumptions were found false:

- **§1's "fixed 128-token block" experimental unit does not exist.**
  `divide_length=128` is only the periodic cadence at which every layer
  *checks* whether to compact (`modeling.py:601`); actual eviction is
  decided independently per layer (one `R1KV` instance per layer) and
  independently per KV head within a layer (`topk` over the sequence
  dimension, per head, `r1_kv.py:75-82`) — never over a contiguous
  128-token span as one object. Repaired unit: `(compaction_event_id,
  layer_index, kv_head_index, absolute_token_position,
  pre_compaction_storage_position)`, a single cache slot.
- **§5's shadow-FullKV KV-recovery method does not recover R-KV's true
  pre-eviction state**, for any sampled event beyond the run's first
  compaction (true of every eligible event under the protocol's own
  eligibility rule): the real R-KV run's token was produced by attending
  over an already-compressed history, while a shadow-FullKV replay never
  experiences any eviction anywhere in its prefix — a different
  cache-policy trajectory, not a reconstruction of the same one. Repaired
  source: a read-only instrumentation hook capturing the real pre-gather
  tensors immediately before `R1KV.update_kv`'s eviction gather; if
  unattachable at implementation time, the intervention is blocked for
  that run rather than silently substituted.

Also repaired: the intervention design (equal-byte add-back for evicted
candidates vs. retained-only physical removal for retained controls,
analyzed separately and never pooled into one utility distribution;
"mask/zero" replaced by physical tensor-slice removal, since a zeroed
key/value slot is not equivalent to removal from the attention softmax);
an exact numeric gate table replacing every vague threshold ("e.g.
0.10-0.15", "meaningful fraction", "say 8 of 12", "dramatically", "low or
near-zero", "strong correlation", "notably above"), reusing this
repository's existing `0.10` accuracy-plausibility ceiling and `0.70`
`meaningful_retention_ceiling` where applicable and stating the purpose of
every new threshold explicitly; and confirmation of two concrete B1A
prerequisites this repository does not yet have — `src/kvcot/generation/policies.py`
calls only `rkv.monkeypatch.replace_qwen2` unconditionally (Candidate A's
Llama-8B checkpoint would silently run with stock, unpatched Llama
attention while still being labeled `rkv_b1024` — a silent mislabeling
defect, not a crash) and `src/kvcot/utils/answers.py` is numeric-only (no
symbolic-equivalence checking for MATH-500's fractions/algebraic answers).
B0's method-pivot verdict (M1/M2/M3 all still non-survives) is unchanged
and not reopened.

Full repair: `docs/B0_5_PROTOCOL_REPAIR.md`. Superseded passages in
`docs/B0_5_DISCOVERY_PROTOCOL.md` and `docs/B0_5_FEASIBILITY_AUDIT.md` are
marked inline, not deleted; `docs/b0_5_decision.json` retains every
original field and adds `superseded_by`/`b0_5_r_*` fields pointing to the
repair.

**B0.5-R VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION** — supersedes
B0.5's original "READY FOR B1 DISCOVERY-HARNESS IMPLEMENTATION" below.
Authorizes only CPU-side B1A prerequisite implementation (MATH-500
verifier, architecture-aware R-KV dispatch, decision/provenance schema,
read-only compaction instrumentation, CPU tests). Does not authorize B1B,
GPU use, model inference, or any method implementation. The `CLAUDE.md`
§4 model-freeze amendment remains ungranted.

## 2026-07-19 — Phase B0.5 (superseded by B0.5-R above): causal false-negative discovery protocol and operating-point feasibility gate (documentation-only; no method or harness implemented; no GPU used, no model inference, no model weights or datasets downloaded; no MATH-500 manifest/config/evaluator/result directory created; no code under `src/`, `tests/`, `configs/`, `scripts/`, `results/`, `schemas/`, or `third_party/` touched; no frozen §1/§4/§8/§9 value changed)

Run on branch `research/b0-5-discovery-protocol`, created from
`research/phase-b0-method-pivot` at commit `68b56f1` (itself created for
this same session's separate B0-doc additions below), which sits on `main`
at the B0-merge tip `f7e9dcc` (parent `8d5aa21`, confirmed a pure
fast-forward, zero content diff). Purpose: decide whether a narrower,
non-method *discovery* question — do deployed R-KV compressors evict
blocks with high counterfactual future utility that cheap deployable
signals miss ("unexplained causal false negatives") — justifies requesting
a later, CPU-developed B1 harness. This does not reverse B0's method-pivot
verdict (M1/M2/M3 all still non-survives) and makes no method-novelty
claim of its own.

**B0 audit:** all three verdicts upheld on review (`docs/b0_5_decision.json`
`b0_audit_verdict`) — M1's kill rests on three independent full-text-quoted
overlaps (ArborKV, IntentKV, ThinKV), not snippet-tier evidence; M2's kill
rests on directly quoted MDP-state definitions (ForesightKV) plus this
repository's own locally-audited R-KV rescoring behavior; M3's verdict is a
self-consistency argument (new constraint metric insufficient by this
project's own predeclared standard) independent of any single paper's
evidence tier.

**Documentation corrections:** `README.md` gained a "Current status" section
ahead of the original diagnostic framing, which is now explicitly labeled
"Historical research question" — no historical sentence removed or edited.
`docs/EXPERIMENT.md` gained a top-of-file ARCHIVAL NOTICE banner (frozen
historical protocol; A3 killed diagnostic novelty; B0 found no novel
method; no new run without a separately approved redesign) — no body
content below it was edited.

**Discovery protocol** (`docs/B0_5_DISCOVERY_PROTOCOL.md`): experimental
unit is a fixed 128-token block (aligned to R-KV's own audited compaction
schedule, `docs/UPSTREAM_AUDIT.md` H4); intervention is a single-block
rescue (evicted candidates, recovered via a shadow-FullKV teacher-forced
replay of the R-KV run's own tokens) or single-block ablation (retained
controls), always relative to the real deployed R-KV cache at a sampled
compaction event — never a full-cache ablation, never a natural-eviction
substitute. Primary metric: reference-token NLL increase over a fixed
48-token horizon (continuous, not final-answer correctness). Baseline-signal
controls are mandatory, including the entropy/logit-margin/branching-marker
confound identified below. Sample size (12 examples × 3 events × 4 blocks =
144 labeled blocks) is feasibility-bounded, not power-bounded, and stated as
such.

**Feasibility audit** (`docs/B0_5_FEASIBILITY_AUDIT.md`): three candidate
operating points compared on primary evidence — **Candidate A (selected)**
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` + MATH-500 + R-KV budget 1024,
backed by a primary-source "lossless at 34%/1024 tokens" claim directly
from the pinned `third_party/R-KV` submodule's own README/paper
(arXiv:2505.24133) — the first candidate operating point in this project's
history with real prior public evidence of viability, unlike the retired
GSM8K b128 point. Candidate B (same model, AIME-24, budget 1536, "lossless
at 10%/1536") rejected as secondary due to small competition-year
population and ~5× longer traces. Candidate C (`DeepSeek-R1-Distill-Qwen-7B`
+ MATH-500) rejected as weaker-evidence-tier (no primary-source
R-KV-specific lossless-budget number found this session). GPU cost model:
144,240 core forward-token-equivalents × 3 safety factor = 432,720; at an
estimated 35-50 tok/s on a single RTX 3090, projects to 2.4-3.4 GPU-hours,
under the 4-hour ceiling with a stated margin — an estimate, not a
measurement; the protocol's own stopping rules require empirical
re-projection after the first example before any future run continues.

**Literature** (`docs/B0_5_SEARCH_LOG.md`): 13 of 16 mandatory-list papers
reused from B0/A3's already-fetched, evidence-graded records (saturation
argument reviewed and endorsed, not merely repeated); 3 new fetches this
session (Runtime-Certified Bounded-Error Quantized Attention
arXiv:2605.20868, Near-Oracle KV Selection via Pre-hoc Sparsity
arXiv:2602.08329, EntmaxKV arXiv:2605.21649) — none is a KV-eviction-rescue
method, none overlaps this protocol's methodology or operating point. One
open, non-load-bearing attribution gap logged: a "counterfactual occlusion"
32-token-window importance-labeling precedent surfaced by search, source
paper not conclusively identified this session.

**Overall: B0.5 VERDICT: READY FOR B1 DISCOVERY-HARNESS IMPLEMENTATION**
(`docs/b0_5_decision.json`). This authorizes only a later, user-requested,
CPU-developed B1 harness — **it does not authorize any GPU use, model
inference, model/dataset download, or method implementation.** A real
blocker remains even for that harness's eventual GPU run: `CLAUDE.md` §4's
model freeze names only `DeepSeek-R1-Distill-Qwen-1.5B`; using Candidate A
requires a separate, dated `CHANGELOG.md`/`CLAUDE.md` amendment **before**
any such run, not granted by this entry. B0's method-pivot verdict is
unchanged; PSS/CPSS/EAS/Delta_EAS are unchanged and were not reused as
B0.5's primary quantity.

Files created: `docs/B0_5_DISCOVERY_PROTOCOL.md`,
`docs/B0_5_FEASIBILITY_AUDIT.md`, `docs/B0_5_SEARCH_LOG.md`,
`docs/b0_5_decision.json` (`python -m json.tool` passes). Files updated:
`PLAN.md` (status + roadmap), `CHANGELOG.md` (this entry), `README.md`
(status section reorder + historical label), `docs/EXPERIMENT.md`
(archival banner).

## 2026-07-19 — Phase B0 addendum: quantization-overthinking confound and pilot measurement checklist (documentation-only; same non-modification guarantees as the B0 entry below)

Added, on branch `research/phase-b0-method-pivot` (commit `68b56f1`, before
the B0.5 branch above was cut): a new `docs/METHOD_NOVELTY_MATRIX.md` §3a
and `docs/method_novelty_matrix.json` record for "Quantized Reasoning
Models Think They Need to Think Longer, but They Do Not" (arXiv:2606.00206,
verified by direct fetch — DeepSeek-R1-Distill-Qwen 1.5B/7B/14B,
DeepSeek-R1-Distill-Llama 8B, QwQ-32B; GPTQ/AWQ weight-only vs. FlatQuant
W4A4KV4/W8A8KV8 end-to-end quantization; same-prefix matched-generation
control; token-level KL/entropy/logit-margin diagnostic; up to 52% of
failures are a correct intermediate answer abandoned). Not a KV-eviction
method and credited with no M1/M2/M3 overlap change — its role is a
required measurement-validity confound, recorded as new
`docs/METHOD_PIVOT_SPEC.md` §5a: any future causal-false-negative claim
must be shown to survive conditioning on entropy, logit margin, and
branching-marker frequency, since quantization noise alone is already known
to produce behavior changes correlated with next-token entropy
(Spearman ρ=0.92), independent of any specific information being destroyed.
Also added `docs/PIVOT_PILOT_PROTOCOL.md`, a measurement checklist (natural-run
counts/ratios/termination/overthinking-marker/gold-appearance/abandonment/
compaction statistics, plus controlled fixed-trace-replay entropy/KL/
logit-margin/branching-mass measurements) for a future, still-unauthorized
method pilot — not authorized to run under the still-BLOCKED B0 verdict.

Files created: `docs/PIVOT_PILOT_PROTOCOL.md`. Files updated:
`docs/METHOD_NOVELTY_MATRIX.md`, `docs/METHOD_PIVOT_SPEC.md`,
`docs/method_novelty_matrix.json` (`python -m json.tool` passes).

## 2026-07-19 — Phase B0: method pivot specification and adversarial method-novelty gate (documentation-only; no frozen §1/§4/§8/§9 value changed; no code/config/test/schema/result/manifest modified; no GPU used, no model inference, no model weights downloaded; no MATH-500 file created; R-KV submodule untouched; A1/A2/A3 artifacts untouched)

Run on branch `research/phase-b0-method-pivot`, created from `main` at the
A3 merge commit `e0b6c68`. Purpose: decide whether FaithKV has a
sufficiently novel candidate *method* (not diagnostic — A3 closed that) to
justify a later GPU pilot. Search cutoff 2026-07-19; 6 targeted web
queries + 20 successful direct arXiv/GitHub fetches + reuse of frozen A3
records; ~40 works screened, 13 abstracts reviewed, 7 full texts
inspected via fetch, CASK repository re-fetched, R-KV audited locally.
Full log: `docs/B0_SEARCH_LOG.md`. Narrative matrix:
`docs/METHOD_NOVELTY_MATRIX.md`. Machine-readable:
`docs/method_novelty_matrix.json` (25 records, `python -m json.tool`
passes). Spec and verdict: `docs/METHOD_PIVOT_SPEC.md`.

Three candidates were evaluated against a prospective (untested)
causal-false-negative failure hypothesis, defined in
`docs/METHOD_PIVOT_SPEC.md` §5 — A2 motivates but cannot evidence it:

- **M1 — residual causal-utility protection: PARTIAL — INSUFFICIENT
  METHOD NOVELTY.** Strongest threats: ArborKV (arXiv:2605.22106 —
  leave-one-out KV-block zeroing with answer-accuracy supervision
  calibrating a lightweight online estimator); IntentKV (arXiv:2606.09916
  — zero-initialized learned residual head explicitly correcting "cases
  the rule scorer misses" on top of a deployable heuristic retention
  score); ThinKV v2 (arXiv:2510.01290 — counterfactual segment-ablation
  KL directly determining a deployed reasoning compression policy);
  ForesightKV (arXiv:2602.03203 — post-eviction loss increase inside the
  training reward of a deployed reasoning evictor); Adaptive Filtering
  (arXiv:2607.13205 — intervention-diagnosed systematic-error correction
  of an existing score); protected partitions in CASK/VaSE/arXiv:2605.18053.
  Remaining delta is an unstudied intersection of published components —
  insufficient under the same predefined rule that killed A3's N3.
- **M2 — interaction-aware dynamic rescue: KILLED.** ForesightKV's GRPO
  stage is an MDP whose state is "the current remaining KV cache at step
  t"; Neural Garbage Collection (arXiv:2604.18002) conditions learned
  eviction on cache state for reasoning models; the pinned R-KV itself
  already rescores at every compaction over the retained set.
- **M3 — faithfulness-constrained memory allocation: PARTIAL —
  INSUFFICIENT METHOD NOVELTY.** ReasonAlloc (arXiv:2606.11164) and the
  Ada-KV/LKV lineage own the constrained-allocation machinery; swapping in
  a causal-dependence constraint is a new metric in a known optimizer,
  explicitly insufficient under the predeclared decision standard.

**Overall: METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET.** B1 is
not permitted under this result and has not started; no GPU rental, model
inference, MATH-500 implementation, or method implementation is authorized
by this entry. §10 f=1 stability remains UNRESOLVED (not a B0 task);
GSM8K b128 remains retired.

Files created: `docs/METHOD_PIVOT_SPEC.md`, `docs/METHOD_NOVELTY_MATRIX.md`,
`docs/B0_SEARCH_LOG.md`, `docs/method_novelty_matrix.json`. Files updated:
`PLAN.md` (status + roadmap), `CHANGELOG.md` (this entry), `README.md`
(one-paragraph B0 status link), `docs/EXPERIMENT.md` §11 (one bracketed
dated note fixing a genuine stale current-status sentence that still named
"a MATH-500 longer-trace feasibility design" as the next experimental work
— written before A3/B0 blocked that path; the historical sentence itself
is preserved). No other file changed. The frozen A3 artifacts
(`docs/RELATED_WORK_MATRIX.md`, `docs/A3_SEARCH_LOG.md`,
`docs/related_work_matrix.json`) were read, cited, and not modified.

## 2026-07-19 — Phase A3: adversarial literature matrix and diagnostic novelty kill-check (documentation-only; no frozen §1/§4/§8/§9 value changed; no code/config/test/schema modified; no GPU used, no model inference, no historical result artifact touched; no MATH-500 code added; R-KV submodule untouched)

Literature-only gate, run before any further experimental work per
`PLAN.md`'s "literature matrix" next step. Cutoff: 2026-07-19. 24 distinct
search queries across 5 rounds (mandatory-paper lookup, screened-paper
lookup, semantic/combination variants, CASK code-level deep dive,
Fixed-Contract deep dive); 20 papers fully entered into the matrix (9
mandatory papers from the task brief + 11 screened); 2 papers (CASK,
Fixed-Contract) escalated to direct primary-source fetch (arXiv abstract
page; for CASK also the official GitHub README and the
`replay_reference_fidelity.py` evaluation script source) since they are the
two papers capable of independently deciding the verdict. Full
methodology and query log: `docs/A3_SEARCH_LOG.md`. Full matrix (narrative):
`docs/RELATED_WORK_MATRIX.md`. Machine-readable: `docs/related_work_matrix.json`
(schema-validated, `python -m json.tool` passes).

**Highest-threat paper: CASK (arXiv:2604.10900, "Core-Aware Selective KV
Compression for Reasoning Traces"), released 2026-04-13.** Its official,
publicly released evaluation harness (`scripts/replay_reference_fidelity.py`,
fetched directly from `github.com/Skyline-23/CASK`) implements the exact
fixed-generated-trace / teacher-forced / cache-policy-varying replay
primitive this repository's narrower novelty claim (N1) rested on: it reads
a FullKV reference continuation, feeds its tokens through a compressed-cache
condition via strict teacher forcing (`target_token =
continuation_ids[step_idx:step_idx+1]` fed unconditionally every step,
regardless of the candidate model's own prediction), and records per-step
fidelity/agreement/cache-size statistics — applied to reasoning models
(AIME24/AIME25) under decode-time KV eviction. It does not implement an
early-answering/omitted-suffix intervention (the replay always consumes the
full reference continuation, never truncates it), does not report a
predeclared accuracy-neutral gate as a headline claim boundary, and does not
produce a held-out per-example mechanism taxonomy.

**Verdicts:**
- **N1 (diagnostic primitive): DOES NOT SURVIVE** — killed by CASK's public
  evaluation code, three months before this repository's Phase A2 commit.
- **N2 (reasoning + decode-time KV application): DOES NOT SURVIVE** — CASK
  kills the "reasoning model, decode-time KV eviction" half directly; Lanham
  et al. (arXiv:2307.13702, 2023) independently kills the
  "early-answering-as-a-technique" half (no KV-cache axis in Lanham).
- **N3 (accuracy-neutral intervention-based FaithKV gap): SURVIVES** as an
  empirical gap — no reviewed paper combines KV-cache compression,
  early-answering/omitted-suffix intervention, a predeclared accuracy-neutral
  gate, realized-memory matching, held-out evaluation, and per-example
  mechanism classification, together, on a reasoning model.
- **Overall: DIAGNOSTIC SURVIVAL VERDICT: DOES NOT SURVIVE.** Per the
  project's predefined rule, an earlier implementation of the essential
  matched-trace decode-time cache diagnostic (CASK) means the diagnostic
  does not survive overall, even though the specific N3 empirical
  intersection remains open — that intersection is an application of two
  independently known ingredients (CASK-style replay; Lanham-style early
  answering) to a new setting, not by itself a new method contribution.
- **PHASE B: BLOCKED — DIAGNOSTIC NOT NOVEL.**

**Claims removed from this repository's documentation as a result (see
`README.md`, `docs/RELATED_WORK_MATRIX.md` §16/Part 16 mapping):** this
repository does not claim, and must not claim going forward, to be the
first fixed-trace FullKV/compressed-KV diagnostic, the first teacher-forced
KV-policy replay, the first reasoning-trace replay under decode-time
compression, the first early-answering CoT-faithfulness diagnostic, the
first causal reasoning-to-answer intervention, or the first separation of
accuracy and reasoning faithfulness. The repository never made most of
these claims explicitly (its `CLAIM_BOUNDARY_NOTICE` already disclaimed
faithfulness conclusions), but this entry records the check as performed
and the boundary as now literature-grounded rather than merely
self-imposed.

**Also verified during this pass (A1/A2 spot-check, no A1/A2 numeric claim
changed):** independently recomputed every A2 headline number directly
from the committed `results/tables/gsm8k_v3_b128_failure_atlas.csv` (50
rows, 50 unique `source_row_index`, FullKV 33/50, R-KV 13/50, correctness
distribution 12/21/1/16, retention mean 0.3596/median 0.3485, compaction
mean 3.9/median 4.0, 9/50 `identical_through_think` with 3 correct→wrong
flips, 41/50 diverge inside the reasoning region) — all match the committed
`results/decisions/gsm8k_v3_b128_failure_atlas_summary.json` and prior
CHANGELOG entries exactly; no discrepancy found. Two genuine documentation
gaps (not data/code defects) were found and fixed by adding prose-only
caveats to `docs/EXPERIMENT.md` §11 (no historical artifact modified): (1)
the "0/50 pairs diverge before first compaction" finding rests on a short
pre-compaction observation window (mean 15.6 / median 13.5 generated tokens,
~4.0% of the mean 450.7-token trace, since first compaction fires at
absolute position 129 for all 50 pairs on this schedule) and is therefore
weak temporal, not causal, evidence; (2) the atlas's own committed
"decoded text after `</think>`" excerpts (`results/tables/
gsm8k_v3_b128_failure_atlas.md`) show substantial step-by-step mathematical
re-derivation continuing past the closing marker on all three
correct→wrong flip rows, so `</think>` is confirmed to be a token-format
boundary, not a validated semantic reasoning/answer boundary — the
`reasoning`/`post_think_answer` split must be read as a literal token-
position classification, not proof that everything after `</think>` is
answer-only text.

## 2026-07-19 — Phase A2: deterministic GSM8K protocol-v3 failure atlas (CPU-only, post-hoc diagnostic; no frozen §1/§4/§8/§9 value changed; no GPU used, no model/tokenizer loaded, no generation rerun, no historical `results/raw/`, `results/gate_artifacts/`, or `results/decisions/*_accuracy_gate.json` file modified)

New `src/kvcot/failure_atlas.py` + `kvcot failure-atlas` CLI command build a
deterministic, tested atlas over the 50 committed protocol-v3
FullKV/R-KV-B128 GSM8K pairs from the immediately preceding entry, pairing
records by `(source_row_index, global_seed)` (never file order), verifying
both `.jsonl.gz` gate artifacts against their committed `.sha256` first, and
never mutating any committed artifact. `tests/unit/test_failure_atlas.py`
(55 tests) and `tests/unit/test_cli_failure_atlas.py` (5 tests) cover the
prompt-offset coordinate arithmetic (mandatory regression: prompt=200,
generated-index=40, first-compaction-absolute=230 → absolute divergence 240,
classified `after_first_compaction`, not `before`), the `</think>`-relative
divergence classification, and pairing/integrity failure modes (duplicate
keys, missing counterparts, mismatched question hashes/prompts,
count!=50, cross-condition provenance mismatch, tampered/missing checksum
sidecars). Full CPU suite: 418 passed (`pytest -m "not gpu" tests/`), up
from 358 before this entry. The command was run twice against the committed
`.gz` artifacts and produced byte-identical output hashes both times.

Committed outputs: `results/tables/gsm8k_v3_b128_failure_atlas.csv` (50
rows), `results/tables/gsm8k_v3_b128_failure_atlas.md`,
`results/decisions/gsm8k_v3_b128_failure_atlas_summary.json`
(`diagnostic_label: post_hoc_diagnostic`, `operating_point_valid: false`,
`hypothesis_status: not_tested`).

Every headline number was independently recomputed from the atlas and
matches the previous entry's manually-derived figures exactly: FullKV
33/50, R-KV 13/50 (both_correct 12, full_only 21, rkv_only 1, both_wrong
16); realized retention mean 0.360/median 0.349 (48/50 below 0.5, 50/50 at
or below 0.7); compaction count mean 3.9/median 4.0 (range 2-7); 0/50 pairs
diverge before their first R-KV compaction event in the shared absolute
(prompt + generated-index) coordinate system, all 50 diverge at or after
it; 9/50 pairs are token-identical through `</think>` (source rows 30, 176,
262, 271, 491, 543, 616, 1115, 1143), of which exactly 3 are correct→wrong
flips (rows 30, 271, 1115) and 6 are correct→correct. No mismatch against
the prior manual analysis was found. New in this entry (not previously
computed): 41/50 pairs first diverge *inside* the reasoning span itself
(before either side's `</think>` starts), vs. 9/50 only in the post-`</think>`
answer region — the identical-through-think flips are the minority failure
pattern, not the typical one, at this retired operating point. Zero cap
hits, zero malformed answers, and zero malformed/missing `</think>`
boundaries were observed in either condition across all 50 pairs.

Claim boundary (restated per §1, and embedded in every atlas output):
this is a post-hoc diagnostic over a RETIRED operating point (0.40 accuracy
drop against the 0.10 pilot ceiling) and cannot establish that any observed
failure pattern occurs at an accuracy-preserving operating point. It
generates hypotheses for later held-out testing only — it does not test,
pass, or fail the §1 research question, and none of its correlations are
causal claims.

## 2026-07-19 — Documentation-only status update: protocol-v3 natural accuracy gate failed; GSM8K b128 operating point retired (no frozen §1/§4/§8/§9 value changed; no code/config/test/schema modified; no raw or gate result file touched)

Documentation-only change recording the outcome of the protocol-v3 natural
R-KV accuracy gate. Only `README.md`, `docs/EXPERIMENT.md`,
`docs/GPU_VALIDATION_PLAN.md`, and this file were edited; no source, config,
test, schema, `results/raw/`, `results/gate_artifacts/`,
`results/decisions/`, or `third_party/` content was modified. Every number
below was recomputed from committed artifacts —
`results/decisions/early_gap_v3_b128_accuracy_gate.json` and the two
`results/gate_artifacts/early_gap_v3_b128_*.jsonl.gz` files (each verified
against its committed `.sha256`).

- **The gate failed.** Over all 50 natural records of each condition, FullKV
  answered 33/50 (66%) correctly and natural R-KV b128 13/50 (26%) — a 40pp
  drop against the 0.10 pilot ceiling (`gate_passed: false`; 12
  both-correct, 21 FullKV-only-correct, 1 R-KV-only-correct; analyzer commit
  `9a89409`). The fixed-trace analysis path (`run_fixed_trace_analysis`)
  correctly exited 1 without computing PSS/CPSS, so no protocol-v3 PSS/CPSS
  decision exists and `hypothesis_status` remains `not_tested`. This pilot
  gate never spoke to distributional accuracy preservation (§9 /
  CLAUDE.md §8.5); it only kept the pilot off an absurd operating point.
- **Compression was real.** R-KV compacted 2–7 times per example (mean 3.9;
  FullKV 0); final measured `instantaneous_retention_ratio` averaged 0.36
  (median 0.35), 48/50 below 0.5 and 50/50 below 0.7. The collapse is not an
  artifact of compression failing to fire.
- **Divergence never preceded compaction.** In absolute sequence positions
  (prompt length + generated index), 0/50 pairs diverged before their first
  compaction event; all 50 diverged at or after it.
- **Post-hoc diagnostic, not a mechanism claim.** 9/50 pairs (source rows
  30, 176, 262, 271, 491, 543, 616, 1115, 1143) generated byte-identical
  tokens through `</think>` under both policies; 3 flipped correct→wrong
  afterward — row 30 (109 → `71.\overline{1}`), row 271 (20 → 12), row 1115
  (5 → 6). Recorded as hypothesis-generating signposts collected at a
  degraded (26%) operating point, NOT as evidence about the §1 research
  question or any internal mechanism.
- **Operating point retired.** FullKV traces on this manifest run 276–847
  generated tokens (median ~440), leaving no fixed budget that is
  simultaneously accuracy-plausible and meaningfully compressing. The
  GSM8K + `DeepSeek-R1-Distill-Qwen-1.5B` + b128 operating point is retired;
  no further GSM8K b128/b160 runs are planned. Next experimental work: a
  CPU-only failure atlas over the existing 50 pairs, then a MATH-500
  longer-trace feasibility design with separate calibration and held-out
  manifests. The earlier one-example frozen fixed-trace result (row 30's
  probe answered 109) is superseded by this failed natural gate and must not
  be cited as evidence.

## 2026-07-18 — Analysis-path fixes: selected-population accuracy defect, strict-gate hardening, provenance (no frozen §1/§4/§8/§9 value changed; no generation/replay/compression/PSS-CPSS math touched; no completed raw GPU record modified)

An external audit of the completed protocol-v3 code found one mandatory
correctness defect in the final CPU analysis path, two robustness defects
around it, and a set of reproducibility/test-accounting gaps. All verified
against the code before fixing. The GPU generation, R-KV integration,
replay/snapshot machinery, frozen-cache probing, retention measurement, and
schedule-simulator code are untouched.

- **THE selected-population accuracy defect (mandatory).**
  `kvcot.analysis.fixed_trace.run_fixed_trace_analysis` filtered the
  canonical base records down to the `--selection-file` subset BEFORE
  building the natural-accuracy screen, so `build_accuracy_screen` paired
  the 10 selected FullKV records (all correct BY CONSTRUCTION — selection
  requires FullKV correctness) against the natural R-KV records:
  `full_accuracy` came out 10/10 instead of the true 33/50, silently
  biasing `pilot_accuracy_plausible` toward passing. Fixed: the strict
  natural-accuracy gate (`build_strict_accuracy_gate`) now runs over ALL
  natural records of both conditions, before any selection filtering; only
  pair construction/PSS/CPSS/curves/eligibility are selection-scoped. The
  decision JSON now records the complete `strict_accuracy_gate` object
  (never just the inner `accuracy_screen`), `build_screen_validity` fails
  the whole screen when `gate_passed` is not True (a failed gate can never
  produce `hypothesis_status: "screened"`), and `run_fixed_trace_analysis`
  exits 1 (after still writing the decision JSON documenting why) on gate
  failure. The prior GPU-produced records themselves are unaffected — the
  defect lived purely in the CPU analysis path; no decision computed from
  it was ever published as a result.
- **`check-fixed-trace-accuracy` accepted `--limit`/`--problem-index`/
  `--seed` (mandatory).** Those flags fed `expected_n`, so a partial pair
  of natural files could be blessed as "the expected experiment" by passing
  a matching restriction. Removed outright; both
  `cmd_check_fixed_trace_accuracy` and `cmd_analyze_fixed_trace` now derive
  the expected natural count from one shared helper
  (`kvcot.cli._expected_stage_record_count`: manifest rows after the
  config's own `limit` × resolved seeds — 50 for the current stage) that
  takes no CLI-args input at all, so the two commands can never derive
  different expectations.
- **`build_strict_accuracy_gate` violated its "never raises" contract
  (mandatory).** It extracted `(source_row_index, global_seed)` keys by
  direct subscripting BEFORE schema validation, so a malformed record
  (missing `dataset`/`global_seed`, or not a dict) raised `KeyError` out of
  the function instead of returning `gate_passed: False`. Rewritten:
  every row is schema-validated first; keys/identities/conditions/accuracy
  are computed only from valid rows; any invalid row fails the gate with a
  per-index reason and `accuracy_screen: null`; the returned object always
  has the full stable shape.
- **Analysis provenance + input hashes.** The fixed-trace decision JSON and
  the accuracy-gate JSON now record `analysis_provenance` (the ANALYZER's
  own git commit + dirty flag — the GPU data producer's commit is already
  on every raw record and the two must never be conflated: the raw pilot
  data was produced at `ef9bb1e...`, analysis code moves independently) and
  `input_sha256` over every file read (base, natural R-KV, both probe
  files, selection; config + lock hashes on the gate JSON).
- **Selection-file validation.** `kvcot.cli._load_fixed_trace_selection`
  silently collapsed duplicate candidate entries via a dict comprehension.
  Now rejected: duplicate candidate `base_record_id`/`source_row_index`
  entries, `n_ranked`/`n_predicted_eligible` disagreeing with the actual
  candidate entries, candidates' own `selected` flags disagreeing with
  `selected_base_record_ids`, and selected ids not present in the canonical
  base file. Count checks key off `n_ranked` (the entries actually written),
  never `n_candidates_considered` (which legitimately counts the 17
  rejected-before-ranking rows the real committed selection file never
  wrote as candidates).
- **f=1 stability test accounting
  (`tests/integration/test_probe_stability_gpu.py`).** Two defects: it ran
  10 of the pre-registered 20 smoke rows, and it incremented `n_valid`
  BEFORE checking probe-answer extractability, so an unextractable probe
  counted as valid-but-unstable while the assertion message claimed a
  both-extractable denominator. A valid pair now requires: base not
  cap-hit, base answer extracted, think span parsed, probe not stopped on
  `max_new_tokens`, probe answer extracted; per-example diagnostics are
  returned; a crashed child process is detected by polling instead of
  blocking the full queue timeout, and child exceptions propagate as
  structured errors. The 0.90 threshold and control suffix are NOT touched.
  **The prior 7/10 result remains unresolved** — this changes what is
  counted, not what passes; the corrected 20-row run has not happened.
- **`scripts/verify_environment.sh` now enforces instead of printing:**
  Python 3.12.x, torch `2.6.0+cu124` (CUDA 12.4), transformers `4.55.4`
  (was a warning), flash-attn `2.7.4.post1`, CXX11 ABI False, CUDA + BF16
  availability, and the pinned R-KV commit are all FATAL on mismatch; a
  real BF16 FlashAttention kernel is executed (`causal=True`, shape +
  finiteness checked); the requirements-lock checker now parses the
  flash_attn direct-URL pin (the old `==`-only parser skipped that line —
  the single most state-critical pin — entirely) and reads installed
  versions via `importlib.metadata`, since `pip freeze` reports direct-URL
  installs in `pkg @ url` form.
- **Dry-run reporting**: `analyze-fixed-trace --dry-run` now reports the
  natural R-KV file, the selection file, the expected natural record count,
  and the strict-gate requirement for v3 stages;
  `check-fixed-trace-accuracy --dry-run` prints `expected_n` and that
  partial overrides are disabled.
- **Regression tests** for all of the above (27 new tests; the CPU-side
  suite grows from 331 to 358), including: selected-analysis-uses-all-50 (asserts
  `full_accuracy == 33/50`-shaped population, `n_accuracy_pairs == 50`,
  `n_shared == 10`), partial/wrong-condition/identity-mismatch natural
  R-KV rejection, gate-failure-blocks-"screened", never-raises gate inputs
  (missing `dataset`, missing `global_seed`, non-dict), CLI flag removal,
  selection duplicate/count-consistency rejection, and explicit
  protocol-v2-unchanged coverage (v2 stages: no gate built, no natural file
  read, `strict_accuracy_gate: null`, rc 0).
- **Deferred by design (documented, not implemented):** the producer-side
  identity extension (schema 1.5.0: `stage_config_sha256`,
  `lock_config_sha256`, `data_producer_tree_sha` over the generation-
  critical tree) is future-stage work. Existing pilot records stay schema
  1.4.0 exactly as produced (commit `ef9bb1e`, environment recorded in
  `requirements-lock.txt`/`results/run_manifests/`); no schema migration or
  rewrite of completed raw data.

## 2026-07-19 — Fixed-trace protocol v3, third pass: setup-script fix, selection/accuracy-gate hardening, safe-default reversal (secondary, additive; no frozen §1/§4/§8/§9 value changed)

A third review of `66f477e` independently re-verified the cache-schedule
simulator against all 90 archived protocol-v2 GPU measurements (zero
mismatches) and confirmed the core v3 design sound, but found one real
infra bug and five remaining code gaps, plus a genuine documentation/
provenance contradiction in the workflow. Fixed here, still before any GPU
spend:

- **`scripts/setup_vast.sh` would repeat a previously-hit setup failure.**
  `logs/setup.log` (committed) preserves the exact prior failure verbatim:
  `pip install -r requirements.txt` tried to build `flash-attn`'s wheel in
  a fresh, PEP-517-isolated build environment that does not see the
  already-installed `torch` from the venv (`ModuleNotFoundError: No module
  named 'torch'`), even though torch was installed in an earlier, separate
  `pip install torch==2.6.0 ...` step. `logs/flash_attn_install.log` shows
  the manual fix that worked: installing flash-attn separately with
  `--no-build-isolation`. The script now does exactly that — installs
  everything else from `requirements.txt` except `flash-attn`, then
  installs `flash-attn` on its own with `--no-build-isolation`.
- **Selection-file loading trusted `selected_base_record_ids` without
  cross-checking it against `selected_source_row_indices` or `candidates`
  at all.** A selection file with `selected_source_row_indices` and
  `selected_base_record_ids` disagreeing with each other, or with a
  selected id's own candidate entry, or with a selected id marked
  `predicted_eligible: false`, or containing duplicate entries, or an
  `n_selected` not matching the actual list length, was silently accepted.
  `kvcot.cli._load_fixed_trace_selection` now cross-validates the full
  tuple — every selected `(base_record_id, source_row_index)` pair must
  agree with its own candidate entry, which must itself be
  `predicted_eligible: true`; no duplicates; `n_selected` must match.
  Additionally, `--max-selected` silently overriding the stage config's
  pre-registered `fixed_trace.max_selected_examples` is now a loud warning
  at write time, and the loader now REJECTS a selection file whose
  `max_selected` disagrees with the CURRENT config's pre-registered value —
  closing the loophole without removing the CLI override's legitimate
  debugging use.
- **`build_strict_accuracy_gate` never checked that the R-KV file's own
  `condition` field actually equalled the requested `--replay-condition`**
  — only that all R-KV records agreed with EACH OTHER. A file that
  accidentally contained `condition="full"` records passed as the "R-KV"
  side (e.g. the wrong file given) could still pass every other check
  (counts, keys, identity) and report `gate_passed: True`. Fixed: new
  `expected_rkv_condition` parameter, wired from `--replay-condition` in
  `kvcot check-fixed-trace-accuracy`, rejects this exact case.
- **`_verify_selection_completeness` only checked for MISSING fractions,
  never extra ones, and silently filtered out any probe-file
  `base_record_id` outside the selection instead of treating that superset
  as an error.** Both are now exact-equality checks: each selected
  example's fraction set must equal the 9 frozen fractions EXACTLY (a
  stray 10th fraction value now fails loudly), and the full set of
  `base_record_id`s present in each probe file must equal the selection's
  set EXACTLY — a probe file containing extra, unselected examples (e.g.
  the wrong, larger file passed in) now aborts analysis instead of being
  quietly scoped down.
- **An unrecognized `probe_cache_mode` silently behaved like `"native"`.**
  This is exactly backwards for a safety feature whose entire point is
  preventing silent contamination — a typo intended to request
  `frozen_at_cut` protection must never quietly fall back to the
  unprotected path. `branch_and_probe` now validates `probe_cache_mode`
  at the very top of the function and raises `ValueError` for anything
  other than `"native"`/`"frozen_at_cut"`. The prior pass's own
  `test_invalid_probe_cache_mode_is_never_silently_treated_as_native` test
  asserted the wrong (unsafe) behavior explicitly — reversed to
  `test_invalid_probe_cache_mode_raises_value_error`.
- **`git_dirty` reported `true` on essentially every record after the first
  command in a session, independent of whether the code matched a
  committed state.** `results/run_manifests/`, `results/selections/`, and
  `results/decisions/` are all deliberately NOT gitignored (they're meant
  to be committed, per `README.md`'s layout) — but that means the moment
  any `kvcot` command writes its first output file, `git status --porcelain`
  reports it as untracked, and the old `git_is_dirty()` counted untracked
  files as dirty. `kvcot.runtime.git_is_dirty()` now uses
  `--untracked-files=no`, so only actual modifications to TRACKED files
  (source, config, or a regenerated `requirements-lock.txt`) count.
  Verified against a real throwaway git repo (`tests/unit/
  test_runtime_git.py`), not a mocked subprocess. `docs/GPU_VALIDATION_PLAN.
  md` now also explicitly instructs committing the freshly-regenerated
  `requirements-lock.txt` right after `setup_vast.sh` (a genuine tracked-
  file change, unlike the untracked-artifact false positive above) —
  before running anything else.
- **Still true, unchanged from the review's own conclusion**: the two real-
  model GPU tests exist and are structurally sound but have not been run on
  this commit (no GPU here); they use synthetic traces and deliberately
  shrunk budget/divide_length constants to force the schedule/eviction
  scenarios deterministically within a short fixture (same convention as
  every pre-existing GPU test in this repository); they do not construct a
  real `FixedTraceProbeRecord` or assert `protocol_version="v3"` at the
  record level — that requires the full CLI path (real tokenizer, real
  model, real R-KV policy) and remains correctly scoped to the mandatory
  one-example gate against the actual production budget=128/divide_length
  =128 config, which is already documented as a required, non-skippable
  step in `docs/GPU_VALIDATION_PLAN.md` and cannot be replaced by any CPU-
  side test. 331 CPU tests pass (up from 313).

## 2026-07-18 — Fixed-trace protocol v3, second pass: selection-to-replay wiring, strict accuracy gate, screen-validity fix (secondary, additive; no frozen §1/§4/§8/§9 value changed)

A second review of the 2026-07-17 protocol-v3 commit (`45bf2d8`) found the
frozen-cache design and cache-schedule simulator sound (independently
re-checked against the archived protocol-v2 raw GPU data: 0 mismatches
across all 90 (example, fraction) comparisons), but four gaps that would
have wasted GPU time or produced an untrustworthy screen on the next real
run. Fixed here, still before any GPU spend:

- **The selection file was written but never consumed.** `kvcot
  inspect-fixed-trace --write-selection` wrote `results/selections/
  {stage}.json`, but neither `replay-fixed-trace` nor `analyze-fixed-trace`
  had any way to read it back — the only way to run a selected subset was
  one `--problem-index` invocation per example, reloading the model every
  time. Fixed: both commands gained `--selection-file`
  (`kvcot.cli._load_fixed_trace_selection`), which verifies the selection
  was computed against THIS exact config/base-file/budget/divide_length/
  stage_name before trusting it (raises `SelectionFileMismatchError`
  otherwise), then restricts the run to exactly the selected
  `base_record_id`s. `replay-fixed-trace` refuses to proceed (`SystemExit`)
  if the selection's `n_selected` is already below `min_eligible_examples`.
  The selection path and its SHA-256 are now recorded on every
  `replay-fixed-trace` `RunManifest` (`RunManifest.selection_path`/
  `selection_file_sha256`, new optional fields).
- **`--max-selected` capped the ranked candidate list BEFORE filtering to
  predicted-eligible ones** (`kvcot.cli._write_fixed_trace_selection`) — a
  real bug: if the first `--max-selected` candidates by `source_row_index`
  happened to be predicted-ineligible, the selection could come back with
  `n_selected=0` even though plenty of eligible candidates existed later in
  the ranking. Fixed: candidates are ranked (uncapped) first, filtered to
  `predicted_eligible`, and ONLY THEN capped. New `FixedTraceSettings.
  max_selected_examples` lets the cap be pre-registered in the stage config
  itself, not chosen after seeing the selection; `--max-selected` on the
  CLI still overrides it when explicitly given.
- **The documented accuracy-gate run order was impossible.**
  `docs/GPU_VALIDATION_PLAN.md` said to run `analyze-fixed-trace` right
  after natural R-KV generation, before any fixed-trace replay, to get the
  pilot accuracy screen — but `analyze-fixed-trace`/`run_fixed_trace_
  analysis` unconditionally reads the fixed-trace probe files too, which do
  not exist yet at that point in the order; the command would simply crash.
  Separately, `build_accuracy_screen`'s intersection-only pairing means one
  matching `(source_row_index, global_seed)` pair could report
  `pilot_accuracy_plausible: True` even while 49 of 50 R-KV records were
  never generated at all. Fixed with a new CPU-only command, **`kvcot
  check-fixed-trace-accuracy`**, that reads ONLY the natural `full.jsonl`/
  `{replay_condition}.jsonl` files (never fixed-trace probes) and wraps the
  pairing in `kvcot.analysis.fixed_trace.build_strict_accuracy_gate`, which
  requires (all independently, never silently skipped): an exact expected
  record count on both sides, no duplicate `(source_row_index, global_seed)`
  keys, an IDENTICAL key set between the two files (not just a
  large-enough intersection), every record schema-valid, one shared
  `(config_sha256, upstream_rkv_commit, model_revision, tokenizer_revision)`
  identity matching the current invocation's own config/lock, consistent
  `condition` fields, and `pilot_accuracy_plausible`. Writes `results/
  decisions/{stage_name}_accuracy_gate.json`, exits nonzero on any failure.
  `build_accuracy_screen` itself is unchanged (still an honest, low-level
  intersection-based pairing helper) — the strict gate wraps it, rather
  than changing what it means.
- **Screen-level validity still gated on `actual_compression_rate`, not
  `meaningful_compression_rate`, even under `require_meaningful_compression
  =True`.** The per-PAIR eligibility gate (`FixedTraceEligibility.
  rkv_meaningful_compression_at_f1`, 2026-07-17) worked correctly, but
  `build_screen_validity` kept checking the old any-eviction rate at the
  SCREEN level regardless — a batch where every example evicted at least
  one token (`actual_compression_rate=1.0`) but almost none evicted enough
  to be "meaningful" could still report `screen_valid: True`. Fixed: when
  `require_meaningful_compression=True`, the screen gate now checks
  `meaningful_compression_rate` against new setting `FixedTraceSettings.
  min_meaningful_compression_rate` (default 0.7) INSTEAD of
  `actual_compression_rate`/`min_actual_compression_rate`; v2 stages
  (`require_meaningful_compression=False`, the default) are completely
  unaffected.
- **Analysis completeness guard added**: `run_fixed_trace_analysis` gained
  `selected_base_record_ids` (wired from `--selection-file`) — when given,
  `kvcot.analysis.fixed_trace._verify_selection_completeness` requires
  every selected example to have all 9 fractions recorded under BOTH
  replay policies, and raises `ValueError` (abort, never silent attrition)
  if any are missing — a partially-completed replay (e.g. 5 of 20 selected
  examples actually written) could otherwise still clear
  `min_eligible_examples` and produce a decision JSON as if the screen were
  actually complete. All curves/counts are also scoped to exactly the
  selected set once a selection is active (`n_shared == n_selected`
  becomes an enforced invariant, not just a hope).
- Still true from the 2026-07-17 entry: no GPU exists on this machine.
  `tests/integration/test_rkv_schedule_prediction_gpu.py` (new) and
  `tests/integration/test_frozen_probe_gpu.py` (new) implement the two
  mandatory GPU gates that entry promised but did not yet contain — both
  collect and auto-skip cleanly here (13 GPU tests skipped, up from 11) and
  must be run for real, and pass, before trusting either the schedule
  simulator or `frozen_at_cut` against a real model.

## 2026-07-17 — Fixed-trace protocol v3: frozen-cache causal probe, meaningful-compression gating, exact cache-schedule simulator (secondary, additive; no frozen §1/§4/§8/§9 value changed)

**Protocol v2's real GPU screen (`configs/early_gap_v2_b128.yaml`, committed
result `results/decisions/early_gap_v2_b128_fixed_trace.json`) produced
`n_eligible=3` against a `min_eligible_examples=5` floor and
`mean_f1_rkv_retention_ratio=0.7456` against a `max_mean_f1_retention_ratio
=0.7` ceiling — `screen_valid: false`, `hypothesis_status: "not_tested"`, as
designed.** This is a valid negative screening outcome, not a bug in the
correctness machinery (patched-noop parity, no-state-leak, and replay
identity GPU gates all passed; every one of the 180 fixed-trace probes
produced a valid boxed answer; cross-file/schema identity checks held). Two
mechanisms specifically explain why it came back invalid, diagnosed from the
raw per-example records already in the committed decision JSON:

1. **`n_actual_compression_active=10/10` overstated real compression.**
   `rkv_actual_compression_at_f1` (added in v2) only requires the physical
   cache to be smaller than the FullKV-equivalent slot count by ANY amount —
   `source_row_index=148` counts as "compression active" at
   `rkv_f1_retention_ratio=0.9959`, i.e. one token evicted out of ~240. Six
   of the ten shared examples sit at retention ≥0.94; the schedule mechanics
   audited in `docs/UPSTREAM_AUDIT.md` H4 (periodic `divide_length=128`
   compaction checks, real eviction gated on `kv_cache_len >= budget`) mean a
   trace can cross the schedule boundary long before it has accumulated
   enough tokens for the eviction itself to matter. An "any eviction"
   boolean cannot distinguish this from a trace that lost most of its cache.
2. **The v2 probe protocol lets R-KV compact again while writing its own
   answer.** Five of the ten shared examples (`source_row_index` 30, 176,
   262, 271, 307) failed eligibility via `rkv_evicted_during_answer_probe` —
   a real compaction event fired during the teacher-forced closing-marker/
   suffix/greedy-answer tokens fed by `branch_and_probe`, after the snapshot
   the probe was supposed to measure. This is a protocol confound, not
   evidence about the hypothesis: it measures a cache state that moved again
   after the reasoning cut, not the cut itself.

Both are addressed here, **before any further GPU spend**, without touching
the frozen v2 output (`configs/early_gap_v2_b128.yaml`,
`results/decisions/early_gap_v2_b128_fixed_trace.json`, and the raw v2
probes are never modified, resumed into, or reinterpreted under v3 —
protocol v2 is an archived, invalid-but-real result, not a draft).

- **`kvcot.analysis.rkv_schedule`** (new, torch-free): an exact CPU
  simulator of the upstream schedule/trigger mechanics audited in
  `docs/UPSTREAM_AUDIT.md` H4 and §3.1/3.3 (`self.length` cumulative
  counter incremented once per top-level forward call; the compression flag
  for call N+1 is `self.length % divide_length == 0` computed at the end of
  call N; the very first forward call in a fresh process always attempts
  eviction regardless of schedule, since `compression` initializes to
  `None`; an attempted eviction is a no-op whenever the current physical
  cache length is still below `budget`). Used to *predict*, from a FullKV
  base record alone (prompt length + generated-token count), the physical
  R-KV cache length and retention ratio at every probe fraction before
  spending any GPU time — this is what makes deterministic, outcome-blind
  trace selection (below) possible on this CPU-only machine.
- **`kvcot.generation.replay.branch_and_probe`** gained
  `probe_cache_mode: Literal["native", "frozen_at_cut"] = "native"`
  (default preserves exact v2/existing behavior and all existing tests
  unchanged). `"frozen_at_cut"` forces `compression=False` on every R-KV
  layer before every teacher-forced/generated token fed during the probe
  (close marker, control suffix, and each greedily generated answer token) —
  addresses failure mode 2 above by construction, not by post-hoc filtering:
  the cache snapshot the probe branched from cannot be disturbed by the
  model's own answer-writing. A fresh per-layer cache-length assertion
  (`final_length == snapshot_length + tokens_fed`, checked every layer) now
  raises loudly if this is ever violated, rather than silently producing an
  ineligible pair after the GPU time to produce it is already spent.
- **`kvcot.config.FixedTraceSettings`** gained (all with defaults that leave
  `configs/early_gap_v2_b128.yaml` byte-behavior-identical):
  `meaningful_retention_ceiling` (default 0.7), `require_meaningful_compression`
  (default `False`), `min_meaningfully_compressed_scored_fractions` (default
  0), `probe_cache_mode` (default `"native"`), `max_pilot_accuracy_drop`
  (default 0.10).
- **`kvcot.analysis.fixed_trace`**: new `rkv_meaningful_compression_at_f1`
  eligibility field — `rkv_f1_retention_ratio <= meaningful_retention_ceiling`
  — additive alongside (never replacing) the existing
  `rkv_actual_compression_at_f1` "any eviction" diagnostic; only enforced as
  an eligibility gate when `require_meaningful_compression=True` (v2 configs
  never set this, so v2's frozen eligibility semantics and its archived
  decision JSON are unaffected). New CPSS (Compression-Active
  Prefix-Sufficiency Sensitivity) metric — `compute_cpss`/`compute_delta_cpss`
  — restricted to the subset of the 7 scored fractions where
  `rkv_retention_at_fraction <= meaningful_retention_ceiling` (requires at
  least `min_compressed_scored_fractions_for_cpss`, default 2, else `None`) —
  a DIFFERENT metric from PSS/Delta_PSS, additive, never pooled with it. New
  per-fraction `retention_summary_by_fraction` (count/mean/median/min/max/
  meaningful-compression-rate) and `compression_rate_by_fraction` in the
  decision JSON — this is what would have shown the sawtooth retention
  pattern (0.994 at f=0.125 down to 0.746 at f=1) without hand-decoding raw
  records. New `all_shared_full_curve`/`all_shared_rkv_curve` keys carry
  forward the EXACT existing `full_curve`/`rkv_curve` semantics (still
  present, unchanged, for v2 backward compatibility) under clearer names;
  new `*_eligible_only` curve variants are additive.
- **New natural-accuracy screen** (`build_accuracy_screen` in
  `kvcot.analysis.fixed_trace`, wired automatically into `kvcot
  analyze-fixed-trace`/`run_fixed_trace_analysis` — no separate CLI command,
  one artifact per stage rather than two): pairs natural (non-fixed-trace)
  `full.jsonl`/`rkv_b{budget}.jsonl` base accuracy on the SAME manifest rows
  — v2 never generated a natural R-KV b128 run, so it could not establish
  `pilot_accuracy_plausible` (deliberately not named `accuracy_neutral` —
  §8.5 of `CLAUDE.md`/the build brief already states the primary paired
  200-problem accuracy check, `kvcot.analysis.stats.paired_accuracy_diff`,
  is the only test allowed to claim distributional accuracy preservation;
  this is a small-n stop/continue gate only). Only attempted for stages that
  set `require_meaningful_compression: true` (the v2/v3 discriminator) —
  `screen_valid` in the v3 fixed-trace decision is `False` whenever this
  screen is missing (natural R-KV base file not found) or
  `accuracy_difference_rkv_minus_full < -max_pilot_accuracy_drop`.
- **`kvcot inspect-fixed-trace --write-selection`** (new flag): deterministic
  trace selection using ONLY the FullKV base file's own correctness/cap/
  think-parse validity plus `kvcot.analysis.rkv_schedule`'s predicted
  retention — never any fixed-trace probe answer, PSS, or CPSS value (a
  selection that could see outcomes would not be a pre-registered screen).
  Writes `results/selections/{stage_name}.json` with the config hash, base
  file SHA-256, per-candidate predicted retention at every fraction, and
  selected/rejected row indices with reasons.
- **`configs/early_gap_v3_b128.yaml`** (new stage, never a resumption of
  `early_gap_v2_b128.yaml`'s `output_dir`): `limit: 50` (natural generation
  only — selection then narrows to a smaller replayed set),
  `probe_cache_mode: frozen_at_cut`, `require_meaningful_compression: true`,
  `min_meaningfully_compressed_scored_fractions: 2`.
- **Schema bump 1.3.0 -> 1.4.0** (`kvcot.schemas`): `FixedTraceProbeRecord`
  gained `protocol_version` (`Literal["v2", "v3"]`, default `"v2"` so
  existing v2 records/fixtures continue to validate unchanged),
  `probe_cache_mode`, `meaningful_compression_at_cut`,
  `compressed_scored_fraction`. `SCHEMA_VERSION` bump means a stale
  1.3.0-shaped record fails validation outright under the new
  `Literal["1.4.0"]` — exactly the same discipline the 1.2.0->1.3.0 and
  1.1.0->1.2.0 bumps established; a v2 output directory must never be
  resumed under this schema.
- **Still open**: this is all CPU-buildable/testable infrastructure — no
  GPU exists on this machine (unchanged from every prior entry). The exact
  schedule simulator's predictions have not been cross-checked against real
  GPU-measured retention (the raw v2 fixed-trace probe JSONL is gitignored
  per `README.md`'s stated layout and was not preserved outside the GPU
  host), only against the audited mechanics and hand-derived unit cases —
  `tests/integration/test_rkv_schedule_prediction_gpu.py` (new, GPU-only)
  is the mandatory one-example gate before trusting it against real
  generations. Natural R-KV b128 generation on the 50-example manifest has
  not been run — the accuracy screen above cannot report a real number
  until it is. `docs/GPU_VALIDATION_PLAN.md` §(new) documents the required
  run order; do not skip the one-example frozen-probe/schedule-prediction
  gate before committing to the full 20-example replay.

## 2026-07-16 — Fixed-trace protocol v2, fourth pass: policy-role validation, resume identity gap (secondary, additive; no frozen §1/§4/§8/§9 value changed)

A fourth external review of the third-pass cross-file identity commit found
the identity checks it added still had two gaps, both in the same spirit
(catching a mislabeled/stale file before it silently corrupts a comparison):

- **A record's own `replay_policy_condition` was never checked against the
  file it was loaded from.** `_assert_consistent_identity`/
  `_assert_shared_trace_source` (prior entry) check `config_sha256`/
  `upstream_rkv_commit`/`model_revision`/`tokenizer_revision` and
  `trace_source_condition` agreement, but nothing checked that a probe
  file's records actually declare the replay policy the filename convention
  implies. A `full_on_full_fixed_trace_probes.jsonl` file whose records
  declare `replay_policy_condition="rkv_b128"` (e.g. from an accidental file
  swap or rename) was silently accepted, which can flip which curve gets
  called FullKV vs. R-KV. Fixed: `load_fixed_trace_records` now raises if any
  row's `replay_policy_condition` disagrees with the `replay_condition` it
  was called with. Also added: `_validate_base_records` now checks the
  canonical base file's own `condition` field against `trace_condition`, so
  a base file recorded under a different condition (e.g. an R-KV file
  passed in as the canonical trace by mistake) is rejected too.
- **`cmd_replay_fixed_trace`'s `--resume` identity check omitted
  `model_revision`/`tokenizer_revision`.** `FixedTraceProbeRecord` has
  carried both fields since the 1.2.0 -> 1.3.0 bump (prior entry)
  specifically so cross-file identity could be checked — but the
  `expected_identity` dict `cmd_replay_fixed_trace` builds for `--resume`
  still only carried `config_sha256`/`upstream_rkv_commit`, so resuming into
  a fixed-trace probe file recorded under a stale model/tokenizer revision
  was accepted at resume time. `run_fixed_trace_analysis` would eventually
  reject the resulting mixed-identity output directory, but only after
  wasting GPU time producing it. Fixed: `expected_identity` now also carries
  `model_revision`/`tokenizer_revision`, matching what `cmd_generate` already
  did.
- Also corrected two stale comments found during this pass: a
  `_validate_fixed_trace_probe_records` docstring still claimed
  `load_fixed_trace_records` "keeps the last" duplicate `(base_record_id,
  fraction)` row, when it has raised on that case since the prior entry; and
  a schema test's comment still described the fixed-trace suffix as always
  empty, a protocol-v1 behavior protocol v2 (two entries ago) replaced with a
  non-empty teacher-forced boxed-answer prefix. `docs/SCHEMA.md` and the
  archived protocol-v1 README were also still citing schema `"1.2.0"` after
  the prior entry's bump to `"1.3.0"`.
- 224 CPU tests pass (up from 217 — new coverage for both validation gaps
  plus the two stale-comment fixes). GPU test files still only collect and
  skip (11 skipped) — no GPU exists in this environment. Same still-open
  items as the prior entry (unrecoverable b256/b1024 raw data, unimplemented
  MATH-500 answer equivalence, no reviewed PR for this pass either).

## 2026-07-16 — Fixed-trace protocol v2, third pass: config-limit ignored, cross-file identity, duplicate rows (secondary, additive; no frozen §1/§4/§8/§9 value changed)

A third external review of the protocol-v2 hardening commit found two
blocking repository bugs (not scientific-mechanics bugs — the anchor
extraction, budget selection, and f=1-eviction fixes from the prior two
entries were all confirmed correct) that would have let a GPU rerun either
silently process the wrong number of examples or silently pair
inconsistent data:

- **`StageConfig.limit` was completely ignored.** `_load_manifest_filtered`
  only ever consulted `args.limit` (the CLI `--limit` flag); a stage
  config's own `limit:` (e.g. `early_gap_v2_b128.yaml`'s `limit: 10`
  against a 50-row manifest) had no effect at all. Every documented
  fixed-trace command in `docs/GPU_VALIDATION_PLAN.md` omits an explicit
  `--limit`, relying entirely on the config's declared limit — so the
  documented "ten-example screen" was actually running against all 50
  rows. Fixed: `effective_limit = args.limit if args.limit is not None
  else stage.limit`. Verified end-to-end with the real config
  (`--dry-run`, no `--limit`): `generate` now reports `rows: 10`,
  `replay-fixed-trace` reports `planned examples: 10` /
  `planned probe records: 90`, matching the documented n=10 exactly.
- **Cross-file identity was never checked.** `_validate_base_records`/
  `_validate_fixed_trace_probe_records` (added in the prior entry) each
  only verified ONE file's own internal consistency — nothing compared the
  canonical base file against either fixed-trace probe file, or either
  probe file against the other. A base file from one config/model/upstream
  pin could be silently paired against probe files from a different run.
  Fixed: `FixedTraceProbeRecord` gained `model_revision`/
  `tokenizer_revision` fields (schema bumped `1.2.0` -> `1.3.0`) so its
  identity is directly comparable to `BaseRunRecord`'s; a new
  `_assert_consistent_identity` cross-checks all three files' identities
  against each other, and `cmd_analyze_fixed_trace` now also passes the
  CURRENT invocation's own `(config_sha256, upstream_commit, model_revision,
  tokenizer_revision)` — computed from `args.config` and the freshly loaded
  lock, previously loaded and silently discarded — so stale data cannot be
  analyzed even if it happens to be internally self-consistent.
- **Duplicate `(base_record_id, fraction)` rows were silently overwritten.**
  `load_fixed_trace_records` now raises on a duplicate key instead of
  letting the later row win silently — such a duplicate can only arise
  from a corrupted, hand-edited, or improperly concatenated file (the
  writer itself already refuses a duplicate `record_id` within one run).
- **`require_boxed_extraction` was a dead config field.** Declared in every
  fixed-trace stage config but never read by any code (boxed extraction
  was always required unconditionally). Changed from a plain `bool` to a
  frozen `Literal[True]` — settable to `True` (or omitted), never silently
  disabled to `False` with no effect.
- Fixed a stale docstring still naming the retired
  `no_rkv_eviction_during_scored_probes` field (renamed
  `no_rkv_eviction_during_answer_probes` two entries ago) and corrected
  `docs/GPU_VALIDATION_PLAN.md`'s one-example-gate instructions, which
  implied `--limit 1` applies to `analyze-fixed-trace` too (it takes no
  such flag — it only reads what `replay-fixed-trace` already wrote).
- **Still open, unchanged**: no GPU exists in this environment to actually
  exercise any of this — 217 CPU tests pass (up from 203), GPU test files
  still only collect and skip. Raw b256/b1024 probe data remains
  unrecoverable through code (§ prior entry). MATH-500 answer equivalence
  remains unimplemented. This round also went directly to `main` without a
  reviewed PR, same as the prior two — flagged again here since external
  review has now raised it twice; a subsequent change may switch to a
  branch+PR flow if that continues to matter.

## 2026-07-16 — Fixed-trace protocol v2 hardening: f=1 eviction gap, budget too large, analysis-input validation (secondary, additive; no frozen §1/§4/§8/§9 value changed)

External review of the first protocol-v2 commit (`20e2ad6`, merged as
`b883fd3`) found the anchor-extraction fix correct but three remaining gaps
that would have let a GPU rerun waste money on a screen that still cannot
produce a valid result, plus one real eligibility bug. Fixed here, still
before any GPU spend:

- **b512/b1024 cannot compress on this manifest; b256 falls short too.**
  Recalculating from the real GPU data already collected
  (`logs/b512_accuracy_compaction.log`): observed prompt+think lengths on
  the `gsm8k_calibration_50` sample never exceed budget 512 or 1024 at all
  (`mean_final_retention_ratio: 0.98` — confirms this), and exceed budget
  256 on at most ~6/10 traces — structurally below
  `FixedTraceSettings.min_actual_compression_rate` (0.70), and even
  maximally aggressive compaction at 256 cannot bring mean retention under
  the 0.70 ceiling on traces this short. **Added
  `configs/early_gap_v2_b128.yaml`** (new `stage_name`/`output_dir`, never a
  resumption of an `early_gap_b*.yaml` directory) as the first budget with
  a realistic chance of clearing both thresholds on this manifest.
  Thresholds themselves were **not** weakened to make an existing budget
  pass — per the review's explicit instruction, a budget too large for the
  data is fixed by picking a smaller budget (or longer traces), not by
  lowering the bar.
- **`kvcot inspect-fixed-trace` strengthened** (`src/kvcot/cli.py`) with two
  new arithmetic-only stop conditions, on top of the existing "nothing
  exceeds the budget" check: (1) `fraction_of_traces_longer_than_budget` is
  an upper bound on the achievable `actual_compression_rate` — if that
  bound is already below `min_actual_compression_rate`, the eligibility
  gate is mathematically unreachable at this budget; (2)
  `mean_optimistic_retention` (`budget/length` per trace, the most
  aggressive possible compaction) is a lower bound on achievable mean
  retention — if even that best case exceeds
  `max_mean_f1_retention_ratio`, no real run can pass either. Both checks
  only run when the stage config declares `fixed_trace:` settings.
- **Eligibility gap: answer-time eviction was never checked for the f=1
  anchor itself** (`src/kvcot/analysis/fixed_trace.py`,
  `FixedTraceEligibility`) — the check only scanned the 7 scored fractions.
  Every scored fraction's match is scored against the f=1 anchor's own
  answer, so an eviction while the anchor was writing ITS OWN answer is
  exactly as disqualifying as one on a scored fraction; a synthetic
  f=1-only-eviction case previously came back eligible with zero failure
  reasons. `no_rkv_eviction_during_answer_probes` (renamed from
  `no_rkv_eviction_during_scored_probes`) now covers
  `PROBE_FRACTIONS_SCORED + (1.0,)`. Regression test added:
  `test_f1_only_answer_time_eviction_makes_pair_ineligible`.
- **`run_fixed_trace_analysis` now validates every input record**
  (`src/kvcot/analysis/fixed_trace.py`, `_validate_base_records`/
  `_validate_fixed_trace_probe_records`) against `BaseRunRecord`/
  `FixedTraceProbeRecord` (rejecting a stale `schema_version` outright, via
  the `Literal["1.2.0"]` field) and checks every record shares one coherent
  `(config_sha256, upstream_rkv_commit[, model_revision, tokenizer_revision])`
  identity, before any pairing/scoring happens. Previously the analysis read
  JSONL as plain dicts with no schema check at load time, so a protocol-v1
  directory (or one mixing two different runs) could be silently
  "analyzed" as if it were valid current input.
- **Archived stale protocol-v1 decision JSONs**
  (`results/decisions/early_gap_b{256,512,1024}_fixed_trace.json`,
  `early_gap_b512_accuracy_compaction.json`) to
  `results/decisions/archive/protocol_v1_2026-07-16/` (with a README
  explaining why) rather than leaving them under names that look like
  current results. These files are `schema_version "1.1.0"`, `git_dirty:
  true`, and — per the diagnosis above — describe a screen that produced
  zero eligible examples; they must never be read as evidence for or
  against G1. Corresponding raw probe data for b256/b1024 (only b512 was
  ever actually generated) does not exist and cannot be reconstructed —
  a fresh GPU run under `early_gap_v2_b128.yaml` is required.
- **Still open, unchanged from the prior entry**: MATH-500 answer
  equivalence is still not implemented (plain string equality only) — do
  not switch to MATH-500 traces until that lands, tested. Protocol v2 has
  not yet been exercised on a real GPU — CPU tests (203 passing) and
  `python -m py_compile` are the only checks possible on this build
  machine; the one-example GPU gate in `docs/GPU_VALIDATION_PLAN.md` is
  still required before any 10-example rerun.

## 2026-07-16 — Fixed-trace protocol v2: boxed-answer prefix, realized-compression gating (secondary, additive; no frozen §1/§4/§8/§9 value changed)

**Protocol v1 produced no scientific result.** The first fixed-trace GPU
screen (b512, seed=42, n=10) ran end-to-end cleanly — sampled base accuracy
9/10 under both conditions, zero cap hits on generation — but `n_eligible =
0` at every budget tested, for two independent, diagnosable reasons found by
decoding the raw probe text:

1. **The f=1 anchor was garbage on every example.** The fixed-trace suffix
   was deliberately empty (`FIXED_TRACE_SUFFIX_TEXT = ""`, to avoid cueing
   recomputation), and probe decoding used the frozen 48-token budget
   (`configs/lock.yaml`'s `probes.max_new_tokens`). R1-Distill's answer mode
   is a verbose structured write-up that essentially never reaches a
   `\boxed{...}` (or even an explicit `Final answer:`) within 48 tokens, so
   extraction fell through to the conservative final-number fallback tier on
   nearly every probe and grabbed an incidental mid-sentence number as the
   "anchor" — noise, not an answer. Every reported PSS/curve value from that
   screen was contaminated (fallback-extracted noise compared against
   fallback-extracted noise) and must not be read as evidence in either
   direction.
2. **Eligibility gated on a recorded compaction EVENT COUNT, not realized
   compression.** At the exact budget boundary R-KV can record a compaction
   event that evicts zero tokens (`kvcot.generation.replay`'s documented
   boundary case) — `rkv_had_replay_compaction` (`count > 0`) let such pairs
   through as "eligible" even though the physical cache never actually
   shrank.

Neither failure says anything about the underlying hypothesis (G1) — this
screen tested the elicitation machinery and found it broken, not the
compression question. All kill criteria from the earlier design chats remain
live and untriggered; the infra (gates, replay, schemas, eligibility logic)
is fully reusable once these two defects are fixed. Fixed here, before any
rerun:

- **`src/kvcot/probes/templates.py`**: `FIXED_TRACE_SUFFIX_TEXT` changed from
  `""` to `"\n\nFinal answer: \\boxed{"` — a teacher-forced FORMAT prefix
  (identical across conditions, fed as plain tokens exactly like the closing
  `</think>` marker), never a natural-language recomputation instruction
  ("solve again"/"recalculate"/"use the question"/"explain your answer" are
  all still forbidden, per the module's own documented rationale).
- **`src/kvcot/config.py`**: new `FixedTraceSettings` (own
  `probe_max_new_tokens` default 64, `min_eligible_examples`,
  `min_actual_compression_rate`, `max_mean_f1_retention_ratio`), attached as
  `StageConfig.fixed_trace`, required (not optional) by
  `cmd_replay_fixed_trace`/`cmd_analyze_fixed_trace` — deliberately
  **separate** from the frozen `configs/lock.yaml` `probes.max_new_tokens:
  48`, so a fixed-trace-motivated change can never silently alter the frozen
  primary EAS experiment. `configs/early_gap_b{256,512,1024}.yaml` each gained
  a `fixed_trace:` block.
- **`src/kvcot/utils/answers.py`**: `has_complete_boxed_answer` (stop
  predicate for probe decoding) and `answers_match_or_none` (three-valued
  match — `None` means "could not extract," `False` means "a valid but
  different answer"; the two must never be conflated, since coercing the
  first into the second hides extraction breakage inside what looks like a
  normal disagreement rate). `answers_match` (the frozen primary path's
  two-valued match) is unchanged.
- **`src/kvcot/generation/replay.py`**: `branch_and_probe` accepts an
  optional `stop_predicate` (checked after every generated token, in
  addition to EOS) so fixed-trace decoding halts the instant a box closes —
  never used by the frozen primary replay-probe path. `ProbeResult` gained
  `stop_reason`, `final_absolute_position`, `final_cache_lengths_per_layer`
  so callers can detect an eviction that happened *while writing the answer*
  itself, not just at the reasoning cut.
- **`src/kvcot/cli.py`** (`cmd_replay_fixed_trace`): extraction now runs over
  the reconstructed prefix+generated text (`probe_extraction_text`), never
  generated tokens alone; the stop predicate is wired in; realized retention
  and actual-compression are measured at every snapshot
  (`replay_retention_at_cut`, `actual_compression_at_cut` — physical cache
  length vs. FullKV-equivalent slots, never the configured budget);
  answer-time eviction is detected (`probe_actual_eviction_during_answer`).
  New CPU-only `kvcot inspect-fixed-trace` command: reports think-span/
  prompt+think-span length statistics against the configured R-KV budget and
  refuses to proceed if no trace in the file is even longer than the budget
  (this cannot prove compression will happen, only rule out the case where
  it definitely cannot) — run this before spending GPU time on
  `replay-fixed-trace`.
- **`src/kvcot/schemas.py`**: `FixedTraceProbeRecord` gained
  `probe_extraction_text`, `probe_stop_reason`, `probe_cap_hit`,
  `replay_retention_at_cut`, `actual_compression_at_cut`,
  `probe_cache_length_final_per_layer`, `probe_actual_eviction_during_answer`.
  `SCHEMA_VERSION` bumped `1.1.0` -> `1.2.0`, and every record's
  `schema_version` is now `Literal["1.2.0"]` (not just a string default) —
  a stale-schema record now fails Pydantic validation outright instead of
  being silently accepted. **Old protocol-v1 output directories must not be
  resumed under protocol v2** — start a fresh `output_dir`.
- **`src/kvcot/analysis/fixed_trace.py`**: eligibility (`FixedTraceEligibility`)
  reworked around realized compression (`rkv_actual_compression_at_f1`,
  `no_rkv_eviction_during_scored_probes`) instead of a recorded event count,
  plus new gates on each side's own f=1 anchor being a `"boxed"` extraction
  (`full_f1_anchor_boxed`/`rkv_f1_anchor_boxed` — a fallback anchor is never
  accepted) and on the canonical trace's own base answer being correct
  (`canonical_trace_base_correct`). PSS is `None` (never `0.0`) whenever a
  side's own anchor is invalid/fallback or any scored fraction failed to
  extract; `Delta_PSS` is additionally `None` whenever the pair fails full
  eligibility (in particular: no actual R-KV compression, or an answer-time
  eviction) even if both PSS values are individually defined. Descriptive
  curves (`fixed_trace_curve_by_fraction`) now return `None` for a fraction
  with zero valid measurements, never `0.0` — the two are different claims.
  New screen-level validity gate (`build_screen_validity`,
  `build_fixed_trace_decision`): `screen_valid` requires enough eligible
  examples, a high enough realized-compression rate, and low enough realized
  retention (all from `FixedTraceSettings`); `hypothesis_status` is
  `"not_tested"` when any of those fail, and even when the screen is valid
  this module never reports "positive"/"negative"/"gap exists"/"gap does not
  exist" — only descriptive counts, per its existing kill/continue-screen
  discipline.
- **GPU test process isolation** (`tests/integration/test_replay_gpu.py`,
  `test_probe_stability_gpu.py`): every patched-R-KV test and the FullKV
  identity test now run inside their own `multiprocessing.get_context
  ("spawn")` subprocess (never `fork`), mirroring the pattern already used in
  `test_patched_noop_parity_gpu.py`. Previously, several of these tests ran
  directly in the shared pytest process — since the R-KV monkeypatch on
  `transformers.models.qwen2` is process-global with no per-instance undo
  (`docs/UPSTREAM_AUDIT.md` H1), and `kvcot.generation.state.
  declare_process_mode` already refuses a second, conflicting mode in one
  process, mixing stock/patched tests (or two different R-KV configs) in one
  process was unsafe or outright broken (`reset_active_mode_for_testing()`
  only clears kvcot's own tracking variable, not the underlying monkeypatch).
  `_load_rkv_model` now calls `declare_process_mode("patched")` before
  `replace_qwen2(...)`, matching every real loader
  (`kvcot.generation.policies._PatchedPolicyBase.load`).
- **Scope note**: MATH-500 support (verified answer-equivalence for
  fractions/radicals/decimals, distinct from GSM8K's plain string
  equality) and longer-trace budget calibration are deliberately **not**
  included in this entry — planned as a follow-up once the corrected
  protocol passes its one-example GPU gate on GSM8K, per the original
  design's stated validation order.

## 2026-07-16 — Fixed-trace prefix-sufficiency screen (secondary, additive; no frozen §1/§4/§8/§9 value changed)

Added on branch `early-gap-fixed-trace`, still pre-GPU. This is an
**addition alongside** the frozen `replay-probe`/EAS/Delta_EAS pipeline, not
a replacement or a modification of it — every §1/§4/§8/§9 frozen value in
`CLAUDE.md`/`configs/lock.yaml` is unchanged, `replay-probe` itself is
byte-for-byte unmodified, and the frozen research question (§1) remains this
repository's headline claim.

**Motivation.** `replay-probe`/EAS scores each condition's probe answer
against that SAME condition's own sampled base answer (§8). FullKV and R-KV
each generate their own natural trace, so a Delta_EAS effect could in
principle be partly attributable to the traces themselves differing between
conditions, not only to the cache policy — the frozen design already
controls for this at the level that matters for the primary claim (§8.5's
both-correct-and-compression-active subset conditions on correctness per
problem), but it does not isolate the cache-policy question in the most
literal possible way: replaying one identical token sequence under two
different policies. This addition does exactly that, as a secondary,
smaller-sample screen — a kill/continue check, not a second primary result.

- **New commands**: `kvcot replay-fixed-trace` and `kvcot analyze-fixed-trace`
  (`src/kvcot/cli.py`). `replay-fixed-trace` reads its canonical token
  sequence from one condition's base file (`--trace-condition`, default
  `full`) but loads the model and applies cache-policy replay under a
  possibly-different condition (`--replay-condition`) — both replay
  policies teacher-force identical prompt and reasoning tokens; only the
  cache policy varies. `replay-probe` is untouched and remains the on-policy
  diagnostic; both commands can coexist against the same stage's output
  directory.
- **New metric**: Prefix-Sufficiency Sensitivity (PSS) / Delta_PSS
  (`src/kvcot/analysis/fixed_trace.py`) — mean mismatch rate against each
  replay policy's own greedy f=1 answer (never the trace source's sampled
  natural answer, which would reintroduce the sampled-vs-greedy confound
  §7 of `docs/EXPERIMENT.md` already documents for the original f=1
  stability probe). `Delta_PSS = PSS_full - PSS_rkv`, same subtraction
  order and sign meaning as `Delta_EAS` (positive => R-KV less sensitive to
  truncation). This is a **different metric** from EAS/Delta_EAS — never
  pool or directly compare the two. No p-value or confidence interval is
  computed at this sample size (`configs/early_gap_b512.yaml`: n=10,
  one seed) — descriptive counts only.
- **New schema**: `FixedTraceProbeRecord` (`src/kvcot/schemas.py`),
  distinguishing `trace_source_condition` from `replay_policy_condition` —
  a distinction `ProbeRunRecord` has no field for, since it never needed
  one. `SCHEMA_VERSION` bumped `1.0.0` -> `1.1.0`. `kvcot validate-run` now
  dispatches on each record's own `record_type` field instead of filename
  pattern-matching (`_schema_for_record`) — a `..._fixed_trace_probes.jsonl`
  file still ends in `_probes.jsonl`, so filename-based dispatch would have
  silently misvalidated it against `ProbeRunRecord`.
- **New configs**: `configs/early_gap_b512.yaml` (primary, 10-example,
  seed=42 screen) plus `early_gap_b256.yaml`/`early_gap_b1024.yaml`
  (budget-escalation fallbacks — step down only if compression rarely
  fires at 512, step up only if it fires but breaks accuracy, never step
  down after breaking accuracy).
- **Scope note**: an earlier draft of this change also proposed a
  mistake-insertion probe (corrupting a verified intermediate arithmetic
  step and testing whether the answer changes). That is **not implemented**
  here — `CLAUDE.md` §1 and `README.md`'s Scope section both explicitly and
  repeatedly list "mistake insertion" as out of scope for this repository,
  and the technique is a standard chain-of-thought-faithfulness probe from
  the literature, which is exactly the category of conclusion §1's
  "Forbidden conclusions" clause exists to rule out. Implementing it would
  require un-freezing that boundary first, with its own dated entry here —
  deliberately deferred rather than done silently alongside an otherwise
  in-scope addition.

## 2026-07-15 — Second pre-GPU audit: orchestration/pipeline completeness (no frozen §4 setting changed)

A second audit found the orchestration layer (CLI commands and the shell
scripts driving them) incomplete in ways that would have made the frozen
`scripts/run_stage1b.sh`/`run_stage2.sh` fail outright or silently skip
work, independent of the generation/replay correctness fixes above.

- **`kvcot calibrate-budget` was a stub** that always printed "no results
  exist" and returned failure. Implemented for real: reads each candidate
  budget's `results/decisions/stage1b_budget_<N>.json` (now actually written
  by a new `cmd_analyze` branch for `stage1b_budget_*` stages, using only
  `generate` output — Stage 1B's calibration decision never needed probes),
  reports the smallest budget passing both gates to
  `results/decisions/stage1b_recommendation.json`, and deliberately does NOT
  auto-write `configs/selected_operating_point.yaml` (§10: that stays a
  manual, reviewed step).
- **`replay-probe --condition rkv_selected` was broken** — `generate`
  resolved the placeholder to `rkv_b{budget}`, but `replay-probe` never did,
  so it looked for a nonexistent `rkv_selected.jsonl` and passed the literal
  placeholder to `build_policy()`. Both commands now share one
  `_resolve_condition` helper so this can't drift again.
- **Compaction events were still recorded once per R-KV layer inside
  `kvcot.generation.replay`'s `CompactionTracker`** (`compaction.note_event()`
  was called inside the per-layer sync loop) — the same "events x n_layers"
  inflation the first audit fixed in `cli.py`/`decode.py`, still present in
  the replay path itself. `_sync_layer_after_call` now only reports whether
  ITS layer fired; a new `_note_event_once` cross-checks all R-KV layers
  agree and records the event exactly once.
- **`--resume` never actually checked identity.** `kvcot.utils.io`'s own
  docstring promises "schema-valid completed records with matching
  config/model/upstream hashes"; the real logic only checked `record_id`
  membership. Added `_verify_resumable_record_ids` (schema validation +
  config/model/tokenizer/upstream-commit match, dotted-path comparison) and
  wired it into both `generate` and `replay-probe`'s resume paths — a
  mismatch now refuses loudly with a clear diagnostic instead of silently
  mixing identities in one output file.
- **Question hashes were computed but never checked.** `cmd_generate` now
  re-hashes each manifest row's question text and compares it to the
  manifest's own recorded `question_hash` before generating anything against
  it (catches a corrupted/hand-edited manifest one layer earlier, §5).
- **`RetentionSummary` was defined but never populated.** `cmd_generate` now
  measures it at end of each R-KV/patched-noop base generation from data
  already computed in that command (physical cache lengths, final absolute
  position) — no extra GPU passes. `ProvenanceRetentionSummary` (prompt/
  reasoning-token retention) and `BaseRunRecord.replay_state_hash` remain
  unpopulated — computing them without extra GPU cost would require
  restructuring `generate_base`'s hot decode loop to track full per-KV-head
  provenance, which risked meaningfully increasing Stage 2 wall-clock; out
  of scope for this pass, flagged here rather than silently left as a gap.
- **The three snapshot hashes on `ProbeRunRecord` hashed proxies, not
  content** (`snapshot_cache_hash` hashed only cache-length shapes;
  `snapshot_provenance_hash` hashed only the event-step list;
  `snapshot_state_hash` hashed two integers) — none would actually detect a
  divergence in the data they're named after. `replay-probe` now hashes the
  real K/V tensor bytes, the real per-layer/per-KV-head absolute source
  positions, and the real scheduling/bookkeeping state respectively.
- **`RunManifest` was imported but never constructed**; `kvcot.analysis.plots`
  functions existed but were never called; the stage0_smoke.yaml-advertised
  "throughput measurement + Stage 2 wall-clock extrapolation" didn't exist.
  `generate`/`replay-probe` now write a `RunManifest` per invocation
  (`{condition}_generate_manifest.json` / `{condition}_replay_probe_manifest.json`);
  `analyze` now writes `results/figures/agreement_curve.png` and
  `delta_eas_distribution.png` for stage2-shaped stages, and a rough
  throughput/wall-clock extrapolation decision JSON for `stage0_smoke`.
  `plot_realized_retention` remains unwired — no per-snapshot retention data
  source exists without the schema change noted above.
- **`Makefile`'s `dry-run` target used `--condition rkv`**, which is not a
  condition any stage config defines (`stage0_smoke.yaml` defines `rkv_b96`)
  — fixed to `rkv_b96`.

## 2026-07-15 — Pre-GPU correctness fixes (no frozen §4 setting changed)

Bug fixes found in a pre-run audit against the pinned upstream + transformers
4.55.4 semantics. None of these alters a `configs/lock.yaml` frozen setting;
they fix defects that would otherwise crash the probe stage or corrupt
provenance. Recorded here for traceability.

- **Blocker 1 — probe branch from an empty cache.** `restore_snapshot`
  item-assigned into `cache.key_cache[i]` on a freshly constructed
  `DynamicCache()`. On transformers 4.55.4 `key_cache` is a deprecated
  `KeyValuesWrapper` property whose `__setitem__` does `setattr(layers[idx],…)`
  with no growth, and a bare `DynamicCache()` pre-creates only ONE layer → the
  probe stage raised `IndexError` at layer 1. Fixed by populating the fresh
  cache through the public `cache.update(...)` path (`_populate_fresh_cache`).
- **Blocker 2 — inflated compaction count.** Base generation counted
  `events × n_layers` and stored a per-layer `[0,1,2,…]` enumeration in
  `compaction_event_steps`. `generate_base` now tracks true events at their
  absolute positions (one count, assert all R-KV layers agree). GPU test's
  `n_compactions` and the `>=2` hard gate corrected to count events, not
  events×layers.
- **#3 — cut-position arithmetic** in `replay-probe` dropped `think_start_index`
  (masked only by the pre-opened `<think>` template); fixed to match the
  documented replay contract, `cut_index` recomputed accordingly.
- **#4 — replay EOS asymmetry**: replay now stops once every requested snapshot
  is captured (all ≤ think_end), so it no longer feeds the trailing answer/EOS
  the base run never fed.
- **#5 — exact-budget assert crash**: `kv_cache_len == budget` records a
  zero-eviction compaction event; the equality assert in `_sync_layer_after_call`
  now treats bookkeeping growth as the event ground-truth and only raises on the
  genuinely-impossible reverse (cache shrank with no event).
- **#6** deleted dead `generate_probe_answer`. **#7** Stage-2 `--dry-run` now
  prints a clean prerequisite error instead of a traceback; base records now
  carry `dataset_config/revision/fingerprint` and the full R-KV `method_config`.
- **Analysis wiring**: `kvcot analyze` now actually computes the Stage-2 primary
  result (EAS → Delta_EAS → Wilcoxon/bootstrap/accuracy + attrition funnel) via
  the new `kvcot.analysis.pipeline`, instead of importing those helpers and
  never calling them. Stage 1A measurability now counts real answer-changes.

## 2026-07-15 — Initial build

- Repository built from scratch per the original build brief, on a
  CPU-only machine (no GPU, no model weights downloaded).
- Upstream R-KV pinned at commit `45eaa7d69d20b7388321f077020a610d9afb65bd`
  (verified to exist before use — `docs/UPSTREAM_AUDIT.md` §0).
- Model/tokenizer revision pinned at
  `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` (resolved via HF metadata API,
  no weights).
- All frozen settings in `configs/lock.yaml` set to their brief-specified
  values for the first time — nothing to diff against, so no prior value is
  listed.
- Four GSM8K manifests (smoke=20, calibration=50, main=200, disjoint) and
  one MATH-500 backup manifest (100 rows, levels 3-5) frozen with real
  network access, seed=13.
- No GPU code executed; no Stage 0-2 run.
