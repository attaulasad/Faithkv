# B1 independent-audit repair

This document supersedes the "READY FOR INDEPENDENT AUDIT" verdict in
`docs/B1_FINAL_CPU_CLOSURE.md` for the current status of this branch. That
verdict was found **incomplete** by a subsequent independent audit (this
document's task brief). The historical body of `B1_FINAL_CPU_CLOSURE.md`
and `docs/B1_FINAL_REPAIR_LEDGER.md` remain preserved as historical record;
this document is authoritative for current status.

## 0. Starting state (recorded before any edit)

```
branch:            research/b1b-r4-final-b2a-closure
HEAD:               7ef13ae566e7c3e699e5143405baf76a81078edf
origin branch:      7ef13ae566e7c3e699e5143405baf76a81078edf (identical)
working tree:       clean
7ef13ae ancestor:   confirmed (git merge-base --is-ancestor ... HEAD)
3c853cf ancestor:   confirmed (git merge-base --is-ancestor ... HEAD)
submodule:          -45eaa7d69d20b7388321f077020a610d9afb65bd third_party/R-KV
                    (uninitialized/not checked out on this machine, per
                    CLAUDE.md's sparse-checkout note -- unchanged by this pass)
```

Both required commits (`7ef13ae...`, `3c853cf...`) were confirmed ancestors
of `HEAD` before any edit. No newer commits existed on the branch or
origin beyond `7ef13ae...` at session start. The working tree was clean.

## 1. Evidence-based findings table

Every finding below was confirmed against the actual current code (not
assumed from the task brief) before any repair began.

| ID | Claim | Code evidence (pre-repair) | Status | Risk | Repair |
|----|-------|---------------------------|--------|------|--------|
| H1.1 | Worker failure envelope discards all partial evidence | `b2a_worker_entry.py:123-124`: `partial_measurements=None, determinism_policy=None` unconditionally on any exception | **CONFIRMED** | High — a real CUDA OOM mid-run would leave zero forensic evidence | Fixed: `worker_partial_evidence.py` (new), worker bodies wrapped, envelope threading |
| H1.2/H1.3 | No typed partial-evidence structure; worker bodies can't report structured failure | `run_fullkv_worker`/`run_rkv_worker` had zero `try/except` around their ~280-line bodies | **CONFIRMED** | High | Fixed: bodies wrapped in `try/except`, `WorkerBodyFailure` raised with `PartialWorkerEvidence` |
| H1.5 | No coordinator-authored termination record on timeout | `run_both_workers_via_subprocess`'s `preserve_timeout_logs` wrote only stdout/stderr logs | **CONFIRMED** | Medium | Fixed: `_write_termination_record` writes `termination.json` on timeout and nonzero-exit |
| H1 (orchestrator) | Pass-2 execution exceptions and per-pair-loop exceptions propagate bare, losing all prior pairs | `orchestrator.run_example`: no `try/except` around `run_pass2_capture` call or the per-pair loop body | **CONFIRMED** | High | Fixed: both wrapped, `ExampleResult.aborted`/`abort_failure_type`/`abort_is_oom` added |
| H1 (Pass 1) | Pass-1 exceptions silently discarded (bare `except Exception:` with no `as exc`) | `orchestrator.py` (pre-repair) mapped ANY Pass-1 exception to `STAGE_NATURAL_RUN_INVALID` with zero diagnostic info | **CONFIRMED** (found during this pass's own hostile audit, not in the original brief's explicit list) | Medium | Fixed: same `aborted`/`abort_failure_type`/`abort_is_oom` fields populated; stage name unchanged (no consumer regression) |
| H2.1/H2.2 | `capture_and_parity` phase measures the wrong operation | `b2a_workers.py`: `timer.measure("capture_and_parity", lambda: compare_call_boundary_traces(...))`, called AFTER `run_example` returns | **CONFIRMED, exact** | Medium — misleads anyone reading the timing evidence about when capture/parity work happens | Fixed: renamed to `call_trace_comparison`; real capture/parity work already timed under `rkv_pass2_prefill`/`rkv_pass2_decode`/`snapshot_creation` (unchanged, already real) |
| H2.4 | Startup/load runtime projection undercounts one-time setup | `b2a_execute.py`: summed only `{role}_worker_startup` + `model_load` | **CONFIRMED, exact** | Medium | Fixed: sums all 5 already-real, non-overlapping, sequential phases (`{role}_worker_startup`, `snapshot_tokenizer_resolution`, `tokenizer_load`, `model_load`, `post_load_validation`) |
| H2.5 | No coordinator-observed process-launch timing | Not measured anywhere | **CONFIRMED** | Low-medium (diagnostic only) | Fixed: `WorkerCoordinationResult.coordinator_observed_process_seconds`, exported as a separate `process_overhead_diagnostic` (never summed into the projection) |
| H2.6 | Pair timing source not synchronized/conservative | `build_runtime_projection` already required exactly 12 durations and used `max()` | **ALREADY FIXED** (false alarm re-derived from the brief's assumed defect list) | — | No change needed |
| H2.7 | Failed timing evidence replaced with `0.0` | `execution_measurement.py`'s `SynchronizedTimer`/`CudaMemoryMeasurer` already record `failure_type`/`failure_message` and never zero out a failed duration | **ALREADY FIXED** | — | No change needed |
| H3.1/H3.2 | Mismatch evidence has no expected/observed values | `replay_evidence`'s `first_mismatch` helper returned a bare index only | **CONFIRMED, exact** | High — a mismatch could not be diagnosed without re-running the model | Fixed: `kvcot.discovery.mismatch.build_mismatch_record` (new canonical schema), wired into token/logical-call/actual-call/compaction comparisons |
| H3.7 | Invalid Pass-2 example discards the actual replayed tokens | `orchestrator.py`'s `pass2_result.valid is False` branch built `ExampleResult(...)` without `pass2_replayed_token_ids=...`, defaulting to `()` despite `pass2_result.replayed_token_ids` holding real data | **CONFIRMED, exact** | High | Fixed: threaded through |
| H4.1/H4.2 | Coordinator trusts a bare `verified=True` boolean for the RTX-3090 gate | `b2a_execute.py`: `single_rtx3090_verified = (fullkv.device_evidence.get("verified") is True and rkv.device_evidence.get("verified") is True)` — no raw-field re-derivation, no FullKV/R-KV agreement check | **CONFIRMED, exact** | High (GPU-execution-time defect; unreachable on this CPU-only build) | Fixed: `strict_device.verify_device_gate_from_raw_evidence` recomputes from raw fields and requires cross-worker agreement |
| H4.3 | No CLI preflight artifact to cross-check against | No such artifact exists anywhere in the execute path | **CONFIRMED** | Medium | **Closed in round 2** (§3) — CLI now calls `verify_single_rtx3090`, writes real `preflight.json`, threads `cli_device_preflight` into a 3-way coordinator check |
| H4.4-H4.7 | Snapshot/tokenizer/dataset-row raw-evidence re-validation, local-tokenizer-only prompt verification | Partially present (`_verify_resolved_prompt_identity` already re-fetches/re-tokenizes/re-hashes); no dedicated typed snapshot-evidence re-validation beyond field-equality checks already in `b2a_execute.py` | **PARTIALLY CONFIRMED** | Medium | **H4.4/H4.6 closed in round 2** (§3) via `snapshot_boundary.verify_snapshot_evidence_raw`; H4.5/H4.7 remain open (§5) |
| H5 | Pre-branch memory estimate omits shape-derived allocations (bridge/scored-horizon K/V growth, query-cache growth, per-token logits/log-softmax) | `check_pre_branch_memory` accounts `snapshot_clone_bytes` (real, shape-derived) + `selected_vector_bytes` + `known_temporary_bytes` (caller-supplied, currently only `2 * vocab_size * 4` for logits/log-softmax) — no K/V-growth-across-branch-horizon term | **CONFIRMED** | Medium (GPU-execution-time defect; unreachable on this CPU-only build) | **Closed in round 2** (§3) — full componentized estimate |
| H6.1/H6.2 | Final artifact gates are existence-only checks | `b2a_execute.py`: `attempt_files_verified = required_attempt_files.issubset(existing)`; `worker_envelopes_verified = all((path/"envelope.json").is_file() ...)` | **CONFIRMED, exact** — matches the hostile-audit grep pattern `issubset(existing)` verbatim | High (GPU-execution-time defect; unreachable on this CPU-only build) | **Closed in round 2** (§3) — `kvcot.discovery.attempt_verification` |
| H7 | Provenance/progress lifecycle completeness | `collect_execution_provenance` already exports git/software/hardware; `build_attempt_references` already computes a per-file sha256/size manifest; no immutable start/end (`completion.json`) split, no dedicated device/hardware preflight artifact | **PARTIALLY CONFIRMED** | Medium | **Immutable start/end split and device-preflight artifact closed in round 2** (§3); H7.4's stage-completeness verifier remains open (§5) |
| H8.5 (hostile grep) | `device_map="auto"` in production path | `kvcot/generation/policies.py:94,161` — this is the **primary Qwen-1.5B pipeline's** model loader, never the B2A/discovery path (`kvcot.discovery.strict_device.load_fullkv_discovery_model`/`load_rkv_discovery_model` already use explicit `device_map={"": device}`) | **FALSE ALARM for the discovery/B2A path; pre-existing, out-of-scope for this repair** (primary pipeline, frozen by CLAUDE.md §4, not part of the B1B/B2A harness this brief scopes) | — | No change — out of scope |
| H8.5 (hostile grep) | Bare `except Exception:` instances | `b2a_workers.py:_framework_seed_for_env` (already justified in an existing docstring — auxiliary env value only, real validation happens in the worker itself); `manifest_prepare.py` (3 instances, pre-existing, **not investigated** — flagged, not verified) | Mixed: one **valid with exact proof** (pre-existing, documented), three **not investigated** | Low-medium | Flagged as open in §5 |
| H8.2 | No unified contract-consistency test | No such test existed | **CONFIRMED** | Medium | **Closed in round 2** (§3) — `test_contract_consistency.py`, 10 tests |

## 2. What round 1 repaired (Gates H1-H3, partial H4)

### Gate H1 — durable partial worker evidence (complete)

- New module `src/kvcot/discovery/worker_partial_evidence.py`: typed
  `PartialWorkerEvidence` (pydantic), `WorkerBodyFailure` exception
  (chains the original exception as `__cause__`), `classify_failure`
  (OOM/timeout detection by exception type/message), and
  `capture_partial_evidence` (reads whatever local variables — `timer`,
  `memory_meter`, `determinism_policy`, `device_evidence`, `actual_calls`,
  `example_result`, attrition counters — were already bound in the failing
  function's own `locals()` at the moment of failure).
- `src/kvcot/discovery/b2a_workers.py`: `run_fullkv_worker`/`run_rkv_worker`
  bodies are now wrapped in `try/except Exception`, tracking the last
  completed stage via a wrapped `_progress` callback, raising
  `WorkerBodyFailure` with a full `PartialWorkerEvidence` snapshot on any
  failure. A new `if example_result.aborted: raise ...` check promotes an
  aborted `run_example` return (see below) into the same failure path, so
  its rich partial pair/attrition evidence is captured too.
- `src/kvcot/discovery/orchestrator.py`: `run_example` now catches
  exceptions from `run_pass2_capture` (new `STAGE_PASS2_EXECUTION_EXCEPTION`
  attrition stage) and from the per-pair evaluation loop (new
  `STAGE_UNEXPECTED_PAIR_EXCEPTION` stage), returning an `ExampleResult`
  with `aborted=True` and every pair/identity/mutation-report/attrition
  entry completed before the abort preserved intact. The pre-existing
  Pass-1 exception handler (a bare `except Exception:` mapping any
  exception to `STAGE_NATURAL_RUN_INVALID` with zero diagnostic
  information — found during this pass's own hostile audit, not in the
  original brief) now populates the same `aborted`/`abort_failure_type`/
  `abort_is_oom` fields, without changing the funnel stage name.
- `src/kvcot/discovery/attrition.py`: two new stage constants
  (`STAGE_PASS2_EXECUTION_EXCEPTION`, `STAGE_UNEXPECTED_PAIR_EXCEPTION`).
- `src/kvcot/discovery/worker_envelope.py`: `WorkerEnvelope` gained
  explicit typed `failure_stage`, `last_completed_stage`, `is_oom`,
  `is_timeout` fields (never buried only inside the unconstrained
  `partial_measurements` blob); `build_failure_envelope` accepts them.
- `src/kvcot/discovery/b2a_worker_entry.py`: the failure branch now
  detects `WorkerBodyFailure` specifically, threads its `.evidence` into
  the envelope, and reports the ORIGINAL cause exception's type/message
  (never the wrapper's own composed message) as `error_type`/
  `error_message` — `traceback.format_exc()` still captures the full
  chained traceback. The durable "failed" progress event is now appended,
  and the envelope written, BEFORE environment variables are restored
  (H1.4's required ordering).
- `src/kvcot/discovery/b2a_workers.py` (`run_both_workers_via_subprocess`):
  a new `_write_termination_record` writes a coordinator-authored
  `termination.json` (attestor=coordinator, termination kind, timeout
  flag, return code, command, stdout/stderr hashes, last durable progress
  event, worker-authored-envelope-present flag) on both a subprocess
  timeout and a nonzero exit — distinguishing a coordinator-witnessed
  external termination from a worker-authored failure envelope, never
  fabricating the latter.
- **Tests**: `test_worker_partial_evidence.py` (9), 5 new tests in
  `test_orchestrator_partial_failure.py` (pre-branch-guard-raises,
  Pass-2-capture-raises, Pass-1-raises, normal-completion-not-aborted,
  Pass-2-token-mismatch-preserves-tokens), 2 new tests in
  `test_b2a_worker_entry.py`, 3 new tests in `test_b2a_workers.py`
  (termination record on timeout, on nonzero exit, coordinator process
  timing), plus one existing test
  (`test_run_rkv_worker_fails_closed_when_fake_kv_cluster_disagrees_with_frozen_config`)
  updated to expect the new `WorkerBodyFailure` wrapping (the underlying
  `WorkerFailedError` is preserved as `__cause__`; asserted explicitly).

### Gate H2 — semantically correct timing and projection (mostly complete)

- Renamed the misleadingly-named `capture_and_parity` timing phase to
  `call_trace_comparison` (`b2a_workers.py`, `final_contract.py`'s
  `RKV_REQUIRED_TIMING_PHASES`) — it only ever timed
  `compare_call_boundary_traces`, never the real capture/gather/parity
  work, which genuinely happens earlier, inside the already-real,
  already-synchronized `rkv_pass2_prefill`/`rkv_pass2_decode`/
  `snapshot_creation` phases.
- Expanded the startup/load runtime-projection component
  (`b2a_execute.py`) from 2 phases to the 5 genuinely one-time,
  non-overlapping, sequential phases per worker; every summed phase is
  listed explicitly in a code comment per the audit's "prove they do not
  overlap, prove none is omitted" requirement.
- Added coordinator-observed process-launch timing
  (`WorkerCoordinationResult.coordinator_observed_process_seconds`,
  measured via `time.perf_counter()` around each `_launch_worker` call)
  and a derived `process_overhead_diagnostic` (worker-internal
  startup/load, worker-internal inference, unattributed overhead) — a
  separate, honestly-labeled diagnostic never summed into
  `runtime_projection` itself.
- H2.6 (synchronized pair-timing source) and H2.7 (failed timing survives
  as real evidence, never `0.0`) were found ALREADY correctly implemented
  and are unchanged.
- **Not done**: H2.3's full "injected callbacks/spies proving call nesting
  and ordering" test matrix was not built as a generic framework; the
  rename and projection fixes are covered by targeted tests instead
  (`test_b2a_execute_coordinator.py`: 3 new tests; `test_b2a_workers.py`:
  1 new test).

### Gate H3 — replay and mismatch evidence (partially complete)

- New module `src/kvcot/discovery/mismatch.py`:
  `build_mismatch_record(expected, observed) -> MismatchRecord` with
  `matched`, `first_mismatch_index`, `expected_value`, `observed_value`,
  `expected_length`, `observed_length`, `mismatch_kind` (distinguishing
  `value_differs`/`expected_ends_first`/`observed_ends_first`/`matched`),
  never indexing beyond either sequence.
- Wired into `replay_evidence`'s four comparisons (token, logical-call,
  actual-call, compaction) in `b2a_workers.py` — the original bare
  `*_first_mismatch` index fields are preserved (existing consumers in
  `b2a_execute.py` are unaffected) and new `*_mismatch` keys carry the
  full record.
- Fixed `orchestrator.py`'s Pass-2-invalid branch to thread
  `pass2_result.replayed_token_ids` into the returned `ExampleResult`
  (previously discarded, defaulting to an empty tuple).
- **Tests**: `test_mismatch.py` (7 tests covering the H3.8 negative-test
  list: value mismatch, expected-shorter, observed-shorter, empty
  sequences, dict-element comparison for call records, export
  round-trip), plus 1 new orchestrator test proving the token-mismatch
  preservation.
- **Not done**: H3.4-H3.6's "complete ordered expected/observed call
  entry at the first mismatch" (beyond the single expected/observed
  VALUE already in `MismatchRecord`) and H3.7's explicit `Pass2Result`/
  worker-result schema field additions (the token evidence now survives
  via `ExampleResult.pass2_replayed_token_ids`, but no NEW top-level
  schema field was added to `RKVWorkerResult` itself for this).

### Gate H4 — device/snapshot verification (partially complete)

- New `strict_device.verify_device_gate_from_raw_evidence`: recomputes
  `single_rtx3090_verified` from raw fields on both workers'
  independently-reported device evidence (`visible_gpu_count == 1`,
  `device_index == 0`, GPU name contains "RTX 3090", VRAM within a frozen
  plausibility range, driver/CUDA/cuDNN version strings present) AND
  requires FullKV/R-KV agreement on GPU name, device index, VRAM, compute
  capability, driver, CUDA runtime, and cuDNN — never trusting either
  worker's bare `verified=True` alone. Wired into `b2a_execute.py`'s
  `final_gate_result` construction.
- **Tests**: 10 new tests in `test_strict_device.py` covering the H4.8
  list (wrong GPU name with `verified=True`, two visible GPUs, wrong
  device index, insufficient/malformed VRAM — 4 parametrized cases,
  mismatched FullKV/R-KV device evidence — 2 sub-cases, missing
  driver/CUDA/cuDNN, `verified=False`, empty evidence dicts).
- **Not done** (remaining gaps, all GPU-execution-time defects
  unreachable on this CPU-only build, but real code-level gaps):
  - H4.3: no CLI execute-mode preflight artifact exists to cross-check
    the coordinator's device gate against; the coordinator only compares
    FullKV vs. R-KV, never a third independent CLI-level observation.
  - H4.4/H4.6: no typed, independently-re-validated snapshot-evidence
    dataclass (file inventory, incomplete/lock-file checks, shard
    validation) beyond the field-equality checks `b2a_execute.py` already
    performs against the manifest.
  - H4.5: `_verify_resolved_prompt_identity` already re-tokenizes via the
    frozen tokenizer name/revision with `local_files_only` implied by
    reusing `manifest_prepare`'s own resolution path, but no NEW explicit
    assertion was added proving it cannot fall back to a network path if
    the local snapshot is missing (not verified this pass).
  - H4.7: the tokenizer-only-validation-vs-production-resolver
    distinction documented in `docs/B1_TOKENIZER_ONLY_VALIDATION.json` was
    not re-examined this pass.

## 3. Round 2 — closing the remaining Gates H4-H8 (forward-only, same branch)

A second repair pass, on top of the round-1 commit above, closed the
majority of what round 1 left open. Evidence-based, confirmed against the
actual code before implementation, same discipline as round 1.

### Gate H4 — closed (CLI preflight + snapshot re-validation)

- `kvcot.cli.cmd_b2a_calibrate --execute` used to write a trivial
  `preflight.json` (`{"passed": True, ...}`, a literal, written BEFORE the
  CUDA availability check even ran) and never called `verify_single_rtx3090`
  at all. It now calls the same raw-evidence producer the workers use,
  writes the REAL result into `preflight.json`, and threads it into the
  coordinator as `cli_device_preflight`.
- `strict_device.verify_device_gate_from_raw_evidence` gained an optional
  third observation parameter — when the CLI preflight is supplied, the
  gate now cross-checks THREE independent observations (CLI, FullKV,
  R-KV), not two; backward compatible (`None` preserves the original
  two-way check).
- New `snapshot_boundary.verify_snapshot_evidence_raw`: re-validates a
  worker-reported `VerifiedLocalSnapshot.__dict__` (repository identity,
  asset type, exact-SHA revision agreement, `local_files_only`, non-empty
  file inventory, no incomplete/lock files, required config/tokenizer/
  weight files present) from the raw JSON dict alone — the coordinator's
  `snapshot_verified` no longer trusts a bare `verified=True` plus a
  single `resolved_revision` comparison.
- Tests: 4 new three-way-check tests in `test_strict_device.py`, 15 new
  tests in `test_snapshot_boundary.py`, 2 new CLI tests in
  `test_cli_b2a_calibrate.py` (real preflight evidence written; refusal +
  failure.json on a device-verification failure).

### Gate H5 — closed (componentized pre-branch memory estimate)

- `check_pre_branch_memory` no longer takes an opaque caller-computed
  `known_temporary_bytes` — it now derives, from the snapshot's own real
  tensor shapes and the frozen bridge/scored-horizon counts
  (`BRIDGE_TOKEN_COUNT`, `SCORED_HORIZON`): `per_token_kv_growth_bytes`
  (new `snapshot_growth_bytes_per_token`), `complete_horizon_kv_growth_bytes`,
  a conservative `append_realloc_temporary_bytes` (documented, not exact),
  an explicit `query_cache_growth_bytes=0` (R-KV's query cache is a
  fixed-size window — documented, not omitted), `logits_bytes`,
  `log_softmax_bytes`, `nll_scalar_bytes`, and `position_tracking_bytes`
  (new `snapshot_position_tracking_bytes_per_token`). Every component is
  independently visible on `PreBranchMemoryEvidence`, not folded into one
  total. Negative/wrong-type inputs and an empty-cache snapshot now raise
  instead of silently computing a wrong estimate.
- Tests: 6 new tests in `test_execution_measurement.py` proving the exact
  formula by hand, shape-scaling (2x heads/layers -> 2x growth), and
  independence of `selected_vector_bytes` from pre-existing cache length.

### Gate H6 — closed (content-verified attempt artifact lifecycle)

- New module `kvcot.discovery.attempt_verification`:
  `verify_attempt_artifacts` parses every required pre-final artifact and
  cross-validates envelope/result hash agreement, timing/memory/pair-
  identity/semantic-swap/replay-evidence mirrors against their source,
  command role/check/capture_output correctness, envelope success/role/
  attempt-id agreement, and progress-journal parseability/non-emptiness —
  replacing the pure-existence `required_attempt_files.issubset(existing)`.
  `verify_worker_envelopes` replaces the pure `.is_file()` check with
  parse-and-validate (role, success, no error recorded).
- Tests: 16 new tests in `test_attempt_verification.py`, covering the full
  H6.7 mutation list (result byte, envelope hash, timing, memory, pair
  identity, replay token, semantic mutation report, command role, attempt
  ID, malformed JSON/JSONL, missing file, empty progress journal).

### Gate H7 — closed (immutable start/end split + device preflight artifact)

- `cmd_b2a_calibrate --execute` now writes a separate, immutable
  `completion.json` (finished_at timestamp, outcome, exit code, gate-passed
  flag, artifact path) in a `finally` block, guaranteed to exist whether
  the command reaches a clean pass, a gate failure, or an uncaught
  exception — `invocation.json` is never rewritten.
- The CLI's `preflight.json` (Gate H4 above) already IS the dedicated
  post-CUDA-inspection device artifact H7.3 asks for, distinct from the
  CPU-safe `provenance.json` written earlier, before any CUDA call.
- Tests: 2 new tests in `test_cli_b2a_calibrate.py` (completion record on
  gate failure and on an uncaught exception).

### Gate H8 — contract-consistency test built; CI/audit re-confirmed

- New `test_contract_consistency.py` (10 tests): every required timing
  phase is genuinely emitted in production code (not just declared);
  `capture_and_parity` is permanently barred from reappearing;
  `FINAL_MANDATORY_GATE_CONDITIONS` has no duplicates and the dry-run's
  printed gate list exactly matches it; every mandatory gate has a real,
  directly-executed negative test; every `STAGE_*` constant is registered
  in `STAGE_ORDER` (both directions); `WorkerEnvelope`'s failure fields
  match what `PartialWorkerEvidence` can actually populate, and every
  field `b2a_worker_entry.py` reads off it is real; `MismatchRecord`'s
  export schema is frozen; the old inline `required_attempt_files`/
  `issubset(existing)` pattern is barred from reappearing as code.
- Hostile grep re-run: no new occurrence of `partial_measurements=None`,
  a functional `issubset(existing)`, or a fabricated `verified=True`
  outside already-justified, evidence-conditioned call sites.

## 4. Round 3 — closing H2.2, H4.5, H6.4, H7.4, and the `manifest_prepare.py` audit

A third, forward-only pass, prompted by a user request to verify the round-2
"remaining gaps" list rather than take it at face value. Each item was
re-investigated against the actual code; the ones that were genuinely
still open were closed.

- **`manifest_prepare.py`'s 3 bare `except Exception:` instances**
  (flagged as "not investigated" after round 2) — investigated and
  confirmed **valid with exact proof**, not blockers: (1)
  `_snapshot_weight_shaped_files`'s `scan_cache_dir()` call returns
  `frozenset()` on failure, a deliberately conservative default for a
  pure before/after weight-file safety-net comparison (an empty
  pre-snapshot makes any later-appearing file show as "new", which is
  the CORRECT direction for a safety check, never suppressing it); (2)/(3)
  two old-manifest-parse fallbacks that treat a non-parsing file as
  "not yet resolved"/"no prior identity", both conservative, both already
  gated behind explicit `--force` where relevant.
- **Gate H2.2 (sub-phase granularity) — closed.** `capture_update_kv`
  (`capture.py`) gained an optional `capture_timer_fn` parameter, threaded
  through `pass2.run_pass2_capture` and `orchestrator.run_example`
  (both new optional parameters, defaulting to `None`, preserving every
  pre-existing caller's exact behavior) down to `b2a_workers.py`'s
  `timer.measure`. It times exactly the `_build_capture_record` call --
  the REAL gather reconstruction, gather-parity check, and
  absolute-position-parity check -- under a new, accurately-named
  `capture_gather_and_parity` phase (added to
  `RKV_REQUIRED_TIMING_PHASES`), firing once per selected target (3 times
  per example). Tests: 3 new in `test_capture.py` (timer invoked around
  the real record-building call, never invoked for a non-target
  pass-through call, prior behavior preserved when omitted), 1 new in
  `test_orchestrator_pair_execution_policy.py` (end-to-end: fires exactly
  3 times through the real synthetic-harness Pass 2 path).
- **Gate H4.5 — closed.** `_verify_resolved_prompt_identity` used to call
  `_render_and_tokenize` with only `tokenizer_name`/`tokenizer_revision`,
  resolved through `huggingface_hub`'s ordinary (potentially
  network-touching) lookup -- never proven local-only, despite this
  function's whole purpose being to verify a strict local-snapshot
  boundary. It now resolves the exact local tokenizer snapshot FIRST via
  `snapshot_boundary.resolve_local_snapshot` (the same function the
  workers themselves use), failing closed with `B2AExecutionRefused` if
  unavailable, and loads the tokenizer from that exact verified path with
  `local_files_only=True`. `manifest_prepare._render_and_tokenize` gained
  a matching optional `local_only_path` parameter (default `None`,
  preserving `prepare-b2a-manifest`'s existing, deliberately
  network-capable tokenizer-only resolution unchanged). Tests: 2 new in
  `test_b2a_execute_coordinator.py` (fails closed when the local snapshot
  is unavailable; loads from the exact verified path when available).
- **Gate H6.4 — closed.** Confirmed a genuine duplicate: the worker body
  ALREADY appends a `(stage, "completed")` progress event live, via
  `_production_progress_callback`, for every named phase (snapshot
  resolution, tokenizer load, model-load completion, runtime
  verification, Pass 1, Pass 2, compact-target conversion, each real
  pair, no-op) as it completes -- yet `b2a_worker_entry.py`'s success
  path ALSO replayed the final `timing_evidence` list and re-appended a
  second `"completed"` event for every one of those same phases, using
  the identical stage-name mapping. The redundant post-hoc replay loop is
  removed; progress is now written live only. Test: 1 new in
  `test_b2a_worker_entry.py` proving each required stage appears in
  `progress.jsonl` exactly once, not twice, for a worker body that
  emits live progress exactly like the real ones do.
- **Gate H7.4 — closed.** New `attempt_verification
  .verify_progress_stage_completeness` checks a parsed progress journal
  against the exact stage names this repository's OWN worker
  bodies/entry point actually emit (derived from the code, not from an
  aspirational list broader than what a worker journal can produce; CLI/
  coordinator-level stages like "device preflight" live in `preflight.json`/
  `final.json` instead and are out of this function's scope by design),
  wired into `verify_attempt_artifacts`'s existing progress-journal check.
  Tests: 2 new in `test_attempt_verification.py` (a truncated journal
  fails; the checker reports the specific missing stage names), plus the
  existing fixture's `_progress_lines` helper was updated to emit the
  complete, real per-role stage list.

## 5. Full local validation performed (after all three rounds)

```
python -m compileall src tests                     -> exit 0
python -m pytest --collect-only -q                  -> 1116 tests collected
python -m pytest -m "not gpu" -q                    -> 1102 passed, 14 deselected
python -m kvcot prepare-b2a-manifest --dry-run       -> exit 0, no download/write
python -m kvcot b2a-calibrate --dry-run              -> exit 0, no CUDA/model execution
python -m kvcot --help                               -> exit 0
git diff --check                                    -> exit 0 (CRLF-normalization
                                                         warnings only, no real
                                                         whitespace errors)
```

One unrelated, pre-existing flake was observed and independently confirmed
non-reproducible in isolation and re-runs: `test_math_verifier.py
::test_accepted_equivalences`, a subprocess-timeout-sensitive symbolic-
equivalence check (`kvcot.utils.math_verifier`, last touched in an
unrelated 2026-07-19 commit, `a393de4`, predating every repair round in
this document) occasionally hits its frozen 5.0-second child-process
timeout under machine load. Re-run in isolation multiple times, it passed
every time except when run concurrently with another heavy test session.
Not modified (out of scope; modifying a frozen timeout "merely to make a
test pass" is explicitly prohibited) and not counted as a repair-round
regression.

No GPU inference, no CUDA, no model-weight download, no B2A/B2B execution,
no Vast.ai activity of any kind occurred during any round.
`configs/lock.yaml` and `third_party/R-KV` are byte-for-byte unchanged.

## 6. Remaining gaps (explicit, not claimed complete, after all three rounds)

- **Gate H4.7**: the tokenizer-only-validation-vs-production-resolver
  distinction documented in `docs/B1_TOKENIZER_ONLY_VALIDATION.json` was
  not re-examined this round.
- **Gate H8.6**: no separate, formally-written manual call-graph trace
  document — the equivalent understanding was built through direct code
  reading (this document's evidence table and the round-by-round sections
  above cite exact file/line locations for every claim) rather than
  produced as a standalone artifact.
- **`kvcot/generation/policies.py`'s `device_map="auto"`** (2 instances)
  is the PRIMARY Qwen-1.5B pipeline's model loader, frozen by CLAUDE.md
  §4 and out of scope for this B1B/B2A-harness-focused repair — confirmed
  not part of the discovery/B2A path. Left unchanged deliberately.

That is the complete remaining-gap list. Every other item named across all
three rounds' findings tables is now closed and tested.

## 7. Verdict

```
B1 FINAL CPU CLOSURE VERDICT:
INCOMPLETE — B2A/GPU REMAIN BLOCKED
```

Gates H1, H2, H3, H4, H5, H6, and H7 are fully repaired and tested for the
CPU-auditable path. Gate H8's contract-consistency test is built; CI is
verified valid; the hostile audit found no unexplained production-path
hit. The only remaining items (§6) are H4.7's documentation distinction
and H8.6's formal (vs. evidenced-inline) call-graph document — both are
audit-formality items, not functional defects, but per the completion
standard an unfinished item may not be called optional, so the verdict
remains INCOMPLETE.

No B2A result exists. No B2B result exists. No real CUDA timing exists. No
RTX 3090 memory measurement exists. Frozen-row real eligibility remains
unknown. No FaithKV method exists. B2A remains blocked pending another
independent audit. B2B remains blocked.

## 8. Round 4 (2026-07-21) — final bounded independent-audit repair (F1–F10)

Round 3's §7 claimed the only remaining items were "audit-formality items,
not functional defects" (H4.7 documentation and the H8.6 call-graph
document). **That was an overclaim.** A subsequent independent audit
verified nine further *functional* execution-boundary defects (F1–F9)
plus the two documented formalities (F10). Round 4 fixes all of them,
starting from commit `419bbc0020b374d6c4a2085a7a04ff293d7ec680`.

| ID | Confirmed defect | Repair (files) |
|---|---|---|
| F1 | Both worker exception handlers reported `failing_stage=_last_completed_stage` — the last *completed* stage, not the failing one — and omitted `attempt_id`/`last_completed_stage`. | Typed `WorkerExecutionState` (`worker_partial_evidence.py`); tracked `timer.measure` wrapper + explicit stage enters in both worker bodies (`b2a_workers.py`); handlers now pass `failing_stage=current_stage`, `last_completed_stage`, `attempt_id` (from `KVCOT_B2A_ATTEMPT_ID`). 12 injection tests in `test_final_audit_repairs.py`. |
| F2 | `PartialWorkerEvidence` declared `failed_pair_identities`/`no_op_evidence`/`replay_evidence` but `capture_partial_evidence` never populated them, and dropped minimized targets, pair failure details, no-op identity, pre-branch guard evidence, compaction positions, and abort state. | Shared single-implementation helpers `derive_failed_pair_identities`/`build_no_op_evidence`/`build_replay_evidence`/`derive_compaction_positions` (`b2a_evidence.py`) called by BOTH the success path and `capture_partial_evidence`; evidence schema extended with all listed fields plus `example_aborted`/abort type/message/OOM. 6 preservation tests. |
| F3 | `MemoryPhaseEvidence` recorded only `failure_type`, no message. | `failure_message` added (`execution_measurement.py`); failure records preserve full phase/before/peak/after/reset/sync fields; never replaced with zero-valued success records. |
| F4 | Verifier did not parse `invocation.json`/`preflight.json`/`provenance.json`; no completion/process-outcome/command-identity/saved-result/typed/progress-ordering/hash validation. | `attempt_verification.py` rewritten into the one authoritative verifier: top-level artifact parsing (attempt-ID agreement, timestamps, start<=finish, sanitized argv, path/hash agreement, preflight hardware schema, completion outcome/exit-code agreement, provenance git/submodule), exact worker command identity (incl. `text` mode, timeout, duplicate/contradictory flag rejection), saved-vs-coordinator result equality, typed `WorkerEnvelope`/`FullKVWorkerResult`/`RKVWorkerResult` validation, `process_outcome.json` (new coordinator-owned artifact), and full progress-journal validation (known stages/statuses, per-event IDs/timestamps, monotonic singleton ordering, no duplicate singleton completion, start-before-completion, exactly 12 unique real-pair + 1 no-op completions, no failure events, result-before-envelope). `verify_final_reference_manifest` recomputes every hash. |
| F5 | `final.json` was written before `completion.json`, so its reference set could never include the completion record; `invocation.json` had no start timestamp. | Coordinator now writes `completion.json` (attempt ID, finished_at, outcome/exit-code/gate agreement, intended final path) BEFORE building the reference manifest and writing `final.json` LAST (`b2a_execute.py`); `final.json` is the only artifact excluded from its own reference set; final-write failure preserves all pre-final artifacts, writes atomic `final_write_failure.json`, never overwrites `completion.json`, raises `B2AFinalWriteError` (nonzero); CLI completion write is now a write-if-missing fallback; `invocation.json` gains a real UTC `started_at`. |
| F6 | Provenance omitted total RAM, OS/kernel detail, origin-branch SHA, and hardcoded `3c853cf` as sole start authority. | `collect_execution_provenance` (`attempt_artifacts.py`): branch/HEAD/origin-main/origin-branch SHAs, dirty state + status/staged/unstaged/untracked, `419bbc0` starting commit + three-way required ancestry (`419bbc0`/`7ef13ae`/`3c853cf`), submodule observed/expected/match, full system block (OS, platform, kernel release/version, arch, CPU, logical CPUs, `total_physical_ram_bytes` via psutil/sysconf/GlobalMemoryStatusEx with honest `None`, disk free for artifacts and model cache), 9-package software block, GPU-evidence cross-references. No credentials recorded. |
| F7 | Device gate verified card identity but not explicit requested-device identity or complete CPU/disk/meta/offload placement. | `requested_device` added to `StrictDeviceEvidence` + raw gate + three-way agreement fields (`strict_device.py`); `ParameterPlacementEvidence` gains `unique_devices` (full identity strings) + `requested_device` (`runtime_evidence.py`); new mandatory final condition `no_offload_and_placement_verified` (`final_contract.py`, wired in `b2a_execute.py`) via `verify_placement_from_raw_evidence`. 11 placement negatives + device-gate negatives. |
| F8 | Coordinator-side snapshot check did not reproduce index/shard validation. | `VerifiedLocalSnapshot` exports inventory hash, per-file sizes, file count, index files + content hashes, referenced/missing shards, recognized weights (`snapshot_boundary.py`); `verify_snapshot_evidence_raw` revalidates all of it (internal consistency of hash and total size included); `revalidate_snapshot_evidence_against_directory` recomputes from disk when the path is readable (never network). 8 negatives. |
| F9 | Timing/memory contracts compared phase-name sets; duplicate singletons passed. | Exact multiplicity maps + `_phase_counts` (`final_contract.py`): singletons exactly 1, `capture_gather_and_parity`/`snapshot_creation` exactly 3, 12 unique real-pair + 1 no-op complete phases each exactly 1 with every subphase exactly 1; prefill/decode timing counts checked against raw actual-call evidence (FullKV exact; R-KV prefill exact, decode floor — branch steps also record decode actual calls); rejects non-finite/zero/negative durations, failed-as-completed records, and unsynchronized memory records. |
| F10 | H4.7/H8.6 formalities open. | `docs/B1_TOKENIZER_ONLY_VALIDATION_CLARIFICATION.md` (three-path distinction, no cross-claiming); `docs/B1_FINAL_EXECUTION_CALL_GRAPH.md` (full success + 12 failure paths, durable artifact per node). |

Additional round-4 corrections while closing F4: the required progress
stage "runtime verification" was never actually emitted in production
(`post_load_validation` ran under `timer.measure`, which writes no journal
event) — it now runs under `measured(...)`; worker `command.json` records
`text: True`; the R-KV frozen-config check has its own tracked stage
(`runtime R-KV config verification`).

### CI status (honest report)

`.github/workflows/cpu-tests.yml` already exists and is the equivalent CPU
workflow the round-4 brief describes (compileall, collect-only,
`-m "not gpu"`, `git diff --check`; no CUDA, no downloads). However, every
recorded run — including the one for starting commit `419bbc0` — **failed
without starting**: GitHub's check-run annotation reads *"The job was not
started because your account is locked due to a billing issue."* This is
an account-level GitHub Actions lock, not a code failure, and cannot be
fixed from this repository. CI status is therefore **failed** (account
locked), and per the completion rule the round-4 verdict below remains
INCOMPLETE even though every F1–F10 repair, focused test, full non-GPU
suite run, and dry-run passes locally.

## 9. Round 4 verdict

```
B1 FINAL CPU CLOSURE VERDICT:
INCOMPLETE — B2A/GPU REMAIN BLOCKED
```

All F1–F10 repairs are complete and tested locally; the sole open item is
independent CI evidence, blocked by the GitHub account lock above. No GPU
inference was run. No model weights were downloaded. No B2A result exists.
No B2B result exists. No real CUDA timing exists. No RTX 3090 memory
measurement exists. Frozen-row real eligibility remains unknown. An
independent audit is still required. No FaithKV method exists.
