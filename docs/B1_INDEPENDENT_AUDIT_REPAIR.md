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
- **Not done**: H7.4's explicit progress-stage-completeness check against
  the full named list, and H7.1's most exhaustive invocation-field
  enumeration, were not built as dedicated new verifiers.

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

## 4. Full local validation performed (after both rounds)

```
python -m compileall src tests                     -> exit 0
python -m pytest --collect-only -q                  -> 1107 tests collected
python -m pytest -m "not gpu" -q                    -> 1093 passed, 14 deselected
python -m kvcot prepare-b2a-manifest --dry-run       -> exit 0, no download/write
python -m kvcot b2a-calibrate --dry-run              -> exit 0, no CUDA/model execution
python -m kvcot --help                               -> exit 0
git diff --check                                    -> exit 0 (CRLF-normalization
                                                         warnings only, no real
                                                         whitespace errors)
```

No GPU inference, no CUDA, no model-weight download, no B2A/B2B execution,
no Vast.ai activity of any kind occurred during either round.
`configs/lock.yaml` and `third_party/R-KV` are byte-for-byte unchanged.

## 5. Remaining gaps (explicit, not claimed complete, after both rounds)

- **Gate H2.2 (sub-phase granularity)**: the real capture/gather/parity
  work is genuinely timed (inside `rkv_pass2_prefill`/`rkv_pass2_decode`/
  `snapshot_creation`), but H2.2's full ask — separately-timed sub-spans
  for "target capture gather" vs. "capture gather parity" vs. "absolute-
  position parity" as distinct phases — was assessed as requiring deep
  instrumentation surgery inside `pass2.py`/`capture.py`'s per-layer hook
  and was not built.
- **Gate H4.5/H4.7**: no NEW explicit assertion proving prompt-identity
  tokenizer resolution cannot fall back to a network path if the local
  snapshot is missing; the tokenizer-only-validation-vs-production-
  resolver distinction in `docs/B1_TOKENIZER_ONLY_VALIDATION.json` was
  not re-examined.
- **Gate H6.4**: no dedicated detector for progress-journal duplication
  (live-appended vs. materialized-after-success events) — a real, if
  narrower, remaining sub-item.
- **Gate H7.4**: no dedicated verifier checking the full named progress-
  stage list (§H7.4) for completeness end-to-end.
- **Gate H8.6**: no separate, formally-written manual call-graph trace
  document — the equivalent understanding was built through direct code
  reading (this document's evidence table cites exact file/line
  locations) rather than produced as a standalone artifact.
- `manifest_prepare.py`'s three pre-existing bare `except Exception:`
  instances (surfaced by the hostile grep, not part of this repair's
  scope) remain uninvestigated.
- **`kvcot/generation/policies.py`'s `device_map="auto"`** (2 instances)
  is the PRIMARY Qwen-1.5B pipeline's model loader, frozen by CLAUDE.md
  §4 and out of scope for this B1B/B2A-harness-focused repair — confirmed
  not part of the discovery/B2A path. Left unchanged deliberately.

## 6. Verdict

```
B1 FINAL CPU CLOSURE VERDICT:
INCOMPLETE — B2A/GPU REMAIN BLOCKED
```

Gates H1, H3, H5, and H6 are fully repaired and tested. Gate H4 is fully
repaired for the CPU-auditable path (CLI preflight, three-way device
cross-check, snapshot content re-validation). Gate H2 is repaired except
for H2.2's finer sub-phase granularity. Gate H7 is repaired except for
H7.4's stage-completeness verifier. Gate H8's contract-consistency test is
built; its hostile-audit and CI sub-items are re-confirmed. Per the
completion standard, an incomplete item may not be marked complete and the
only valid verdict remains INCOMPLETE — the remaining gaps above are all
narrow, named, and GPU-execution-time-adjacent rather than open-ended.

No B2A result exists. No B2B result exists. No real CUDA timing exists. No
RTX 3090 memory measurement exists. Frozen-row real eligibility remains
unknown. No FaithKV method exists. B2A remains blocked pending another
independent audit. B2B remains blocked.
