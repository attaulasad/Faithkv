# B1 final execution call graph (H8.6)

Formal trace of the one-example B2A `--execute` path as implemented on this
branch, node by node, with the durable artifact each node owns. This
documents CPU-verified code only: **no GPU inference was run, no model
weights were downloaded, no B2A result exists, no B2B result exists, no
real CUDA timing exists, no RTX 3090 memory measurement exists, and no
FaithKV method exists.**

Notation: `A -> B` means A calls B; the bracketed path names the durable
artifact the node writes (all inside the immutable
`results/decisions/b2a_attempt_<UTC>_<id>/` directory unless noted).

## 1. Success path

```
CLI (kvcot.cli.cmd_b2a_calibrate --execute)
-> attempt creation      create_attempt_directory            [directory itself + fullkv/ rkv/]
-> invocation            atomic_write_json                   [invocation.json: attempt_id, started_at (UTC),
                                                              sanitized argv, config_path, manifest_path, env]
-> config/manifest load  load_discovery_config /
                         load_b2a_one_example_manifest       [none on success]
-> provenance            collect_execution_provenance        [provenance.json: git (branch, HEAD, origin/main,
                                                              origin-branch SHA, dirty/status/staged/unstaged/
                                                              untracked, 419bbc0 starting commit + three-way
                                                              required ancestry, R-KV submodule observed/expected/
                                                              match), system (OS/platform/kernel/arch/CPU/logical
                                                              CPUs/total RAM/disk free), software versions,
                                                              GPU-evidence cross-references]
-> CUDA + device preflight  verify_single_rtx3090            [preflight.json: passed, config/manifest hashes,
                                                              raw StrictDeviceEvidence incl. requested_device]
-> coordinator           b2a_execute.run_b2a_calibration
   -> local prompt verification  _verify_resolved_prompt_identity
                         (resolve_local_snapshot tokenizer, local_files_only)   [folded into final.json:
                                                              dataset_row_verification]
   -> worker launch      b2a_workers.run_both_workers_via_subprocess
      -> FullKV subprocess  python -m kvcot.discovery.b2a_worker_entry --role fullkv
                                                             [fullkv/command.json (argv/timeout/check/
                                                              capture_output/text), fullkv/stdout.log,
                                                              fullkv/stderr.log, fullkv/progress.jsonl]
         -> worker body  b2a_workers.run_fullkv_worker
            (WorkerExecutionState tracks current/last-completed stage;
             snapshot resolution -> tokenizer load -> model load ->
             post-load validation -> natural generation ->
             result construction)                            [fullkv/result.json,
                                                              fullkv/result.json.envelope.json]
      -> FullKV result/envelope validation  _validate_atomic_worker_envelope
                                                             [fullkv/envelope.json, fullkv/timing.json,
                                                              fullkv/memory.json]
      -> R-KV subprocess   python -m kvcot.discovery.b2a_worker_entry --role rkv
                                                             [rkv/command.json, rkv/stdout.log, rkv/stderr.log,
                                                              rkv/progress.jsonl]
         -> worker body  b2a_workers.run_rkv_worker
            -> Pass 1          orchestrator.run_example -> pass1.run_natural_pass1
            -> Pass 2          pass2.run_pass2_capture (token-identical replay)
            -> capture/parity  capture.capture_update_kv (capture_gather_and_parity x3)
            -> compact conversion  compact_target (selected-only)
            -> memory admission    check_pre_branch_memory (per pair, tracked stage)
            -> 12 real pairs + 1 no-op  pipeline.build_swap_pair_record
            -> result construction (tracked stage)           [rkv/result.json,
                                                              rkv/result.json.envelope.json]
      -> R-KV result/envelope validation                     [rkv/envelope.json, rkv/timing.json,
                                                              rkv/memory.json, rkv/pair_identities.json,
                                                              rkv/semantic_swaps.json, rkv/replay_evidence.json]
      -> coordinator process outcome                         [process_outcome.json: attempt_id, return codes,
                                                              timeout states, partial_success,
                                                              coordinator-observed durations]
   -> coordinator verification  attempt_verification.verify_attempt_artifacts
      (typed results/envelopes, exact command identity, saved-vs-supplied
       results, process outcome, full progress-journal validation)          [result recorded in final.json:
                                                              attempt_verification_reasons]
   -> gate derivation    final_contract.evaluate_final_gates
      (30 mandatory conditions incl. no_offload_and_placement_verified,
       exact timing/memory multiplicities, snapshot raw + on-disk
       revalidation, three-way device gate)                  [gate_result / final_gate_result in final.json]
   -> completion         atomic_write_json                   [completion.json: attempt_id, finished_at, outcome,
                                                              exit_code, gate_passed, intended final path,
                                                              config/manifest hashes — written BEFORE final.json]
   -> pre-final verification  verify_attempt_artifacts (again, now
      including completion.json)                             [pre_final_verification in final.json]
   -> reference manifest  build_attempt_references(exclude=final.json)
      (relative path, semantic role, size, SHA-256 for every pre-final
       artifact including completion.json)                   [attempt_artifacts in final.json]
   -> final.json         atomic_write_json — written LAST    [final.json]
-> CLI completion fallback  (skipped: completion.json already exists)
-> exit code 0 (gates passed) / 2 (gates failed)
```

## 2. Failure paths

| Failure | Node | Durable artifact(s) |
|---|---|---|
| config failure | CLI `load_discovery_config`/manifest load raises | `failure.json` (`stage=config_or_manifest_validation`), `invocation.json`; `completion.json` written by CLI fallback (`outcome=exception`) |
| preflight failure | CLI blockers or `verify_single_rtx3090` raises | `failure.json` (`stage=preflight`/`cuda_preflight`/`device_preflight`), `completion.json` fallback |
| model-load failure | worker body, `WorkerExecutionState.current_stage="model_load"` | worker `result.json.envelope.json` failure envelope with `failure_stage="model_load"`, `last_completed_stage="tokenizer_load"`, attempt ID, and full `PartialWorkerEvidence`; `progress.jsonl` `failed` event; coordinator `termination.json` (`nonzero_exit`); `failure.json`; `completion.json` fallback |
| Pass-1 failure | `run_example` aborts; worker promotes to raise | failure envelope with `failing_stage=rkv_pass1_prefill/decode`, Pass-1 partial evidence; same downstream artifacts as model-load failure |
| Pass-2 failure | `run_example` aborts | failure envelope with `failing_stage=rkv_pass2_*`, Pass-1 token evidence preserved (`example_aborted=True`) |
| pair failure (contained) | branch evaluation raises inside `build_swap_pair_record` | pair recorded in `pair_failure_details` + `failed_pair_identities` (typed stage/detail naming the failing subphase); worker still writes `result.json`; exact-count gates fail closed |
| pair failure (abort) | clone/mutation/admission raises in the pair loop | `run_example` returns `aborted=True`; worker raises; failure envelope carries attempted/completed/failed identities, pre-branch guard evidence, minimized targets, replay evidence, no-op evidence when already constructed |
| OOM | any worker stage | as above, plus `is_oom=True` (message/type classification), memory-phase failure record with `failure_type` AND `failure_message`, peak counters preserved |
| timeout | coordinator `subprocess.TimeoutExpired` | `termination.json` (`termination_kind=timeout`), preserved `stdout.log`/`stderr.log`, `failure.json` (`timed_out=true`), `completion.json` fallback |
| malformed result | coordinator schema validation fails | `WorkerFailedError` -> `failure.json` (partial FullKV result folded in when present), `completion.json` fallback |
| envelope failure | `_validate_atomic_worker_envelope` fails | `WorkerFailedError` -> `failure.json`, `completion.json` fallback |
| verification failure | pre-final `verify_attempt_artifacts` fails while gates passed | `B2AExecutionRefused` -> `failure.json`; `completion.json` already written, never overwritten |
| final-write failure | `final.json` atomic write raises | every pre-final artifact preserved; `final_write_failure.json` (best-effort, atomic); `completion.json` never overwritten; `B2AFinalWriteError` -> nonzero exit; never reported as a completed successful attempt |

## 3. What this graph is not

This document traces implemented, CPU-tested code paths (real worker bodies
exercised with injected deterministic fakes; the coordinator exercised with
a fake subprocess runner). It is not evidence of a real GPU execution:
until a separately authorized B2A run exists, every timing, memory, and
device value this graph's artifacts would contain remains unproduced.
