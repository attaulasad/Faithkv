# B1 final CPU execution-boundary closure

Starting ancestor: `3c853cff34e52d792cd0e5a96d1a5369f17f8047` on
`research/b1b-r4-final-b2a-closure`. This is a forward-only repair. The
frozen lock and `third_party/R-KV` pin are unchanged.

## Verdict

**B1 FINAL CPU CLOSURE VERDICT: READY FOR INDEPENDENT AUDIT — B2A/GPU
REMAIN BLOCKED.**

Compilation passed, 1,012 tests collected, and the complete non-GPU suite
passed with 998 tests and 14 GPU-marked deselections. Authorization does not
change: B2A and all GPU execution remain blocked pending an independent audit.

No B2A result exists.

No B2B result exists.

No RTX 3090 timing exists.

No FaithKV method exists.

Independent audit is required before any GPU authorization.

## Implemented boundary

- Pass 2 converts selected captures to `CompactBranchTarget` objects before
  pair evaluation, retains only selected K/V vectors and scalars beside the
  pristine snapshot, enforces a selected-count/head-dimension/dtype-derived
  bound, and releases the full captures and `Pass2Result` container.
- Branch evaluation retains only 48 NLL scalars and a hash. Baseline and
  swapped working caches run sequentially. Discovery restore transfers the
  exclusively-owned working snapshot tensors into the live cache, avoiding a
  second full mutable cache allocation.
- The actual production R-KV worker body is exercised on CPU through Pass 1,
  token-identical Pass 2, three selected events, twelve unique real pairs,
  one no-op, positive semantic mutation checks, and result construction.
- One synchronized timer owns all CUDA timing boundaries. Measurement code
  exclusively owns peak-memory resets. Model-load peaks participate in the
  maximum, and the pre-branch guard uses actual snapshot/vector/dtype sizes.
- Execute-mode loaders require one explicit RTX 3090 and local immutable
  snapshots. They use `device_map={"": "cuda:0"}` and
  `local_files_only=True`; auto placement and CPU/disk offload are forbidden.
- Actual adapter calls supply batch/shape/device/dtype evidence. Raw token,
  call, compaction, event, pair, mutation, and no-op evidence is retained.
- Execute mode creates an immutable attempt directory before validation,
  atomically writes JSON, preserves commands/logs/timeouts/partial success,
  journals progress, validates worker envelopes, and hashes artifact files.

## Runtime and memory contract

The exported projection is:

`FullKV startup+load + R-KV startup+load + 12 × (FullKV natural + R-KV Pass 1 + R-KV Pass 2) + 144 × max(B2A real-pair duration)`.

The no-op is excluded from B2B projection. Diagnostic subphases are not
added to their enclosing Pass/pair durations. The VRAM gate uses the maximum
of allocated and reserved peaks across every recorded phase, including load.

## Mandatory final gates

The canonical list is `FINAL_MANDATORY_GATE_CONDITIONS` in
`kvcot.discovery.final_contract`:

- git_clean_verified
- rkv_submodule_match
- single_rtx3090_verified
- local_model_snapshot_verified
- local_tokenizer_snapshot_verified
- dataset_row_identity_verified
- prompt_identity_verified
- fullkv_generation_matches_expected
- rkv_generation_matches_expected
- workers_generation_match
- actual_batch_size_verified
- complete_token_trace_match
- complete_call_trace_match
- complete_compaction_trace_match
- capture_gather_parity
- absolute_position_parity
- selected_event_ids_exact
- unique_real_pair_count_exact
- events_with_four_unique_pairs_exact
- no_duplicate_pair_identity
- authorized_no_op_identity_exact
- positive_semantic_swap_parity
- no_op_exact_parity
- all_required_timings_present
- all_required_memory_phases_present
- runtime_within_limit
- peak_vram_within_limit
- worker_envelopes_verified
- attempt_artifacts_verified

Missing, extra, unavailable, or non-boolean final gate evidence is rejected.

## Tokenizer-only validation

The permitted tokenizer-only validation succeeded for
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B@6a6f4aa4197940add57724a7707d069478df56b1`
in an isolated cache. It resolved `LlamaTokenizerFast`, EOS `128001`, the
frozen chat-template/prompt hashes, and 105 prompt tokens. The before
inventory was empty; the after inventory contains only config, tokenizer,
and Hugging Face metadata files. No `.safetensors`, `.bin`, weight index, or
weight shard was downloaded. Exact file sizes and SHA-256 hashes are in
`docs/B1_TOKENIZER_ONLY_VALIDATION.json`.

## Validation and limits

Focused CPU tests cover compact-storage/reachability, worker success and
failure paths, timing/memory ownership, strict snapshot/device checks, raw
evidence, envelope/attempt durability, and final gate fail-closed behavior.
The final compile passed; 1,012 tests collected; the complete non-GPU suite
reported 998 passed and 14 deselected. Both required dry-runs passed. The diff,
Git/submodule, and hostile-search audits completed without an open blocker.

No CUDA inference, B2A execution, B2B execution, Vast.ai activity, model
weight loading, or model-weight download was performed during this repair.

## Hostile-search classification

The final repository-wide search is classified as follows:

| Hit | Classification | Proof |
|---|---|---|
| `time.monotonic()` and primary-pipeline peak resets in `cli.py` | unrelated path | Existing non-discovery generate/replay paths; B2A workers use `SynchronizedTimer`. |
| `time.monotonic` default in `orchestrator.py` | generic CPU diagnostic | Execute mode explicitly supplies `perf_counter`; authoritative B2A pair and subphase evidence comes from the synchronized timer passed as the operation/phase runner. |
| `reset_peak_memory_stats` in `execution_measurement.py` | valid with proof | This is the sole B2A measurement owner; state reset contains no peak reset. |
| `device_map="auto"` text | valid with proof | Documentation describing the rejected failure mode; strict loaders use an explicit one-device map. |
| `_commit_hash` in `runtime_evidence.py` | valid compatibility fallback | Production supplies revisions from `VerifiedLocalSnapshot`; the private field is consulted only by injected/legacy callers without verified snapshot evidence and cannot satisfy final snapshot gates. |
| `shutil.rmtree` in worker coordination | test-only compatibility | Only the no-attempt temporary-directory path used by legacy CPU tests is removed; execute mode always supplies and preserves an attempt directory. |
| `atomic_write_text` occurrences | valid with proof | Fsync + refusal-to-overwrite helper, not direct `Path.write_text`. |
| `placeholder`/`deferred` in config, schemas, CLI, and import-discipline comments | unrelated path or explanatory text | No B2A evidence value or completed measurement is a placeholder. |
| `parity_check_passed=True` in capture construction | valid with proof | Reached only after tensor equality, absolute-position, kept-index, and score-component checks have all passed. |
| historical `INCOMPLETE` records | historical documentation | Preserved as required; the final closure record supersedes status without rewriting history. |

Removed from the B2A path: manual monotonic timing, `default=str`, the
at-least-four pair gate, direct artifact `write_text`, auto device mapping,
synthetic prompt-tensor batch evidence, and unexplained “never invoked” text.

## Manual call-graph audit

`cmd_b2a_calibrate --execute` creates the attempt and performs CPU/CUDA
preflight, then calls `run_b2a_calibration`. The coordinator alone launches
`b2a_worker_entry` twice. Each entry calls exactly one canonical body in
`b2a_workers`: FullKV uses `run_natural_pass1`; R-KV calls
`orchestrator.run_example`, which alone performs Pass 1, Pass 2, compact
conversion, the twelve real pairs, and the no-op. Pair construction alone
calls the compact branch evaluator and semantic swap. The coordinator alone
reconstructs final gates and writes `final.json`. No shadow worker, pair,
projection, gate, or artifact path remains.
