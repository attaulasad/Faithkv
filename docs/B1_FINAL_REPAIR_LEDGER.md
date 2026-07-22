# B1 Final Repair Ledger

> **SUPERSEDED.** An independent audit subsequently found the "Complete"
> status of every row below to be incomplete against the actual executable
> code (loose `verified` booleans, a misleadingly-named timing phase, bare
> mismatch indices, and total loss of partial worker evidence on failure,
> among other confirmed gaps). The historical rows below are preserved
> unedited. **See §"Independent-audit repair pass" at the end of this file,
> and `docs/B1_INDEPENDENT_AUDIT_REPAIR.md`, for the current, authoritative
> status.**

Starting commit: `3c853cff34e52d792cd0e5a96d1a5369f17f8047`.

| ID | Defect / risk / root cause | Files and implementation | Focused tests | Full-suite status | Open questions | Status |
|---|---|---|---|---|---|---|
| A1-A5 | Full Pass-2 capture records remained reachable throughout pair evaluation, retaining cache-length-scaled tensors. | `compact_target.py`, `orchestrator.py`, `pipeline.py`: selected-only K/V/scalar conversion with a shape/dtype-derived bound; explicit Pass2Result release; pair construction consumes `CompactBranchTarget`. | `test_compact_target.py`; existing B1B integration tests | 998 passed, 14 deselected | None. | Complete |
| A6-A7 | Production scoring accumulated 48 vocabulary logits and returned a live cache; this could overlap branch lifetimes. | `branch_eval.py`, `pipeline.py`: direct compact evaluator returns NLL scalars/hash only and drops live state before return. | Branch evaluator and branch lifetime tests | 998 passed, 14 deselected | None. | Complete |
| B1-B4 | Worker CPU path stopped at cap-hit and never proved a complete success result. | `test_b2a_workers_real_bodies.py`: deterministic Torch CPU causal LM and R-KV bookkeeping drive the production body through 3 events, 12 real pairs, 1 no-op, and final schema construction. | Controlled worker-body and worker-entry/coordinator end-to-end tests | 998 passed, 14 deselected | None. | Complete |
| B5 | Failure modes could lose or accept partial/malformed evidence. | Failure matrix plus atomic envelope validation, timeout log preservation, exact raw identity recomputation, and strict timing/memory validators. | Worker/coordinator/evidence/attempt focused suites | 998 passed, 14 deselected | None. | Complete |
| C1-C2 | Timing was scattered and pair subphases were absent. | `execution_measurement.py`, `pipeline.py`: one synchronized timer, parent spans, exception evidence, and clone/restore/score/release/mutation/record subphases. | `test_execution_measurement.py`; worker phase assertions | 998 passed, 14 deselected | No real timings exist until an authorized GPU run. | Complete |
| C3 | Projection excluded startup/model load. | Exact load-inclusive projection exports every formula component and uses max of 12 real pairs; no-op excluded. | Projection and coordinator tests | 998 passed, 14 deselected | None. | Complete |
| C4-C5 | Peak reset ownership/phase coverage and pre-branch admission were incomplete. | Measurement-only resets; per-phase allocated/reserved peaks including model load; ownership-transfer restore; shape/dtype-derived 22-GiB guard. | Measurement and worker-body tests | 998 passed, 14 deselected | No RTX 3090 memory observation exists. | Complete |
| D1 | Discovery load could permit ambiguous device placement. | `strict_device.py`: exactly one RTX 3090, explicit device/map, no auto/offload/meta, complete device evidence. | `test_strict_device.py` | 998 passed, 14 deselected | Requires independent GPU-host audit before authorization. | Complete |
| D2 | Snapshot identity depended on loader-private metadata. | `snapshot_boundary.py`: public Hub cache metadata, exact 40-SHA, local-only path, incomplete/shard/config/disk checks. | `test_snapshot_boundary.py` | 998 passed, 14 deselected | Model snapshot intentionally not downloaded. | Complete |
| D3 | Tokenizer boundary was unvalidated against the real pinned asset. | Isolated tokenizer-only validation and immutable inventory in `B1_TOKENIZER_ONLY_VALIDATION.json`. | Live tokenizer validation plus inventory/hash audit | 998 passed, 14 deselected | First symlink-mode attempt failed; successful copy-mode attempt is recorded. | Complete |
| E1-E5 | Batch, selected-event, pair, no-op, and replay claims depended on aggregates or incomplete traces. | Actual call recorder plus raw token/call/compaction lists/hashes/mismatches, exact stable pair IDs, mutation reports, and no-op arrays/state hashes. | Call-trace, evidence, worker, coordinator tests | 998 passed, 14 deselected | None. | Complete |
| E6-E8 | Expected generation, exact row identity, and final gate names were incomplete. | `final_contract.py`, coordinator re-verification: three generation comparisons, question/gold/raw/manifest/prompt hashes, exact fail-closed 29-gate map. | `test_final_contract.py`; coordinator tests | 998 passed, 14 deselected | None. | Complete |
| F1-F5 | Flat artifacts could omit failed attempts, logs, envelopes, and provenance. | `attempt_artifacts.py`, entry/coordinator/CLI: attempt-first lifecycle, atomic refusal-to-overwrite writes, journals, commands/logs/timeouts, hashes, Git/software/hardware/invocation provenance. | Attempt, worker-entry, and timeout tests | 998 passed, 14 deselected | Final artifact cannot hash itself; it hashes every pre-final referenced file. | Complete |
| G1-G4 | Contract drift, stale dry-run, and hostile patterns were not centrally audited. | Canonical constants drive validators and dry-run gate listing; closure doc mirrors the list. | Consistency test, both CLI dry-runs, Git/diff/submodule audit, and classified hostile search. | 998 passed, 14 deselected | None. | Complete |

Final validation: compilation passed; 1,012 tests collected; 998 passed and
14 GPU-marked tests were deselected. The final verdict is **READY FOR
INDEPENDENT AUDIT — B2A/GPU REMAIN BLOCKED**.

---

## Independent-audit repair pass (forward-only, on top of `7ef13ae566e7c3e699e5143405baf76a81078edf`)

Full detail, evidence citations, and remaining-gap list:
`docs/B1_INDEPENDENT_AUDIT_REPAIR.md`. Summary table below; historical rows
above are NOT edited or re-marked.

| ID | Confirmed defect | Risk | Root cause | Files changed | Focused tests | Full-suite result | Open questions | Status |
|---|---|---|---|---|---|---|---|---|
| H1 | Worker failure envelope discarded ALL partial evidence (`partial_measurements=None, determinism_policy=None` unconditional) | High | No structured partial-evidence capture existed; worker bodies had no `try/except` around ~280-line bodies | `worker_partial_evidence.py` (new), `b2a_workers.py`, `orchestrator.py`, `attrition.py`, `worker_envelope.py`, `b2a_worker_entry.py` | `test_worker_partial_evidence.py`, `test_orchestrator_partial_failure.py`, +tests in `test_b2a_worker_entry.py`/`test_b2a_workers.py` | 1039 passed, 14 deselected | None for the repaired paths; `manifest_prepare.py`'s 3 bare `except Exception:` not investigated | Complete |
| H2 | `capture_and_parity` phase measured a post-hoc trace comparison, not real capture/parity work; startup/load projection summed only 2 of 5 real one-time phases; no process-launch overhead diagnostic | Medium | Misleading phase name; incomplete phase-sum list; no coordinator-side process timing | `b2a_workers.py`, `final_contract.py`, `b2a_execute.py` | 3 new tests in `test_b2a_execute_coordinator.py`, 1 in `test_b2a_workers.py` | 1039 passed, 14 deselected | H2.3's generic call-nesting spy-test framework not built | Complete for the confirmed defects; H2.3 framework open |
| H3 | Mismatch evidence exported only a bare index, no expected/observed values; Pass-2-invalid `ExampleResult` discarded the actual replayed tokens | High | No canonical mismatch schema; `orchestrator.py` failure-path constructor omitted an already-computed field | `mismatch.py` (new), `b2a_workers.py`, `orchestrator.py` | `test_mismatch.py` (7), 1 new orchestrator test | 1039 passed, 14 deselected | H3.4-H3.6's full ordered-call-entry-at-mismatch and dedicated schema fields on `RKVWorkerResult` itself not added | Partial |
| H4 | Coordinator derived `single_rtx3090_verified` from a bare worker-reported `verified=True`, no raw-field recomputation, no cross-worker agreement check | High (GPU-time; unreachable here) | Gate trusted a self-reported flag instead of recomputing | `strict_device.py`, `b2a_execute.py` | 10 new tests in `test_strict_device.py` | 1039 passed, 14 deselected | No CLI preflight artifact; no typed snapshot-evidence re-validation; H4.5/H4.7 not (re-)verified | Partial |
| H5 | Pre-branch memory guard omits shape-derived branch-horizon K/V growth, query-cache growth, per-token temporary storage beyond a fixed logits/log-softmax term | Medium (GPU-time; unreachable here) | Not repaired this pass | none | none | — | Full componentized estimate (H5.2) not built | Open |
| H6 | Final artifact gates (`attempt_files_verified`, `worker_envelopes_verified`) are existence-only checks (`issubset(existing)`, `.is_file()`) | High (GPU-time; unreachable here) | Not repaired this pass | none | none | — | Canonical content/hash-verifying attempt verifier (H6.2-H6.7) not built | Open |
| H7 | No immutable start/end artifact split; no dedicated post-CUDA device preflight artifact distinct from CPU-safe provenance | Medium | Not repaired this pass | none | none | — | H7.1-H7.5 largely open; `collect_execution_provenance`/`build_attempt_references` already cover a real subset | Open |
| H8 | Hostile audit re-run; CI workflow verified (not modified, already correct) | — | — | (audit only) | — | 1039 passed, 14 deselected; `python -m compileall`, `--collect-only`, both CLI dry-runs, `python -m kvcot --help`, `git diff --check` all exit 0 | H8.2's unified contract-consistency test not built | Partial |

### Round 2 (forward-only, on top of round 1's commit)

Closes the majority of what round 1 left open. Full detail:
`docs/B1_INDEPENDENT_AUDIT_REPAIR.md` §3.

| ID | Confirmed defect | Risk | Files changed | Focused tests | Full-suite result | Open questions | Status |
|---|---|---|---|---|---|---|---|
| H4 (remainder) | No CLI preflight artifact; coordinator trusted `snapshot_evidence["verified"]` + one field | Medium-High (GPU-time) | `cli.py`, `strict_device.py`, `snapshot_boundary.py`, `b2a_execute.py` | 4 new in `test_strict_device.py`, 15 new in `test_snapshot_boundary.py`, 2 new in `test_cli_b2a_calibrate.py` | 1093 passed, 14 deselected | H4.5/H4.7 (network-fallback proof, tokenizer-validation distinction) not re-verified | Complete for H4.1-H4.4/H4.6 |
| H5 | Pre-branch memory guard omitted shape-derived branch-horizon growth | Medium (GPU-time) | `execution_measurement.py`, `b2a_workers.py` | 6 new in `test_execution_measurement.py` | 1093 passed, 14 deselected | None for the repaired components | Complete |
| H6 | Artifact gates were existence-only | High (GPU-time) | `attempt_verification.py` (new), `b2a_execute.py` | 16 new in `test_attempt_verification.py` | 1093 passed, 14 deselected | None for the repaired scope | Complete |
| H7 (remainder) | No immutable start/end split; no dedicated device-preflight artifact | Medium | `cli.py` (`completion.json`; `preflight.json` now real, from H4) | 2 new in `test_cli_b2a_calibrate.py` | 1093 passed, 14 deselected | H7.4's stage-completeness verifier not built | Complete except H7.4 |
| H8.2 | No unified contract-consistency test | Medium | `test_contract_consistency.py` (new) | 10 new tests | 1093 passed, 14 deselected | H8.6's formal call-graph document not produced | Complete |

**Round 2 verdict: B1 FINAL CPU CLOSURE VERDICT: INCOMPLETE — B2A/GPU
REMAIN BLOCKED.** Remaining gaps are narrow and named (H2.2 sub-phase
granularity, H4.5/H4.7, H6.4 progress-duplication detection, H7.4
stage-completeness, H8.6 formal document, `manifest_prepare.py`'s 3
uninvestigated bare excepts) — see `docs/B1_INDEPENDENT_AUDIT_REPAIR.md`
§5. No B2A result exists. No B2B result exists. No real CUDA timing
exists. No RTX 3090 memory measurement exists. No FaithKV method exists.

### Round 3 (forward-only, on top of round 2's commit)

Prompted by a user request to VERIFY round 2's own "remaining gaps" list
rather than accept it at face value. Each item was re-investigated against
the actual code before being touched. Full detail:
`docs/B1_INDEPENDENT_AUDIT_REPAIR.md` §4.

| ID | Confirmed defect | Risk | Files changed | Focused tests | Full-suite result | Open questions | Status |
|---|---|---|---|---|---|---|---|
| `manifest_prepare.py` excepts | Flagged "not investigated" in round 1/2 | — | none (audit only) | none | 1102 passed, 14 deselected | — | Investigated: all 3 valid with exact proof, not blockers |
| H2.2 | No sub-phase timing for the real target-capture-gather/parity computation | Medium | `capture.py`, `pass2.py`, `orchestrator.py`, `b2a_workers.py`, `final_contract.py` | 3 new in `test_capture.py`, 1 new in `test_orchestrator_pair_execution_policy.py` | 1102 passed, 14 deselected | None | Complete |
| H4.5 | Prompt-identity tokenizer resolution not proven local-only | Medium | `manifest_prepare.py`, `b2a_execute.py` | 2 new in `test_b2a_execute_coordinator.py` | 1102 passed, 14 deselected | None | Complete |
| H6.4 | Progress journal double-recorded every named stage (live + post-hoc replay) | Medium | `b2a_worker_entry.py` | 1 new in `test_b2a_worker_entry.py` | 1102 passed, 14 deselected | None | Complete |
| H7.4 | No stage-completeness verifier | Low-Medium | `attempt_verification.py` | 2 new in `test_attempt_verification.py` | 1102 passed, 14 deselected | None | Complete |

One unrelated, pre-existing flake was observed during validation
(`test_math_verifier.py::test_accepted_equivalences`, a subprocess-timeout-
sensitive test in a module last touched 2026-07-19, predating every round
in this document) — confirmed non-reproducible in isolation across
multiple re-runs, not modified, not counted as a regression.

**Round 3 verdict: B1 FINAL CPU CLOSURE VERDICT: INCOMPLETE — B2A/GPU
REMAIN BLOCKED.** Remaining gaps are now only H4.7 (a documentation
distinction) and H8.6 (a formal call-graph document, vs. the equivalent
evidence already cited inline throughout this ledger and the audit
document) — see `docs/B1_INDEPENDENT_AUDIT_REPAIR.md` §6. No B2A result
exists. No B2B result exists. No real CUDA timing exists. No RTX 3090
memory measurement exists. No FaithKV method exists.

## Round 4 (2026-07-21) — final F1–F10 repairs

Round 3's closing claim that "remaining gaps are now only H4.7 … and
H8.6" was an overclaim: an independent audit verified nine further
functional execution-boundary defects. Starting commit:
`419bbc0020b374d6c4a2085a7a04ff293d7ec680`.

| ID | Defect | Files | Focused tests | Status |
|---|---|---|---|---|
| F1 | `failing_stage` populated with the last *completed* stage; `attempt_id`/`last_completed_stage` omitted | `worker_partial_evidence.py` (`WorkerExecutionState`), `b2a_workers.py` | 12 stage-injection tests (`test_final_audit_repairs.py`), updated `test_b2a_workers_real_bodies.py` | Complete |
| F2 | `failed_pair_identities`/`no_op_evidence`/`replay_evidence` and most available `ExampleResult` fields dropped on failure | `b2a_evidence.py` (shared helpers), `worker_partial_evidence.py`, `b2a_workers.py` | 6 preservation tests + structural single-derivation test | Complete |
| F3 | Memory failure records had no `failure_message` | `execution_measurement.py` | memory-failure message/field tests | Complete |
| F4 | Artifact verification incomplete (no top-level parsing, command identity, saved-result equality, typed validation, process outcome, progress ordering, hash recomputation) | `attempt_verification.py`, `b2a_workers.py` (`process_outcome.json`, `text: True`), `b2a_execute.py` | 38 tests in `test_attempt_verification.py` (rewritten fixture + new negatives) | Complete |
| F5 | `final.json` written before `completion.json`; no invocation start timestamp | `b2a_execute.py` (lifecycle + `B2AFinalWriteError` + `final_write_failure.json`), `cli.py` (started_at, write-if-missing completion), `attempt_artifacts.py` (semantic-role reference manifest) | lifecycle + final-write-failure + reference-manifest tests | Complete |
| F6 | Provenance missing RAM/OS/kernel/origin-branch/three-way ancestry; `3c853cf` sole start authority | `attempt_artifacts.py` | provenance content tests | Complete |
| F7 | No explicit requested-device or complete placement boundary in the mandatory final gate | `strict_device.py`, `runtime_evidence.py`, `final_contract.py`, `b2a_execute.py` | 11 placement negatives, requested-device negatives, three-way mismatch | Complete |
| F8 | Coordinator snapshot check weaker than resolver (no index/shard/inventory revalidation) | `snapshot_boundary.py`, `b2a_execute.py` | malformed-index/missing-shard/fake-inventory/wrong-hash/wrong-size/wrong-path/valid-snapshot tests | Complete |
| F9 | Timing/memory contracts tolerated duplicate singleton phases | `final_contract.py` | duplicate-singleton/pair, capture-count, invalid-duration, actual-call-agreement, memory-duplication tests | Complete |
| F10 | H4.7/H8.6 formalities | `docs/B1_TOKENIZER_ONLY_VALIDATION_CLARIFICATION.md`, `docs/B1_FINAL_EXECUTION_CALL_GRAPH.md` | doc-name coverage via `test_final_contract.py` | Complete |

Local validation (round 4, newly observed): `python -m compileall src
tests` clean; 1,187 tests collected; full non-GPU suite **1,173 passed,
14 deselected**; both dry-runs exit 0; `git diff --check` clean.

CI: `.github/workflows/cpu-tests.yml` is the equivalent CPU workflow, but
every recorded run fails without starting — GitHub reports "The job was
not started because your account is locked due to a billing issue." No
completed successful CI run exists for any commit on this branch.

**Round 4 verdict: B1 FINAL CPU CLOSURE VERDICT: INCOMPLETE — B2A/GPU
REMAIN BLOCKED** (sole open item: independent CI evidence, blocked by the
GitHub account lock). No GPU inference was run. No model weights were
downloaded. No B2A result exists. No B2B result exists. No real CUDA
timing exists. No RTX 3090 memory measurement exists. No FaithKV method
exists.
