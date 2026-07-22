# Plan and status

## B2A-R2 FORENSIC PAIR-RECORD PERSISTENCE REPAIR, AUDIT ROUND 2 (2026-07-22)

An independent audit found round 1 (immediately below) left
`verify_pair_record_artifacts` never-fatal -- a future V2 attempt could
still be reported successful with a missing, incomplete, duplicated,
mismatched, or corrupt `rkv/pair_records.json`/`rkv/scientific_summary.json`.
Fixed, CPU-only: `overall_passed` now ANDs in a third gate factor,
`scientific_pair_artifacts_verified`, derived from
`verify_pair_record_artifacts` and `isinstance(rkv, RKVWorkerResultV2)` --
failure now produces `exit_code=2`/`outcome="gate_failed"`, never a false
success. `parse_rkv_worker_result` now rejects (rather than silently
misclassifies as legacy V1) a payload labeled `schema_version=
"rkv_worker_result.v2"` that is missing `pair_records`. Ten new
coordinator-level tests assert `overall_passed`/exit code/`completion.json`
directly. No GPU, no re-run, `FINAL_MANDATORY_GATE_CONDITIONS` unmodified.
Full detail: `docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md`
§10.

```text
B2A-R2 FORENSIC CLOSURE VERDICT:
PAIR-RECORD PERSISTENCE REPAIRED -- READY FOR INDEPENDENT REVIEW; B2A-R3/B2B REMAIN BLOCKED
```

## Prior status: B2A-R2 forensic pair-record persistence repair, round 1 (2026-07-22)

CPU-only durable-artifact repair -- no GPU, no inference, no re-run. B2A-R1
and B2A-R2 have both already executed (see the B2A-R1/R2 section
immediately below, now historical) -- any wording elsewhere in this
repository implying B2A has never run refers to a status prior to
2026-07-22 and is superseded here.

B2A-R2 (`fb6f5081d47f45f4b4f9258c25e6883d`, qualified row
`test/number_theory/820.json`) ran to completion: 27/28 legacy-gate
conditions passed; the sole failure, `runtime_within_limit` (5.01 projected
GPU-hours vs. the 4.00-hour limit), remains the frozen verdict, unchanged
by this repair: `B2A-R2 FINAL VERDICT: FAIL -- B2B BLOCKED`.

A post-run audit of the preserved archive found that "full per-pair
evidence" was too broad: execution accounting, identities, and mutation/
parity evidence survived, but the twelve real interventions' scientific
outcomes (`swap_gain`, 48-value `baseline_per_token_nll`/
`swapped_per_token_nll`) were never exported by `RKVWorkerResult` and are
unrecoverable from the archive (the GPU instance is gone; no value was
estimated, inferred, or backfilled). Repaired: `RKVWorkerResult` split
into structurally-versioned `RKVWorkerResultV1` (legacy, unmodified,
still parseable) / `RKVWorkerResultV2` (adds a REQUIRED `pair_records`
field); `run_rkv_worker` now populates it directly from
`ExampleResult.pair_records`; every future successful attempt durably
writes `rkv/pair_records.json` and `rkv/scientific_summary.json` (a pure
CPU summary including a dependency-free tie-aware Spearman correlation);
a new, deliberately-standalone
`kvcot.discovery.attempt_verification.verify_pair_record_artifacts`
checks population completeness, identity agreement, and exact
recomputation -- wired into the live coordinator as additional,
never-fatal evidence, never a new mandatory gate condition. Full detail:
`docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md`;
`docs/B2A_R2_RESULT_2026-07-22.md` §8 (dated clarification; also corrects
a decimal-GB-labeled-as-GiB documentation arithmetic error in that
document's memory table -- does not change any verdict).

**No B2A-R3 attempt is authorized by this document. B2B remains blocked.
No FaithKV method exists.**

```text
B2A-R2 FORENSIC CLOSURE VERDICT:
PAIR-RECORD PERSISTENCE REPAIRED -- READY FOR INDEPENDENT REVIEW; B2A-R3/B2B REMAIN BLOCKED
```

## Prior status: B2A-R1 failure closure and B2A-R2 pre-registration (2026-07-22)

B2A-R1 (the single attempt CLAUDE.md §1c authorized) executed against
`example_index=0`: both FullKV and R-KV workers ran (return code 0 each),
so the attempt is scientifically consumed, but it produced ZERO R-KV
compaction events (prompt=105 tokens, generated=449 tokens, far under
budget=1024) -- an ineligible calibration that tested no eviction at all.
Full evidence and root-cause analysis:
`docs/B2A_R1_FAILURE_AND_B2A_R2_PROTOCOL_2026-07-22.md`;
`docs/evidence/B2A_R1_ATTEMPT_INDEX_2026-07-22.json`.

Two repairs made before any B2A-R2 inference, neither touching scientific
configuration: (1) the coordinator's `build_runtime_projection` no longer
raises on an insufficient real-pair count -- it resolves to an explicit,
never-fabricated "unavailable" outcome, failing the runtime gate closed;
(2) the MATH-500 answer verifier's calling convention into `math-verify`
was corrected (confirmed root cause: bare, non-`\boxed{}` compound LaTeX is
unreliably parsed by that library's fallback path) -- a general fix, not
tuned to the specific observed answer.

B2A-R2 is pre-registered: a committed, deterministically-ordered 12-row
MATH-500 candidate manifest (`configs/discovery/b2a_r2_candidate_manifest.json`),
FullKV-only qualification (`kvcot qualify-b2a-row`, R-KV never imported)
against 10 frozen conditions, stopping at the first qualified row -- no
qualified row means immediate stop. The qualified row, if any, is frozen
into a replacement one-example manifest only by a fail-closed function
that rejects any hash/identity mismatch or row substitution. Exactly one
further `b2a-calibrate --execute` attempt is then authorized, under the
unchanged §1c/§4c settings.

No B2A-R2 result exists yet as of this entry. No B2B result exists. No
FaithKV method exists. B2B and any FaithKV method implementation remain
blocked pending a separate, future, independent authorization.

## Historical status

## Prior status: B2A one-example GPU authorization (2026-07-22)

B1 CPU closure is complete; CPU CI is green at run
[29892965613](https://github.com/asad073-ui/Faithkv/actions/runs/29892965613)
(commit `a4f6e4298eba10d037ca7e6570fe6d69aad2472f`). Bounded GPU mechanical
validation is complete on the Vast RTX 3090 host: 12 of 14 collected
GPU-marked tests pass (replay identity, stock-vs-patched no-op parity,
cross-example state isolation, snapshot restoration, R-KV schedule
prediction, and more). The remaining two
(`test_probe_stability_gpu.py`'s f=1 stability checks — FullKV 17/20, R-KV
b256 15/20, against the 0.90 threshold) are a preserved, unmodified
historical Stage 0 result for the archived Qwen-1.5B/GSM8K early-answering
protocol; they are non-blocking for B2A, which exercises a disjoint
mechanism (greedy decoding throughout, no early-answering control suffix,
`DeepSeek-R1-Distill-Llama-8B`/MATH-500). See
`docs/B2A_ONE_EXAMPLE_GPU_AUTHORIZATION_2026-07-22.md` and CLAUDE.md
§1c/§4c for the full authorization and its exact scope: exactly one B2A
(`b2a-calibrate --execute`) engineering-calibration attempt, one frozen
example, hard 22 GiB / 4.00 GPU-hour limits.

No B2A result exists yet. No B2B result exists. No FaithKV method exists.
B2B and any FaithKV method implementation remain blocked pending a
separate, future, independent authorization.

## Prior status: Phase B1 final CPU closure, round 4 — B1 FINAL CPU CLOSURE VERDICT: INCOMPLETE — B2A/GPU REMAIN BLOCKED (2026-07-21)

**Round 4 (2026-07-21).** The round-3 paragraph below claimed
only two documentation formalities remained — that was an overclaim. A
final independent audit confirmed nine further functional
execution-boundary defects (F1–F9: wrong failing-stage attribution,
dropped partial R-KV evidence, missing memory failure messages, incomplete
artifact verification, `final.json` written before `completion.json`,
incomplete provenance, a narrower-than-claimed device/placement gate,
weak snapshot revalidation, duplicate-tolerant timing/memory contracts)
plus the two formalities (F10). All ten are repaired and locally
validated (1,187 collected; 1,173 passed / 14 deselected; dry-runs exit
0). The sole open item is independent CI evidence: GitHub Actions is
locked at the account level (billing), so no run can start. See
`docs/B1_INDEPENDENT_AUDIT_REPAIR.md` §8/§9 (authoritative),
`docs/B1_FINAL_REPAIR_LEDGER.md` "Round 4", and
`docs/B1_FINAL_EXECUTION_CALL_GRAPH.md`.

Round 3 (historical, overclaim corrected above): three forward-only
repair passes closed Gates H1-H7 in full and H8.2; it claimed the only
remaining items were H4.7 and H8.6, "both audit-formality items rather
than functional defects" — incorrect, per the round-4 findings.
Non-GPU suite at round 3: 1102 passed, 14 deselected.

No B2A result exists. No B2B result exists. No RTX 3090 timing exists. No
FaithKV method exists. Independent audit is required before any GPU
authorization.

## Current status: Phase B1 execution-boundary closure — focused completion pass on B1B-R4.1 — INCOMPLETE; GPU/B2A/B2B STILL BLOCKED

**Phase B1 execution-boundary closure (2026-07-20, CHANGELOG.md) is a
forward completion pass on top of the already-pushed B1B-R4.1 commit**
(`4e45beac1912a0a7852a034420732a10d0d703e7`, itself already self-recorded
INCOMPLETE), fixing five concrete defects a fresh evidence-based audit (16
numbered claims, all 16 confirmed against the actual current code)
found: `reset_patched_state` no longer secretly resets CUDA peak-memory
stats (the R-KV worker's Pass-2 state construction was silently wiping
Pass 1's already-accumulated peak); baseline/swapped branch evaluation now
releases down to a compact score object the instant each branch finishes,
not just the initial snapshot clone (the prior pass's own gap, closed one
layer deeper); the semantic swap on an already-owned snapshot clone no
longer clones the cache a second time; the `semantic_swap_parity` gate
condition now requires POSITIVE attempted/passed counts instead of
absence-of-a-failure-record; three new gate conditions replace a bare
per-event pair count with exact, duplicate-detecting pair-identity
accounting. 20 new CPU tests; full non-GPU suite: 970 passed, 0 failed.
Full detail, including an HONEST itemized list of everything still open
(largest: `Pass2Result.target_captures` still retains full capture tensors
through all 12+1 pair evaluations — no `CompactBranchTarget` conversion
was built; plus no CUDA-synchronized timing, no VRAM phase split, no full
worker-level success test, no Hub snapshot resolver, no strict single-GPU
load path, no immutable attempt-directory architecture, and more):
`docs/B1_EXECUTION_BOUNDARY_FINAL_CLOSURE.md`. **Status: B1
EXECUTION-BOUNDARY CLOSURE VERDICT: INCOMPLETE — B2A/GPU REMAIN BLOCKED.
No discovery result exists. No method exists.**

## Prior status: Phase B1B-R4.1 — focused completion amendment on B1B-R4 — INCOMPLETE; GPU/B2A/B2B STILL BLOCKED

**Phase B1B-R4.1 (2026-07-20, CHANGELOG.md) is a forward completion pass on
top of the already-pushed B1B-R4 commit** (`4d7971b7b09c004c4670bfde
3939416ab550ea71`), fixing seven concrete defects an evidence-based audit
of the actual current code found (not the full list a much larger task
brief assumed — several assumed defects, e.g. non-Pydantic schemas, turned
out to already be fixed): one authoritative Pass-1/Pass-2 provenance state
(removing a Pass-2 shadow `LayerProvenance` track); selected-event count
now derived from the frozen Pass-1 plan instead of surviving pair records;
structured per-pair failure evidence actually populated (was always empty
in production); the capture-minimization bound now enforced in production,
not only in its own test file; baseline/swapped branch snapshots now
released sequentially, proven via `weakref`, instead of both live at once;
semantic-swap parity/byte-delta evidence now derived from the real
mutation report instead of hard-coded `True`/`0`; `PYTHONHASHSEED` now set
on the subprocess environment before worker launch instead of ineffectively
inside the already-running process. One new gate condition
(`semantic_swap_parity`) was added. Every repair has a new CPU test (25 new
tests total); full non-GPU suite: 950 passed, 0 failed. Full detail,
including an HONEST, itemized list of everything the originating task
brief asked for that this pass did NOT implement (CUDA-synchronized
timing, model-load VRAM phases, a strict single-GPU load path, a Hub
snapshot identity resolver, an immutable attempt-directory architecture,
the remaining gate conditions, and more):
`docs/B1B_R4_1_FINAL_CLOSURE.md`. **Status: B1 FINAL CLOSURE VERDICT:
INCOMPLETE — B2A/GPU REMAIN BLOCKED. No discovery result exists. No method
exists.**

## Prior status: Phase B1B-R4 — final executable, measurement, and worker-evidence closure — IMPLEMENTED FOR REVIEW; GPU/B2A/B2B STILL BLOCKED

**Phase B1B-R4 (2026-07-20, CHANGELOG.md) repairs the remaining defects
found during an independent audit of the merged PR #19 (B1B-R3, commit
`fa117046bea2a2c492e17cd91276b2e3c6d59f7f`)**: FullKV sampling replaced
with exact greedy generation reusing R-KV Pass 1's own loop; framework
determinism applied and recorded independently in both worker processes;
`NoOpMode` now actually controls pair construction
(`kvcot.discovery.orchestrator.PairExecutionPolicy`) instead of only
documenting intent; five trajectory/parity conditions derived
independently instead of copied from one umbrella boolean; resolved-vs-
requested revision read back via `transformers`' own commit-hash
attributes; batch size/parameter placement/one-example scope derived from
real observations; per-pair timing measured individually (never an
aggregate bucket multiplied by 144); branch-restored compaction history
reconstructed from the snapshot instead of reset; VRAM gate uses
`max(allocated, reserved)`; the weight-cache guard scoped to manifest
preparation only; partial FullKV evidence preserved on R-KV failure;
durable per-attempt worker envelopes; collision-resistant artifact naming;
bounded selected-capture minimization; one canonical FullKV/R-KV worker
API exercised by CPU tests against injected fakes. This pass's own
adversarial self-review found and fixed three further defects (documented
in `docs/B1B_R4_FINAL_B2A_CLOSURE.md` §5). Full detail:
`docs/B1B_R4_FINAL_B2A_CLOSURE.md`. **Status: B1B-R4 implemented, ready for
independent CPU audit. GPU, B2A, and B2B remain blocked. No discovery
result exists. No method exists.** The full 12-real+1-no-op success path
is not exercised by any `run_rkv_worker`-level CPU test (a fake-model
fixture limitation, documented in that document's §4) — the orchestrator-
level success path IS exercised extensively elsewhere.

## Prior status: Phase B1B-R3 — executable B2A boundary and evidence producer — IMPLEMENTED FOR REVIEW; GPU/B2A/B2B STILL BLOCKED

**Phase B1B-R3 (2026-07-20, CHANGELOG.md) repairs twelve defects found
during an independent audit of the merged PR #18 (B1B-R2, commit
`7034e46b516eff656b5508d9253ee02b13405f95`)**: invalid `RKVPolicy`
construction, an unresolvable prompt identity, a stub answer verifier,
split real-model provenance state, per-token (rather than per-branch)
snapshot restoration, inconsistent swap bookkeeping, hard-coded B2A
evidence, a FullKV/R-KV single-process conflict, ambiguous no-op
accounting, a missing failure artifact, oversized selected captures
(NOT fully repaired — see below), and frozen-vs-runtime R-KV configuration
drift. Also resolves `configs/discovery/b2a_one_example_manifest.json`'s
prompt identity for real via the new `kvcot prepare-b2a-manifest` command
(CPU-only: one pinned MATH-500 row, the pinned tokenizer's config files
only — no model weights), and corrects a non-reproducible `raw_content_hash`
found in the process. `kvcot b2a-calibrate --execute`'s complete subprocess
coordinator/worker architecture and evidence producer are now real,
reviewable code, exercised end-to-end on CPU via a mocked worker runner
(`tests/unit/discovery/test_b2a_execute_coordinator.py`) — but every path
requiring CUDA is still never invoked. No new `CLAUDE.md` exception was
needed. Full detail: `docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md`. **Status:
B1B-R3 implemented, ready for independent CPU audit. GPU, B2A, and B2B
remain blocked. No discovery result exists. No method exists.** Defect 11
(further capture-size tightening) was assessed and deliberately deferred,
not silently claimed complete — see that document's §2 Defect 11 entry.

## Prior status: Phase B1B-R2 — real-model boundary and B2A preflight — IMPLEMENTED FOR REVIEW; GPU/B2A/B2B STILL BLOCKED

**Phase B1B-R2 (2026-07-20, CHANGELOG.md) repairs eight defects found
during independent review of B1B-R1** (device/dtype parity, target-only
bounded capture, complete-`ModelStateSnapshot` branching, exact
prefill/decode call boundaries, a fully frozen/hashed discovery
configuration, a frozen MATH-500 dataset revision plus one-example
manifest, corrected 144-branch accounting, and a mandatory-field B2A gate),
and adds the one-example-only `kvcot b2a-calibrate` command
(`--dry-run` fully exercised; `--execute`'s code path is implemented but
never invoked — every precondition fails closed on this CPU-only, no-model
build). No new `CLAUDE.md` exception was needed — this pass stays inside
the CPU-side harness architecture §1b/§4b already authorized. Full detail:
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md`. **Status: B1B-R2
implemented for review. GPU, B2A, and B2B remain blocked. No discovery
result exists. No method exists.**

## Prior status: Phase B1B-R1 — B1A defect repairs and B1B CPU harness architecture integration — IMPLEMENTED FOR REVIEW; GPU STILL BLOCKED

**Phase B1B-R1 (2026-07-20, CHANGELOG.md) repairs six B1A defects found
during independent review of PR #16, and adds a new, dated `CLAUDE.md`
§1b/§4b exception that narrowly authorizes CPU-side B1B **harness
architecture** work (Pass 1/Pass 2 orchestration, branch construction/
evaluation, attrition accounting, `plan-discovery --dry-run`, a
documentation-only future B2A contract) — exercised only against
dependency-injected synthetic/deterministic components in CPU tests, never
a real model. No model inference or GPU use is authorized by this
exception; B2A (GPU calibration) and B2B (the bounded discovery pilot)
still require their own separate, future, dated authorization, exactly as
§1a already required. Full detail:
`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`. **Status: B1A repairs and B1B
CPU harness architecture implemented for review. GPU, B2A, and Vast.ai
remain blocked. No discovery result exists. No method exists.**

**Phase B0.5-R2.2 (2026-07-19, CHANGELOG.md) does two things in one
consolidated pass.** First, it reconciles a real (if narrow) contradiction:
`CLAUDE.md`'s original blanket "no 7B support" line technically
contradicted the already-selected Llama-8B discovery operating point
(`docs/b0_5_decision.json`'s `selected_operating_point`, chosen back in
Phase B0.5) — resolved with a new, dated `CLAUDE.md` §1a/§4a exception that
authorizes CPU-side infrastructure only for `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
+ MATH-500, explicitly not a method, not GPU/inference authorization, and
not general benchmark-suite expansion; the original Qwen-1.5B/GSM8K
pipeline (`CLAUDE.md` §1/§4's original tables, `configs/lock.yaml`) is
unchanged. Second, it implements every B1A CPU prerequisite
`docs/B0_5_R2_1_FINAL_PROTOCOL.md` §11/§14 had named as outstanding, plus
repairs three further defects found during this pass: the
`first_affected_logit_absolute_position` schema field name conflated an
input position with a logit-target position (now two explicit fields);
depth-stratum sampling assigned each event's depth directly from its
chronological draw order (now independently permuted, removing the
layer-depth/event-time confound); and entropy/logit-margin were classified
"mandatory" without ever being operationally defined (now frozen exactly,
`kvcot.discovery.uncertainty`). Full detail:
`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`.

**Status: B0.5-R2.2 authority reconciliation complete; B1A CPU
prerequisites implemented and CPU-validated. B1B/B2A/B2B/GPU/Vast.ai
remain unauthorized.** New code, all CPU-tested, no GPU used: architecture-
aware R-KV monkeypatch dispatch (`kvcot.discovery.dispatch`, wired into
`kvcot.generation.policies`), a MATH-500 symbolic-equivalence verifier
isolated per-comparison in a child process with a frozen 5-second timeout
(`kvcot.utils.math_verifier`), the active discovery pairwise-provenance
schema (`kvcot.discovery.schemas`, `schema_version="b0_5_r2_2.v1"`),
deterministic sampling utilities with golden-vector tests
(`kvcot.discovery.sampling`), a per-instance read-only capture-wrapper
prerequisite (`kvcot.discovery.capture`), a fixed-shape within-head swap
primitive (`kvcot.discovery.swap`), and a strengthened complete-branch-output
no-op control (`kvcot.discovery.branch_eval`). No discovery hypothesis
result exists — none is claimed. `CLAUDE.md` §1/§4's original Qwen-1.5B
freeze is unchanged; no GPU run of any kind is authorized by this phase.

## Prior status: B0.5-R2.1 final protocol correction ran — READY FOR B1A PREREQUISITE IMPLEMENTATION (CPU prerequisites only; no GPU authorized)

**Phase B0.5-R2.1 (2026-07-19, CHANGELOG.md) is the final B0.5 protocol
correction.** It fixed an off-by-one timing defect in B0.5-R2's branch
estimand: the forward call that consumes the event token `x_t` already
produces the logits predicting `x_{t+1}` *during the same call the swap
fires in*, so the swap cannot affect them — `x_{t+1}` must be fed
identically into both branches as one unscored "bridge" token first, and
the 48-token scored window starts one token later, at `x_{t+2}`
(`bridge_tokens=1`, `scored_horizon=48`,
`minimum_future_tokens_after_event=49`). It also froze exact, reproducible
SHA-256-seeded `random.Random` sampling algorithms for event, layer,
KV-head, evicted-candidate, and donor selection (previously an
unrestricted hash for layer/head, which did not actually guarantee
early/middle/late depth coverage, and a plain ascending-position tie-break
for candidates/donors, which was systematically edge-biased) — the
repaired layer rule restricts each selected event's draw to its own third
of the model's depth so coverage is a real guarantee, not a probabilistic
tendency. Gate 10 is repaired to nest Spearman association testing
per-example (median of per-example `|rho|`, never pooled across examples)
across eight named mandatory deployable signals, with an explicit
8-evaluable-example floor per signal and a mandatory no-op control
(replacing a donor with its own K/V must produce exactly zero change) —
the outcome set is expanded from pass/fail to
DISCOVERY-SUPPORTING/NOT DISCOVERY-SUPPORTING/**NOT ADJUDICABLE**, so a
data-thinness problem is never misreported as a negative finding. Full
correction: `docs/B0_5_R2_1_FINAL_PROTOCOL.md`. B0.5-R2's fixed-shape
within-head swap, its capture-strategy wrapper, and its aggregation
hierarchy are all unaffected and remain current.**

**B0.5-R2.1 VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION**
(`docs/B0_5_R2_1_FINAL_PROTOCOL.md` §14, `docs/b0_5_decision.json`
`b0_5_r2_1_verdict`) — **this authorizes only B1A: CPU-side prerequisite
implementation (MATH-500 verifier, architecture-aware dispatch, the
repaired pairwise provenance schema with timing fields, the repaired
per-instance capture wrapper, the frozen sampling algorithms, the mandatory
no-op control's CPU unit test, CPU tests generally). It does NOT authorize
B1B, GPU use, model inference, or any method implementation.** `CLAUDE.md`
§4's model freeze (`DeepSeek-R1-Distill-Qwen-1.5B` only) still requires a
separate dated amendment before any GPU run of a later phase.

## Prior status: B0.5-R2 dense-cache repair ran — READY FOR B1A PREREQUISITE IMPLEMENTATION (superseded — see B0.5-R2.1 above)

**[Superseded by B0.5-R2.1 above — the branch-timing definition, the
sampling rule, and gate 10 described just below were found to contain an
off-by-one defect, an under-specified/edge-biased sampling rule, and a
pooled (not per-example-nested) association test, respectively. Preserved
verbatim as the historical record.]**

**Phase B0.5-R2 (2026-07-19, CHANGELOG.md) audited B0.5-R's selected
intervention against the pinned R-KV source and the installed
`transformers==4.55.4` cache implementation and found it not
representable: "equal-byte add-back" and "retained-only physical
ablation" both required changing cache size "at one (layer, kv_head) pair
only," but a dense `(batch, num_kv_heads, seq_len, head_dim)` tensor has
one `seq_len` shared by every head at a layer — a slot cannot be
added/removed for one head while leaving the others unchanged. B0.5-R's
capture-hook claim (a hook reading `final_score`/`indices` as internal
locals inside `R1KV.update_kv`) was also found not implementable as
described. Both are repaired: a fixed-shape **within-head swap**
(`key_cache[L][0,h,r_slot,:] = captured_key_e`, net physical bytes always
exactly 0) captured via a per-instance before/after wrapper plus
independent, exact recomputation of R-KV's real windowed score formula
(a previously unflagged defect was also found here: R-KV's own persisted
`kept_final_scores` uses a *different*, unwindowed formula than the one
that actually drives eviction — recomputation must replicate the real
formula, not the persisted one). Also repaired: a mandatory two-pass
capture plan (event eligibility is only knowable after natural generation
completes, so a second, token-identical instrumented replay pass is
required — a real cost previously absent from the cost model), and gate
10 (previously did not require any actual positive ranking reversal to
exist). Full repair: `docs/B0_5_R2_DENSE_CACHE_REPAIR.md`. B0.5-R's
corrected decision unit, B1A-1/B1A-2 prerequisites, and B0's method-pivot
verdict are unaffected.**

**B0.5-R2 VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION**
(`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §21, `docs/b0_5_decision.json`
`b0_5_r2_verdict`) — **this authorizes only B1A: CPU-side prerequisite
implementation (MATH-500 verifier, architecture-aware dispatch, the
repaired pairwise provenance schema, the repaired per-instance capture
wrapper, CPU tests). It does NOT authorize B1B, GPU use, model inference,
or any method implementation.** `CLAUDE.md` §4's model freeze
(`DeepSeek-R1-Distill-Qwen-1.5B` only) still requires a separate dated
amendment before any GPU run of a later phase.

## Prior status: B0.5-R protocol repair ran — READY FOR B1A PREREQUISITE IMPLEMENTATION (superseded — see B0.5-R2 above)

**[Superseded by B0.5-R2 above — the selected intervention design and the
capture-hook claim described just below were found not representable/not
implementable. Preserved verbatim as the historical record.]**

**Phase B0.5-R (2026-07-19, CHANGELOG.md) audited B0.5 against the pinned
R-KV source and found two load-bearing assumptions false: the "fixed
128-token block" experimental unit (real unit is a single per-layer,
per-KV-head cache slot — `divide_length=128` is only a compaction cadence,
never an eviction unit) and the shadow-FullKV KV-recovery method (does not
reconstruct R-KV's true pre-eviction state for any event past the run's
first compaction; a read-only pre-compaction capture hook is required
instead). Both are repaired, along with an intervention-design fix
(equal-byte add-back vs. retained-only physical removal, never pooled;
zeroing replaced by physical tensor-slice removal), an exact numeric gate
table (replacing every vague threshold), and confirmation of two concrete
B1A prerequisites this repository does not yet have: architecture-aware
R-KV monkeypatch dispatch (`src/kvcot/generation/policies.py` currently
calls only `replace_qwen2` unconditionally — Candidate A's Llama-8B
checkpoint would silently run unpatched, mislabeled as `rkv_b1024`) and a
MATH-500 symbolic-equivalence verifier (`src/kvcot/utils/answers.py` is
numeric-only). Full repair: `docs/B0_5_PROTOCOL_REPAIR.md`. B0's method-
pivot verdict is unchanged.**

**B0.5-R VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION**
(`docs/B0_5_PROTOCOL_REPAIR.md` §19, `docs/b0_5_decision.json`
`b0_5_r_verdict`) — **this authorizes only B1A: CPU-side prerequisite
implementation (MATH-500 verifier, architecture-aware dispatch,
decision/provenance schema, read-only compaction instrumentation, CPU
tests). It does NOT authorize B1B, GPU use, model inference, or any
method implementation.** `CLAUDE.md` §4's model freeze
(`DeepSeek-R1-Distill-Qwen-1.5B` only) still requires a separate dated
amendment before any GPU run of a later phase.

## Prior status: B0.5 discovery-protocol feasibility gate ran — READY FOR B1 DISCOVERY-HARNESS IMPLEMENTATION (superseded — see B0.5-R above)

**[Superseded by B0.5-R above — the "fixed 128-token block" unit and the
shadow-FullKV KV-recovery method described just below were found false
against the pinned R-KV source. Preserved verbatim as the historical
record.]**

**Phase B0.5 (2026-07-19, CHANGELOG.md) is documentation-only** — it does
not implement a compression method, does not implement the discovery
harness, uses no GPU, downloads no model weights or datasets, and does not
reverse the B0 method-pivot verdict below. It audited B0 (upheld all three
verdicts — `docs/b0_5_decision.json` `b0_audit_verdict`), defined an
untested discovery hypothesis (`docs/B0_5_DISCOVERY_PROTOCOL.md` §0:
"unexplained causal false negatives" — a deployed R-KV policy evicting
blocks with high counterfactual future utility that cheap deployable
signals miss), preregistered a full future protocol for testing it
(`docs/B0_5_DISCOVERY_PROTOCOL.md`), and ran an operating-point feasibility
audit (`docs/B0_5_FEASIBILITY_AUDIT.md`) that selected **Candidate A —
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B` + MATH-500 + R-KV budget 1024**
(primary-source "lossless at 34%/1024 tokens" evidence directly from the
pinned `third_party/R-KV` submodule's own README/paper) over a GSM8K b128
repeat (forbidden — already retired) and two other candidates (AIME-24:
worse population/cost tradeoff; Qwen-7B: weaker evidence tier). Projected
GPU cost: 2.4-3.4 hours (safety-factored), under the 4-hour ceiling.

**B0.5 VERDICT: READY FOR B1 DISCOVERY-HARNESS IMPLEMENTATION**
(`docs/b0_5_decision.json`) — **this authorizes only a later,
user-requested, CPU-developed B1 harness. It does NOT authorize any GPU
use, model inference, or method implementation.** A real blocker remains
even for that harness's eventual GPU run: `CLAUDE.md` §4 freezes the model
as `DeepSeek-R1-Distill-Qwen-1.5B` only, and Candidate A uses a different
model — a dated `CHANGELOG.md`/`CLAUDE.md` amendment is required **before**
any such run and is not granted by B0.5. See
`docs/B0_5_DISCOVERY_PROTOCOL.md`, `docs/B0_5_FEASIBILITY_AUDIT.md`,
`docs/B0_5_SEARCH_LOG.md`, `docs/b0_5_decision.json`.

## Prior status: B0 method-pivot gate ran and FAILED — METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET

**Phase B0 (2026-07-19, CHANGELOG.md) ran the method-pivot specification
and adversarial method-novelty gate** on three candidate methods targeting
a prospective causal-false-negative failure mode (`docs/METHOD_PIVOT_SPEC.md`
§5 — an untested hypothesis, not a finding): M1 residual causal-utility
protection, M2 interaction-aware dynamic rescue, M3
faithfulness-constrained memory allocation. Against a 2026-07-19-cutoff
adversarial search (`docs/METHOD_NOVELTY_MATRIX.md`,
`docs/B0_SEARCH_LOG.md`, `docs/method_novelty_matrix.json`):

- **M1: PARTIAL — INSUFFICIENT METHOD NOVELTY** (ablation supervision →
  ArborKV arXiv:2605.22106; residual-correction architecture → IntentKV
  arXiv:2606.09916; counterfactual-ablation-to-policy pipeline → ThinKV
  arXiv:2510.01290v2; protected partition → CASK/VaSE/arXiv:2605.18053);
- **M2: KILLED** (cache-state-conditional sequential eviction is
  ForesightKV arXiv:2602.03203's MDP and Neural Garbage Collection
  arXiv:2604.18002; R-KV itself already rescores per compaction);
- **M3: PARTIAL — INSUFFICIENT METHOD NOVELTY** (allocation machinery →
  ReasonAlloc arXiv:2606.11164 and the Ada-KV/LKV lineage; a new
  constraint metric is insufficient by the predeclared standard).

**METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET. B1 is not
permitted; no GPU, no MATH-500 work, no method implementation is
authorized.** The only permitted next activity remains method design: a
future candidate must contain a cache operation absent as an operation
class from `docs/METHOD_NOVELTY_MATRIX.md` §5, then re-clear a fresh
B0-style gate.

## Prior status: A3 diagnostic novelty kill-check DOES NOT SURVIVE; PHASE B BLOCKED

**Phase A3 (2026-07-19, CHANGELOG.md) found that CASK (arXiv:2604.10900),
released 2026-04-13, independently implements the fixed-generated-trace /
teacher-forced / cache-policy-varying replay diagnostic this repository's
narrower novelty claim (N1) rested on — confirmed by direct inspection of
CASK's official evaluation code, not just its abstract
(`docs/RELATED_WORK_MATRIX.md` §6.1, §8).** Early answering itself is
independently non-novel since Lanham et al. 2023 (arXiv:2307.13702). No
paper was found that combines both (KV-cache-policy replay + early-
answering/omitted-suffix intervention) under an accuracy-neutral gate with
realized-memory matching and held-out per-example mechanism classification —
that specific empirical intersection remains open, but per the project's
predefined rule it is an application of known ingredients, not a standalone
method contribution, so the overall verdict is still negative:

**DIAGNOSTIC SURVIVAL VERDICT: DOES NOT SURVIVE — PHASE B: BLOCKED —
DIAGNOSTIC NOT NOVEL.** Full matrix: `docs/RELATED_WORK_MATRIX.md`; search
log: `docs/A3_SEARCH_LOG.md`; machine-readable: `docs/related_work_matrix.json`.

This is layered on top of, and does not reverse, the pre-existing GSM8K
b128 status below — the operating point was already retired on independent
(accuracy) grounds before this literature check ran.

## Prior status (2026-07-19): protocol-v3 GSM8K b128 gate FAILED; GSM8K b128 retired; hypothesis `not_tested`

The implementation is complete and its GPU correctness gates have passed, but
the pilot has **not** reached the §1 research question. The protocol-v3
natural R-KV accuracy gate ran on the full 50-pair GSM8K calibration manifest
and failed: FullKV answered 33/50 (66%) correctly, natural R-KV b128 13/50
(26%) — a 40pp drop past the 0.10 pilot ceiling
(`results/decisions/early_gap_v3_b128_accuracy_gate.json`: `gate_passed:
false`). The fixed-trace analysis path exited before computing any PSS/CPSS,
so **no protocol-v3 PSS/CPSS decision exists and `hypothesis_status` remains
`not_tested`** — the research hypothesis is neither supported nor refuted; it
has not been tested.

The GSM8K + `DeepSeek-R1-Distill-Qwen-1.5B` + b128 operating point is
**retired** as structurally unviable — FullKV traces on this manifest run
276–847 generated tokens (median ~440), leaving no fixed budget that is both
accuracy-plausible and meaningfully compressing. No further GSM8K b128/b160
runs are planned.

Full detail and provenance live in the docs updated alongside this entry:
`README.md`, `CHANGELOG.md` (2026-07-19), `docs/EXPERIMENT.md` §11, and
`docs/GPU_VALIDATION_PLAN.md` (2026-07-19 note). This file is the roadmap
summary; those are the source of truth for the numbers.

## Development model

This repository is developed and maintained on a CPU-only, no-GPU machine
(`pytest -m "not gpu" tests/`, `--dry-run`). GPU-dependent work runs on a
rented host and is synced back as committed artifacts. GPU code *has* now been
executed on such a host — the correctness gates, the protocol-v2 fixed-trace
screen (returned `screen_valid=false`), and the failed protocol-v3 natural
accuracy gate above — so the earlier "no GPU code has been executed" status is
obsolete.

## What's done

- Upstream audit (`docs/UPSTREAM_AUDIT.md`): H1-H8 confirmed with
  file:line citations, plus grounding for `mix_lambda`/`retain_ratio`/
  `retain_direction`, plus the `retain_ratio` inertness finding.
- Full package (`src/kvcot/`): schemas, config, data/manifest freezing
  (real GSM8K + MATH-500 data actually downloaded), answer extraction,
  think-span parsing, metrics/stats (sign convention, Pratt zeros, f=0/f=1
  exclusion), generation engine (state reset, sampling, policies, decode,
  provenance, replay), analysis (summaries, plots), CLI, runtime.
- CPU test suite: passes in full (see the build report for the exact
  count).
- GPU correctness gates: **passed** on a rented host — `test_replay_gpu.py`
  (all seven cases), `test_patched_noop_parity_gpu.py`,
  `test_no_state_leak_gpu.py`, determinism and compaction
  (`logs/gpu_validation/*.log`). The §10 f=1 probe-stability control
  (`test_probe_stability_gpu.py`) is the exception — it remains **UNRESOLVED**
  under the corrected validity definition (`docs/GPU_VALIDATION_PLAN.md`).
- Pilot screens run on GPU: protocol-v2 fixed-trace screen
  (`screen_valid=false`, `hypothesis_status=not_tested` — a valid negative
  screening outcome) and the protocol-v3 natural accuracy gate (FAILED, above).
- **Phase A2 — failure atlas (2026-07-19, CHANGELOG.md).** Deterministic,
  tested, CPU-only atlas over the 50 committed protocol-v3 gate pairs
  (`kvcot failure-atlas`, `src/kvcot/failure_atlas.py`):
  `results/tables/gsm8k_v3_b128_failure_atlas.{csv,md}`,
  `results/decisions/gsm8k_v3_b128_failure_atlas_summary.json`. Headline
  recomputation matches the prior manual analysis exactly (0/50 diverge
  before first compaction; 9/50 identical through `</think>`, 3 of those
  flip correct→wrong: rows 30, 271, 1115) and adds a new finding: 41/50
  pairs first diverge *inside* the reasoning span itself, so the
  identical-through-think flip is the minority pattern, not the typical
  one, at this retired operating point. Still `post_hoc_diagnostic` /
  `hypothesis_status: not_tested` — this does not test the §1 hypothesis.
- **Phase A3 — adversarial literature matrix and diagnostic novelty
  kill-check (2026-07-19, CHANGELOG.md).** `docs/RELATED_WORK_MATRIX.md`,
  `docs/A3_SEARCH_LOG.md`, `docs/related_work_matrix.json` (20 papers,
  schema-validated). CASK (arXiv:2604.10900) independently implements this
  repository's core fixed-trace/teacher-forced replay diagnostic primitive
  (confirmed against its official evaluation code); Lanham et al.
  (arXiv:2307.13702) independently established early answering. **DIAGNOSTIC
  SURVIVAL VERDICT: DOES NOT SURVIVE — PHASE B: BLOCKED — DIAGNOSTIC NOT
  NOVEL.** A specific empirical intersection (KV-cache replay + early
  answering + accuracy gate + held-out per-example classification) remains
  unstudied but is not, by itself, a new method.
- **Phase B0 — method pivot specification and adversarial method-novelty
  gate (2026-07-19, CHANGELOG.md).** `docs/METHOD_PIVOT_SPEC.md`,
  `docs/METHOD_NOVELTY_MATRIX.md`, `docs/B0_SEARCH_LOG.md`,
  `docs/method_novelty_matrix.json` (25 records, schema-validated).
  Documentation-only; no code/config/test/schema/result touched, no GPU,
  no inference. Outcome: M1 PARTIAL, M2 KILLED, M3 PARTIAL — **METHOD
  PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET** (see Current status).
- All docs: `UPSTREAM_AUDIT.md`, `REPLAY_DESIGN.md`, `EXPERIMENT.md`,
  `PROBE_PROTOCOL.md` (real tokenizer output), `SCHEMA.md`,
  `REPRODUCIBILITY.md`, `GPU_VALIDATION_PLAN.md`.

## What's next (CPU-only; no new GPU rental; Phase B/MATH-500 blocked)

1. ~~**Failure atlas** over the existing 50 gate pairs~~ — **done, 2026-07-19**
   (Phase A2 above).
2. ~~**Literature matrix** situating this negative pilot result against prior
   faithfulness / KV-compression work.~~ — **done, 2026-07-19** (Phase A3
   above): **DOES NOT SURVIVE**, Phase B blocked under the diagnostic's old
   novelty story.

**MATH-500 implementation and any other Phase B work remain BLOCKED** —
not merely "not yet started" — until a genuinely new technique is designed
and approved; the current diagnostic combination (fixed-trace replay +
early answering + KV compression) is not, by itself, that new technique
(`docs/RELATED_WORK_MATRIX.md` §16).

That design phase has now run once: **Phase B0 (2026-07-19) evaluated
three candidate methods and returned BLOCKED — NO NOVEL METHOD YET** (see
Prior status above). **B1-as-a-method-implementation is still not
permitted under that B0 result** — it would have required at least one
SURVIVES PROVISIONALLY candidate, and there is none.

Separately, **Phase B0.5 (2026-07-19) evaluated a narrower, non-method
*discovery* question**, Phase B0.5-R (2026-07-19) repaired that protocol's
experimental unit and capture source against the pinned R-KV source,
Phase B0.5-R2 (2026-07-19) then found B0.5-R's own selected intervention
not representable in a dense KV tensor and repaired it to a fixed-shape
within-head swap, and Phase B0.5-R2.1 (2026-07-19) then fixed an
off-by-one branch-timing defect, froze the exact sampling algorithms, and
repaired gate 10 (see Current status above) — the current authorized
verdict is **READY FOR B1A PREREQUISITE IMPLEMENTATION**
(`docs/B0_5_R2_1_FINAL_PROTOCOL.md`), which supersedes B0.5-R2's, B0.5-R's,
and B0.5's prior verdicts. This permits only B1A: CPU-side prerequisite
implementation (MATH-500 symbolic-equivalence verifier, architecture-aware
R-KV monkeypatch dispatch, the repaired pairwise provenance schema with
timing fields, the repaired per-instance read-only capture wrapper with
exact score recomputation, the frozen deterministic sampling algorithms,
the mandatory no-op control's CPU unit test, CPU unit/integration tests —
code living in `src/kvcot`, testable via `--dry-run` exactly like every
other stage in this repository). **It still does not authorize any GPU
run, model inference, or method implementation** — those each require
their own separate authorization (B1B/B2A/B2B/C0,
`docs/B0_5_R2_1_FINAL_PROTOCOL.md` §11), and the model-freeze amendment
(`CLAUDE.md` §4) that a GPU run of a later phase would additionally need
has not been granted. No MATH-500 manifest, config, evaluator, or result
directory has been created by B0, B0.5, B0.5-R, B0.5-R2, or B0.5-R2.1. The
§10 f=1 stability control remains UNRESOLVED (not a
B0/B0.5/B0.5-R/B0.5-R2/B0.5-R2.1 task); the GSM8K b128 operating point
remains retired.

3. **Phase C — GPU rental.** No new GPU host is rented until a redesigned,
   non-retired, genuinely-novel experiment is specified and approved. The
   retired GSM8K b128 operating point is not re-run, and Phase C does not
   begin before a design phase addressing the A3 verdict is complete.

## Open decisions needing human input

- **License.** Not chosen. See `README.md`.
- **Whether to pursue MATH-500 at all, and under what redesigned method.**
  The old "Stage 1A decides GSM8K vs MATH-500" decision was already moot
  (GSM8K b128 retired on accuracy grounds); Phase A3 adds a second, deeper
  reason a MATH-500 rerun of the SAME diagnostic would not be worth GPU
  spend even if GSM8K accuracy had passed — the diagnostic combination
  itself does not clear the novelty bar. Any MATH-500 work needs both the
  fresh feasibility design AND a design response to the A3 gap, not the
  current frozen configuration.
- **§10 f=1 stability control.** UNRESOLVED, and a separate Stage-0
  prerequisite that any future non-retired stage must clear on its own terms
  (`docs/GPU_VALIDATION_PLAN.md`, 2026-07-19).
- **What the genuinely new technique should be.** Still open. Phase B0
  (2026-07-19) tested three candidates against the literature and none
  survived (M1 PARTIAL, M2 KILLED, M3 PARTIAL —
  `docs/METHOD_PIVOT_SPEC.md`). This remains the single open question
  blocking any Phase B successor; the B0 matrix's operation tables
  (`docs/METHOD_NOVELTY_MATRIX.md` §5–§7) are the map of what is already
  taken.

## Changes to frozen settings

None. Retiring an operating point changes no frozen §1/§4/§8/§9 value in
`configs/lock.yaml`; the retirement is recorded in `CHANGELOG.md`
(2026-07-19) as a documentation-only status update. Any future change to
frozen values still requires a dated `CHANGELOG.md` entry first, per the
build brief.
