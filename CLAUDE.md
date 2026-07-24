# CLAUDE.md

Frozen decisions for this repository. Loaded automatically at the start of
future sessions working in this directory. These are excerpts (Sections 1,
4, 8, 9) of the original build brief, preserved verbatim where the exact
wording matters (sign conventions, forbidden conclusions) — do not
paraphrase these away in future edits without updating `CHANGELOG.md` first.

## Section 1 — Research question and claim boundary

One narrow question:

> At an accuracy-preserving operating point, does decoding-time R-KV
> compression reduce a reasoning model's **behavioral dependence on the
> omitted suffix of its visible reasoning trace**?

Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, only.

Intervention: **early answering**. Generate a complete reasoning response.
Replay the exact generated tokens one at a time under the same cache
policy. Branch at several fractions of the thinking span. Close the
thinking block. Force a short final answer. Compare that answer to the
untruncated base response **from the same condition and same seed**.

**Allowed conclusion:** lower sensitivity to truncating visible reasoning
under R-KV.

**Forbidden conclusions:** that the chain is fake, decorative, unfaithful
to the model's "true thoughts," or that we observed internal cognition.
This measures counterfactual behavioral dependence on visible generated
tokens. Nothing else. Enforce this in docstrings and in every generated
summary string (`kvcot.probes.early_answering.CLAIM_BOUNDARY_NOTICE`).

**No method lives in this repository.** Do not implement faithfulness-aware
eviction, KIVI, mistake insertion, vLLM, SGLang, multi-GPU, an LLM judge, or
a benchmark suite. Scope control is worth more than empty stub files — do
not create placeholder modules for out-of-scope work. The blanket
prohibition on additional model/architecture support (previously stated
here as "7B support") is **narrowly superseded by §1a below**, which
authorizes CPU-side infrastructure only for one additional architecture
(`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`) for a bounded discovery track —
it remains true, without exception, that no *method* is implemented, no
general benchmark-suite expansion is authorized, and no other model or
architecture beyond that one narrow exception is in scope.

### Section 1a — Discovery-only exception (dated 2026-07-19, B0.5-R2.2)

Added by `docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`, superseding
nothing above — the original Qwen-1.5B/GSM8K research question, claim
boundary, and "no method lives in this repository" rule remain the frozen
primary pipeline, unmodified. This is a narrow, dated, explicit exception,
not a redefinition of §1:

- The original pipeline remains `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  **only** — §1's research question, model line, and claim boundary are
  unchanged by this exception.
- A bounded **discovery track** (B1A CPU prerequisites only, as of this
  date) may add architecture support for
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` — dispatch/monkeypatch
  plumbing, state-reset generalization, and construction-parity tests, never
  a change to the primary pipeline's model.
- **MATH-500** may be supported only for this bounded discovery track, never
  substituted for GSM8K in the primary pipeline.
- This is **infrastructure support and failure discovery, not method
  implementation** — no faithfulness-aware eviction, no new compression
  policy, no accuracy or faithfulness claim of any kind is authorized by
  this exception.
- **No model inference or GPU use is authorized by this amendment.** Every
  line item above is CPU-side code and CPU-side tests only.
- A **separate, explicit, future authorization is still required** before
  B2A (GPU calibration) or any B1B/B2B discovery-pilot activity, and before
  any Vast.ai (or other GPU host) activity of any kind.
- This repository still contains **no final faithfulness-aware compression
  method** — this exception does not create, imply, or move toward one.
- Support for `DeepSeek-R1-Distill-Llama-8B` under this exception must
  **not** be described, in any document, as general benchmark-suite
  expansion — it is scoped exactly to the discovery track defined in
  `docs/B0_5_R2_1_FINAL_PROTOCOL.md` and
  `docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`, nothing broader.

### Section 1b — Bounded B1B CPU-harness-architecture exception (dated 2026-07-20, B1B-R1)

Added by `docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`, superseding nothing
above and nothing in §1a — the original Qwen-1.5B/GSM8K pipeline and the
Llama-8B/MATH-500 B1A CPU-prerequisite exception are both unchanged. §1a
stated that "a separate, explicit, future authorization is still required
before B2A ... or any B1B/B2B discovery-pilot activity". This subsection is
that separate, dated, explicit authorization — narrow, and only for what is
listed below:

- Authorizes CPU-side implementation of the **B1B harness architecture
  only**: Pass-1 natural-run bookkeeping contracts, Pass-2 token-identical
  replay/capture orchestration, branch construction and evaluation wiring,
  and attrition accounting — built with **dependency-injected synthetic and
  deterministic components exercised only in CPU tests**, never against a
  real model, real weights, or a real dataset.
- Authorizes a CPU-only `kvcot plan-discovery --dry-run` planning command
  and a documentation/validation-only future one-example B2A contract
  (schema and hard-stop-condition definitions; the contract is never
  executed by this exception).
- **No model inference or GPU use is authorized by this amendment.** No
  line item above ever loads a real model, a real tokenizer, or a real
  dataset; every path exercised by this exception's tests uses injected
  fakes only.
- Does **not** authorize B2A (one-example GPU calibration) or B2B (the
  bounded discovery pilot) execution — both still require their own
  separate, future, dated authorization, exactly as §1a already stated.
  Does not authorize any Vast.ai or other GPU-host activity of any kind.
- This repository still contains **no final faithfulness-aware compression
  method** and this exception implements **no learned eviction policy** —
  it is harness plumbing only, never a method.
- Does not weaken, narrow, or reinterpret any prohibition in §1 or §1a
  (vLLM, SGLang, multi-GPU, an LLM judge, a benchmark suite, KIVI, mistake
  insertion, or any method implementation remain fully prohibited).

### Section 1c — B2A one-example GPU authorization (dated 2026-07-22)

Added by
`docs/B2A_ONE_EXAMPLE_GPU_AUTHORIZATION_2026-07-22.md`, superseding nothing
above — this is the separate, explicit, dated authorization §1a and §1b
both said was still required before B2A. It authorizes exactly one
`b2a-calibrate --execute` attempt against the committed one-example
manifest (`configs/discovery/b2a_one_example_manifest.json`) under
`configs/discovery/llama8b_math500_b1024.yaml`, unmodified — nothing
broader:

- Authorizes exactly one execution attempt of the existing B1B harness
  architecture against real weights: one frozen MATH-500 example, FullKV
  and R-KV (budget 1024) workers, the existing 12-real-pair-plus-one-no-op
  design, on one RTX 3090, batch size 1, no CPU/disk/meta offload, peak
  tracked CUDA memory <= 22 GiB, projected complete-pilot runtime <= 4.00
  GPU-hours.
- Does **not** authorize B2B (the bounded discovery pilot), a 12-example
  run of any kind, a second B2A attempt, any method implementation, any new
  eviction criterion, or any change to a scientific threshold, event
  selection, pair selection, candidate/donor definition, model, dataset
  row, revision, seed, budget, or generation config.
- The two failed historical results in
  `tests/integration/test_probe_stability_gpu.py` (FullKV 17/20, R-KV b256
  15/20 against the 0.90 f=1 stability threshold) are **not** blockers for
  this authorization and are **not** altered, rerun, tuned, or redescribed
  as passing — see `docs/B2A_ONE_EXAMPLE_GPU_AUTHORIZATION_2026-07-22.md`
  §3 for why that archived Qwen-1.5B/GSM8K early-answering measurement
  (sampled base continuation, greedy probe continuation conditioned on an
  inserted control suffix) does not bear on the disjoint Llama-8B/MATH-500
  B2A mechanism (greedy throughout, no control suffix, causal
  candidate/donor swaps).
- This repository still contains **no final faithfulness-aware compression
  method** and this authorization implements **no learned eviction
  policy** — it is one bounded engineering-calibration execution, never a
  method, and never a scientific pilot result.
- Once FullKV or R-KV inference begins under this authorization, the
  attempt is scientifically consumed — no automatic or unauthorized second
  attempt.

### Section 1d — B2A-R1 failure closure and B2A-R2 pre-registration (dated 2026-07-22)

Added by
`docs/B2A_R1_FAILURE_AND_B2A_R2_PROTOCOL_2026-07-22.md`, superseding
nothing above. The single attempt §1c authorized (B2A-R1, against
`example_index=0`) ran: FullKV/R-KV inference began (return code 0 for
both workers), so the attempt is consumed, but it produced **zero R-KV
compaction events** (prompt=105 tokens, generated=449 tokens, far under
budget=1024) — an ineligible calibration that tested no eviction at all,
not a scientific result of any kind. This subsection is the separate,
explicit, dated authorization for exactly one further attempt, B2A-R2,
against a row selected by a pre-registered, deterministic, outcome-blind
procedure:

- A committed candidate manifest
  (`configs/discovery/b2a_r2_candidate_manifest.json`, 12 level-5 MATH-500
  rows from the SAME pinned dataset revision, ordered by a fixed
  content-derived hash — never by observed generation length or outcome).
- FullKV-only qualification (`kvcot qualify-b2a-row`, R-KV never imported)
  attempting those 12 candidates in committed order, stopping at the first
  one satisfying all 10 frozen conditions (§4 of the protocol doc) — no
  qualified row means immediate stop, B2A-R2/B2B remain blocked.
- The qualified row frozen into a replacement
  `configs/discovery/b2a_one_example_manifest.json` only by
  `kvcot.discovery.b2a_r2_freeze.freeze_qualified_row`, which fails closed
  on any hash/identity mismatch or arbitrary-row substitution attempt.
- Exactly one `b2a-calibrate --execute` attempt against that frozen row,
  under the EXACT SAME settings §1c/§4c already fixed — no threshold,
  budget, model, dataset, or gate change of any kind.
- Two repairs made before B2A-R2, neither touching scientific
  configuration: (1) the coordinator no longer raises on an insufficient
  real-pair count, it resolves to a clean `gate_failed` outcome with an
  explicitly unavailable (never fabricated) runtime projection; (2) the
  MATH-500 answer verifier's calling convention into `math_verify` was
  corrected (bare compound LaTeX is now re-wrapped in `\boxed{}` before
  verification — a general parsing-boundary fix, confirmed against the
  installed `math-verify==0.9.0` package directly, not tuple-specific and
  not tuned to the observed answer).
- Does **not** authorize B2B, a 12-example run, a third B2A attempt, any
  method implementation, or any change to `configs/lock.yaml`,
  `third_party/R-KV`, the pinned R-KV revision, or any parity/provenance/
  memory/timing/device-placement gate.
- Once FullKV or R-KV inference begins under THIS authorization, B2A-R2 is
  likewise scientifically consumed — no automatic or unauthorized further
  attempt.

### Section 1e — B2A-R3 protocol freeze (dated 2026-07-22)

Added by
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md`, superseding
nothing above. PR #20 (the B2A-R2 forensic pair-record persistence repair)
merged into `main` (merge commit
`9e78bc5edda0f0086d9e9aaea98896ac24caa7b0`); B2A-R2 remains consumed and
cannot be rerun (`B2A-R2 FINAL VERDICT: FAIL -- B2B BLOCKED`, unchanged).
This subsection records that the B2A-R3 runtime-qualified calibration
protocol is now frozen:

```text
B2A-R3 protocol is frozen.
CPU implementation and CPU tests are authorized after independent audit.
GPU activity remains prohibited.
```

Explicitly allowed by this subsection, once that independent audit of the
frozen protocol document passes: CPU-only implementation of the B2A-R3
candidate manifest, deterministic candidate generator, runtime predictor,
qualification evaluator, and selected-row freezer described in the
protocol document; CPU-only tests for all of the above, using injected
synthetic/deterministic components exactly as §1b's harness-architecture
exception required; static candidate/qualification manifests produced by
CPU-only code; `kvcot`'s CPU-only dry-run planning paths; documentation.

Explicitly **not** authorized by this subsection: FullKV candidate
qualification against real weights; R-KV execution; pair evaluation; any
CUDA initialization; B2A-R3 execution; B2B; any FaithKV method
implementation. CPU implementation becomes the next proposed action only
after an independent audit of the frozen protocol document — this
subsection does not itself authorize Step 3 (CPU implementation) to begin;
it authorizes only that the protocol exists in frozen form and states what
Step 3 will be permitted to do once that separate audit passes.

- Does **not** authorize B2A-R3 execution, B2B, GPU rental, or FaithKV
  method implementation of any kind.
- Does **not** change any threshold, budget, model, dataset, or gate frozen
  by §1/§1a/§1b/§1c/§1d/§4/§4a/§4b/§4c.
- Does **not** modify `configs/lock.yaml` or the pinned R-KV revision.

### Section 1f — B2A-R3 protocol audit repair (dated 2026-07-22)

Added by
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`, superseding nothing
above. An independent audit of the protocol §1e froze found 18 defects
(missing §22; a contradictory "CPU IMPLEMENTATION AUTHORIZED" status line
next to audit-gated prose; rounded runtime constants; an unfrozen
mixed-level candidate ordering; an unfrozen exclusion set; undefined
thinking-span/trace-completeness predicates; a self-referential hash
ambiguity; a nondeterministic candidate-manifest timestamp; every
artifact path left merely "recommended"; a self-contradictory wall-time
section; an undefined attempt-consumption lifecycle; a historical
branch-constant conflict; dynamic (unfrozen) final-gate inheritance;
imprecise no-op tolerance wording; inconsistent schema field names; CPU
scope ambiguous about authorizing a real qualification artifact; and an
unresolved repository-identity discrepancy). All 18 are repaired directly
in `docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md`; the finding
ledger is `docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`.

```text
B2A-R3 protocol repairs are committed.
Independent re-audit is pending.
Step 3 remains blocked.
GPU remains prohibited.
```

This subsection does **not** authorize CPU implementation. The repairing
author does not self-certify this protocol — a separate, genuinely
independent re-audit of this repair commit is the next required action
before Stage A (§1e) can begin. No threshold, budget, model, dataset, or
gate frozen by §1/§1a-§1e/§4/§4a-§4c is changed by this subsection; no
source code, test, or configuration file is modified by it;
`configs/lock.yaml` and the pinned R-KV revision are unchanged.

### Section 1g — B2A-R3 Step 2B implementation-contract closure (dated 2026-07-22)

Added by
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`'s "Step 2B" section,
superseding nothing above. A second independent re-audit of §1f's repair
commit (`81e11cb57202e0d4b434aabb347963ae3c34b80b`) found the original 18
defects materially repaired, but identified seven remaining
implementation-level ambiguities that could still have let Step 3 invent
behavior the protocol never actually pinned down: an unfrozen
qualification-condition tuple (R3-AUDIT-19); incomplete artifact
provenance schemas (R3-AUDIT-20); unresolved selected-manifest hash
semantics (R3-AUDIT-21); a candidate-row schema insufficient for
deterministic freezing (R3-AUDIT-22); a Markdown authorization-document
hash conflicting with the JSON self-hash rule (R3-AUDIT-23); an
authorization-claim consumption design that was scan-then-write rather
than globally atomic (R3-AUDIT-24); and protocol-identity fields conflated
under one ambiguous `protocol_version` name (R3-AUDIT-25). All seven are
repaired directly in
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` (§10.5, §12.1,
§12.3-12.9, §13, §14.4, §22); the finding ledger is
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`'s "Step 2B" section.

```text
Step 2B implementation-contract repairs committed.
Independent re-audit required.
Step 3 remains blocked.
GPU remains prohibited.
```

This subsection does **not** authorize CPU implementation, Stage B FullKV
qualification, Stage C B2A-R3 execution, B2B, or FaithKV method
implementation. The repairing author does not self-certify this
protocol — a separate, genuinely independent re-audit of this Step 2B
commit is the next required action before Stage A (§1e) can begin. No
threshold, budget, model, dataset, or gate frozen by
§1/§1a-§1f/§4/§4a-§4c is changed by this subsection; no source code, test,
or configuration file is modified by it; `configs/lock.yaml` and the
pinned R-KV revision are unchanged.

### Section 1h — B2A-R3 Step 2B independent re-audit and Step 3 Stage-A CPU implementation (dated 2026-07-23)

Added by
`docs/B2A_R3_STEP2B_INDEPENDENT_REAUDIT_2026-07-23.md`, superseding
nothing above. A genuinely independent re-audit of the Step 2B repair
commit (§1g, folded into
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` and
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`) found all seven Step 2B
findings (R3-AUDIT-19 through R3-AUDIT-25) materially repaired, with one
narrow explanatory correction (the protocol's own claim that `1.20` is
"exactly representable in IEEE-754 binary64" is imprecise — the gate
itself, exact `==` comparison against the same frozen Python float
literal, is unchanged). Two hashes central to the frozen predictor/
qualification contract (`runtime_source_artifact_sha256` over
`docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json`, and
`generation_config_sha256` over the frozen `generation:` payload) and the
frozen `2775`-passes/`2776`-fails integer runtime boundary were
independently reproduced and matched exactly.

```text
INDEPENDENT STEP 2B RE-AUDIT: PASS
STEP 3 STAGE-A CPU IMPLEMENTATION AUTHORIZED
STAGE B FULLKV QUALIFICATION PROHIBITED
STAGE C B2A-R3 EXECUTION PROHIBITED
GPU / CUDA / MODEL INFERENCE PROHIBITED
```

This subsection authorizes exactly Step 3 Stage-A: CPU-only implementation
of the B2A-R3 candidate manifest/generator, runtime predictor, pure
qualification evaluator (exercised only against synthetic/injected
evidence), artifact verifiers, a synthetic-only selected-row freezer, the
atomic authorization-claim mechanism (exercised only against synthetic
fixtures — no real claim is ever created by Stage A), a
separately-constructed `AttemptProvenancePolicy` that does not modify
`attempt_verification.py`'s historical `REQUIRED_BRANCH` constant or any
historical B2A-R1/R2 verification, CPU-only dry-run/verification CLI
commands, and CPU tests for all of the above. It also authorizes
generating and committing exactly one real, deterministic,
outcome-blind B2A-R3 candidate manifest
(`configs/discovery/b2a_r3_candidate_manifest.json`) — built from the
pinned MATH-500 dataset's already-public content, requiring no model, no
CUDA, and no GPU.

- Does **not** authorize Stage B (FullKV qualification against real
  weights), Stage C (B2A-R3 execution), B2B, any CUDA initialization, any
  R-KV import outside its historical B2A-R1/R2 role, a real qualification
  artifact, a real selected-manifest replacement, a real authorization
  claim, or any FaithKV method implementation.
- Does **not** change any threshold, budget, model, dataset, revision, or
  gate frozen by §1/§1a-§1g/§4/§4a-§4c/§8/§9. Does not modify
  `configs/lock.yaml` or the pinned R-KV revision.
- Does not modify `src/kvcot/discovery/attempt_verification.py`'s
  historical `REQUIRED_BRANCH` constant, and does not weaken any
  historical B2A-R1/R2 verification.

### Section 1i — B2A-R3 Step 3 Stage-A independent-audit repair round (dated 2026-07-23)

Added by
`docs/B2A_R3_STEP3_STAGE_A_INDEPENDENT_AUDIT_REPAIR_2026-07-23.md`,
superseding no frozen scientific setting or authorization boundary. The
independent audit of Stage-A implementation SHA
`6778fd119c33f5025f328321a33eb7c62ad86d20` identified twelve blocking
verification defects (P0-1 through P0-9 and P1-1 through P1-3). The
bounded repair round added strict independent replay for qualification,
schedule, timing, candidate/token/prompt identities, authorization
preconditions, exact post-claim paths, multiprocess claim races, and the
complete selection provenance chain.

```text
STEP 3 STAGE-A AUDIT REPAIRS IMPLEMENTED —
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED
```

- This is a CPU-only repair and verification round. It does not
  self-certify Stage A and does not authorize Stage B or Stage C.
- No GPU/CUDA initialization, model or tokenizer execution, FullKV or
  R-KV run, real qualification/freeze/claim/attempt, B2B execution, or
  FaithKV method implementation is authorized or performed.
- Historical B2A-R1/R2 artifacts and verification semantics, the committed
  B2A-R3 candidate-manifest content, `configs/lock.yaml`, and the R-KV
  gitlink remain unchanged.
- The next required action is an independent re-audit of the final repair
  SHA.

### Section 1j — B2A-R3 Step 3R4 CPU protocol alignment and Stage-B readiness repair (dated 2026-07-23)

Added by
`docs/B2A_R3_STAGE_B_READINESS_REPAIR_2026-07-23.md` and
`docs/B2A_R3_STAGE_A_PROTOCOL_ALIGNMENT_AMENDMENT_2026-07-23.md`,
superseding nothing above except the narrow qualification-outcome schema/
version fields the dated amendment names explicitly (§3 of that
document). A re-audit of Step 3 Stage-A SHA
`7062f3cb8a6f555d3b67cf9e9be3bd6710e78120` identified six findings: a
qualification-outcome schema that duplicated artifact-level identity
fields onto every attempted candidate and persisted a redundant nested
runtime-prediction object; a FullKV timing helper that spliced two memory
phases into its timing vocabulary and used the wrong phase order; an
authorization-precondition check that derived its enforced policy from
the claim itself rather than the tracked authorization document; a
public claim-consumption API that accepted an arbitrary `claims_root`
instead of the one deterministic global path; a missing canonical
adapter from a FullKV worker result into qualification evidence; and a
missing qualification-artifact builder/atomic-writer/coordinator
architecture. All six are repaired, each with a dedicated adversarial
regression suite.

```text
STEP 3R4 CPU PROTOCOL ALIGNMENT AND STAGE-B READINESS REPAIRS IMPLEMENTED —
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED
```

- This is a CPU-only repair and governance round. It does not self-certify
  its own work and does not authorize Stage B or Stage C.
- No GPU/CUDA initialization, model or tokenizer execution, FullKV or
  R-KV run, real qualification/freeze/claim/attempt, B2B execution, or
  FaithKV method implementation is authorized or performed.
- Historical B2A-R1/R2 artifacts and verification semantics, the committed
  B2A-R3 candidate-manifest content (canonical hash unchanged:
  `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42`),
  `configs/lock.yaml`, and the R-KV gitlink remain unchanged.
- The next required action is an independent re-audit of the final
  Step 3R4 repair SHA.

### Section 1k — B2A-R3 Step 3R4 independent re-audit repair, round 2 (dated 2026-07-23)

Added by `docs/B2A_R3_STEP3R4_REPAIR2_2026-07-23.md`, superseding nothing
above except the narrow `authorized_maximum_candidates`/qualification-
outcome-version fields that document names explicitly. An independent
re-audit of §1j's SHA (`187236426b5fb48321c18a91556cf0d560918494`) returned
**FAIL**, with seven blocking findings plus one report-inconsistency: (1)
no production path existed from a real FullKV worker result into
`FullKVWorkerResultR3` — the adapter only ever converted a hand-built one;
(2)/(3) the qualification-only timing/memory validators rejected the real
worker's own output (`before_model_load`/`post_load_baseline` genuinely
appear as timing phases, `tokenizer_load`/`post_load_validation`
genuinely appear as memory phases, and `answer_verification` is genuinely
nested before `fullkv_complete_natural_generation`); (4) the qualification
coordinator always handed a worker the full frozen per-candidate timeout
regardless of remaining authorized phase time, and never rechecked elapsed
time after a worker completed; (5) the atomic qualification-artifact
writer performed only shallow (self-hash/schema) verification, never full
semantic re-derivation; (6) `qualification_stopped_reason` had no
membership validator and the artifact carried no record of the
authorization's actual `maximum_candidates`, so an exhaustion claim could
not be independently checked; (7) `verify_authorization_preconditions`/
`claim_authorization` accepted `git_state` and `repository_root` as two
independent, unbound parameters. All seven are repaired, each with a
dedicated regression suite exercising the real (injected, non-fake-shaped)
worker path where applicable; the 37-vs-53 field-count report
inconsistency is corrected (53 is, and always was, the correct count —
`QUALIFICATION_OUTCOME_V2_FIELD_NAMES`, already contract-tested).

```text
STEP 3R4 INDEPENDENT RE-AUDIT REPAIR ROUND 2 IMPLEMENTED —
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED
```

- This is a CPU-only repair round. It does not self-certify its own work
  and does not authorize Stage B or Stage C.
- No GPU/CUDA initialization, real model or tokenizer weights, a real
  FullKV or R-KV run, real qualification/freeze/claim/attempt, B2B
  execution, or FaithKV method implementation is authorized or performed.
  The new `kvcot.discovery.b2a_r3_qualification_worker` module reuses the
  canonical `run_fullkv_worker` body unmodified and is exercised only
  against injected fakes (a deterministic, torch-CPU-tensor model/
  tokenizer/cuda backend) — no line in this repair round initializes real
  CUDA or loads real weights.
- Historical B2A-R1/R2 artifacts and verification semantics, the committed
  B2A-R3 candidate-manifest content (canonical hash unchanged:
  `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42`),
  `configs/lock.yaml`, and the R-KV gitlink remain unchanged.
- `QUALIFICATION_ARTIFACT_SCHEMA_VERSION` is bumped v2 to v3 (a new
  required `authorized_maximum_candidates` field) — no real qualification
  artifact has ever been produced under v2 either, so this reinterprets
  nothing historical.
- The next required action is an independent re-audit of the final
  repair SHA from this round.

### Section 1l -- B2A-R3 Stage-B binding repair (dated 2026-07-23)

Added by `docs/B2A_R3_STAGE_B_BINDING_REPAIR_2026-07-23.md`, superseding
nothing above except the narrow persisted Stage-B authorization-binding
and qualification-artifact schema details named here. An independent
re-audit of SHA `6828e2f263da64bf552e605fb39a140af5be0c07` found that the
new v4 authorization fields and fixed Stage-B runner were substantive,
but that persisted Stage-B binding verification still incorrectly reused
the pre-claim/current-HEAD clean-worktree gate, the new subprocess wrapper
did not pass the frozen child environment, governance still recorded v3,
and remote CI had not yet run for the repair SHA.

```text
B2A-R3 STAGE-B BINDING REPAIR IMPLEMENTED --
READY FOR INDEPENDENT RE-AUDIT;
STAGE B FULLKV QUALIFICATION REMAINS BLOCKED;
REMOTE CI REQUIRED ON FINAL SHA
```

- This is a CPU-only repair round. It does not self-certify its own work
  and does not authorize Stage B or Stage C.
- No GPU/CUDA initialization, real model or tokenizer weights, real
  FullKV or R-KV run, real B2A-R3 attempt, B2B execution, or FaithKV method
  implementation is authorized by this section.
- `QUALIFICATION_ARTIFACT_SCHEMA_VERSION` is now
  `faithkv-b2a-r3-qualification-artifact-v4`. Schema v4 requires the
  persisted Stage-B authorization fields:
  `authorized_phase_wall_time_limit_seconds`,
  `stage_b_authorization_id`, `authorization_document_sha256`, and
  `authorization_claim_canonical_sha256`.
- The fixed-path Stage-B command contract is
  `kvcot run-b2a-r3-stage-b-qualification --claim <claim-json>`. The
  command does not expose candidate order, maximum candidates,
  phase-wall-time limit, per-candidate timeout, output path, claims root,
  config path, candidate-manifest path, or repository root as operator
  overrides.
- The command contract exists for auditability only. Stage B remains
  blocked until a genuinely independent re-audit accepts the final repair
  SHA and remote CI is green for that exact SHA.
- The next required action after this commit is pushing the branch and
  obtaining remote CI evidence for the exact pushed SHA.

An independent re-audit of SHA
`ebb69d52810c94916ec4955d2c0848f597eadaf8` then found one remaining
fundamental blocker: the v1 authorization document/claim semantics required
the committed authorization document to contain the SHA of the commit that
contained the document itself. That self-reference is impossible in a
normal Git workflow.

- `AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION` is now
  `faithkv-b2a-r3-authorization-claim-v2`, and
  `AUTHORIZATION_DOCUMENT_SCHEMA_VERSION` is now
  `faithkv-b2a-r3-stage-authorization-document-v2`.
- Authorization now separates `authorized_code_commit_sha` (the audited
  CPU implementation commit) from `observed_execution_commit_sha` (the
  later clean commit that contains the dated authorization document).
- The authorization document names only `authorized_code_commit_sha`; the
  claim records `observed_execution_commit_sha`.
- Before claim consumption, the verifier requires clean current `HEAD` to
  equal `observed_execution_commit_sha`, requires
  `authorized_code_commit_sha` to be its ancestor, allows only the exact
  dated authorization document to differ between those commits, retrieves
  the authorization document bytes from the execution commit, and pins the
  R-KV gitlink at both commits.
- Persisted Stage-B verification uses the claim's historical execution
  commit for authorization-document bytes instead of reading the current
  checkout.
- No real B2A-R3 claim exists under the v1 field semantics, so the schema
  bump reinterprets nothing historical.

### Section 1m -- B2A-R3 GPU-host-neutral Stage-B preflight test repair (dated 2026-07-24)

Added by
`docs/B2A_R3_GPU_HOST_NEUTRAL_PREFLIGHT_TEST_REPAIR_2026-07-24.md`,
superseding nothing above. A Stage-B FullKV qualification preflight
attempt against the `stage-b-2026-07-23-final` authorization (execution
commit `4d559070df95def18fe5b649e2a7523d32bdba95`) on a rented Vast.ai RTX
3090 host was correctly **blocked before claim consumption**: every
precondition passed except the mandatory CPU test suite, which reported 3
failures. No FullKV or R-KV inference started, no model weights were
loaded for execution, no qualification artifact was produced, and the
external claim was never consumed.

Root cause: three CPU tests
(`test_run_fullkv_worker_requires_cuda_when_no_fake_backend_injected`,
`test_run_rkv_worker_requires_cuda_when_no_fake_backend_injected`,
`test_cuda_clean_refusal_is_not_wrapped_when_no_fake_backend_injected`)
called the production worker call shape (`_cuda`/`_load_model`/
`_load_tokenizer` all omitted) without controlling the physical host's
real `torch.cuda.is_available()` result. Frozen on a CPU-only machine
where that call always returns `False`, they observed the intended clean
`WorkerFailedError("... requires CUDA ...")` refusal; on a real
GPU-visible host it returns `True`, so they instead fell through into real
snapshot resolution and failed on an unrelated `SnapshotBoundaryError`.
This is a test-environment determinism defect, not a production worker
defect -- `src/kvcot/discovery/b2a_workers.py`'s CUDA-availability guard
and `WorkerFailedError` message are unchanged by this repair.

```text
B2A-R3 GPU-HOST-NEUTRAL PREFLIGHT TEST REPAIR AUTHORIZED — CPU TESTS ONLY

OLD CLAIM:
UNCONSUMED; SUPERSEDED WHEN THE REPAIR BRANCH ADVANCES

AUTHORIZED:
DETERMINISTIC REPAIR OF THREE HOST-DEPENDENT NO-CUDA TESTS

PROHIBITED:
PRODUCTION SOURCE CHANGES
SCIENTIFIC CONFIGURATION CHANGES
MODEL INFERENCE
FULLKV/R-KV EXECUTION
CLAIM CONSUMPTION
NEW STAGE-B AUTHORIZATION
```

- Authorizes exactly: monkeypatching only `torch.cuda.is_available` to
  `False` in the three named tests so they force the deterministic clean
  no-CUDA refusal branch regardless of physical host hardware, while
  continuing to omit `_cuda`/`_load_model`/`_load_tokenizer`/
  `_fresh_cache_factory`/`_device` so the production default-selection
  call shape is exercised unchanged; narrowly-scoped host-neutrality
  assertions (e.g. failing loudly if snapshot/model resolution is
  unexpectedly reached) may be added.
- Does **not** authorize any change under `src/`, `configs/`,
  `third_party/R-KV/`, or `results/`; any change to the model, tokenizer,
  dataset, cache budget, runtime threshold, VRAM limit, R-KV revision, or
  any of the 27 qualification conditions; marking the three tests
  `@pytest.mark.gpu`, skipping them, xfailing them, weakening their
  assertions, or accepting `SnapshotBoundaryError` in their place;
  modifying the existing dated Stage-B authorization document; or
  regenerating, editing, or consuming the existing external claim.
- The existing external claim (`authorization_id=stage-b-2026-07-23-final`,
  `canonical_sha256=992d7ebf68efcce14aca4bec49a932f8ba7d23517c9c7f0a1e5d11f5e46f5ec1`)
  remains unconsumed. Once this repair advances branch HEAD past
  `4d559070df95def18fe5b649e2a7523d32bdba95`, the claim's bound
  `observed_execution_commit_sha` no longer equals current clean `HEAD`,
  so it becomes **superseded** and must never be executed. A
  byte-identical copy is preserved at
  `/workspace/faithkv-superseded-claims/faithkv-stage-b-claim-unconsumed-2026-07-23.json`
  without modifying the original.
- Repair completed and locally validated same-day: each test now
  monkeypatches only `torch.cuda.is_available` to `False` (no fake `_cuda`
  passed), plus a narrow guard failing loudly if snapshot resolution is
  unexpectedly reached. On this Vast.ai RTX 3090 host, the 3 targeted
  tests pass both with the GPU visible and with `CUDA_VISIBLE_DEVICES=""`;
  the two full relevant modules pass (69 passed); the complete non-GPU
  suite passes (1850 passed, 14 deselected, 0 failed -- up from 1847
  passed / 3 failed); `compileall`, collection, and `git diff --check` are
  all clean. `src/`, `configs/`, `third_party/R-KV/`, and `results/` are
  byte-identical to the pre-repair execution commit.
- Stage B remains blocked. The next required actions are exact-SHA GitHub
  Actions CPU CI on the final repair commit, then an independent re-audit
  of that exact SHA, before any new, separate, dated Stage-B authorization
  may be produced against a newly audited code commit.

### Section 1n -- B2A-R3 Stage-B qualification evidence acceptance (dated 2026-07-24)

Added by
`docs/evidence/B2A_R3_STAGE_B_QUALIFICATION_ACCEPTANCE_2026-07-24.md`,
superseding nothing above. This records persistence-only acceptance of the
audited, independently-verified B2A-R3 Stage-B FullKV qualification
evidence produced under authorization ID `stage-b-2026-07-24-r2-final`
(audited code SHA `4117baea139f745ceeff85039258445639e85049`, execution
SHA `16d01ebe5c0659330bd78ccff96b9e64aea787ac`).

```text
B2A-R3 STAGE-B EVIDENCE ACCEPTANCE COMPLETE — PERSISTENCE ONLY

ACCEPTED:
CONSUMED CLAIM (canonical_sha256 68d055876a2260b179681fb276b79c37b6d1f987ae1899658fc969fcd05af975)
QUALIFICATION ARTIFACT (canonical_sha256 4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d)

SELECTED ROW:
test/number_theory/631.json (ordinal 1, answer 36, 7 eligible compaction events,
projected B2B runtime ~2.7949 GPU-hours)

PROHIBITED:
STAGE C
R-KV EXECUTION
PRODUCTION SELECTED-ROW FREEZER EXECUTION
CHANGES TO THE SELECTED MANIFEST OR SELECTION PROVENANCE
```

- Both artifacts pass `kvcot verify-b2a-r3-candidates` and
  `kvcot verify-b2a-r3-qualification`, and both `canonical_sha256` fields
  independently re-verify via `verify_canonical_sha256` against the exact
  values recorded in the accepted independent audit.
- `claimed_at_utc` on the consumed claim is the external claim payload
  timestamp only, **not** the canonical atomic-consumption timestamp — no
  protocol field represents the exact atomic-consumption instant;
  filesystem timestamps remain auxiliary evidence only.
- The production selected manifest
  (`configs/discovery/b2a_one_example_manifest.json`) still contains the
  historical row (`test/number_theory/820.json`); no
  `results/decisions/b2a_r3_selection_provenance.json` exists. Neither is
  created, modified, or authorized by this acceptance.
- Full identity, hash, and CI binding recorded in
  `docs/evidence/B2A_R3_STAGE_B_QUALIFICATION_ACCEPTANCE_2026-07-24.md`.
  Stage C remains blocked.

### Section 1o -- B2A-R3 Stage-B evidence-acceptance test-assumption repair (dated 2026-07-24)

Added by
`docs/B2A_R3_STAGE_B_EVIDENCE_ACCEPTANCE_TEST_ASSUMPTION_REPAIR_2026-07-24.md`,
superseding nothing above. Exact-SHA CI on the §1n evidence-acceptance
commit failed 2 of 1850 non-GPU tests
(`test_no_production_claims_directory_touched_by_dry_run`,
`test_no_production_files_written`): both hard-asserted the real
authorization-claims directory / qualification-artifact path must never
exist in the repository, an assumption §1n's own committed evidence
directly falsified. Repaired to assert the real invariant instead --
before/after no-touch (existence and byte-content unchanged) -- rather
than flat non-existence. Test files only; no change under `src/`,
`configs/`, `third_party/R-KV/`, or `results/`. Full non-GPU suite: 1850
passed / 14 deselected / 0 failed locally. Stage C remains blocked.

## Section 4 — Frozen settings

Fixed unless a dated `CHANGELOG.md` entry is added **before** the run.
Executable source of truth: `configs/lock.yaml`.

| Item | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Model/tokenizer revision | `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` |
| Dtype | BF16 |
| Attention backend | `flash_attention_2` primary; `sdpa` available for the determinism test. Fail loudly if unavailable — never switch silently mid-run. |
| Batch size | 1 |
| Base generation | sampling, temperature 0.6, top-p 0.95, one sequence |
| Base cap | `max_new_tokens=6144` (**never** `max_length`) |
| Seeds | 13, 42, 2026 |
| R-KV window | 8 |
| R-KV mix lambda | 0.1 |
| R-KV retain ratio | 0.2 (**inert** under `retain_direction=last` — docs/UPSTREAM_AUDIT.md §6.3) |
| R-KV retain direction | `last` |
| Compression schedule | `divide_method=step_length`, `divide_length=128` |
| Compression content | `all` |
| Probe fractions (all probed) | 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 |
| Fractions **scored** into EAS | 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875 (7 values) |
| f=0.0 | descriptive no-chain baseline — **excluded from EAS** |
| f=1.0 | stability control — **excluded from EAS** |
| Probe decoding | greedy/deterministic, `max_new_tokens=48` |

Explicit **batch-1, token-by-token decode loop** for base generation *and*
replay — never `model.generate()` on the state-critical path
(`kvcot.generation.decode`, `docs/REPLAY_DESIGN.md` §2 explains why call
shape specifically matters here, not just as a style rule). Per-example
seed via SHA-256 of `(global_seed, dataset_name, problem_index)`
(`kvcot.utils.seeding.derive_seed`) — FullKV and R-KV always receive the
identical derived seed.

### Section 4a — Discovery-only exception (dated 2026-07-19, B0.5-R2.2)

**The Qwen-1.5B model row in the table above is unchanged.** This
subsection is a separate, clearly-labeled, dated exception for the bounded
discovery track only — it does not edit, replace, or silently override any
row in the §4 table, and `configs/lock.yaml` is **not** changed by this
exception (no discovery GPU configuration is authorized or executed).

| Item | Discovery-track-only value |
|---|---|
| Additional model (discovery track only) | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Additional dataset (discovery track only) | MATH-500 |
| Scope | CPU-side B1A prerequisites only (architecture dispatch, state-reset generalization, MATH-500 verifier, discovery schema, deterministic sampling, read-only capture wrapper prerequisite, fixed-shape swap primitive, no-op branch test) |
| GPU/inference authorization | **None.** Not granted by this table or this document. |
| Method authorization | **None.** No compression policy is implemented under this exception. |

Any GPU run under this exception requires its own separate, future, dated
authorization (B2A calibration at minimum) — this table only unblocks the
CPU-side code listed above from contradicting §1's/§4's original blanket
freeze.

### Section 4b — Bounded B1B CPU-harness-architecture exception (dated 2026-07-20, B1B-R1)

**No row in the §4/§4a tables above is changed.** `configs/lock.yaml` is
**not** changed by this exception; `configs/discovery/llama8b_math500_b1024.yaml`
is a separate, discovery-track-only file this exception adds, never merged
into `configs/lock.yaml`.

| Item | B1B-harness-only value |
|---|---|
| Scope | CPU-side harness architecture only: Pass-1/Pass-2 orchestration, branch construction/evaluation, attrition accounting, `plan-discovery --dry-run`, future B2A contract (documentation/validation only) |
| Component wiring | Dependency-injected synthetic/deterministic components in CPU tests only |
| GPU/inference authorization | **None.** Not granted by this table or this document. |
| Method authorization | **None.** No compression policy or learned eviction policy is implemented under this exception. |
| B2A/B2B execution | **Not authorized.** Requires its own separate, future, dated authorization. |

### Section 4c — B2A one-example execution settings (dated 2026-07-22)

**No row in the §4/§4a/§4b tables above is changed.** `configs/lock.yaml` is
**not** changed by this table. These are the execution-time settings for
the single attempt authorized by §1c, resolved entirely from
`configs/discovery/llama8b_math500_b1024.yaml` and
`configs/discovery/b2a_one_example_manifest.json` — this table records them
for audit purposes, it does not itself set them.

| Item | B2A one-example value |
|---|---|
| Example scope | Exactly one frozen MATH-500 row, identified by the committed manifest |
| Model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, pinned revision, resolved from the discovery config only |
| Generation mode | Greedy (`do_sample=False`), batch size 1, no early-answering control suffix |
| Workers | FullKV and R-KV (budget 1024), existing 12-real-pair-plus-one-no-op design, unmodified |
| Hardware | One visible RTX 3090, no CPU/disk/meta offload |
| Memory limit | Peak tracked CUDA memory <= 22 GiB — hard gate, not weakened on failure |
| Runtime limit | Projected complete-pilot runtime <= 4.00 GPU-hours — hard gate, not weakened on failure |
| Attempt count | Exactly one `b2a-calibrate --execute` attempt; consumed once FullKV or R-KV inference begins |
| B2B authorization | **None.** Not granted by this table or §1c. |

## Section 8 — Metrics and statistics

For problem `i`, condition `c`, seed `s`, fraction `f`:

```
match_{i,c,s}(f) = 1 iff normalized probe answer at f == normalized untruncated base answer, SAME condition, SAME seed
```
Not matched to gold. Not matched across conditions.

### Why f=0 is excluded from EAS

On the both-correct subset both base answers equal gold, so they are
identical. If no compaction has fired by end of prefill — guaranteed
whenever budget > prompt length — the R-KV cache at f=0 **is** the FullKV
cache, so `match_full(0) ≡ match_rkv(0)` by construction and the term
cancels out of the difference. Including it dilutes the effect, and dilutes
it by a **budget-dependent** amount. Probe f=0, report its curve point as
the descriptive no-chain baseline, exclude it from EAS.

### Early-Answer Sensitivity

```
EAS_{i,c,s} = mean over f in {0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875} of (1 - match_{i,c,s}(f))
```

```
Delta_EAS_{i,s} = EAS_{i,FullKV,s} - EAS_{i,RKV,s}
```

**Positive `Delta_EAS` is the hypothesized direction: the R-KV answer is
less sensitive to truncation of its visible trace under R-KV.** A sign
error here silently inverts the entire result
(`kvcot.analysis.metrics.compute_delta_eas` docstring restates this — never
recompute the sign convention independently anywhere else).

Seven scored fractions (not three), because `Delta_EAS` is quantized to
multiples of `1/|F|`; at `|F|=3` ties swamp Wilcoxon.

### Primary eligibility (§8.3)

A (problem, seed) pair is eligible iff: both conditions' base answers
correct; both think spans parse; both f=1 stability probes match their own
base answer; the R-KV run had ≥1 actual compaction; no required record
missing. A problem enters primary analysis if **≥2 of its 3 seed pairs are
eligible** — average `Delta_EAS` over eligible seeds, **exactly one number
per problem**, never pool (problem, seed) rows as independent samples
(`kvcot.analysis.metrics.aggregate_problem_delta_eas` is the only place
that averaging is allowed to happen).

### Attrition funnel is mandatory (§8.4)

Every eligibility filter is potentially correlated with the treatment.
`kvcot.analysis.summaries.build_attrition_funnel_table` emits it; if R-KV
loses substantially more problems at any stage, that belongs in the
headline, not a footnote.

### Frozen tests (§8.5) — implement these and no others

- **Primary:** two-sided Wilcoxon signed-rank over problem-level
  `Delta_EAS`, **Pratt zero handling primary**, zero-drop (`wilcox`) as
  sensitivity, exact-zero count always reported
  (`kvcot.analysis.stats.wilcoxon_delta_eas`).
- **Primary CI:** percentile bootstrap 95% CI of mean `Delta_EAS`, 10,000
  resamples over problems, fixed seed `20260715`
  (`kvcot.analysis.stats.bootstrap_ci_mean`).
- **Accuracy match (headline, not a footnote):** paired base-accuracy
  difference FullKV vs. R-KV on the full 200-problem main split, bootstrap
  95% CI (`kvcot.analysis.stats.paired_accuracy_diff`). "Accuracy-
  preserving operating point" is load-bearing in the research question.

**The primary control is the both-correct subset**, which conditions on
correctness per problem. The Stage 1B/2 accuracy checks only keep the pilot
off an absurd operating point — they do not independently establish
distributional accuracy preservation. State this explicitly wherever the
result is reported (`docs/EXPERIMENT.md` §9).

## Section 9 — Realized retention naming

**No condition may be named "R-KV 10%" or any percentage.** The condition
is `RKV-B{budget}` (schema/config spelling: `rkv_b{budget}`). Realized
retention is *measured* per snapshot (`RetentionSummary` in
`src/kvcot/schemas.py`: `instantaneous_retention_ratio =
physical_cache_slots / fullkv_equivalent_slots`), never configured, and
never used to name a condition. `tests/unit/test_no_ten_percent_naming.py`
enforces this structurally (validator rejection) and by repo-wide grep for
the literal banned phrase.

## Session-specific notes for this repository

- Built entirely on a CPU-only Windows machine; `torch`/GPU code paths are
  implemented but never executed here. See `docs/GPU_VALIDATION_PLAN.md`
  before running anything on a real GPU host.
- `kvcot.generation` and `kvcot.cli` defer every torch/transformers import
  to inside the function bodies that actually need a GPU — `kvcot.analysis`
  and `kvcot.utils` never import torch at all (enforced by
  `tests/unit/test_no_analysis_torch_import.py`). Preserve this discipline
  in any new module.
- `third_party/R-KV` is checked out via a cone-mode sparse-checkout limited
  to `HuggingFace/` on this machine (Windows `MAX_PATH` issue with vendored
  `vLLM/` config files, irrelevant to this repo's scope anyway) — do a full
  `git submodule update --init --recursive` on the GPU host instead.
- License has not been chosen — see `README.md`'s License section. Do not
  pick one without asking.
