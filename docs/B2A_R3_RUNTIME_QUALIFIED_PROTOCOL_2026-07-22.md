# B2A-R3 Runtime-Qualified Calibration Protocol (dated 2026-07-22)

This document was frozen once (commit `93b6ba8`), then **repaired** in
response to a first independent audit (Step 2A, R3-AUDIT-01 through
R3-AUDIT-18), then **repaired again** in response to a second independent
re-audit of that repair (Step 2B, R3-AUDIT-19 through R3-AUDIT-25). This
is the Step 2B-repaired version. See
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md` for the full
audit-finding ledger (both rounds) and §22 below for the repair
disposition.

## 1. Protocol identity and status

```text
Protocol: B2A-R3 Runtime-Qualified Calibration
Date frozen: 2026-07-22
Date repaired (Step 2A): 2026-07-22 (same day, following first independent audit)
Date repaired (Step 2B): 2026-07-22 (same day, following second independent re-audit)
Branch: research/b2a-r3-runtime-qualified-calibration
Parent closure commit: 0fa42a7edb88e766b5665547af15a5b52e823066
Step 2A repair commit: 81e11cb57202e0d4b434aabb347963ae3c34b80b
```

```text
B2A-R3 STATUS:
PROTOCOL IMPLEMENTATION CONTRACT REPAIRED (STEP 2B) — INDEPENDENT
RE-AUDIT REQUIRED
CPU IMPLEMENTATION BLOCKED
GPU EXECUTION PROHIBITED
```

The original status line in the first-frozen version of this document
("PROTOCOL FROZEN — CPU IMPLEMENTATION AUTHORIZED — GPU EXECUTION
PROHIBITED") was itself an audit finding (R3-AUDIT-02,
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`): it read as an
unconditional grant while the surrounding paragraph gated it on an
independent audit that had not happened. This status block is corrected
to be unambiguous on its own: **no stage of this protocol is authorized
yet.** A second independent re-audit of the Step 2A repair (commit
`81e11cb57202e0d4b434aabb347963ae3c34b80b`) found the original 18 defects
materially repaired, but identified seven further implementation-level
ambiguities (R3-AUDIT-19 through R3-AUDIT-25) that could still have let
Step 3 invent behavior the protocol never actually pinned down (an unfrozen
qualification-condition tuple, incomplete artifact schemas, unresolved
selected-manifest hash semantics, an underspecified candidate-row freezing
contract, a Markdown-vs-JSON hashing conflict, a non-atomic authorization-
claim design, and conflated protocol-identity fields). All seven are now
repaired (§10.5, §12.1, §12.3-12.9, §13, §14.4; full ledger in
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`'s "Step 2B" section).
CPU implementation, CPU tests, candidate-manifest generation, and every
other Stage A activity (§14) become authorized only after a genuinely
independent re-audit of THIS Step 2B-repaired document — a review pass by
someone other than whoever wrote this repair — confirms the repair is
sound. **The repairing author does not self-certify this protocol** (§22).
GPU activity remains prohibited regardless of any future CPU-stage
authorization outcome, and requires its own separate, dated authorization
exactly as §14 (Stage B, Stage C) already required.

## 2. Background and failure-specific motivation

B2A-R2 (the single attempt authorized by `CLAUDE.md` §1d) demonstrated
that the causal-swap harness can mechanically execute end to end:

- FullKV and R-KV answers were both correct (`8` == `8`) —
  `docs/B2A_R2_RESULT_2026-07-22.md` §4.
- Replay was token-identical (Pass 1/Pass 2 confirmed) — same document, §4.
- Compression occurred: observed retention ratio 0.2797 (real, substantial
  compression) — same document, §4, "Compaction / event / pair evidence".
- Eligible events were measured: 22 observed compaction events, 20
  eligible — same location.
- Twelve real interventions and one no-op calibration interventions
  completed (12/12 real pairs, 1/1 no-op) — same location.
- Pair records are now durably persisted after the merged forensic repair
  (`docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md`, merged
  into `main` via PR #20, merge commit
  `9e78bc5edda0f0086d9e9aaea98896ac24caa7b0`).
- Peak tracked CUDA memory was 15.87 GiB / 15.98 GiB (allocated/reserved),
  under the 22 GiB limit — `docs/B2A_R2_RESULT_2026-07-22.md` §4, "Timing
  and memory" (corrected-GiB row).
- B2A-R2 failed only the runtime gate: `runtime_within_limit: False` —
  same document, §4, "Every gate". Its projected complete-pilot runtime
  was approximately 5.01 GPU-hours (18034.6s) against a 4.00-hour hard
  limit — same location.
- The selected trace (`test/number_theory/820.json`) generated 4822
  tokens (`docs/B2A_R2_RESULT_2026-07-22.md` §4, "FullKV and R-KV
  evidence"); combined with its 109-token prompt this is the 4931
  total-token figure the qualification pass itself predicted
  (`docs/B2A_R2_RESULT_2026-07-22.md` §2, ordinal 2 row; cross-verified
  against `results/decisions/b2a_r2_qualification.json`, fields
  `generated_token_count: 4822` and `total_processed_tokens: 4931`).
- The failure was trace length and repeated per-example inference cost
  driving up a 12-example/144-pair projection, not intervention semantics
  — `docs/B2A_R2_RESULT_2026-07-22.md` §6, "Verdict": "not a software
  defect, not fixed by retrying". The same section states the inherent
  tension explicitly: "a row needs enough length to trigger real events,
  but enough length drives up the per-example and projected-pilot cost."

No historical artifact above is rewritten, reinterpreted, or re-scored by
this protocol. B2A-R2 remains `B2A-R2 FINAL VERDICT: FAIL -- B2B BLOCKED`,
unchanged.

## 3. Research question

```text
Can the existing causal-swap harness produce a complete, mechanically valid,
analyzable B2A calibration under the four-GPU-hour projected B2B budget by
selecting a shorter, independently qualified reasoning trace?
```

B2A-R3 must answer:

1. Can all mechanical gates pass?
2. Can all pair outcomes be durably preserved?
3. Can projected runtime remain below four GPU-hours?
4. Is the causal signal measurable rather than at floor?

Explicitly:

- B2A-R3 is not the paper's method.
- B2A-R3 is not a method contribution.
- B2A-R3 does not establish R-KV causal mismatch by itself.
- B2A-R3 is a bounded engineering calibration.

## 4. Fixed scientific settings

Every row below is verified against a repository source, not assumed.
Unchanged by this repair (§S of the repair ledger confirms no scientific
scope drift).

| Setting | Frozen value | Source |
|---|---:|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `configs/discovery/llama8b_math500_b1024.yaml` `model.name`/`model.revision` (`6a6f4aa4197940add57724a7707d069478df56b1`) |
| Dataset | HuggingFaceH4/MATH-500, same pinned revision | `configs/discovery/llama8b_math500_b1024.yaml` `dataset.revision`: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`; identical value independently confirmed in `docs/B2A_R2_RESULT_2026-07-22.md` §3 |
| R-KV cache budget | 1024 | `configs/discovery/llama8b_math500_b1024.yaml` `rkv.budget`; file name `llama8b_math500_b1024.yaml` |
| Divide length | 128 | `configs/discovery/llama8b_math500_b1024.yaml` `rkv.divide_length` |
| Bridge tokens | 1 | `src/kvcot/discovery/constants.py` `BRIDGE_TOKEN_COUNT = 1` |
| Scored future horizon | 48 | `src/kvcot/discovery/constants.py` `SCORED_HORIZON = 48` |
| Selected events | 3 | `src/kvcot/discovery/constants.py` `EVENTS_SELECTED_PER_EXAMPLE = 3` (aliased `B2A_SELECTED_EVENTS`) |
| Candidate/donor pairs per event | 4 | `src/kvcot/discovery/constants.py` `REAL_PAIR_EVALUATIONS_PER_EVENT = CANDIDATES_PER_EVENT(2) * DONORS_PER_EVENT(2) = 4` |
| Real interventions | 12 | `src/kvcot/discovery/constants.py` `B2A_REAL_PAIR_EVALUATIONS_TOTAL = 3 * 4 = 12` |
| No-op calibration interventions | 1 | `src/kvcot/discovery/constants.py` `B2A_NOOP_CALIBRATION_COUNT = 1` |
| Batch size | 1 | `configs/discovery/llama8b_math500_b1024.yaml` `generation.batch_size` |
| GPU | One RTX 3090 | `CLAUDE.md` §4c, "Hardware" row |
| VRAM hard limit | 22 GiB | `CLAUDE.md` §4c, "Memory limit" row; `src/kvcot/discovery/b2a_qualification.py` `QUALIFICATION_MEMORY_LIMIT_BYTES = 22 * 1024**3` |
| Qualification runtime target | 3.60 GPU-hours (exact) | New B2A-R3 protocol decision (§7) — not a pre-existing repository value |
| Final runtime hard gate | 4.00 GPU-hours (exact) | `CLAUDE.md` §4c, "Runtime limit" row; `src/kvcot/discovery/execution_measurement.py` docstring; `docs/B2A_R2_RESULT_2026-07-22.md` §4 |
| Positive-gain threshold | 0.01 nats (exact) | `src/kvcot/discovery/scientific_summary.py:24` `_MEANINGFUL_GAIN_THRESHOLD` |
| Correlation ceiling | 0.30 (exact) | `docs/b0_5_decision.json` `b0_5_r2_1_repaired_gate_10`; `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §16 |
| No-op numerical-parity tolerance | 1e-9 (exact, absolute) | `src/kvcot/discovery/schemas.py:51` `_FLOAT_DIFF_TOLERANCE` |

No intended value above conflicts with the frozen B2A-R2 protocol or
`CLAUDE.md`. The two rows without a pre-existing source (qualification
runtime target, and the safety multiplier introduced in §7) are explicitly
marked as new, not presented as inherited.

```text
B2A-R3 changes only candidate qualification and row selection.
The scientific causal intervention remains unchanged.
```

## 5. Non-goals

B2A-R3 explicitly excludes:

- Changing the R-KV budget (1024).
- Changing the divide length (128).
- Shortening the scored horizon (48).
- Removing the bridge token (1).
- Reducing the number of selected events (3).
- Reducing the real-pair count (12) or no-op count (1).
- Weakening the no-op gate (§16, §17).
- Selecting a row based on causal outcomes (§6).
- Running multiple B2A-R3 attempts automatically (§14).
- Designing FaithKV.
- Running B2B.
- Claiming scientific novelty.

## 6. Outcome-blind qualification contract

### Allowed qualification information

- Dataset row identifier (`unique_id`).
- Dataset metadata such as `level`, `subject`.
- Problem text and its content hash.
- Prompt-token count.
- FullKV answer correctness.
- Answer-verifier result/status.
- Thinking-span parse validity and trace completeness (exact predicates:
  §10.1).
- Generation-cap status.
- Generated-token count.
- Total prompt-plus-generation length.
- FullKV runtime (wall seconds).
- Static R-KV compaction schedule prediction
  (`kvcot.analysis.rkv_schedule.predicted_compaction_event_positions`).
- Predicted eligible-event count
  (`kvcot.discovery.pass1.eligible_event_positions`, applied to the
  predicted schedule).
- Predicted future-token availability per candidate event.
- Frozen runtime predictor inputs and output (§7).

### Forbidden qualification information

- Swap gains (`swap_gain`).
- Pair-level baseline/swapped NLL (`baseline_per_token_nll`,
  `swapped_per_token_nll`).
- Candidate/donor causal outcomes of any kind.
- R-KV score-margin versus swap-gain correlation
  (`score_margin_e_minus_r`, `spearman_score_margin_vs_swap_gain`).
- Semantic-swap results.
- The no-op result.
- Pair-record content (`SwapPairRecord`, `rkv/pair_records.json`).
- The scientific summary (`rkv/scientific_summary.json`).
- Any B2A-R3 R-KV execution output.
- Any pair-level scientific signal from any candidate.

```text
The selected row must be frozen before any forbidden outcome exists.
```

The qualification implementation (Step 3) must not import
`kvcot.discovery.b2a_workers.run_rkv_worker`, any R-KV pair-evaluation
path, `kvcot.discovery.schemas.SwapPairRecord`, or
`kvcot.discovery.scientific_summary`. This mirrors B2A-R2's own qualifier
(`src/kvcot/discovery/b2a_qualification.py`), whose module docstring
already states: "R-KV is never imported, patched, or loaded here."

## 7. Runtime predictor

### 7.1 Ambiguity resolution: pair-time reference (mandatory, resolved)

The maximum of B2A-R2's 12 observed real-pair durations, to full recorded
precision, not a rounded display value.

- **Source artifact:** `docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json`,
  object `runtime_and_memory`, field `per_real_pair_seconds`. Independently
  cross-checked against the rounded, human-readable value in
  `docs/B2A_R2_RESULT_2026-07-22.md` §4 ("Conservative per-real-pair
  seconds | 10.25 (max of the 12)") and against source code
  `src/kvcot/discovery/execution_measurement.py:416`
  (`conservative_pair = max(b2a_real_pair_seconds)`).
- **Number of observations:** 12 (`observed_real_pair_duration_count ==
  required_real_pair_duration_count == B2A_REAL_PAIR_EVALUATIONS_TOTAL ==
  12`).
- **Precise statistic:** maximum, never mean/median — confirmed both in
  source (`max(...)`) and in `per_real_pair_projection_seconds`'s
  docstring (`src/kvcot/discovery/b2a_evidence.py:289-294`): "the MAXIMUM
  total time among the completed real pair evaluations -- never the mean
  or an aggregate bucket."
- **Whether it is conservative:** yes — the maximum overstates per-pair
  cost relative to the observed population, pushing the runtime
  projection upward rather than downward.

### 7.2 Ambiguity resolution: meaningful compression (mandatory, resolved)

Unchanged from the first-frozen version of this protocol:

```python
def derive_meaningful_compression_observed(
    *, selected_event_count: int, observed_retention_ratio: float
) -> bool:
    return selected_event_count >= 1 and observed_retention_ratio < 1.0
```

Source: `src/kvcot/discovery/b2a_evidence.py:285-286`. Inherited unchanged
— B2A-R2 passed this exact condition
(`docs/B2A_R2_RESULT_2026-07-22.md` §4, "Every gate": "meaningful_
compression_observed" listed among the passing conditions), so there is
no documented failure to repair.

This execution-time predicate is distinct from the **qualification-time**
proxy (§10.2), which additionally freezes a new, more conservative minimum
of 6 predicted eligible events (raised from B2A-R2's 3) — justified by
B2A-R2's own qualification-vs-measured attrition (31 predicted / 29
eligible at qualification time vs. 22 measured / 20 eligible at execution
time, `docs/B2A_R2_RESULT_2026-07-22.md` §4).

### 7.3 Frozen exact runtime constants

Every constant below is either read verbatim from
`docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json`'s `runtime_and_memory`
object, or algebraically derived from those exact values (never rounded
intermediate arithmetic). This repairs R3-AUDIT-03: the first-frozen
version of this protocol used the rounded display values `10.25` and
`~19.0`, which are not sufficient for a byte-identical CPU predictor
implementation or its tests.

```text
REFERENCE_TOTAL_TOKENS            = 4931
    Source: docs/B2A_R2_RESULT_2026-07-22.md §2 (ordinal-2 row) and
    results/decisions/b2a_r2_qualification.json (generated_token_count:
    4822, total_processed_tokens: 4931). Integer, exact by construction
    (109 prompt + 4822 generated).

REFERENCE_EXAMPLE_SECONDS         = 1378.3004406290129
    Source: docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json
    runtime_and_memory.per_example_total_wall_seconds (exact float as
    recorded).

REFERENCE_PAIR_SECONDS            = 10.247917714063078
    Source: docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json
    runtime_and_memory.per_real_pair_seconds (exact float as recorded;
    this is the same MAXIMUM-of-12 statistic §7.1 resolves, at full
    precision rather than the rounded "10.25" display value).

REFERENCE_PROJECTED_TOTAL_SECONDS = 18034.603590369457
    Source: docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json
    runtime_and_memory.projected_total_seconds (exact float as recorded;
    B2A-R2's OWN projection, computed by build_runtime_projection at
    execution time -- used here only to algebraically back out the setup
    component below, never as a B2A-R3 candidate's own projection).

REFERENCE_SETUP_SECONDS           = 19.298151996218968
    Derivation (algebraic, not a separately measured raw field):
        REFERENCE_SETUP_SECONDS
            = REFERENCE_PROJECTED_TOTAL_SECONDS
              - B2B_EXAMPLE_COUNT * REFERENCE_EXAMPLE_SECONDS
              - B2B_REAL_PAIR_COUNT * REFERENCE_PAIR_SECONDS
            = 18034.603590369457 - 12*1378.3004406290129 - 144*10.247917714063078
            = 19.298151996218968 (verified by direct recomputation,
              recorded in the Step 2A final report, §W)
    This is the combined FullKV-plus-R-KV one-time startup/model-load
    time. The individual FullKV-vs-R-KV split is NOT separately preserved
    in the repository (docs/B2A_R2_RESULT_2026-07-22.md §4 states this
    explicitly) -- REFERENCE_SETUP_SECONDS is a reconstructed combined
    value, not a raw measured field, and this protocol does not claim
    otherwise.

SAFETY_MULTIPLIER                 = 1.20 (exact)
    New B2A-R3 protocol decision, not read from any artifact. See §7.2 of
    the first-frozen protocol's reasoning (retained): a further, smaller
    margin layered on top of already-conservative reference statistics
    (the per-pair maximum, and B2A-R2's own real measured per-example
    cost), replacing the old, much larger x3 pre-GPU planning factor
    (docs/B0_5_PROTOCOL_REPAIR.md §15) that applied to a guessed, not
    measured, throughput.

QUALIFICATION_TARGET_HOURS        = 3.60 (exact)
FINAL_RUNTIME_LIMIT_HOURS         = 4.00 (exact)
B2B_EXAMPLE_COUNT                 = 12
    Source: src/kvcot/discovery/constants.py B2B_PILOT_EXAMPLE_COUNT;
    src/kvcot/discovery/execution_measurement.py build_runtime_projection
    default example_count=12.
B2B_REAL_PAIR_COUNT               = 144
    Source: src/kvcot/discovery/constants.py
    B2B_PILOT_TOTAL_REAL_PAIR_EVALUATIONS = 12 * 3 * 4 = 144;
    execution_measurement.py build_runtime_projection default
    real_pair_count=144.
```

**Gate decisions use these unrounded values exactly.** Human-facing CLI or
log output may round a displayed number only *after* the boolean gate
(`qualification_passed = projected_gpu_hours <= 3.60`) has already been
computed from the full-precision values above — never before, and never
by rounding an intermediate value mid-formula.

### 7.4 Frozen predictor formula

```text
reference_seconds_per_token
    = REFERENCE_EXAMPLE_SECONDS / REFERENCE_TOTAL_TOKENS

predicted_example_seconds
    = reference_seconds_per_token
      × candidate_total_tokens
      × SAFETY_MULTIPLIER

predicted_pair_seconds
    = REFERENCE_PAIR_SECONDS
      × SAFETY_MULTIPLIER

projected_total_seconds
    = REFERENCE_SETUP_SECONDS
      + B2B_EXAMPLE_COUNT × predicted_example_seconds
      + B2B_REAL_PAIR_COUNT × predicted_pair_seconds

projected_gpu_hours
    = projected_total_seconds / 3600
```

**Verified multiplicities.** `B2B_EXAMPLE_COUNT = 12` and
`B2B_REAL_PAIR_COUNT = 144` are read directly from the existing,
already-frozen projection logic (`execution_measurement.py`,
`constants.py`, cited above in §7.3) — not assumed by this protocol.

Frozen gate:

```text
qualification_passed = projected_gpu_hours <= 3.60
```

Final hard gate, unchanged and separate:

```text
runtime_within_limit (measured, at execution time) <= 4.00 GPU-hours
```

Qualifying at 3.60 does not itself satisfy the 4.00 final gate — the
actual B2A-R3 attempt's own measured evidence must still separately pass
`runtime_within_limit`, exactly as B2A-R2's did (and failed).

### 7.5 Continuous token ceiling and frozen integer test boundary

Solving §7.4's formula for `candidate_total_tokens` at
`projected_gpu_hours = QUALIFICATION_TARGET_HOURS` gives the continuous
token ceiling:

```text
continuous_token_ceiling = 2775.0857674895859...
```

**Step 3 must not hard-code this decimal.** It must derive it from the
frozen constants in §7.3 by solving the formula, so that if any reference
constant were ever amended by a future dated protocol revision, the
ceiling recomputes correctly rather than silently going stale.

Frozen integer boundary (recorded here so Step 3's tests can assert
against a known-correct expectation without re-deriving it blind):

```text
candidate_total_tokens = 2775  → projected_gpu_hours = 3.5999041059674...  → PASS
candidate_total_tokens = 2776  → projected_gpu_hours = 3.6010221756819...  → FAIL
```

Both values were independently recomputed from §7.3's exact constants
during this repair (Step 2A final report, §W) and matched the ceiling
above to full double-precision agreement. Step 3 must verify these two
exact expectations against its own implemented formula as a CPU test —
if the implementation disagrees with either row, the implementation (not
this frozen expectation) is wrong.

Rounded values (`10.25`, `~19.0`, "approximately 19", "roughly 2,750") are
removed from every ACTIVE predictor reference in this document. They may
still appear inside historical B2A-R2 narrative (§2, §7.1) where they are
clearly attributed to a human-readable historical document as presentation
rounding, never as an active predictor constant.

The predictor must preserve, for every candidate evaluated: every raw
input (candidate prompt/generated/total token counts, FullKV wall
seconds); every constant in §7.3; intermediate values
(`reference_seconds_per_token`, `predicted_example_seconds`,
`predicted_pair_seconds`); final `projected_total_seconds` and
`projected_gpu_hours`; a predictor version string; and the source
timing-artifact hash the `REFERENCE_*` constants were read from. No
predictor input may be manually overridden by an operator at
qualification time — a candidate is rejected outright if any required
predictor input is missing (§10.3).

## 8. Deterministic candidate pool

- Same pinned MATH-500 revision: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
  (§4).
- Candidate pool size: 16 candidates total (8 level-4, 8 level-5).
- Qualification is permitted for only the first 8 candidates in the final
  interleaved manifest order (§8.2); qualification stops at the first
  passing candidate.
- No earlier passing candidate is ever replaced by a later one.
- The candidate manifest is committed (Step 3) before any GPU
  qualification inference runs, exactly as B2A-R2's was.
- Row IDs (`unique_id`), content hashes, `level`, and the ordering key are
  preserved for every candidate (§12.1 schema table).

The pool is not hand-selected based on expected answer correctness,
expected trace length, mathematical topic/subject, generated length, or
any prior model behavior — membership is determined solely by (a)
pinned-dataset content, (b) the level filter, (c) the frozen exclusion set
(§8.1), and (d) the deterministic ordering key (§9), interleaved per §8.2.

### 8.1 Frozen prior-row exclusion set (repairs R3-AUDIT-05)

All 13 `unique_id` values already used by B2A-R1 or B2A-R2 are excluded.
This table is the single reproducible source of truth — Step 3 must not
run `git show` dynamically against repository history; it must use this
committed list (or verify a generated list reproduces `EXCLUSION_SET_
SHA256` below).

| `unique_id` | `source_attempt` | `source_manifest_or_commit` | `source_ordinal` |
|---|---|---|---:|
| `test/precalculus/807.json` | B2A-R1 (executed, `example_index=0`) | commit `9fe27a2` (`configs/discovery/b2a_one_example_manifest.json` content immediately before B2A-R2's freeze overwrote it; verified via `git show 9fe27a2:configs/discovery/b2a_one_example_manifest.json`) | n/a (not part of a candidate manifest) |
| `test/number_theory/427.json` | B2A-R2 (candidate, qualification-attempted) | `configs/discovery/b2a_r2_candidate_manifest.json` (`canonical_sha256 = ac2dcc4550a89f2cfa701acd608a8087b4a1ebaa0ea05eb15d8f71e3434ee0ec`) | 0 |
| `test/counting_and_probability/51.json` | B2A-R2 (candidate, qualification-attempted) | same manifest | 1 |
| `test/number_theory/820.json` | B2A-R2 (candidate, qualified and EXECUTED) | same manifest | 2 |
| `test/prealgebra/1961.json` | B2A-R2 (candidate, not attempted) | same manifest | 3 |
| `test/intermediate_algebra/1354.json` | B2A-R2 (candidate, not attempted) | same manifest | 4 |
| `test/algebra/2277.json` | B2A-R2 (candidate, not attempted) | same manifest | 5 |
| `test/intermediate_algebra/966.json` | B2A-R2 (candidate, not attempted) | same manifest | 6 |
| `test/precalculus/675.json` | B2A-R2 (candidate, not attempted) | same manifest | 7 |
| `test/counting_and_probability/894.json` | B2A-R2 (candidate, not attempted) | same manifest | 8 |
| `test/intermediate_algebra/2022.json` | B2A-R2 (candidate, not attempted) | same manifest | 9 |
| `test/counting_and_probability/181.json` | B2A-R2 (candidate, not attempted) | same manifest | 10 |
| `test/precalculus/323.json` | B2A-R2 (candidate, not attempted) | same manifest | 11 |

Correction of prior wording (R3-AUDIT-05): the first-frozen protocol said
all 12 B2A-R2 candidates are excluded "because FullKV qualification
inference already ran against three of them" — that was an imprecise
justification, since 9 of the 12 were never actually inference-attempted
(only ordinals 0-2 were). The actual, correct rule is simpler and does not
depend on which ones were inference-attempted: **the entire committed
B2A-R2 candidate manifest is excluded because every row in it was already
a designated, content-addressed candidate under a prior round's frozen
selection procedure** — re-offering any of them to B2A-R3 would blur which
round's outcome-blind procedure actually selected a row, independent of
whether qualification inference reached that specific row. This is a
protocol decision, not a claim about inference history.

Exactly 13 unique IDs (verified: no duplicates — `test/number_theory/
820.json` is the only ID appearing in both a "source" sense across
attempts, and it is listed once, at its B2A-R2 candidate-manifest
ordinal).

**Canonical exclusion payload** (UTF-8, ordered lexicographically ascending
by `unique_id` string, one per line, trailing newline required):

```text
test/algebra/2277.json
test/counting_and_probability/181.json
test/counting_and_probability/51.json
test/counting_and_probability/894.json
test/intermediate_algebra/1354.json
test/intermediate_algebra/2022.json
test/intermediate_algebra/966.json
test/number_theory/427.json
test/number_theory/820.json
test/prealgebra/1961.json
test/precalculus/323.json
test/precalculus/675.json
test/precalculus/807.json
```

```text
EXCLUSION_SET_SHA256 = 0c46510c79a22d08e8fd610104a527e867f821a540063a5059b51a660d25bc69
```

(SHA-256 of the exact 13-line UTF-8 payload above, each line terminated by
`\n` including the last. Recomputed and verified during this repair —
Step 2A final report, §W.)

Step 3 must either (a) hard-code this exact 13-row list and its hash, or
(b) generate the exclusion list from the same two sources (B2A-R1's frozen
row, B2A-R2's committed candidate manifest) and verify the result
reproduces `EXCLUSION_SET_SHA256` exactly, failing closed on any mismatch.
Dynamic `git show` invocation against repository history is permitted only
as a one-time verification step to originally freeze this list (already
done, in this document) — it must not be part of Step 3's production
candidate-generation code path.

### 8.2 Exact mixed-level candidate construction (repairs R3-AUDIT-04)

Frozen construction, in order:

1. Load every row from the pinned MATH-500 revision
   (`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`).
2. Verify every row's columns exactly match
   `src/kvcot/discovery/manifest_prepare.py`'s `EXPECTED_MATH500_COLUMNS`
   (`("problem", "solution", "answer", "subject", "level", "unique_id")`).
3. Reject any duplicate `unique_id` (fail loudly, matching
   `build_candidate_manifest`'s existing behavior).
4. Remove every row whose `unique_id` is in the frozen exclusion set
   (§8.1).
5. Keep only rows with `level == "4"` or `level == "5"` (string
   comparison, matching the dataset's own bare-digit-string convention
   confirmed in `src/kvcot/discovery/b2a_r2_candidates.py:127-130`).
6. Compute the frozen B2A-R3 ordering hash (§9) for every eligible row.
7. Sort the level-4 subset independently by `(ordering_hash ascending,
   unique_id ascending)`.
8. Sort the level-5 subset independently by the same key,
   **independently** — the two subsets are never sorted together at this
   step.
9. Take the first 8 rows of the sorted level-4 subset ("level-4 rank
   0..7").
10. Take the first 8 rows of the sorted level-5 subset ("level-5 rank
    0..7").
11. Interleave the two ranked lists exactly, level-4 first:

```text
ordinal 0  = level-4 rank 0        ordinal 8  = level-4 rank 4
ordinal 1  = level-5 rank 0        ordinal 9  = level-5 rank 4
ordinal 2  = level-4 rank 1        ordinal 10 = level-4 rank 5
ordinal 3  = level-5 rank 1        ordinal 11 = level-5 rank 5
ordinal 4  = level-4 rank 2        ordinal 12 = level-4 rank 6
ordinal 5  = level-5 rank 2        ordinal 13 = level-5 rank 6
ordinal 6  = level-4 rank 3        ordinal 14 = level-4 rank 7
ordinal 7  = level-5 rank 3        ordinal 15 = level-5 rank 7
```

12. Assign final `candidate_ordinal` values (0-15) only after
    interleaving — never before.

**Frozen consequences:**

```text
candidate count                       = 16
level-4 count                         = 8
level-5 count                         = 8
qualification limit                   = first 8 candidates (ordinals 0-7)
level-4 count within first 8          = 4  (ordinals 0,2,4,6)
level-5 count within first 8          = 4  (ordinals 1,3,5,7)
first level in manifest order         = level 4 (ordinal 0)
```

The final 16-row manifest is **never** globally re-sorted after
interleaving — the interleave order in step 11 is the final,
`candidate_ordinal`-assigning order. No subject, problem length, answer,
generated length, historical model behavior, or expected difficulty may
influence ordering at any step.

## 9. Exact deterministic ordering algorithm

Unchanged from the first-frozen protocol: B2A-R3 reuses B2A-R2's existing
ordering-hash construction (`src/kvcot/discovery/b2a_r2_candidates.py:44-53`,
`_ordering_hash`), parameterized by a new protocol-version string, rather
than inventing a new canonical-form scheme. The only change from the
first-frozen version is where in the pipeline it is applied: §8.2 now
applies it, then sorts, **per level independently**, before interleaving —
it does not sort the full eligible population globally by this key (that
would defeat the fixed 8-and-8 level mixture).

- **Hash algorithm:** SHA-256 (`kvcot.utils.hashing.sha256_text`).
- **Canonical payload:** identity fields only, pipe-separated, UTF-8:

```text
f"{candidate_order_protocol_version}|{dataset_revision}|{model_revision}|budget={budget}|{unique_id}"
```

(repairs R3-AUDIT-25: this is `candidate_order_protocol_version`, §12.9's
first identity field — never a generic, unqualified `protocol_version`.
`src/kvcot/discovery/b2a_r2_candidates.py`'s existing `_ordering_hash`
binds its own module-level `CANDIDATE_MANIFEST_PROTOCOL_VERSION` constant
into this same payload position; Step 3's B2A-R3 version reuses that
function parameterized by `candidate_order_protocol_version`'s value
instead, per this section's own instruction below.)

  Deliberately excludes problem text, subject, level, or any other content
  that could correlate with expected difficulty or trace length.
- **Frozen B2A-R3 protocol-version string:** `"faithkv-b2a-r3-row-order-v1"`
  — distinct from B2A-R2's `"faithkv-b2a-r2-row-order-v1"`
  (`CANDIDATE_MANIFEST_PROTOCOL_VERSION`,
  `src/kvcot/discovery/b2a_r2_candidates.py:39`), so no B2A-R3 ordering
  hash can coincide with a B2A-R2 one for the same `unique_id`.
- **Sort direction:** ascending by the hex-digest string.
- **Tie-breaking rule:** ascending `unique_id` string comparison —
  total and reproducible even in the astronomically unlikely case of an
  exact digest collision.
- **Whole-manifest hash:** `canonical_sha256` (§11 of this document defines
  the standardized hashing rule for all new B2A-R3 artifacts).

This protocol deliberately does not adopt a new NFC/CRLF-normalized,
problem-text-inclusive canonical form — reuse of an existing, working,
already-tested convention is preferred over reinvention, per this
document's own discipline. Step 3 must reuse
`kvcot.discovery.b2a_r2_candidates`'s existing `_ordering_hash`
construction (parameterized by the new protocol-version string), not
reimplement it independently.

## 10. Candidate qualification gates

A row qualifies only when every gate below passes.

### 10.1 Correctness and trace integrity (repairs R3-AUDIT-06, R3-AUDIT-07)

Frozen exact predicates. This repairs the first-frozen protocol's
undefined "thinking span parses correctly" / "trace is complete" prose.

```text
THINK_PARSE_SUCCESS_STATUSES = {
    "ok",
    "generation_prompt_preopened_ok",
}
```

Source: `src/kvcot/probes/early_answering.py`, `find_think_span`
(`ThinkSpanResult.think_parse_status` literal values). The function
returns exactly four possible statuses: `"ok"`,
`"generation_prompt_preopened_ok"` (both successful parses — the second
is the "preopened" case where the chat template's generation prompt
already ends inside an open think block), `"no_open_marker"`, and
`"no_close_marker"` (both failures). `THINK_PARSE_SUCCESS_STATUSES` is
exactly the two success values.

```text
thinking_span_valid =
    think_parse_status in THINK_PARSE_SUCCESS_STATUSES
    and think_start_index is not None
    and think_end_index is not None
    and think_start_index >= 0
    and think_end_index >= think_start_index
    and think_end_index <= generated_token_count
```

```text
trace_complete =
    not generation_cap_hit
    and thinking_span_valid
    and answer_verification_status != "unverifiable"
```

```text
fullkv_answer_correct =
    answer_verification_status == "correct"
```

Correctness is kept separate from structural completeness: a trace may be
structurally complete (`trace_complete = True`) but wrong
(`answer_verification_status == "incorrect"`) — it then fails the
correctness gate, never the structural gate. Conflating the two would
make a wrong-but-complete trace indistinguishable from a genuinely
malformed one during debugging.

**Evidence origin (frozen for Step 3, not implemented now):**

- Thinking-span evidence must be produced by the canonical FullKV worker,
  using the existing `find_think_span` implementation
  (`src/kvcot/probes/early_answering.py`) — the qualification evaluator
  must consume typed worker evidence, never construct a second,
  independent parser.
- Step 3 may extend the FullKV worker result schema with a
  backward-compatible, versioned shape to carry this evidence; historical
  artifacts (B2A-R1, B2A-R2) must remain parseable exactly as they are
  today — this protocol does not require or permit retroactively
  reprocessing them.
- Required future fields on that (not-yet-implemented) worker-result
  extension: `think_parse_status`, `think_start_index`, `think_end_index`,
  `generation_prompt_preopened_think`, `thinking_span_valid`,
  `trace_complete`. None of these fields are implemented by this
  documentation-only task.

Remaining 10.1 conditions (unchanged from the first-frozen protocol):

- FullKV answer is correct (`fullkv_answer_correct` above).
- Answer verifier returns a valid (not `unverifiable`) decision — mirrors
  `fullkv_answer_verifiable`/`fullkv_answer_correct` in
  `src/kvcot/discovery/b2a_qualification.py:136-138`.
- Generation cap is not hit (`no_cap_hit`, same source, line 136).
- Prompt and generation token counts are available.
- FullKV timing evidence is complete.
- No required field is manually inserted after execution — every field is
  produced by the same `run_fullkv_worker` path B2A-R2's qualifier reused.

### 10.2 Compression feasibility

- Total sequence length exceeds the R-KV budget (1024).
- Static prediction indicates real compaction
  (`kvcot.analysis.rkv_schedule.predicted_compaction_event_positions`).
- **At least 6 predicted eligible events** (raised from B2A-R2's frozen
  qualification minimum of 3 — `QUALIFICATION_MINIMUM_EVENTS = 3`,
  `src/kvcot/discovery/b2a_qualification.py:38`). New, explicitly more
  conservative B2A-R3-only qualification decision — justified in §7.2.
- **At least 3 of the predicted events each have** one bridge token
  (`BRIDGE_TOKEN_COUNT = 1`) and at least 48 future scored tokens
  (`SCORED_HORIZON = 48`) — together, 49 total future tokens required per
  event, matching `MINIMUM_FUTURE_TOKENS_AFTER_EVENT = 49`
  (`src/kvcot/discovery/constants.py:14-16`) and B2A-R2's own
  `at_least_three_events_have_49_future_tokens` condition.
- Event positions use absolute token indexing; boundary arithmetic
  matches the existing B0.5/B1 protocol
  (`src/kvcot/discovery/pass1.py` `eligible_event_positions`, reused
  unchanged, never reimplemented for B2A-R3).

### 10.3 Runtime qualification

- Projected runtime (§7.4's formula, §7.3's exact constants) is at most
  3.60 GPU-hours.
- All predictor inputs (§7.3 reference fields, candidate token counts)
  are present.
- The predictor version matches this frozen protocol.
- The safety multiplier is exactly 1.20 — never overridden per candidate.
- No runtime value is manually overridden.
- The candidate is rejected outright when any predictor input is missing
  (never silently defaulted).

### 10.4 Deterministic selection

- Candidates are evaluated strictly in the final interleaved manifest
  order (§8.2, §9).
- At most the first 8 candidates (ordinals 0-7) are evaluated.
- Evaluation stops immediately after the first passing row.
- Every attempted row (pass or fail) is preserved in the qualification
  artifact (§12).
- Every rejection reason is preserved.
- No row after a pass is ever evaluated.
- If all 8 candidates fail, no row is selected — qualification produces no
  selected row (§18, "No candidate qualifies").

### 10.5 Frozen qualification-condition tuple (repairs R3-AUDIT-19)

§10.1-10.4 above define the substantive rules; this subsection freezes the
one CONTRACT Step 3's evaluator and any future verifier must implement
against — an exact ordered tuple of condition names, an exact derivation
of `qualified`/`failed_conditions` from it, and a complete boolean
definition for every name. Before this repair, §10 was implementable in
more than one way (different condition names, a different ordering inside
`failed_conditions`, a missing condition) — exactly the defect R3-AUDIT-14
already repaired for the *final* mechanical gate tuple (§16.1). This
subsection is that same treatment applied to the *qualification* gate.

This tuple is new and specific to B2A-R3. It is never confused with, and
does not modify, B2A-R2's historical 10-name `QUALIFICATION_CONDITIONS`
(`src/kvcot/discovery/b2a_qualification.py`) — B2A-R3's qualification
gate is more granular (it adds explicit identity/hash-match and
prompt-identity conditions B2A-R2's simpler qualifier never had) because
it must qualify against the fuller runtime-predictor and cross-artifact
binding machinery this protocol adds (§7, §9, §12).

#### 10.5.1 Frozen exact ordered tuple

```python
B2A_R3_QUALIFICATION_CONDITIONS = (
    "no_cap_hit",
    "answer_verifiable",
    "fullkv_answer_correct",
    "thinking_span_valid",
    "trace_complete",
    "prompt_token_count_present",
    "generated_token_count_present",
    "fullkv_timing_complete",
    "candidate_manifest_hash_match",
    "config_hash_match",
    "dataset_identity_match",
    "model_identity_match",
    "tokenizer_identity_match",
    "generation_config_hash_match",
    "prompt_identity_match",
    "batch_size_is_one",
    "all_parameters_on_requested_cuda",
    "no_offload_verified",
    "peak_memory_within_limit",
    "sequence_exceeds_budget",
    "predicted_compaction_present",
    "predicted_event_count_at_least_six",
    "at_least_three_events_have_49_future_tokens",
    "runtime_inputs_complete",
    "runtime_predictor_version_match",
    "safety_multiplier_exact",
    "projected_runtime_within_qualification_target",
)
```

```text
Exact count: 27 conditions. All 27 names are unique.
```

#### 10.5.2 Frozen derivation

```python
qualified = all(conditions[name] for name in B2A_R3_QUALIFICATION_CONDITIONS)

failed_conditions = [
    name
    for name in B2A_R3_QUALIFICATION_CONDITIONS
    if not conditions[name]
]
```

`conditions` is a mapping from every name in the tuple to a concrete
`bool` (never `None`, never a non-bool truthy/falsy value, never a string).
`failed_conditions` preserves the tuple's own order, never an
implementation-dependent iteration order (e.g. dict insertion order of a
differently-constructed mapping).

#### 10.5.3 Mandatory rejection rules

The evaluator and any future verifier must reject, as a hard failure (not
a warning, not a silent default):

- A `conditions` mapping missing any of the 27 names.
- A `conditions` mapping containing any name outside the 27.
- Any condition value that is not a concrete Python `bool` (in particular,
  `0`/`1`/`None`/a string is rejected, never coerced).
- A qualification outcome whose named condition fields (§12.5) do not
  exactly match its own `conditions` map (no condition may be "stored"
  under a different field name than the one in the tuple).
- A `failed_conditions` array that is not exactly `[name for name in
  B2A_R3_QUALIFICATION_CONDITIONS if not conditions[name]]` (wrong order,
  missing entry, or extra entry).
- A `qualified` value that disagrees with `all(conditions.values())`.

#### 10.5.4 Condition definitions

Each condition below is fully and exactly defined. Where a condition
reuses a predicate already frozen elsewhere in this protocol, this section
cites that definition rather than restating it independently — there is
exactly one authoritative definition per predicate anywhere in this
document.

**`no_cap_hit`**

```text
no_cap_hit = (cap_hit is False)
```

**`answer_verifiable`**

```text
answer_verifiable = (answer_verification_status != "unverifiable")
```

**`fullkv_answer_correct`**

```text
fullkv_answer_correct = (answer_verification_status == "correct")
```

Identical to §10.1's `fullkv_answer_correct`; not redefined, only listed
here as a named member of the frozen tuple.

**`thinking_span_valid`**

The exact, unchanged §10.1 predicate:

```text
thinking_span_valid =
    think_parse_status in THINK_PARSE_SUCCESS_STATUSES
    and think_start_index is not None
    and think_end_index is not None
    and think_start_index >= 0
    and think_end_index >= think_start_index
    and think_end_index <= generated_token_count
```

**`trace_complete`**

The exact, unchanged §10.1 predicate:

```text
trace_complete =
    not generation_cap_hit
    and thinking_span_valid
    and answer_verification_status != "unverifiable"
```

**`prompt_token_count_present`**

```text
prompt_token_count_present =
    type(prompt_token_count) is int
    and prompt_token_count > 0
```

`type(x) is int`, never `isinstance(x, int)` — Python's `bool` is an `int`
subclass, so `isinstance(True, int)` is `True`; `type(x) is int` is the
one check that actually rejects a boolean masquerading as a count.

**`generated_token_count_present`**

```text
generated_token_count_present =
    type(generated_token_count) is int
    and generated_token_count > 0
    and generated_token_count == len(natural_generated_token_ids)
```

The qualification artifact (§12.5) may store only
`generated_token_ids_sha256` rather than the complete token array, but
this equality must be established by the worker/evaluator boundary code
from the actual `natural_generated_token_ids` array BEFORE that array is
discarded and only its count/hash persisted into the artifact — never
trusted as a caller-supplied integer with no underlying array to check it
against.

**`fullkv_timing_complete`**

```text
fullkv_timing_complete =
    type(fullkv_wall_seconds) in (int, float)
    and type(fullkv_wall_seconds) is not bool
    and math.isfinite(fullkv_wall_seconds)
    and fullkv_wall_seconds >= 0.0
    and fullkv_timing_evidence is present and complete,
        per the existing FullKV worker timing-evidence contract
        (the SAME `timing_evidence`/`FullKVWorkerResult` timing contract
        `kvcot.discovery.b2a_workers.run_fullkv_worker` already produces
        and `kvcot.discovery.attempt_verification` already cross-checks
        via `{role}/timing.json` == `result.json`'s `timing_evidence` —
        never a second, independently invented phase-timing list)
```

**`candidate_manifest_hash_match`**

```text
candidate_manifest_hash_match =
    ( the qualification runner independently recomputed and verified
      the candidate manifest's own canonical_sha256, per §12.1's
      five-step verification procedure )
    and ( qualification_artifact.candidate_manifest_canonical_sha256
          == candidate_manifest.canonical_sha256 )
```

**`config_hash_match`**

```text
config_hash_match =
    qualification_artifact.config_sha256
    == config_identity("configs/discovery/llama8b_math500_b1024.yaml")
```

using the repository's existing `kvcot.config.config_identity` (itself
`sha256_file`) — never a second, independently defined config-hash
algorithm.

**`dataset_identity_match`**

```text
dataset_identity_match = all of:
    worker_dataset_repo    == "HuggingFaceH4/MATH-500"
    worker_dataset_config  == "default"
    worker_dataset_split   == "test"
    worker_dataset_revision == "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    worker-reported unique_id        == candidate.unique_id
    worker-reported source_example_index == candidate.source_example_index
    candidate.raw_row_sha256   reproduces sha256_json(candidate.row)
    candidate.problem_sha256   reproduces sha256_text(candidate.row["problem"])
    candidate.gold_answer_sha256 reproduces sha256_text(candidate.row["answer"])
```

(the last three reuse §12.4's frozen verification formulas, not a second
definition of them).

**`model_identity_match`**

```text
model_identity_match =
    worker_model_name     == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    and worker_model_revision == "6a6f4aa4197940add57724a7707d069478df56b1"
```

**`tokenizer_identity_match`**

```text
tokenizer_identity_match =
    worker_tokenizer_name     == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    and worker_tokenizer_revision == "6a6f4aa4197940add57724a7707d069478df56b1"
```

**`generation_config_hash_match`**

Frozen generation-configuration payload (byte-for-byte the `generation:`
block of `configs/discovery/llama8b_math500_b1024.yaml`, confirmed by
direct inspection during this repair):

```json
{
  "generation_mode": "greedy",
  "do_sample": false,
  "temperature": null,
  "top_p": null,
  "batch_size": 1,
  "max_new_tokens": 6144,
  "framework_seed": 13,
  "attention_backend": "flash_attention_2",
  "cache_implementation": "DynamicCache",
  "no_offload_required": true
}
```

```text
generation_config_sha256 = sha256_json(the exact object above)
                          = b67ed818bb94e0a674e01c1400caf2e005c41c12203d6a29b0f0f5c69eef01a3
```

Computed twice during this repair, using the repository's actual
`kvcot.utils.hashing.sha256_json`, from two independent calls — both
calls returned this identical 64-lowercase-hex value. Never guessed, never
manually calculated.

```text
generation_config_hash_match =
    qualification_artifact.generation_config_sha256 == the frozen value above
```

**`prompt_identity_match`**

```text
prompt_identity_match =
    observed_prompt_token_ids_sha256 == expected_prompt_token_ids_sha256

where expected_prompt_token_ids_sha256 is derived by the canonical FullKV
path from:
    - the exact candidate raw row (candidate.row, §12.4)
    - the frozen tokenizer revision (6a6f4aa4197940add57724a7707d069478df56b1)
    - the canonical kvcot.discovery.manifest_prepare._render_and_tokenize
      path (never a second, independently-invented rendering path)
    - the frozen chat-template rendering arguments (one user-role message,
      add_generation_prompt=True, tokenize=True — kvcot.discovery.manifest
      .ChatTemplateRenderingConfig's existing frozen convention)

and observed_prompt_token_ids_sha256 is the hash of the token IDs the
FullKV worker actually used for inference.
```

Both `expected_prompt_token_ids_sha256` and `observed_prompt_token_ids_
sha256` must be preserved on the qualification outcome (§12.5) so this
condition is independently re-derivable — a bare caller-supplied
`prompt_identity_match=True` boolean with no underlying hash pair is
forbidden and must be rejected by the evaluator.

**`batch_size_is_one`**

```text
batch_size_is_one = (actual_batch_size == 1)
```

**`all_parameters_on_requested_cuda`**

Derived from raw `ParameterPlacementEvidence`
(`src/kvcot/discovery/strict_device.py`'s
`verify_placement_from_raw_evidence`/its per-worker `_ok` helper), never
from only the legacy summary boolean `every_parameter_on_cuda` when the
stricter raw fields are available:

```text
all_parameters_on_requested_cuda =
    parameter_placement_evidence.requested_device == "cuda:0"
    and parameter_placement_evidence.every_parameter_on_cuda is True
    and parameter_placement_evidence.parameter_count > 0
    and list(parameter_placement_evidence.unique_device_types) == ["cuda"]
    and list(parameter_placement_evidence.unique_devices) == ["cuda:0"]
```

**`no_offload_verified`**

```text
no_offload_verified =
    parameter_placement_evidence.no_offload_verified is True
    and ( parameter_placement_evidence.hf_device_map is None
          or every value in hf_device_map maps to cuda:0/"0"/"cuda",
             never "cpu"/"disk"/"meta"/"auto" )
```

Rejects CPU placement, disk offload, meta-device placement, device-map
offload, or any unverified placement — reusing
`verify_placement_from_raw_evidence`'s existing per-worker rule, never a
second, independently-written placement check.

**`peak_memory_within_limit`**

```text
peak_memory_within_limit =
    type(peak_cuda_allocated_bytes) is int
    and type(peak_cuda_reserved_bytes) is int
    and peak_cuda_allocated_bytes >= 0
    and peak_cuda_reserved_bytes >= 0
    and max(peak_cuda_allocated_bytes, peak_cuda_reserved_bytes)
        <= 22 * 1024**3
```

**`sequence_exceeds_budget`**

```text
sequence_exceeds_budget = (total_processed_tokens > 1024)
```

**`predicted_compaction_present`**

```text
predicted_compaction_present =
    len(predicted_compaction_event_positions) >= 1
```

**`predicted_event_count_at_least_six`**

```text
predicted_event_count_at_least_six =
    predicted_event_count == len(predicted_compaction_event_positions)
    and predicted_event_count >= 6
```

This is the total-predicted-events count (§10.2's raised-from-3-to-6
minimum) — never conflated with the next condition's 3-eligible-event
count.

**`at_least_three_events_have_49_future_tokens`**

Uses the existing, unchanged `eligible_event_positions`
(`src/kvcot/discovery/pass1.py`) applied to the predicted schedule:

```text
eligible_event_count == len(eligible_event_indices)
and at_least_three_events_have_49_future_tokens = (eligible_event_count >= 3)
```

The 6-event minimum above applies to TOTAL predicted compaction events;
this 3-event minimum applies only to the subset of those events satisfying
the exact bridge-token-plus-48-scored-future-token condition
(`MINIMUM_FUTURE_TOKENS_AFTER_EVENT = 49`). These are two different counts
over two different (nested) sets and must never be conflated or computed
from the same variable.

**`runtime_inputs_complete`**

```text
runtime_inputs_complete = all of:
    every §7.3 reference constant is present and unmodified
    candidate_total_tokens (= total_processed_tokens) is present
    no NaN, no Infinity, anywhere in the above
    no negative timing value anywhere in the above
    no missing token count
    no boolean value substituted for an integer/float input
        (type(x) is bool is rejected wherever an int/float is required)
    no manually overridden constant (every §7.3 constant matches its
        frozen literal value exactly)
```

A candidate missing any required predictor input is rejected outright —
never silently defaulted (§7.5's existing rule, restated here as a named
condition rather than only prose).

**`runtime_predictor_version_match`**

```text
runtime_predictor_version = "faithkv-b2a-r3-runtime-predictor-v1"

runtime_predictor_version_match =
    (the predictor implementation's own version string
     == "faithkv-b2a-r3-runtime-predictor-v1")
```

**`safety_multiplier_exact`**

```text
safety_multiplier_exact = (safety_multiplier == 1.20)
```

Exact protocol equality (`==`) on the stored configuration value — `1.20`
is exactly representable in IEEE-754 binary64 as literal `1.2`, so this is
a genuine equality check, never a tolerance-based comparison.

**`projected_runtime_within_qualification_target`**

```text
projected_runtime_within_qualification_target =
    (projected_gpu_hours <= 3.60)
```

using §7.4's exact unrounded formula and §7.3's exact constants — never a
rounded intermediate value (§7.3's existing rule).

## 11. Qualification wall-time and bounded execution (repairs R3-AUDIT-11)

This section defines, but does **not** itself authorize, a bounded
FullKV-only qualification phase — that phase requires its own future,
separate authorization (§14, Stage B).

The first-frozen version of this protocol both said `VERIFY BEFORE
FREEZE` for the wall-time limit AND required Step 3 to "calculate and
display the protocol-derived wall-time limit" without freezing a formula
to derive it from — a direct contradiction (R3-AUDIT-11). This is
repaired by freezing the one number that IS a mathematical consequence of
already-frozen values (the per-candidate worker timeout and the
candidate cap), while being explicit that this number is an **envelope**,
not an authorized GPU budget or the actual Stage B wall-time limit:

```text
PER_CANDIDATE_WORKER_TIMEOUT_SECONDS = 7200
    Source: src/kvcot/discovery/constants.py B2A_WORKER_TIMEOUT_SECONDS
    (the existing, unmodified per-worker subprocess timeout).
QUALIFICATION_CANDIDATE_LIMIT        = 8   (§8, §10.4)
ABSOLUTE_TIMEOUT_ENVELOPE_SECONDS    = 57600   (= 8 × 7200, pure arithmetic)
```

Explicitly:

- `57600` is only the mathematical envelope `8 × 7200`. It is **not** an
  authorized GPU budget, and it is **not** the final qualification
  phase-wide wall-time limit.
- Step 3's CPU dry-run may display this envelope as a computed value.
- Step 3's dry-run planning output must also display, verbatim:

```text
qualification_phase_wall_time_limit = null
gpu_qualification_authorized = false
wall_time_authorization_required = true
```

- A separate Stage B authorization document must freeze a stricter,
  actual phase-wide wall-time limit before any GPU qualification runs.
  This protocol does not invent that number — B2A-R2's own qualification
  pass needed only ~9.9 minutes across 3 attempted candidates
  (`docs/B2A_R2_RESULT_2026-07-22.md` §2), a single data point that is
  informative but not a worst-case bound for up to 8 attempts against a
  wider candidate pool.
- Step 3 must not invent or apply a real GPU qualification wall-time
  limit of its own.
- The absence of a Stage B wall-time limit does **not** block CPU
  implementation (Stage A) — it blocks only Stage B GPU use.

## 12. Candidate and qualification artifact schemas

Conceptual schemas only — no Python classes, and no runtime artifacts of
any kind, are implemented or generated by this task.

### 12.1 Canonical hashing rule (repairs R3-AUDIT-08)

Standardized field name for every new B2A-R3 artifact's self-hash:
`canonical_sha256` (never `manifest_hash`/`artifact_hash`/other
synonyms — repairs R3-AUDIT-16's inconsistent naming for this field
specifically).

```text
canonical_sha256 = sha256_json(payload with canonical_sha256 omitted)
```

using the repository's existing, unmodified `kvcot.utils.hashing
.sha256_json` (`json.dumps(obj, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)`, then SHA-256 hex digest) — never a second,
independently-defined hash function.

Verification procedure (frozen for Step 3):

1. Parse the JSON object.
2. Require exactly one field named `canonical_sha256`, a lowercase 64-hex-
   character string.
3. Remove the `canonical_sha256` field from the parsed object.
4. Recompute `sha256_json` over the remaining object.
5. Require exact string equality with the value removed in step 2.

A payload must never be hashed while still containing its own hash field
— this rule applies to the candidate manifest, the qualification artifact,
the selection-provenance artifact, and the authorization-claim artifact
(§14.4-14.5) — every one of which is a JSON object containing its own
`canonical_sha256` field.

**Scope boundary (repairs R3-AUDIT-23): this rule applies ONLY to JSON
artifacts that actually contain a `canonical_sha256` field.** It does
NOT apply to:

- The dated Markdown authorization documents (§12.2's
  `docs/B2A_R3_STAGE_{B,C}_..._AUTHORIZATION_<date>.md` pattern) — these
  are Markdown files, not JSON objects, so `sha256_json`'s
  self-referential-field-omission procedure cannot apply to them (there is
  no JSON object to parse and no field to omit). Their hash is
  `authorization_document_sha256 = sha256_file(the exact committed
  Markdown document)` — a plain whole-file byte hash, using the
  repository's existing `kvcot.utils.hashing.sha256_file` (the same
  helper `kvcot.config.config_identity` already uses for config-file
  hashing) — never `sha256_json`, and never a second, independently
  invented hash function.
- The selected one-example execution manifest
  (`configs/discovery/b2a_one_example_manifest.json`,
  `B2AOneExampleManifest`) — see §12.7/R3-AUDIT-21: this file has never
  carried a `canonical_sha256` field and this protocol does not add one to
  it. Its hash is the existing external `manifest_hash()` method
  (`selected_manifest_sha256`, §12.7), never `canonical_sha256`.

Every reference to "the canonical hashing rule" elsewhere in this document
(§12.3, §12.5, §12.6, §12.7's `qualification_artifact_canonical_sha256`/
`candidate_manifest_canonical_sha256` fields, §12.8's authorization-claim
schema) means the JSON `canonical_sha256`/`sha256_json` rule above, applied
only to the JSON artifact it is attached to — never applied to a Markdown
document or to the selected one-example manifest.

### 12.2 Frozen exact artifact paths (repairs R3-AUDIT-10)

Every "recommended path" in the first-frozen protocol is replaced with one
exact, unambiguous path:

```text
Candidate manifest:
    configs/discovery/b2a_r3_candidate_manifest.json
    (mirrors configs/discovery/b2a_r2_candidate_manifest.json's existing
    location convention)

Qualification artifact:
    results/decisions/b2a_r3_qualification.json
    (mirrors results/decisions/b2a_r2_qualification.json's existing
    location convention)

Selected one-example execution manifest:
    configs/discovery/b2a_one_example_manifest.json
    (SAME file B2A-R1/B2A-R2 used -- the freezer, §13, replaces its
    content in place, exactly as kvcot.discovery.b2a_r2_freeze already
    does; no new filename is introduced for this artifact)

Selection provenance:
    results/decisions/b2a_r3_selection_provenance.json
    (companion file, mirroring kvcot.discovery.b2a_r2_freeze
    .SelectionProvenance's existing role; B2A-R2's own equivalent forensic
    index additionally lives at
    docs/evidence/B2A_R2_SELECTION_PROVENANCE_2026-07-22.json, but that is
    a documentation-evidence copy assembled during forensic writeup, not
    the primary artifact -- results/decisions/ is where the primary,
    machine-authored qualification/selection artifacts already live for
    B2A-R2, so B2A-R3's primary selection-provenance artifact uses the
    same directory)

Attempt directory root (Stage C execution, when authorized):
    results/decisions/b2a_r3_attempt_<UTC-timestamp>_<attempt_id>/
    (reuses B2A-R2's exact existing directory-naming convention, e.g.
    results/decisions/b2a_attempt_20260722T101253300941Z_
    fb6f5081d47f45f4b4f9258c25e6883d/ -- only the "b2a_" prefix becomes
    "b2a_r3_" to distinguish rounds)

Global authorization claim (repairs R3-AUDIT-24 — replaces the prior
"inside the attempt directory" design; see §14.4):
    results/decisions/b2a_r3_authorization_claims/<authorization_id>.json
    (one deterministic path per authorization_id, created via an
    exclusive-create filesystem primitive BEFORE the attempt directory
    exists -- creation of THIS file is the consumption event, not a copy
    later placed inside the attempt directory)

Attempt-directory claim copy/reference (written only AFTER the global
claim above already exists, never a second authorization-bearing claim):
    <attempt directory root>/authorization_claim.json

Dated authorization documents (naming pattern, not a literal path -- the
date is filled in at authorization time, never fabricated in advance):
    docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_<YYYY-MM-DD>.md
    docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_<YYYY-MM-DD>.md
```

No path above is qualified with "recommended," "such as," or an
alternative — each is the one path Step 3 must use.

### 12.3 Candidate manifest schema (repairs R3-AUDIT-20, R3-AUDIT-25)

| Field | Type | Nullable/Required | Meaning | Hash inclusion | Source |
|---|---|---|---|---|---|
| `artifact_schema_version` | string | required | `"faithkv-b2a-r3-candidate-manifest-v1"` | included | new (§12.9) |
| `candidate_order_protocol_version` | string | required | `"faithkv-b2a-r3-row-order-v1"` (§9) | included | new (§12.9) |
| `dataset_repo` | string | required | `"HuggingFaceH4/MATH-500"` | included | §4 |
| `dataset_config` | string | required | `"default"` | included | §4 |
| `dataset_split` | string | required | `"test"` | included | §4 |
| `dataset_revision` | string | required | `"6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"` (§4) | included | `configs/discovery/llama8b_math500_b1024.yaml` |
| `model_name` | string | required | `"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"` | included | same config |
| `model_revision` | string | required | `"6a6f4aa4197940add57724a7707d069478df56b1"` (§4) | included | same config |
| `tokenizer_name` | string | required | `"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"` | included | same config |
| `tokenizer_revision` | string | required | `"6a6f4aa4197940add57724a7707d069478df56b1"` | included | same config |
| `budget` | integer | required | `1024` | included | same config |
| `config_path` | string | required | `"configs/discovery/llama8b_math500_b1024.yaml"` | included | §12.2 |
| `config_sha256` | string (64-hex) | required | `config_identity(config_path)` (`sha256_file`, §10.5) | included | `kvcot.config.config_identity` |
| `generation_config_sha256` | string (64-hex) | required | §10.5's frozen `b67ed818bb94e0a674e01c1400caf2e005c41c12203d6a29b0f0f5c69eef01a3` | included | §10.5 |
| `exclusion_set_sha256` | string (64-hex) | required | §8.1's `EXCLUSION_SET_SHA256` | included | §8.1 |
| `candidate_count` | integer | required | `16` | included | §8.2 |
| `qualification_limit` | integer | required | `8` | included | §8.2, §10.4 |
| `level_mixture` | object | required | `{"level_4": 8, "level_5": 8}` | included | §8.2 |
| `candidates` | array | required | ordered list of candidate rows (§12.4), in final interleaved order | included | §8.2 |
| `canonical_sha256` | string (64-hex) | required | this manifest's own self-hash (§12.1) | **excluded from its own computation** | §12.1 |

`created_at` and any other timestamp, random ID, filesystem path, or
network-fetch timestamp are **removed from the canonical payload**
(repairs R3-AUDIT-09): the committed candidate manifest must be
reproducible byte-for-byte from the same dataset and protocol inputs
alone. Any operational timestamp of interest may be recorded in a
separate, noncanonical log file, never inside this manifest's hashed
content.

The verifier must reject a manifest with a field name outside this table,
a missing required field, or a `candidates` array whose length disagrees
with `candidate_count`.

### 12.4 Candidate row schema (repairs R3-AUDIT-20, R3-AUDIT-22)

| Field | Type | Nullable/Required | Meaning | Hash inclusion | Source |
|---|---|---|---|---|---|
| `candidate_ordinal` | integer, 0-15 | required | final position after interleaving (§8.2) | included | mirrors `CandidateRow.candidate_ordinal` |
| `source_example_index` | integer | required | row's index in the pinned dataset file | included | mirrors `CandidateRow.source_example_index` |
| `unique_id` | string | required | MATH-500 `unique_id` | included | mirrors `CandidateRow.unique_id` |
| `subject` | string | required | MATH-500 `subject` column | included | mirrors `CandidateRow.subject` |
| `level` | integer, 4 or 5 | required | MATH-500 `level` column | included | mirrors `CandidateRow.level` |
| `row` | object | required | the COMPLETE pinned MATH-500 row, embedded (never refetched — see below) | included | mirrors `CandidateRow.row` |
| `raw_row_sha256` | string (64-hex) | required | `sha256_json(row)` | included | mirrors `CandidateRow.raw_row_sha256` |
| `problem_sha256` | string (64-hex) | required | `sha256_text(row["problem"])` | included | mirrors `CandidateRow.problem_sha256` |
| `gold_answer_sha256` | string (64-hex) | required | `sha256_text(row["answer"])` | included | mirrors `CandidateRow.gold_answer_sha256` |
| `ordering_hash` | string (64-hex) | required | §9's per-row ordering key | included | mirrors `CandidateRow.ordering_hash` |

No field on this row may carry a predicted or observed model outcome —
mirrors B2A-R2's own `CandidateRow`, which stores only identity/content
fields.

**Frozen embedded-row columns (repairs R3-AUDIT-22):** `row` must have
exactly the columns `("problem", "solution", "answer", "subject", "level",
"unique_id")` — `src/kvcot/discovery/manifest_prepare.py`'s
`EXPECTED_MATH500_COLUMNS`, reused unchanged.

**Frozen verification formulas, mandatory for both the candidate generator
and any future verifier:**

```text
sha256_json(row) == raw_row_sha256
sha256_text(row["problem"]) == problem_sha256
sha256_text(row["answer"]) == gold_answer_sha256

row["unique_id"] == unique_id
row["subject"] == subject
int(row["level"]) == level
```

**Frozen embed-not-refetch rule (repairs R3-AUDIT-22):** the complete
pinned row is embedded in `row` at manifest-construction time. The future
selected-row freezer (§13) MUST use this embedded `row` and MUST NOT
refetch the dataset during selected-row freezing — this is not left as an
implementation choice. This mirrors
`kvcot.discovery.b2a_r2_freeze.freeze_qualified_row`'s existing behavior
exactly (it reads `candidate_row["row"]` and recomputes `sha256_json(row)`
against the stored `raw_row_sha256`, never re-fetching from the network),
and is required so freezing stays deterministic and immune to a
since-changed live source. The candidate manifest's own top-level pinned
dataset identity (`dataset_repo`/`dataset_config`/`dataset_split`/
`dataset_revision`, §12.3) plus each row's own content hashes together
bind every embedded row back to the frozen dataset-selection procedure
(§8, §9) without requiring a second network fetch to verify it.

### 12.5 Qualification outcome schema (repairs R3-AUDIT-19, R3-AUDIT-20)

Per attempted candidate. Every field named in §10.5's condition
definitions must be present so every one of the 27 conditions is
independently re-derivable from this record alone (plus the candidate
manifest and config file it is read against).

| Field | Type | Nullable/Required | Meaning | Hash inclusion | Source |
|---|---|---|---|---|---|
| `candidate_ordinal` | integer | required | which manifest row this outcome is for | included | mirrors `CandidateQualificationOutcome.candidate_ordinal` |
| `source_example_index` | integer | required | mirrors the manifest row's own field | included | §12.4 |
| `unique_id` | string | required | row identity, redundant with the manifest for audit convenience | included | mirrors same-named field |
| `raw_row_sha256` | string (64-hex) | required | echoes the candidate row's own hash (§12.4) | included | §12.4 |
| `problem_sha256` | string (64-hex) | required | echoes the candidate row's own hash (§12.4) | included | §12.4 |
| `gold_answer_sha256` | string (64-hex) | required | echoes the candidate row's own hash (§12.4) | included | §12.4 |
| `worker_dataset_repo` | string | required | worker-reported, must equal `"HuggingFaceH4/MATH-500"` (§10.5) | included | new (§10.5) |
| `worker_dataset_config` | string | required | worker-reported, must equal `"default"` | included | new (§10.5) |
| `worker_dataset_split` | string | required | worker-reported, must equal `"test"` | included | new (§10.5) |
| `worker_dataset_revision` | string | required | worker-reported, must equal §4's pinned revision | included | new (§10.5) |
| `worker_model_name` | string | required | worker-reported, must equal §4's pinned model name | included | new (§10.5) |
| `worker_model_revision` | string | required | worker-reported, must equal §4's pinned model revision | included | new (§10.5) |
| `worker_tokenizer_name` | string | required | worker-reported, must equal §4's pinned tokenizer name | included | new (§10.5) |
| `worker_tokenizer_revision` | string | required | worker-reported, must equal §4's pinned tokenizer revision | included | new (§10.5) |
| `expected_prompt_token_ids_sha256` | string (64-hex) | required | derived via the canonical rendering path (§10.5's `prompt_identity_match`) | included | new (§10.5) |
| `observed_prompt_token_ids_sha256` | string (64-hex) | required | the hash the FullKV worker actually used | included | new (§10.5) |
| `prompt_token_count` | integer | required | mirrors `CandidateQualificationOutcome.prompt_token_count` | included | same |
| `generated_token_count` | integer | required | mirrors same-named field | included | same |
| `generated_token_ids_sha256` | string | required | mirrors same-named field | included | same |
| `total_processed_tokens` | integer | required | prompt + generated | included | same |
| `cap_hit` | boolean | required | mirrors same-named field | included | same |
| `extracted_answer` | string, nullable | nullable | mirrors same-named field | included | same |
| `answer_verification_status` | string | required | `"correct"`/`"incorrect"`/`"unverifiable"` | included | same |
| `think_parse_status` | string | required | §10.1 | included | new (§10.1) |
| `think_start_index` | integer, nullable | nullable | §10.1 | included | new (§10.1) |
| `think_end_index` | integer, nullable | nullable | §10.1 | included | new (§10.1) |
| `generation_prompt_preopened_think` | boolean | required | §10.1 (`ThinkSpanResult` field, reused unchanged) | included | new (§10.1) |
| `thinking_span_valid` | boolean | required | §10.1's derived predicate | included | new (§10.1) |
| `trace_complete` | boolean | required | §10.1's derived predicate | included | new (§10.1) |
| `fullkv_wall_seconds` | float | required | mirrors same-named field | included | same |
| `fullkv_timing_evidence` | object | required | the existing FullKV worker timing-evidence contract, reused unchanged | included | new (§10.5) |
| `requested_device` | string | required | must equal `"cuda:0"` | included | new (§10.5) |
| `parameter_placement_evidence` | object | required | raw `ParameterPlacementEvidence` (`requested_device`, `every_parameter_on_cuda`, `no_offload_verified`, `parameter_count`, `unique_device_types`, `unique_devices`, `hf_device_map`) | included | `src/kvcot/discovery/strict_device.py` |
| `actual_batch_size` | integer | required | mirrors same-named worker field | included | same |
| `peak_cuda_allocated_bytes` | integer | required | mirrors same-named field | included | same |
| `peak_cuda_reserved_bytes` | integer | required | mirrors same-named field | included | same |
| `peak_cuda_tracked_bytes` | integer | required | `max(allocated, reserved)` | included | same |
| `predicted_compaction_event_positions` | array of int | required | mirrors same-named field | included | same |
| `predicted_event_count` | integer | required | mirrors same-named field | included | same |
| `eligible_event_indices` | array of int | required | mirrors same-named field | included | same |
| `eligible_event_count` | integer | required | mirrors same-named field | included | same |
| `reference_seconds_per_token` | float | required | §7.4 intermediate | included | new (§7.4) |
| `predicted_example_seconds` | float | required | §7.4 intermediate | included | new (§7.4) |
| `predicted_pair_seconds` | float | required | §7.4 intermediate | included | new (§7.4) |
| `projected_total_seconds` | float | required | §7.4 output | included | new (§7.4) |
| `projected_gpu_hours` | float | required | §7.4 output | included | new (§7.4) |
| `safety_multiplier` | float | required | must equal `1.20` exactly | included | §7.3 |
| `runtime_predictor_version` | string | required | must equal `"faithkv-b2a-r3-runtime-predictor-v1"` | included | §10.5 |
| `conditions` | object (bool map) | required | §10.5's 27 named conditions, by name, to a concrete `bool` | included | §10.5 |
| `qualified` | boolean | required | §10.5.2's derivation | included | §10.5 |
| `failed_conditions` | array of string | required | §10.5.2's derivation, tuple order preserved | included | §10.5 |

The verifier must reject an outcome record containing a field name outside
this table, or missing any required field above.

### 12.6 Qualification artifact schema (repairs R3-AUDIT-20, R3-AUDIT-25)

Top level.

| Field | Type | Nullable/Required | Meaning | Hash inclusion | Source |
|---|---|---|---|---|---|
| `artifact_schema_version` | string | required | `"faithkv-b2a-r3-qualification-artifact-v1"` | included | new (§12.9) |
| `candidate_order_protocol_version` | string | required | `"faithkv-b2a-r3-row-order-v1"` (§9) | included | new (§12.9) |
| `qualification_protocol_version` | string | required | `"faithkv-b2a-r3-qualification-v1"` | included | new (§12.9) |
| `runtime_predictor_version` | string | required | `"faithkv-b2a-r3-runtime-predictor-v1"` | included | §10.5, §12.9 |
| `candidate_manifest_path` | string | required | `"configs/discovery/b2a_r3_candidate_manifest.json"` | included | §12.2 |
| `candidate_manifest_canonical_sha256` | string | required | §12.3's `canonical_sha256`, bound at read time | included | §12.1 |
| `config_path` | string | required | `"configs/discovery/llama8b_math500_b1024.yaml"` | included | §12.2 |
| `config_sha256` | string (64-hex) | required | `config_identity(config_path)` | included | §10.5 |
| `generation_config_sha256` | string (64-hex) | required | §10.5's frozen value | included | §10.5 |
| `dataset_repo` | string | required | `"HuggingFaceH4/MATH-500"` | included | §4 |
| `dataset_config` | string | required | `"default"` | included | §4 |
| `dataset_split` | string | required | `"test"` | included | §4 |
| `dataset_revision` | string | required | `"6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"` | included | §4 |
| `model_name` | string | required | `"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"` | included | §4 |
| `model_revision` | string | required | `"6a6f4aa4197940add57724a7707d069478df56b1"` | included | §4 |
| `tokenizer_name` | string | required | `"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"` | included | §4 |
| `tokenizer_revision` | string | required | `"6a6f4aa4197940add57724a7707d069478df56b1"` | included | §4 |
| `budget` | integer | required | `1024` | included | §4 |
| `runtime_source_artifact_path` | string | required | `"docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json"` | included | §7.3 |
| `runtime_source_artifact_sha256` | string (64-hex) | required | `f9eff6c1785df69406309157a6921a6fa7729dfb4fb5bfe43fa3401e0691443e` (`sha256_file` of the committed file, §7.3) | included | §7.3 |
| `attempted` | array | required | one entry per §12.5 outcome, in attempt order | included | new |
| `attempted_candidate_count` | integer | required | length of `attempted` | included | new |
| `first_passing_candidate_ordinal` | integer, nullable | nullable | `null` if no candidate qualified | included | new |
| `selected_unique_id` | string, nullable | nullable | `null` if no candidate qualified | included | standardized name (was `selected_row_id`/`selected_unique_id` inconsistently) |
| `selection_status` | string | required | e.g. `"selected"` / `"no_candidate_qualified"` | included | new |
| `qualification_stopped_reason` | string | required | why evaluation stopped (first pass, or all 8 exhausted) | included | new |
| `attempt_started_at_utc` | string (ISO 8601) | required | operational timestamp — permitted here (this artifact describes one real future execution, unlike the candidate manifest, §12.3) | included | new |
| `attempt_completed_at_utc` | string (ISO 8601) | required | same | included | new |
| `canonical_sha256` | string (64-hex) | required | this artifact's own self-hash (§12.1) | **excluded from its own computation** | §12.1 |

Unlike the candidate manifest, the qualification artifact **may** include
operational timestamps in its canonical, hashed payload — it describes one
real future execution, not a reproducible-from-inputs-alone static
manifest, so a timestamp here does not break determinism of the manifest
itself.

`runtime_source_artifact_sha256` above was computed twice during this
repair using the repository's actual `kvcot.utils.hashing.sha256_file`
against the committed `docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json`
— both calls returned the identical value recorded above. Never guessed,
never manually calculated.

The verifier must reject an artifact with a field name outside this table,
a missing required field, or an `attempted` array whose length disagrees
with `attempted_candidate_count`.

### 12.7 Selection provenance schema (repairs R3-AUDIT-20, R3-AUDIT-21, R3-AUDIT-25)

| Field | Type | Nullable/Required | Meaning | Hash inclusion | Source |
|---|---|---|---|---|---|
| `artifact_schema_version` | string | required | `"faithkv-b2a-r3-selection-provenance-v1"` | included | new (§12.9) |
| `qualification_artifact_path` | string | required | `"results/decisions/b2a_r3_qualification.json"` | included | mirrors `SelectionProvenance.qualification_artifact_path` |
| `qualification_artifact_canonical_sha256` | string | required | §12.6's `canonical_sha256` (renamed per §12.1) | included | same |
| `candidate_manifest_path` | string | required | `"configs/discovery/b2a_r3_candidate_manifest.json"` | included | mirrors `SelectionProvenance.candidate_manifest_path` |
| `candidate_manifest_canonical_sha256` | string | required | §12.3's `canonical_sha256` (renamed per §12.1) | included | same |
| `selected_manifest_path` | string | required | `"configs/discovery/b2a_one_example_manifest.json"` | included | §12.2 |
| `selected_manifest_sha256` | string (64-hex) | required | `B2AOneExampleManifest.manifest_hash()` — an EXTERNAL hash, never a field added to the manifest itself (repairs R3-AUDIT-21) | included | `src/kvcot/discovery/manifest.py` |
| `selected_manifest_hash_algorithm` | string | required | `"B2AOneExampleManifest.manifest_hash-v1"` | included | new (repairs R3-AUDIT-21) |
| `selected_ordinal` | integer | required | matches the qualification artifact's `first_passing_candidate_ordinal` | included | mirrors existing naming in `kvcot.discovery.b2a_r2_freeze` |
| `selected_unique_id` | string | required | matches the candidate row at `selected_ordinal` | included | standardized name |
| `selection_protocol_version` | string | required | `"faithkv-b2a-r3-selection-v1"` | included | new (§12.9) |
| `row_raw_sha256` | string (64-hex) | required | echoes the selected candidate row's `raw_row_sha256` (§12.4) | included | mirrors `SelectionProvenance.row_raw_sha256` |
| `prompt_token_ids_sha256` | string (64-hex) | required | mirrors same-named field | included | same |
| `tokenizer_revision_used_for_prompt_hash` | string | required | mirrors same-named field | included | same |
| `canonical_sha256` | string (64-hex) | required | this artifact's own self-hash | **excluded** | §12.1 |

**Frozen rule (repairs R3-AUDIT-21):** `selected_manifest_sha256` is never
named `selected_manifest_canonical_sha256` — the selected one-example
manifest does not contain, and this protocol does not add, a
`canonical_sha256` self-hash field, so §12.1's JSON self-hash rule never
applies to it. `selected_manifest_sha256` is computed exactly as
`B2AOneExampleManifest.manifest_hash()` already computes it today
(`sha256_json(self.model_dump(mode="json"))`, an external hash over the
manifest's own complete field set) — no change to `manifest.py`, no new
field added to `B2AOneExampleManifest`.

The verifier must reject a selection-provenance record with a field name
outside this table, or missing any required field above.

### 12.8 Authorization claim schema (repairs R3-AUDIT-20, R3-AUDIT-23, R3-AUDIT-24, R3-AUDIT-25)

See §14.4 for the full atomic-claim lifecycle this schema is written
under. This subsection is the one, complete, authoritative schema table
`kvcot.discovery` code must implement — no other section restates or
supersedes it:

| Field | Type | Nullable/Required | Meaning | Hash inclusion | Source |
|---|---|---|---|---|---|
| `artifact_schema_version` | string | required | `"faithkv-b2a-r3-authorization-claim-v1"` | included | new (§12.9) |
| `authorization_id` | string | required | matches the dated authorization document's `authorization_id`; must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$` | included | §14.3, §14.4 |
| `authorization_stage` | string | required | `"fullkv_qualification"` or `"b2a_r3_execution"` | included | §14.3 |
| `authorization_document_path` | string | required | e.g. `docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_<date>.md` | included | §12.2 |
| `authorization_document_sha256` | string (64-hex) | required | `sha256_file` of that Markdown document's committed bytes (repairs R3-AUDIT-23 — never `sha256_json`) | included | §12.1, §14.3 |
| `authorized_repository` | string | required | must equal `"asad073-ui/Faithkv"` (§21) | included | §14.4 |
| `authorized_branch` | string | required | the authorization document's named branch | included | §14.3 |
| `authorized_commit_sha` | string | required | commit the authorization document names | included | §14.3 |
| `observed_repository` | string | required | actual repository at claim time | included | §14.4 |
| `observed_branch` | string | required | actual branch at claim time | included | §14.4 |
| `observed_commit_sha` | string | required | actual `HEAD` at claim time | included | §14.4 |
| `required_ancestor_shas` | array of string | required | every ancestor commit verified per §14.5's `AttemptProvenancePolicy` | included | §14.5 |
| `required_rkv_sha` | string | required | the R-KV submodule SHA the authorization document requires | included | §14.5 |
| `observed_rkv_sha` | string | required | the R-KV submodule SHA actually observed at claim time | included | §14.5 |
| `candidate_manifest_canonical_sha256` | string | required | binds the claim to one exact candidate manifest | included | §12.3 |
| `qualification_artifact_canonical_sha256` | string, nullable | `null` for Stage B; required for Stage C | binds a Stage C claim to one exact qualification outcome | included | §12.6 |
| `selected_manifest_sha256` | string, nullable | `null` for Stage B; required for Stage C | binds a Stage C claim to one exact selected row (repairs R3-AUDIT-21 — never `selected_manifest_canonical_sha256`) | included | §12.7 |
| `selected_manifest_hash_algorithm` | string, nullable | `null` for Stage B; `"B2AOneExampleManifest.manifest_hash-v1"` for Stage C | names the algorithm the previous field used | included | §12.7 |
| `attempt_id` | string | required | this attempt's own identifier | included | mirrors B2A-R2's `attempt_id` convention |
| `global_claim_path` | string | required | `"results/decisions/b2a_r3_authorization_claims/<authorization_id>.json"` (repairs R3-AUDIT-24) | included | §14.4 |
| `attempt_directory_path` | string | required | the attempt directory this claim's execution proceeds into | included | §14.4 |
| `claimed_at_utc` | string (ISO 8601) | required | operational timestamp — permitted (describes one real claim event) | included | new |
| `canonical_sha256` | string (64-hex) | required | this claim's own self-hash | **excluded** | §12.1 |

**Frozen claim-verifier rejection list.** The verifier must reject a claim
that has: an unknown `authorization_stage`; a wrong `authorization_
document_sha256` (recomputed via `sha256_file` against the committed
document and compared); a wrong `authorized_repository`/`authorized_
branch` vs. what is observed; a wrong `observed_commit_sha` (does not
equal `authorized_commit_sha`); a missing required ancestor commit (any
entry in `required_ancestor_shas` that does not verify as an ancestor of
`observed_commit_sha`); a wrong R-KV SHA (`observed_rkv_sha !=
required_rkv_sha`); a wrong `candidate_manifest_canonical_sha256`; for
Stage C, a wrong `qualification_artifact_canonical_sha256` or a wrong
`selected_manifest_sha256`/`selected_manifest_hash_algorithm`; any field
name outside this table; any required field missing; an invalid
`canonical_sha256` (§12.1's five-step verification failing); an invalid
`authorization_id` (§14.4.1's pattern failing); or a `global_claim_path`/
`attempt_directory_path` that disagrees with the deterministic paths
§14.4.1/§14.4.2 derive.

### 12.9 Frozen per-artifact protocol-identity fields (repairs R3-AUDIT-25)

Four separate, independently-versioned identity concepts exist across the
new B2A-R3 artifacts. None of them share a single ambiguous
`protocol_version` field — each artifact names its own identity field(s)
explicitly, and a bare `protocol_version` never appears in any new B2A-R3
schema table (§12.3, §12.5, §12.6, §12.7, §12.8 above all already use the
separated names; this subsection is the one place that states all four
together for cross-reference).

**Candidate manifest:**

```text
artifact_schema_version         = "faithkv-b2a-r3-candidate-manifest-v1"
candidate_order_protocol_version = "faithkv-b2a-r3-row-order-v1"
```

**Qualification artifact:**

```text
artifact_schema_version   = "faithkv-b2a-r3-qualification-artifact-v1"
qualification_protocol_version = "faithkv-b2a-r3-qualification-v1"
runtime_predictor_version = "faithkv-b2a-r3-runtime-predictor-v1"
```

**Selection provenance:**

```text
artifact_schema_version    = "faithkv-b2a-r3-selection-provenance-v1"
selection_protocol_version = "faithkv-b2a-r3-selection-v1"
```

**Authorization claim:**

```text
artifact_schema_version = "faithkv-b2a-r3-authorization-claim-v1"
```

(one concept only — no second sub-version is needed for the claim
schema.)

Historical B2A-R1/B2A-R2 artifacts retain their historical field names and
meanings (e.g. `b2a_r2_candidates.CANDIDATE_MANIFEST_PROTOCOL_VERSION`'s
single `protocol_version` field) — this standardization applies only to
new B2A-R3 artifacts and never edits a historical schema.

## 13. Selected-row freeze contract

A future hash-verified freezer (Step 3) must:

- Accept only a row present in the committed candidate manifest
  (`configs/discovery/b2a_r3_candidate_manifest.json`, §12.2).
- Accept only the first passing candidate recorded in the qualification
  artifact (`results/decisions/b2a_r3_qualification.json`, §12.2).
- Verify the candidate manifest's `canonical_sha256`.
- Verify the qualification artifact's `canonical_sha256`.
- Verify the dataset revision.
- Verify the row ID and problem hash.
- Reject arbitrary row substitution.
- Reject selection of a later passing row when an earlier one passed.
- Reject manual artifact editing (any field mismatch against the
  regenerated hash is a hard failure).
- Read the selected candidate's complete pinned row from the candidate
  manifest's own embedded `row` field (§12.4) — **never refetch the
  dataset during freezing** (R3-AUDIT-22's frozen embed-not-refetch rule).
- Produce the one-row selected manifest
  (`configs/discovery/b2a_one_example_manifest.json`, §12.2) with full
  provenance hashes, and the companion selection-provenance artifact
  (`results/decisions/b2a_r3_selection_provenance.json`, §12.7), including
  that artifact's `selected_manifest_sha256` field — computed as
  `B2AOneExampleManifest.manifest_hash()` over the just-written selected
  manifest, never named `selected_manifest_canonical_sha256` (§12.1,
  R3-AUDIT-21: the selected manifest has no `canonical_sha256` field of
  its own).

This mirrors `kvcot.discovery.b2a_r2_freeze.freeze_qualified_row`'s
existing behavior exactly (`docs/B2A_R2_RESULT_2026-07-22.md` §3).

No selected manifest may exist before FullKV qualification (Stage B, §14)
has run, and no FullKV qualification may run before Stage B's own
authorization claim (§14.4) has been atomically written.

## 14. Authorization and attempt-consumption boundaries

### 14.1 Stage A — Authorized only after independent audit of this document

```text
CPU implementation, CPU tests, and dry-run planning only
```

Repairs R3-AUDIT-17: the first-frozen protocol's Stage A list was
ambiguous about whether it covered generating a *production* qualification
artifact. It does not. Frozen precisely (§Q of the repair ledger):

Stage A **allows**, once independently audited:

- Candidate-manifest generation (a real, committed
  `configs/discovery/b2a_r3_candidate_manifest.json`, built from the real
  pinned MATH-500 dataset content — this is deterministic, outcome-blind,
  CPU-only, and touches no model).
- The pure runtime predictor (§7) as code, exercised only against
  synthetic/injected inputs in tests, or against the real candidate
  manifest's token counts if those were themselves obtained without any
  model inference (they are not — see below).
- The pure qualification evaluator (§10) as code, exercised only against
  synthetic/injected `CandidateQualificationOutcome`-shaped fixtures in
  tests.
- Synthetic qualification artifacts in tests (never a real one).
- A qualification-artifact parser/verifier (schema/hash validation only).
- The selected-row freezer implementation, exercised only against
  synthetic/injected fixtures in tests.
- The authorization-claim implementation (schema, atomic-write logic),
  exercised only against synthetic/injected fixtures in tests.
- Dry-run planning (`kvcot plan-discovery --dry-run`-style commands).
- CPU tests for all of the above.

Stage A **forbids**, even after independent audit:

- A *production* qualification artifact based on real FullKV output.
- Real model output of any kind.
- FullKV qualification (running the FullKV worker against real weights).
- CUDA initialization.
- R-KV.
- Pair evaluation.
- A real selected-row manifest derived from an actual qualification run
  (only a synthetic one, in a test, is permitted).

The candidate manifest is the one artifact that IS real and committed
under Stage A, because it requires no model, no CUDA, and no GPU — only
the pinned dataset's already-public content and pure Python hashing. The
qualification *artifact*, by contrast, describes the outcome of running
the FullKV worker against real weights, which Stage A never authorizes.

### 14.2 Stage B — Future separate authorization

```text
One bounded FullKV-only qualification
```

Prerequisites: this protocol independently audited and the audit
recorded; CPU implementation complete; green CPU CI; candidate manifest
committed and hash-verified; the runtime predictor independently checked;
a phase-wide qualification wall-time limit frozen by that future
authorization (§11); a dated authorization document committed (§14.3).

Stage B does not authorize B2A-R3 execution.

### 14.3 Stage C — Future separate authorization

```text
Exactly one B2A-R3 execution
```

Prerequisites: qualification complete; the first passing row frozen; all
qualification artifacts archived; selected-row provenance audited;
projected runtime at most 3.60 GPU-hours; green CPU CI; a dated execution
authorization document committed.

Stage C does not authorize B2B.

- No automatic rerun.
- No hidden retry.
- No replacement row after a failure.
- No B2A-R4 authorization is created or implied by this document.
- No attempt reset after failure — a consumed attempt stays consumed,
  exactly as B2A-R1 and B2A-R2 remain consumed (`CLAUDE.md` §1c, §1d).

### 14.4 Authorization document and claim lifecycle (repairs R3-AUDIT-12, R3-AUDIT-24)

The first-frozen protocol required "an attempt-consumption guard" without
defining its lifecycle; R3-AUDIT-12 froze a two-part model (a tracked
authorization document plus a claim). A second independent re-audit found
that model's claim step ("search the attempt root for any existing valid
claim, then write a new one") was a scan-then-write pattern — inherently
racy, since two concurrent processes could each complete the scan before
either writes, both then proceeding toward CUDA/model activity under the
same authorization. This subsection replaces the claim mechanism entirely
with a **globally exclusive, atomically created claim at one
deterministic path**, never a scan. **Not implemented by this task** —
frozen now, for Step 3 to implement.

**Tracked authorization document** (one per Stage B or Stage C grant,
committed at the exact path pattern in §12.2, immutable after commit):

```text
authorization_id
authorization_stage            (one of: "fullkv_qualification", "b2a_r3_execution")
authorized_repository
authorized_branch
authorized_commit_sha
authorization_document_sha256  (sha256_file of this document's own committed
                                 bytes -- see §12.1's Markdown/JSON scope
                                 boundary; the document cannot hash itself
                                 before it is committed, so this field is
                                 recorded by the claim, §12.8, not by the
                                 document)
candidate_manifest_canonical_sha256
qualification_artifact_canonical_sha256   (Stage C only)
selected_manifest_sha256                  (Stage C only; repairs R3-AUDIT-21
                                            -- never "..._canonical_sha256")
selected_manifest_hash_algorithm          (Stage C only:
                                            "B2AOneExampleManifest.manifest_hash-v1")
maximum_candidates                        (Stage B only)
phase_wall_time_limit_seconds             (Stage B only)
created_at_utc
```

**14.4.1 Frozen global claim path (repairs R3-AUDIT-24)**

```text
results/decisions/b2a_r3_authorization_claims/<authorization_id>.json
```

One deterministic path per `authorization_id` — never inside an attempt
directory, never scanned for, always computed directly from the
authorization document's own `authorization_id` field.

**Frozen authorization-ID validation** (rejected values are a hard
failure, never sanitized/coerced):

```text
pattern: ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$
```

Rejects: any `/` or `\`, any `..` sequence, whitespace, control
characters, an empty string, and any ID longer than 128 characters.

**14.4.2 Frozen claim operation — creation IS consumption**

Before any CUDA initialization, model loading, tokenizer loading for
execution, or GPU worker launch, in this exact order:

1. Complete every non-GPU pre-claim verification: authorization document
   path and byte hash (`authorization_document_sha256`, §12.1's
   `sha256_file` rule); authorization stage; repository; branch; exact
   authorized commit; required ancestor commits (§14.5); the R-KV
   submodule SHA; the candidate-manifest hash; the qualification-artifact
   hash (Stage C only); the selected-manifest hash (Stage C only, §12.7);
   a clean worktree.
2. Compute the deterministic global claim path from `authorization_id`
   (§14.4.1) — a pure function, never a filesystem scan.
3. Create the claim path using exclusive-create semantics:
   `os.open(claim_path, O_WRONLY | O_CREAT | O_EXCL, mode)` (or an
   equivalent operation whose success is atomic across concurrent
   processes on the host filesystem). **The success of this single
   operation IS the consumption event** — not a subsequent successful
   write, not a subsequent successful GPU run.
4. Only after step 3 succeeds: write the complete canonical JSON claim
   payload (§12.8 schema) to the exclusively created file descriptor,
   flush it, and `fsync` it.
5. Only after the global claim (steps 3-4) succeeds: create the new
   immutable attempt directory (§12.2 path pattern).
6. Store, inside the attempt directory, either a byte-identical copy of
   the completed claim or an immutable reference recording the global
   claim's own path and hash — never a second, independently-authoritative
   claim.
7. Only after both the global claim and the attempt-directory record exist
   may execution proceed toward tokenizer/model/CUDA activity.

**14.4.3 Frozen crash semantics**

```text
The existence of ANY filesystem entry at the deterministic global claim
path means the authorization is permanently consumed.
```

This includes a complete valid claim, a partially written claim, an empty
claim, a corrupt claim, a claim left by a crash, a claim left by a killed
process, and a claim left by a failed preflight step that ran after step 3
above already succeeded. A malformed claim means **consumed and
corrupted**, never **available for retry**. No deletion, repair, or manual
replacement of the claim file restores authorization. A later attempt
requires a new authorization document carrying a new `authorization_id`
— never reuse of an existing ID's claim path.

**14.4.4 Mandatory concurrency test**

A future CPU test must prove: two concurrent claim attempts for the same
`authorization_id` produce exactly one successful exclusive creation and
one refusal (e.g. by asserting `FileExistsError`/`OSError` on the loser of
a race over the same path, or by direct simulation of the `O_EXCL`
semantics). No scan-then-write design of any kind is permitted to pass
this test.

**14.4.5 Frozen dry-run behavior**

CPU-only planning, verification, synthetic tests, and dry-run commands
must not create the global claim directory, must not create a claim file
at any path, must not create an attempt directory, must report
`authorization_claim_created = false`, and must report
`authorization_consumed = false`.

**14.4.6 Worktree/provenance behavior**

The future provenance verifier must: require a clean worktree BEFORE
claim creation (never after); recognize only the exact active global
claim path and the exact active attempt-directory root as expected
post-claim artifacts (never a broader "anything under `results/decisions/`
is fine" rule); reject any other dirty, staged, modified, or untracked
path exactly as today; never place a claim anywhere under `configs/`; and
never weaken historical B2A-R1/R2 clean-worktree verification in the
course of adding this.

### 14.5 Versioned provenance policy (repairs R3-AUDIT-13)

`src/kvcot/discovery/attempt_verification.py` contains
`REQUIRED_BRANCH = "research/b1b-r4-final-b2a-closure"`, a historical,
module-level constant tied to the branch B2A-R1/B2A-R2 executed on. **This
protocol does not instruct Step 3 to replace that historical global** —
doing so would silently reinterpret B2A-R1/B2A-R2's own historical
verification. Instead, Step 3 must introduce a separately-constructed
policy object:

```text
AttemptProvenancePolicy
    provenance_policy_version   ("faithkv-b2a-r3-provenance-policy-v1" —
                                 repairs R3-AUDIT-25: never a bare
                                 "protocol_version"; this is a fifth,
                                 distinct identity concept from the four
                                 named in §12.9, versioning this in-memory
                                 policy object itself, not a persisted
                                 artifact)
    required_repository
    required_branch
    required_commit_sha
    required_ancestor_shas
    required_rkv_sha
    authorization_id
    authorization_document_sha256
```

Rules:

1. Historical B2A-R1/R2 verification retains its historical, unmodified
   provenance policy (the existing `REQUIRED_BRANCH` constant and its
   surrounding checks in `attempt_verification.py`).
2. B2A-R3 uses a separately constructed `AttemptProvenancePolicy` instance,
   populated from its own future dated authorization document (§14.4) —
   never from `REQUIRED_BRANCH`.
3. The verifier must not depend on one single global "current branch"
   constant for both historical and B2A-R3 verification.
4. The B2A-R3 observed `HEAD` must equal the authorized commit exactly
   (`authorized_commit_sha`).
5. The observed branch and repository must equal the authorization
   document's `authorized_branch`/`authorized_repository`.
6. All `required_ancestor_shas` must verify as ancestors.
7. The R-KV submodule must match `required_rkv_sha`.
8. The worktree must be clean, excluding only the active immutable
   attempt artifact root, under the existing provenance rules (matching
   B2A-R2's own `dirty`/`staged_paths`/`unstaged_paths` accounting in
   `docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json`).
9. No Step 3 code may weaken historical (B2A-R1/R2) verification in the
   course of adding B2A-R3 support.

This protocol does **not** freeze the future Stage B/C execution commit
now — the future authorization document (§14.4) provides that exact
commit at authorization time, not this protocol.

## 15. CPU implementation requirements for Step 3

Documented, not implemented, by this task:

1. B2A-R3 candidate-manifest schema (§12.3-12.4).
2. Deterministic candidate generator, including the level-interleave
   construction (§8, §9).
3. Prior-row exclusion logic, verified against `EXCLUSION_SET_SHA256`
   (§8.1).
4. Runtime predictor, using §7.3's exact constants (§7).
5. Qualification evaluator, including §10.1's exact thinking-span/
   trace-completeness predicates (§10).
6. First-qualified-row selector (§10.4).
7. Qualification artifact writer (§12.5-12.6).
8. Hash-verified selected-row freezer (§13).
9. Authorization-claim implementation and attempt-consumption guard
   (§14.4).
10. A separately-constructed `AttemptProvenancePolicy` for B2A-R3, without
    modifying historical verification (§14.5).
11. Dry-run planning CLI, including the wall-time envelope display (§11).
12. Pair-record persistence requirements inherited unchanged from B2A-R2
    (`rkv/pair_records.json`, `rkv/scientific_summary.json`,
    `verify_pair_record_artifacts`, `verify_pair_record_population`).
13. Final/completion/CLI outcome consistency, inherited unchanged from the
    B2A-R2 forensic repair (one authoritative `overall_passed`, computed
    once, read everywhere).

Required later CPU tests: deterministic ordering (per-level, then
interleave); canonicalization stability; prior-row exclusion against
`EXCLUSION_SET_SHA256`; level-mixture enforcement (8/8, then 4/4 in the
first 8); manifest `canonical_sha256` stability and exclusion of its own
field from its hash; predictor arithmetic exactness against §7.3's exact
constants; the frozen 2775-passes/2776-fails integer boundary (§7.5);
safety-multiplier enforcement; rejection above 3.60 hours; missing-input
rejection; thinking-span/trace-completeness predicate exactness (§10.1);
event-count calculation; future-horizon calculation; first-pass stopping;
eight-candidate limit; later-candidate substitution refusal; no R-KV
imports during qualification; no pair outcomes during qualification; no
CUDA from dry-run; selected-manifest hash verification; authorization-claim
atomicity and consumption-on-claim (never on success); `AttemptProvenancePolicy`
correctness without modifying `REQUIRED_BRANCH`; the frozen
`FINAL_MANDATORY_GATE_CONDITIONS` tuple reproduced exactly (§16); mandatory
pair persistence; CLI/final/completion consistency.

## 16. Mechanical B2A-R3 acceptance gates (repairs R3-AUDIT-14, R3-AUDIT-15)

### 16.1 Frozen final-gate tuple

The first-frozen protocol said B2A-R3 "inherits whatever
`FINAL_MANDATORY_GATE_CONDITIONS` is at Step 3 implementation time" — a
dynamic, unfrozen inheritance rule, not an actual frozen protocol
(R3-AUDIT-14). Repaired: the exact ordered tuple, copied verbatim from the
starting commit of this repair, is now part of the frozen protocol.

```text
Source path:   src/kvcot/discovery/final_contract.py
Source commit: 93b6ba869eb5e555684704a6d1f2250f16884768
Blob SHA-256:  0b4063023b4da5cf33b5a1e419fc9577db363b6b
Exact count:   30 conditions
```

```text
FINAL_MANDATORY_GATE_CONDITIONS = (
    "git_clean_verified",
    "rkv_submodule_match",
    "single_rtx3090_verified",
    "local_model_snapshot_verified",
    "local_tokenizer_snapshot_verified",
    "dataset_row_identity_verified",
    "prompt_identity_verified",
    "fullkv_generation_matches_expected",
    "rkv_generation_matches_expected",
    "workers_generation_match",
    "actual_batch_size_verified",
    "complete_token_trace_match",
    "complete_call_trace_match",
    "complete_compaction_trace_match",
    "capture_gather_parity",
    "absolute_position_parity",
    "selected_event_ids_exact",
    "unique_real_pair_count_exact",
    "events_with_four_unique_pairs_exact",
    "no_duplicate_pair_identity",
    "authorized_no_op_identity_exact",
    "positive_semantic_swap_parity",
    "no_op_exact_parity",
    "no_offload_and_placement_verified",
    "all_required_timings_present",
    "all_required_memory_phases_present",
    "runtime_within_limit",
    "peak_vram_within_limit",
    "worker_envelopes_verified",
    "attempt_artifacts_verified",
)
```

Step 3 must preserve this exact tuple (order and membership). Any
addition, removal, renaming, or semantic change to this tuple requires a
dated protocol amendment and its own independent audit — no silent
inheritance from a later source-code version is permitted, even if the
source has moved on by the time Step 3 is implemented.

### 16.2 Frozen legacy-gate tuple

The legacy gate (`evaluate_b2a_gate`, distinct from the final gate above)
remains independently mandatory and is likewise frozen exactly:

```text
Source path:   src/kvcot/discovery/b2a_contract.py
Source commit: 93b6ba869eb5e555684704a6d1f2250f16884768
Blob SHA-256:  4d63a83a81855cfcbc7defc74ef332946870f112
Exact count:   29 conditions
```

```text
MANDATORY_GATE_CONDITIONS = (
    "token_identical_replay",
    "prefill_decode_boundary_parity",
    "compaction_position_equality",
    "capture_gather_parity",
    "absolute_position_parity",
    "no_op_numerical_parity",
    "semantic_swap_parity",
    "dataset_revision_match",
    "dataset_row_identity_match",
    "manifest_hash_match",
    "prompt_token_hash_match",
    "model_revision_match",
    "tokenizer_revision_match",
    "generation_config_hash_match",
    "rkv_config_hash_match",
    "no_offload_verified",
    "batch_size_verified",
    "runtime_within_limit",
    "peak_vram_within_limit",
    "one_example_only",
    "meaningful_compression_observed",
    "sufficient_eligible_events",
    "selected_event_count_exact",
    "real_pair_count_exact",
    "no_op_count_exact",
    "all_required_pair_evaluations_completed",
    "unique_real_pair_count_exact",
    "events_with_four_unique_pairs_exact",
    "no_duplicate_pair_identity",
)
```

Note for the record: B2A-R2's own historical documents reported "27 of 28"
conditions passing — that count reflects the tuple's membership as it
stood on 2026-07-22 at B2A-R2's execution time, before subsequent CPU-only
repairs (e.g. the F7 device/offload repair) added conditions to both
tuples. The counts above (30 and 29) are the exact, current counts at this
repair's starting commit, not a restatement of B2A-R2's historical count —
the two are expected to differ, and this document does not paper over that
difference.

Any addition, removal, renaming, or semantic change to either tuple
requires a dated protocol amendment and independent audit, exactly as
§16.1 states for the final-gate tuple. The source code remains the
runtime source of truth; the code must reproduce these frozen protocol
tuples exactly, not the other way around.

### 16.3 Acceptance gate table

| Category | Required result |
|---|---|
| FullKV answer | Correct |
| R-KV answer | Correct |
| Answer verifier | Valid |
| Generation cap | Not hit |
| Thinking span | Valid (§10.1's exact `thinking_span_valid` predicate) |
| Compression | `derive_meaningful_compression_observed` passes (§7.2) |
| Measured eligible events | At least 3 |
| Selected events | Exactly 3 |
| Real pair records | Exactly 12 |
| No-op records | Exactly 1 |
| Baseline NLL arrays | Exactly 48 values per pair (`SwapPairRecord.baseline_per_token_nll`, `min_length=max_length=SCORED_HORIZON`) |
| Swapped NLL arrays | Exactly 48 values per pair (same schema, `swapped_per_token_nll`) |
| Pair records | Durably persisted (`rkv/pair_records.json`) |
| Pair artifact | Present and `canonical_sha256`-verified |
| Scientific summary | Present and `canonical_sha256`-verified |
| Replay | Token-identical |
| No-op gain | See §16.4 (repairs R3-AUDIT-15's tolerance wording) |
| Peak VRAM | At most 22 GiB |
| Runtime | At most 4.00 GPU-hours |
| `completion.json` | Internally consistent |
| `final.json` | Internally consistent |
| CLI exit code | 0 only when `overall_passed` is true |
| Archive | Externally verified before instance destruction |
| `FINAL_MANDATORY_GATE_CONDITIONS` | All 30 conditions in §16.1's exact tuple pass |
| `MANDATORY_GATE_CONDITIONS` (legacy) | All 29 conditions in §16.2's exact tuple pass |

Qualifying at 3.60 GPU-hours (§7, §10.3) does not itself satisfy the final
4.00-hour gate — the actual attempt's own measured runtime evidence must
still separately pass `runtime_within_limit`.

### 16.4 Corrected no-op tolerance wording (repairs R3-AUDIT-15)

The first-frozen protocol mixed "exact equality" with "`_close(...)`,
tolerance `1e-9`" in the same row. Inspected directly
(`src/kvcot/discovery/schemas.py:250-274`, `_noop_invariants`), the
implemented rule is actually **two different checks, not one**:

- **Array-level check (literal exact equality):**
  `baseline_per_token_nll != swapped_per_token_nll` is a Python list `!=`
  comparison — element-by-element EXACT equality, deliberately never
  `math.isclose`/`allclose` (the source's own comment: "a genuine no-op
  swap... must reproduce bit-for-bit identical logits and therefore
  identical NLL, not merely 'close' NLL").
- **Derived `swap_gain` check (numerical tolerance):**
  `swap_gain == 0.0` is checked via `_close(self.swap_gain, 0.0)`, where
  `_close` uses the module-level `_FLOAT_DIFF_TOLERANCE = 1e-9`
  (`src/kvcot/discovery/schemas.py:51,54`) — an absolute-tolerance
  numerical-parity check, not literal `==`.

Frozen wording for every reference to this gate in this protocol and in
Step 3's own documentation:

```text
No-op gain: baseline_per_token_nll and swapped_per_token_nll must be
element-by-element EXACTLY equal (Python list ==, zero tolerance). The
derived swap_gain must equal 0.0 within the frozen absolute tolerance
1e-9, as implemented by the repository's _close predicate
(src/kvcot/discovery/schemas.py). These are two distinct checks on two
distinct quantities -- never described as a single "exact equality"
check, and the 1e-9 tolerance is not changed by this protocol.
```

## 17. Scientific mechanism gate

Kept separate from mechanical acceptance (§16).

Frozen thresholds, verified from the prior protocol:

- **Positive-gain threshold: `swap_gain > 0.01` nats.** Source:
  `_MEANINGFUL_GAIN_THRESHOLD = 0.01` in
  `src/kvcot/discovery/scientific_summary.py:24`, computed into
  `gain_above_0_01_count` (line 94). Inherited unchanged from the original
  B0.5-era discovery-pattern rule (`docs/b0_5_decision.json`
  `b0_5_r2_1_repaired_gate_10`).
- **Correlation ceiling: `|Spearman rho| < 0.30`.** Source: the same
  `b0_5_r2_1_repaired_gate_10` decision and
  `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §16; computed for a single B2A
  attempt as `spearman_score_margin_vs_swap_gain`
  (`src/kvcot/discovery/scientific_summary.py:104-107,120`).
- **No-op gain:** see §16.4's corrected wording — element-by-element exact
  equality on the NLL arrays, and the derived `swap_gain == 0.0` within the
  frozen absolute tolerance `1e-9`. Not changed by this repair.

**Necessary single-example scale adaptation, stated explicitly, not
silently copied** (unchanged from the first-frozen protocol): the original
`b0_5_r2_1_repaired_gate_10` rule was designed for a 12-**example**
B2B-scale population. B2A (and B2A-R3) runs exactly **one** example with
12 real pairs. The measurable, non-fabricated, single-example analogue
this protocol freezes for B2A-R3 is:

```text
At least 4 of the 12 real pairs in the single B2A-R3 attempt:
swap_gain > 0.01 nats
(gain_above_0_01_count >= 4, out of real_pair_count == 12)

The single computed value:
|spearman_score_margin_vs_swap_gain| < 0.30
(there is one example, so there is one correlation value, not a median
across examples)

No-op gain: per §16.4's corrected wording
```

This is a new B2A-R3 protocol decision for the unit of measurement (pair,
not example), reusing the exact, unchanged numeric thresholds (`0.01`,
`0.30`) the prior protocol already froze. It is, in the current codebase,
descriptive rather than gating: `scientific_summary.py`'s fields are
computed and persisted but are not wired into either frozen gate tuple in
§16 as a pass/fail condition.

Interpretation:

- The mechanical gate (§16) answers whether the harness result is valid.
- This scientific gate answers whether the causal-mismatch signal
  survives at all.
- A mechanical pass does not imply scientific success.
- Scientific failure does not invalidate mechanically correct artifacts.

## 18. Kill gates and decision tree

**No candidate qualifies:**

```text
STOP — DO NOT RUN B2A-R3.
Do not weaken correctness, event, compression, or runtime thresholds.
Any new candidate-pool design requires a protocol amendment and independent audit.
```

**Mechanical failure:**

```text
No scientific conclusion.
No B2B.
No automatic rerun.
Repair the failure-specific path on CPU.
```

**Gains are at floor:**

```text
Do not run B2B.
The intervention is not measurable enough to support method design.
Revise the intervention or retarget.
```

**R-KV score correlates strongly with causal gain:**

```text
Do not claim R-KV causal mismatch.
The proposed gap is substantially weakened.
```

**Runtime exceeds four hours:**

```text
Do not authorize B2A-R4 automatically.
Audit whether the predictor, selected trace, or pair cost failed.
A new attempt requires a new failure-specific protocol.
```

**Pair persistence fails:**

```text
No scientific conclusion.
Repair persistence on CPU before considering any new attempt.
```

**Mechanical pass and causal signal survives:**

```text
B2B may be proposed, but remains separately blocked pending authorization.
```

## 19. Governance and prohibited work

Until a separate, dated GPU authorization exists (Stage B and/or Stage C,
§14), the following remain prohibited:

- GPU rental.
- CUDA initialization.
- Model download.
- Model loading.
- FullKV candidate qualification.
- R-KV inference.
- Pair evaluation.
- B2A-R3 execution.
- B2B execution.
- FaithKV method implementation.
- Threshold modification made in order to obtain a pass.

An authorization document, once claimed (§14.4), is consumed regardless of
outcome — no failure mode of any kind restores it, and no automatic or
unauthorized further attempt is permitted.

## 20. Protocol freeze checklist

- [x] Research question frozen (§3).
- [x] Scientific settings verified against repository sources (§4).
- [x] Outcome-blind allowed signals frozen (§6).
- [x] Forbidden signals frozen (§6).
- [x] Pair-timing statistic verified, exact value frozen (§7.1, §7.3).
- [x] Meaningful-compression definition verified (§7.2).
- [x] Runtime formula frozen with exact constants and multiplicities
      verified from source (§7.3, §7.4).
- [x] Continuous token ceiling and integer test boundary frozen (§7.5).
- [x] Safety multiplier frozen and justified as new (§7.3).
- [x] Deterministic candidate canonicalization frozen, reusing B2A-R2's
      construction (§9).
- [x] Candidate level mixture and exact interleave algorithm frozen (§8.2).
- [x] Prior-row exclusion set frozen with a verifiable hash (§8.1).
- [x] First-pass rule frozen (§10.4).
- [x] Eight-candidate cap frozen (§8.2, §10.4).
- [x] Thinking-span/trace-completeness predicates frozen (§10.1).
- [x] Canonical hashing rule frozen, self-referential ambiguity resolved
      (§12.1).
- [x] Exact artifact paths frozen, no "recommended" language remains
      (§12.2).
- [x] Qualification artifact schema frozen with standardized field names
      (§12.3-12.8).
- [x] Selected-row freezer contract frozen (§13).
- [x] Qualification wall-time contradiction resolved; envelope frozen,
      phase-wide limit deferred to Stage B without contradiction (§11).
- [x] Authorization-claim lifecycle frozen (§14.4).
- [x] Provenance-policy contract frozen without touching the historical
      `REQUIRED_BRANCH` constant (§14.5).
- [x] Both frozen final/legacy gate tuples copied verbatim with source
      commit and blob hash (§16.1, §16.2).
- [x] No-op tolerance wording corrected to distinguish exact-equality
      from numerical-tolerance checks (§16.4).
- [x] Scientific gates frozen, with the single-example scale adaptation
      stated explicitly (§17).
- [x] Kill gates frozen (§18).
- [x] Repository identity resolved, not left as an open ambiguity (§21).
- [x] GPU remains prohibited pending separate authorization (§14, §19).
- [x] §22 (this repair's disposition) exists exactly once.
- [x] Exact 27-condition qualification tuple frozen, with full condition
      definitions and rejection rules (§10.5).
- [x] Candidate manifest, per-candidate outcome, and qualification
      artifact schemas completed with every field elsewhere required
      (§12.3, §12.5, §12.6).
- [x] Selected-manifest hash semantics resolved as an external
      `manifest_hash()` call, never a new self-hash field (§12.1, §12.7).
- [x] Candidate-row schema frozen with embedded `row`, verification
      formulas, and an explicit embed-not-refetch rule (§12.4, §13).
- [x] Markdown authorization-document hashing (`sha256_file`) frozen
      separately from the JSON `canonical_sha256`/`sha256_json` rule
      (§12.1, §12.8).
- [x] Authorization-claim consumption replaced with a globally exclusive,
      atomically created claim at one deterministic path (§14.4).
- [x] Four separate per-artifact protocol-identity fields frozen; no new
      B2A-R3 schema uses a bare `protocol_version` (§12.9).

## 21. Repository identity (repairs R3-AUDIT-18)

```text
repository = asad073-ui/Faithkv
```

Verified directly against the locally configured remote (`git remote -v`
→ `https://github.com/asad073-ui/Faithkv.git`) at this repair's starting
commit. This is now a resolved fact, not an open ambiguity.

Historical note: an earlier task's wording referred to the repository as
`attaulasad/Faithkv`. That was a documentation mismatch in that task's own
prompt text (an operator/org-name confusion), not a different repository
or a real identity conflict — every git-level fact (branch, HEAD, ancestor
commits, submodule SHA) matched exactly regardless of it. This is
mentioned here only as a resolved historical note, never as an active
ambiguity requiring further action.

## 22. Independent protocol audit and repair disposition

This section repairs R3-AUDIT-01 (the first-frozen protocol referenced
"§22" in §1 without any such section existing). It now records TWO repair
rounds: Step 2A (R3-AUDIT-01 through R3-AUDIT-18, the first independent
audit) and Step 2B (R3-AUDIT-19 through R3-AUDIT-25, a second independent
re-audit of the Step 2A repair commit). All 25 findings are repaired.

**Step 2A audit findings and repairs, summarized** (full detail:
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`):

| Finding | Issue | Repaired in |
|---|---|---|
| R3-AUDIT-01 | Missing §22 despite being referenced | This section |
| R3-AUDIT-02 | Contradictory authorization status (bare "AUTHORIZED" vs. audit-gated prose) | §1 |
| R3-AUDIT-03 | Rounded runtime constants (`10.25`, `~19.0`) insufficient for exact CPU tests | §7.1, §7.3, §7.5 |
| R3-AUDIT-04 | Mixed-level candidate construction left the 8-and-8/interleave order unfrozen | §8.2 |
| R3-AUDIT-05 | Exclusion sources described but not frozen as a directly implementable set | §8.1 |
| R3-AUDIT-06 | "Thinking span parses correctly" left undefined | §10.1 |
| R3-AUDIT-07 | "Trace is complete" left undefined | §10.1 |
| R3-AUDIT-08 | Self-referential hash field naming/computation ambiguity | §12.1 |
| R3-AUDIT-09 | Candidate manifest schema included a nondeterministic `created_at` | §12.3 |
| R3-AUDIT-10 | Every artifact path was only "recommended" | §12.2 |
| R3-AUDIT-11 | Wall-time section both said `VERIFY BEFORE FREEZE` and demanded a formula that didn't exist | §11 |
| R3-AUDIT-12 | Attempt-consumption guard required with no defined lifecycle | §14.4 |
| R3-AUDIT-13 | Historical `REQUIRED_BRANCH` global conflicts with a future B2A-R3 commit | §14.5 |
| R3-AUDIT-14 | Final-gate tuple dynamically inherited "whatever exists at Step 3 time" | §16.1 |
| R3-AUDIT-15 | No-op tolerance wording mixed "exact equality" with a numeric tolerance | §16.4 |
| R3-AUDIT-16 | Inconsistent field names (`candidate_index`/`candidate_ordinal`, `manifest_hash`/`canonical_sha256`/`artifact_hash`, `selected_row_id`/`selected_unique_id`) | §12.3-12.8 |
| R3-AUDIT-17 | CPU scope ambiguous about whether a real qualification artifact was authorized | §14.1 |
| R3-AUDIT-18 | Repository identity left as an unresolved discrepancy | §21 |

**Step 2B audit findings and repairs, summarized** (full detail:
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md`, "Step 2B" section):

| Finding | Issue | Repaired in |
|---|---|---|
| R3-AUDIT-19 | Qualification-condition tuple not frozen (Step 3 could invent names/order) | §10.5 |
| R3-AUDIT-20 | Artifact provenance schemas (candidate manifest, outcome, qualification artifact) incomplete | §12.3, §12.5, §12.6 |
| R3-AUDIT-21 | Selected-manifest hash semantics unresolved ("if that file gains one") | §12.1, §12.7, §13 |
| R3-AUDIT-22 | Candidate rows insufficient for deterministic freezing (embed-vs-refetch left open) | §12.4, §13 |
| R3-AUDIT-23 | Markdown authorization hash conflicted with the JSON self-hash rule | §12.1, §12.8, §14.4 |
| R3-AUDIT-24 | Authorization-claim consumption was scan-then-write, not globally atomic | §14.4 |
| R3-AUDIT-25 | Protocol-identity fields conflated under one ambiguous `protocol_version` | §12.9, §12.3, §12.6, §12.7, §12.8, §14.5 |

```text
ALL 25 FINDINGS (STEP 2A: 18, STEP 2B: 7) REPAIRED IN THIS COMMIT.

The repairing author does not self-certify this protocol.

STEP 3 CPU IMPLEMENTATION REMAINS BLOCKED UNTIL A SEPARATE
INDEPENDENT RE-AUDIT VERIFIES THIS STEP 2B COMMIT.

STAGE B FULLKV QUALIFICATION, STAGE C B2A-R3 EXECUTION, ALL GPU/CUDA
ACTIVITY, B2B, AND FAITHKV METHOD IMPLEMENTATION REMAIN PROHIBITED.
```
