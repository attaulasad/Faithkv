# B2A-R3 Runtime-Qualified Calibration Protocol (dated 2026-07-22)

## 1. Protocol identity and status

```text
Protocol: B2A-R3 Runtime-Qualified Calibration
Date frozen: 2026-07-22
Branch: research/b2a-r3-runtime-qualified-calibration
Parent closure commit: 0fa42a7edb88e766b5665547af15a5b52e823066
```

```text
B2A-R3 STATUS:
PROTOCOL FROZEN — CPU IMPLEMENTATION AUTHORIZED — GPU EXECUTION PROHIBITED
```

This status becomes valid only after, in order: protocol review (this
document's own internal self-audit, §22 below); documentation validation
(the diff checks recorded when this document was committed); commit;
push; and a genuinely independent audit of this document (a separate
review pass, not authored by whoever froze this document). **It does not
authorize GPU activity of any kind.** CPU-only implementation and CPU-only
tests become authorized only after that independent audit passes (§14,
Stage A).

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
| Qualification runtime target | 3.60 GPU-hours | **New B2A-R3 protocol decision** (§7) — not a pre-existing repository value; see justification below |
| Final runtime hard gate | 4.00 GPU-hours | `CLAUDE.md` §4c, "Runtime limit" row; `src/kvcot/discovery/execution_measurement.py` docstring; `docs/B2A_R2_RESULT_2026-07-22.md` §4 |

No intended value above conflicts with the frozen B2A-R2 protocol or
`CLAUDE.md`. The one row without a pre-existing source (qualification
runtime target, 3.60 GPU-hours) is explicitly marked as new, not
presented as inherited.

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
- Thinking-span parse validity.
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

### Ambiguity resolution: pair-time reference (mandatory, resolved)

`10.25` seconds is the **maximum** of 12 observed real-pair durations from
B2A-R2's own execution, not the mean, median, or a rounded representative
value.

- **Source artifact:** `docs/B2A_R2_RESULT_2026-07-22.md` §4, "Timing and
  memory" table: `Real-pair durations | 9.14s - 10.25s (12 values)` and
  `Conservative per-real-pair seconds | 10.25 (max of the 12)`.
- **Field name:** `conservative_real_pair_seconds` (dataclass field,
  `RuntimeProjection`) / `conservative_pair` (local variable),
  `src/kvcot/discovery/execution_measurement.py:357,416`.
- **Number of observations:** 12 (`observed_real_pair_duration_count ==
  required_real_pair_duration_count == B2A_REAL_PAIR_EVALUATIONS_TOTAL ==
  12`).
- **Precise statistic:** `conservative_pair = max(b2a_real_pair_seconds)`
  — literal source code,
  `src/kvcot/discovery/execution_measurement.py:416`. Independently
  confirmed by `per_real_pair_projection_seconds`'s docstring in
  `src/kvcot/discovery/b2a_evidence.py:289-294`: "the MAXIMUM total time
  among the completed real pair evaluations -- never the mean or an
  aggregate bucket."
- **Precise numeric value:** `10.25` seconds (range across the 12 was
  9.14s-10.25s).
- **Whether it is conservative:** yes. Using the maximum of 12 observed
  durations, rather than the mean or median, overstates per-pair cost
  relative to the population actually observed, which pushes the runtime
  projection upward (more conservative, i.e. less likely to
  under-project) rather than downward.

This value is **inherited unchanged** from B2A-R2 — it is a property of
the existing `build_runtime_projection` function, not a new B2A-R3
threshold. B2A-R3's own qualified row will produce its own 12 real-pair
durations at execution time (Stage C, §14); the reference value used
during CPU-only qualification planning (Stage A) is B2A-R2's own
already-measured `10.25`, used as the best available prior estimate
before any B2A-R3 inference occurs.

### Ambiguity resolution: meaningful compression (mandatory, resolved)

The exact previously authorized, currently-implemented definition of "real
and meaningful compression" for a single B2A attempt is:

```python
def derive_meaningful_compression_observed(
    *, selected_event_count: int, observed_retention_ratio: float
) -> bool:
    return selected_event_count >= 1 and observed_retention_ratio < 1.0
```

- **Source:** `src/kvcot/discovery/b2a_evidence.py:285-286`.
- **Definition basis:** a combination of actual selected-event count
  (must be at least 1) and cache-length reduction (`observed_retention_ratio
  < 1.0`, i.e. the R-KV physical cache is strictly smaller than the FullKV
  cache would have been) — not compaction count alone, not evicted-token
  count, and not a compressed-token-fraction ceiling.
- **How it is used:** this predicate is what feeds the
  `meaningful_compression_observed` field on `B2AGateEvidence`
  (`src/kvcot/discovery/b2a_contract.py:211`), one of the legacy
  mandatory gate conditions B2A-R2 passed
  (`docs/B2A_R2_RESULT_2026-07-22.md` §4, "Every gate": "meaningful_
  compression_observed" listed among the passing conditions).
- **Inherited or newly frozen:** **inherited unchanged.** There is no
  documented B2A-R2 failure implicating this predicate — B2A-R2 passed it
  — so it is copied exactly, not amended.

This execution-time predicate is distinct from, and does not replace, the
**qualification-time** (pre-execution, FullKV-only) proxy used to predict
compression feasibility before any R-KV inference runs. That proxy already
exists as two of B2A-R2's ten frozen `QUALIFICATION_CONDITIONS`
(`src/kvcot/discovery/b2a_qualification.py:40-51`):
`predicted_schedule_has_at_least_three_events` (`predicted_event_count >=
minimum_events`, default `QUALIFICATION_MINIMUM_EVENTS = 3`) and
`at_least_three_events_have_49_future_tokens` (`eligible_event_count >=
minimum_events`). §10.2 below inherits both predicates and additionally
freezes a **new, more conservative B2A-R3-only** minimum of 6 predicted
eligible events (raised from B2A-R2's 3) — see §10.2's justification,
which cites B2A-R2's own qualification-vs-measured attrition (31
predicted / 29 eligible at qualification time vs. 22 measured / 20
eligible at execution time, `docs/B2A_R2_RESULT_2026-07-22.md` §4) as the
evidence for that margin. This is flagged explicitly as a new decision,
not presented as an unchanged copy of B2A-R2's qualification minimum.

### Frozen predictor

Reference fields (all from B2A-R2's own real, measured execution — never
fabricated, never estimated):

```text
reference_total_tokens        = 4931   (109 prompt + 4822 generated;
                                         docs/B2A_R2_RESULT_2026-07-22.md
                                         §2 ordinal-2 row and §4;
                                         results/decisions/
                                         b2a_r2_qualification.json
                                         fields generated_token_count=4822,
                                         total_processed_tokens=4931)
reference_example_seconds     = 1378.3004406290129
                                         (docs/evidence/
                                         B2A_R2_ATTEMPT_INDEX_2026-07-22.json
                                         field per_example_total_wall_seconds;
                                         rounded 1378.30 in
                                         docs/B2A_R2_RESULT_2026-07-22.md §4)
reference_pair_seconds        = 10.25  (max of 12; see resolution above)
reference_setup_seconds       = ~19.0  (algebraically backed out:
                                         18034.6 - 12*1378.30 - 144*10.25
                                         ~= 19.0s; the FullKV-vs-R-KV
                                         startup split is not separately
                                         preserved in the repository —
                                         docs/B2A_R2_RESULT_2026-07-22.md
                                         §4)
safety_multiplier             = 1.20   (new B2A-R3 decision, see below)
```

**Why `safety_multiplier = 1.20` and not the old ×3 planning factor:** the
original pre-GPU cost model (`docs/B0_5_PROTOCOL_REPAIR.md` §15) applied a
×3 safety factor to a *guessed* 35-50 tok/s throughput range, because no
real measurement existed yet. B2A-R2 replaced that guess with a real,
measured per-example and per-pair cost (§15's own words: "B2A ... replaces
this planning number with a measured one"). Stacking a further ×3 onto an
already-measured value, which itself already uses the conservative
maximum (never the mean) per-pair statistic, would be excessively
conservative to the point of making almost no row qualify. `1.20` is a new,
explicitly smaller B2A-R3 margin layered on top of two statistics that are
already conservative by construction (the per-pair maximum, and using
B2A-R2's own real per-example cost as the reference rather than a faster
hypothetical trace). This is a protocol decision, not a value read from
any existing artifact — recorded here as new, not inherited.

Frozen formula:

```text
reference_seconds_per_token
    = reference_example_seconds / reference_total_tokens

predicted_example_seconds
    = reference_seconds_per_token
      × candidate_total_tokens
      × safety_multiplier

predicted_pair_seconds
    = reference_pair_seconds
      × safety_multiplier

projected_total_seconds
    = reference_setup_seconds
      + 12 × predicted_example_seconds
      + 144 × predicted_pair_seconds

projected_gpu_hours
    = projected_total_seconds / 3600
```

**Verified multiplicities.** `12` and `144` are not assumed by this
protocol — they are read directly from the existing, already-frozen
projection logic:

- `src/kvcot/discovery/execution_measurement.py:363-372`,
  `build_runtime_projection(..., example_count: int = 12, real_pair_count:
  int = 144)`.
- `src/kvcot/discovery/constants.py:29-36`:
  `B2B_PILOT_EXAMPLE_COUNT = 12`;
  `B2B_PILOT_TOTAL_REAL_PAIR_EVALUATIONS = B2B_PILOT_EXAMPLE_COUNT *
  EVENTS_SELECTED_PER_EXAMPLE * REAL_PAIR_EVALUATIONS_PER_EVENT = 12 * 3 *
  4 = 144`.

`12` is the number of examples in a full B2B pilot; `144` is 12 examples
times 12 real-pair evaluations per example (3 selected events × 4
candidate/donor pairs each). The projection formula answers "if a B2B
pilot ran 12 examples like this candidate, each producing per-pair costs
like B2A-R2's own measured maximum, what would total runtime be" — the
same question B2A-R2's own `build_runtime_projection` already answers,
reused here unchanged for B2A-R3 candidate qualification.

Gates:

```text
Qualification pass:
projected_gpu_hours <= 3.60

Final hard gate:
projected_gpu_hours <= 4.00
```

The 3.60-hour target is a safety margin, not a replacement for the
4.00-hour gate — qualifying at or under 3.60 hours does not itself
satisfy the final gate; the actual B2A-R3 attempt (Stage C, §14) must
still measure and pass `runtime_within_limit <= 4.00` from its own real
evidence, exactly as B2A-R2's did (and failed).

The predictor must preserve, for every candidate evaluated: every raw
input (candidate prompt/generated/total token counts, FullKV wall
seconds); every constant used (`reference_*`, `safety_multiplier`);
intermediate values (`reference_seconds_per_token`,
`predicted_example_seconds`, `predicted_pair_seconds`); final
`projected_total_seconds` and `projected_gpu_hours`; a predictor version
string; and the source timing-artifact hashes the `reference_*` constants
were read from. No predictor input may be manually overridden by an
operator at qualification time — a candidate is rejected outright if any
required predictor input is missing (§10.3).

If a token-length ceiling is ever displayed to an operator during CPU
dry-run planning, it must be shown as a computed consequence of solving
the formula above for `candidate_total_tokens` at `projected_gpu_hours =
3.60`, never as an independently hand-chosen threshold.

## 8. Deterministic candidate pool

- Same pinned MATH-500 revision: `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`
  (§4).
- Excludes every row already used by B2A-R1 and B2A-R2:
  - B2A-R1's executed row: `test/precalculus/807.json` (`example_index=0`
    — verified via `git show 9fe27a2:configs/discovery/
    b2a_one_example_manifest.json`, the manifest content immediately
    before B2A-R2's freeze overwrote it).
  - Every one of the 12 unique IDs in B2A-R2's committed candidate
    manifest (`configs/discovery/b2a_r2_candidate_manifest.json`,
    `canonical_sha256 = ac2dcc4550a89f2cfa701acd608a8087b4a1ebaa0ea05eb
    15d8f71e3434ee0ec`) — all 12 are excluded, not only the one that was
    actually executed (`test/number_theory/820.json`), because FullKV
    qualification inference already ran against three of them (ordinals
    0-2) and content-based exclusion of the whole committed manifest is
    unambiguous and costs nothing given MATH-500's size. This is a
    protocol decision, stated explicitly rather than left ambiguous.
  - Total excluded rows: 13 unique `unique_id` values (1 from B2A-R1 + 12
    from B2A-R2's manifest, with no overlap other than
    `test/number_theory/820.json` already being one of the 12).
- Candidate pool size: 16 candidates total.
- Level mixture: **8 level-4 candidates, 8 level-5 candidates** — this
  mixture is compatible with the pinned dataset (MATH-500's `level`
  column is a bare digit string 1-5,
  `src/kvcot/discovery/manifest_prepare.py` `EXPECTED_MATH500_COLUMNS`)
  and does not conflict with B2A-R2's own candidate design (B2A-R2 used
  level-5 only, `CANDIDATE_LEVEL = 5` in
  `src/kvcot/discovery/b2a_r2_candidates.py`; B2A-R3 additionally
  considers level-4 to widen the pool of MATH-500 rows away from a
  level-5-only population without touching level ≤3 rows, which are
  markedly shorter and less likely to trigger real eviction — consistent
  with `docs/B2A_R1_FAILURE_AND_B2A_R2_PROTOCOL_2026-07-22.md`'s own
  stated reason for restricting to level-5: "the longest, hardest
  problems, most likely to produce a long enough generated trace to
  actually exercise R-KV's budget=1024 eviction trigger").
- Qualification is permitted for only the first 8 candidates in
  manifest order; qualification stops at the first passing candidate.
- No earlier passing candidate is ever replaced by a later one.
- The candidate manifest is committed (Step 3) before any GPU
  qualification inference runs, exactly as B2A-R2's was.
- Row IDs (`unique_id`), content hashes, `level`, and the ordering key are
  preserved for every candidate, mirroring `CandidateRow`'s existing
  fields (`src/kvcot/discovery/b2a_r2_candidates.py:76-93`).

The pool is not hand-selected based on expected answer correctness,
expected trace length, mathematical topic/subject, or any prior model
behavior — membership is determined solely by (a) pinned-dataset content,
(b) the level filter above, (c) the exclusion set above, and (d) the
deterministic ordering key (§9).

## 9. Exact deterministic ordering algorithm

B2A-R3 **reuses B2A-R2's existing, already-implemented ordering-hash
construction** (`src/kvcot/discovery/b2a_r2_candidates.py:44-53`,
`_ordering_hash`) rather than introducing a new canonical-form scheme.
There is no documented B2A-R2 failure implicating this construction — the
ordering hash is not what B2A-R2 failed on (it failed the runtime gate) —
so per this protocol's own discipline ("copy the existing frozen
definition exactly unless there is a documented failure requiring an
amendment"), it is reused, extended only with a new protocol-version
string so the ordering-hash space is distinct from B2A-R2's own (this
also structurally prevents any accidental collision with B2A-R2's
ordering).

Frozen construction:

- **Hash algorithm:** SHA-256 (`kvcot.utils.hashing.sha256_text`, the same
  helper `b2a_r2_candidates.py` uses).
- **Canonical fields, in order:** `protocol_version`, `dataset_revision`,
  `model_revision`, `budget`, `unique_id` — identity fields only, deliberately
  excluding problem text, subject, level, or any other content that could
  correlate with expected difficulty or trace length (this exclusion is
  the existing convention's own stated purpose:
  "`_ordering_hash`... a fixed function of identity fields ONLY... never a
  function of anything observed about generation").
- **Field order, separators, encoding:** exactly
  `f"{protocol_version}|{dataset_revision}|{model_revision}|budget=
  {budget}|{unique_id}"`, UTF-8 encoded (`sha256_text`'s existing
  behavior) — pipe-separated, no additional normalization step, matching
  `_ordering_hash`'s literal implementation.
- **Protocol version string (new, B2A-R3-only):**
  `"faithkv-b2a-r3-row-order-v1"` — distinct from B2A-R2's
  `"faithkv-b2a-r2-row-order-v1"` (`CANDIDATE_MANIFEST_PROTOCOL_VERSION`,
  `src/kvcot/discovery/b2a_r2_candidates.py:39`), so no B2A-R3 ordering
  hash can coincide with a B2A-R2 one for the same `unique_id`.
- **Sort direction:** ascending by the hex-digest string, exactly as
  `scored.sort(key=lambda item: item[0])` already does in
  `build_candidate_manifest`.
- **Tie-breaking rule (new, made explicit for completeness):** ascending
  `unique_id` string comparison. SHA-256 collisions between two distinct
  `unique_id` values are not expected to occur in a pool of this size; this
  tie-breaker exists so the ordering is total and reproducible even in
  the astronomically unlikely case of an exact digest match.
- **Whole-manifest hash:** `manifest["canonical_sha256"] =
  sha256_json(manifest)`, exactly as `build_candidate_manifest` already
  computes it (`src/kvcot/discovery/b2a_r2_candidates.py:204`).

This protocol deliberately does **not** adopt a new NFC/CRLF-normalized,
problem-text-inclusive canonical form, because doing so would diverge from
an existing, working, already-tested convention without a documented
reason to change it. Reuse, not reinvention, is the governing choice here
(matching this document's own instruction to prefer existing repository
canonicalization helpers). The Step 3 CPU implementation must reuse
`kvcot.discovery.b2a_r2_candidates`'s existing `_ordering_hash` construction
(parameterized by the new protocol-version string above), not
reimplement it independently.

## 10. Candidate qualification gates

A row qualifies only when every gate below passes.

### 10.1 Correctness and trace integrity

- FullKV answer is correct.
- Answer verifier returns a valid (not `unverifiable`) decision — mirrors
  `fullkv_answer_verifiable`/`fullkv_answer_correct` in
  `src/kvcot/discovery/b2a_qualification.py:136-138`.
- Generation cap is not hit (`no_cap_hit`, same source, line 136).
- Thinking span parses correctly and the trace is complete.
- Prompt and generation token counts are available.
- FullKV timing evidence is complete.
- No required field is manually inserted after execution — every field is
  produced by the same `run_fullkv_worker` path B2A-R2's qualifier reused
  (`src/kvcot/discovery/b2a_qualification.py` module docstring).

### 10.2 Compression feasibility

- Total sequence length exceeds the R-KV budget (1024).
- Static prediction indicates real compaction
  (`kvcot.analysis.rkv_schedule.predicted_compaction_event_positions`).
- The repository-verified "meaningful compression" condition (§7's
  resolved definition, `derive_meaningful_compression_observed`) is the
  execution-time gate this qualification is designed to make likely to
  pass; qualification itself uses the pre-execution proxy below.
- **At least 6 predicted eligible events** (raised from B2A-R2's frozen
  qualification minimum of 3 — `QUALIFICATION_MINIMUM_EVENTS = 3`,
  `src/kvcot/discovery/b2a_qualification.py:38`). This is a **new,
  explicitly more conservative B2A-R3-only qualification decision**,
  justified as follows:
  - Only 3 events are ultimately selected per example (§4).
  - The predictor is a static approximation and may overestimate measured
    eligibility.
  - B2A-R2's own qualification predicted more eligible events than were
    later measured at execution time: 31 predicted / 29 eligible at
    qualification vs. 22 measured / 20 eligible at execution
    (`docs/B2A_R2_RESULT_2026-07-22.md` §4: "Qualification's pure-arithmetic
    *prediction* was 31 events / 29 eligible; the *measured* real-run
    counts, 22 / 20, differ somewhat... both were comfortably above the
    required minimum of 3").
  - Doubling the qualification-time minimum from 3 to 6 preserves the
    same comfortable margin B2A-R2 happened to have (its measured count,
    20, was roughly 6-7× the old minimum of 3) while explicitly protecting
    against a candidate whose predicted count is only marginally above 3
    and could plausibly attrit below the 3 actually needed once measured.
  - This does not weaken any existing gate — it raises a threshold, and
    only at qualification time (a pre-execution proxy), never at
    execution time.
- **At least 3 of the predicted events each have** one bridge token
  (`BRIDGE_TOKEN_COUNT = 1`) and at least 48 future scored tokens
  (`SCORED_HORIZON = 48`) — together, 49 total future tokens required per
  event, matching `MINIMUM_FUTURE_TOKENS_AFTER_EVENT = 49`
  (`src/kvcot/discovery/constants.py:14-16`) and B2A-R2's own
  `at_least_three_events_have_49_future_tokens` condition
  (`src/kvcot/discovery/b2a_qualification.py:140`).
- Event positions use absolute token indexing; boundary arithmetic
  matches the existing B0.5/B1 protocol (`src/kvcot/discovery/pass1.py`
  `eligible_event_positions`, reused unchanged, never reimplemented for
  B2A-R3).

### 10.3 Runtime qualification

- Projected runtime (§7's formula) is at most 3.60 GPU-hours.
- All timing fields (`reference_*` inputs, candidate token counts) are
  present.
- The predictor version matches this frozen protocol.
- The safety multiplier is exactly 1.20 (§7) — never overridden per
  candidate.
- No runtime value is manually overridden.
- The candidate is rejected outright when any predictor input is missing
  (never silently defaulted).

### 10.4 Deterministic selection

- Candidates are evaluated strictly in manifest order (§9).
- At most the first 8 candidates are evaluated (§8).
- Evaluation stops immediately after the first passing row.
- Every attempted row (pass or fail) is preserved in the qualification
  artifact (§12).
- Every rejection reason is preserved.
- No row after a pass is ever evaluated.
- If all 8 candidates fail, no row is selected — qualification produces no
  selected row (§18, "No candidate qualifies").

## 11. Qualification wall-time and bounded execution

This section defines, but does **not** itself authorize, a bounded
FullKV-only qualification phase — that phase requires its own future,
separate authorization (§14, Stage B).

Frozen for that future phase:

- One RTX 3090.
- FullKV only — no R-KV import.
- Maximum 8 candidates attempted.
- Stop at the first passing row.
- No pair evaluation, no pair artifacts, no causal outcomes of any kind.

**Wall-time limit:** `VERIFY BEFORE FREEZE` in the sense that this
document does not invent one. B2A-R2's own qualification pass needed only
~9.9 minutes across 3 attempted candidates (39.1s + 87.8s + 465.4s;
`docs/B2A_R2_RESULT_2026-07-22.md` §2), but that is a single observed data
point, not a worst-case bound — a candidate that survives longer before
hitting an incorrect answer, or whose generation runs close to the
`max_new_tokens=6144` cap, could take substantially longer, and B2A-R3
permits up to 8 attempts (not 3). No repository artifact independently
establishes a worst-case per-candidate FullKV qualification wall-time,
distinct from the existing 7200-second (2-hour) per-worker subprocess
timeout (`B2A_WORKER_TIMEOUT_SECONDS`,
`src/kvcot/discovery/constants.py:59`), which bounds a single worker
invocation but was not designed as a qualification-phase-wide planning
number.

```text
The implementation must calculate and display the protocol-derived wall-time
limit during CPU dry-run planning. A separate authorization document must freeze
the final qualification wall-time before GPU use.
```

## 12. Candidate and qualification artifact schemas

Conceptual schemas only — no Python classes are implemented by this
document.

### Candidate manifest

Recommended path: `artifacts/b2a_r3/manifests/candidates.json`.

Required fields:

```text
protocol_version
dataset_name
dataset_revision
dataset_revision_hash
exclusion_manifest_hash
candidate_count
qualification_limit
level_mixture
canonicalization_version
ordering_algorithm
ordering_seed_or_domain_separator
candidates
manifest_hash
created_at
```

Each candidate:

```text
candidate_index
dataset_row_id
level
problem_hash
ordering_key
excluded_prior_attempt_check
```

No model outcome of any kind is stored in the committed candidate
manifest — mirrors B2A-R2's own `CandidateRow`, which stores only
identity/content fields (`src/kvcot/discovery/b2a_r2_candidates.py:76-93`),
never a predicted or observed outcome.

### Qualification artifact

Recommended path: `artifacts/b2a_r3/qualification/qualification.json`.

Each attempted candidate:

```text
protocol_version
candidate_manifest_hash
candidate_index
dataset_revision
dataset_row_id
problem_hash
prompt_hash
fullkv_correct
answer_verifier_status
generation_cap_hit
thinking_span_valid
trace_complete
prompt_tokens
generated_tokens
total_tokens
fullkv_runtime_seconds
predicted_compactions
predicted_eligible_events
events_with_required_future
meaningful_compression_predicted
reference_seconds_per_token
predicted_example_seconds
predicted_pair_seconds
projected_total_seconds
projected_gpu_hours
qualification_passed
rejection_reasons
attempt_started_at
attempt_completed_at
```

Top level:

```text
protocol_version
candidate_manifest_hash
attempted_candidate_count
first_passing_candidate_index
selected_row_id
selected_row_hash
selection_status
qualification_stopped_reason
runtime_predictor_version
runtime_source_artifact_hashes
artifact_hash
```

Both schemas require canonical, deterministic hashing of their own
content (`manifest_hash`/`artifact_hash`), mirroring
`build_candidate_manifest`'s existing `canonical_sha256 =
sha256_json(manifest)` pattern.

## 13. Selected-row freeze contract

A future hash-verified freezer (Step 3) must:

- Accept only a row present in the committed candidate manifest.
- Accept only the first passing candidate recorded in the qualification
  artifact.
- Verify the candidate-manifest hash.
- Verify the qualification-artifact hash.
- Verify the dataset revision.
- Verify the row ID and problem hash.
- Reject arbitrary row substitution.
- Reject selection of a later passing row when an earlier one passed.
- Reject manual artifact editing (any field mismatch against the
  regenerated hash is a hard failure).
- Produce a one-row selected manifest with full provenance hashes.

This mirrors `kvcot.discovery.b2a_r2_freeze.freeze_qualified_row`'s
existing behavior exactly (`docs/B2A_R2_RESULT_2026-07-22.md` §3: "verified
the qualification artifact's hash chain against the committed candidate
manifest, re-checked every one of the selected candidate's own recorded
conditions, and only then replaced" the one-example manifest).

Recommended future selected manifest:
`artifacts/b2a_r3/manifests/selected.json`.

No selected manifest may exist before FullKV qualification (Stage B, §14)
has run.

## 14. Authorization and attempt-consumption boundaries

### Stage A — Authorized after independent audit of this document

```text
CPU implementation, CPU tests, and dry-run planning only
```

Allowed: schemas; the deterministic candidate generator (reusing §9);
the runtime predictor (§7); the qualification evaluator (§10); the
first-pass selector (§10.4); the freezer (§13); attempt guards; a dry-run
CLI; mocked/injected CPU tests only.

Forbidden: model loading; CUDA initialization; FullKV inference; R-KV;
pair evaluation.

### Stage B — Future separate authorization

```text
One bounded FullKV-only qualification
```

Prerequisites: this protocol independently audited; CPU implementation
complete; green CPU CI; candidate manifest committed and hash-verified;
the runtime predictor independently checked; the qualification wall-time
(§11) frozen by that future authorization; a dated authorization document
committed.

Stage B does not authorize B2A-R3 execution.

### Stage C — Future separate authorization

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

## 15. CPU implementation requirements for Step 3

Documented, not implemented, by this task:

1. B2A-R3 candidate-manifest schema (§12).
2. Deterministic candidate generator (§8, §9).
3. Prior-row exclusion logic (§8).
4. Runtime predictor (§7).
5. Qualification evaluator (§10).
6. First-qualified-row selector (§10.4).
7. Qualification artifact writer (§12).
8. Hash-verified selected-row freezer (§13).
9. Attempt-consumption guard (§14).
10. Dry-run planning CLI.
11. Pair-record persistence requirements inherited unchanged from B2A-R2
    (`rkv/pair_records.json`, `rkv/scientific_summary.json`,
    `verify_pair_record_artifacts`, `verify_pair_record_population`).
12. Final/completion/CLI outcome consistency, inherited unchanged from the
    B2A-R2 forensic repair (`overall_passed` computed once, read
    everywhere — `docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md`
    §11).

Required later CPU tests: deterministic ordering; canonicalization
stability; prior-row exclusions; level-mixture enforcement; manifest hash
stability; predictor arithmetic exactness; safety-multiplier enforcement;
rejection above 3.60 hours; missing-input rejection; event-count
calculation; future-horizon calculation; first-pass stopping;
eight-candidate limit; later-candidate substitution refusal; no R-KV
imports during qualification; no pair outcomes during qualification; no
CUDA from dry-run; selected-manifest hash verification;
attempt-consumption enforcement; mandatory pair persistence; CLI/final/
completion consistency.

## 16. Mechanical B2A-R3 acceptance gates

| Category | Required result |
|---|---|
| FullKV answer | Correct |
| R-KV answer | Correct |
| Answer verifier | Valid |
| Generation cap | Not hit |
| Thinking span | Valid |
| Compression | Existing exact meaningful-compression gate passes (§7's resolved `derive_meaningful_compression_observed`) |
| Measured eligible events | At least 3 |
| Selected events | Exactly 3 |
| Real pair records | Exactly 12 |
| No-op records | Exactly 1 |
| Baseline NLL arrays | Exactly 48 values per pair (`SwapPairRecord.baseline_per_token_nll`, `min_length=max_length=SCORED_HORIZON`) |
| Swapped NLL arrays | Exactly 48 values per pair (same schema, `swapped_per_token_nll`) |
| Pair records | Durably persisted (`rkv/pair_records.json`) |
| Pair artifact | Present and hash-verified |
| Scientific summary | Present and hash-verified |
| Replay | Token-identical |
| No-op gain | Zero within the frozen tolerance: `swap_gain == 0.0` per `_close(a, b)` with `_FLOAT_DIFF_TOLERANCE = 1e-9` (`src/kvcot/discovery/schemas.py:51,54,259-267`) — an exact-equality schema invariant on `baseline_per_token_nll == swapped_per_token_nll`, not merely a numeric closeness check |
| Peak VRAM | At most 22 GiB |
| Runtime | At most 4.00 GPU-hours |
| `completion.json` | Internally consistent |
| `final.json` | Internally consistent |
| CLI exit code | 0 only when `overall_passed` is true |
| Archive | Externally verified before instance destruction |

The exact condition names and count for "every gate" above are defined by
`FINAL_MANDATORY_GATE_CONDITIONS` (`src/kvcot/discovery/final_contract.py`)
and `evaluate_b2a_gate` (`src/kvcot/discovery/b2a_contract.py`) in the
current codebase. This protocol does not restate or re-derive that exact
list or count, because it has already evolved since B2A-R2's historical
run (e.g. the F7 device/offload repair added
`no_offload_and_placement_verified` after B2A-R2 executed) — restating a
stale count here would silently drift out of sync with the source of
truth. B2A-R3 inherits whatever that list is at Step 3 implementation
time, unchanged by this protocol.

Qualifying at 3.60 GPU-hours (§7, §10.3) does not itself satisfy the final
4.00-hour gate — the actual attempt's own measured runtime evidence must
still separately pass `runtime_within_limit` (§7).

## 17. Scientific mechanism gate

Kept separate from mechanical acceptance (§16).

Frozen thresholds, verified from the prior protocol:

- **Positive-gain threshold: `swap_gain > 0.01` nats.** Source:
  `_MEANINGFUL_GAIN_THRESHOLD = 0.01` in
  `src/kvcot/discovery/scientific_summary.py:24`, computed into
  `gain_above_0_01_count` (line 94). This value is inherited unchanged
  from the original B0.5-era discovery-pattern rule
  (`docs/b0_5_decision.json` `b0_5_r2_1_repaired_gate_10`: "at least 4 of
  12 examples show >=1 valid pair with swap_gain > 0.01 nats, strict";
  `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §16).
- **Correlation ceiling: `|Spearman rho| < 0.30`.** Source: the same
  `b0_5_r2_1_repaired_gate_10` decision and
  `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §16; computed for a single B2A
  attempt as `spearman_score_margin_vs_swap_gain`
  (`src/kvcot/discovery/scientific_summary.py:104-107,120`, a tie-aware
  Spearman correlation between `score_margin_e_minus_r` and `swap_gain`
  over the attempt's valid real pairs).
- **No-op gain:** zero within the frozen tolerance — identical to §16's
  row (`_close(swap_gain, 0.0)`, tolerance `1e-9`).

**Necessary single-example scale adaptation, stated explicitly, not
silently copied.** The original `b0_5_r2_1_repaired_gate_10` rule was
designed for a 12-**example** B2B-scale population: "at least 4 of 12
examples" and "median absolute **per-example** Spearman rho... computed
separately within each example (never pooled)" across 8 deployable
signals (`docs/b0_5_decision.json`). B2A (and B2A-R3) runs exactly **one**
example with 12 real pairs, never 12 examples — there is no population of
examples to pool or take a median across. The measurable, non-fabricated,
single-example analogue that this protocol freezes for B2A-R3 is:

```text
At least 4 of the 12 real pairs in the single B2A-R3 attempt:
swap_gain > 0.01 nats
(gain_above_0_01_count >= 4, out of real_pair_count == 12)

The single computed value:
|spearman_score_margin_vs_swap_gain| < 0.30
(there is one example, so there is one correlation value, not a median
across examples)

No-op gain: zero within the frozen 1e-9 tolerance
```

This is a **new B2A-R3 protocol decision** for the unit of measurement
(pair, not example), reusing the exact, unchanged numeric thresholds
(`0.01`, `0.30`) the prior protocol already froze. It is stated here as a
scale adaptation, not presented as an unmodified verbatim carry-over of
the population-level rule. It is also, in the current codebase,
descriptive rather than gating: `scientific_summary.py`'s fields are
computed and persisted (per the merged forensic repair) but are not wired
into `FINAL_MANDATORY_GATE_CONDITIONS` as a pass/fail condition — matching
this section's own framing below.

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

## 20. Protocol freeze checklist

- [x] Research question frozen (§3).
- [x] Scientific settings verified against repository sources (§4).
- [x] Outcome-blind allowed signals frozen (§6).
- [x] Forbidden signals frozen (§6).
- [x] Pair-timing statistic verified (max of 12, §7).
- [x] Meaningful-compression definition verified (§7).
- [x] Runtime formula frozen, multiplicities verified from source (§7).
- [x] Safety multiplier frozen and justified as new (§7).
- [x] Deterministic candidate canonicalization frozen, reusing B2A-R2's
      construction (§9).
- [x] Candidate level mixture frozen (§8).
- [x] First-pass rule frozen (§10.4).
- [x] Eight-candidate cap frozen (§8, §10.4).
- [x] Qualification artifact schema frozen (§12).
- [x] Selected-row freezer contract frozen (§13).
- [x] Authorization stages frozen (§14).
- [x] Mechanical gates frozen, deferring exact count to source of truth
      (§16).
- [x] Scientific gates frozen, with the single-example scale adaptation
      stated explicitly (§17).
- [x] Kill gates frozen (§18).
- [x] GPU remains prohibited pending separate authorization (§14, §19).

## 21. Repository/authorization discrepancy noted for the record

The task instructions that produced this document referred to the
repository as `attaulasad/Faithkv`. This repository's actual configured
remote (`git remote -v`) is `https://github.com/asad073-ui/Faithkv.git`.
All git-level facts verified in Phase A (branch, HEAD, ancestor commits,
submodule SHA) matched exactly regardless of this naming discrepancy,
which appears to be an operator/org-name mismatch rather than a different
repository — recorded here rather than silently resolved, per this
document's own discipline of not silently inferring unverified facts.
