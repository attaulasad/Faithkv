# B0.5 — Causal false-negative discovery protocol (preregistration, not authorized to run)

Phase B0.5 artifact (2026-07-19). Branch `research/b0-5-discovery-protocol`,
created from `research/phase-b0-method-pivot` at commit `68b56f1` (which sits
on top of the B0 merge commit `f7e9dcc`, itself on top of B0's own commit
`8d5aa21` — verified ancestor, see `docs/B0_5_FEASIBILITY_AUDIT.md` §0).
Roles applied: strict senior LLM-inference researcher; adversarial ICLR
reviewer; research engineer preserving historical evidence.

**This is not B1. Nothing in this document is authorized to run.** No
compression method is implemented here, no discovery harness is
implemented here, no GPU command has been run, no dataset manifest exists,
no model weights or datasets were downloaded. This is a preregistration for
a future, separately authorized GPU pilot, plus the feasibility judgment
needed to decide whether requesting that later phase is worthwhile.

## 0. Research question — untested hypothesis, not a finding

> At an operating point that first passes a predeclared natural-accuracy
> plausibility gate and a realized-compression gate, does a deployed
> reasoning KV compressor evict blocks that (1) it ranks safe to remove,
> (2) are actually removed, (3) have high counterfactual future utility
> under a controlled KV-only intervention, (4) are not identified by
> established cheap signals, (5) occur frequently enough to motivate a
> distinct algorithmic operation?

Call such blocks **unexplained causal false negatives**. This phrase does
not imply they have already been observed.

**Status: UNTESTED HYPOTHESIS.** `docs/METHOD_PIVOT_SPEC.md` §5 already
states this; this document does not weaken or strengthen that status. The
retired GSM8K b128 operating point's A2 failure atlas is explicitly **not**
existence evidence — its operating point failed the accuracy gate by 40pp
(`docs/METHOD_PIVOT_SPEC.md` §3-§4), so nothing observed there can be
attributed to a *deployed, accuracy-plausible* policy's behavior.

## 1. Experimental unit: fixed 128-token block

**Unit chosen: a fixed block of `divide_length=128` consecutive generated
tokens**, aligned exactly to this repository's already-audited R-KV
compaction schedule (`docs/UPSTREAM_AUDIT.md` H4; `kvcot.analysis.rkv_schedule`
already predicts these boundaries from a FullKV trace alone, CPU-only, no
GPU needed).

**Why this unit, and not token-level or semantic thought-block:**

- **Token-level** would require an ablation replay per candidate *token*.
  At even a modest sample (say 500 tokens), this multiplies the cost model
  in §9 by roughly two orders of magnitude relative to a few dozen blocks
  — infeasible under the 4-GPU-hour ceiling (§8).
- **Semantic thought-block** (à la ThinKV's attention-sparsity-derived
  reasoning/execution/transition categories) requires a validated
  thought-boundary classifier this repository does not have and has not
  validated (the A2 failure atlas explicitly notes `</think>` is a
  token-format boundary, not a semantic one — `docs/METHOD_PIVOT_SPEC.md`
  §3 point 4). Building and validating such a classifier is itself
  unbudgeted engineering work, and would confound "is this an unexplained
  false negative" with "is this classifier's segmentation correct."
- **Fixed 128-token blocks** require no new classifier, are already the
  unit R-KV's own compaction schedule operates over, and their boundaries
  are computable entirely on CPU before any GPU time is spent — the same
  property that makes this repo's existing `inspect-fixed-trace
  --write-selection` preflight possible.

Block `k` in a given natural R-KV run is exactly the token span evicted or
retained at compaction checkpoint `k` of that run's schedule.

## 2. Compaction events eligible for sampling

A compaction event `t` (for a given example, seed, natural R-KV run) is
**eligible for sampling** iff all of:

1. It is not the run's first compaction event (excluded — no meaningfully
   compressed retained-set history yet, no realistic "deployed policy"
   state).
2. It is not the run's last compaction event (excluded — leaves no room
   for the future-horizon measurement in §5 before generation ends).
3. At least `horizon` (§5) generated tokens remain in the natural run after
   event `t` (else the reference continuation for teacher-forcing does not
   exist).
4. The natural run did not hit the generation token cap (a capped run's
   post-cap "continuation" is an artifact of truncation, not of the
   model's own stopping behavior — excluded entirely, matching this
   repository's existing cap-hit discipline).

From each eligible example's remaining events, sample `events_per_example`
(§8's chosen value) deterministically via a new derived sub-seed:
`SHA-256(global_seed, dataset_name, problem_index, "b05_event_sample")`
(same construction family as `kvcot.utils.seeding.derive_seed`, extended
with a literal disambiguating suffix — never re-using the base seed for a
second, different purpose). Selection is over eligible events only, never
conditioned on any measured outcome.

## 3. Base-compressor score capture

At each sampled event, log R-KV's own internal joint score
`Z = λ·Importance − (1−λ)·Redundancy` (`docs/UPSTREAM_AUDIT.md` §2, H2-H4)
for every block in the compaction decision, via an additive read-only
instrumentation hook living in `src/kvcot` (a thin logging wrapper around
the pinned R-KV forward pass, analogous to this repo's existing
`kvcot.generation.replay` layer) — **the R-KV submodule itself is not
modified**, satisfying decision criterion 7 (§11 of the task spec, `docs/
B0_5_FEASIBILITY_AUDIT.md` §6). `b_i` for block `i` is this logged score at
the event where it was evicted or retained.

## 4. Which states are actually evicted, and matched retained controls

**Evicted-candidate blocks:** identified directly from the diff of the
retained-token-ID set immediately before and after compaction event `t`
(deterministic bookkeeping already present in R-KV's cache object — no
inference needed). Sample `evicted_per_event` of them via the same
derived-sub-seed procedure as §2 (deterministic among *actually evicted*
blocks at that event — not a free choice among all candidates R-KV
considered, since the discovery question is specifically about blocks the
deployed policy did remove).

**Retained-control blocks:** sampled from blocks the base policy actually
**kept** at event `t`, matched on (a) distance from the compaction boundary
and (b) block length, to the evicted candidates — via the same
deterministic sub-seed. **Selection never depends on any measured utility
`u_i`** (that would be outcome-based selection, forbidden by the task
brief §6 and by this repository's existing outcome-blind-selection
discipline, `kvcot inspect-fixed-trace --write-selection`).

## 5. Reference cache state and the exact KV-only ablation

Let `C_rkv,t` be the **real, deployed R-KV cache** exactly as it existed
immediately after compaction event `t` (the actually-realized compressed
cache — not a hypothetical). This is the reference cache state for every
block sampled at event `t`.

Two, and only two, single-block cache modifications are ever constructed,
always relative to `C_rkv,t`:

- **Rescue intervention (evicted candidates):** `C_rkv,t` does not contain
  block `i`'s true KV entries — they were dropped by the natural eviction,
  not by an experimenter action. To measure `i`'s counterfactual utility,
  reconstruct its true KV entries via a **shadow-FullKV replay**: replay
  the R-KV run's own generated token sequence (the exact tokens it
  produced, teacher-forced) from position 0 up to event `t` under the
  **FullKV** cache policy — FullKV never evicts, so this recovers block
  `i`'s true KV state exactly as it would have been had it never been
  evicted. Insert those recovered entries into a copy of `C_rkv,t`,
  producing `C_rkv,t + {i}`.
- **Ablation intervention (retained controls):** block `j` is already
  present in `C_rkv,t`. Mask/zero its KV entries in a copy of `C_rkv,t`,
  producing `C_rkv,t − {j}`.

**This is intentionally asymmetric and by design:** for evicted candidates,
the "absent" state is the real, natural eviction (nothing synthetic); only
the "present" state is reconstructed. For retained controls, the "present"
state is real; only the "absent" state is synthetic. This is what §6 of the
task brief calls distinguishing *ablation from natural eviction* — exactly
one side of each comparison is ever synthetic, and it is always the side
that did not naturally occur. It also answers "full-cache ablation from
ablation inside an already-compressed cache": every modification is a
single-block perturbation of the already-compressed `C_rkv,t`, never a
perturbation of a full/uncompressed cache.

**What remains fixed during the intervention:** every other block's KV
content, the model weights, the attention backend, the decoding
temperature/mode (greedy, matching this repo's frozen probe convention),
and the token sequence being teacher-forced. Only the single target
block's presence/absence changes between the two branches compared for
that block.

**Are reference tokens teacher-forced? Yes, always** — the tokens scored
in §6 below are the R-KV natural run's own actually-generated continuation,
never a resampled or hypothetical continuation. This isolates the cache
perturbation's effect from generation-sampling noise, the same rationale
`docs/EXPERIMENT.md` §2 gives for why base and probe generation intervene
on token replay rather than bulk re-prefill.

## 6. Future evaluation horizon and metrics

**Horizon:** `horizon = 48` teacher-forced tokens following the compaction
event, reusing this repository's existing frozen probe budget
(`configs/lock.yaml` `probes.max_new_tokens: 48`) rather than inventing a
new number — a value this project has already validated as long enough to
observe an answer-relevant change (`docs/EXPERIMENT.md` §2, §7) without
being needlessly expensive.

**Primary metric — reference-token NLL increase over the fixed horizon:**
for a block `i` with "present" cache `C_present` and "absent" cache
`C_absent` (per §5's asymmetric construction),

```
u_i = mean_{k=1..horizon} [ -log P(ref_token_k | C_absent) ]
    - mean_{k=1..horizon} [ -log P(ref_token_k | C_present) ]
```

i.e. `u_i` = how much worse (higher NLL, mean per token, horizon-length
normalized so it is comparable across events) the reference continuation
becomes when block `i` is absent rather than present. Positive `u_i` means
block `i` had positive utility. **Continuous, token-level — never
final-answer correctness**, per the task brief's explicit requirement that
the primary metric not be answer-correctness (too coarse and expensive to
power a discovery pilot at this `n`).

**Secondary metrics** (computed from the *same* forward passes — no
additional GPU cost, since the full logit vector at each horizon position
is already materialized):

- Answer-token NLL increase (the same quantity as the primary metric, but
  restricted to the final boxed-answer tokens specifically, when the
  horizon reaches them).
- FullKV–R-KV-style KL divergence between the `C_present`/`C_absent`
  next-token distributions at each horizon position (full-vocabulary KL if
  affordable within the horizon; a top-k-restricted approximation
  otherwise — whichever is used, the choice and its approximation error
  must be stated in the eventual run's report, never silently swapped).
- Top-1 agreement between `C_present`/`C_absent` at each horizon position.
- Target-token log-probability under each condition (a raw ingredient of
  the primary metric, reported separately for diagnosis).
- Probability mass on the branching/overthinking marker set (§7) under
  each condition, at each horizon position — the direct distributional
  analogue of `docs/PIVOT_PILOT_PROTOCOL.md` §2 measurement 7.

**Labeled as secondary, not primary, per the task brief's default**: none
of these is promoted to primary without a specific, stated reason, which
this protocol does not have.

## 7. Baseline-signal comparison (mandatory, not optional)

An unexplained false negative must not mean merely "R-KV missed it." For
every labeled block (evicted candidate and retained control alike), record
these baseline signals **at the moment of the compaction decision**,
wherever technically available without violating the FlashAttention/
deployability constraints already documented for this class of signal
(`docs/METHOD_NOVELTY_MATRIX.md` §4):

- R-KV's own base score `b_i` (§3 — already required, repeated here as a
  baseline for emphasis: an "unexplained" false negative must be low-`b_i`
  by construction, or the discovery question is vacuous).
- Accumulated or recent attention to the block (R-KV already logs an
  attention-based importance component internally — reuse it; do not
  recompute a materialized full attention matrix, which is the exact
  FlashAttention-incompatibility EpiKV/VaSE flag as a deployability bar).
- Recency / position (block start position, distance from the compaction
  boundary, distance from the trace start).
- Value-norm and key-norm (VaSE's and EpiKV's signal families,
  `docs/METHOD_NOVELTY_MATRIX.md` §4).
- Entropy and top-1/top-2 logit margin **at the block's own token
  positions during the natural run** — the mandatory confound control from
  `docs/METHOD_PIVOT_SPEC.md` §5a (arXiv:2606.00206). This is not optional:
  any claimed unexplained false negative must be checked against whether
  it sits at a locally uncertain (high-entropy, low-margin) decision point,
  which alone could explain a large `u_i` without any causal information
  loss.
- Hidden-state change across the block (EpiKV's signal family).
- Block length (fixed at 128 by construction here, so this signal is
  degenerate for this protocol's block choice — recorded anyway for
  provenance, not used as a discriminating feature).
- Thought-category / boundary labels: **not available** — no validated
  classifier exists in this repository (§1); recorded as `null`, not
  fabricated.
- LazyEviction-style recurrence information (whether the block's tokens
  reappear in the retained-attention pattern shortly after being
  candidate-marked) — available from R-KV's own bookkeeping without a new
  attention pass.
- ForesightKV-style future-attention oracle: **non-deployable by
  construction** (it requires the future context that does not exist at
  decision time); included only as a diagnostic upper bound, clearly
  labeled `oracle_non_deployable: true` in the schema, never compared to
  `b_i` as if it were a fair deployable baseline.
- Any other signal from VaSE/InfoKV/EpiKV/CAOTE/Learning to Evict directly
  computable from quantities R-KV or a lightweight added hook already
  produces; signals requiring separate model training (e.g., a from-scratch
  learned evictor) are out of scope for a discovery pilot and are not
  computed.

**Predeclared success/failure judgment (frozen before any GPU result):**

1. **Primary — rank-based, threshold-free:** Spearman rank correlation
   between `b_i` (and separately, each baseline signal) and `u_i`, computed
   over all labeled blocks (evicted candidates and retained controls
   pooled, since both populations get the same intervention-and-measure
   treatment per §5). A **low or near-zero correlation between every
   deployable baseline signal and `u_i`, while `u_i` itself shows
   non-trivial variance**, is the discovery-supporting pattern; a strong
   correlation with any one signal means that signal already explains the
   utility variation and no new operation is motivated.
2. **Primary — recall-style:** among the blocks R-KV actually evicted,
   what fraction of the empirical top-quartile-by-`u_i` blocks (defined
   over the *pooled* evicted+control sample, not a post-hoc per-example
   quantile, to avoid small-sample quantile instability) were evicted
   versus retained? A recall notably above the base eviction rate itself
   (i.e., R-KV disproportionately evicts high-utility blocks) is the
   discovery-supporting pattern. **No other quadrant threshold (e.g., an
   arbitrary "bottom 20% / top 20%" cut) is used** unless a specific reason
   is given and frozen in a future document before that run's results
   exist — the task brief explicitly forbids defaulting to that pattern.
3. **Dependence accounting:** blocks from the same example are not
   independent samples. Any correlation/recall statistic is reported both
   pooled and clustered-by-example (e.g., a cluster-robust or
   permutation-based standard error, or at minimum a by-example spread of
   the statistic) — this is a discovery pilot, so a full mixed-effects
   treatment is not required, but pretending within-example blocks are iid
   is not acceptable either.
4. **No statistical-equivalence or generality claim** is permitted from
   this `n` regardless of outcome — matching this repository's existing
   discipline for Stage 1B's `coarse_screen` (`docs/EXPERIMENT.md` §8) and
   applying it here even more strongly, since this pilot's `n` is smaller.

## 8. Sample size (feasibility-bounded — see `docs/B0_5_FEASIBILITY_AUDIT.md` §5 for the GPU cost derivation)

| Quantity | Value |
|---|---|
| `n_examples` | 12 |
| `events_per_example` | 3 |
| `evicted_per_event` | 2 |
| `control_per_event` | 2 |
| Total labeled blocks | 12 × 3 × 4 = 144 |
| `horizon` | 48 tokens |

These are small-pilot numbers, stated as such throughout (§7.4). They are
not powered for a publication-grade result; they are sized to fit the
4-GPU-hour ceiling with a stated safety margin (`docs/B0_5_FEASIBILITY_AUDIT.md`
§5) while producing enough labeled blocks for a first descriptive read of
the rank-correlation and recall statistics in §7.

## 9. Invalid-example and cap-hit handling

- Any example where the natural FullKV base answer is wrong is excluded
  from event sampling (mirrors this repository's existing both-correct
  eligibility discipline, loosened here since accuracy correctness is not
  this pilot's primary axis — but a wrong-answer trace is not a reasonable
  basis for a future-behavior utility measurement either).
- Any natural run (FullKV shadow replay or R-KV) that hits the generation
  token cap is excluded entirely from event sampling (§2 criterion 4).
- Any sampled event whose shadow-FullKV replay fails a schema/identity
  check against the R-KV run it is meant to shadow (same discipline as
  `kvcot.analysis.fixed_trace`'s existing identity validation) is dropped,
  logged, and counted in an attrition report — never silently skipped.
- `cap_hit_flag` is recorded per labeled block's horizon window
  specifically (distinct from the whole-run cap-hit check above): if the
  natural run's own continuation is shorter than `horizon` tokens past the
  event (should already be excluded by §2 criterion 3, but checked again
  at label-construction time as a hard assertion, not just a soft filter).

## 10. Deterministic seeds

All three frozen base seeds (`13, 42, 2026`, `configs/lock.yaml`) are
reused as the outer seed for `kvcot.utils.seeding.derive_seed`. Two new,
explicitly named sub-seed derivations are introduced (§2, §4) — both
literal-suffix extensions of the existing SHA-256 construction, never a
new, undocumented randomization source. No manual/hand-picked example,
event, or block selection occurs anywhere in this protocol.

## 11. Calibration / held-out separation

The primary statistics (§7, points 1-2) are rank-based and threshold-free,
so no calibration-fitted percentile or cutoff is needed for them, and no
calibration/held-out split is required for the primary analysis. If any
future report adds a quadrant-style (e.g., "top-k") diagnostic beyond §7's
predeclared statistics, it must reserve a held-out calibration subset
(e.g., 3 of the 12 examples) to fix that threshold **before** looking at
the remaining 9 examples' outcomes — stated here so a future implementer
does not skip it under time pressure.

## 12. Machine-readable future artifact schema

One JSON-lines record per labeled block:

```json
{
  "schema_version": "b0_5.v1",
  "example_id": "string",
  "dataset": "math500",
  "seed": 42,
  "compaction_event_id": "int (ordinal within the run)",
  "block_id": "string",
  "block_role": "evicted_candidate | retained_control",
  "block_start_token": "int",
  "block_end_token": "int",
  "distance_from_compaction": "int",
  "base_score_b_i": "float",
  "u_i_primary_reference_nll_delta": "float",
  "u_i_secondary": {
    "answer_token_nll_delta": "float | null",
    "kl_divergence_or_approx": "float",
    "kl_is_approximation": "bool",
    "top1_agreement_rate": "float",
    "target_logprob_present": "float",
    "target_logprob_absent": "float",
    "branching_marker_mass_present": "float",
    "branching_marker_mass_absent": "float"
  },
  "baseline_signals": {
    "attention_recent": "float | null",
    "value_norm": "float | null",
    "key_norm": "float | null",
    "recency_position": "int",
    "entropy_at_block": "float | null",
    "logit_margin_at_block": "float | null",
    "hidden_state_delta": "float | null",
    "recurrence_flag": "bool | null",
    "future_attention_oracle": "float | null",
    "oracle_non_deployable": true
  },
  "thought_category": null,
  "cap_hit_flag": "bool",
  "valid_flag": "bool",
  "invalid_reason": "string | null"
}
```

## 13. Preregistered stopping rules

1. Run exactly `n_examples=12`, `events_per_example=3`,
   `evicted_per_event=2`, `control_per_event=2` (§8) — no optional stopping
   based on how results look partway through.
2. Before spending GPU time on labeling, the natural-accuracy and
   realized-compression gates (`docs/B0_5_FEASIBILITY_AUDIT.md` §7) must
   pass on the natural runs alone. If either gate fails, **stop** — do not
   proceed to event sampling or labeling, and report the gate failure as
   the pilot's outcome (a valid negative result, per this repository's own
   precedent with `screen_valid: false`).
3. Measure real per-token throughput after the first example's natural
   FullKV and R-KV generations complete, recompute the projected total
   wall-clock time from that empirical rate, and **stop before proceeding**
   if the recomputed projection exceeds the authorized GPU-hour ceiling —
   the same discipline `docs/EXPERIMENT.md` §7 already requires before
   Stage 2 ("Throughput is measured and a Stage 2 wall-clock estimate is
   printed before Stage 2 is authorized").
4. No candidate/control block count, horizon, or event count is increased
   mid-run to chase significance.
