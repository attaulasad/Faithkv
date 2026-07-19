# B0 — Method pivot specification and novelty gate

Phase B0 artifact (2026-07-19). Companion files:
`docs/METHOD_NOVELTY_MATRIX.md` (adversarial matrix and per-candidate
verdicts), `docs/B0_SEARCH_LOG.md`, `docs/method_novelty_matrix.json`.
Documentation-only: no method implemented, no MATH-500 infrastructure
created, no GPU used, no model inference run, no frozen §1/§4/§8/§9 value
changed.

## 1. Executive verdict

**METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET.**

- M1 (residual causal-utility protection): **PARTIAL — INSUFFICIENT
  METHOD NOVELTY**
- M2 (interaction-aware dynamic rescue): **KILLED**
- M3 (faithfulness-constrained memory allocation): **PARTIAL —
  INSUFFICIENT METHOD NOVELTY**

No candidate reaches SURVIVES PROVISIONALLY. B1 is not authorized.

## 2. Why the old Phase B is cancelled

The original Phase B would have implemented MATH-500 infrastructure to run
the *diagnostic* (fixed-trace replay + early answering under R-KV) on a
longer-trace dataset. Phase A3 (2026-07-19, frozen artifacts
`docs/RELATED_WORK_MATRIX.md` / `docs/A3_SEARCH_LOG.md` /
`docs/related_work_matrix.json`) found the diagnostic primitive is prior
art — CASK's official evaluation code implements fixed-generated-trace,
teacher-forced, cache-policy-varying replay on reasoning models
(arXiv:2604.10900, released 2026-04-13), and early answering is Lanham et
al. 2023 (arXiv:2307.13702). **DIAGNOSTIC SURVIVAL VERDICT: DOES NOT
SURVIVE; PHASE B: BLOCKED — DIAGNOSTIC NOT NOVEL.** That cancellation is
permanent for the old diagnostic claim: rerunning the same combination on
MATH-500 would spend GPU budget on an application of known ingredients.
This B0 phase exists because the project's non-negotiable goal is a
genuinely new KV-cache compression *technique*, not an audit paper.

## 3. Valid conclusions from A2

The A2 failure atlas (50 committed protocol-v3 GSM8K pairs, retired b128
operating point) validly supports only:

1. At the retired, accuracy-invalid operating point, all 50 R-KV pairs
   first diverged from their FullKV counterparts strictly after the first
   compaction event (with a short pre-compaction observation window —
   mean 15.6 generated tokens — so this is weak temporal, not causal,
   evidence).
2. 41/50 pairs first diverged inside the nominal reasoning region; 9/50
   remained token-identical through the literal `</think>` boundary, and 3
   of those 9 flipped correct→wrong.
3. Realized retention (mean ≈0.3596, median ≈0.3485) and compaction counts
   (mean 3.9) at that operating point.
4. `</think>` is a token-format boundary; the atlas's own excerpts show
   mathematical derivation continuing past it.

## 4. Conclusions A2 cannot support

- That any specific evicted token or block *caused* any failure.
- That causal false negatives (§5) exist, at this or any operating point.
- That a causal-importance or causal-rescue method would have prevented
  the observed failures.
- That the reasoning-region/post-think classification is semantic.
- Anything about an accuracy-neutral operating point — A2's operating
  point failed the accuracy gate by 40pp.

A2 motivates *looking for* the §5 failure mode. It is not evidence the
failure mode exists. Any future document citing A2 as causal evidence is
in error.

## 5. Exact causal-false-negative hypothesis

Prospective, untested hypothesis a future method would target:

> At an accuracy-neutral, meaningfully compressed operating point (both
> gates predeclared and passed), a deployable KV importance score `b`
> assigns low rank — low enough to be evicted or heavily compressed — to
> some reasoning states whose KV information has high counterfactual
> utility `u` for future reasoning or answer formation, where `u` is
> measured by a controlled KV-only intervention (ablate the state, hold
> everything else fixed, measure a preregistered future-behavior quantity
> such as future-token NLL, answer-token NLL, or continuation-distribution
> divergence).

A **causal false negative** is a state (token or block) satisfying all of:
(1) ranked safe-to-remove by the deployable score; (2) actually removed or
heavily compressed at the operating point; (3) high `u` under the
controlled intervention; (4) whose removal changes future reasoning or
answer behavior; (5) at an operating point that passed the predeclared
natural-accuracy and realized-compression gates.

Status: **existence unproven.** No experiment in this repository or in the
reviewed literature establishes (1)–(5) jointly. LazyEviction's "Token
Importance Recurrence" and VaSE's catastrophic value-eviction findings are
adjacent observational evidence that *temporal* and *magnitude* false
negatives exist for specific scores, at aggregate level; neither is the
per-state causal quantity defined here.

## 6. M1 formal specification — residual causal-utility protection

- Let `b_i ∈ ℝ` be the deployable base score for KV state/block `i`
  (e.g., R-KV's mixed attention/redundancy score at a compaction event).
- Offline label: `u_i = Q(model, cache) − Q(model, cache \ {i})` where `Q`
  is a preregistered future-behavior quantity (future-reference NLL,
  answer-token NLL, or continuation-distribution divergence) and
  `cache \ {i}` denotes KV-only ablation of `i` with the visible token
  sequence held fixed.
- Residual target: `r_i = u_i − E[u_i | b_i, s_i]` where `s_i` is a fixed
  vector of known cheap signals (value norm, entropy, position, hidden-
  state delta), and the conditional expectation is fit on calibration
  data.
- Learned component: lightweight predictor `r̂_i = g_θ(features_i)`
  trained offline on `(features_i, r_i)`.
- Online operation: base policy evicts as usual into budget `B − B_rescue`;
  the rescue partition retains the top-`B_rescue` states by `r̂_i` among
  states the base policy would evict; total physical bytes = `B` exactly.

**Verdict: PARTIAL — INSUFFICIENT METHOD NOVELTY** (full argument:
`docs/METHOD_NOVELTY_MATRIX.md` §8–§9, §12). Component-wise prior art:
ablation supervision → ArborKV/ThinKV/ForesightKV; residual architecture →
IntentKV (zero-init residual head "for cases the rule scorer misses");
score-correction-at-deployment → CAOTE/Adaptive Filtering; protected
partition → CASK/VaSE/2605.18053. The residual parameterization is
expressively equivalent to giving `b_i` to an absolute predictor.

## 7. M2 formal specification — interaction-aware dynamic rescue

- After compaction event `t` produces retained set `C_t`, recompute for
  borderline states `i` (those within a margin of the eviction threshold)
  a conditional risk `r̂_i(C_t)` — the predicted residual utility of `i`
  *given the current compressed cache* — and rescue states whose
  conditional risk rose relative to their unconditional estimate; same
  fixed byte budget.

**Verdict: KILLED.** ForesightKV's GRPO stage already formulates
sequential eviction as an MDP whose state is "the current remaining KV
cache at step t"; Neural Garbage Collection conditions learned eviction on
cache state for reasoning models; the base R-KV rescores at every
compaction over the retained set. The only non-published word in M2 is
"residual," which is M1's and falls with it.

## 8. M3 formal specification — faithfulness-constrained memory allocation

- Decision variables: per-(layer, head, segment) budgets `B_ℓ,h,g` and/or
  precisions.
- Objective: minimize total bytes, or minimize continuation distortion at
  fixed bytes.
- Constraints: (a) natural-accuracy gate (predeclared ceiling on paired
  accuracy drop); (b) realized physical byte budget `Σ B_ℓ,h,g ≤ B`;
  (c) bounded degradation of a preregistered causal reasoning-dependence
  proxy `D` (e.g., an omitted-suffix dependence quantity) relative to
  FullKV: `|D_compressed − D_full| ≤ ε`.

**Verdict: PARTIAL — INSUFFICIENT METHOD NOVELTY.** The allocation
machinery under (a)+(b) is ReasonAlloc/LKV/Ada-KV-lineage prior art;
constraint (c) is a new metric in an old optimizer, which the decision
standard explicitly rules insufficient, and the metric family itself is
established (Lanham lineage, per the frozen A3 matrix).

## 9. Candidate pseudocode (implementation-independent; recorded for the archive)

```
# M1 (offline)
for trace in calibration_traces:                 # natural FullKV traces
    for block i in trace:
        u[i] = Q(replay(trace))  -  Q(replay(trace, kv_ablate=i))
fit E_hat(u | b, s)   on {(b[i], s[i], u[i])}
fit g_theta(features) on residuals r[i] = u[i] - E_hat(u[i] | b[i], s[i])

# M1 (online, per compaction event)
keep_base   = base_policy.select(cache, budget = B - B_rescue)
candidates  = cache - keep_base                  # states base would evict
keep_rescue = top_k(candidates, key = g_theta, k = B_rescue)
cache       = keep_base + keep_rescue            # physical bytes == B

# M2 (online) — as M1, then after each compaction:
for i in borderline(candidates):
    r_cond[i] = g_theta(features_i, summary(C_t))   # cache-conditional
swap rescue members where r_cond rank differs      # budget unchanged

# M3 (offline search / online schedule)
argmin_{B_lhg} distortion  s.t.  accuracy_gate, sum(B_lhg) <= B,
                                 |D_compressed - D_full| <= eps
```

None of this is implemented; nothing in `src/` changed in B0.

## 10. Offline and online components

| | Offline | Online |
|---|---|---|
| M1 | KV-ablation labeling runs; conditional-expectation fit; residual predictor training | base policy + `g_θ` scoring of eviction candidates + rescue top-k |
| M2 | M1's, plus cache-summary featurization | M1's, plus borderline re-scoring after each compaction |
| M3 | constrained search for allocation schedule; proxy measurement runs | fixed schedule lookup (cheap) |

## 11. Actual-memory accounting

All candidates were specified against **realized physical KV bytes**,
never nominal token budgets, consistent with this repository's existing
discipline (`RetentionSummary.instantaneous_retention_ratio =
physical_cache_slots / fullkv_equivalent_slots`; §9 naming rule — no
percentage-named conditions). M1/M2: rescue partition bytes are carved out
of, not added to, the baseline budget (`B_rescue` bytes fewer for the base
policy), so any comparison is at matched physical memory including
predictor state (θ is model-external and constant). M3: `Σ B_ℓ,h,g ≤ B`
counts bytes after quantization. Any future method phase must also count
auxiliary state (e.g., M2's cache summary) against the budget.

## 12. Expected computational overhead

- M1 offline: one KV-ablation replay per labeled block per trace — the
  dominant cost, O(blocks × trace length) forward passes per trace;
  this is why any pilot must be small and gated.
- M1 online: one linear/MLP evaluation per eviction candidate per
  compaction — negligible next to a decode step.
- M2 online: adds borderline re-scoring per compaction — still small, but
  the featurization of `C_t` must avoid materializing attention (else it
  breaks the FlashAttention constraint that EpiKV/VaSE document as the
  deployability bar).
- M3: offline search cost; online negligible.

Recorded for completeness; overhead was not a factor in any verdict.

## 13. Required future evidence (for any successor candidate)

1. **Existence evidence:** per-state controlled KV-ablation measurements
   at an accuracy-neutral, meaningfully compressed operating point showing
   states that are (base-score-low, utility-high) at a rate materially
   above chance — the §5 hypothesis itself.
2. **Predictability evidence:** those states are predictable from
   deployable features better than every cheap proxy in
   `docs/METHOD_NOVELTY_MATRIX.md` §4's table (value norm, entropy,
   hidden-state delta, recurrence statistics) — otherwise the cheap proxy
   *is* the method and it is already published.
3. **Operating-point evidence:** a (model, dataset, budget) triple passing
   both predeclared gates; GSM8K + R1-Distill-1.5B + b128 is retired and
   cannot be it.
4. **Baseline evidence:** matched-byte comparisons against, at minimum,
   R-KV, VaSE, and one learned evictor (ForesightKV or Learning to Evict).

None of this evidence exists today; none may be collected without a
separately authorized phase.

## 14. Candidate-specific falsification tests (design only — none run)

- **M1 motivating-failure test:** at a gate-passing operating point, label
  `u_i` for a sample of base-policy-evicted states; if the joint
  distribution of `(b_i, u_i)` shows no excess mass in the
  (low `b`, high `u`) quadrant beyond what value-norm/entropy proxies
  already flag, the motivating failure is absent and M1-style methods are
  pointless regardless of novelty.
- **M2 motivating-failure test:** measure `u_i` for the same states before
  and after compaction events; if `u_i` rankings are stable across cache
  states (high rank correlation), interaction effects are negligible and
  M2's premise is false.
- **M3 motivating-failure test:** at matched bytes and matched accuracy,
  measure whether the causal-dependence proxy `D` actually varies across
  allocation schedules; if `D` is flat, the constraint never binds and M3
  reduces to existing allocation.

These tests are cheap relative to a full method build and would each be a
legitimate *first* GPU experiment for a future authorized phase — but
running them is not authorized by B0.

## 15. Claims explicitly forbidden

This repository must not claim, now or in any future writeup, absent new
evidence and a new novelty gate:

- that causal false negatives exist (untested hypothesis, §5);
- that A2 provides causal evidence for them (it cannot, §4);
- that M1/M2/M3 or any variant works, beats baselines, or preserves
  accuracy (nothing was run);
- that residual prediction over a base compressor is a novel algorithmic
  operation (IntentKV; `docs/METHOD_NOVELTY_MATRIX.md` §8.4);
- that cache-state-conditional importance is a novel operation
  (ForesightKV, Neural GC);
- that counterfactual thought/block ablation supervision is a novel
  operation (ThinKV, ArborKV);
- that a protected reasoning-core partition is a novel operation (CASK,
  VaSE, arXiv:2605.18053);
- the A3-forbidden diagnostic-novelty claims (unchanged, frozen);
- any condition named by a retention percentage (§9 rule, unchanged).

## 16. B0 verdict

**METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET.**

M1: PARTIAL — INSUFFICIENT METHOD NOVELTY. M2: KILLED. M3: PARTIAL —
INSUFFICIENT METHOD NOVELTY. Per the predeclared standard, PROCEED
requires at least one SURVIVES PROVISIONALLY; there is none. The negative
outcome is recorded without softening, and no fourth candidate was
invented to avoid it.

## 17. What B1 would do (not started, and not currently permitted)

Recorded only so a future reader knows what was *not* authorized: had B0
passed, B1 would have been a paper-only design phase producing (a) a
preregistered existence/falsification protocol for §14's motivating-
failure test on a new candidate operating point, (b) a baseline plan, and
(c) a GPU cost envelope — all still without implementation or GPU use,
each requiring separate authorization. Because B0 is BLOCKED, the only
permitted next activity is further method design that produces a candidate
whose central cache operation is absent from
`docs/METHOD_NOVELTY_MATRIX.md` §5's operation table as a class, followed
by a fresh B0-style gate re-run against the then-current literature.
