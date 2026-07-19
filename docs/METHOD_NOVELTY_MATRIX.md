# B0 — Method-novelty matrix and adversarial kill-check

Phase B0 artifact (2026-07-19). Companion files: `docs/METHOD_PIVOT_SPEC.md`
(candidate specifications and verdict), `docs/B0_SEARCH_LOG.md` (search
methodology), `docs/method_novelty_matrix.json` (machine-readable records).
This document does not modify the frozen A3 artifacts
(`docs/RELATED_WORK_MATRIX.md`, `docs/A3_SEARCH_LOG.md`,
`docs/related_work_matrix.json`).

Roles applied: strict senior LLM-inference researcher; adversarial ICLR
reviewer; research engineer preserving historical evidence. The working
hypothesis of this whole document is the hostile reviewer's summary —
"this is just ArborKV-style KV ablation used to train a ThinKV/CASK-style
protected-cache policy on top of R-KV" — treated as the claim to test, not
a strawman.

## 1. Search methodology

Cutoff 2026-07-19. Engines: general web search (US) + direct arXiv
abstract/HTML/PDF fetches + one official GitHub repository fetch (CASK) +
reuse of the A3 matrix's already-verified records (R-KV, KIVI, ShotKV,
LazyEviction, VaSE, CASK evaluation-code inspection). Full query log,
screening tiers, and evidence-quality limitations: `docs/B0_SEARCH_LOG.md`.
Key limitation, stated up front: fetched pages were read through a
fetch-and-summarize tool that returns verbatim quotes; "full text
inspected" below means "full HTML text fetched and interrogated with
targeted questions, quotes captured," not a human-style cover-to-cover
read. Where a conclusion depends on a detail that tool did not quote, the
field is UNKNOWN.

## 2. Candidate definitions

- **M1 — Residual causal-utility protection.** Offline: controlled KV-only
  ablation produces per-state counterfactual utility labels `u_i`; a
  lightweight predictor is trained on the residual `r_i = u_i − E[u_i |
  b_i, baseline signals]`, where `b_i` is an existing deployable
  compression score. Online: the base eviction policy runs unchanged
  except that a strictly budgeted rescue partition retains the states with
  the highest predicted residual, at the same total physical KV bytes.
- **M2 — Interaction-aware dynamic rescue.** After each compaction,
  recompute causal-residual risk for borderline states conditional on the
  current compressed cache (importance is not frozen at token creation),
  rescuing states whose marginal utility rose because of what was already
  evicted; fixed byte budget.
- **M3 — Faithfulness-constrained memory allocation.** Allocate
  token/layer/head memory under a fixed physical byte budget subject to a
  bounded-degradation constraint on a preregistered causal
  reasoning-dependence proxy, plus a natural-accuracy gate.

## 3. Main paper-by-candidate matrix

Overlap levels: HIGH / MEDIUM / LOW / NONE / UNKNOWN. Full per-paper
fields in `docs/method_novelty_matrix.json`.

| Paper | arXiv | Evidence tier | M1 | M2 | M3 |
|---|---|---|---|---|---|
| ArborKV | 2605.22106 | full text (fetch) | **HIGH** | MEDIUM | MEDIUM |
| IntentKV | 2606.09916 | full text (fetch) | **HIGH** | LOW | NONE |
| ForesightKV | 2602.03203 | full text (fetch) | **HIGH** | **HIGH** | LOW |
| ThinKV v2 | 2510.01290 | full text (fetch) | **HIGH** | MEDIUM | MEDIUM |
| CASK | 2604.10900 | code + abstract | MEDIUM | UNKNOWN | LOW |
| Learning to Evict (KVP) | 2602.10238 | abstract | **HIGH** | UNKNOWN | LOW |
| Adaptive Filtering | 2607.13205 | full text (fetch) | **HIGH** (conceptual) | NONE | LOW |
| CAOTE | 2504.14051 | abstract | MEDIUM | MEDIUM | NONE |
| VaSE | 2606.03928 | abstract + A3 | MEDIUM | LOW | NONE |
| InfoKV | 2606.26875 | abstract | MEDIUM | LOW | NONE |
| EpiKV | 2606.26472 | abstract | LOW | LOW | NONE |
| Protection Is (Nearly) All You Need | 2605.18053 | abstract | MEDIUM (effect size) | LOW | LOW |
| Functional importance | 2601.03066 | abstract | MEDIUM | MEDIUM | NONE |
| Thought Anchors | 2506.19143 | abstract | MEDIUM | LOW | NONE |
| R-KV | 2505.24133 | code (submodule) | MEDIUM (base) | **HIGH** | NONE |
| KIVI | 2402.02750 | A3 reuse | NONE | NONE | LOW |
| ShotKV | 2502.01941 | A3 reuse | LOW | NONE | LOW |
| LazyEviction | 2506.15969 | A3 reuse | MEDIUM | MEDIUM | NONE |
| ReasonAlloc | 2606.11164 | abstract | NONE | LOW | **HIGH** |
| Neural Garbage Collection | 2604.18002 | abstract | MEDIUM | **HIGH** | NONE |
| LookaheadKV | 2603.10899 | abstract | MEDIUM | UNKNOWN | NONE |
| Lookahead Q-Cache | 2505.20334 | abstract | LOW | MEDIUM | NONE |
| Locret | 2410.01805 | search only | MEDIUM | UNKNOWN | NONE |
| LKV | 2605.06676 | search only | LOW | UNKNOWN | MEDIUM |
| IndexMem | 2605.25475 | search only | MEDIUM | UNKNOWN | NONE |

## 3a. Adjacent-compression threat (weight/activation/KV quantization, not KV eviction)

Not part of the M1/M2/M3 overlap table above (different compression mechanism
entirely — quantization, not eviction), but load-bearing for M1's motivating
hypothesis (`docs/METHOD_PIVOT_SPEC.md` §5) and recorded here per the same
adversarial discipline. Verified 2026-07-19 by direct fetch of the arXiv
abstract and HTML full text (`arxiv.org/abs/2606.00206`,
`arxiv.org/html/2606.00206`); quotes captured, not a full cover-to-cover
read — same limitation as every other B0 record (§1).

| Field | Entry |
|---|---|
| Paper | Quantized Reasoning Models Think They Need to Think Longer, but They Do Not |
| arXiv | 2606.00206 |
| Compression | Weight PTQ (GPTQ, AWQ, 3-/4-bit); FlatQuant also quantizes activations and KV (W4A4KV4, W8A8KV8) |
| Reasoning model | DeepSeek-R1-Distill-Qwen 1.5B/7B/14B, DeepSeek-R1-Distill-Llama 8B, QwQ-32B |
| Same-prefix control | Yes — "we run both models on the same MATH-500 prompts under identical generation prefixes to isolate the effects of quantization" |
| Diagnostic | Token-level KL divergence between quantized and full-precision output distributions; next-token entropy (Spearman ρ=0.92 with KL); top-1/top-2 logit margin; overthinking markers ("wait", "but", "alternatively") |
| Failure | Correct intermediate answer reached, then abandoned — "in up to 52% of the quantized models' failures, models reach the right answer in intermediate reasoning steps but do not output it as a final answer" |
| Method | Training-free fixed logit penalty on a curated set of overthinking/branching markers, cutting CoT length 12-23% |
| Cache intervention | No KV-entry rescue or eviction — the fix operates on output logits, not on cache contents; FlatQuant quantizes KV cache values but performs no per-state rescue |
| Direct overlap | Medium diagnostic overlap (KL/entropy/logit-margin position-level diagnostic is close kin to a controlled-intervention causal-utility measurement); low method overlap (no cache operation of any kind) |
| Threat to FaithKV | Entropy-amplified branching may explain some compression failures independent of any per-state KV-information loss — a confound M1's motivating hypothesis (§5 causal-false-negative existence) must rule out, not just a competing method |
| Required response | Any future rescue-utility claim must be shown to hold beyond entropy, logit-margin, and branching-marker controls — see the added threat paragraph in `docs/METHOD_PIVOT_SPEC.md` §5a |

This record does not change any M1/M2/M3 verdict in §12 — it is a
different compression family (quantization, not eviction) and does not
implement a KV-cache operation, so it cannot itself kill or corroborate a
cache-operation novelty claim. Its relevance is entirely to §5's
*measurement validity*: a future discovery pilot (`docs/B0_5_DISCOVERY_PROTOCOL.md`)
must separate "the compressor removed causally necessary information" from
"the compressor's perturbation nudged a locally uncertain, high-entropy
token decision and triggered overthinking/branching that would have
happened from any sufficiently large perturbation, KV or otherwise."

## 4. Signal comparison

| Signal | Already owned by | Deployable | FlashAttention-safe | Reasoning-specific evidence |
|---|---|---|---|---|
| Value norm / magnitude | VaSE | yes | yes (VaSE claims FA2) | yes (Qwen3, beats R-KV) |
| Stochastic diversity | VaSE | yes | yes | yes |
| Entropy / predictive uncertainty | InfoKV | yes | plausible | yes (DeepSeek-R1) |
| Forward influence on future context | InfoKV | yes | plausible | yes |
| Hidden-state change | EpiKV | yes | yes (explicit) | yes (MATH-500/AIME) |
| Key/value variation | EpiKV, MomentKV-family (screened) | yes | yes | partial |
| Attention-output error of eviction | CAOTE (closed form) | yes | attention-score-dependent | partial |
| Future attention (oracle) | ForesightKV Golden Eviction | offline label only | n/a | yes (R1-Distill-7B) |
| Learned future utility (K/V inputs) | Learning to Evict; Locret; LookaheadKV | yes | yes (no attention matrix needed) | limited |
| Counterfactual segment-ablation KL | ThinKV (offline, per thought type) | as offline calibration | n/a | yes |
| Leave-one-out KV-block accuracy delta | ArborKV MSVE labels | as offline calibration | n/a | yes (tree reasoning) |
| Post-eviction loss increase | ForesightKV RL reward | as training signal | n/a | yes |
| Residual correction of a rule score | IntentKV | yes | yes | no (agent tool-use) |
| Role/structural correction of a base score | Adaptive Filtering | yes | yes | no (structured QA) |

Conclusion of the signal comparison: **residual causal prediction does not
introduce a signal type absent from this table — it combines the
counterfactual-ablation label family (ArborKV/ThinKV/ForesightKV) with the
residual-correction architecture family (IntentKV/CAOTE/Adaptive
Filtering).** Its only unclaimed content is the specific pairing, i.e., an
intersection.

## 5. Cache-operation comparison

| Operation | Already owned by |
|---|---|
| Protected core / reserve partition inside a capped budget | CASK (core), VaSE (large-value protection), Protection-2605.18053 (10% boundary reserve), ForesightKV (recency window), IntentKV (forced set), R-KV (recent window) |
| Meta-correction layered on an arbitrary base eviction policy | CAOTE (explicitly "meta-heuristic ... with any token eviction method"), Adaptive Filtering (role layer over H2O/SnapKV) |
| Additive learned residual over a rule score, zero-init to recover the rule | IntentKV |
| Eviction decisions conditional on the current remaining cache | ForesightKV (MDP state = current remaining cache), Neural GC, CAOTE (implicitly), R-KV/H2O lineage (per-compaction rescoring) |
| Refresh of thought/category assignment during decode | ThinKV (τ=128 interval) |
| Delayed eviction to catch late-blooming importance | LazyEviction |
| Rescue/restoration of already-evicted state | ArborKV (lazy rehydration on tree revisit), IndexMem (latent-memory residual readouts), CacheFlow-family (systems-level restoration) |
| Budget allocation across layers/heads/segments under a cap | ReasonAlloc, LKV, Ada-KV lineage, Adaptive Filtering (role buckets) |

Every cache operation named in M1/M2/M3 appears in this table.

## 6. Offline-supervision comparison

| Supervision | Already owned by | Counterfactual? | Per-instance? |
|---|---|---|---|
| Leave-one-out KV-block zeroing → answer-accuracy delta | ArborKV | yes (KV-level) | labels per block; deployed predictor generalizes |
| Segment ablation → answer-distribution KL (50 rollouts) | ThinKV | yes (segment-level) | no — collapsed to static per-type ranks |
| Post-eviction loss increase on impacted tokens | ForesightKV (reward) | yes (set-level) | policy-conditional |
| Future-attention oracle (Golden Eviction) | ForesightKV (distillation) | no (observational) | yes |
| Future-use substring labels | IntentKV | no | yes |
| Task-reward policy gradient | Neural GC | no (end-to-end) | policy |
| Role-ban accuracy counterfactual | Adaptive Filtering | yes (role-level) | no — static role shares |
| Long-context SFT-derived CIS | Locret | unverified | yes |
| Greedy likelihood-preserving deletion ranks | 2601.03066 | yes (text-level) | yes |
| Counterfactual sentence resampling | Thought Anchors | yes (text-level) | yes |

The specific cell "per-state KV-only ablation labels, residualized against
a fixed base compressor's score, on linear reasoning traces" is empty. But
the row and column ingredients all exist, and A3's own predefined rule —
an unstudied intersection of known components is not automatically a new
method — was the exact ground on which this repository's N3 was ruled
insufficient. Applying a weaker standard to our own method candidate than
we applied to our own diagnostic would be motivated reasoning.

## 7. Dynamic-update comparison

| Behavior | Already owned by |
|---|---|
| Rescore at every compaction over the retained set | R-KV itself (audited in `docs/UPSTREAM_AUDIT.md`), H2O lineage |
| Sequential eviction as an MDP over the current cache | ForesightKV (GRPO stage), Neural GC |
| Interval-based category refresh | ThinKV |
| Lagged eviction for importance recurrence | LazyEviction |
| Pseudo-future-query rescoring | Lookahead Q-Cache |
| Deletion-conditional importance (text level) | Greedy pruning (2601.03066) |

M2's framing — "importance conditional on the current compressed cache,
rather than a score frozen when the token was created" — describes
published behavior (ForesightKV's MDP state is literally "the current
remaining KV cache at step t"; Neural GC conditions on cache state; the
base R-KV already refreshes scores per compaction). What no paper does is
refresh a *causal-residual* estimate specifically — but that estimate is
M1's, and M2 has no independent content once M1 falls.

## 8. Detailed threat memos

### 8.1 ThinKV (arXiv:2510.01290, v2 2026-05-07) — required analysis

- **Thought identification:** attention-sparsity signatures against
  KDE-calibrated thresholds; three types (reasoning/execution/transition);
  refreshed every τ=128 decode steps.
- **Counterfactual importance:** yes — "we measure the counterfactual
  importance of each segment Y_i by computing the KL divergence between
  A's distributions obtained with and without Y_i, averaged over 50
  rollouts." Answer distributions are compared directly.
- **KV or visible tokens suppressed:** segments ablated for the offline
  measurement; the deployed policy operates on KV states (quantization +
  eviction).
- **Measurements determine the deployed policy:** yes — the offline KL
  hierarchy (R ≫ E ≫ T) fixes the deployed rank map ρ and precision map ψ
  (R→8-bit, E→4-bit, T→2-bit) and the eviction ordering.
- **Static or instance-specific:** static per thought type at deployment.
  Instance-specific outliers (high-importance backtracking T-thoughts
  whose removal "causes the model to loop endlessly") are handled by a
  uniform minimum-retention floor (ℛ_min = 4), not by per-instance
  prediction.
- **Rankings refreshed:** category assignment refreshed at τ=128
  intervals; **no re-scoring after eviction; not conditional on the
  compressed cache**.
- **Residual over another compressor:** no.
- **Protects causal false negatives under matched memory:** partially and
  bluntly — the minimum-retention floor is exactly a crude
  false-negative guard, motivated by an observed catastrophic ablation
  outcome.
- **Consequence:** M1 cannot claim the counterfactual-ablation→policy
  pipeline, and cannot claim the observation that some low-ranked states
  are catastrophic to evict. What ThinKV does not do: per-instance
  prediction, residualization against a base score, budgeted rescue.
  Per instruction, M1 is NOT credited novelty merely for finer granularity.

### 8.2 ArborKV (arXiv:2605.22106) — required analysis

- **What is masked:** the KV cache of each closed thought block,
  individually zeroed (leave-one-out), on full-retention trajectories.
- **Masking applied to KV blocks directly:** yes.
- **Leave-one-out:** yes, explicitly.
- **Supervision:** change in answer accuracy.
- **Trajectory count:** not captured in the fetched text (UNKNOWN).
- **What is learned:** MSVE — a linear model σ(θᵀφ_i) over three features
  (controller search value, boundary next-token entropy, accumulated
  attention), calibrated offline; θ fixed at inference.
- **Used online:** yes, at policy update events (block boundary, active-
  leaf transition, memory pressure).
- **Tree-specific:** yes, intrinsically — distance-to-active-leaf decay
  e^(−λΔ·Δ_i) and rehydration-on-backtrack have no meaning in linear CoT
  decode.
- **Absolute or residual:** absolute.
- **Conditioned on previous evictions:** no — features are block content
  plus tree topology.
- **Does it kill M1 or M2?** It kills M1's supervision pipeline claim
  ("offline KV ablation supervises an online importance predictor" is
  published, on reasoning workloads, at matched memory). It does not kill
  M2 (not eviction-conditional) — ForesightKV does that.

### 8.3 Functional-importance work — required analysis

| Work | Removes visible tokens | Removes KV states | Scores answer likelihood | Joint likelihood | Dynamic recomputation | Trains a deployable compressor | Identifies false negatives of an existing KV score |
|---|---|---|---|---|---|---|---|
| Greedy pruning (2601.03066) | yes | no | yes (likelihood objective) | objective-dependent | yes (greedy iterative) | no (diagnostic + distillation data) | no |
| Thought Anchors (2506.19143) | resamples sentences | no | answer-outcome via resampling | no | no | no | no |
| Lanham et al. (2307.13702, A3) | truncates/perturbs CoT text | no | answer change | no | no | no | no |
| Thinking Drafts (2505.13774, A3) | edits draft steps | no | draft-answer consistency | partial | no | no | no |
| ThinKV (offline stage) | segment ablation | KV-relevant | answer-distribution KL | no | no | yes (policy calibration) | no |
| ArborKV (calibration) | no | yes (block zeroing) | answer accuracy | no | no | yes (MSVE) | no |

The last column is the only empty one, and it is exactly the column
Adaptive Filtering (2607.13205) fills at role granularity for false
*positives*: a controlled suppression experiment diagnosing an existing
deployable score's systematic error, driving a deployed correction. The
directional variant (false negatives, per-state, causal labels) is an
unfilled cell, not an unfilled operation class.

### 8.4 KVP / learned eviction (Learning to Evict, 2602.10238) — required analysis

- **"Future utility":** predicted usefulness of a cached token for future
  decoding steps, derived from pre-computed generation traces.
- **Reward:** "a holistic reward, derived from future utility, that
  evaluates the quality of the ranking across all cache budgets"
  (budget-agnostic).
- **Inputs:** "only key and value vectors" — deployable without the
  attention matrix.
- **Offline traces:** yes, explicitly.
- **Lightweight deployment:** yes — per-head agents, no LLM modification.
- **Static or conditional rankings:** UNKNOWN at abstract level.
- **Would a residual-over-baseline reformulation be substantive or
  cosmetic?** Cosmetic. A learned ranker that receives the baseline score
  `b_i` as an input feature can represent any residual function of it;
  IntentKV demonstrates the explicit-residual variant is already published
  practice, including the zero-initialization trick that makes the learned
  part a strict correction. The reformulation changes the parameterization,
  not the algorithm.

### 8.5 CASK (2604.10900) — required analysis

- **Core/scratch selection:** signal unspecified in every source inspected
  (abstract, README, A3 code inspection of the evaluation harness);
  UNKNOWN.
- **Is the core already a rescue partition?** It is a budgeted protected
  partition of reasoning states "that anchors answer formation and
  intermediate state." It is not (per available evidence) selected by
  counterfactual supervision or targeted at another policy's false
  negatives — but at reviewer altitude, "a reserved region of
  reasoning-critical KV states protected from compression at matched
  budget" is CASK's headline shape, and M1's rescue partition would be
  described as a differently-selected core.
- **Scratch consolidation vs eviction:** representative-based
  consolidation (merging), not pure eviction; two-stage prefix-eviction /
  decode-consolidation.
- **Scores refreshed:** UNKNOWN.
- **Safety diagnostics vs residual utility:** CASK's "lost representative
  mass" and "kappa-dispersion" are geometric preservation diagnostics, not
  behavioral-utility measures; they do not implement residual causal
  utility.
- **Does M1 merely rename protected-core selection?** The selection *rule*
  differs (learned causal residual vs unspecified/heuristic), but the
  cache *operation* (maintain a protected subset inside the budget) is the
  same. M1's operation-level novelty therefore rests entirely on the
  selection rule — which §8.1–8.4 show is itself assembled from published
  parts.

### 8.6 VaSE, InfoKV, EpiKV, CAOTE — signal-level analysis

See §4 table. The four papers jointly occupy: value norm (VaSE),
stochastic diversity (VaSE), entropy/uncertainty (InfoKV), forward
influence (InfoKV), hidden-state change (EpiKV), attention-output error
with value integration (CAOTE, closed form, meta-heuristic over arbitrary
base policies). All are training-free and deployable; VaSE and EpiKV
document FlashAttention compatibility; VaSE beats R-KV on Qwen3 reasoning
models at 4× compression. **Residual causal prediction would provide
information beyond these signals only if measured counterfactual utility
diverges from every cheap proxy in a learnable way — an empirical
hypothesis nobody has verified for linear reasoning traces, not an
algorithmic novelty.** The honest statement: M1 combines the ablation
label family with a corrective architecture; it does not introduce a new
signal class.

## 9. Reviewer attack for M1

"M1 is IntentKV's zero-initialized residual head, retrained with
ArborKV-style leave-one-out KV-ablation labels (or ForesightKV-style
post-eviction loss labels), bolted onto R-KV as the base score, with the
correction spent through a CASK-style protected partition. Every named
component cites to a specific published method; the contribution is a
recombination plus a domain transfer (tree→linear, agent-history→reasoning
trace). The residual parameterization is expressively equivalent to
feeding b_i to an absolute predictor, which Learning to Evict and
ForesightKV already deploy. The budgeted rescue reserve is a trust-region
deployment constraint, not an algorithm. And Protection-2605.18053 warns
that scoring refinements are empirically second-order once structural
protection is in place."

Defense available: no inspected paper performs per-state causal-ablation
labeling *of a fixed base compressor's errors* with a *bounded-override*
deployment. Assessment of the defense: true, and insufficient — it is a
new cell in a dense grid, exactly the situation A3's predefined rule
classified as an application, not a method.

## 10. Reviewer attack for M2

"M2 is ForesightKV's MDP (state = current remaining cache) with the word
'causal-residual' inserted, or equivalently Neural GC's cache-conditional
policy restricted to borderline states. The base method R-KV already
recomputes its scores at every compaction over the retained set, so
'rankings go stale after eviction' is already partially handled by the
thing being improved; LazyEviction handles delayed importance;
Lookahead Q-Cache handles future-query staleness. The only new word is
'residual,' which imports M1's problems."

Defense available: none independent of M1. If M1's residual signal is not
novel, conditioning that signal on the cache state is an implementation
schedule, and the unconditional-importance gap M2 targets is already
closed by ForesightKV/NGC at the operation level.

## 11. Reviewer attack for M3

"M3 is constrained optimization — the oldest framing in the allocation
literature (Ada-KV lineage, LKV, ReasonAlloc, which already does
hierarchical decoding-time allocation for reasoning models under a global
cap) — with one constraint term swapped for a faithfulness proxy whose
own component metrics (Lanham-style dependence, Thinking-Drafts-style
draft reliance, RFEval-style causal influence) are established prior art
per the frozen A3 matrix. A new constraint in an old optimizer is a new
objective, not a new cache method; the candidate description itself
concedes this risk."

Defense available: none found. The empty cell (nobody constrains
allocation on a causal-dependence proxy) is real but is a metric choice.

## 12. Candidate verdicts

- **M1 — Residual causal-utility protection: PARTIAL — INSUFFICIENT
  METHOD NOVELTY.** Each central sub-operation is individually published:
  offline leave-one-out KV-block ablation supervising a lightweight online
  estimator (ArborKV); counterfactual segment ablation with
  answer-distribution comparison determining a deployed reasoning
  compression policy (ThinKV); learned residual correction of a deployable
  retention heuristic's misses, zero-initialized to recover the base rule
  (IntentKV); intervention-diagnosed systematic-error correction of an
  existing score at matched budget (Adaptive Filtering); counterfactual
  post-eviction degradation inside the training signal of a deployed
  reasoning evictor (ForesightKV); budgeted protected partitions (CASK,
  VaSE, 2605.18053). The residual parameterization is expressively
  cosmetic (§8.4). The remaining delta — per-state causal labels of a
  fixed base compressor's false negatives, spent through a bounded rescue
  reserve, on linear reasoning traces — is an unstudied intersection of
  known components, which this project's predefined rule (applied to kill
  its own N3 in A3) classifies as an application, not a method. Not
  KILLED outright: no single inspected paper performs the composite
  operation. Not SURVIVES: fails "more than an empirical intersection."
- **M2 — Interaction-aware dynamic rescue: KILLED.** The central
  operation — eviction decisions conditional on the current compressed
  cache, refreshed across sequential compactions, for reasoning models —
  is performed by ForesightKV's GRPO stage (state = "the current remaining
  KV cache at step t") and by Neural Garbage Collection, and the base
  compressor R-KV already rescores at every compaction. The residual
  qualifier does not rescue M2 because it inherits M1's status.
- **M3 — Faithfulness-constrained memory allocation: PARTIAL —
  INSUFFICIENT METHOD NOVELTY.** Constrained decoding-time budget
  allocation for reasoning models exists (ReasonAlloc; learned variants
  LKV/Ada-KV lineage). The only new element is the constraint metric —
  explicitly insufficient under the §9 decision standard ("more than a new
  metric"), and pre-flagged as such in the task's own candidate
  description. Checked absence: no inspected paper uses a causal
  reasoning-dependence constraint in allocation.

No fourth candidate is proposed: every design direction generated during
the adversarial self-review re-decomposed into cells of §5–§7's tables,
and the B0 instruction forbids inventing a candidate to avoid a negative
outcome.

## 13. Overall B0 verdict

**METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD YET.**

No candidate reaches SURVIVES PROVISIONALLY. M2 is killed; M1 and M3 fail
the novelty bar for the same reason A3's N3 did: they are precise,
falsifiable, *unstudied intersections of published components*, and the
project's own standard — correctly — does not count that as a method
contribution. A future candidate must contain a cache-algorithmic
operation absent as an *operation class* (not just as a cell) from §5's
table, or must produce evidence that changes what the operation is (e.g.,
a demonstrated failure mode no published mechanism can express).

## 14. Remaining uncertainties

1. Fetch-summarizer mediation: all "full text inspected" claims are
   tool-assisted; verbatim quotes were captured, but details the tool did
   not quote may have been missed. The verdict is robust to this risk in
   the negative direction only (more detail could only add overlaps, not
   remove the ones quoted).
2. LookaheadKV (2603.10899): supervision and residual-vs-absolute status
   UNKNOWN (abstract only, ICLR 2026). Could only worsen M1's position.
3. CASK core-selection signal: UNKNOWN in all inspected sources; if it
   turned out to be ablation-supervised, M1's position would worsen
   further.
4. Locret's "causal importance score" operationalization: unverified from
   primary source this session.
5. LKV / IndexMem / KVzip / KVpop / MomentKV / Judge Q / SkipKV / KVSlimmer
   / PagedEviction: screened at search-snippet level only; none was needed
   for any verdict, but a future B-phase touching learned allocation or
   eviction-loss compensation must review them fully.
6. ArborKV trajectory counts and calibration-set size: not captured.
7. Neural GC's replay-attention-mask counterfactual machinery: reported in
   a search snippet, unconfirmed in the fetched PDF summary.
8. All 2026 arXiv IDs postdate this assistant's training corpus; every
   such paper was verified by live fetch this session, but version churn
   after 2026-07-19 is uncovered.
