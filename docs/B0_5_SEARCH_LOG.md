# B0.5 search log

Phase B0.5 artifact. Search date: 2026-07-19. This log covers only the
incremental literature work performed for B0.5; it does not repeat the B0
search (`docs/B0_SEARCH_LOG.md`, frozen) or the A3 search
(`docs/A3_SEARCH_LOG.md`, frozen), both of which are **reused, not
re-run**, for the papers they already cover.

## 1. Mandatory-list disposition

The task brief's mandatory re-check list, cross-referenced against what
B0/A3 already verified:

| Paper | arXiv | Disposition this session |
|---|---|---|
| ArborKV | 2605.22106 | **Reused from B0** (`full_text_inspected_via_fetch`, `docs/method_novelty_matrix.json`). Not re-fetched. |
| IntentKV | 2606.09916 | Reused from B0 (full text). Not re-fetched. |
| ForesightKV | 2602.03203 | Reused from B0 (full text). Not re-fetched. |
| ThinKV v2 | 2510.01290 | Reused from B0 (full text). Not re-fetched. |
| Neural Garbage Collection | 2604.18002 | Reused from B0 (partial-yield full-text fetch). Not re-fetched. |
| Learning to Evict | 2602.10238 | Reused from B0 (abstract-level). Not re-fetched. |
| CASK | 2604.10900 | Reused from B0 (code + abstract) and A3 (evaluation-code line-level inspection). Not re-fetched. |
| VaSE | 2606.03928 | Reused from B0 (abstract + A3). Confirmed still current via this session's WebSearch (query 2, §2 below) — same 4.4%/4.9% Qwen3 improvement figures resurfaced, no contradicting newer version found. |
| InfoKV | 2606.26875 | Reused from B0 (abstract). Not re-fetched. |
| EpiKV | 2606.26472 | Reused from B0 (abstract). One fresh fetch attempt this session (§3) to locate a "counterfactual occlusion" methodology surfaced by search; the PDF fetch returned only metadata/citations, not the methods section — **inconclusive, not a new finding**, logged as an open uncertainty (§4). |
| CAOTE | 2504.14051 | Reused from B0 (abstract). Not re-fetched. |
| LazyEviction | 2506.15969 | Reused from B0/A3. Not re-fetched. |
| ReasonAlloc | 2606.11164 | Reused from B0 (partial-yield fetch). Not re-fetched. |
| Runtime-Certified Bounded-Error Quantized Attention | 2605.20868 | **New this session** — direct abstract fetch, §2. |
| Pre-hoc Sparsity | 2602.08329 | **New this session** — direct abstract fetch, §2 (real title: "Near-Oracle KV Selection via Pre-hoc Sparsity for Long-Context Inference"). |
| EntmaxKV | 2605.21649 | **New this session** — direct abstract fetch, §2 (real title: "EntmaxKV: Support-Aware Decoding for Entmax Attention"). |

**Why reuse rather than re-fetch 13 of 16:** B0's search (six days-old at
most, same literature cutoff date 2026-07-19) already performed full-text
or code-level inspection on the highest-stakes items (ArborKV, IntentKV,
ForesightKV, ThinKV, CASK) and abstract-level review on the rest, with a
saturation argument recorded in `docs/B0_SEARCH_LOG.md` ("Saturation
status") that the assistant re-derived and endorsed in this session's B0
audit (see the accompanying chat response, §2 "B0 audit verdict"). B0.5's
own literature burden is narrower than B0's: B0 had to establish whether
*any* of these papers implements the exact M1/M2/M3 cache operations
(method novelty). B0.5 only needs to establish whether any of them
performs the *specific discovery methodology* this protocol proposes
(per-state causal KV-ablation labeling against known baseline signals at a
gate-passing operating point) closely enough to change the feasibility
picture — re-reading already-quoted full texts for that narrower question
would not change B0's captured quotes, none of which describe that
combination (`docs/METHOD_NOVELTY_MATRIX.md` §6's empty-cell finding
already covers this).

## 2. New fetches this session

- **Runtime-Certified Bounded-Error Quantized Attention (arXiv:2605.20868)**,
  fetched via `WebFetch` (abstract page). Confirmed: a KV *quantization*
  method (tiered INT8/INT4 GPU cache with FP16 CPU fallback and per-head
  error bounds), not an eviction or rescue method; no per-state
  causal-utility labeling; not reasoning-model-specific. **No overlap with
  B0.5's discovery methodology or operating point.**
- **Near-Oracle KV Selection via Pre-hoc Sparsity (arXiv:2602.08329)**,
  fetched via `WebFetch` (abstract page; the task brief's working title
  "Pre-hoc Sparsity" matches this paper's actual title). Confirmed: a
  pre-attention KV selection method with an information-theoretic
  mutual-information bound on dropped mass; benchmarked on GSM8K/CoQA/
  LongBench, not on DeepSeek-R1-Distill or QwQ; no causal/counterfactual
  ablation methodology. **No overlap.**
- **EntmaxKV: Support-Aware Decoding for Entmax Attention (arXiv:2605.21649)**,
  fetched via `WebFetch` (abstract page). Confirmed: exploits entmax
  attention's exact-zero support property for KV-cache-traffic reduction
  during decoding; not training-free (assumes entmax kernels already
  trained in); not reasoning-model-specific; no causal ablation
  methodology. **No overlap.**

All three new fetches returned real, existing papers (titles independently
confirmed against arXiv metadata) with no causal-false-negative-discovery
methodology and no reasoning-model-specific evaluation, so **none of them
changes the B0.5 feasibility or novelty picture** — they are recorded for
completeness of the mandatory list, not because they moved any decision.

## 3. WebSearch queries run this session

1. `KV cache eviction counterfactual ablation utility label false negative reasoning model 2026` —
   surfaced VaSE, ForesightKV, EpiKV, MomentKV, and "When Does Value-Aware
   KV Eviction Help? A Fixed-Contract Diagnostic for Non-Monotone Cache
   Compression" (arXiv:2605.08234, already known from A3 as "Fixed-Contract").
   Also surfaced a snippet describing a **"counterfactual occlusion"**
   ground-truth importance-labeling procedure (32-token sliding window,
   stride 16, padding + greedy regeneration, "important" iff the
   regenerated answer changes; reported important-fraction ≈0.20 on
   MATH-500 and 0.52-0.64 on AIME) — a close methodological precedent to
   this protocol's own KV-ablation labeling, but operating at the
   text/token-occlusion level, not the KV-state level.
2. `R-KV DeepSeek-R1-Distill-Qwen-7B OR QwQ accuracy budget compression MATH-500 AIME 2026` —
   surfaced base-model accuracy figures (DeepSeek-R1-Distill-Qwen-7B: 92.8%
   MATH-500, 55.5% AIME-2024) used in `docs/B0_5_FEASIBILITY_AUDIT.md`
   Candidate C, and a snippet claiming "R-KV improved from 37.20% to
   42.60% on MATH500 under extreme budget constraints" for a 7B backbone —
   **flagged as unverified, search-snippet-level only**; the exact
   comparison (R-KV vs. what baseline, at what budget) was not confirmed
   against a primary source this session, and is not relied upon for any
   verdict-bearing claim (Candidate C is already the audit's
   weakest-evidence-tier candidate; this number does not promote it).
3. `"counterfactual occlusion" 32-token window reasoning span KV cache importance label answer changes` —
   attempted to pin down the source of the query-1 occlusion snippet.
   Returned several candidate titles (EpiKV, Adaptive Filtering, EchoKV,
   LazyEviction, KVzip, and others) without a single, clearly-attributed
   source in the search-summary text itself.

## 4. Unresolved attribution: the "counterfactual occlusion" methodology

Two direct-fetch attempts to confirm which paper contains the 32-token
sliding-window occlusion procedure (query 3's candidates: Adaptive
Filtering `2607.13205` and EpiKV `2606.26472`) both came back negative or
inconclusive:

- Adaptive Filtering's full text (`arxiv.org/html/2607.13205`, re-fetched
  this session) describes a **different** counterfactual procedure
  (excluding all tokens of one structural *role* from an H2O candidate
  set, on synthetic schema-dense data, n=50 prompts) — confirmed **not**
  the source of the 32-token/stride-16/MATH-500/AIME occlusion snippet.
- EpiKV's PDF (`arxiv.org/pdf/2606.26472`) returned only front-matter/
  citation content on fetch, not the methods section — inconclusive, not
  a negative result.
- A follow-up abstract fetch of `arxiv.org/abs/2605.07234`
  ("Reformulating KV Cache Eviction Problem for Long-Context LLM
  Inference," surfaced by query 1) failed with a transient tool-availability
  error and was not retried within this session's time budget.

**Status: unresolved.** This occlusion methodology is real (multiple
independent search snippets describe consistent numeric details — window
size, stride, important-fraction statistics for MATH-500 vs. AIME), but
its exact source paper is not pinned down this session. It is **not**
treated as existence evidence for anything B0.5 concludes (no verdict in
this phase depends on it), but it is recorded here as a genuine design
precedent worth locating properly before a future B1 harness is built: a
published, per-window (not per-KV-state) counterfactual labeling procedure
with reported important-fraction statistics on exactly MATH-500 and AIME
is directly relevant to sanity-checking this protocol's own expected
labeled-block utility distribution (`docs/B0_5_DISCOVERY_PROTOCOL.md` §7).
A future session should re-attempt this attribution before relying on it
for anything stronger than a sanity check.

## 5. What was not searched

- No Semantic Scholar / OpenReview / ACL Anthology systematic sweep (same
  gap B0's log already declares).
- No non-English literature.
- No forward-citation-database pass; forward coverage is approximated by
  the 2026-dated WebSearch queries above, same limitation B0's log states
  for its own queries.
- No attempt to independently verify the "R-KV improved from 37.20% to
  42.60%" 7B/MATH500 snippet (§3, query 2) against a primary source —
  explicitly not relied upon for any decision.

## 6. Saturation judgment for B0.5's specific question

B0.5 does not need to saturate "is there a novel method here" (B0 already
did, negatively, and B0.5 makes no method-novelty claim, per decision
criterion 8). B0.5 needs to saturate a narrower question: **does any
inspected paper already run the exact discovery methodology this protocol
proposes, at a gate-passing operating point, making this pilot redundant?**
No paper found this session or reused from B0/A3 does — the closest
precedents (ArborKV's leave-one-out KV-block accuracy-delta labeling;
ThinKV's counterfactual segment-KL labeling; the unresolved text-occlusion
procedure in §4) either operate at a different granularity, use a
different supervision target (answer accuracy instead of continuous
future-token NLL), or are not KV-state-level. This session's search is
sufficient to support proceeding to a feasibility judgment; it does not
claim to be an exhaustive forward-citation sweep.
