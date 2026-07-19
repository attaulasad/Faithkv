# A3 — Adversarial literature matrix and diagnostic novelty kill-check

Companion machine-readable data: `docs/related_work_matrix.json`. Search
methodology and query log: `docs/A3_SEARCH_LOG.md`. This document is a
literature and documentation gate only — it changes no frozen `configs/
lock.yaml` value, no historical result artifact, and no experimental code.

## 1. Executive summary

This repository's research question (`CLAUDE.md` §1) rests on a specific
diagnostic combination: hold a model's own generated reasoning trace fixed,
vary only the KV-cache policy (FullKV vs. R-KV) or the amount of that trace
visible to a forced answer, and measure counterfactual behavioral
dependence. An adversarial literature search (cutoff 2026-07-19) found that
the core fixed-trace / teacher-forced replay primitive this repository's
narrower novelty claim (N1) rests on is **already implemented and publicly
released** by CASK (arXiv:2604.10900, code released), three months before
this repository's Phase A2 commit. Early answering and causal
draft-to-answer dependence measurement — the other half of the combination —
were independently well-established by 2023 (Lanham et al.) and remain an
active area through 2026 (Thinking Drafts, RFEval, Thought Branches,
FaithCoT-Bench), none of which touch KV-cache compression. No paper found in
this search combines KV-cache compression with an early-answering/
omitted-CoT-suffix intervention under a predeclared accuracy-neutral gate,
matched realized memory, held-out evaluation, and per-example mechanism
classification — but that remaining combination is an empirical
intersection of independently known components, not by itself a new
diagnostic technique. Overall verdict: **DIAGNOSTIC SURVIVAL VERDICT: DOES
NOT SURVIVE.** See §16 for the required verdict statement in full.

## 2. Exact FaithKV diagnostic extracted from the repository

From `CLAUDE.md` §1/§4 (verbatim structure, paraphrased minimally for this
summary — the frozen wording remains in `CLAUDE.md`):

> Generate a complete reasoning response (FullKV, sampling). Replay the
> exact generated tokens one at a time under the same cache policy AND under
> R-KV. Branch at several fractions of the thinking span. Close the thinking
> block. Force a short final answer. Compare that answer to the untruncated
> base response from the SAME condition and SAME seed.

Key structural properties that any candidate "killer" paper must match to
count as `equivalent_diagnostic: YES` rather than `PARTIAL`:

1. The object held fixed across the two compared conditions is a
   **model-generated token trace** (not a prompt, not a selector
   configuration).
2. The intervention is **omission of a visible suffix** (early answering),
   not a content edit, not a resampling of alternative continuations.
3. The varying factor is **decode-time KV-cache policy** (FullKV vs. a
   compressed condition), not a static context/retrieval selector.
4. The comparison is **same-condition, same-seed** (§8 of `CLAUDE.md`) — an
   R-KV probe answer is compared to R-KV's OWN untruncated answer, never to
   FullKV's, and never to gold.
5. It runs behind a **predeclared, checked accuracy-neutral/-preserving
   gate** before any causal claim is entertained (§4/§8.3/§9).
6. Realized (measured), not nominal, memory/retention is what is reported
   and gated on.
7. Per-example mechanism classification (e.g. this repo's `reasoning_region_
   category`, `identical_through_think`) is produced over a held-out set.

No paper found in this search satisfies all seven simultaneously (§10/§16).

## 3. Terminology and non-equivalences

Per the task's mandatory distinctions, applied throughout this matrix:

- **Same prompt vs. same generated trace** — CASK and this repository both
  fix the same GENERATED trace (not just the same prompt) across compared
  conditions; most compression-method papers (R-KV, ThinKV, VaSE, RLKV,
  Tactic, LazyEviction, KaVa) only share the same prompt, letting each
  condition generate its own trace naturally.
- **Same seed vs. same token IDs** — this repository's frozen primary
  pipeline (EAS/Delta_EAS) uses same-seed, naturally-generated,
  condition-specific traces (§8, `match_{i,c,s}(f)` is same-condition,
  same-seed); its SECONDARY fixed-trace screen (`docs/EXPERIMENT.md` §11)
  and CASK both go further and force literal same-token-ID replay via
  teacher forcing. These are different rigor levels; only the second is
  where CASK is directly on point.
- **Fixed prefix vs. fixed complete trace** — Lanham et al.'s early
  answering and Thought Branches both fix a PREFIX and let the model
  continue from there (under standard attention, no compression); CASK and
  this repository's fixed-trace screen fix the COMPLETE reference
  continuation and teacher-force it token-by-token regardless of length.
- **Teacher-forced training vs. diagnostic replay** — KaVa uses compressed
  KV-cache trajectories as a TRAINING signal (distillation); CASK and this
  repository use teacher forcing purely as an EVALUATION/diagnostic replay
  mechanism, never for gradient updates.
- **Token equality vs. decoded-text equality** — this repository's atlas
  (`src/kvcot/failure_atlas.py`) and CASK's replay harness both compare at
  the TOKEN level; Lanham-style early-answering studies typically compare
  decoded/extracted final ANSWERS, not token IDs.
- **Continuation fidelity vs. causal faithfulness** — CASK's
  `replay_reference_fidelity.py` measures whether a compressed condition's
  *predicted distribution* tracks a reference trace it is being fed
  (fidelity/agreement/NLL) — a measure of how similar the compressed
  model's behavior is to the reference under forced tokens. This
  repository's EAS/Delta_EAS and PSS/Delta_PSS measure whether a condition's
  OWN final answer changes when part of ITS OWN trace is withheld — a
  measure of behavioral dependence on visible tokens. These are related but
  distinct estimands; CASK's protocol never withholds anything from the
  reference — it always replays the full continuation.
- **Answer accuracy vs. reasoning faithfulness** — this repository's own
  discipline (`CLAUDE.md` §8.5, "the both-correct subset... conditions on
  correctness... does not independently establish distributional accuracy
  preservation") already draws this line; RFEval's "faithfulness... decoupled
  from accuracy" independently reinforces the same discipline in the pure
  faithfulness-literature space.
- **Selector quality vs. reasoning dependence** — the Fixed-Contract paper
  (§8 below) diagnoses whether a context SELECTOR picked the right evidence
  tokens for a later retrieval-style answer; this repository diagnoses
  whether an ANSWER depends on a portion of its own visible REASONING trace.
  Different objects of study entirely, despite the shared "hold most things
  fixed, vary one slot" design philosophy.
- **Sparse attention vs. physical KV removal** — several screened
  compression methods (Tactic, StreamingLLM in its sink-window form) are
  attention-sparsification schemes that may retain the physical cache while
  masking it, rather than physically evicting entries the way R-KV does
  (audited in `docs/UPSTREAM_AUDIT.md`); this distinction does not change
  any N1–N3 verdict here but is noted for completeness.
- **Nominal budget vs. realized memory** — this repository names conditions
  `RKV-B{budget}` and separately measures `instantaneous_retention_ratio`
  (`CLAUDE.md` §9); CASK's `terminal_saved_ratio`/realized cache cardinality
  measurement is the same discipline, independently arrived at.
- **Post-hoc similarity vs. controlled intervention** — this repository's
  Phase A2 atlas (already retired-operating-point, post-hoc) is explicitly
  NOT the same category as its own frozen primary early-answering
  intervention (§4/§8) — see `PLAN.md`/`docs/EXPERIMENT.md` for that
  internal distinction, unrelated to the external novelty question here.
- **Mechanism discussion vs. mechanism identification** — CASK's "core vs.
  scratch" is a DESIGN-TIME token classification used to build its own
  compression policy (which tokens to protect), not a POST-HOC per-example
  diagnostic classification of WHY a specific example's behavior changed,
  which is what this repository's `reasoning_region_category`/
  `identical_through_think`/`divergence_relation_to_first_compaction` fields
  are.

## 4. Search methodology

See `docs/A3_SEARCH_LOG.md` for the full query log, round structure, and
saturation reasoning. Summary: 24 distinct queries across 5 rounds
(mandatory-paper lookup, screened-paper lookup, semantic/combination
variants, CASK code-level deep dive, Fixed-Contract deep dive), using
`WebSearch` and `WebFetch` against arXiv, GitHub, OpenReview, and
HuggingFace-Papers-indexed content. Two papers (CASK, Fixed-Contract) were
escalated to direct primary-source fetch (abstract page, and for CASK also
the official GitHub README and evaluation script source) because they are
the two papers capable of independently deciding the N1/N2 verdict; the
other 18 rest on `WebSearch` result-summary confidence, which is recorded
explicitly in each paper's `uncertainties` field in the JSON rather than
silently upgraded to a firmer verdict than the evidence supports.

## 5. Main literature matrix

Full field-by-field data: `docs/related_work_matrix.json` (20 papers,
schema-validated). Condensed view:

| paper_id | threat_level | n1 | n2 | n3 | equivalent_diagnostic |
|---|---|---|---|---|---|
| cask_2604_10900 | **EXACT_KILLER** | YES | PARTIAL | NO | PARTIAL |
| fixed_contract_2605_08234 | PARTIAL_MEDIUM | PARTIAL | NO | NO | NO |
| lanham_2307_13702 | PARTIAL_HIGH | PARTIAL | PARTIAL | NO | PARTIAL |
| thinking_drafts_2505_13774 | PARTIAL_HIGH | PARTIAL | PARTIAL | NO | PARTIAL |
| rfeval_2602_17053 | PARTIAL_HIGH | NO | PARTIAL | NO | NO |
| thought_branches_2510_27484 | PARTIAL_MEDIUM | NO | PARTIAL | NO | NO |
| faithcot_bench_2510_04040 | PARTIAL_MEDIUM | NO | PARTIAL | NO | NO |
| shotkv_2502_01941 | ADJACENT | NO | NO | NO | NO |
| hold_onto_that_thought_2512_12008 | ADJACENT | NO | NO | NO | NO |
| early_stopping_cot_2509_14004 | ADJACENT | NO | NO | NO | NO |
| rkv_2505_24133 | BACKGROUND | NO | NO | NO | NO |
| thinkv_2510_01290 | BACKGROUND | NO | NO | NO | NO |
| vase_2606_03928 | BACKGROUND | NO | NO | NO | NO |
| rlkv_2510_08525 | BACKGROUND | NO | NO | NO | NO |
| kava_2510_02312 | BACKGROUND | NO | NO | NO | NO |
| tactic_2502_12216 | BACKGROUND | NO | NO | NO | NO |
| lazyeviction_2506_15969 | BACKGROUND | NO | NO | NO | NO |
| kivi_2402_02750 | BACKGROUND | NO | NO | NO | NO |
| h2o_2306_14048 | BACKGROUND | NO | NO | NO | NO |
| streamingllm_2309_17453 | BACKGROUND | NO | NO | NO | NO |

## 6. Detailed threat memos

### 6.1 CASK — arXiv:2604.10900 (highest threat)

**Exact method.** CASK is a KV-compression METHOD (core/scratch
partitioning + two-stage prefix-eviction/decode-consolidation), but its
**official evaluation harness**, `scripts/replay_reference_fidelity.py`
(fetched directly from `github.com/Skyline-23/CASK`, 2026-07-19),
independently implements the fixed-trace teacher-forced replay primitive:

- Reference tokens come from a prior run's own recorded `output` field
  (a FullKV or high-budget generation) — `reference_output =
  str(record.get("output", ""))`.
- At every decode step the loop feeds the REFERENCE token, not whatever the
  candidate (compressed) model would itself produce: `target_token =
  continuation_ids[step_idx : step_idx + 1]`, fed via `input_ids=
  target_token.unsqueeze(0)`. This is strict teacher forcing.
- Per-step statistics recorded: target-token log-probability, top-1/top-5
  agreement, first-mismatch position, and realized (not nominal) cache size
  read from `past_key_values`/the compressor's own state.
- The loop always consumes the entire reference continuation
  (`for step_idx in range(int(continuation_ids.numel()))`) — there is no
  early-stopping or omitted-suffix branch anywhere in this file.

**Exact trace control.** `TOKEN_IDENTICAL_REPLAY` — the strongest level on
the controlled vocabulary, matching this repository's own secondary
fixed-trace screen (`docs/EXPERIMENT.md` §11) almost exactly in mechanism.

**Exact cache intervention.** Decode-time, physical (CASK's core/scratch
consolidation, compared against `fullkv` and `triattention` baselines at
matched nominal budgets, evaluated by realized `terminal_saved_ratio`).

**What varies.** The cache/compression policy (CASK vs. FullKV vs.
TriAttention) while the reference token trajectory is held identical via
teacher forcing.

**What remains fixed.** The complete generated continuation (never
truncated in this harness).

**Measured outcome.** Per-step fidelity (agreement/NLL against the
reference) and realized memory saved — a continuation-fidelity metric, not
an answer-dependence metric.

**Causal estimand.** None in the strict counterfactual sense used by this
repository — CASK measures similarity-to-reference under forced tokens, not
"what does the model do differently when it must act without seeing part of
its own trace."

**Reasoning relevance.** Direct — evaluated on AIME24/AIME25 (reasoning
math benchmarks) plus long-context summarization sets, explicitly framed
around "decode-time reasoning trace" compression.

**Accuracy and memory conditions.** Reports fidelity/accuracy as the
headline result itself, not behind a predeclared accuracy-neutral STOP/
CONTINUE gate checked before a separate causal experiment (this
repository's structure, per `CLAUDE.md` §4/§8.3, is gate-first,
causal-experiment-second). Memory is realized/measured (`matched_memory:
true`).

**Strongest overlap.** The N1 diagnostic PRIMITIVE (fixed reference trace +
teacher-forced replay across cache policies + per-step
fidelity/cache-statistics recording) is implemented, publicly released, and
demonstrated on reasoning models under decode-time KV eviction three months
before this repository's Phase A2 commit.

**Strongest distinction.** No early-answering/omitted-suffix intervention;
no accuracy-neutral predeclared gate as a headline claim boundary; no
per-example held-out mechanism taxonomy; different task domain (AIME/
summarization, not GSM8K); "core vs. scratch" is a compression-policy design
classification, not a post-hoc failure-mechanism classification.

**Is the distinction substantive or terminological?** Substantive on the
N3 axes (early-answering causal intervention, accuracy gate, per-example
taxonomy are genuinely absent, not just differently named), but the N1
primitive overlap is not terminological either — it is the same mechanism
implemented for the same general purpose (isolate cache-policy effects by
holding the trace fixed).

**Claim killed.** "This repository is the first to implement a fixed
generated-trace / teacher-forced replay diagnostic that varies decode-time
KV-cache policy while holding the visible trace fixed, applied to reasoning
models." (N1, and the reasoning/decode-time-KV half of N2.)

**Claim surviving.** "This repository is the first to combine that
primitive specifically with an early-answering/omitted-CoT-suffix
intervention, behind a predeclared accuracy-neutral gate, with realized-
memory matching, held-out evaluation, and per-example mechanism
classification, on a reasoning model." (Remaining N3 gap — see §12.)

### 6.2 When Does Value-Aware KV Eviction Help? A Fixed-Contract Diagnostic — arXiv:2605.08234

See dedicated §7 below for the full 13-question breakdown. Summary: the
"contract" fixed is the SELECTOR's own configuration/decision slots, not a
generated reasoning trace; evaluated on LongBench (retrieval/long-context
QA/summarization), not reasoning/CoT benchmarks; no evidence of KV-cache
teacher-forced replay of a generated trace, no early-answering intervention,
no reasoning-model framing. **FIXED-CONTRACT VERDICT: PARTIAL OVERLAP
ONLY** (methodology philosophy only — "hold most things fixed, vary one
slot" — not the specific fixed-TRACE object).

### 6.3 Hold Onto That Thought — arXiv:2512.12008

A benchmark comparing EXISTING eviction methods' (H2O, SnapKV variant, etc.)
aggregate accuracy on reasoning tasks (GSM8K/MATH500-scale) under natural,
unconstrained generation. No fixed-trace replay, no causal intervention, no
per-example mechanism taxonomy beyond dataset-level accuracy tables.
Threat: ADJACENT (relevant motivating context; structurally the same
category of evidence as this repository's own retired protocol-v3 accuracy
gate, not a competing diagnostic).

### 6.4 R-KV — arXiv:2505.24133

The compression METHOD this repository tests (pinned submodule, already
audited file:line in `docs/UPSTREAM_AUDIT.md`). Its own paper reports
standard aggregate accuracy-vs-budget curves under natural generation — no
fixed-trace diagnostic of any kind. Threat: BACKGROUND (object of study, not
a competing diagnostic).

### 6.5 ThinKV — arXiv:2510.01290

Compression method (thought-type-aware hybrid quantization/eviction),
near-lossless accuracy at <5% cache on DeepSeek-R1-Distill/GPT-OSS/
AceReason. Aggregate accuracy evaluation only. Threat: BACKGROUND.

### 6.6 VaSE — arXiv:2606.03928

Compression method (value-magnitude-aware stochastic eviction for
reasoning models). Notable finding used to MOTIVATE the method design —
evicting high-magnitude value states causes "repetitive reasoning loops" —
but this is a method-design observation, not a fixed-trace or causal-
intervention diagnostic, and not a held-out per-example taxonomy. Threat:
BACKGROUND.

### 6.7 Lanham et al., Measuring Faithfulness in Chain-of-Thought Reasoning — arXiv:2307.13702

The original source of the early-answering intervention this repository's
own protocol descends from (`CLAUDE.md` §1: "Branch at several fractions of
the thinking span... Force a short final answer... Compare that answer to
the untruncated base response"). Operates entirely at the text/prompt
level under standard, uncompressed attention — no KV-cache axis whatsoever.
See §10 for the full comparison against later faithfulness work. Threat:
PARTIAL_HIGH (kills novelty of early-answering-as-a-technique in general,
which this repository never claimed — its `CLAIM_BOUNDARY_NOTICE` exists
precisely because of prior work like this).

### 6.8 Measuring the Faithfulness of Thinking Drafts — arXiv:2505.13774

"Draft-to-answer faithfulness" (draft reliance + draft-answer consistency)
via TEXTUAL counterfactual-step editing of the draft — conceptually
adjacent to omitted-suffix dependence, mechanically a content edit, not a
KV-cache-policy substitution. No compression axis. Threat: PARTIAL_HIGH on
the "reasoning-model draft-to-answer causal dependence" framing only.

### 6.9 RFEval — arXiv:2602.17053

Large benchmark (7,186 instances, 7 tasks, 12 LRMs) explicitly decoupling
causal influence of reasoning from accuracy via "output-level
interventions." No KV-cache manipulation found. Threat: PARTIAL_HIGH,
reinforcing (not newly establishing) that accuracy/faithfulness decoupling
and causal-influence measurement are actively, recently benchmarked at
scale without any compression axis.

### 6.10-6.20 Remaining screened papers

RLKV (2510.08525), KaVa (2510.02312), Tactic (2502.12216), LazyEviction
(2506.15969), ShotKV/"Can LLMs Maintain..." (2502.01941), KIVI (2402.02750),
H2O (2306.14048), StreamingLLM (2309.17453), Thought Branches (2510.27484),
Early Stopping CoT/ES-CoT (2509.14004), FaithCoT-Bench (2510.04040) — full
detail in `docs/related_work_matrix.json`; none rises above ADJACENT/
PARTIAL_MEDIUM and none contributes an independent N1/N2/N3 kill beyond
what CASK and Lanham-lineage papers already establish. Thought Branches
(resampling-based causal partial-CoT analysis) and FaithCoT-Bench
(instance-level unfaithfulness DETECTION benchmark) are the two most
conceptually adjacent of this group but neither touches KV-cache
compression. ES-CoT is flagged specifically because its name ("early
stopping") is easily confused with this repository's "early answering" —
they are different operations (efficiency-motivated convergence-based
truncation vs. diagnostic forced-truncation-at-a-fraction).

## 7. Fixed-Contract special analysis

### Does the Fixed-Contract Diagnostic kill FaithKV?

1. **What exactly is fixed by its "contract"?** The selector's own
   setup/configuration — per the abstract, "holds the selector's setup
   fixed and changes one decision slot at a time." Not a generated model
   output.
2. **Is a generated trajectory part of the contract?** Not found after
   inspecting the abstract page and a targeted question set fetched
   2026-07-19; the described object is a context/evidence SELECTOR's
   decision process for long-context QA, not a decode trajectory.
3. **Are tokens teacher-forced?** Not confirmed from the fetched abstract;
   no explicit teacher-forcing language found (contrast with CASK's
   explicit `replay_reference_fidelity.py`).
4. **Is the same continuation replayed across policies?** Not evidenced;
   the diagnostic varies selector decision slots, not cache-policy-under-
   fixed-continuation.
5. **Is FullKV compared?** Not confirmed from the fetched excerpt.
6. **Is physical decode-time eviction used?** Ambiguous from the abstract —
   the described failure modes ("miss the evidence future decoding needs,"
   "break related evidence when fitting scores into a small cache") suggest
   a context-selection/eviction budget for evidence tokens feeding a later
   decode-time answer, closer to a retrieval-KV-budget problem than
   reasoning-CoT decode-time compression.
7. **What selector decision changes?** Per abstract: "one decision slot at
   a time" among the three named failure modes (miss evidence / mis-rank
   evidence / break related evidence when packing into budget).
8. **Is output measured at token or sequence level?** Not confirmed from
   the fetched excerpt; the reported metric ("the probe is positive on
   72.6% of positive-margin cells") suggests a per-cell (per selector
   decision) measurement, not per-generated-token.
9. **Does it study reasoning or retrieval?** Retrieval/long-context QA —
   evaluated on LongBench, a retrieval/summarization/QA benchmark suite, not
   a reasoning-CoT benchmark.
10. **Does it distinguish trajectory corruption from continuation
    behavior?** Not evidenced — no generated trajectory is described as
    the fixed object at all.
11. **Does it test causal faithfulness?** It tests whether a SELECTOR's
    decisions causally matter for downstream task accuracy (an ablation-style
    causal test on selector decisions), not whether a model's FINAL ANSWER
    is causally dependent on its own VISIBLE REASONING trace.
12. **Does it require an accuracy-neutral operating point?** Not confirmed;
    `accuracy_gate` recorded as `ACCURACY_REPORTED_ONLY` (task accuracy is
    the outcome measured, not a predeclared gate checked before a separate
    causal claim).
13. **Which of N1, N2, N3, if any, does it kill?** None outright. It
    provides `PARTIAL` overlap with N1 at the level of general experimental
    philosophy only ("fixed contract, vary one slot" is analogous in spirit
    to this repository's "fixed everything, vary cache policy or suffix
    visibility"), but the object held fixed, the task domain, and the
    causal target are all different. Does not touch N2 (not reasoning/CoT)
    or N3 (no accuracy-neutral gate, no reasoning-trace fixing, no
    early-answering).

**FIXED-CONTRACT VERDICT: PARTIAL OVERLAP ONLY.** The selector contract is
not automatically a fixed generated trace, confirming the task's own stated
expected hypothesis for this paper.

## 8. CASK special analysis

(Consolidated from §6.1's evidence base; this section answers Part 8's
checklist explicitly.)

- **How is the reference continuation generated?** Read from a prior run's
  recorded `output` field — a FullKV or high-budget run, per the paper's
  "full-KV continuation fidelity" framing and the code's generic `record`
  loader (any prior condition's output could technically be loaded as
  `reference_output`, but the paper's comparisons are always framed against
  `fullkv`).
- **FullKV or trusted high-budget run?** FullKV, per the abstract's own
  language ("full-KV continuation fidelity").
- **Do candidate conditions receive exactly the same token IDs?** Yes —
  `target_token = continuation_ids[step_idx : step_idx + 1]` fed
  unconditionally every step.
- **Is replay teacher-forced?** Yes, confirmed directly from the fetched
  source file, not inferred from the paper's prose alone.
- **Do candidate cache tensors differ?** Yes — CASK vs. TriAttention vs.
  FullKV are different cache-management policies operating on the same fed
  tokens.
- **Is compression active during replay?** Yes — the candidate condition's
  own compression policy (CASK/TriAttention) runs live during the replay,
  producing the compressed cache the fidelity/agreement statistics are
  measured against.
- **Is the replay decode-time?** Yes.
- **Are next-token distributions recorded at every step?** Yes — log-probs,
  top-1/top-5 agreement per step.
- **Is final-answer dependence measured?** Not directly — the metric family
  is per-step fidelity/agreement against the reference, not "does the
  FINAL ANSWER change." A large first-mismatch-position statistic is the
  closest proxy, but it is not framed as, or validated as, an answer-level
  dependence metric.
- **Is an early-answer or omitted-suffix intervention performed?** No — the
  loop always consumes the full reference continuation.
- **Matched nominal budget or matched realized memory?** Both reported —
  nominal `--budget` argument, cross-checked against realized
  `terminal_saved_tokens`/cache cardinality.
- **Is held-out evaluation used?** Not confirmed (UNKNOWN) from the fetched
  excerpts.
- **Is "core vs. scratch" equivalent to FaithKV's per-example mechanism
  classification?** No — core/scratch is a DESIGN-TIME token-level
  classification CASK uses to build its own compression policy (decide
  what to protect), computed once per generation to configure compression;
  it is not a POST-HOC, per-example classification of observed failure
  mechanism across a held-out evaluation set, which is what this
  repository's `reasoning_region_category`/`identical_through_think`/
  `divergence_relation_to_first_compaction` fields are.

Confirmed, from direct inspection of the official implementation rather
than terminology alone:

- CASK implements token-identical teacher-forced replay. **Confirmed.**
- The visible trajectory is fixed. **Confirmed.**
- Cache policy/state differs across compared conditions. **Confirmed.**
- This kills N1. **Confirmed** (see §6.1, §16).
- It substantially kills broad N2. **Confirmed** for the "reasoning model +
  decode-time KV eviction" half of N2's application description; does not
  kill the "early-answering/omitted-suffix" half, which Lanham-lineage work
  (§10) separately addresses.
- It does not implement FaithKV's early-answer suffix-dependence estimand.
  **Confirmed** — full-continuation-only replay, no truncation branch.
- It does not satisfy the complete N3 operating-point and held-out package.
  **Confirmed** — no predeclared accuracy-neutral gate as a headline claim
  boundary, no confirmed held-out split, no per-example post-hoc mechanism
  taxonomy.

CASK is not dismissed merely because it frames replay as an evaluation
protocol rather than a paper contribution — per the task's explicit
instruction, that framing does not change what the code demonstrably does,
which is exactly the N1 primitive.

## 9. CoT-faithfulness special analysis

### Does prior CoT-faithfulness work already provide FaithKV's causal diagnostic?

| Paper | Early answering | Truncates/omits suffix | Perturbs content | Reasoning fixed | Draft-to-answer dependence | Causal influence measured | Faithfulness vs. accuracy separated | Manipulates hidden state | Manipulates KV cache | Compares FullKV vs. compressed KV | Causal estimand vs. FaithKV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Lanham et al. (2307.13702) | Yes (originates it) | Yes | Yes (mistakes/paraphrase, separate tests) | Yes (per-test) | Implicitly (via early answering) | Yes | Yes | No | No | No | **Partial** — same suffix-omission mechanic, no cache axis |
| Thinking Drafts (2505.13774) | No (uses counterfactual step insertion, not truncation) | No (edits content, doesn't omit) | Yes | Yes | Yes (named "draft reliance") | Yes | Yes | UNKNOWN | No | No | **Different** — content-edit, not suffix-omission |
| RFEval (2602.17053) | UNKNOWN (described as "output-level interventions") | UNKNOWN | Yes (counterfactual) | UNKNOWN | Yes | Yes | Yes (explicitly decoupled) | UNKNOWN | No | No | **Different/partial** — mechanics not confirmed beyond summary |
| Thought Branches (2510.27484) | No (resampling from a prefix, not forced truncation+answer) | Partial (studies partial-CoT impact) | No (resamples rather than edits) | No (fixed PREFIX only, not complete trace) | Yes | Yes | Implicitly | No | No | No | **Different** — resampling-based, not forced-answer-based |

**Required distinction, applied:** prior faithfulness work (Lanham,
Thinking Drafts, RFEval, Thought Branches) can and does kill the novelty of
"early answering," "causal answer-dependence measurement," and "accuracy/
faithfulness decoupling" as general TECHNIQUES — none of these was ever
this repository's claim (its `CLAIM_BOUNDARY_NOTICE` explicitly disclaims
faithfulness conclusions and restricts the allowed conclusion to
counterfactual behavioral dependence on visible tokens). None of these four
papers implements or combines with ANY form of KV-cache compression, so
none of them independently establishes the complete FaithKV comparison
(FullKV vs. R-KV under the same suffix-omission intervention). Combining
KNOWN early answering with KNOWN compressed-cache replay (the CASK-style
primitive) may remain empirically unstudied as of this search, but per the
task's own framing, that combination is not automatically a new method —
it is an application of two independently known ingredients to a new
setting, which is exactly the character of the "remaining empirical gap"
in §12.

## 10. N1/N2/N3 verdicts

- **N1 — Diagnostic primitive.** DOES NOT SURVIVE. CASK's official,
  publicly released evaluation code (`replay_reference_fidelity.py`)
  independently implements fixed-generated-trace, teacher-forced,
  cache-policy-varying replay with per-step fidelity/cache-statistics
  recording, three months before this repository's Phase A2 commit.
- **N2 — Reasoning + decode-time KV application.** DOES NOT SURVIVE. CASK
  applies the N1 primitive specifically to reasoning models (AIME24/AIME25)
  under decode-time KV eviction — the "reasoning model, decode-time KV
  eviction" half of N2's application description is directly demonstrated.
  The "early-answering/reasoning-to-answer behavior" half of N2 is
  independently non-novel via Lanham/Thinking-Drafts/RFEval, though never
  combined by any of them with a KV-cache axis.
- **N3 — Accuracy-neutral intervention-based FaithKV gap.** SURVIVES AS AN
  EMPIRICAL GAP ONLY (not as a standalone method contribution — see §16 for
  why this still yields an overall DOES NOT SURVIVE verdict per the
  predefined rule). No paper found in this search combines: KV-cache
  compression (any method) + early-answering/omitted-CoT-suffix
  intervention + predeclared accuracy-neutral gate + realized-memory
  matching + held-out evaluation + per-example mechanism classification, on
  a reasoning model.

## 11. Claims killed or weakened

See §16 (Part 16 mapping) for the itemized list this repository's
documentation must not make going forward.

## 12. Defensible remaining empirical gap

"No reviewed work was found that compares FullKV and decode-time compressed
KV policies using omitted-CoT-suffix answer-dependence interventions after
validating natural accuracy and realized memory at a held-out operating
point." This is an empirical gap in the literature as searched (§4,
`docs/A3_SEARCH_LOG.md`'s saturation discussion), not a new method — it
describes an unstudied INTERSECTION of independently known ingredients
(CASK-style fixed-trace/cache-policy replay; Lanham-style early answering;
standard accuracy-gate discipline), not a new primitive.

## 13. Reviewer attack

See §19 of the final response / adversarial self-review discipline applied
throughout this document: "This is already CASK replay combined with Lanham
early answering" is treated as the working hypothesis this entire matrix
was built to test, not as a strawman to defeat. §6.1, §6.7, and §9 above
are the direct answers.

## 14. Required future citations

Any future PLAN.md redesign phase must cite, at minimum: CASK (2604.10900)
as prior art for the fixed-trace/teacher-forced replay primitive; Lanham et
al. (2307.13702) as prior art for early answering; Thinking Drafts
(2505.13774) and RFEval (2602.17053) as prior art for causal
draft-to-answer/reasoning-dependence measurement decoupled from accuracy;
and should re-run this search closer to any future submission date, since
the KV-compression-for-reasoning literature is moving fast (multiple
mandatory/screened papers here were posted within the 2026-04 to 2026-06
window alone).

## 15. Phase B decision

`PHASE B: BLOCKED — DIAGNOSTIC NOT NOVEL` (see `PLAN.md`, `CHANGELOG.md`
2026-07-19 A3 entry for the operational consequence).

# Diagnostic Survival Verdict

**N1 — Diagnostic primitive:**
DOES NOT SURVIVE

**N2 — Reasoning + decode-time KV application:**
DOES NOT SURVIVE

**N3 — Accuracy-neutral intervention-based FaithKV gap:**
SURVIVES

CASK (arXiv:2604.10900) kills the fixed-trace replay primitive: its
official, publicly released evaluation code performs exactly the
fixed-generated-trace, teacher-forced, cache-policy-varying replay this
repository's N1 claims novelty for, applied to reasoning models under
decode-time KV eviction — killing N1 and substantially killing the broad
reasoning/decode-time-KV-application half of N2. Early answering and causal
CoT-to-answer dependence measurement already exist in faithfulness
literature independent of CASK — Lanham et al. (2023) originated early
answering as a truncate-and-force-answer intervention; Thinking Drafts and
RFEval (2025-2026) independently established causal, accuracy-decoupled
reasoning-dependence measurement at scale — none of which touch KV-cache
compression, killing the remaining early-answering-novelty half of N2 by a
different route. No reviewed work was found combining all of: KV-cache
compression, early-answering/omitted-suffix intervention, a predeclared
accuracy-neutral gate, realized-memory matching, held-out evaluation, and
per-example mechanism classification, together, on a reasoning model — N3
is a remaining empirical intersection of already-known components. The
remaining intersection is not by itself a new technique — assembling two
independently documented primitives (CASK-style cache-policy replay;
Lanham-style early answering) into a new experimental setting is an
application, not a method contribution, under the project's own
non-negotiable "genuinely new method" bar for a research contribution.

DIAGNOSTIC SURVIVAL VERDICT: DOES NOT SURVIVE

PHASE B: BLOCKED — DIAGNOSTIC NOT NOVEL
