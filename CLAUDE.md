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
