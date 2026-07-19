# B0.5 — Operating-point feasibility audit and GPU cost model

**[SUPERSEDED IN PART — see `docs/B0_5_PROTOCOL_REPAIR.md` (B0.5-R,
2026-07-19)].** §7's gate wording below uses non-exact qualifiers ("e.g.
0.10-0.15", "meaningful fraction", "dramatically", "say 8 of 12") that the
task's readiness standard requires to be exact numeric cutoffs with exact
denominators — replaced in B0.5-R §12 (reusing this repository's existing
`0.10` accuracy-plausibility ceiling and `0.70` `meaningful_retention_ceiling`
exactly, and fixing new thresholds — 9/12, ±0.20, 22 GiB — explicitly,
each with a stated rationale). The candidate selection (§3-§4 below), the
GPU cost model's *caveats* (§5 below — already correctly labeled as an
estimate, not a measurement), and the submodule-modification check (§6
below) are **not** found defective and are not superseded. §5's specific
**shadow-FullKV prefix reconstruction** line items (the "~1,500 tokens"
row and the "1,500 shadow-FullKV prefix" cost-subtotal term) describe a
cost driver that no longer exists in the repaired design — B0.5-R §6
replaces shadow-FullKV reconstruction with a read-only pre-compaction
capture hook, and B0.5-R §15 gives the corrected, separated cost
breakdown. The overall estimate/measurement caveat discipline in §5
survives; these specific line items are superseded by B0.5-R §15. Text
below is otherwise unchanged.

Phase B0.5 artifact (2026-07-19). Companion to
`docs/B0_5_DISCOVERY_PROTOCOL.md` (what would be measured) and
`docs/B0_5_SEARCH_LOG.md` (literature verification log). Documentation and
source-inspection only — no GPU used, no model weights or datasets
downloaded, no inference run.

## 0. Repository-state audit (required before any edit)

```
git status --short                    -> clean, before this branch's edits
git branch --show-current (initial)   -> research/phase-b0-method-pivot
git rev-parse HEAD (initial)          -> 8d5aa21d039fe728316aee3006d5f74f5545ca0b
git submodule status                  -> -45eaa7d69d20b7388321f077020a610d9afb65bd third_party/R-KV
                                          (leading "-" = not checked out on this machine;
                                          gitlink SHA matches configs/lock.yaml's pin)
git diff --check                      -> clean, no whitespace errors
```

`origin` is `https://github.com/attaulasad/Faithkv.git`. `git log --oneline
origin/main -8` confirms `f7e9dcc` ("Merge pull request #13 from
attaulasad/research/phase-b0-method-pivot") is the tip of `main`, and
`8d5aa21` is its immediate parent with **zero content diff** between them
(`git diff 8d5aa21 origin/main --stat` returns nothing) — the PR #13 merge
was a pure fast-forward, so `8d5aa21` is confirmed an ancestor of `HEAD`
and of `origin/main`. Before starting B0.5 work, this session's Phase-B0
documentation additions (the quantization-threat paper and pilot
measurement checklist requested alongside B0.5, see the session's other
commit) were committed on `research/phase-b0-method-pivot` as their own
commit (`68b56f1`) so that a dedicated B0.5 branch could be cut from a
clean state without losing or mixing that work. `research/b0-5-discovery-protocol`
was then created from `68b56f1`. No R-KV submodule content was touched.

## 1. Constraints (from the task brief, restated)

- One RTX 3090, 24 GB VRAM.
- Models no larger than approximately 7-8B parameters.
- Discovery pilot target: under 4 GPU-hours.
- No distributed execution, no training of large models.
- No model or dataset download during this documentation phase.
- The retired GSM8K + `DeepSeek-R1-Distill-Qwen-1.5B` + b128 operating
  point is forbidden (`docs/METHOD_PIVOT_SPEC.md` §2-4).

## 2. Why GSM8K + 1.5B is not reconsidered here

Already retired on independent accuracy grounds (`docs/EXPERIMENT.md`,
"Protocol v3 outcome"): FullKV 33/50, R-KV b128 13/50, a 40pp accuracy
collapse, `gate_passed: false`. B0.5 does not revisit this — the task
brief forbids it explicitly, and nothing in this audit's literature pass
changes that verdict.

## 3. Candidate operating points

All three below use the **pinned R-KV method** (`third_party/R-KV`,
commit `45eaa7d69d20b7388321f077020a610d9afb65bd`) with its existing
default hyperparameters (`window_size=8`, `mix_lambda=0.1`,
`divide_method=step_length`, `divide_length=128`) unless noted — reusing,
not modifying, the pinned upstream source, consistent with decision
criterion 7 (no R-KV submodule change).

### Candidate A (preferred) — DeepSeek-R1-Distill-Llama-8B + MATH-500, budget 1024

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Dataset | MATH-500 (500 problems; already a frozen backup dataset for this repository, `docs/EXPERIMENT.md` §1 note, so its manifest-freezing tooling already exists in `src/kvcot/data.py` in principle — not exercised in this phase) |
| Precision | BF16 (matches this repo's existing frozen dtype convention, `configs/lock.yaml`) |
| Max context/generation length | ~16,384 (R-KV's own reported max gen len for this model/dataset pair, `third_party/R-KV/README.md` "Datasets" table) |
| Compressor | R-KV, fixed budget `B_budget = 1024` |
| Candidate budgets | 1024 (34% ratio, R-KV's own reported lossless-@-fixed-tokens point) |
| Expected realized compression | ~34% retention at the lossless point per the source table below; this repo's own `RetentionSummary.instantaneous_retention_ratio` would still have to *measure*, not assume, the realized value on this repo's own replay engine |
| Public accuracy evidence | **Primary source**, `third_party/R-KV/README.md` "Budget at a Glance" table (also the arXiv:2505.24133 paper this submodule is pinned to): R1-Llama-8B on MATH-500 is reported **lossless at 34% ratio / 1024 fixed tokens** |
| Expected RTX 3090 memory use | ~16 GB BF16 weights (8B params × 2 bytes) + fixed R-KV buffers (`B_budget + B_buffer` per layer/head, bounded, not growing with sequence length — this is exactly R-KV's own selling point) + activations; expected to fit in 24 GB at batch=1 with headroom, though not verified on this machine (no GPU here) |
| Expected runtime | See §5's cost model |
| Compatibility with this repo + pinned R-KV | High — this exact (model, dataset) pair is the one R-KV's own README benchmarks against, so the pinned submodule's code path is already exercised by its own authors for this combination |
| Licensing/access uncertainty | None found — `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` is a public, ungated Hugging Face model (same access tier as the currently-frozen 1.5B model, `README.md` "Model access" note) |
| Strongest reason it may fail | The "lossless" claim is the R-KV paper's own eval protocol (their own prompt template, 64-candidate pass@1 aggregation) — **not yet reproduced under this repository's own batch-1, token-by-token replay engine and chat-template conventions.** The retired GSM8K point's failure came from exactly this kind of untested assumption; Candidate A is better-evidenced than GSM8K/1.5B/b128 ever was (real primary-source numbers exist here; none existed for the retired point before it was tried), but "better evidenced" is not "guaranteed to pass its own predeclared gate" — see §7. |

### Candidate B — DeepSeek-R1-Distill-Llama-8B + AIME-24, budget 1536

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (same as A) |
| Dataset | AIME-24 (≈30 problems per year of competition; small population) |
| Precision | BF16 |
| Max context/generation length | ~32,768 (`third_party/R-KV/README.md`) |
| Compressor | R-KV, fixed budget `B_budget = 1536` |
| Candidate budgets | 1536 (10% ratio — R-KV's own reported lossless-@-fixed-tokens point for this pair) |
| Expected realized compression | ~10% retention at the lossless point (primary source, same table) |
| Public accuracy evidence | Primary source, same README/paper table: R1-Llama-8B on AIME-24 lossless at 10% ratio / 1536 fixed tokens |
| Expected RTX 3090 memory use | Same weight footprint as A; longer traces (avg 15,536 tokens/solution) mean a larger transient buffer/activation footprint during generation even with R-KV's fixed KV budget, and a much longer wall-clock per example |
| Expected runtime | Substantially higher per-example than A — traces average ~5.2× longer than MATH-500's |
| Compatibility | Same code path as A |
| Licensing/access uncertainty | None found (AIME-24 problem sets are widely available; this repository would need to freeze its own manifest, not download from an unreviewed source, per this repo's existing manifest-freezing discipline) |
| Strongest reason it may fail | **Population size.** ~30 problems/year means `n_examples=12` (§8 of the protocol) would consume nearly half of one year's entire AIME set, leaving little room for a held-out calibration split (`docs/B0_5_DISCOVERY_PROTOCOL.md` §11) without pooling multiple years (introducing difficulty-distribution heterogeneity across years as a new confound). Combined with ~5× longer traces, this candidate's GPU-hour cost is materially higher for the same `n_examples` (§5) — it does not fit the 4-hour ceiling at the same sample size as Candidate A without either shrinking `n_examples` further (worsening the population problem) or shrinking `events_per_example`/`horizon` (weakening the discovery signal). Retained as a secondary/stretch candidate, not preferred. |

### Candidate C — DeepSeek-R1-Distill-Qwen-7B + MATH-500 (weaker evidence tier)

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| Dataset | MATH-500 |
| Precision | BF16 |
| Max context/generation length | Not directly published in the R-KV README's own table (that table only covers R1-Llama-8B and R1-Qwen-14B); general-purpose public benchmarks report this model at 92.8% MATH-500 / 55.5% AIME-2024 pass rate (WebSearch-level source, not the pinned R-KV paper itself — see `docs/B0_5_SEARCH_LOG.md`) |
| Compressor | R-KV; budget would have to be chosen by interpolation/analogy to the 8B/14B table, not read directly off a published number for this exact model |
| Candidate budgets | Not pinned to a specific published lossless point for this model — this is the candidate's central weakness |
| Expected realized compression | Unverified for this exact model |
| Public accuracy evidence | **Secondary/search-summary tier only** for the R-KV-specific accuracy-vs-budget curve on this exact model; the base (uncompressed) accuracy numbers (92.8%/55.5%) are better-attested but come from a general benchmark source, not verified against a primary paper this session |
| Expected RTX 3090 memory use | ~14 GB BF16 weights, likely the most comfortable VRAM margin of the three candidates |
| Expected runtime | Comparable to Candidate A per-token, unverified overall since the lossless budget point is unknown |
| Compatibility | R-KV's monkeypatch targets Qwen2/Qwen3/Llama attention modules (`docs/UPSTREAM_AUDIT.md` §1) — Qwen2-family compatibility is expected but the exact `DeepSeek-R1-Distill-Qwen-7B` architecture variant was not independently confirmed against the pinned `rkv/monkeypatch.py` this session |
| Licensing/access uncertainty | None found; public ungated model |
| Strongest reason it may fail | No primary-source R-KV lossless-budget number for this exact model exists in the evidence gathered this session — choosing a budget would require either a fresh calibration sweep (spending part of the 4-GPU-hour budget just to find a workable budget, leaving less for the actual discovery labeling) or an unverified interpolation from the 8B/14B table, both weaker starting positions than Candidate A. |

## 4. Recommendation

**Candidate A is selected**, per §11 criterion 1 of the task brief ("one
feasible non-retired operating point is identified"). It is the only
candidate with a directly-published, primary-source lossless accuracy
point for the exact (model, dataset, budget) triple, within the VRAM/size
constraint, with a manageable per-example cost. Candidate B is documented
as a secondary/stretch option (better compression ratio extremity, worse
population size and cost); Candidate C is documented as considered and
not preferred (weaker evidence tier). **MATH-500 was not chosen by
default** — it was compared against AIME-24 on trace length, population
size, and primary-evidence availability (§3, Candidate B's "strongest
reason it may fail"), per the task brief's explicit instruction not to
default to it.

**Not yet satisfied, and explicitly flagged as a blocker independent of
this audit's READY/BLOCKED call:** `CLAUDE.md` §4 freezes the model as
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, **only**, for this
repository's frozen pipeline. Candidate A uses a different model. Per this
repository's own governing rule ("Frozen unless a dated `CHANGELOG.md`
entry is added **before** the run"), moving to Candidate A for an actual
GPU pilot requires a separate, explicit, dated `CHANGELOG.md`/`CLAUDE.md`
amendment **before** that pilot runs — this document does not perform
that amendment and B0.5 does not have the authority to grant it
unilaterally as a side effect of a feasibility audit. This is listed again
in `docs/b0_5_decision.json`'s `blockers`.

## 5. GPU cost model

Assumptions stated explicitly; all are estimates, not measurements (no GPU
was used to produce them).

| Quantity | Value | Basis |
|---|---|---|
| Natural FullKV generations | 12 | `docs/B0_5_DISCOVERY_PROTOCOL.md` §8 `n_examples` |
| Natural R-KV generations | 12 | same |
| Examples | 12 | same |
| Sampled compaction events / example | 3 | protocol §8 |
| Candidate evicted blocks / event | 2 | protocol §8 |
| Retained control blocks / event | 2 | protocol §8 |
| Unablated (baseline) replay passes | 1 / event = 36 total | protocol §5-§6: one shared baseline continuation per event, reused by every block sampled at that event |
| Ablated/rescued replay passes | 4 / event = 144 total | 2 evicted-rescue + 2 retained-ablation branches per event |
| Shadow-FullKV prefix reconstructions | 1 / event = 36 total | needed once per event to recover evicted blocks' true KV entries (protocol §5); shared across that event's evicted candidates |
| Teacher-forced tokens / replay branch | 48 | protocol §6 `horizon` |
| Avg. natural-run length (prompt + generation) | ~3,400 tokens | MATH-500 avg solution length 2,979 tokens (`third_party/R-KV/README.md`) + an assumed ~400-token prompt |
| Avg. shadow-FullKV prefix length to reach a sampled event | ~1,500 tokens | rough midpoint of a ~2,979-token average trace, since sampled events (§2, excluding first/last) skew toward the middle of the trace |

**Total estimated forward-token-equivalents (core, no safety factor):**

- Natural runs: 12 examples × 2 conditions × 3,400 ≈ **81,600**
- Per-event replay cost: (1,500 shadow-FullKV prefix + 48 baseline + 4×48
  branch tokens) = 1,740 per event × 36 events ≈ **62,640**
- **Core subtotal ≈ 144,240 token-forward-equivalents**

**Safety factor: ×3** (covers retries, cap-hit examples needing
resampling/replacement, batching inefficiency at batch=1, and the fact
that prefill and decode are not actually uniform-cost per token — this
factor is a deliberately conservative blanket multiplier, not a precise
correction) → **≈ 432,720 token-forward-equivalents.**

**Throughput assumption:** R-KV's own published batch-1 decode throughput
on an A100 for an 8B model is ~70-80 tok/s (`third_party/R-KV/README.md`
efficiency table: 75.44 tok/s FullKV, 80.46-80.95 tok/s R-KV, at batch=1).
An RTX 3090 is generally estimated at roughly 50-70% of an A100's BF16
throughput for a similarly-sized dense model (general hardware-comparison
knowledge, not a source specific to this model — flagged as an estimate,
`docs/b0_5_decision.json` `uncertainties`). Using a conservative range of
**35-50 tok/s**:

- At 50 tok/s: 432,720 / 50 ≈ 8,654 s ≈ **2.4 GPU-hours**.
- At 35 tok/s (more conservative): 432,720 / 35 ≈ 12,363 s ≈ **3.4 GPU-hours**.

**Both estimates are under the 4-GPU-hour ceiling**, with the pessimistic
estimate leaving roughly 15% margin. Per `docs/B0_5_DISCOVERY_PROTOCOL.md`
§13 stopping rule 3, the actual future pilot must re-measure real
throughput after example 1 and re-project before continuing — this
estimate is a planning number, not a guarantee, and the protocol already
requires the harness to check itself against the ceiling empirically
rather than trust this table blindly.

## 6. B1-harness-without-submodule-modification check

The instrumentation this protocol needs (§3-§5 of the discovery protocol:
reading R-KV's internal score, diffing retained-token-ID sets, constructing
a single-block-modified copy of a cache, running a shadow-FullKV replay)
is all achievable by code living in `src/kvcot` that *calls into* the
pinned `third_party/R-KV` package (as this repository's existing
`kvcot.generation.replay`/`kvcot.generation.policies` modules already do),
without editing any file under `third_party/R-KV`. This mirrors the
existing pattern audited in `docs/UPSTREAM_AUDIT.md` (a process-global
monkeypatch applied *from* this repository's own code, not a fork of the
upstream source). **Criterion 7 is satisfiable without submodule
modification.**

## 7. Gates (predeclared, to apply in a future pilot — not run here)

**[SUPERSEDED — see `docs/B0_5_PROTOCOL_REPAIR.md` §12 for the exact
numeric replacement of every gate below (exact ceiling, exact denominator,
exact equality rule). Text below preserved as the historical proposal.]**

1. **Natural-accuracy plausibility gate:** on the 12-example calibration
   subset, R-KV's natural accuracy must not fall further below FullKV's
   natural accuracy than a predeclared ceiling (recommend reusing this
   repository's existing `pilot_accuracy_plausible` ceiling convention,
   e.g. 0.10-0.15, rather than inventing a new number) — this is the exact
   check that would have caught the GSM8K b128 failure earlier, applied
   here to Candidate A before any labeling GPU-time is spent (§13 stopping
   rule 2).
2. **Realized physical compression gate:** measured
   `instantaneous_retention_ratio` (this repository's existing schema
   field, `src/kvcot/schemas.py`) must be materially below 1.0 on a
   meaningful fraction of natural R-KV runs — reusing this repository's
   existing `require_meaningful_compression` machinery
   (`docs/EXPERIMENT.md` §11) rather than a new ad hoc check.
3. **Sufficient eligible compaction events:** each of the 12 examples must
   yield at least `events_per_example=3` eligible events (§2 of the
   discovery protocol) after cap-hit and end-of-trace exclusions; if fewer
   than, say, 8 of 12 examples clear this, the pilot's event population is
   too thin and the pilot should not proceed at this `n_examples`.
4. **Absence of catastrophic cap-hit bias:** cap-hit rate must not differ
   dramatically between FullKV and R-KV natural runs (a large asymmetry
   would mean the two conditions' generation-length distributions are not
   comparable, contaminating any downstream comparison) — reported, not
   silently averaged over, per this repository's existing attrition-funnel
   discipline (`docs/EXPERIMENT.md` §10).

**Claim boundary, restated:** none of these gates, even if passed, permits
calling Candidate A "accuracy preserving" or "accuracy neutral" in any
distributional sense — they only keep the pilot off an operating point
already known to be broken, exactly as this repository's existing
`coarse_screen`/`pilot_accuracy_plausible` gates are labeled
(`docs/EXPERIMENT.md` §8, §11 "Accuracy population vs. analysis
population").
