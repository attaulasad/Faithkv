# Experiment design

## 1. Research question and claim boundary

> At an accuracy-preserving operating point, does decoding-time R-KV
> compression reduce a reasoning model's behavioral dependence on the
> omitted suffix of its visible reasoning trace?

Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` only. Method: R-KV
(`third_party/R-KV`, pinned commit `45eaa7d69d20b7388321f077020a610d9afb65bd`)
only. Dataset: GSM8K test split, with a frozen MATH-500 backup if Stage 1A
shows GSM8K lacks measurement range (§4 of `docs/EXPERIMENT.md` §6 below).

**Allowed conclusion:** lower sensitivity to truncating visible reasoning
under R-KV, relative to FullKV, at a matched accuracy-preserving budget.

**Forbidden conclusions:** that the chain is fake, decorative, unfaithful
to the model's "true thoughts," or that we observed internal cognition.
This measures counterfactual behavioral dependence on *visible, generated*
tokens under a specific intervention (early answering). Nothing else. This
boundary is repeated as `kvcot.probes.early_answering.CLAIM_BOUNDARY_NOTICE`
and embedded in every generated summary string, table, and figure caption
this repository produces — not just stated here.

## 2. Intervention: early answering

Generate a complete reasoning response under a condition (FullKV or R-KV).
Replay the exact generated tokens one at a time under the same cache
policy (never a bulk re-prefill of a truncated prefix — see
`docs/REPLAY_DESIGN.md` §1 for why that would be a different, invalid
experiment). Branch at nine fractions of the think span. Close the think
block. Force a short final answer. Compare that answer to the untruncated
base response **from the same condition and same seed** — never to gold,
never across conditions.

## 3. Frozen settings

See `configs/lock.yaml` (executable) — this section explains the
non-obvious choices, not restate the table.

- **Base decoding is sampled (temperature 0.6, top-p 0.95); probe decoding
  is greedy.** These are deliberately different: greedy for the 48-token
  probe removes sampling noise from the question "did truncation change the
  answer," isolating the effect of interest. The ban on greedy decoding in
  the brief applies only to the long base generation.
- **`max_new_tokens`, never `max_length`.** `kvcot.generation.decode` has no
  code path that accepts or derives a total-sequence-length cap — see that
  module's docstring.
- **R-KV `retain_ratio=0.2` is inert under the frozen `retain_direction=last`**
  (`docs/UPSTREAM_AUDIT.md` §6.3) — recorded in `configs/lock.yaml` as a
  comment so a future reader doesn't waste time tuning a dead parameter.

## 4. Stage funnel and gates

| Stage | Purpose | Key gate |
|---|---|---|
| 0 — smoke | Machinery correctness: coherence, parity, replay identity, f=1 stability, throughput extrapolation | All pass criteria in this doc §7 below |
| 1A — measurability | Does GSM8K have enough early-answering measurement range at all? | `results/decisions/stage1a_baseline_measurability.json`'s `recommendation` |
| 1B — calibration | Which budget is both compression-active and accuracy-plausible? | Two independent gates, §8 below |
| 2 — main pilot | The actual comparison, n=200, 3 seeds | Guarded by `configs/selected_operating_point.yaml` (refuses to start without it) |

## 5. Metrics — sign convention (repeated from `kvcot.analysis.metrics`, load-bearing enough to restate)

```
match_{i,c,s}(f)  = 1 iff probe answer at f == same condition's own untruncated base answer, same seed
EAS_{i,c,s}       = mean over f in {0.125, ..., 0.875} of (1 - match_{i,c,s}(f))     [7 fractions; f=0, f=1 excluded — §8.1]
Delta_EAS_{i,s}   = EAS_{i,FullKV,s} - EAS_{i,RKV,s}
```

**Positive `Delta_EAS` is the hypothesized direction**: R-KV's answer is
*less* sensitive to truncation of its own visible trace than FullKV's. A
sign error here silently inverts the entire result.

## 6. Why f=0 is excluded from EAS

On the both-correct subset both base answers equal gold, so they are
identical. If no compaction has fired by end of prefill (guaranteed
whenever `budget > prompt_length`), the R-KV cache at f=0 *is* the FullKV
cache, so `match_full(0) == match_rkv(0)` by construction and the term
cancels out of the difference — including it dilutes the effect by a
*budget-dependent* amount, which would make EAS sensitivity vary across the
Stage 1B candidates being compared. f=0 is still probed and reported as a
descriptive no-chain baseline curve point; it is just not part of the score.

## 7. Stage 0 pass criteria

- Stock FullKV output is coherent and mechanically extractable.
- Think parsing succeeds on ≥90% of non-cap-hit smoke traces.
- `patched_noop` vs. stock FullKV parity passes (`test_patched_noop_parity_gpu.py`).
- Replay identity passes (`test_replay_gpu.py` hard gates).
- The R-KV smoke fixture (`rkv_b96`) triggers ≥2 real compactions.
- At f=1, ≥90% of valid probes reproduce their own base answer, for both
  conditions (`test_probe_stability_gpu.py`).
- Throughput is measured and a Stage 2 wall-clock estimate is printed
  before Stage 2 is authorized.

**On f=1 failure:** do not tune the statistic. The base response sampled
its own post-think continuation at T=0.6 while the probe teacher-forces a
fixed control suffix and decodes greedily — a mismatch here measures suffix
sensitivity plus sampling noise, not only protocol instability. Inspect
`kvcot.probes.templates.CONTROL_SUFFIX_TEXT` and the probe decoding config
first, then report and ask.

## 8. Stage 1B's two gates have very different statistical power

- **Compaction activation** (≥1 compaction in ≥50% of valid calibration
  traces) is a near-deterministic function of trace length vs. budget. n=50
  is ample power for this gate — it is a real, load-bearing check.
- **Accuracy** at n=50 has a 95% CI of roughly ±14 percentage points on a
  proportion. This repository never claims equivalence at this sample size.
  The Stage 1B accuracy check is implemented and always labeled
  `coarse_screen` (`kvcot.analysis.summaries.build_stage1b_budget_decision`)
  — it only rejects a budget whose calibration accuracy CI *excludes*
  FullKV's point estimate, i.e. it screens off an absurd operating point,
  nothing stronger. The real accuracy comparison — the one that can support
  "accuracy-preserving" in the research question — happens at n=200 in
  Stage 2 (§9 below, §8.5 of the build brief).

Never silently choose the budget numerically closest to any particular
target retention fraction — the smallest budget passing *both* gates is
recommended; if none passes both, Stage 1B stops and reports that GSM8K
provides no accuracy-plausible, compression-active operating point for this
pilot, rather than picking one anyway.

## 9. The primary statistical control, stated explicitly (§8.5 of the build brief)

**The primary analysis's control is the both-correct-and-compression-active
subset** (§8.3 eligibility: both conditions' base answers correct, both
think spans parsed, both f=1 probes stable, R-KV had ≥1 real compaction).
This conditions on correctness *per problem* — it is not a random sample of
GSM8K, and it is not meant to be. The Stage 1B `coarse_screen` and the
Stage 2 headline paired-accuracy-with-CI (`kvcot.analysis.stats.paired_accuracy_diff`)
only keep the pilot off an operating point where R-KV has visibly broken
accuracy; they do not, on their own, establish that R-KV "preserves
accuracy" in any distributional sense. This framing has to survive review,
which is why it is stated here in plain language rather than left implicit
in the eligibility filter's code.

## 10. Attrition is treatment-correlated — report it, don't average it away

Every eligibility filter (§8.3) is plausibly correlated with the treatment:
compression may inflate or shorten generation length, changing cap-hit
rates, which changes which problems ever reach an eligible pair. §8.4's
attrition funnel (`kvcot.analysis.summaries.build_attrition_funnel_table`,
written to `results/tables/attrition_funnel.csv`) exists specifically so a
reader can see whether R-KV loses substantially more problems at any stage
— that finding, if present, belongs in the headline, not a footnote.

## 11. Secondary, additive diagnostic: fixed-trace prefix-sufficiency (added 2026-07-16, see CHANGELOG.md)

Everything in §1-§10 above describes the frozen, primary pipeline
(`replay-probe`/EAS/Delta_EAS) and is unchanged by this section. This
section documents a smaller, **secondary** screen that runs alongside it,
never in place of it — the research question in §1 remains this
repository's headline claim.

**Why.** §5's match rule is "same condition, same seed" — FullKV and R-KV
are each scored against their *own* sampled natural trace. That is the
correct design for §1's research question (a reasoning model's dependence
on *its own* visible trace), but it means a Delta_EAS effect could in
principle be attributable in part to the two conditions' traces differing
from each other, not only to the cache policy. `kvcot replay-fixed-trace`
isolates the cache-policy question alone: it replays ONE canonical trace
(always FullKV's own generated tokens — R-KV never supplies the canonical
trace, since the whole point is holding the token sequence fixed while only
the cache policy varies) under both FullKV and R-KV cache policies.

**Metric.** Prefix-Sufficiency Sensitivity (PSS), scored against each
replay policy's own greedy f=1 answer under the shared trace — never
against the trace source's sampled natural answer, which would reintroduce
the sampled-vs-greedy confound §7 above already documents for the f=1
stability probe. `Delta_PSS = PSS_full - PSS_rkv`, same subtraction order
and sign meaning as Delta_EAS. PSS/Delta_PSS is a **different metric**;
never pool or directly compare it with EAS/Delta_EAS values
(`kvcot.analysis.fixed_trace` module docstring).

**Sample size.** `configs/early_gap_b512.yaml` runs n=10, one seed —
descriptive counts only (`n_positive`/`n_negative`/`n_ties`/`mean_delta_pss`),
no p-value or confidence interval. This is a kill/continue screen, not a
claim of any distributional result — the same discipline §8 above applies
to Stage 1B's `coarse_screen` (n=50, ~±14pp CI, never labeled
`equivalence`) applies here even more strongly: n=10 has far less power
still.

**Claim boundary.** Unchanged from §1: this still measures counterfactual
behavioral dependence on visible generated tokens under an intervention
(here, cache-policy substitution over a fixed trace, rather than
truncation) — not internal faithfulness, not whether reasoning is "real."

**Protocol v2 (2026-07-16).** Protocol v1's first GPU screen (b512, n=10)
produced `n_eligible = 0` — zero scientific information about the
hypothesis, not a negative result. Root cause: an empty fixed-trace suffix
plus the frozen 48-token probe budget meant the f=1 anchor almost never
reached a `\boxed{...}` (R1-Distill's answer mode is a verbose write-up),
so extraction fell through to the conservative final-number fallback and
"anchored" against noise; separately, eligibility gated on a recorded
compaction *event count*, which can be nonzero with zero actual eviction at
the exact budget boundary. See `CHANGELOG.md`'s 2026-07-16 entry for the
full diagnosis and fix (a teacher-forced boxed-answer format prefix,
`FixedTraceSettings`, and eligibility gated on realized compression). Every
number reported from a protocol-v1 run (`schema_version` `"1.1.0"`) must be
treated as uninformative, not as evidence in either direction — do not
resume a protocol-v1 output directory under protocol v2.

Before spending GPU time on a rerun, run the CPU-only preflight:

```bash
kvcot inspect-fixed-trace --config configs/early_gap_b512.yaml --trace-condition full
```

against an already-generated FullKV base file. It reports think-span and
prompt+think-span length statistics against the configured R-KV budget and
refuses to proceed if no trace in the manifest is even longer than the
budget — R-KV cannot compress a sequence shorter than its own budget, so
this at least rules out repeating the "budget larger than every trace"
failure mode cheaply, before any replay.
