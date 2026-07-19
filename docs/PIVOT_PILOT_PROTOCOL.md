# B0 — Pilot measurement plan for a future, separately authorized method pilot

Phase B0 artifact (2026-07-19). Companion to `docs/METHOD_PIVOT_SPEC.md`
(candidate specs and verdict) and `docs/METHOD_NOVELTY_MATRIX.md`
(adversarial novelty matrix). **Documentation only: nothing in this file
is authorized to run.** Per `docs/METHOD_PIVOT_SPEC.md` §16, the METHOD
PIVOT VERDICT is BLOCKED — NO NOVEL METHOD YET, so no GPU pilot,
implementation, or dataset infrastructure is permitted yet. This file
exists so that *if* a future candidate clears a fresh novelty gate, the
measurement plan for its first pilot does not have to be designed from
scratch, and so that the entropy/overthinking confound recorded in
`docs/METHOD_PIVOT_SPEC.md` §5a is answerable from the first pilot's data
rather than discovered as a gap afterward.

This is a measurement checklist, not a full preregistration. A full
preregistration (experimental unit, ablation procedure, stopping rules,
gates, cost model) is a separate, larger artifact —
`docs/B0_5_DISCOVERY_PROTOCOL.md` — written for a distinct, later-authorized
phase (B0.5) and is the document of record for anything beyond "what to
measure." Where the two overlap, `docs/B0_5_DISCOVERY_PROTOCOL.md` is
authoritative; this file is retained as the original B0-phase measurement
list for provenance.

## 1. Natural-run measurements (FullKV and R-KV, per example)

For every natural (untruncated) FullKV and R-KV generation in a future
pilot, report:

1. **Generated-token count** (per condition).
2. **R-KV/FullKV length ratio** (`len(R-KV) / len(FullKV)`, same example,
   same seed) — a compression-side-effect-on-length indicator, distinct
   from realized retention.
3. **EOS termination rate** — fraction of generations that end via a real
   end-of-sequence token rather than the token cap.
4. **Cap-hit rate** — fraction of generations that hit `max_new_tokens`
   without EOS; cap-hit generations are a known confound for any
   length-based or overthinking-based measurement (an artificially
   truncated trace cannot be scored for "did it stop reasoning early").
5. **Overthinking-marker count and density** — count of branching/hedging
   markers ("wait", "but", "alternatively", and this repository's own
   marker list once fixed) per generation and per 100 generated tokens,
   for both conditions — the same marker family arXiv:2606.00206
   (`docs/METHOD_NOVELTY_MATRIX.md` §3a) uses, so any future R-KV-vs-FullKV
   difference here can be directly compared against a quantization-induced
   baseline rate rather than assumed novel.
6. **Whether the gold answer appeared before the final answer** — a
   scan of intermediate reasoning for the gold value, independent of
   whether it was the value ultimately output; directly operationalizes
   the arXiv:2606.00206 failure mode ("reach the right answer in
   intermediate reasoning steps but do not output it") for this
   repository's own model/method pair.
7. **Whether a previously correct intermediate answer was abandoned** —
   stronger than (6): requires an explicit intermediate final-answer-shaped
   assertion (not just the bare gold value appearing) that is later
   contradicted or replaced by a different final answer.
8. **Compaction count** — number of real R-KV compaction events (FullKV is
   always 0 by construction).
9. **Relation between compaction and later branching markers** — for each
   compaction event, whether a branching/overthinking marker occurs within
   a fixed following window (e.g., the next N generated tokens), reported
   as a rate, to check whether compaction events are followed by
   overthinking markers more often than a matched non-compaction baseline
   position — this is the natural-run-level analogue of the entropy
   confound in `docs/METHOD_PIVOT_SPEC.md` §5a and a prerequisite for any
   claim that a compaction event "caused" a downstream behavior change,
   which natural-run co-occurrence alone cannot establish (that requires
   the controlled intervention in §2 below).

## 2. Controlled fixed-trace-replay measurements (at selected positions)

For a controlled KV-only intervention replaying a fixed, teacher-forced
trace under varying cache policy (the existing `replay-fixed-trace`
machinery's intervention shape, per `docs/REPLAY_DESIGN.md` and
`docs/EXPERIMENT.md` §11 — reused as an *instrument*, not as a claim that
this pilot is the same experiment), record at each selected position:

1. **FullKV entropy** — predictive entropy of the next-token distribution
   under FullKV at that position.
2. **R-KV entropy** — same, under R-KV's cache state at that position.
3. **Top-1/top-2 logit margin** — `z_(1) - z_(2)` for both conditions,
   the same quantity arXiv:2606.00206 uses to explain quantization
   fragility; small margins flag decision-boundary sensitivity regardless
   of cache policy.
4. **FullKV-R-KV KL divergence, or an efficient approximation** — the
   direct position-level divergence between the two conditions'
   next-token distributions (or a cheaper proxy — e.g., top-k-restricted
   KL — if full-vocabulary KL is too expensive at pilot scale; the
   approximation choice and its error must be stated, not silently
   substituted).
5. **Top-1 agreement** — whether the two conditions' greedy next-token
   argmax matches at that position.
6. **Target-token log probability** — log-probability each condition
   assigns to the actual (reference/teacher-forced) next token, enabling a
   reference-token-NLL-style quantity without committing to a specific
   aggregation yet (aggregation and horizon are `docs/B0_5_DISCOVERY_PROTOCOL.md`'s
   job, not this checklist's).
7. **Probability mass on branching markers** — summed probability the
   next-token distribution assigns to the curated branching/overthinking
   marker set, under both conditions — the direct causal-mechanism analogue
   of measurement (5) in §1, now at the distributional level rather than
   the realized-token level.
8. **Distance from the nearest compaction** — signed or unsigned token
   distance from the measured position to the nearest R-KV compaction
   event, so that any entropy/KL/margin/branching-mass effect can be
   checked for compaction-proximity dependence rather than assumed uniform.

## 3. Why these lists exist here and not only in the pilot code

Per this repository's discipline (`CLAUDE.md`, `docs/METHOD_PIVOT_SPEC.md`
§15), no method or pilot harness is implemented in this documentation-only
phase. Recording the measurement plan now, before any implementation, means
a future implementer works against a reviewed list rather than inventing
one under time pressure once GPU access is available — and means the §5a
entropy/overthinking confound has a concrete, checked-off answer plan
rather than remaining a rhetorical caveat.

## 4. Status

**Not authorized to run.** No dataset manifest, config, evaluator, or
result directory exists for this plan. Authorization requires a candidate
method that first clears a fresh novelty gate (`docs/METHOD_PIVOT_SPEC.md`
§16) and a separate feasibility/authorization decision
(`docs/B0_5_FEASIBILITY_AUDIT.md`, `docs/b0_5_decision.json`).
