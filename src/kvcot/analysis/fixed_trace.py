"""Fixed-trace prefix-sufficiency analysis — a secondary, additive
diagnostic. NOT the frozen primary result (`kvcot.analysis.metrics`/
`.pipeline`/`.stats`/`.summaries`, EAS/Delta_EAS, §8 of the build brief) and
does not replace or modify it.

=== Why this exists, alongside EAS/Delta_EAS, not instead of it ===

`replay-probe`/EAS matches each condition's probe answer against that SAME
condition's own untruncated base answer (kvcot.analysis.metrics.compute_match).
That is the frozen, correct design for the frozen research question — but it
means FullKV and R-KV are each scored against a *different* natural trace
(each condition samples its own base generation), so a Delta_EAS effect could
in principle be partly attributable to the traces themselves differing, not
only to the cache policy. This module isolates the cache-policy question
alone: `kvcot.cli.cmd_replay_fixed_trace` replays ONE canonical trace (always
FullKV's own generated tokens) under both FullKV and R-KV cache policies, so
both conditions teacher-force identical prompt and reasoning tokens — only
the policy varies.

=== Why the match target is each policy's own f=1 answer, not the trace
source's sampled natural answer ===

The canonical trace was generated with SAMPLED decoding (temperature 0.6).
Replaying it and then closing the think block with the fixed boxed-answer
prefix (kvcot.probes.templates.render_fixed_trace_suffix) and decoding
greedily is a different decoding procedure than the one that produced the
trace's own recorded answer — so a naive comparison against the trace
source's own answer would conflate "did truncation change the answer" with
"does greedy teacher-forced replay reproduce a temperature-0.6 sample,"
exactly the confound docs/EXPERIMENT.md §7 already documents for the
ORIGINAL f=1 stability probe. Using each policy's own greedy f=1 replay as
the anchor (kvcot.schemas.FixedTraceProbeRecord.normalized_f1_anchor_answer)
removes that confound: every fraction, including f=1 itself, is compared
against a same-policy, same-decoding-procedure reference.

=== Protocol v2 (2026-07-16, CHANGELOG.md) ===

Protocol v1 (empty fixed-trace suffix, frozen 48-token probe budget, event-
count-based eligibility) produced n_eligible=0 at every budget tested — not
a negative result, no result at all. Two independent failures, diagnosed
from the raw probe text:

  1. The anchor (f=1 probe) almost never reached a `\\boxed{...}` within 48
     tokens — R1-Distill's answer mode is a verbose structured write-up.
     Extraction fell through to the conservative final-number fallback tier
     and grabbed an incidental mid-sentence number as the "anchor," which is
     noise, not an answer.
  2. A recorded R-KV compaction event is not sufficient evidence of actual
     compression — at the exact budget boundary R-KV can record an event
     that evicts zero tokens (kvcot.generation.replay's documented boundary
     case), so "≥1 compaction event" let through pairs where the physical
     cache never actually shrank.

Protocol v2 fixes both: a teacher-forced boxed-answer format prefix
(kvcot.probes.templates.FIXED_TRACE_SUFFIX_TEXT) makes extraction reliable
within a short budget, and eligibility now requires the physical cache to
have actually shrunk at the cut (`actual_compression_at_cut`), not just a
nonzero event count. `rkv_had_replay_compaction` (event-count-based) is
retired as a *gate* — compaction event counts remain available on each
record as a diagnostic, but never substitute for the realized-compression
check.

=== Metric: Prefix-Sufficiency Sensitivity (PSS) ===

For problem i, replay policy c (full or a specific rkv_b{budget}):

    PSS_{i,c} = mean over f in {0.125, ..., 0.875} of (1 - matches_f1_anchor(f))
    Delta_PSS_i = PSS_{i,full} - PSS_{i,rkv}

Positive Delta_PSS: R-KV is less sensitive to truncation of a SHARED
reasoning prefix than FullKV is, under the same trace. Same subtraction
order (full - rkv) as the frozen Delta_EAS convention, for the same reason:
PSS is a mismatch-rate ("sensitivity") metric, so less-sensitive-under-R-KV
means a SMALLER PSS_rkv, so full-minus-rkv is positive in the hypothesized
direction. This is a DIFFERENT metric from EAS/Delta_EAS — do not compare
PSS/Delta_PSS values against EAS/Delta_EAS ones, and do not average or pool
them; they are scored against different anchors over different trace
sources.

PSS_{i,c} is None (not 0.0) whenever this policy's own anchor is invalid
(missing), a fallback (non-boxed) extraction, or any scored fraction is
missing/failed to extract (`_pss_for_side`) — 0.0 means "a valid 0%
mismatch rate," which is a completely different, and false, claim.
Delta_PSS_i is additionally None (even when both PSS values are individually
defined) whenever the pair does not meet full eligibility (§ below) — in
particular, whenever R-KV shows no actual compression at the cut or evicted
further while answering, since in that regime R-KV behaves like FullKV and a
delta would say nothing about compression's effect.

=== Sample-size discipline ===

This module never computes a p-value or a confidence interval. It is a
kill/continue screen at n<=50 (in practice n=10 for the first pass,
configs/early_gap_b512.yaml), not a claim of any distributional result.
`build_screen_validity` additionally refuses to characterize the result
("positive"/"negative"/"gap exists"/"gap does not exist") at all unless the
screen clears minimum eligible-example, realized-compression-rate, and
retention thresholds (kvcot.config.FixedTraceSettings) — otherwise
`hypothesis_status` is "not_tested".

CLAIM BOUNDARY (§1, restated per repository convention): this measures
counterfactual behavioral dependence on the visible, generated
chain-of-thought under truncation. Nothing here licenses any statement about
internal faithfulness or cognition.

Must never import torch (tests/unit/test_no_analysis_torch_import.py) —
every input is a plain dict read from JSONL, exactly like
kvcot.analysis.metrics/.pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from kvcot.config import PROBE_FRACTIONS_ALL, PROBE_FRACTIONS_SCORED
from kvcot.probes.early_answering import CLAIM_BOUNDARY_NOTICE
from kvcot.utils.io import read_jsonl, write_json

if TYPE_CHECKING:
    from kvcot.config import FixedTraceSettings

PSS_METRIC_NOTICE = (
    "PSS/Delta_PSS is a SEPARATE, secondary, additive metric from EAS/Delta_EAS "
    "(kvcot.analysis.metrics) — it is scored against each replay policy's own "
    "greedy f=1 answer under a SHARED canonical (FullKV) trace, not against "
    "each condition's own sampled natural base answer. Do not compare or pool "
    "PSS/Delta_PSS values with EAS/Delta_EAS ones."
)


def think_parsed_ok(think_parse_status: str) -> bool:
    return think_parse_status in ("ok", "generation_prompt_preopened_ok")


def compute_pss(
    matches_by_fraction: dict[float, bool | None],
    scored_fractions: tuple[float, ...] = PROBE_FRACTIONS_SCORED,
) -> float | None:
    """PSS_{i,c} = mean over the 7 scored fractions of (1 - matches_f1_anchor(f)).

    Requires all 7 scored-fraction matches to be defined, mirroring
    kvcot.analysis.metrics.compute_eas's refusal to silently average over a
    partial set — a missing or undefined fraction makes PSS undefined for
    this (problem, seed, policy), not a partial mean.
    """
    values: list[float] = []
    for f in scored_fractions:
        if f not in matches_by_fraction:
            return None
        m = matches_by_fraction[f]
        if m is None:
            return None
        values.append(0.0 if m else 1.0)
    return sum(values) / len(values)


def compute_delta_pss(pss_full: float | None, pss_rkv: float | None) -> float | None:
    """Delta_PSS = PSS_full - PSS_rkv. Positive => R-KV less sensitive to
    truncation of a SHARED reasoning prefix. See module docstring for why
    this subtraction order (not rkv - pss_full) is correct for a
    mismatch-rate metric — the same convention as compute_delta_eas, applied
    independently here since PSS is a different metric and must not import
    from or delegate to kvcot.analysis.metrics's EAS-specific implementation."""
    if pss_full is None or pss_rkv is None:
        return None
    return pss_full - pss_rkv


def _matches_by_fraction_vs_anchor(
    probes: dict[float, dict], fractions: tuple[float, ...]
) -> dict[float, bool | None]:
    out: dict[float, bool | None] = {}
    for f in fractions:
        rec = probes.get(f)
        if rec is None:
            continue
        out[f] = rec.get("matches_f1_anchor_answer")
    return out


def _is_boxed(rec: dict | None) -> bool:
    return bool(rec is not None and rec.get("probe_extraction_status") == "boxed")


def _pss_for_side(
    group: dict[float, dict], scored_fractions: tuple[float, ...] = PROBE_FRACTIONS_SCORED
) -> float | None:
    """PSS for one replay policy's side of a pair. None (not 0.0) whenever
    this side's own f=1 anchor is missing or a non-boxed (e.g. fallback)
    extraction — a fallback anchor is documented noise (module docstring),
    never a usable reference point — or whenever `compute_pss` itself finds
    a missing/failed-extraction scored fraction."""
    f1 = group.get(1.0)
    if not _is_boxed(f1):
        return None
    matches = _matches_by_fraction_vs_anchor(group, scored_fractions)
    return compute_pss(matches, scored_fractions)


def _all_scored_extractable(matches: dict[float, bool | None], scored: tuple[float, ...]) -> bool:
    return all(f in matches and matches[f] is not None for f in scored)


@dataclass(frozen=True)
class FixedTraceEligibility:
    """A stricter, purpose-built eligibility check for this small screen —
    deliberately not reusing kvcot.analysis.metrics.EligibilityCheck, since
    the two pipelines are scored against different anchors and this one adds
    a canonical-trace cleanliness bar (no cap hit, base itself correct) that
    the primary EAS pipeline does not enforce at the pairing stage.

    Protocol v2: gates on realized compression (`rkv_actual_compression_at_f1`,
    `no_rkv_eviction_during_scored_probes`), never on a recorded compaction
    EVENT COUNT alone — see module docstring for why that changed.
    """

    full_f1_anchor_boxed: bool
    rkv_f1_anchor_boxed: bool
    full_f1_anchor_correct: bool
    rkv_f1_anchor_correct: bool
    full_all_scored_extractable: bool
    rkv_all_scored_extractable: bool
    rkv_actual_compression_at_f1: bool
    no_rkv_eviction_during_scored_probes: bool
    canonical_trace_base_correct: bool
    canonical_trace_did_not_hit_cap: bool
    canonical_trace_think_parsed: bool

    @property
    def eligible(self) -> bool:
        return (
            self.full_f1_anchor_boxed
            and self.rkv_f1_anchor_boxed
            and self.full_f1_anchor_correct
            and self.rkv_f1_anchor_correct
            and self.full_all_scored_extractable
            and self.rkv_all_scored_extractable
            and self.rkv_actual_compression_at_f1
            and self.no_rkv_eviction_during_scored_probes
            and self.canonical_trace_base_correct
            and self.canonical_trace_did_not_hit_cap
            and self.canonical_trace_think_parsed
        )

    @property
    def failure_reasons(self) -> list[str]:
        reasons = []
        if not self.full_f1_anchor_boxed:
            reasons.append("full_f1_anchor_not_boxed")
        if not self.rkv_f1_anchor_boxed:
            reasons.append("rkv_f1_anchor_not_boxed")
        if not self.full_f1_anchor_correct:
            reasons.append("full_f1_anchor_incorrect")
        if not self.rkv_f1_anchor_correct:
            reasons.append("rkv_f1_anchor_incorrect")
        if not self.full_all_scored_extractable:
            reasons.append("full_scored_probe_extraction_failed")
        if not self.rkv_all_scored_extractable:
            reasons.append("rkv_scored_probe_extraction_failed")
        if not self.rkv_actual_compression_at_f1:
            reasons.append("rkv_no_actual_compression_at_f1")
        if not self.no_rkv_eviction_during_scored_probes:
            reasons.append("rkv_evicted_during_answer_probe")
        if not self.canonical_trace_base_correct:
            reasons.append("canonical_trace_base_incorrect")
        if not self.canonical_trace_did_not_hit_cap:
            reasons.append("canonical_trace_cap_hit")
        if not self.canonical_trace_think_parsed:
            reasons.append("canonical_trace_think_parse_failed")
        return reasons


@dataclass(frozen=True)
class FixedTraceRecords:
    replay_condition: str
    trace_source_condition: str | None  # None only for an empty file
    # base_record_id -> {fraction: fixed-trace probe record dict}
    probes_by_base: dict[str, dict[float, dict]] = field(default_factory=dict)


def load_fixed_trace_records(path: str | Path, replay_condition: str) -> FixedTraceRecords:
    probes_by_base: dict[str, dict[float, dict]] = {}
    trace_source_condition: str | None = None
    for rec in read_jsonl(path):
        probes_by_base.setdefault(rec["base_record_id"], {})[float(rec["fraction"])] = rec
        if trace_source_condition is None:
            trace_source_condition = rec["trace_source_condition"]
        elif rec["trace_source_condition"] != trace_source_condition:
            raise ValueError(
                f"{path}: mixed trace_source_condition values "
                f"({trace_source_condition!r} and {rec['trace_source_condition']!r}) in one "
                "fixed-trace probe file — a fixed-trace analysis requires every record in a "
                "file to share the same canonical trace source."
            )
    return FixedTraceRecords(
        replay_condition=replay_condition,
        trace_source_condition=trace_source_condition,
        probes_by_base=probes_by_base,
    )


def _assert_shared_trace_source(full_probes: FixedTraceRecords, rkv_probes: FixedTraceRecords, expected: str) -> None:
    """Mandatory test #8 (build brief §20): FullKV and R-KV fixed-trace
    probes computed from different canonical traces must never be silently
    compared — that would defeat the entire fixed-trace design (§ module
    docstring: "both conditions teacher-force identical ... tokens"). Checked
    once here, up front, rather than per-pair, since it is a whole-file
    invariant, not a per-example one."""
    for label, recs in (("full", full_probes), ("rkv", rkv_probes)):
        if recs.trace_source_condition is not None and recs.trace_source_condition != expected:
            raise ValueError(
                f"{label} fixed-trace probes were computed from trace_source_condition="
                f"{recs.trace_source_condition!r}, but this analysis expected {expected!r} — "
                "refusing to compare probes replayed from two different canonical traces."
            )


@dataclass(frozen=True)
class FixedTracePairResult:
    source_row_index: int
    seed: int
    base_record_id: str
    eligibility: FixedTraceEligibility
    pss_full: float | None
    pss_rkv: float | None
    delta_pss: float | None  # None unless eligible AND both PSS defined
    full_f1_boxed: bool
    rkv_f1_boxed: bool
    rkv_actual_compression_at_f1: bool
    rkv_f1_retention_ratio: float | None


def build_fixed_trace_pairs(
    base_records: list[dict],
    full_probes: FixedTraceRecords,
    rkv_probes: FixedTraceRecords,
    scored_fractions: tuple[float, ...] = PROBE_FRACTIONS_SCORED,
) -> list[FixedTracePairResult]:
    """One FixedTracePairResult per canonical-trace base record for which
    BOTH replay policies have at least one fixed-trace probe on record —
    i.e. "shared" (kvcot's n_shared). Eligibility (§ FixedTraceEligibility)
    is computed but not pre-filtered here, mirroring
    kvcot.analysis.pipeline.build_pair_results: callers decide what to do
    with ineligible pairs (report them in the attrition-style per_example
    listing) rather than having them silently vanish.
    """
    results: list[FixedTracePairResult] = []
    for base in base_records:
        base_id = base["record_id"]
        full_group = full_probes.probes_by_base.get(base_id)
        rkv_group = rkv_probes.probes_by_base.get(base_id)
        if full_group is None or rkv_group is None:
            continue  # not shared -- attrition, not eligibility

        full_matches = _matches_by_fraction_vs_anchor(full_group, scored_fractions)
        rkv_matches = _matches_by_fraction_vs_anchor(rkv_group, scored_fractions)
        pss_full = _pss_for_side(full_group, scored_fractions)
        pss_rkv = _pss_for_side(rkv_group, scored_fractions)

        full_f1 = full_group.get(1.0)
        rkv_f1 = rkv_group.get(1.0)

        rkv_f1_retention = None
        if rkv_f1 is not None:
            retention = rkv_f1.get("replay_retention_at_cut")
            if retention is not None:
                rkv_f1_retention = retention.get("instantaneous_retention_ratio")

        no_eviction_during_scored = all(
            rkv_group.get(f, {}).get("probe_actual_eviction_during_answer", False) is not True
            for f in scored_fractions
        )

        elig = FixedTraceEligibility(
            full_f1_anchor_boxed=_is_boxed(full_f1),
            rkv_f1_anchor_boxed=_is_boxed(rkv_f1),
            full_f1_anchor_correct=bool(full_f1 and full_f1.get("f1_anchor_is_correct") is True),
            rkv_f1_anchor_correct=bool(rkv_f1 and rkv_f1.get("f1_anchor_is_correct") is True),
            full_all_scored_extractable=_all_scored_extractable(full_matches, scored_fractions),
            rkv_all_scored_extractable=_all_scored_extractable(rkv_matches, scored_fractions),
            rkv_actual_compression_at_f1=bool(rkv_f1 and rkv_f1.get("actual_compression_at_cut") is True),
            no_rkv_eviction_during_scored_probes=no_eviction_during_scored,
            canonical_trace_base_correct=base.get("is_correct") is True,
            canonical_trace_did_not_hit_cap=not bool(base.get("cap_hit", True)),
            canonical_trace_think_parsed=think_parsed_ok(base["think_span"]["think_parse_status"]),
        )

        delta = compute_delta_pss(pss_full, pss_rkv) if elig.eligible else None

        results.append(
            FixedTracePairResult(
                source_row_index=base["dataset"]["source_row_index"],
                seed=base["global_seed"],
                base_record_id=base_id,
                eligibility=elig,
                pss_full=pss_full,
                pss_rkv=pss_rkv,
                delta_pss=delta,
                full_f1_boxed=elig.full_f1_anchor_boxed,
                rkv_f1_boxed=elig.rkv_f1_anchor_boxed,
                rkv_actual_compression_at_f1=elig.rkv_actual_compression_at_f1,
                rkv_f1_retention_ratio=rkv_f1_retention,
            )
        )
    return results


def fixed_trace_curve_by_fraction(
    records: FixedTraceRecords, fractions: tuple[float, ...] = PROBE_FRACTIONS_ALL
) -> dict[float, float | None]:
    """Descriptive match-vs-f1-anchor rate curve for ONE replay policy across
    all 9 probe fractions — same shape/philosophy as
    kvcot.analysis.pipeline.agreement_curve_by_fraction, but keyed against
    this policy's own f=1 anchor rather than each condition's own natural
    base answer. f=1 itself is included (its match rate is, by construction,
    how often an f=1-vs-f=1 comparison is defined at all, i.e. how often the
    anchor itself extracted an answer) — descriptive only, not scored.

    None (not 0.0) for a fraction with zero valid measurements — 0.0 would
    silently claim "a valid 0% match rate," which is a different, false,
    statement from "no valid data exists at this fraction."
    """
    curve: dict[float, float | None] = {}
    for f in fractions:
        matches: list[bool] = []
        for group in records.probes_by_base.values():
            rec = group.get(f)
            if rec is None:
                continue
            m = rec.get("matches_f1_anchor_answer")
            if m is not None:
                matches.append(m)
        curve[f] = (sum(1.0 for m in matches if m) / len(matches)) if matches else None
    return curve


def fixed_trace_compression_rate_by_fraction(
    records: FixedTraceRecords, fractions: tuple[float, ...] = PROBE_FRACTIONS_ALL
) -> dict[float, float | None]:
    """Diagnostic-only: fraction of this replay policy's probes at each
    fraction whose cache had ACTUALLY shrunk relative to FullKV-equivalent
    slots (`actual_compression_at_cut`), never the recorded-event-count
    proxy. None for a fraction with no probes on record."""
    curve: dict[float, float | None] = {}
    for f in fractions:
        flags: list[bool] = []
        for group in records.probes_by_base.values():
            rec = group.get(f)
            if rec is None or "actual_compression_at_cut" not in rec:
                continue
            flags.append(bool(rec["actual_compression_at_cut"]))
        curve[f] = (sum(1.0 for v in flags if v) / len(flags)) if flags else None
    return curve


def match_rate_delta_rkv_minus_full(
    full_curve: dict[float, float | None], rkv_curve: dict[float, float | None]
) -> dict[float, float | None]:
    """rkv_match_rate - full_match_rate, per fraction. NOT full - rkv: this is
    a match-rate (not a mismatch-rate) metric, so the less-sensitive-under-R-KV
    direction is a HIGHER match rate, i.e. a POSITIVE rkv-minus-full delta —
    the opposite subtraction order from compute_delta_pss, which is scored
    over a mismatch rate. Do not swap this without re-deriving both signs
    together (see module docstring). None whenever either side is None."""
    out: dict[float, float | None] = {}
    for f in full_curve:
        if f not in rkv_curve:
            continue
        full_v, rkv_v = full_curve[f], rkv_curve[f]
        out[f] = (rkv_v - full_v) if (full_v is not None and rkv_v is not None) else None
    return out


def build_screen_validity(
    n_eligible: int,
    actual_compression_rate: float | None,
    mean_f1_rkv_retention_ratio: float | None,
    settings: "FixedTraceSettings",
) -> tuple[bool, list[str]]:
    """§ Step 11: whether this screen cleared the minimum bar to be
    interpreted AT ALL — insufficient eligible examples, compression that
    essentially never fired, or realized retention indistinguishable from
    FullKV are each, independently, reasons this screen tested nothing about
    the compression hypothesis (kill/continue gate, not a significance test).
    """
    reasons: list[str] = []
    if n_eligible < settings.min_eligible_examples:
        reasons.append(
            f"n_eligible ({n_eligible}) below min_eligible_examples "
            f"({settings.min_eligible_examples})"
        )
    if actual_compression_rate is None or actual_compression_rate < settings.min_actual_compression_rate:
        reasons.append(
            f"actual_compression_rate ({actual_compression_rate}) below "
            f"min_actual_compression_rate ({settings.min_actual_compression_rate})"
        )
    if mean_f1_rkv_retention_ratio is None or mean_f1_rkv_retention_ratio > settings.max_mean_f1_retention_ratio:
        reasons.append(
            f"mean_f1_rkv_retention_ratio ({mean_f1_rkv_retention_ratio}) above "
            f"max_mean_f1_retention_ratio ({settings.max_mean_f1_retention_ratio})"
        )
    return (len(reasons) == 0, reasons)


def build_fixed_trace_decision(
    n_shared: int,
    pair_results: list[FixedTracePairResult],
    full_curve: dict[float, float | None],
    rkv_curve: dict[float, float | None],
    settings: "FixedTraceSettings",
) -> dict:
    eligible = [p for p in pair_results if p.eligibility.eligible]
    deltas = [p.delta_pss for p in eligible if p.delta_pss is not None]
    n_eligible = len(deltas)
    mean_delta_pss = (sum(deltas) / n_eligible) if n_eligible > 0 else None
    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    n_ties = sum(1 for d in deltas if d == 0)

    n_boxed_f1_full = sum(1 for p in pair_results if p.full_f1_boxed)
    n_boxed_f1_rkv = sum(1 for p in pair_results if p.rkv_f1_boxed)
    n_actual_compression_active = sum(1 for p in pair_results if p.rkv_actual_compression_at_f1)
    actual_compression_rate = (n_actual_compression_active / n_shared) if n_shared > 0 else None

    retention_values = [p.rkv_f1_retention_ratio for p in pair_results if p.rkv_f1_retention_ratio is not None]
    mean_f1_rkv_retention_ratio = (
        sum(retention_values) / len(retention_values) if retention_values else None
    )

    screen_valid, screen_invalid_reasons = build_screen_validity(
        n_eligible=n_eligible,
        actual_compression_rate=actual_compression_rate,
        mean_f1_rkv_retention_ratio=mean_f1_rkv_retention_ratio,
        settings=settings,
    )
    # Never report a characterization of the result ("positive"/"negative"/
    # "gap exists"/"gap does not exist") — this screen's job is kill/continue,
    # not a significance claim. "screened" only means the validity gates
    # passed, i.e. mean_delta_pss/the curves above are backed by enough
    # eligible, actually-compressed examples to be worth reading at all —
    # it is still a descriptive count, not a statistical result.
    hypothesis_status = "screened" if screen_valid else "not_tested"

    return {
        "claim_boundary_notice": CLAIM_BOUNDARY_NOTICE,
        "metric_notice": PSS_METRIC_NOTICE,
        "sign_convention": (
            "positive delta_pss (pss_full - pss_rkv) => R-KV less sensitive to truncation of a "
            "SHARED reasoning prefix; positive match_rate_delta_rkv_minus_full => same direction"
        ),
        "statistical_note": (
            "descriptive counts only -- no p-value or confidence interval is computed at this "
            "sample size; this is a kill/continue screen, not the primary result"
        ),
        "n_shared": n_shared,
        "n_eligible": n_eligible,
        "n_boxed_f1_full": n_boxed_f1_full,
        "n_boxed_f1_rkv": n_boxed_f1_rkv,
        "n_actual_compression_active": n_actual_compression_active,
        "actual_compression_rate": actual_compression_rate,
        "mean_f1_rkv_retention_ratio": mean_f1_rkv_retention_ratio,
        "mean_delta_pss": mean_delta_pss,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_ties": n_ties,
        "screen_valid": screen_valid,
        "screen_invalid_reasons": screen_invalid_reasons,
        "hypothesis_status": hypothesis_status,
        "full_curve": {str(k): v for k, v in full_curve.items()},
        "rkv_curve": {str(k): v for k, v in rkv_curve.items()},
        "match_rate_delta_rkv_minus_full": {
            str(k): v for k, v in match_rate_delta_rkv_minus_full(full_curve, rkv_curve).items()
        },
        "per_example": [
            {
                "source_row_index": p.source_row_index,
                "seed": p.seed,
                "base_record_id": p.base_record_id,
                "eligible": p.eligibility.eligible,
                "failure_reasons": p.eligibility.failure_reasons,
                "pss_full": p.pss_full,
                "pss_rkv": p.pss_rkv,
                "delta_pss": p.delta_pss,
                "rkv_actual_compression_at_f1": p.rkv_actual_compression_at_f1,
                "rkv_f1_retention_ratio": p.rkv_f1_retention_ratio,
            }
            for p in pair_results
        ],
    }


def run_fixed_trace_analysis(
    output_dir: str | Path,
    trace_condition: str,
    replay_condition: str,
    stage_name: str,
    settings: "FixedTraceSettings",
) -> int:
    """End-to-end: read the canonical base file plus both replay policies'
    fixed-trace probe files from `output_dir`, pair/score them, and write
    `results/decisions/{stage_name}_fixed_trace.json`. Keyed by `stage_name`
    (not a fixed filename) so the b256/b512/b1024 screens
    (configs/early_gap_b*.yaml) never overwrite each other's decision file.
    """
    output_dir = Path(output_dir)
    base_path = output_dir / f"{trace_condition}.jsonl"
    full_probes_path = output_dir / f"{trace_condition}_on_{trace_condition}_fixed_trace_probes.jsonl"
    rkv_probes_path = output_dir / f"{replay_condition}_on_{trace_condition}_fixed_trace_probes.jsonl"

    base_records = list(read_jsonl(base_path))
    full_probes = load_fixed_trace_records(full_probes_path, trace_condition)
    rkv_probes = load_fixed_trace_records(rkv_probes_path, replay_condition)
    _assert_shared_trace_source(full_probes, rkv_probes, trace_condition)

    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    full_curve = fixed_trace_curve_by_fraction(full_probes)
    rkv_curve = fixed_trace_curve_by_fraction(rkv_probes)

    decision = build_fixed_trace_decision(len(pairs), pairs, full_curve, rkv_curve, settings)
    out_path = Path("results/decisions") / f"{stage_name}_fixed_trace.json"
    write_json(out_path, decision)
    print(
        f"wrote {out_path}: n_shared={decision['n_shared']} n_eligible={decision['n_eligible']} "
        f"mean_delta_pss={decision['mean_delta_pss']} hypothesis_status={decision['hypothesis_status']}"
    )
    return 0
