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

from kvcot.config import PROBE_FRACTIONS_ALL, PROBE_FRACTIONS_SCORED, FixedTraceSettings
from kvcot.probes.early_answering import CLAIM_BOUNDARY_NOTICE
from kvcot.schemas import SCHEMA_VERSION, BaseRunRecord, FixedTraceProbeRecord
from kvcot.utils.io import read_jsonl, write_json

PSS_METRIC_NOTICE = (
    "PSS/Delta_PSS is a SEPARATE, secondary, additive metric from EAS/Delta_EAS "
    "(kvcot.analysis.metrics) — it is scored against each replay policy's own "
    "greedy f=1 answer under a SHARED canonical (FullKV) trace, not against "
    "each condition's own sampled natural base answer. Do not compare or pool "
    "PSS/Delta_PSS values with EAS/Delta_EAS ones."
)

CPSS_METRIC_NOTICE = (
    "CPSS/Delta_CPSS (protocol v3, CHANGELOG.md 2026-07-17) is a SEPARATE metric "
    "from both PSS/Delta_PSS and EAS/Delta_EAS. It restricts the mean to the "
    "subset of the 7 scored fractions where R-KV's OWN measured retention at "
    "that fraction is <= meaningful_retention_ceiling ('compression-active' "
    "fractions) -- protocol v2's PSS averaged over every scored fraction "
    "regardless of whether R-KV had actually compressed anything yet at that "
    "fraction, which dilutes a real effect by a budget/schedule-dependent "
    "amount (the same reasoning CLAUDE.md/the build brief already applies to "
    "excluding f=0 from EAS). Requires at least "
    "min_compressed_scored_fractions_for_cpss fractions to clear the ceiling, "
    "else None. Do not compare or pool CPSS/Delta_CPSS with PSS/Delta_PSS or "
    "EAS/Delta_EAS values."
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


def compute_cpss(
    matches_by_fraction: dict[float, bool | None],
    active_fractions: frozenset[float] | set[float],
    min_active_fractions: int,
) -> float | None:
    """Compression-Active Prefix-Sufficiency Sensitivity (protocol v3,
    CHANGELOG.md 2026-07-17): mean over ONLY `active_fractions` (the subset
    of the 7 scored fractions where R-KV's own measured retention cleared
    `meaningful_retention_ceiling` — the SAME set for both FullKV's and
    R-KV's side of a pair, since it is determined once from R-KV's own
    retention and passed in here unchanged by the caller) of
    `(1 - matches_f1_anchor(f))`.

    None (not 0.0) whenever `active_fractions` has fewer than
    `min_active_fractions` members, or any active fraction's match is
    missing/undefined — same "None means undefined, not zero" discipline as
    `compute_pss`."""
    if len(active_fractions) < min_active_fractions:
        return None
    values: list[float] = []
    for f in active_fractions:
        if f not in matches_by_fraction:
            return None
        m = matches_by_fraction[f]
        if m is None:
            return None
        values.append(0.0 if m else 1.0)
    if not values:
        return None
    return sum(values) / len(values)


def compute_delta_cpss(cpss_full: float | None, cpss_rkv: float | None) -> float | None:
    """Delta_CPSS = CPSS_full - CPSS_rkv. Same sign convention and rationale
    as compute_delta_pss/compute_delta_eas (positive => R-KV less sensitive)
    — computed independently here, never by delegating to either of those,
    since CPSS is a separate metric scored over a different fraction set."""
    if cpss_full is None or cpss_rkv is None:
        return None
    return cpss_full - cpss_rkv


def _retention_by_fraction(
    group: dict[float, dict], fractions: tuple[float, ...]
) -> dict[float, float | None]:
    """This side's own measured `instantaneous_retention_ratio` at each
    fraction (kvcot.schemas.RetentionSummary, via `replay_retention_at_cut`)
    — None wherever the fraction's record or its retention summary is
    missing."""
    out: dict[float, float | None] = {}
    for f in fractions:
        rec = group.get(f)
        if rec is None:
            out[f] = None
            continue
        retention = rec.get("replay_retention_at_cut") or {}
        out[f] = retention.get("instantaneous_retention_ratio")
    return out


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
    `no_rkv_eviction_during_answer_probes`), never on a recorded compaction
    EVENT COUNT alone — see module docstring for why that changed.
    """

    full_f1_anchor_boxed: bool
    rkv_f1_anchor_boxed: bool
    full_f1_anchor_correct: bool
    rkv_f1_anchor_correct: bool
    full_all_scored_extractable: bool
    rkv_all_scored_extractable: bool
    rkv_actual_compression_at_f1: bool
    # Covers PROBE_FRACTIONS_SCORED *and* the f=1 anchor itself — every
    # fraction's match is scored against the f=1 anchor's own answer, so an
    # eviction that happened while the anchor was writing ITS OWN answer is
    # exactly as disqualifying as one during a scored fraction's answer (an
    # untrustworthy anchor contaminates every comparison against it, not
    # just the scored-fraction side). A prior version of this check only
    # scanned the 7 scored fractions and missed an f=1-only eviction.
    no_rkv_eviction_during_answer_probes: bool
    canonical_trace_base_correct: bool
    canonical_trace_did_not_hit_cap: bool
    canonical_trace_think_parsed: bool

    # --- Protocol v3 additions (2026-07-17, CHANGELOG.md) ---
    # rkv_f1_retention_ratio <= meaningful_retention_ceiling — a SUBSTANTIAL
    # measured retention drop, never just "not exactly 1.0"
    # (rkv_actual_compression_at_f1 above remains available as a diagnostic,
    # but protocol v2's real screen showed it lets a 0.9959-retention example
    # count as "compression active"). Computed unconditionally by the caller
    # so it is always available as a diagnostic even when not gated on.
    rkv_meaningful_compression_at_f1: bool = False
    # How many of the 7 scored fractions individually clear
    # meaningful_retention_ceiling on R-KV's own side.
    n_meaningfully_compressed_scored_fractions: int = 0
    # When False (v2 default), eligible/failure_reasons below reproduce v2's
    # exact semantics — this field and n_meaningfully_compressed_scored_
    # fractions are diagnostics only. When True (v3), eligibility ALSO
    # requires the meaningful-compression gate.
    require_meaningful_compression: bool = False
    min_meaningfully_compressed_scored_fractions: int = 0

    @property
    def eligible(self) -> bool:
        base = (
            self.full_f1_anchor_boxed
            and self.rkv_f1_anchor_boxed
            and self.full_f1_anchor_correct
            and self.rkv_f1_anchor_correct
            and self.full_all_scored_extractable
            and self.rkv_all_scored_extractable
            and self.rkv_actual_compression_at_f1
            and self.no_rkv_eviction_during_answer_probes
            and self.canonical_trace_base_correct
            and self.canonical_trace_did_not_hit_cap
            and self.canonical_trace_think_parsed
        )
        if not self.require_meaningful_compression:
            return base
        return (
            base
            and self.rkv_meaningful_compression_at_f1
            and self.n_meaningfully_compressed_scored_fractions
            >= self.min_meaningfully_compressed_scored_fractions
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
        if not self.no_rkv_eviction_during_answer_probes:
            reasons.append("rkv_evicted_during_answer_probe")
        if not self.canonical_trace_base_correct:
            reasons.append("canonical_trace_base_incorrect")
        if not self.canonical_trace_did_not_hit_cap:
            reasons.append("canonical_trace_cap_hit")
        if not self.canonical_trace_think_parsed:
            reasons.append("canonical_trace_think_parse_failed")
        if self.require_meaningful_compression:
            if not self.rkv_meaningful_compression_at_f1:
                reasons.append("rkv_no_meaningful_compression_at_f1")
            if self.n_meaningfully_compressed_scored_fractions < self.min_meaningfully_compressed_scored_fractions:
                reasons.append("too_few_meaningfully_compressed_scored_fractions")
        return reasons


@dataclass(frozen=True)
class FixedTraceRecords:
    replay_condition: str
    trace_source_condition: str | None  # None only for an empty file
    # base_record_id -> {fraction: fixed-trace probe record dict}
    probes_by_base: dict[str, dict[float, dict]] = field(default_factory=dict)


def load_fixed_trace_records(path: str | Path, replay_condition: str) -> FixedTraceRecords:
    """Read one fixed-trace probe JSONL file, grouped by
    `(base_record_id, fraction)`. A duplicate `(base_record_id, fraction)`
    pair is rejected outright (§ external review 2026-07-16) rather than
    silently letting the later row overwrite the earlier one in
    `probes_by_base` — `JsonlWriter` already refuses to append a duplicate
    `record_id` within one run, so two rows sharing a `(base_record_id,
    fraction)` key can only mean the file was corrupted, hand-edited, or
    concatenated from two different runs, none of which should be silently
    "resolved" by picking whichever row happened to come last.

    Every row's own `replay_policy_condition` must equal `replay_condition`
    (§ external review 2026-07-16): the caller passes `replay_condition`
    based on which ROLE this file is being read for (e.g. `full_on_full_
    fixed_trace_probes.jsonl` is always read as the "full" replay policy's
    probes) — nothing about a filename is otherwise checked, so a file that
    was swapped or renamed with its sibling would be silently accepted and
    read as the wrong policy, which can flip the sign of the whole
    comparison. Checking the record's own declared field, not just the
    filename convention, catches exactly that swap."""
    probes_by_base: dict[str, dict[float, dict]] = {}
    trace_source_condition: str | None = None
    for rec in read_jsonl(path):
        base_id, fraction = rec["base_record_id"], float(rec["fraction"])
        if rec["replay_policy_condition"] != replay_condition:
            raise ValueError(
                f"{path}: record {rec.get('record_id')!r} declares replay_policy_condition="
                f"{rec['replay_policy_condition']!r}, but this file is being read as the "
                f"{replay_condition!r} replay policy's probes — refusing to analyze a file whose "
                "records' declared policy does not match the role this file is being loaded for "
                "(this is exactly what a swapped or renamed pair of fixed-trace probe files would "
                "look like)."
            )
        group = probes_by_base.setdefault(base_id, {})
        if fraction in group:
            raise ValueError(
                f"{path}: duplicate (base_record_id={base_id!r}, fraction={fraction}) — "
                f"record_ids {group[fraction].get('record_id')!r} and {rec.get('record_id')!r} both "
                "claim this key. Refusing to silently let one overwrite the other; the file is "
                "corrupted, hand-edited, or concatenated from more than one run."
            )
        group[fraction] = rec
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
    # --- Protocol v3 additions (2026-07-17, CHANGELOG.md) ---
    cpss_full: float | None = None
    cpss_rkv: float | None = None
    delta_cpss: float | None = None  # None unless eligible AND both CPSS defined
    active_scored_fractions: tuple[float, ...] = ()  # fractions where R-KV cleared meaningful_retention_ceiling
    rkv_retention_by_scored_fraction: dict[float, float | None] = field(default_factory=dict)


def build_fixed_trace_pairs(
    base_records: list[dict],
    full_probes: FixedTraceRecords,
    rkv_probes: FixedTraceRecords,
    scored_fractions: tuple[float, ...] = PROBE_FRACTIONS_SCORED,
    settings: "FixedTraceSettings | None" = None,
) -> list[FixedTracePairResult]:
    """One FixedTracePairResult per canonical-trace base record for which
    BOTH replay policies have at least one fixed-trace probe on record —
    i.e. "shared" (kvcot's n_shared). Eligibility (§ FixedTraceEligibility)
    is computed but not pre-filtered here, mirroring
    kvcot.analysis.pipeline.build_pair_results: callers decide what to do
    with ineligible pairs (report them in the attrition-style per_example
    listing) rather than having them silently vanish.

    `settings` defaults to `FixedTraceSettings()` (protocol v2 semantics:
    `require_meaningful_compression=False`) when omitted, so every existing
    caller/test that does not pass it gets byte-identical behavior to
    before this parameter existed.
    """
    if settings is None:
        settings = FixedTraceSettings()
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
        rkv_meaningful_compression_at_f1 = (
            rkv_f1_retention is not None and rkv_f1_retention <= settings.meaningful_retention_ceiling
        )

        rkv_retention_by_scored_fraction = _retention_by_fraction(rkv_group, scored_fractions)
        active_fractions = frozenset(
            f for f, r in rkv_retention_by_scored_fraction.items()
            if r is not None and r <= settings.meaningful_retention_ceiling
        )

        # §Codex review 2026-07-16: must cover the f=1 anchor itself, not
        # only the 7 scored fractions — every fraction's match is scored
        # against the anchor's own answer, so an eviction while the anchor
        # was writing that answer is exactly as disqualifying.
        no_eviction_during_answer = all(
            rkv_group.get(f, {}).get("probe_actual_eviction_during_answer", False) is not True
            for f in (*scored_fractions, 1.0)
        )

        elig = FixedTraceEligibility(
            full_f1_anchor_boxed=_is_boxed(full_f1),
            rkv_f1_anchor_boxed=_is_boxed(rkv_f1),
            full_f1_anchor_correct=bool(full_f1 and full_f1.get("f1_anchor_is_correct") is True),
            rkv_f1_anchor_correct=bool(rkv_f1 and rkv_f1.get("f1_anchor_is_correct") is True),
            full_all_scored_extractable=_all_scored_extractable(full_matches, scored_fractions),
            rkv_all_scored_extractable=_all_scored_extractable(rkv_matches, scored_fractions),
            rkv_actual_compression_at_f1=bool(rkv_f1 and rkv_f1.get("actual_compression_at_cut") is True),
            no_rkv_eviction_during_answer_probes=no_eviction_during_answer,
            canonical_trace_base_correct=base.get("is_correct") is True,
            canonical_trace_did_not_hit_cap=not bool(base.get("cap_hit", True)),
            canonical_trace_think_parsed=think_parsed_ok(base["think_span"]["think_parse_status"]),
            rkv_meaningful_compression_at_f1=rkv_meaningful_compression_at_f1,
            n_meaningfully_compressed_scored_fractions=len(active_fractions),
            require_meaningful_compression=settings.require_meaningful_compression,
            min_meaningfully_compressed_scored_fractions=settings.min_meaningfully_compressed_scored_fractions,
        )

        delta = compute_delta_pss(pss_full, pss_rkv) if elig.eligible else None

        cpss_full = compute_cpss(full_matches, active_fractions, settings.min_compressed_scored_fractions_for_cpss)
        cpss_rkv = compute_cpss(rkv_matches, active_fractions, settings.min_compressed_scored_fractions_for_cpss)
        delta_cpss = compute_delta_cpss(cpss_full, cpss_rkv) if elig.eligible else None

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
                cpss_full=cpss_full,
                cpss_rkv=cpss_rkv,
                delta_cpss=delta_cpss,
                active_scored_fractions=tuple(sorted(active_fractions)),
                rkv_retention_by_scored_fraction=rkv_retention_by_scored_fraction,
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


def fixed_trace_curve_by_fraction_eligible_only(
    records: FixedTraceRecords,
    eligible_base_record_ids: set[str],
    fractions: tuple[float, ...] = PROBE_FRACTIONS_ALL,
) -> dict[float, float | None]:
    """Same match-vs-f1-anchor curve as `fixed_trace_curve_by_fraction`, but
    restricted to `eligible_base_record_ids` (protocol v3, CHANGELOG.md
    2026-07-17) — the all-shared curves conflate examples where R-KV never
    meaningfully compressed, or where an answer-time eviction contaminated
    the anchor, with the examples the screen actually trusts. Callers must
    never present an all-shared curve as the eligible scientific result;
    this function exists so both are available side by side."""
    curve: dict[float, float | None] = {}
    for f in fractions:
        matches: list[bool] = []
        for base_id, group in records.probes_by_base.items():
            if base_id not in eligible_base_record_ids:
                continue
            rec = group.get(f)
            if rec is None:
                continue
            m = rec.get("matches_f1_anchor_answer")
            if m is not None:
                matches.append(m)
        curve[f] = (sum(1.0 for m in matches if m) / len(matches)) if matches else None
    return curve


def retention_summary_by_fraction(
    records: FixedTraceRecords,
    meaningful_retention_ceiling: float,
    fractions: tuple[float, ...] = PROBE_FRACTIONS_ALL,
) -> dict[float, dict | None]:
    """Per-fraction distribution of this replay policy's own measured
    `instantaneous_retention_ratio` (protocol v3, CHANGELOG.md 2026-07-17) —
    count/mean/median/min/max plus the meaningful-compression rate at that
    fraction. This is what would have shown protocol v2's sawtooth pattern
    (retention ~0.994 at f=0.125 down to ~0.746 at f=1.0) directly in the
    decision JSON, instead of requiring a manual pass over raw records.
    None for a fraction with zero valid retention measurements."""
    summary: dict[float, dict | None] = {}
    for f in fractions:
        values: list[float] = []
        for group in records.probes_by_base.values():
            rec = group.get(f)
            if rec is None:
                continue
            retention = rec.get("replay_retention_at_cut") or {}
            r = retention.get("instantaneous_retention_ratio")
            if r is not None:
                values.append(r)
        if not values:
            summary[f] = None
            continue
        sorted_values = sorted(values)
        n = len(sorted_values)
        mid = n // 2
        median = sorted_values[mid] if n % 2 == 1 else (sorted_values[mid - 1] + sorted_values[mid]) / 2
        summary[f] = {
            "count": n,
            "mean": sum(values) / n,
            "median": median,
            "min": min(values),
            "max": max(values),
            "meaningful_compression_rate": sum(1 for v in values if v <= meaningful_retention_ceiling) / n,
        }
    return summary


def build_accuracy_screen(
    full_base_records: list[dict],
    rkv_base_records: list[dict],
    settings: "FixedTraceSettings",
) -> dict:
    """Protocol v3 (CHANGELOG.md 2026-07-17): pairs NATURAL (non-fixed-trace)
    base accuracy on the SAME manifest rows/seeds — protocol v2 never
    generated a natural R-KV b128 run on its 10-example manifest, so it
    could not establish even a small-n stop/continue signal that R-KV
    accuracy was in the same ballpark as FullKV's.

    Deliberately named `pilot_accuracy_plausible`, never `accuracy_neutral`
    — CLAUDE.md/the build brief §8.5 reserves that claim for the frozen
    primary paired 200-problem accuracy check
    (`kvcot.analysis.stats.paired_accuracy_diff`); this is a small-n
    kill/continue gate only, with no p-value or confidence interval.
    """
    def _key(row: dict) -> tuple[int, int]:
        return (row["dataset"]["source_row_index"], row["global_seed"])

    full_by_key = {_key(r): r for r in full_base_records}
    rkv_by_key = {_key(r): r for r in rkv_base_records}
    shared_keys = sorted(set(full_by_key) & set(rkv_by_key))
    n = len(shared_keys)

    def _correct(row: dict) -> bool:
        return row.get("is_correct") is True

    full_correct = sum(1 for k in shared_keys if _correct(full_by_key[k]))
    rkv_correct = sum(1 for k in shared_keys if _correct(rkv_by_key[k]))
    both_correct = sum(1 for k in shared_keys if _correct(full_by_key[k]) and _correct(rkv_by_key[k]))
    full_only_correct = sum(1 for k in shared_keys if _correct(full_by_key[k]) and not _correct(rkv_by_key[k]))
    rkv_only_correct = sum(1 for k in shared_keys if _correct(rkv_by_key[k]) and not _correct(full_by_key[k]))

    full_accuracy = (full_correct / n) if n > 0 else None
    rkv_accuracy = (rkv_correct / n) if n > 0 else None
    accuracy_difference_rkv_minus_full = (
        (rkv_accuracy - full_accuracy) if (full_accuracy is not None and rkv_accuracy is not None) else None
    )
    pilot_accuracy_plausible = (
        accuracy_difference_rkv_minus_full is not None
        and accuracy_difference_rkv_minus_full >= -settings.max_pilot_accuracy_drop
    )

    return {
        "statistical_note": (
            "descriptive counts only -- no p-value or confidence interval; a small-n "
            "stop/continue pilot gate, never the frozen primary paired_accuracy_diff"
        ),
        "n_accuracy_pairs": n,
        "full_correct": full_correct,
        "rkv_correct": rkv_correct,
        "both_correct": both_correct,
        "full_only_correct": full_only_correct,
        "rkv_only_correct": rkv_only_correct,
        "full_accuracy": full_accuracy,
        "rkv_accuracy": rkv_accuracy,
        "accuracy_difference_rkv_minus_full": accuracy_difference_rkv_minus_full,
        "max_pilot_accuracy_drop": settings.max_pilot_accuracy_drop,
        "pilot_accuracy_plausible": pilot_accuracy_plausible,
    }


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
    accuracy_screen: dict | None = None,
) -> tuple[bool, list[str]]:
    """§ Step 11: whether this screen cleared the minimum bar to be
    interpreted AT ALL — insufficient eligible examples, compression that
    essentially never fired, or realized retention indistinguishable from
    FullKV are each, independently, reasons this screen tested nothing about
    the compression hypothesis (kill/continue gate, not a significance test).

    `accuracy_screen` (protocol v3, CHANGELOG.md 2026-07-17): optional
    `build_accuracy_screen(...)` output. Omitted (`None`, the v2 default and
    every existing caller) skips this check entirely — protocol v2 never
    generated a natural R-KV run to build one from, so nothing changes for
    it. When given, a missing/implausible pilot accuracy screen invalidates
    the whole screen: the research question is explicitly scoped to an
    "accuracy-preserving operating point" (CLAUDE.md §1), so a screen that
    cannot even clear a small-n plausibility check has no business reporting
    `hypothesis_status: "screened"`.
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
    if accuracy_screen is not None and not accuracy_screen.get("pilot_accuracy_plausible"):
        reasons.append(
            f"pilot_accuracy_plausible is False (accuracy_difference_rkv_minus_full="
            f"{accuracy_screen.get('accuracy_difference_rkv_minus_full')}, "
            f"max_pilot_accuracy_drop={settings.max_pilot_accuracy_drop})"
        )
    return (len(reasons) == 0, reasons)


def build_fixed_trace_decision(
    n_shared: int,
    pair_results: list[FixedTracePairResult],
    full_curve: dict[float, float | None],
    rkv_curve: dict[float, float | None],
    settings: "FixedTraceSettings",
    full_probes: "FixedTraceRecords | None" = None,
    rkv_probes: "FixedTraceRecords | None" = None,
    accuracy_screen: dict | None = None,
) -> dict:
    """`full_probes`/`rkv_probes`/`accuracy_screen` are protocol v3 additions
    (CHANGELOG.md 2026-07-17), all optional and defaulting to `None` so
    every existing call site/test (which never passes them) is unaffected —
    when omitted, the new additive decision-JSON keys they would populate
    (`retention_summary_by_fraction`, `compression_rate_by_fraction`, the
    `*_eligible_only` curves) are simply left empty/`None` rather than
    computed from data the caller didn't provide.
    """
    eligible = [p for p in pair_results if p.eligibility.eligible]
    eligible_base_ids = {p.base_record_id for p in eligible}
    deltas = [p.delta_pss for p in eligible if p.delta_pss is not None]
    n_eligible = len(deltas)
    mean_delta_pss = (sum(deltas) / n_eligible) if n_eligible > 0 else None
    n_positive = sum(1 for d in deltas if d > 0)
    n_negative = sum(1 for d in deltas if d < 0)
    n_ties = sum(1 for d in deltas if d == 0)

    cpss_deltas = [p.delta_cpss for p in eligible if p.delta_cpss is not None]
    n_cpss_defined = len(cpss_deltas)
    mean_delta_cpss = (sum(cpss_deltas) / n_cpss_defined) if n_cpss_defined > 0 else None
    n_positive_cpss = sum(1 for d in cpss_deltas if d > 0)
    n_negative_cpss = sum(1 for d in cpss_deltas if d < 0)
    n_ties_cpss = sum(1 for d in cpss_deltas if d == 0)

    n_boxed_f1_full = sum(1 for p in pair_results if p.full_f1_boxed)
    n_boxed_f1_rkv = sum(1 for p in pair_results if p.rkv_f1_boxed)
    n_actual_compression_active = sum(1 for p in pair_results if p.rkv_actual_compression_at_f1)
    actual_compression_rate = (n_actual_compression_active / n_shared) if n_shared > 0 else None
    n_meaningful_compression_active = sum(1 for p in pair_results if p.eligibility.rkv_meaningful_compression_at_f1)
    meaningful_compression_rate = (n_meaningful_compression_active / n_shared) if n_shared > 0 else None

    retention_values = [p.rkv_f1_retention_ratio for p in pair_results if p.rkv_f1_retention_ratio is not None]
    mean_f1_rkv_retention_ratio = (
        sum(retention_values) / len(retention_values) if retention_values else None
    )

    screen_valid, screen_invalid_reasons = build_screen_validity(
        n_eligible=n_eligible,
        actual_compression_rate=actual_compression_rate,
        mean_f1_rkv_retention_ratio=mean_f1_rkv_retention_ratio,
        settings=settings,
        accuracy_screen=accuracy_screen,
    )
    # Never report a characterization of the result ("positive"/"negative"/
    # "gap exists"/"gap does not exist") — this screen's job is kill/continue,
    # not a significance claim. "screened" only means the validity gates
    # passed, i.e. mean_delta_pss/the curves above are backed by enough
    # eligible, actually-compressed examples to be worth reading at all —
    # it is still a descriptive count, not a statistical result.
    hypothesis_status = "screened" if screen_valid else "not_tested"

    full_curve_eligible_only = (
        fixed_trace_curve_by_fraction_eligible_only(full_probes, eligible_base_ids) if full_probes is not None else {}
    )
    rkv_curve_eligible_only = (
        fixed_trace_curve_by_fraction_eligible_only(rkv_probes, eligible_base_ids) if rkv_probes is not None else {}
    )
    compression_rate_by_fraction = (
        fixed_trace_compression_rate_by_fraction(rkv_probes) if rkv_probes is not None else {}
    )
    retention_summary = (
        retention_summary_by_fraction(rkv_probes, settings.meaningful_retention_ceiling) if rkv_probes is not None else {}
    )

    return {
        "claim_boundary_notice": CLAIM_BOUNDARY_NOTICE,
        "metric_notice": PSS_METRIC_NOTICE,
        "cpss_metric_notice": CPSS_METRIC_NOTICE,
        "sign_convention": (
            "positive delta_pss (pss_full - pss_rkv) => R-KV less sensitive to truncation of a "
            "SHARED reasoning prefix; positive delta_cpss (cpss_full - cpss_rkv) => same direction "
            "restricted to compression-active fractions; positive match_rate_delta_rkv_minus_full "
            "=> same direction"
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
        "n_meaningful_compression_active": n_meaningful_compression_active,
        "meaningful_compression_rate": meaningful_compression_rate,
        "mean_f1_rkv_retention_ratio": mean_f1_rkv_retention_ratio,
        "mean_delta_pss": mean_delta_pss,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_ties": n_ties,
        "n_cpss_defined": n_cpss_defined,
        "mean_delta_cpss": mean_delta_cpss,
        "n_positive_cpss": n_positive_cpss,
        "n_negative_cpss": n_negative_cpss,
        "n_ties_cpss": n_ties_cpss,
        "accuracy_screen": accuracy_screen,
        "screen_valid": screen_valid,
        "screen_invalid_reasons": screen_invalid_reasons,
        "hypothesis_status": hypothesis_status,
        # Unchanged since protocol v2 -- all-shared curves (never filtered to
        # eligible examples). Kept under these exact keys for backward
        # compatibility with the archived v2 decision JSON's consumers.
        "full_curve": {str(k): v for k, v in full_curve.items()},
        "rkv_curve": {str(k): v for k, v in rkv_curve.items()},
        # Same data, explicit "all_shared" names (protocol v3) so a reader
        # cannot mistake these for the eligible-only curves below.
        "all_shared_full_curve": {str(k): v for k, v in full_curve.items()},
        "all_shared_rkv_curve": {str(k): v for k, v in rkv_curve.items()},
        "full_curve_eligible_only": {str(k): v for k, v in full_curve_eligible_only.items()},
        "rkv_curve_eligible_only": {str(k): v for k, v in rkv_curve_eligible_only.items()},
        "compression_rate_by_fraction": {str(k): v for k, v in compression_rate_by_fraction.items()},
        "retention_summary_by_fraction": {str(k): v for k, v in retention_summary.items()},
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
                "cpss_full": p.cpss_full,
                "cpss_rkv": p.cpss_rkv,
                "delta_cpss": p.delta_cpss,
                "active_scored_fractions": list(p.active_scored_fractions),
                "rkv_actual_compression_at_f1": p.rkv_actual_compression_at_f1,
                "rkv_meaningful_compression_at_f1": p.eligibility.rkv_meaningful_compression_at_f1,
                "rkv_f1_retention_ratio": p.rkv_f1_retention_ratio,
            }
            for p in pair_results
        ],
    }


def _record_identity(row: dict) -> tuple:
    """(config_sha256, upstream_rkv_commit, model_revision, tokenizer_revision)
    — the full identity every record in one coherent run must share,
    regardless of record type. Both `BaseRunRecord` and `FixedTraceProbeRecord`
    carry all four fields as of schema 1.3.0 (added to `FixedTraceProbeRecord`
    specifically so this identity is directly comparable across the
    canonical base file and both fixed-trace probe files — `config_sha256`
    alone hashes only the STAGE yaml, never the `configs/lock.yaml` it
    references, so it cannot by itself catch a model/tokenizer revision
    drift between two runs of the identical stage yaml)."""
    provenance = row.get("provenance") or {}
    return (
        row.get("config_sha256"),
        provenance.get("upstream_rkv_commit"),
        row.get("model_revision"),
        row.get("tokenizer_revision"),
    )


def _validate_base_records(rows: list[dict], path: Path, expected_condition: str) -> tuple | None:
    """Reject anything that isn't a schema-valid, current-protocol
    `BaseRunRecord` — in particular, a stale protocol-v1/v2 (`schema_version`
    below the current `Literal`) file fails validation immediately, rather
    than being silently read as plain dicts and analyzed as if it were
    current data. Also requires every row to share one `_record_identity` —
    a file mixing two runs (different config/model/upstream pin) is not a
    single coherent trace source. Returns that shared identity (or `None` if
    `rows` is empty) so the caller can additionally cross-check it against
    the other two files this analysis reads (`_assert_consistent_identity`).

    Also requires every row's own `condition` field to equal
    `expected_condition` (§ external review 2026-07-16) — the base file this
    analysis reads is always the canonical TRACE SOURCE, i.e.
    `trace_condition` (in practice always "full", per kvcot.cli.
    cmd_replay_fixed_trace's critical rule), and a base file actually
    recorded under a different condition means the wrong file was passed in
    as the canonical trace, which nothing else here checks."""
    identities: set[tuple] = set()
    for row in rows:
        try:
            BaseRunRecord.model_validate(row)
        except Exception as e:
            raise ValueError(
                f"{path}: base record {row.get('record_id')!r} failed schema validation against "
                f"BaseRunRecord (schema_version={row.get('schema_version')!r}, expected "
                f"{SCHEMA_VERSION!r}): {e} -- refusing to analyze. This usually means {path} was "
                "produced under an old protocol version — regenerate it under the current "
                "protocol into a fresh output_dir rather than reusing old output."
            ) from e
        if row.get("condition") != expected_condition:
            raise ValueError(
                f"{path}: base record {row.get('record_id')!r} has condition="
                f"{row.get('condition')!r}, but this analysis expected the canonical trace source "
                f"condition {expected_condition!r} -- refusing to treat a "
                f"{row.get('condition')!r}-condition file as the canonical trace source."
            )
        identities.add(_record_identity(row))
    if len(identities) > 1:
        raise ValueError(
            f"{path}: base records were produced under {len(identities)} different "
            f"(config_sha256, upstream_rkv_commit, model_revision, tokenizer_revision) identities "
            f"{sorted(identities)} -- refusing to analyze a file that mixes more than one run."
        )
    return next(iter(identities), None)


def _validate_fixed_trace_probe_records(records: FixedTraceRecords, path: Path) -> tuple | None:
    """Same discipline as `_validate_base_records`, for fixed-trace probe
    files: schema-valid `FixedTraceProbeRecord` (rejects a stale
    `schema_version`) plus one shared `_record_identity` across every row.
    A duplicate `(base_record_id, fraction)` pair is already rejected
    upstream, in `load_fixed_trace_records` itself — it raises rather than
    silently keeping the last row of the same key — so by the time
    `records.probes_by_base` reaches this function, at most one row exists
    per key."""
    identities: set[tuple] = set()
    for base_id, group in records.probes_by_base.items():
        for row in group.values():
            try:
                FixedTraceProbeRecord.model_validate(row)
            except Exception as e:
                raise ValueError(
                    f"{path}: fixed-trace probe record {row.get('record_id')!r} failed schema "
                    f"validation against FixedTraceProbeRecord (schema_version="
                    f"{row.get('schema_version')!r}, expected {SCHEMA_VERSION!r}): {e} -- refusing "
                    f"to analyze. Regenerate {path} under the current protocol into a fresh "
                    "output_dir rather than reusing old output."
                ) from e
            identities.add(_record_identity(row))
    if len(identities) > 1:
        raise ValueError(
            f"{path}: fixed-trace probe records were produced under {len(identities)} different "
            f"(config_sha256, upstream_rkv_commit, model_revision, tokenizer_revision) identities "
            f"{sorted(identities)} -- refusing to analyze a file that mixes more than one run."
        )
    return next(iter(identities), None)


def _assert_consistent_identity(
    labeled_identities: list[tuple[str, tuple | None]],
    expected: tuple | None = None,
) -> None:
    """Cross-file identity check (§ external review 2026-07-16):
    `_validate_base_records`/`_validate_fixed_trace_probe_records` each only
    check ONE file's own internal consistency — this additionally requires
    every non-empty file's identity to agree with every other, and (when
    `expected` is given — the config/lock this analysis was actually
    invoked with) with that expected identity too. Without this, a base
    file from one config/model/upstream pin could be silently paired
    against fixed-trace probe files from a completely different run.

    `None` entries (an empty file — e.g. a fixed-trace probe file with zero
    records yet) are skipped, not treated as a mismatch; an empty file is a
    legitimate "nothing written yet" state reported elsewhere (`n_shared=0`
    in the decision JSON), not an identity conflict.
    """
    present = [(label, ident) for label, ident in labeled_identities if ident is not None]
    if expected is not None:
        present = present + [("current config/lock", expected)]
    if len(present) < 2:
        return
    reference_label, reference = present[0]
    for label, ident in present[1:]:
        if ident != reference:
            raise ValueError(
                f"identity mismatch: {label} has (config_sha256, upstream_rkv_commit, "
                f"model_revision, tokenizer_revision)={ident}, but {reference_label} has "
                f"{reference} -- refusing to analyze data produced under different "
                "configs/models/upstream pins as if it were one coherent run."
            )


def run_fixed_trace_analysis(
    output_dir: str | Path,
    trace_condition: str,
    replay_condition: str,
    stage_name: str,
    settings: "FixedTraceSettings",
    expected_identity: tuple | None = None,
) -> int:
    """End-to-end: read the canonical base file plus both replay policies'
    fixed-trace probe files from `output_dir`, pair/score them, and write
    `results/decisions/{stage_name}_fixed_trace.json`. Keyed by `stage_name`
    (not a fixed filename) so the b256/b512/b1024 screens
    (configs/early_gap_b*.yaml) never overwrite each other's decision file.

    Every input is validated against its Pydantic schema, checked for a
    single coherent identity WITHIN each file, and cross-checked for the
    SAME identity ACROSS all three files (`_validate_base_records`/
    `_validate_fixed_trace_probe_records`/`_assert_consistent_identity`)
    BEFORE any pairing/scoring happens — reading JSONL as plain dicts
    without this would silently accept a stale protocol-v1 directory, a
    directory mixing two different runs, or (given `expected_identity`,
    computed by the caller from the config/lock this invocation is actually
    using) a directory produced under a config/model/upstream pin different
    from the one currently pinned.
    """
    output_dir = Path(output_dir)
    base_path = output_dir / f"{trace_condition}.jsonl"
    full_probes_path = output_dir / f"{trace_condition}_on_{trace_condition}_fixed_trace_probes.jsonl"
    rkv_probes_path = output_dir / f"{replay_condition}_on_{trace_condition}_fixed_trace_probes.jsonl"

    base_records = list(read_jsonl(base_path))
    full_probes = load_fixed_trace_records(full_probes_path, trace_condition)
    rkv_probes = load_fixed_trace_records(rkv_probes_path, replay_condition)
    base_identity = _validate_base_records(base_records, base_path, trace_condition)
    full_probes_identity = _validate_fixed_trace_probe_records(full_probes, full_probes_path)
    rkv_probes_identity = _validate_fixed_trace_probe_records(rkv_probes, rkv_probes_path)
    _assert_consistent_identity(
        [
            (str(base_path), base_identity),
            (str(full_probes_path), full_probes_identity),
            (str(rkv_probes_path), rkv_probes_identity),
        ],
        expected=expected_identity,
    )
    _assert_shared_trace_source(full_probes, rkv_probes, trace_condition)

    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes, settings=settings)
    full_curve = fixed_trace_curve_by_fraction(full_probes)
    rkv_curve = fixed_trace_curve_by_fraction(rkv_probes)

    # Protocol v3 (CHANGELOG.md 2026-07-17): a natural (non-fixed-trace)
    # R-KV base run on the SAME manifest is required to even attempt a
    # pilot accuracy screen -- only look for one when this stage actually
    # requires the meaningful-compression gate (the v2/v3 discriminator);
    # v2 stages never had a natural R-KV run to read here, and must keep
    # producing byte-identical decision JSON (accuracy_screen stays None).
    accuracy_screen: dict | None = None
    if settings.require_meaningful_compression:
        rkv_natural_path = output_dir / f"{replay_condition}.jsonl"
        if rkv_natural_path.exists():
            rkv_natural_records = list(read_jsonl(rkv_natural_path))
            accuracy_screen = build_accuracy_screen(base_records, rkv_natural_records, settings)
        else:
            accuracy_screen = {
                "statistical_note": (
                    f"natural R-KV base file {rkv_natural_path} not found -- accuracy screen "
                    "could not be built"
                ),
                "n_accuracy_pairs": 0,
                "full_accuracy": None,
                "rkv_accuracy": None,
                "accuracy_difference_rkv_minus_full": None,
                "pilot_accuracy_plausible": False,
            }

    decision = build_fixed_trace_decision(
        len(pairs), pairs, full_curve, rkv_curve, settings,
        full_probes=full_probes, rkv_probes=rkv_probes, accuracy_screen=accuracy_screen,
    )
    out_path = Path("results/decisions") / f"{stage_name}_fixed_trace.json"
    write_json(out_path, decision)
    print(
        f"wrote {out_path}: n_shared={decision['n_shared']} n_eligible={decision['n_eligible']} "
        f"mean_delta_pss={decision['mean_delta_pss']} hypothesis_status={decision['hypothesis_status']}"
    )
    return 0
