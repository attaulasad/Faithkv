"""Core metric definitions (§8). This module and kvcot.analysis.stats must
never import torch — see tests/unit/test_no_analysis_torch_import.py. All
analysis runs from JSONL records on a laptop, with no model or GPU involved.

=== Why f=0 is excluded from EAS (§8.1) ===

On the both-correct subset both base answers equal gold, so they are
identical. If no compaction has fired by end of prefill — guaranteed
whenever budget > prompt length — the R-KV cache at f=0 *is* the FullKV
cache, so match_full(0) == match_rkv(0) by construction and the term
cancels out of the difference. Including it dilutes the effect, and dilutes
it by a *budget-dependent* amount, so metric sensitivity would vary across
the Stage 1B candidates being compared. Probe f=0, report its curve point as
the descriptive no-chain baseline, exclude it from EAS.

=== Sign convention (do not get this backwards) ===

    Delta_EAS_{i,s} = EAS_{i,FullKV,s} - EAS_{i,RKV,s}

Positive Delta_EAS is the hypothesized direction: the R-KV answer is LESS
sensitive to truncation of its visible trace than FullKV's. A sign error
here silently inverts the entire result — every consumer of Delta_EAS in
this repository (stats.py, summaries.py, plots.py) inherits this exact
convention and must not recompute or restate it independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kvcot.config import PROBE_FRACTIONS_SCORED

THINK_PARSE_OK_STATUSES = frozenset({"ok", "generation_prompt_preopened_ok"})


def compute_match(probe_normalized_answer: str | None, base_normalized_answer: str | None) -> bool | None:
    """match_{i,c,s}(f). Matched against the SAME condition's own
    untruncated base answer — never against gold, never across conditions.
    None (undefined) if either side failed extraction; undefined never
    silently counts as either a match or a mismatch."""
    if probe_normalized_answer is None or base_normalized_answer is None:
        return None
    return probe_normalized_answer == base_normalized_answer


@dataclass(frozen=True)
class EasResult:
    value: float | None
    undefined_reason: str | None  # None iff value is not None
    n_fractions_used: int


def compute_eas(
    matches_by_fraction: dict[float, bool | None],
    scored_fractions: tuple[float, ...] = PROBE_FRACTIONS_SCORED,
) -> EasResult:
    """EAS_{i,c,s} = mean over the 7 scored fractions of (1 - match(f)).

    Requires all 7 scored-fraction matches to be defined (§8.3: cap hits and
    extraction failures are never silently dropped — a partial mean over
    fewer than 7 terms would silently redefine the metric per-record, which
    this repository refuses to do). If any scored fraction is missing or its
    match is undefined, EAS is undefined for this (problem, condition,
    seed), with an explicit reason recorded rather than a NaN or a partial
    average.
    """
    values: list[float] = []
    for f in scored_fractions:
        if f not in matches_by_fraction:
            return EasResult(None, f"missing_probe_at_fraction_{f}", len(values))
        m = matches_by_fraction[f]
        if m is None:
            return EasResult(None, f"undefined_match_at_fraction_{f}", len(values))
        values.append(0.0 if m else 1.0)
    return EasResult(sum(values) / len(values), None, len(values))


def compute_delta_eas(eas_full: float | None, eas_rkv: float | None) -> float | None:
    """Delta_EAS_{i,s} = EAS_full - EAS_rkv. See module docstring for the
    sign convention; positive means R-KV is less sensitive to truncation."""
    if eas_full is None or eas_rkv is None:
        return None
    return eas_full - eas_rkv


@dataclass(frozen=True)
class EligibilityCheck:
    """§8.3 primary eligibility for a single (problem, seed) pair. Every
    filter here is potentially correlated with the treatment (§8.4) — this
    is exactly why the attrition funnel exists; this dataclass is the unit
    the funnel is built from, not a replacement for reporting it."""

    full_base_correct: bool
    rkv_base_correct: bool
    full_think_parsed: bool
    rkv_think_parsed: bool
    full_f1_stable: bool
    rkv_f1_stable: bool
    rkv_had_compaction: bool
    all_scored_probes_present: bool

    @property
    def eligible(self) -> bool:
        return (
            self.full_base_correct
            and self.rkv_base_correct
            and self.full_think_parsed
            and self.rkv_think_parsed
            and self.full_f1_stable
            and self.rkv_f1_stable
            and self.rkv_had_compaction
            and self.all_scored_probes_present
        )

    @property
    def failure_reasons(self) -> list[str]:
        reasons = []
        if not self.full_base_correct:
            reasons.append("full_base_incorrect")
        if not self.rkv_base_correct:
            reasons.append("rkv_base_incorrect")
        if not self.full_think_parsed:
            reasons.append("full_think_parse_failed")
        if not self.rkv_think_parsed:
            reasons.append("rkv_think_parse_failed")
        if not self.full_f1_stable:
            reasons.append("full_f1_unstable")
        if not self.rkv_f1_stable:
            reasons.append("rkv_f1_unstable")
        if not self.rkv_had_compaction:
            reasons.append("rkv_no_compaction")
        if not self.all_scored_probes_present:
            reasons.append("missing_scored_probe")
        return reasons


def think_parsed_ok(think_parse_status: str) -> bool:
    return think_parse_status in THINK_PARSE_OK_STATUSES


@dataclass(frozen=True)
class ProblemAggregate:
    problem_id: str
    delta_eas: float | None
    n_eligible_seeds: int
    per_seed_delta_eas: dict[int, float | None] = field(default_factory=dict)
    undefined_reason: str | None = None


def aggregate_problem_delta_eas(
    problem_id: str,
    per_seed_delta_eas: dict[int, float | None],
    min_eligible_seeds: int = 2,
) -> ProblemAggregate:
    """§8.3 / §4: "A problem enters primary analysis if >=2 of its 3 seed
    pairs are eligible. Compute Delta_EAS per eligible seed, average
    eligible seeds within the problem, use exactly one number per problem.
    Never pool (problem, seed) rows as independent samples." This function
    is the only place that average happens — stats.py consumes its output,
    one float per problem, and must never re-derive per-seed rows itself.
    """
    eligible_values = [v for v in per_seed_delta_eas.values() if v is not None]
    if len(eligible_values) < min_eligible_seeds:
        return ProblemAggregate(
            problem_id=problem_id,
            delta_eas=None,
            n_eligible_seeds=len(eligible_values),
            per_seed_delta_eas=per_seed_delta_eas,
            undefined_reason=f"only_{len(eligible_values)}_of_{len(per_seed_delta_eas)}_seeds_eligible",
        )
    return ProblemAggregate(
        problem_id=problem_id,
        delta_eas=sum(eligible_values) / len(eligible_values),
        n_eligible_seeds=len(eligible_values),
        per_seed_delta_eas=per_seed_delta_eas,
    )
