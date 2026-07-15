"""Attrition funnel, primary analysis summary, and Stage 1A/1B decision
JSON (§8.4, §10). Must never import torch (see
tests/unit/test_no_analysis_torch_import.py) — everything here runs from
JSONL records on a laptop.

Every human-facing summary string produced by this module embeds
`CLAIM_BOUNDARY_NOTICE` (§1: "Enforce this in docstrings and in every
generated summary string.") so a reader encountering a results table or
decision file in isolation still sees the claim boundary, not just readers
of the source code.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kvcot.analysis.metrics import EligibilityCheck
from kvcot.analysis.stats import BootstrapCI, PairedAccuracyResult, WilcoxonResult
from kvcot.probes.early_answering import CLAIM_BOUNDARY_NOTICE
from kvcot.utils.io import write_json

FUNNEL_STAGES: list[tuple[str, Callable[[dict], bool]]] = [
    ("attempted", lambda r: True),
    ("base_generated_no_error", lambda r: r.get("base_generated", False)),
    ("did_not_hit_cap", lambda r: not r.get("cap_hit", True)),
    ("think_span_parsed", lambda r: r.get("think_parsed", False)),
    ("answer_extracted", lambda r: r.get("extracted", False)),
    ("base_answer_correct", lambda r: r.get("correct", False)),
    ("f1_stability_passed", lambda r: r.get("f1_stable", False)),
]
RKV_ONLY_STAGE = (">=1_compaction", lambda r: r.get("compaction_occurred", False))
ELIGIBLE_STAGE_LABEL = "eligible_pairs"


def _funnel_counts(records: list[dict], is_rkv: bool) -> dict[str, int]:
    """Sequential AND-reduction: each stage's count is how many records
    survive every predicate up to and including that stage, so counts are
    monotonically non-increasing down the funnel — standard funnel
    semantics, and the only semantics that makes "differential" between
    conditions meaningful (§8.4)."""
    stages = list(FUNNEL_STAGES)
    if is_rkv:
        stages.append(RKV_ONLY_STAGE)
    counts: dict[str, int] = {}
    survivors = records
    for name, predicate in stages:
        survivors = [r for r in survivors if predicate(r)]
        counts[name] = len(survivors)
    counts[ELIGIBLE_STAGE_LABEL] = len(survivors)
    return counts


def build_attrition_funnel_table(full_records: list[dict], rkv_records: list[dict]) -> list[dict]:
    """§8.4: "Every eligibility filter is correlated with the treatment...
    Emit an explicit funnel." Returns rows ready to write as CSV: stage,
    n_full, n_rkv, differential (n_full - n_rkv; positive means R-KV lost
    more problems at that stage than FullKV did)."""
    full_counts = _funnel_counts(full_records, is_rkv=False)
    rkv_counts = _funnel_counts(rkv_records, is_rkv=True)

    rows: list[dict] = []
    for name, _ in FUNNEL_STAGES:
        rows.append(
            {
                "stage": name,
                "n_full": full_counts[name],
                "n_rkv": rkv_counts[name],
                "differential": full_counts[name] - rkv_counts[name],
            }
        )
    rows.append(
        {
            "stage": RKV_ONLY_STAGE[0],
            "n_full": None,  # not applicable to FullKV — never silently coerced to 0
            "n_rkv": rkv_counts[RKV_ONLY_STAGE[0]],
            "differential": None,
        }
    )
    rows.append(
        {
            "stage": ELIGIBLE_STAGE_LABEL,
            "n_full": None,  # eligibility is a paired (full, rkv) concept, not per-condition
            "n_rkv": None,
            "differential": None,
        }
    )
    return rows


def write_attrition_funnel_csv(rows: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "n_full", "n_rkv", "differential"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass(frozen=True)
class PrimaryAnalysisSummary:
    wilcoxon_pratt: WilcoxonResult
    wilcoxon_wilcox_sensitivity: WilcoxonResult
    delta_eas_bootstrap_ci: BootstrapCI
    accuracy_headline: PairedAccuracyResult
    n_problems_primary: int
    claim_boundary_notice: str = CLAIM_BOUNDARY_NOTICE


def build_primary_analysis_summary(
    problem_level_delta_eas: list[float],
    full_correct_per_problem: list[bool],
    rkv_correct_per_problem: list[bool],
    bootstrap_seed: int = 20260715,
) -> PrimaryAnalysisSummary:
    """§8.5: the frozen primary tests, run exactly once each, on
    already-aggregated one-number-per-problem inputs (never (problem, seed)
    rows — see kvcot.analysis.metrics.aggregate_problem_delta_eas, which is
    the only place that aggregation is allowed to happen)."""
    from kvcot.analysis.stats import bootstrap_ci_mean, paired_accuracy_diff, wilcoxon_delta_eas

    pratt = wilcoxon_delta_eas(problem_level_delta_eas, zero_method="pratt")
    wilcox = wilcoxon_delta_eas(problem_level_delta_eas, zero_method="wilcox")
    ci = bootstrap_ci_mean(problem_level_delta_eas, seed=bootstrap_seed)
    accuracy = paired_accuracy_diff(full_correct_per_problem, rkv_correct_per_problem, seed=bootstrap_seed)

    return PrimaryAnalysisSummary(
        wilcoxon_pratt=pratt,
        wilcoxon_wilcox_sensitivity=wilcox,
        delta_eas_bootstrap_ci=ci,
        accuracy_headline=accuracy,
        n_problems_primary=len(problem_level_delta_eas),
    )


def write_primary_analysis_json(summary: PrimaryAnalysisSummary, path: str | Path) -> None:
    write_json(
        path,
        {
            "claim_boundary_notice": summary.claim_boundary_notice,
            "n_problems_primary": summary.n_problems_primary,
            "wilcoxon_pratt": {
                "statistic": summary.wilcoxon_pratt.statistic,
                "p_value": summary.wilcoxon_pratt.p_value,
                "n_used": summary.wilcoxon_pratt.n_used,
                "n_zeros": summary.wilcoxon_pratt.n_zeros,
            },
            "wilcoxon_wilcox_sensitivity": {
                "statistic": summary.wilcoxon_wilcox_sensitivity.statistic,
                "p_value": summary.wilcoxon_wilcox_sensitivity.p_value,
                "n_used": summary.wilcoxon_wilcox_sensitivity.n_used,
                "n_zeros": summary.wilcoxon_wilcox_sensitivity.n_zeros,
            },
            "delta_eas_bootstrap_ci": {
                "point_estimate": summary.delta_eas_bootstrap_ci.point_estimate,
                "ci_low": summary.delta_eas_bootstrap_ci.ci_low,
                "ci_high": summary.delta_eas_bootstrap_ci.ci_high,
                "n_resamples": summary.delta_eas_bootstrap_ci.n_resamples,
                "seed": summary.delta_eas_bootstrap_ci.seed,
                "sign_convention": "positive => R-KV less sensitive to truncation than FullKV (EAS_full - EAS_rkv)",
            },
            "accuracy_headline": {
                "full_accuracy": summary.accuracy_headline.full_accuracy,
                "rkv_accuracy": summary.accuracy_headline.rkv_accuracy,
                "diff_rkv_minus_full": summary.accuracy_headline.diff_rkv_minus_full,
                "ci_low": summary.accuracy_headline.ci.ci_low,
                "ci_high": summary.accuracy_headline.ci.ci_high,
                "n_problems": summary.accuracy_headline.n_problems,
            },
        },
    )


def build_stage1a_measurability_decision(
    n_total: int, n_answer_changed_at_any_scored_fraction: int, min_fraction_changed: float = 0.10
) -> dict:
    """§10 Stage 1A: emits a machine-readable `recommendation`, never
    hard-coded favorable. Recommends switching to the frozen MATH-500
    backup if GSM8K shows insufficient measurement range (fewer than
    `min_fraction_changed` of both-correct problems ever change their
    answer at any scored fraction)."""
    fraction_changed = n_answer_changed_at_any_scored_fraction / n_total if n_total > 0 else 0.0
    recommendation = "proceed_on_gsm8k" if fraction_changed >= min_fraction_changed else "switch_to_math500_backup"
    return {
        "claim_boundary_notice": CLAIM_BOUNDARY_NOTICE,
        "n_total": n_total,
        "n_answer_changed_at_any_scored_fraction": n_answer_changed_at_any_scored_fraction,
        "fraction_changed": fraction_changed,
        "min_fraction_changed_threshold": min_fraction_changed,
        "recommendation": recommendation,
    }


def build_stage1b_budget_decision(
    budget: int,
    n_calibration: int,
    n_with_compaction: int,
    compaction_activation_threshold: float,
    full_accuracy_point_estimate: float,
    candidate_accuracy_ci: tuple[float, float],
) -> dict:
    """§10 Stage 1B: two independently-labeled gates. Compaction activation
    is a real, well-powered gate at n=50. The accuracy check is explicitly
    labeled `coarse_screen`, never `equivalence` — n=50 accuracy CIs are far
    too wide (~+/-14pp) to support an equivalence claim."""
    activation_fraction = n_with_compaction / n_calibration if n_calibration > 0 else 0.0
    activation_passed = activation_fraction >= compaction_activation_threshold
    ci_low, ci_high = candidate_accuracy_ci
    coarse_screen_passed = ci_low <= full_accuracy_point_estimate <= ci_high
    return {
        "claim_boundary_notice": CLAIM_BOUNDARY_NOTICE,
        "budget": budget,
        "compaction_activation_gate": {
            "threshold": compaction_activation_threshold,
            "observed_fraction": activation_fraction,
            "passed": activation_passed,
        },
        "accuracy_coarse_screen": {
            "label": "coarse_screen",
            "fullkv_accuracy_point_estimate": full_accuracy_point_estimate,
            "candidate_accuracy_ci": {"low": ci_low, "high": ci_high},
            "candidate_ci_excludes_fullkv_point_estimate": not coarse_screen_passed,
            "passed": coarse_screen_passed,
        },
        "overall_passed": activation_passed and coarse_screen_passed,
    }
