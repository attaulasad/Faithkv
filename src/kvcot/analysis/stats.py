"""Frozen statistical tests (§8.5). No other tests are implemented — the
brief is explicit that this list is exhaustive for the primary analysis;
anything else is a labeled sensitivity analysis, never a second "primary".

Must never import torch (see tests/unit/test_no_analysis_torch_import.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n_used: int
    n_zeros: int
    zero_method: str


def wilcoxon_delta_eas(problem_deltas: list[float], zero_method: str = "pratt") -> WilcoxonResult:
    """Two-sided Wilcoxon signed-rank test over problem-level Delta_EAS
    values (§8.5). `zero_method="pratt"` (zeros retained in ranking) is
    primary; call again with `zero_method="wilcox"` for the zero-drop
    sensitivity variant. The count of exact zeros is always reported,
    regardless of which variant is requested, since it does not depend on
    zero_method.
    """
    if zero_method not in ("pratt", "wilcox", "zsplit"):
        raise ValueError(f"unsupported zero_method: {zero_method}")
    arr = np.asarray(problem_deltas, dtype=float)
    n_zeros = int(np.sum(arr == 0.0))
    if len(arr) == 0:
        raise ValueError("wilcoxon_delta_eas requires at least one problem-level Delta_EAS")
    result = scipy_stats.wilcoxon(arr, zero_method=zero_method, alternative="two-sided")
    return WilcoxonResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        n_used=len(arr),
        n_zeros=n_zeros,
        zero_method=zero_method,
    )


@dataclass(frozen=True)
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    seed: int
    n_observations: int


def bootstrap_ci_mean(
    values: list[float],
    n_resamples: int = 10_000,
    seed: int = 20260715,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Percentile bootstrap 95% CI of the mean, resampled over problems
    (§8.5), fixed RNG seed so the CI is exactly reproducible across runs.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("bootstrap_ci_mean requires at least one observation")
    point_estimate = float(arr.mean())
    rng = np.random.default_rng(seed)
    resample_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample_means[i] = arr[idx].mean()
    alpha = 1.0 - confidence
    lo, hi = np.quantile(resample_means, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapCI(
        point_estimate=point_estimate,
        ci_low=float(lo),
        ci_high=float(hi),
        n_resamples=n_resamples,
        seed=seed,
        n_observations=n,
    )


@dataclass(frozen=True)
class PairedAccuracyResult:
    """Headline accuracy-match result (§8.5): paired base-accuracy
    difference FullKV vs. R-KV, one pair per problem (accuracy already
    aggregated per problem upstream of this function — see note below),
    with a bootstrap CI over problems. Required because "accuracy-
    preserving operating point" is the load-bearing phrase in the research
    question."""

    full_accuracy: float
    rkv_accuracy: float
    diff_rkv_minus_full: float
    ci: BootstrapCI
    n_problems: int


def paired_accuracy_diff(
    full_correct: list[bool],
    rkv_correct: list[bool],
    n_resamples: int = 10_000,
    seed: int = 20260715,
) -> PairedAccuracyResult:
    """`full_correct[i]`/`rkv_correct[i]` must already be one paired
    observation per problem (i.e. each problem contributes exactly one
    entry to each list, in matching order) — this function does not
    aggregate across seeds itself; that is metrics.py's job, so this
    module never silently pools (problem, seed) rows as independent
    samples (§8.3).
    """
    if len(full_correct) != len(rkv_correct):
        raise ValueError("full_correct and rkv_correct must be the same length (paired per problem)")
    n = len(full_correct)
    if n == 0:
        raise ValueError("paired_accuracy_diff requires at least one problem")
    full_arr = np.asarray(full_correct, dtype=float)
    rkv_arr = np.asarray(rkv_correct, dtype=float)
    diffs = rkv_arr - full_arr  # positive => R-KV more accurate than FullKV
    ci = bootstrap_ci_mean(diffs.tolist(), n_resamples=n_resamples, seed=seed)
    return PairedAccuracyResult(
        full_accuracy=float(full_arr.mean()),
        rkv_accuracy=float(rkv_arr.mean()),
        diff_rkv_minus_full=float(diffs.mean()),
        ci=ci,
        n_problems=n,
    )
