import pytest

from kvcot.analysis.metrics import (
    compute_match,
    compute_eas,
    compute_delta_eas,
    EligibilityCheck,
    think_parsed_ok,
    aggregate_problem_delta_eas,
)
from kvcot.analysis.stats import (
    wilcoxon_delta_eas,
    bootstrap_ci_mean,
    paired_accuracy_diff,
)
from kvcot.config import PROBE_FRACTIONS_SCORED


def test_compute_match_defined_cases():
    assert compute_match("5", "5") is True
    assert compute_match("5", "6") is False


def test_compute_match_undefined_on_extraction_failure():
    assert compute_match(None, "5") is None
    assert compute_match("5", None) is None
    assert compute_match(None, None) is None


def test_compute_eas_all_mismatched_gives_1():
    matches = {f: False for f in PROBE_FRACTIONS_SCORED}
    r = compute_eas(matches)
    assert r.value == pytest.approx(1.0)
    assert r.undefined_reason is None
    assert r.n_fractions_used == 7


def test_compute_eas_all_matched_gives_0():
    matches = {f: True for f in PROBE_FRACTIONS_SCORED}
    r = compute_eas(matches)
    assert r.value == pytest.approx(0.0)


def test_compute_eas_partial_known_value():
    # 3 mismatches out of 7 -> EAS = 3/7
    matches = {f: True for f in PROBE_FRACTIONS_SCORED}
    fs = list(PROBE_FRACTIONS_SCORED)
    matches[fs[0]] = False
    matches[fs[1]] = False
    matches[fs[2]] = False
    r = compute_eas(matches)
    assert r.value == pytest.approx(3 / 7)


def test_compute_eas_f0_and_f1_never_enter_even_if_present():
    matches = {f: True for f in PROBE_FRACTIONS_SCORED}
    matches[0.0] = False  # should be ignored; f=0 is not in PROBE_FRACTIONS_SCORED
    matches[1.0] = False  # should be ignored; f=1 is not in PROBE_FRACTIONS_SCORED
    r = compute_eas(matches)
    assert r.value == pytest.approx(0.0)


def test_compute_eas_undefined_when_fraction_missing():
    fs = list(PROBE_FRACTIONS_SCORED)
    matches = {f: True for f in fs[:-1]}  # drop the last scored fraction
    r = compute_eas(matches)
    assert r.value is None
    assert "missing_probe_at_fraction" in r.undefined_reason


def test_compute_eas_undefined_when_match_undefined():
    matches = {f: True for f in PROBE_FRACTIONS_SCORED}
    fs = list(PROBE_FRACTIONS_SCORED)
    matches[fs[0]] = None  # extraction failure at this fraction
    r = compute_eas(matches)
    assert r.value is None
    assert "undefined_match_at_fraction" in r.undefined_reason


def test_delta_eas_sign_convention_positive_means_rkv_less_sensitive():
    # FullKV changes its answer more often (higher EAS) than R-KV under
    # truncation -> R-KV is less sensitive -> Delta_EAS should be positive.
    eas_full = 0.8
    eas_rkv = 0.3
    delta = compute_delta_eas(eas_full, eas_rkv)
    assert delta == pytest.approx(0.5)
    assert delta > 0


def test_delta_eas_sign_convention_negative_means_rkv_more_sensitive():
    eas_full = 0.2
    eas_rkv = 0.6
    delta = compute_delta_eas(eas_full, eas_rkv)
    assert delta == pytest.approx(-0.4)
    assert delta < 0


def test_delta_eas_undefined_propagates():
    assert compute_delta_eas(None, 0.3) is None
    assert compute_delta_eas(0.3, None) is None


def test_think_parsed_ok_statuses():
    assert think_parsed_ok("ok") is True
    assert think_parsed_ok("generation_prompt_preopened_ok") is True
    assert think_parsed_ok("no_close_marker") is False
    assert think_parsed_ok("no_open_marker") is False


def test_eligibility_all_true_is_eligible():
    e = EligibilityCheck(
        full_base_correct=True,
        rkv_base_correct=True,
        full_think_parsed=True,
        rkv_think_parsed=True,
        full_f1_stable=True,
        rkv_f1_stable=True,
        rkv_had_compaction=True,
        all_scored_probes_present=True,
    )
    assert e.eligible is True
    assert e.failure_reasons == []


def test_eligibility_single_failure_reported():
    e = EligibilityCheck(
        full_base_correct=True,
        rkv_base_correct=False,
        full_think_parsed=True,
        rkv_think_parsed=True,
        full_f1_stable=True,
        rkv_f1_stable=True,
        rkv_had_compaction=True,
        all_scored_probes_present=True,
    )
    assert e.eligible is False
    assert e.failure_reasons == ["rkv_base_incorrect"]


def test_eligibility_requires_compaction():
    e = EligibilityCheck(
        full_base_correct=True,
        rkv_base_correct=True,
        full_think_parsed=True,
        rkv_think_parsed=True,
        full_f1_stable=True,
        rkv_f1_stable=True,
        rkv_had_compaction=False,
        all_scored_probes_present=True,
    )
    assert e.eligible is False
    assert "rkv_no_compaction" in e.failure_reasons


def test_aggregate_problem_delta_eas_two_of_three_eligible():
    per_seed = {13: 0.2, 42: None, 2026: 0.4}
    agg = aggregate_problem_delta_eas("p1", per_seed)
    assert agg.delta_eas == pytest.approx(0.3)
    assert agg.n_eligible_seeds == 2
    assert agg.undefined_reason is None


def test_aggregate_problem_delta_eas_only_one_eligible_is_undefined():
    per_seed = {13: 0.2, 42: None, 2026: None}
    agg = aggregate_problem_delta_eas("p1", per_seed)
    assert agg.delta_eas is None
    assert agg.n_eligible_seeds == 1
    assert "only_1_of_3" in agg.undefined_reason


def test_aggregate_problem_delta_eas_produces_exactly_one_number():
    # Structural guard against pooling (problem, seed) as independent samples.
    per_seed = {13: 0.1, 42: 0.2, 2026: 0.3}
    agg = aggregate_problem_delta_eas("p1", per_seed)
    assert isinstance(agg.delta_eas, float)


def test_wilcoxon_pratt_vs_wilcox_differ_with_zeros_present():
    deltas = [0.0, 0.0, 0.1, 0.2, -0.1, 0.3, -0.2, 0.15]
    pratt = wilcoxon_delta_eas(deltas, zero_method="pratt")
    wilcox = wilcoxon_delta_eas(deltas, zero_method="wilcox")
    assert pratt.n_zeros == 2
    assert wilcox.n_zeros == 2  # zero count reported regardless of method
    assert pratt.n_used == len(deltas)
    # wilcox drops zeros before ranking -> different statistic in general
    assert pratt.statistic != wilcox.statistic or pratt.p_value != wilcox.p_value


def test_wilcoxon_zero_count_always_reported():
    deltas = [0.0, 0.1, 0.2, -0.1]
    r = wilcoxon_delta_eas(deltas)
    assert r.n_zeros == 1
    assert r.zero_method == "pratt"


def test_bootstrap_ci_mean_reproducible_with_fixed_seed():
    values = [0.1, 0.2, -0.1, 0.3, 0.0, 0.15, -0.05]
    a = bootstrap_ci_mean(values, n_resamples=500, seed=42)
    b = bootstrap_ci_mean(values, n_resamples=500, seed=42)
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high
    assert a.point_estimate == pytest.approx(sum(values) / len(values))


def test_bootstrap_ci_mean_constant_values_collapses():
    values = [0.5] * 10
    r = bootstrap_ci_mean(values, n_resamples=200, seed=1)
    assert r.ci_low == pytest.approx(0.5)
    assert r.ci_high == pytest.approx(0.5)


def test_paired_accuracy_diff_known_values():
    full_correct = [True, True, False, True, False]
    rkv_correct = [True, False, False, True, True]
    r = paired_accuracy_diff(full_correct, rkv_correct, n_resamples=200, seed=1)
    assert r.full_accuracy == pytest.approx(3 / 5)
    assert r.rkv_accuracy == pytest.approx(3 / 5)
    assert r.diff_rkv_minus_full == pytest.approx(0.0)
    assert r.n_problems == 5


def test_paired_accuracy_diff_requires_equal_length():
    with pytest.raises(ValueError):
        paired_accuracy_diff([True, False], [True], n_resamples=10, seed=1)
