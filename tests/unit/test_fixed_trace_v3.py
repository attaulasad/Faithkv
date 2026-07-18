"""Unit tests for the protocol-v3 additions to kvcot.analysis.fixed_trace
(CHANGELOG.md 2026-07-17): meaningful-compression eligibility gating, the
CPSS/Delta_CPSS metric, per-fraction retention/compression summaries,
eligible-only curves, and the natural-accuracy pilot screen. Every existing
protocol-v2 test in test_fixed_trace_analysis.py must keep passing unchanged
-- these tests only cover the NEW, additive behavior.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kvcot.analysis.fixed_trace import (
    _verify_selection_completeness,
    build_accuracy_screen,
    build_fixed_trace_decision,
    build_fixed_trace_pairs,
    build_screen_validity,
    build_strict_accuracy_gate,
    compute_cpss,
    compute_delta_cpss,
    fixed_trace_curve_by_fraction_eligible_only,
    load_fixed_trace_records,
    retention_summary_by_fraction,
)
from kvcot.config import FixedTraceSettings, PROBE_FRACTIONS_ALL, PROBE_FRACTIONS_SCORED
from kvcot.utils.io import JsonlWriter


# Self-contained fixture helpers (deliberately not imported from
# test_fixed_trace_analysis.py -- tests/unit has no __init__.py, so relative
# imports between test modules don't work under this repo's pytest layout).

def _base_record(src_idx: int, seed: int, *, cap_hit: bool = False, is_correct: bool = True) -> dict:
    return {
        "record_id": f"base-full-ds-{src_idx}-seed{seed}",
        "global_seed": seed,
        "dataset": {"source_row_index": src_idx},
        "cap_hit": cap_hit,
        "is_correct": is_correct,
        "think_span": {"think_parse_status": "generation_prompt_preopened_ok"},
    }


def _fixed_probe_record(
    base_record_id: str,
    fraction: float,
    *,
    trace_source_condition: str = "full",
    replay_policy_condition: str = "full",
    matches_anchor: bool | None,
    f1_anchor_is_correct: bool | None = True,
    probe_extraction_status: str = "boxed",
    actual_compression_at_cut: bool = True,
    probe_actual_eviction_during_answer: bool = False,
    retention_ratio: float = 0.5,
) -> dict:
    return {
        "record_id": f"fixed-probe-x-on-full-{base_record_id}-f{fraction}",
        "base_record_id": base_record_id,
        "fraction": fraction,
        "trace_source_condition": trace_source_condition,
        "replay_policy_condition": replay_policy_condition,
        "matches_f1_anchor_answer": matches_anchor,
        "f1_anchor_is_correct": f1_anchor_is_correct,
        "replay_compaction_count_at_cut": 3,
        "probe_extraction_status": probe_extraction_status,
        "actual_compression_at_cut": actual_compression_at_cut,
        "probe_actual_eviction_during_answer": probe_actual_eviction_during_answer,
        "replay_retention_at_cut": {
            "fullkv_equivalent_slots": 200,
            "physical_cache_slots_per_layer": [int(200 * retention_ratio)],
            "instantaneous_retention_ratio": retention_ratio,
            "post_compaction_budget_tokens": 512,
            "tokens_since_last_compaction": 5,
        },
    }


def _write(path: Path, rows: list[dict]) -> None:
    w = JsonlWriter(path, validator=None)
    for r in rows:
        w.append(r)


# --- compute_cpss / compute_delta_cpss ---

def test_compute_cpss_none_below_min_active_fractions():
    matches = {f: True for f in PROBE_FRACTIONS_SCORED}
    assert compute_cpss(matches, active_fractions={0.5}, min_active_fractions=2) is None


def test_compute_cpss_restricted_to_active_fractions_only():
    matches = {0.125: True, 0.25: False, 0.5: False, 0.625: True}
    # Only 0.25 and 0.5 are "active" -- both mismatches -> CPSS = 1.0,
    # ignoring 0.125/0.625 entirely even though they're present in matches.
    cpss = compute_cpss(matches, active_fractions={0.25, 0.5}, min_active_fractions=2)
    assert cpss == pytest.approx(1.0)


def test_compute_cpss_none_when_active_fraction_missing_from_matches():
    matches = {0.125: True, 0.25: True}
    assert compute_cpss(matches, active_fractions={0.25, 0.5}, min_active_fractions=2) is None


def test_compute_cpss_none_when_active_fraction_match_undefined():
    matches = {0.25: None, 0.5: True}
    assert compute_cpss(matches, active_fractions={0.25, 0.5}, min_active_fractions=2) is None


def test_compute_delta_cpss_sign_convention():
    assert compute_delta_cpss(0.8, 0.3) == pytest.approx(0.5)
    assert compute_delta_cpss(0.2, 0.6) == pytest.approx(-0.4)


def test_compute_delta_cpss_undefined_propagates():
    assert compute_delta_cpss(None, 0.3) is None
    assert compute_delta_cpss(0.3, None) is None


# --- meaningful-compression eligibility gate is OFF by default (v2 semantics preserved) ---

def test_meaningful_compression_gate_off_by_default_v2_semantics_unchanged(tmp_path):
    # rkv retention 0.9959 (protocol v2's actual "148" example) -- any-eviction
    # gate passes, meaningful-compression gate would fail it, but since
    # require_meaningful_compression defaults to False, eligibility must be
    # unaffected by the new field entirely.
    base, base_records, full_probes, rkv_probes = _make_run(
        tmp_path, retention=0.9959, rkv_actual_compression=True,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is True
    assert pairs[0].eligibility.rkv_meaningful_compression_at_f1 is False  # diagnostic, computed regardless


def test_meaningful_compression_gate_rejects_high_retention_when_required(tmp_path):
    settings = FixedTraceSettings(
        min_eligible_examples=1, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0,
        require_meaningful_compression=True, meaningful_retention_ceiling=0.7,
        min_meaningfully_compressed_scored_fractions=2,
    )
    base, base_records, full_probes, rkv_probes = _make_run(
        tmp_path, retention=0.9959, rkv_actual_compression=True, scored_retention=0.99,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes, settings=settings)
    assert pairs[0].eligibility.eligible is False
    assert "rkv_no_meaningful_compression_at_f1" in pairs[0].eligibility.failure_reasons


def test_meaningful_compression_gate_accepts_low_retention_when_required(tmp_path):
    settings = FixedTraceSettings(
        min_eligible_examples=1, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0,
        require_meaningful_compression=True, meaningful_retention_ceiling=0.7,
        min_meaningfully_compressed_scored_fractions=2,
    )
    base, base_records, full_probes, rkv_probes = _make_run(
        tmp_path, retention=0.5, rkv_actual_compression=True, scored_retention=0.5,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes, settings=settings)
    assert pairs[0].eligibility.eligible is True
    assert pairs[0].eligibility.rkv_meaningful_compression_at_f1 is True


def test_too_few_meaningfully_compressed_scored_fractions_rejects(tmp_path):
    settings = FixedTraceSettings(
        min_eligible_examples=1, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0,
        require_meaningful_compression=True, meaningful_retention_ceiling=0.7,
        min_meaningfully_compressed_scored_fractions=5,  # need 5, only give 1 low-retention fraction
    )
    base, base_records, full_probes, rkv_probes = _make_run(
        tmp_path, retention=0.5, rkv_actual_compression=True, scored_retention=0.5, n_low_retention_scored=1,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes, settings=settings)
    assert pairs[0].eligibility.eligible is False
    assert "too_few_meaningfully_compressed_scored_fractions" in pairs[0].eligibility.failure_reasons


# --- FullKV and R-KV use identical active-fraction sets for CPSS ---

def test_cpss_active_fraction_set_is_identical_for_both_sides(tmp_path):
    settings = FixedTraceSettings(min_compressed_scored_fractions_for_cpss=2)
    base, base_records, full_probes, rkv_probes = _make_run(
        tmp_path, retention=0.5, rkv_actual_compression=True, scored_retention=0.5, n_low_retention_scored=3,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes, settings=settings)
    p = pairs[0]
    # Both cpss_full and cpss_rkv are defined and were computed over the SAME
    # active_scored_fractions set (there is only one such set stored on the
    # pair result, used for both sides).
    assert p.cpss_full is not None
    assert p.cpss_rkv is not None
    assert len(p.active_scored_fractions) == 3


def test_cpss_undefined_when_no_fractions_clear_ceiling(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_run(
        tmp_path, retention=0.99, rkv_actual_compression=True, scored_retention=0.99,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].cpss_full is None
    assert pairs[0].cpss_rkv is None
    assert pairs[0].delta_cpss is None


# --- retention_summary_by_fraction / eligible-only curves ---

def test_retention_summary_by_fraction_reports_distribution(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_run(
        tmp_path, retention=0.5, rkv_actual_compression=True, scored_retention=0.6,
    )
    summary = retention_summary_by_fraction(rkv_probes, meaningful_retention_ceiling=0.7)
    s = summary[1.0]
    assert s["count"] == 1
    assert s["mean"] == pytest.approx(0.5)
    assert s["meaningful_compression_rate"] == pytest.approx(1.0)


def test_retention_summary_none_for_fraction_with_no_data(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    # 0.9 is not one of PROBE_FRACTIONS_ALL, so no probe record exists for it.
    summary = retention_summary_by_fraction(rkv_probes, meaningful_retention_ceiling=0.7, fractions=(0.9,))
    assert summary[0.9] is None


def test_eligible_only_curve_excludes_ineligible_examples(tmp_path):
    # Two problems: one eligible (all matches True), one ineligible (cap hit).
    base_a, base_records_a, full_a, rkv_a = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True, src_idx=0)
    base_b, base_records_b, full_b, rkv_b = _make_run(
        tmp_path, retention=0.5, rkv_actual_compression=True, src_idx=1, cap_hit=True, subdir="b",
    )
    # Merge both problems' records into one shared pair of files.
    all_full = list(full_a.probes_by_base.items()) + list(full_b.probes_by_base.items())
    all_rkv = list(rkv_a.probes_by_base.items()) + list(rkv_b.probes_by_base.items())
    merged_full_path = tmp_path / "merged_full.jsonl"
    merged_rkv_path = tmp_path / "merged_rkv.jsonl"
    _write(merged_full_path, [rec for _, group in all_full for rec in group.values()])
    _write(merged_rkv_path, [rec for _, group in all_rkv for rec in group.values()])
    full_probes = load_fixed_trace_records(merged_full_path, "full")
    rkv_probes = load_fixed_trace_records(merged_rkv_path, "rkv_b512")
    base_records = [base_a, base_b]

    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    eligible_ids = {p.base_record_id for p in pairs if p.eligibility.eligible}
    assert len(eligible_ids) == 1

    curve_all = fixed_trace_curve_by_fraction_eligible_only(rkv_probes, {base_a["record_id"], base_b["record_id"]})
    curve_eligible = fixed_trace_curve_by_fraction_eligible_only(rkv_probes, eligible_ids)
    # Eligible-only curve is computed from strictly fewer (or equal) examples.
    assert curve_eligible[1.0] is not None
    assert curve_all[1.0] is not None


# --- build_accuracy_screen ---

def _natural_base_record(src_idx: int, seed: int, is_correct: bool) -> dict:
    return {
        "dataset": {"source_row_index": src_idx},
        "global_seed": seed,
        "is_correct": is_correct,
    }


def test_accuracy_screen_computes_paired_difference():
    full_records = [_natural_base_record(i, 42, True) for i in range(10)]
    rkv_records = [_natural_base_record(i, 42, i != 0) for i in range(10)]  # one flip
    settings = FixedTraceSettings(max_pilot_accuracy_drop=0.10)
    screen = build_accuracy_screen(full_records, rkv_records, settings)
    assert screen["n_accuracy_pairs"] == 10
    assert screen["full_accuracy"] == pytest.approx(1.0)
    assert screen["rkv_accuracy"] == pytest.approx(0.9)
    assert screen["accuracy_difference_rkv_minus_full"] == pytest.approx(-0.1)
    assert screen["pilot_accuracy_plausible"] is True  # exactly at the -0.10 boundary


def test_accuracy_screen_implausible_when_drop_exceeds_threshold():
    full_records = [_natural_base_record(i, 42, True) for i in range(10)]
    rkv_records = [_natural_base_record(i, 42, i < 5) for i in range(10)]  # 50% accuracy
    settings = FixedTraceSettings(max_pilot_accuracy_drop=0.10)
    screen = build_accuracy_screen(full_records, rkv_records, settings)
    assert screen["accuracy_difference_rkv_minus_full"] == pytest.approx(-0.5)
    assert screen["pilot_accuracy_plausible"] is False


def test_accuracy_screen_only_pairs_shared_keys():
    full_records = [_natural_base_record(0, 42, True), _natural_base_record(1, 42, True)]
    rkv_records = [_natural_base_record(0, 42, True)]  # row 1 missing entirely
    settings = FixedTraceSettings()
    screen = build_accuracy_screen(full_records, rkv_records, settings)
    assert screen["n_accuracy_pairs"] == 1


def test_screen_validity_invalidated_by_missing_accuracy_screen():
    settings = FixedTraceSettings(min_eligible_examples=1, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0)
    missing_screen = {"pilot_accuracy_plausible": False, "accuracy_difference_rkv_minus_full": None}
    valid, reasons = build_screen_validity(
        n_eligible=5, actual_compression_rate=1.0, mean_f1_rkv_retention_ratio=0.3, settings=settings,
        accuracy_screen=missing_screen,
    )
    assert valid is False
    assert any("pilot_accuracy_plausible" in r for r in reasons)


def test_screen_validity_gates_on_meaningful_not_actual_compression_rate_when_required():
    # 2026-07-18 review: require_meaningful_compression=True must gate the
    # SCREEN on meaningful_compression_rate, not actual_compression_rate --
    # a batch where every example evicted at least one token (actual=1.0)
    # but almost none evicted enough to be "meaningful" (meaningful=0.1)
    # must NOT pass the screen just because actual_compression_rate looks
    # perfect.
    settings = FixedTraceSettings(
        min_eligible_examples=1, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0,
        require_meaningful_compression=True, min_meaningful_compression_rate=0.7,
    )
    valid, reasons = build_screen_validity(
        n_eligible=5, actual_compression_rate=1.0, mean_f1_rkv_retention_ratio=0.3, settings=settings,
        meaningful_compression_rate=0.1,
    )
    assert valid is False
    assert any("meaningful_compression_rate" in r for r in reasons)


def test_screen_validity_passes_on_high_meaningful_compression_rate_when_required():
    settings = FixedTraceSettings(
        min_eligible_examples=1, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0,
        require_meaningful_compression=True, min_meaningful_compression_rate=0.7,
    )
    valid, reasons = build_screen_validity(
        n_eligible=5, actual_compression_rate=1.0, mean_f1_rkv_retention_ratio=0.3, settings=settings,
        meaningful_compression_rate=0.8,
    )
    assert valid is True
    assert reasons == []


def test_screen_validity_v2_stages_still_gate_on_actual_compression_rate():
    # require_meaningful_compression defaults to False (v2 semantics) --
    # meaningful_compression_rate must be completely ignored, even if given.
    settings = FixedTraceSettings(min_eligible_examples=1, min_actual_compression_rate=0.7, max_mean_f1_retention_ratio=1.0)
    valid, reasons = build_screen_validity(
        n_eligible=5, actual_compression_rate=0.8, mean_f1_rkv_retention_ratio=0.3, settings=settings,
        meaningful_compression_rate=0.0,  # would fail if mistakenly checked
    )
    assert valid is True
    assert reasons == []


def test_decision_meaningful_compression_rate_drives_v3_screen_validity(tmp_path):
    # End-to-end: build_fixed_trace_decision must actually wire
    # meaningful_compression_rate into build_screen_validity when
    # require_meaningful_compression=True.
    settings = FixedTraceSettings(
        min_eligible_examples=1, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0,
        require_meaningful_compression=True, meaningful_retention_ceiling=0.7,
        min_meaningful_compression_rate=0.7,
    )
    # retention=0.9 -> actual_compression_at_cut True (still < fullkv-equivalent
    # in the fixture's construction) but NOT meaningful (0.9 > 0.7 ceiling).
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.9, rkv_actual_compression=True)
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes, settings=settings)
    decision = build_fixed_trace_decision(
        len(pairs), pairs, full_curve={}, rkv_curve={}, settings=settings,
        full_probes=full_probes, rkv_probes=rkv_probes,
    )
    assert decision["meaningful_compression_rate"] == pytest.approx(0.0)
    assert decision["screen_valid"] is False
    assert any("meaningful_compression_rate" in r for r in decision["screen_invalid_reasons"])


def test_screen_validity_unaffected_when_accuracy_screen_omitted():
    # v2 callers never pass accuracy_screen -- must not spuriously invalidate.
    settings = FixedTraceSettings(min_eligible_examples=1, min_actual_compression_rate=0.5, max_mean_f1_retention_ratio=0.7)
    valid, reasons = build_screen_validity(
        n_eligible=5, actual_compression_rate=0.8, mean_f1_rkv_retention_ratio=0.4, settings=settings,
    )
    assert valid is True
    assert reasons == []


# --- decision JSON additive keys never break existing consumers ---

def test_decision_json_carries_all_shared_curve_aliases_and_new_keys(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    decision = build_fixed_trace_decision(
        len(pairs), pairs, full_curve={1.0: 1.0}, rkv_curve={1.0: 1.0}, settings=FixedTraceSettings(min_eligible_examples=1),
        full_probes=full_probes, rkv_probes=rkv_probes,
    )
    # Old keys still present, unchanged (backward compatibility).
    assert decision["full_curve"] == {"1.0": 1.0}
    assert decision["rkv_curve"] == {"1.0": 1.0}
    # New aliases carry the exact same data.
    assert decision["all_shared_full_curve"] == decision["full_curve"]
    assert decision["all_shared_rkv_curve"] == decision["rkv_curve"]
    # New additive keys present.
    assert "full_curve_eligible_only" in decision
    assert "rkv_curve_eligible_only" in decision
    assert "retention_summary_by_fraction" in decision
    assert "compression_rate_by_fraction" in decision
    assert "mean_delta_cpss" in decision
    assert decision["accuracy_screen"] is None  # not passed in this call


def test_decision_json_still_json_serializable_with_new_fields(tmp_path):
    import json

    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    decision = build_fixed_trace_decision(
        len(pairs), pairs, full_curve={}, rkv_curve={}, settings=FixedTraceSettings(min_eligible_examples=1),
        full_probes=full_probes, rkv_probes=rkv_probes,
    )
    json.dumps(decision)


# --- helper: build a one-problem fixture with controllable retention values ---

def _make_run(
    tmp_path: Path,
    *,
    retention: float,
    rkv_actual_compression: bool,
    scored_retention: float | None = None,
    n_low_retention_scored: int | None = None,
    src_idx: int = 0,
    cap_hit: bool = False,
    subdir: str | None = None,
):
    """Like test_fixed_trace_analysis._make_fixed_trace_run, but exposes
    per-fraction retention_ratio control (needed for meaningful-compression/
    CPSS tests, which that helper's fixed retention_ratio=0.5 default
    doesn't expose)."""
    base = _base_record(src_idx, 42, cap_hit=cap_hit)
    scored_retention = scored_retention if scored_retention is not None else retention

    def _retention_for(f: float) -> float:
        if f == 1.0:
            return retention
        if n_low_retention_scored is not None:
            # Only the first N scored fractions (by PROBE_FRACTIONS_SCORED
            # order) get the low retention value; the rest stay high.
            idx = PROBE_FRACTIONS_SCORED.index(f) if f in PROBE_FRACTIONS_SCORED else -1
            return scored_retention if 0 <= idx < n_low_retention_scored else 0.99
        return scored_retention

    full_recs = [
        _fixed_probe_record(base["record_id"], f, matches_anchor=True, retention_ratio=_retention_for(f))
        for f in PROBE_FRACTIONS_ALL
    ]
    rkv_recs = [
        _fixed_probe_record(
            base["record_id"], f, matches_anchor=True, replay_policy_condition="rkv_b512",
            actual_compression_at_cut=rkv_actual_compression, retention_ratio=_retention_for(f),
        )
        for f in PROBE_FRACTIONS_ALL
    ]
    d = tmp_path / subdir if subdir else tmp_path
    d.mkdir(exist_ok=True)
    _write(d / "full.jsonl", [base])
    _write(d / "full_on_full_fixed_trace_probes.jsonl", full_recs)
    _write(d / "rkv_b512_on_full_fixed_trace_probes.jsonl", rkv_recs)

    base_records = [base]
    full_probes = load_fixed_trace_records(d / "full_on_full_fixed_trace_probes.jsonl", "full")
    rkv_probes = load_fixed_trace_records(d / "rkv_b512_on_full_fixed_trace_probes.jsonl", "rkv_b512")
    return base, base_records, full_probes, rkv_probes


# --- selection-file completeness guard (2026-07-18 review) ---

def test_verify_selection_completeness_passes_when_fully_covered(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    _verify_selection_completeness({base["record_id"]}, base_records, full_probes, rkv_probes)  # must not raise


def test_verify_selection_completeness_raises_when_selected_id_missing_from_base(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    with pytest.raises(ValueError, match="not present in the canonical base file"):
        _verify_selection_completeness({"nonexistent-base-id"}, base_records, full_probes, rkv_probes)


def test_verify_selection_completeness_raises_when_full_probes_missing_entirely(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    # Simulate a selected example whose FullKV replay never ran at all.
    empty_full_probes = load_fixed_trace_records(tmp_path / "empty_full.jsonl", "full")
    _write(tmp_path / "empty_full.jsonl", [])
    with pytest.raises(ValueError, match="missing from FullKV fixed-trace probes entirely"):
        _verify_selection_completeness({base["record_id"]}, base_records, empty_full_probes, rkv_probes)


def test_verify_selection_completeness_raises_when_a_fraction_is_missing(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    # Drop one fraction from the R-KV side's group to simulate an
    # incomplete (interrupted) replay.
    del rkv_probes.probes_by_base[base["record_id"]][0.5]
    with pytest.raises(ValueError, match=r"R-KV fraction set does not exactly match \(missing"):
        _verify_selection_completeness({base["record_id"]}, base_records, full_probes, rkv_probes)


def test_verify_selection_completeness_raises_on_unexpected_extra_fraction(tmp_path):
    # 2026-07-19 review: a stray extra (10th) fraction on a selected
    # example must be caught too -- the check must be exact-set equality,
    # not just "nothing missing".
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    rkv_probes.probes_by_base[base["record_id"]][0.95] = dict(
        rkv_probes.probes_by_base[base["record_id"]][1.0]
    )  # a bogus, non-canonical fraction value
    with pytest.raises(ValueError, match=r"R-KV fraction set does not exactly match \(.*unexpected extra"):
        _verify_selection_completeness({base["record_id"]}, base_records, full_probes, rkv_probes)


def test_verify_selection_completeness_raises_on_superset_probe_file(tmp_path):
    # 2026-07-19 review: a probe file containing MORE base_record_ids than
    # the selection accounts for must be a hard failure, never silently
    # filtered away before pairing.
    base, base_records, full_probes, rkv_probes = _make_run(tmp_path, retention=0.5, rkv_actual_compression=True)
    extra_base_id = "base-full-ds-999-seed42"
    rkv_probes.probes_by_base[extra_base_id] = dict(rkv_probes.probes_by_base[base["record_id"]])
    with pytest.raises(ValueError, match="NOT in the selection"):
        _verify_selection_completeness({base["record_id"]}, base_records, full_probes, rkv_probes)


# Note: end-to-end run_fixed_trace_analysis(selected_base_record_ids=...)
# tests live in test_fixed_trace_analysis.py, which already has the
# full-schema-valid BaseRunRecord/FixedTraceProbeRecord fixture builders
# run_fixed_trace_analysis's schema validation requires (_valid_base_run_
# record/_valid_fixed_trace_probe_record) -- the lightweight dicts here are
# sufficient for build_fixed_trace_pairs/_verify_selection_completeness
# directly, but not for the full pipeline's Pydantic validation step.


# --- build_strict_accuracy_gate (2026-07-18 review) ---

def _natural_schema_record(idx: int, seed: int, condition: str, is_correct: bool = True, **overrides) -> dict:
    from kvcot.schemas import (
        BaseRunRecord, DatasetProvenance, MethodConfig, ProvenanceState, ThinkSpanInfo, VersionInfo,
    )

    defaults = dict(
        record_id=f"base-{condition}-ds-{idx}-seed{seed}",
        config_path="configs/early_gap_v3_b128.yaml",
        config_sha256="a" * 64,
        provenance=ProvenanceState(upstream_rkv_commit="45eaa7d69d20b7388321f077020a610d9afb65bd", git_commit="deadbeef", git_dirty=False),
        versions=VersionInfo(python="3.10.0"),
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        model_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tokenizer_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        dataset=DatasetProvenance(dataset_name="gsm8k", source_row_index=idx, question_hash="b" * 64, normalized_gold="42"),
        condition=condition,
        method_config=MethodConfig(method="fullkv" if condition == "full" else "rkv"),
        global_seed=seed,
        derived_seed=123,
        prompt_text="hello",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5, 6],
        decoded_output="42",
        think_span=ThinkSpanInfo(think_start_index=0, think_end_index=3, think_parse_status="generation_prompt_preopened_ok", generation_prompt_preopened_think=True),
        extracted_answer="42",
        extraction_method="boxed",
        is_correct=is_correct,
        cap_hit=False,
        wall_time_seconds=1.0,
        generated_token_count=3,
        compaction_count=0,
        compaction_event_steps=[],
        cache_length_final_per_layer=[3, 3],
    )
    defaults.update(overrides)
    return BaseRunRecord(**defaults).model_dump(mode="json")


def _identity(rec: dict) -> tuple:
    return (rec["config_sha256"], rec["provenance"]["upstream_rkv_commit"], rec["model_revision"], rec["tokenizer_revision"])


def test_strict_accuracy_gate_passes_with_complete_matching_data():
    full_records = [_natural_schema_record(i, 42, "full") for i in range(10)]
    rkv_records = [_natural_schema_record(i, 42, "rkv_b128") for i in range(10)]
    settings = FixedTraceSettings(max_pilot_accuracy_drop=0.10)
    gate = build_strict_accuracy_gate(full_records, rkv_records, expected_n=10, settings=settings)
    assert gate["gate_passed"] is True
    assert gate["reasons"] == []


def test_strict_accuracy_gate_fails_on_missing_records_even_with_one_matching_pair():
    # The exact scenario the review flagged: one matching pair, 9 missing --
    # build_accuracy_screen alone would compute a "perfect" n=1 comparison;
    # the strict gate must fail on count/key-set grounds regardless.
    full_records = [_natural_schema_record(i, 42, "full") for i in range(10)]
    rkv_records = [_natural_schema_record(0, 42, "rkv_b128")]  # only 1 of 10
    settings = FixedTraceSettings(max_pilot_accuracy_drop=0.10)
    gate = build_strict_accuracy_gate(full_records, rkv_records, expected_n=10, settings=settings)
    assert gate["gate_passed"] is False
    assert any("rkv record count" in r for r in gate["reasons"])
    assert any("key sets differ" in r for r in gate["reasons"])


def test_strict_accuracy_gate_fails_on_duplicate_keys():
    full_records = [_natural_schema_record(0, 42, "full", record_id="dup-a"), _natural_schema_record(0, 42, "full", record_id="dup-b")]
    rkv_records = [_natural_schema_record(0, 42, "rkv_b128")]
    settings = FixedTraceSettings()
    gate = build_strict_accuracy_gate(full_records, rkv_records, expected_n=1, settings=settings)
    assert gate["gate_passed"] is False
    assert any("duplicate" in r for r in gate["reasons"])


def test_strict_accuracy_gate_fails_on_schema_invalid_record():
    full_records = [_natural_schema_record(0, 42, "full")]
    bad = _natural_schema_record(0, 42, "rkv_b128")
    del bad["is_correct"]  # required field missing -> schema invalid
    gate = build_strict_accuracy_gate(full_records, [bad], expected_n=1, settings=FixedTraceSettings())
    assert gate["gate_passed"] is False
    assert any("failed schema validation" in r for r in gate["reasons"])


def test_strict_accuracy_gate_fails_on_identity_mismatch_within_rkv_records():
    full_records = [_natural_schema_record(i, 42, "full") for i in range(2)]
    rkv_records = [
        _natural_schema_record(0, 42, "rkv_b128"),
        _natural_schema_record(1, 42, "rkv_b128", model_revision="some-other-revision"),
    ]
    gate = build_strict_accuracy_gate(full_records, rkv_records, expected_n=2, settings=FixedTraceSettings())
    assert gate["gate_passed"] is False
    assert any("different identities" in r for r in gate["reasons"])


def test_strict_accuracy_gate_fails_against_mismatched_expected_identity():
    full_records = [_natural_schema_record(i, 42, "full") for i in range(2)]
    rkv_records = [_natural_schema_record(i, 42, "rkv_b128") for i in range(2)]
    mismatched_expected = ("a" * 64, "45eaa7d69d20b7388321f077020a610d9afb65bd", "some-other-model-revision", "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562")
    gate = build_strict_accuracy_gate(
        full_records, rkv_records, expected_n=2, settings=FixedTraceSettings(), expected_identity=mismatched_expected,
    )
    assert gate["gate_passed"] is False
    assert any("do not share one" in r for r in gate["reasons"])


def test_strict_accuracy_gate_passes_against_matching_expected_identity():
    full_records = [_natural_schema_record(i, 42, "full") for i in range(2)]
    rkv_records = [_natural_schema_record(i, 42, "rkv_b128") for i in range(2)]
    matching_expected = _identity(full_records[0])
    gate = build_strict_accuracy_gate(
        full_records, rkv_records, expected_n=2, settings=FixedTraceSettings(), expected_identity=matching_expected,
    )
    assert gate["gate_passed"] is True


def test_strict_accuracy_gate_fails_on_wrong_condition_field():
    mislabeled = _natural_schema_record(0, 42, "full")
    mislabeled["condition"] = "rkv_b128"  # as if the wrong file were passed in as "full"
    full_records = [mislabeled]
    rkv_records = [_natural_schema_record(0, 42, "rkv_b128")]
    gate = build_strict_accuracy_gate(full_records, rkv_records, expected_n=1, settings=FixedTraceSettings())
    assert gate["gate_passed"] is False
    assert any("condition field is 'full'" in r for r in gate["reasons"])


def test_strict_accuracy_gate_accepts_wrong_rkv_condition_without_expected_rkv_condition():
    # Exact repro from the 2026-07-19 review: the "rkv" file actually
    # contains condition="full" records (e.g. the wrong file was passed as
    # --replay-condition), but counts/keys/identities all match, and the
    # internal-consistency-only check (no expected_rkv_condition given)
    # cannot catch it -- documents the gap this test's sibling below closes.
    full_records = [_natural_schema_record(i, 42, "full") for i in range(5)]
    mislabeled_rkv = [_natural_schema_record(i, 42, "full") for i in range(5)]  # should be rkv_b128
    gate = build_strict_accuracy_gate(full_records, mislabeled_rkv, expected_n=5, settings=FixedTraceSettings())
    assert gate["gate_passed"] is True  # the gap: passes despite being the wrong file


def test_strict_accuracy_gate_rejects_wrong_rkv_condition_when_expected_given():
    # The fix: passing expected_rkv_condition (as kvcot.cli.
    # cmd_check_fixed_trace_accuracy now always does, wired from
    # --replay-condition) catches exactly the scenario above.
    full_records = [_natural_schema_record(i, 42, "full") for i in range(5)]
    mislabeled_rkv = [_natural_schema_record(i, 42, "full") for i in range(5)]
    gate = build_strict_accuracy_gate(
        full_records, mislabeled_rkv, expected_n=5, settings=FixedTraceSettings(),
        expected_rkv_condition="rkv_b128",
    )
    assert gate["gate_passed"] is False
    assert any("this gate was asked to check" in r for r in gate["reasons"])


def test_strict_accuracy_gate_accepts_correct_rkv_condition_when_expected_given():
    full_records = [_natural_schema_record(i, 42, "full") for i in range(5)]
    rkv_records = [_natural_schema_record(i, 42, "rkv_b128") for i in range(5)]
    gate = build_strict_accuracy_gate(
        full_records, rkv_records, expected_n=5, settings=FixedTraceSettings(),
        expected_rkv_condition="rkv_b128",
    )
    assert gate["gate_passed"] is True


def test_strict_accuracy_gate_fails_on_accuracy_drop_even_with_complete_data():
    full_records = [_natural_schema_record(i, 42, "full", is_correct=True) for i in range(10)]
    rkv_records = [_natural_schema_record(i, 42, "rkv_b128", is_correct=(i < 5)) for i in range(10)]
    settings = FixedTraceSettings(max_pilot_accuracy_drop=0.10)
    gate = build_strict_accuracy_gate(full_records, rkv_records, expected_n=10, settings=settings)
    assert gate["gate_passed"] is False
    assert any("pilot_accuracy_plausible" in r for r in gate["reasons"])
    assert gate["accuracy_screen"]["accuracy_difference_rkv_minus_full"] == pytest.approx(-0.5)


# --- build_strict_accuracy_gate must NEVER raise on malformed input
# (2026-07-18 external review: the first pass extracted (source_row_index,
# global_seed) keys by direct subscripting BEFORE schema validation, so a
# malformed record raised KeyError out of the "never raises" function) ---

def test_strict_accuracy_gate_returns_failure_for_missing_dataset():
    good = _natural_schema_record(0, 42, "full")
    bad = _natural_schema_record(1, 42, "full")
    del bad["dataset"]  # _key() would raise KeyError on this pre-fix
    gate = build_strict_accuracy_gate([good, bad], [_natural_schema_record(0, 42, "rkv_b128")], expected_n=2, settings=FixedTraceSettings())
    assert gate["gate_passed"] is False
    assert any("failed schema validation" in r for r in gate["reasons"])
    assert gate["accuracy_screen"] is None  # never computed over malformed records
    assert gate["n_full"] == 2 and gate["n_rkv"] == 1  # stable shape retained


def test_strict_accuracy_gate_returns_failure_for_missing_seed():
    bad = _natural_schema_record(0, 42, "rkv_b128")
    del bad["global_seed"]
    gate = build_strict_accuracy_gate([_natural_schema_record(0, 42, "full")], [bad], expected_n=1, settings=FixedTraceSettings())
    assert gate["gate_passed"] is False
    assert any("failed schema validation" in r for r in gate["reasons"])
    assert gate["accuracy_screen"] is None


def test_strict_accuracy_gate_returns_failure_for_non_dict_record():
    gate = build_strict_accuracy_gate(
        [["not", "a", "dict"]], [42], expected_n=1, settings=FixedTraceSettings(),
    )
    assert gate["gate_passed"] is False
    assert any("failed schema validation" in r for r in gate["reasons"])
    assert gate["accuracy_screen"] is None
    # Full stable shape even on garbage input.
    assert set(gate) >= {"gate_passed", "reasons", "expected_n", "n_full", "n_rkv", "accuracy_screen"}


def test_strict_accuracy_gate_valid_records_still_checked_alongside_invalid_ones():
    # Count/key checks still run over whatever IS valid -- one malformed row
    # must not suppress the other diagnostics.
    good_full = [_natural_schema_record(i, 42, "full") for i in range(3)]
    bad = _natural_schema_record(3, 42, "full")
    del bad["dataset"]
    gate = build_strict_accuracy_gate(good_full + [bad], [_natural_schema_record(0, 42, "rkv_b128")], expected_n=4, settings=FixedTraceSettings())
    assert gate["gate_passed"] is False
    assert any("rkv record count" in r for r in gate["reasons"])
    assert any("failed schema validation" in r for r in gate["reasons"])
