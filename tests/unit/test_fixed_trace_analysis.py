"""Unit tests for kvcot.analysis.fixed_trace — the secondary, additive
prefix-sufficiency (PSS/Delta_PSS) screen. Runs entirely from synthetic JSONL
dicts on the CPU, mirroring tests/unit/test_pipeline.py's/test_metrics.py's
style for the frozen EAS/Delta_EAS pipeline.

Protocol v2 (2026-07-16, CHANGELOG.md) fixtures: every fixed-trace probe
record now carries `probe_extraction_status` (must be "boxed" for an f=1
anchor to count), `actual_compression_at_cut` (realized, not just a
recorded compaction EVENT count), and `probe_actual_eviction_during_answer`.
The base record now also carries `is_correct` (canonical-trace correctness
is part of eligibility, not just "did not hit cap").
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kvcot.analysis.fixed_trace import (
    _assert_shared_trace_source,
    _validate_base_records,
    _validate_fixed_trace_probe_records,
    build_fixed_trace_decision,
    build_fixed_trace_pairs,
    build_screen_validity,
    compute_delta_pss,
    compute_pss,
    fixed_trace_curve_by_fraction,
    load_fixed_trace_records,
    match_rate_delta_rkv_minus_full,
    run_fixed_trace_analysis,
)
from kvcot.config import FixedTraceSettings, PROBE_FRACTIONS_ALL, PROBE_FRACTIONS_SCORED
from kvcot.schemas import BaseRunRecord, FixedTraceProbeRecord
from kvcot.utils.io import JsonlWriter, read_json, read_jsonl


def _base_record(
    src_idx: int,
    seed: int,
    *,
    cap_hit: bool = False,
    parse_status: str = "generation_prompt_preopened_ok",
    is_correct: bool = True,
) -> dict:
    return {
        "record_id": f"base-full-ds-{src_idx}-seed{seed}",
        "global_seed": seed,
        "dataset": {"source_row_index": src_idx},
        "cap_hit": cap_hit,
        "is_correct": is_correct,
        "think_span": {"think_parse_status": parse_status},
    }


def _fixed_probe_record(
    base_record_id: str,
    fraction: float,
    *,
    trace_source_condition: str = "full",
    replay_policy_condition: str = "full",
    matches_anchor: bool | None,
    f1_anchor_is_correct: bool | None = True,
    compaction_count: int = 3,
    probe_extraction_status: str = "boxed",
    actual_compression_at_cut: bool = True,
    probe_actual_eviction_during_answer: bool = False,
    retention_ratio: float = 0.5,
    **extra,
) -> dict:
    rec = {
        "record_id": f"fixed-probe-x-on-full-{base_record_id}-f{fraction}",
        "base_record_id": base_record_id,
        "fraction": fraction,
        "trace_source_condition": trace_source_condition,
        "replay_policy_condition": replay_policy_condition,
        "matches_f1_anchor_answer": matches_anchor,
        "f1_anchor_is_correct": f1_anchor_is_correct,
        "replay_compaction_count_at_cut": compaction_count,
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
    rec.update(extra)
    return rec


def _write(path: Path, rows: list[dict]) -> None:
    w = JsonlWriter(path, validator=None)
    for r in rows:
        w.append(r)


def _make_fixed_trace_run(
    tmp_path: Path,
    *,
    full_matches: dict[float, bool | None],
    rkv_matches: dict[float, bool | None],
    rkv_compaction: int = 3,
    full_f1_correct: bool = True,
    rkv_f1_correct: bool = True,
    full_f1_boxed: bool = True,
    rkv_f1_boxed: bool = True,
    rkv_actual_compression: bool = True,
    rkv_eviction_during_answer: bool = False,
    src_idx: int = 0,
    seed: int = 42,
    cap_hit: bool = False,
    base_is_correct: bool = True,
    parse_status: str = "generation_prompt_preopened_ok",
    rkv_base_record_id: str | None = None,
):
    """Build one problem's worth of fixed-trace fixtures: one canonical base
    record plus one probe record per fraction present in each of
    `full_matches`/`rkv_matches` (a fraction simply absent from the dict
    means "no probe record was ever written for it," e.g. to simulate a
    missing f=1 anchor or a missing scored fraction)."""
    base = _base_record(src_idx, seed, cap_hit=cap_hit, parse_status=parse_status, is_correct=base_is_correct)

    def _status_for(f: float, boxed: bool) -> str:
        if f != 1.0:
            return "boxed"
        return "boxed" if boxed else "final_number_fallback"

    full_recs = [
        _fixed_probe_record(
            base["record_id"], f, matches_anchor=full_matches[f], f1_anchor_is_correct=full_f1_correct,
            probe_extraction_status=_status_for(f, full_f1_boxed),
        )
        for f in PROBE_FRACTIONS_ALL
        if f in full_matches
    ]
    rkv_base_id = rkv_base_record_id if rkv_base_record_id is not None else base["record_id"]
    rkv_recs = [
        _fixed_probe_record(
            rkv_base_id, f, matches_anchor=rkv_matches[f], f1_anchor_is_correct=rkv_f1_correct,
            replay_policy_condition="rkv_b512",
            compaction_count=rkv_compaction,
            probe_extraction_status=_status_for(f, rkv_f1_boxed),
            actual_compression_at_cut=rkv_actual_compression,
            probe_actual_eviction_during_answer=rkv_eviction_during_answer,
        )
        for f in PROBE_FRACTIONS_ALL
        if f in rkv_matches
    ]
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", full_recs)
    _write(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", rkv_recs)

    base_records = [base]
    full_probes = load_fixed_trace_records(tmp_path / "full_on_full_fixed_trace_probes.jsonl", "full")
    rkv_probes = load_fixed_trace_records(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", "rkv_b512")
    return base, base_records, full_probes, rkv_probes


_ALL_MATCH = {f: True for f in PROBE_FRACTIONS_ALL}
_SETTINGS = FixedTraceSettings(min_eligible_examples=1, min_actual_compression_rate=0.7, max_mean_f1_retention_ratio=0.7)


# --- 1. FullKV and R-KV fixed probes share the same base_record_id ---

def test_pairing_requires_shared_base_record_id(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH,
        rkv_base_record_id="base-full-ds-999-seed42",  # deliberately mismatched
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs == []  # nothing shared -- no pair is formed across different base_record_ids


def test_pairing_forms_when_base_record_id_matches(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert len(pairs) == 1
    assert pairs[0].base_record_id == base["record_id"]


# --- 2. Pairing uses source_row_index and global_seed ---

def test_pair_result_carries_source_row_index_and_seed_from_base_record(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, src_idx=17, seed=2026,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert len(pairs) == 1
    assert pairs[0].source_row_index == 17
    assert pairs[0].seed == 2026


# --- 3. Matching uses f=1 (the same policy's own anchor), not the natural base answer ---

def test_pss_reads_only_matches_f1_anchor_field_not_source_base_answer(tmp_path):
    # Every field that would encode "the trace source's own sampled answer"
    # is deliberately set to CONTRADICT matches_f1_anchor_answer here; PSS
    # must follow matches_f1_anchor_answer alone.
    base = _base_record(0, 42)
    full_recs = [
        _fixed_probe_record(
            base["record_id"], f, matches_anchor=True,
            source_base_answer="999", normalized_probe_answer="42", normalized_f1_anchor_answer="1",
        )
        for f in PROBE_FRACTIONS_ALL
    ]
    _write(tmp_path / "probes.jsonl", full_recs)
    recs = load_fixed_trace_records(tmp_path / "probes.jsonl", "full")
    group = recs.probes_by_base[base["record_id"]]
    matches_by_fraction = {f: group[f]["matches_f1_anchor_answer"] for f in PROBE_FRACTIONS_SCORED}
    pss = compute_pss(matches_by_fraction)
    assert pss == pytest.approx(0.0)  # all matched -> PSS 0, despite the contradictory raw-answer fields


# --- 4. Positive Delta_PSS means R-KV is less sensitive ---

def test_delta_pss_sign_convention_positive_means_rkv_less_sensitive():
    delta = compute_delta_pss(pss_full=0.8, pss_rkv=0.3)
    assert delta == pytest.approx(0.5)
    assert delta > 0


def test_delta_pss_sign_convention_negative_means_rkv_more_sensitive():
    delta = compute_delta_pss(pss_full=0.2, pss_rkv=0.6)
    assert delta == pytest.approx(-0.4)
    assert delta < 0


def test_delta_pss_undefined_propagates():
    assert compute_delta_pss(None, 0.3) is None
    assert compute_delta_pss(0.3, None) is None


# --- 5. Missing f=1 makes a pair ineligible (invalid/missing anchor -> PSS None) ---

def test_missing_f1_anchor_makes_pair_ineligible(tmp_path):
    matches_no_f1 = {f: True for f in PROBE_FRACTIONS_ALL if f != 1.0}
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=matches_no_f1, rkv_matches=_ALL_MATCH,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert len(pairs) == 1
    assert pairs[0].eligibility.eligible is False
    assert "full_f1_anchor_not_boxed" in pairs[0].eligibility.failure_reasons
    assert pairs[0].pss_full is None
    assert pairs[0].delta_pss is None


# --- 6. Missing scored fractions make a pair ineligible ---

def test_missing_scored_fraction_makes_pair_ineligible(tmp_path):
    matches_no_half = {f: True for f in PROBE_FRACTIONS_ALL if f != 0.5}
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=matches_no_half,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert len(pairs) == 1
    assert pairs[0].eligibility.eligible is False
    assert "rkv_scored_probe_extraction_failed" in pairs[0].eligibility.failure_reasons
    assert pairs[0].pss_rkv is None
    assert pairs[0].delta_pss is None


def test_missing_scored_extraction_produces_pss_none(tmp_path):
    # A scored fraction present but with an undefined (None) match -- i.e.
    # this fraction's own extraction (or the anchor's) failed.
    rkv_matches_with_failure = {f: True for f in PROBE_FRACTIONS_ALL}
    rkv_matches_with_failure[0.5] = None
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=rkv_matches_with_failure,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].pss_rkv is None
    assert pairs[0].delta_pss is None


# --- 7. R-KV with no ACTUAL compression (not just an event count) is ineligible ---

def test_rkv_zero_replay_compactions_makes_pair_ineligible(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, rkv_compaction=0,
        rkv_actual_compression=False,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert len(pairs) == 1
    assert pairs[0].eligibility.eligible is False
    assert "rkv_no_actual_compression_at_f1" in pairs[0].eligibility.failure_reasons
    assert pairs[0].delta_pss is None


def test_recorded_compaction_event_with_zero_actual_eviction_is_ineligible(tmp_path):
    # The exact-budget-boundary case: a compaction EVENT was recorded
    # (compaction_count > 0) but the physical cache never actually shrank
    # (actual_compression_at_cut False) -- must NOT be treated as compression.
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH,
        rkv_compaction=1, rkv_actual_compression=False,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is False
    assert "rkv_no_actual_compression_at_f1" in pairs[0].eligibility.failure_reasons


def test_actual_compression_flag_is_read_directly_from_the_f1_record(tmp_path):
    # Physical cache smaller than the FullKV-equivalent slot count at the cut
    # is what actual_compression_at_cut encodes (kvcot.cli.cmd_replay_fixed_trace);
    # the pairing layer must surface it unchanged onto the pair result.
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, rkv_actual_compression=True,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].rkv_actual_compression_at_f1 is True


# --- 8. Fallback (non-boxed) f=1 anchor is ineligible, PSS None for that side ---

def test_fallback_f1_anchor_is_ineligible_and_pss_none(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, rkv_f1_boxed=False,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is False
    assert "rkv_f1_anchor_not_boxed" in pairs[0].eligibility.failure_reasons
    assert pairs[0].pss_rkv is None
    assert pairs[0].delta_pss is None


# --- 9. Incorrect boxed f=1 anchor is ineligible ---

def test_incorrect_boxed_f1_anchor_is_ineligible(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, rkv_f1_correct=False,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is False
    assert "rkv_f1_anchor_incorrect" in pairs[0].eligibility.failure_reasons
    # The anchor is still a well-formed boxed answer, so PSS_rkv can still be
    # numerically defined -- only eligibility (hence delta_pss) is affected.
    assert pairs[0].pss_rkv is not None
    assert pairs[0].delta_pss is None


def test_cap_hit_canonical_trace_makes_pair_ineligible(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, cap_hit=True,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is False
    assert "canonical_trace_cap_hit" in pairs[0].eligibility.failure_reasons


def test_canonical_trace_base_incorrect_makes_pair_ineligible(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, base_is_correct=False,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is False
    assert "canonical_trace_base_incorrect" in pairs[0].eligibility.failure_reasons


# --- 10. Answer-time R-KV eviction makes the pair ineligible ---

def test_answer_time_eviction_makes_pair_ineligible(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, rkv_eviction_during_answer=True,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is False
    assert "rkv_evicted_during_answer_probe" in pairs[0].eligibility.failure_reasons
    assert pairs[0].delta_pss is None


def test_f1_only_answer_time_eviction_makes_pair_ineligible(tmp_path):
    # Regression for a real gap found in external review (2026-07-16): the
    # eviction check previously only scanned PROBE_FRACTIONS_SCORED (7
    # fractions), never the f=1 anchor itself -- so a synthetic case where
    # ONLY the R-KV f=1 answer evicted cache tokens was scored eligible with
    # zero failure reasons. Every scored fraction is compared against the
    # f=1 anchor's own answer, so an eviction while the anchor was writing
    # ITS OWN answer must be exactly as disqualifying as one on a scored
    # fraction -- the anchor itself becomes untrustworthy.
    base = _base_record(0, 42, is_correct=True)
    full_recs = [
        _fixed_probe_record(base["record_id"], f, matches_anchor=True)
        for f in PROBE_FRACTIONS_ALL
    ]
    rkv_recs = [
        _fixed_probe_record(
            base["record_id"], f, matches_anchor=True,
            replay_policy_condition="rkv_b512",
            # Every scored fraction is clean -- eviction happens ONLY on f=1.
            probe_actual_eviction_during_answer=(f == 1.0),
        )
        for f in PROBE_FRACTIONS_ALL
    ]
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", full_recs)
    _write(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", rkv_recs)

    base_records = [base]
    full_probes = load_fixed_trace_records(tmp_path / "full_on_full_fixed_trace_probes.jsonl", "full")
    rkv_probes = load_fixed_trace_records(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", "rkv_b512")

    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert len(pairs) == 1
    eligible_with_f1_only_eviction = pairs[0].eligibility.eligible
    failure_reasons = pairs[0].eligibility.failure_reasons
    assert eligible_with_f1_only_eviction is False, (
        "an f=1-only answer-time eviction must make the pair ineligible -- the anchor itself is "
        "untrustworthy even though every SCORED fraction's own probe was clean"
    )
    assert "rkv_evicted_during_answer_probe" in failure_reasons
    assert pairs[0].delta_pss is None


def test_fully_eligible_pair_gets_a_defined_delta_pss(tmp_path):
    full_matches = {f: True for f in PROBE_FRACTIONS_ALL}
    rkv_matches = {f: True for f in PROBE_FRACTIONS_ALL}
    for f in (0.125, 0.375):
        full_matches[f] = False  # FullKV mismatches twice; R-KV never does
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=full_matches, rkv_matches=rkv_matches,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is True
    assert pairs[0].pss_full == pytest.approx(2 / 7)
    assert pairs[0].pss_rkv == pytest.approx(0.0)
    assert pairs[0].delta_pss == pytest.approx(2 / 7)


# --- 11. FullKV and R-KV computed from different canonical traces raise an error ---

def test_mismatched_trace_source_condition_raises(tmp_path):
    # `load_fixed_trace_records` alone only catches an internally-inconsistent
    # single file (see test_single_file_with_two_trace_sources_raises below);
    # the FullKV-vs-R-KV cross-file check is `_assert_shared_trace_source`,
    # called once up front by kvcot.analysis.fixed_trace.run_fixed_trace_analysis.
    base = _base_record(0, 42)
    full_recs = [
        _fixed_probe_record(base["record_id"], f, trace_source_condition="full", matches_anchor=True)
        for f in PROBE_FRACTIONS_ALL
    ]
    # Simulates a user error: this file was actually replayed from a
    # patched_noop-generated trace, not the "full" trace this analysis expects.
    rkv_recs_from_wrong_trace = [
        _fixed_probe_record(
            base["record_id"], f, trace_source_condition="patched_noop",
            replay_policy_condition="rkv_b512", matches_anchor=True,
        )
        for f in PROBE_FRACTIONS_ALL
    ]
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", full_recs)
    _write(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", rkv_recs_from_wrong_trace)
    full_probes = load_fixed_trace_records(tmp_path / "full_on_full_fixed_trace_probes.jsonl", "full")
    rkv_probes = load_fixed_trace_records(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", "rkv_b512")
    with pytest.raises(ValueError):
        _assert_shared_trace_source(full_probes, rkv_probes, "full")


def test_single_file_with_two_trace_sources_raises(tmp_path):
    base_a = _base_record(0, 42)
    base_b = _base_record(1, 42)
    rows = [
        _fixed_probe_record(base_a["record_id"], 1.0, trace_source_condition="full", matches_anchor=True),
        _fixed_probe_record(base_b["record_id"], 1.0, trace_source_condition="patched_noop", matches_anchor=True),
    ]
    _write(tmp_path / "mixed.jsonl", rows)
    with pytest.raises(ValueError):
        load_fixed_trace_records(tmp_path / "mixed.jsonl", "full")


# --- 12. Agreement-rate delta is RKV minus FullKV ---

def test_match_rate_delta_is_rkv_minus_full():
    full_curve = {0.5: 0.4, 0.75: 0.6}
    rkv_curve = {0.5: 0.9, 0.75: 0.6}
    delta = match_rate_delta_rkv_minus_full(full_curve, rkv_curve)
    assert delta[0.5] == pytest.approx(0.5)  # rkv higher match rate -> positive -> less sensitive
    assert delta[0.75] == pytest.approx(0.0)


# --- 13. Empty curve returns None, not 0.0 ---

def test_curve_with_no_valid_measurements_returns_none_not_zero(tmp_path):
    _write(tmp_path / "empty.jsonl", [])
    empty_records = load_fixed_trace_records(tmp_path / "empty.jsonl", "full")
    curve = fixed_trace_curve_by_fraction(empty_records)
    assert all(v is None for v in curve.values())


def test_match_rate_delta_none_when_either_side_none():
    full_curve = {0.5: None, 0.75: 0.6}
    rkv_curve = {0.5: 0.9, 0.75: None}
    delta = match_rate_delta_rkv_minus_full(full_curve, rkv_curve)
    assert delta[0.5] is None
    assert delta[0.75] is None


# --- 14. Analysis produces no p-value at n=10 (or any n) ---

def test_build_fixed_trace_decision_has_no_pvalue_or_ci(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    decision = build_fixed_trace_decision(len(pairs), pairs, full_curve={}, rkv_curve={}, settings=_SETTINGS)
    banned_substrings = ("p_value", "pvalue", "ci_low", "ci_high", "wilcoxon", "bootstrap")
    flat_keys = _flatten_keys(decision)
    for key in flat_keys:
        for banned in banned_substrings:
            assert banned not in key.lower(), f"decision JSON unexpectedly contains a {banned!r}-like key: {key}"
    assert decision["n_shared"] == 1
    assert decision["n_eligible"] == 1


def _flatten_keys(obj, prefix="") -> list[str]:
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            keys.append(full_key)
            keys.extend(_flatten_keys(v, full_key))
    elif isinstance(obj, list):
        for item in obj:
            keys.extend(_flatten_keys(item, prefix))
    return keys


def test_decision_json_round_trips_through_read_jsonl_style_io(tmp_path):
    # Sanity check the output is plain-JSON-serializable (no stray float
    # fraction keys, no non-serializable objects) since kvcot.utils.io.write_json
    # is what actually persists it in kvcot.analysis.fixed_trace.run_fixed_trace_analysis.
    import json

    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    decision = build_fixed_trace_decision(
        len(pairs), pairs, full_curve={0.5: 1.0}, rkv_curve={0.5: 1.0}, settings=_SETTINGS
    )
    json.dumps(decision)  # must not raise


# --- 15. Screen-level validity (§ Step 11) ---

def test_screen_invalid_when_too_few_eligible_examples():
    settings = FixedTraceSettings(min_eligible_examples=5, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0)
    valid, reasons = build_screen_validity(
        n_eligible=2, actual_compression_rate=1.0, mean_f1_rkv_retention_ratio=0.3, settings=settings
    )
    assert valid is False
    assert any("n_eligible" in r for r in reasons)


def test_screen_invalid_when_retention_above_threshold():
    settings = FixedTraceSettings(min_eligible_examples=1, min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=0.7)
    valid, reasons = build_screen_validity(
        n_eligible=5, actual_compression_rate=1.0, mean_f1_rkv_retention_ratio=0.98, settings=settings
    )
    assert valid is False
    assert any("retention_ratio" in r for r in reasons)


def test_screen_invalid_when_compression_rate_too_low():
    settings = FixedTraceSettings(min_eligible_examples=1, min_actual_compression_rate=0.7, max_mean_f1_retention_ratio=1.0)
    valid, reasons = build_screen_validity(
        n_eligible=5, actual_compression_rate=0.1, mean_f1_rkv_retention_ratio=0.3, settings=settings
    )
    assert valid is False
    assert any("compression_rate" in r for r in reasons)


def test_screen_valid_when_all_thresholds_cleared():
    settings = FixedTraceSettings(min_eligible_examples=1, min_actual_compression_rate=0.5, max_mean_f1_retention_ratio=0.7)
    valid, reasons = build_screen_validity(
        n_eligible=5, actual_compression_rate=0.8, mean_f1_rkv_retention_ratio=0.4, settings=settings
    )
    assert valid is True
    assert reasons == []


def test_decision_hypothesis_status_not_tested_when_screen_invalid(tmp_path):
    strict_settings = FixedTraceSettings(min_eligible_examples=5, min_actual_compression_rate=0.7, max_mean_f1_retention_ratio=0.7)
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    decision = build_fixed_trace_decision(len(pairs), pairs, full_curve={}, rkv_curve={}, settings=strict_settings)
    # Only 1 eligible example, well below min_eligible_examples=5.
    assert decision["screen_valid"] is False
    assert decision["hypothesis_status"] == "not_tested"
    assert decision["screen_invalid_reasons"] != []


def test_decision_never_reports_positive_or_negative_characterization(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    decision = build_fixed_trace_decision(len(pairs), pairs, full_curve={}, rkv_curve={}, settings=_SETTINGS)
    assert decision["hypothesis_status"] in ("not_tested", "screened")


# --- Schema/identity validation at analysis load time (§ external review 2026-07-16) ---

def _valid_base_run_record(**overrides) -> dict:
    from kvcot.schemas import DatasetProvenance, MethodConfig, ProvenanceState, ThinkSpanInfo, VersionInfo

    defaults = dict(
        record_id="base-full-gsm8k_calibration_50-0-seed42",
        config_path="configs/early_gap_v2_b128.yaml",
        config_sha256="a" * 64,
        provenance=ProvenanceState(upstream_rkv_commit="45eaa7d69d20b7388321f077020a610d9afb65bd", git_commit="deadbeef", git_dirty=False),
        versions=VersionInfo(python="3.10.0"),
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        model_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tokenizer_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        dataset=DatasetProvenance(dataset_name="gsm8k", source_row_index=0, question_hash="b" * 64, normalized_gold="42"),
        condition="full",
        method_config=MethodConfig(method="fullkv"),
        global_seed=42,
        derived_seed=123,
        prompt_text="hello",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5, 6],
        decoded_output="Final answer: \boxed{42}",
        think_span=ThinkSpanInfo(think_start_index=0, think_end_index=3, think_parse_status="generation_prompt_preopened_ok", generation_prompt_preopened_think=True),
        extracted_answer="42",
        extraction_method="boxed",
        is_correct=True,
        cap_hit=False,
        wall_time_seconds=1.0,
        generated_token_count=3,
        compaction_count=0,
        compaction_event_steps=[],
        cache_length_final_per_layer=[3, 3],
    )
    defaults.update(overrides)
    return BaseRunRecord(**defaults).model_dump(mode="json")


def _valid_fixed_trace_probe_record(**overrides) -> dict:
    from kvcot.schemas import ProvenanceState, RetentionSummary, VersionInfo

    defaults = dict(
        record_id="fixed-probe-rkv_b128-on-full-base-full-gsm8k_calibration_50-0-seed42-f1.0",
        parent_record_id="base-full-gsm8k_calibration_50-0-seed42",
        config_path="configs/early_gap_v2_b128.yaml",
        config_sha256="a" * 64,
        provenance=ProvenanceState(upstream_rkv_commit="45eaa7d69d20b7388321f077020a610d9afb65bd", git_commit="deadbeef", git_dirty=False),
        versions=VersionInfo(python="3.10.0"),
        model_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        tokenizer_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        base_record_id="base-full-gsm8k_calibration_50-0-seed42",
        trace_source_condition="full",
        replay_policy_condition="rkv_b128",
        source_row_index=0,
        global_seed=42,
        normalized_gold="42",
        source_base_answer="42",
        source_base_is_correct=True,
        fraction=1.0,
        think_span_length=100,
        cut_index=100,
        close_marker_token_ids=[151649],
        control_suffix_token_ids=[123, 456],
        probe_decoding_max_new_tokens=64,
        probe_output_token_ids=[9, 9],
        probe_output_text="42}",
        probe_extraction_text="Final answer: \boxed{42}",
        normalized_probe_answer="42",
        probe_extraction_status="boxed",
        probe_stop_reason="boxed_answer_complete",
        probe_cap_hit=False,
        replay_retention_at_cut=RetentionSummary(
            fullkv_equivalent_slots=200, physical_cache_slots_per_layer=[120, 120],
            instantaneous_retention_ratio=0.6, post_compaction_budget_tokens=128,
            tokens_since_last_compaction=5,
        ),
        actual_compression_at_cut=True,
        probe_cache_length_final_per_layer=[125, 125],
        probe_actual_eviction_during_answer=False,
        normalized_f1_anchor_answer="42",
        matches_f1_anchor_answer=True,
        f1_anchor_matches_source_base_answer=True,
        f1_anchor_is_correct=True,
        replay_compaction_count_at_cut=2,
        replay_compaction_event_steps_at_cut=[64, 128],
        snapshot_cache_hash="c" * 64,
        snapshot_provenance_hash="d" * 64,
        snapshot_state_hash="e" * 64,
    )
    defaults.update(overrides)
    return FixedTraceProbeRecord(**defaults).model_dump(mode="json")


def test_validate_base_records_accepts_schema_valid_input(tmp_path):
    _validate_base_records([_valid_base_run_record()], tmp_path / "full.jsonl", "full")  # must not raise


def test_validate_base_records_rejects_stale_schema_version_dict(tmp_path):
    # A raw dict shaped like a protocol-v1 (schema 1.1.0) record -- missing
    # the protocol-v2-only fields entirely, and carrying the old version
    # string. model_validate must reject it, not silently accept it as a
    # plain dict the way analysis used to.
    stale_row = _valid_base_run_record()
    stale_row["schema_version"] = "1.1.0"
    with pytest.raises(ValueError):
        _validate_base_records([stale_row], tmp_path / "full.jsonl", "full")


def test_validate_base_records_rejects_mismatched_identity(tmp_path):
    row_a = _valid_base_run_record(record_id="base-a")
    row_b = _valid_base_run_record(record_id="base-b", config_sha256="f" * 64)
    with pytest.raises(ValueError):
        _validate_base_records([row_a, row_b], tmp_path / "full.jsonl", "full")


def test_validate_base_records_rejects_condition_not_matching_trace_condition(tmp_path):
    # § external review 2026-07-16: a base file recorded under condition=
    # "rkv_b128" being passed in as the canonical trace source (which this
    # analysis always expects to be trace_condition, in practice "full")
    # would mean the wrong file was used as the canonical trace.
    row = _valid_base_run_record(condition="rkv_b128")
    with pytest.raises(ValueError):
        _validate_base_records([row], tmp_path / "full.jsonl", "full")


def test_validate_fixed_trace_probe_records_accepts_schema_valid_input(tmp_path):
    from kvcot.analysis.fixed_trace import FixedTraceRecords

    rec = _valid_fixed_trace_probe_record()
    records = FixedTraceRecords(
        replay_condition="rkv_b128", trace_source_condition="full",
        probes_by_base={rec["base_record_id"]: {1.0: rec}},
    )
    _validate_fixed_trace_probe_records(records, tmp_path / "probes.jsonl")  # must not raise


def test_validate_fixed_trace_probe_records_rejects_stale_schema_version(tmp_path):
    from kvcot.analysis.fixed_trace import FixedTraceRecords

    rec = _valid_fixed_trace_probe_record()
    rec["schema_version"] = "1.1.0"
    records = FixedTraceRecords(
        replay_condition="rkv_b128", trace_source_condition="full",
        probes_by_base={rec["base_record_id"]: {1.0: rec}},
    )
    with pytest.raises(ValueError):
        _validate_fixed_trace_probe_records(records, tmp_path / "probes.jsonl")


def test_run_fixed_trace_analysis_rejects_protocol_v1_shaped_base_file(tmp_path):
    # End-to-end: a base file shaped like protocol v1 (missing the fields
    # BaseRunRecord now requires, or carrying schema_version "1.1.0") must
    # make the whole analysis command fail loudly rather than silently
    # produce a decision JSON from unvalidated dicts.
    stale_base = _valid_base_run_record()
    stale_base["schema_version"] = "1.1.0"
    _write(tmp_path / "full.jsonl", [stale_base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", [])
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", [])

    with pytest.raises(ValueError):
        run_fixed_trace_analysis(
            output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
            stage_name="early_gap_v2_b128", settings=_SETTINGS,
        )


# --- Cross-file and current-config identity checks (§ external review 2026-07-16) ---

def test_assert_consistent_identity_accepts_matching_files():
    from kvcot.analysis.fixed_trace import _assert_consistent_identity

    ident = ("a" * 64, "up", "model-1", "tok-1")
    _assert_consistent_identity([("base", ident), ("full_probes", ident), ("rkv_probes", ident)])  # no raise


def test_assert_consistent_identity_ignores_empty_files():
    from kvcot.analysis.fixed_trace import _assert_consistent_identity

    ident = ("a" * 64, "up", "model-1", "tok-1")
    # An empty rkv_probes file (identity None) must not itself trigger a
    # mismatch -- "nothing written yet" is reported elsewhere (n_shared=0).
    _assert_consistent_identity([("base", ident), ("full_probes", ident), ("rkv_probes", None)])


def test_assert_consistent_identity_rejects_cross_file_mismatch():
    from kvcot.analysis.fixed_trace import _assert_consistent_identity

    base_ident = ("a" * 64, "up", "model-1", "tok-1")
    rkv_ident = ("a" * 64, "up", "model-1", "DIFFERENT-tokenizer")
    with pytest.raises(ValueError):
        _assert_consistent_identity([("base", base_ident), ("full_probes", base_ident), ("rkv_probes", rkv_ident)])


def test_assert_consistent_identity_rejects_mismatch_against_expected():
    from kvcot.analysis.fixed_trace import _assert_consistent_identity

    ident = ("a" * 64, "up", "model-1", "tok-1")
    expected = ("a" * 64, "up", "model-2", "tok-1")  # current lock pins a different model revision
    with pytest.raises(ValueError):
        _assert_consistent_identity([("base", ident), ("full_probes", ident), ("rkv_probes", ident)], expected=expected)


def test_run_fixed_trace_analysis_rejects_base_vs_probes_identity_mismatch(tmp_path):
    # base.jsonl and the R-KV fixed-trace probes were produced under
    # different upstream_rkv_commit pins -- must never be silently paired.
    base = _valid_base_run_record()
    full_rec = _valid_fixed_trace_probe_record(replay_policy_condition="full")
    rkv_rec = _valid_fixed_trace_probe_record(
        provenance={"upstream_rkv_commit": "0000000000000000000000000000000000000000", "git_commit": "deadbeef", "git_dirty": False},
    )
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", [full_rec])
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", [rkv_rec])

    with pytest.raises(ValueError):
        run_fixed_trace_analysis(
            output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
            stage_name="early_gap_v2_b128", settings=_SETTINGS,
        )


def test_run_fixed_trace_analysis_rejects_mismatch_against_current_config(tmp_path):
    # All three files agree with each other, but the invocation's OWN
    # config/lock (expected_identity) pins a different model revision --
    # this must be rejected even though the files are internally consistent.
    base = _valid_base_run_record()
    full_rec = _valid_fixed_trace_probe_record(replay_policy_condition="full")
    rkv_rec = _valid_fixed_trace_probe_record(record_id="fixed-probe-rkv-f1", replay_policy_condition="rkv_b128")
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", [full_rec])
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", [rkv_rec])

    mismatched_expected_identity = (
        base["config_sha256"], base["provenance"]["upstream_rkv_commit"],
        "some-other-model-revision", base["tokenizer_revision"],
    )
    with pytest.raises(ValueError):
        run_fixed_trace_analysis(
            output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
            stage_name="early_gap_v2_b128", settings=_SETTINGS, expected_identity=mismatched_expected_identity,
        )


def test_run_fixed_trace_analysis_accepts_matching_current_config(tmp_path, monkeypatch):
    # run_fixed_trace_analysis writes results/decisions/{stage_name}_fixed_trace.json
    # relative to the CURRENT WORKING DIRECTORY (not output_dir) -- chdir
    # into tmp_path for this one successful-path test so it can't leave a
    # stray file behind in the real repository's results/decisions/.
    monkeypatch.chdir(tmp_path)
    base = _valid_base_run_record()
    full_rec = _valid_fixed_trace_probe_record(replay_policy_condition="full")
    rkv_rec = _valid_fixed_trace_probe_record(record_id="fixed-probe-rkv-f1", replay_policy_condition="rkv_b128")
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", [full_rec])
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", [rkv_rec])

    matching_expected_identity = (
        base["config_sha256"], base["provenance"]["upstream_rkv_commit"],
        base["model_revision"], base["tokenizer_revision"],
    )
    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="early_gap_v2_b128", settings=_SETTINGS, expected_identity=matching_expected_identity,
    )
    assert rc == 0


# --- selection-file completeness guard, end-to-end (2026-07-18 review) ---

def _all_fraction_records(base_record_id: str, replay_policy_condition: str, **overrides) -> list[dict]:
    return [
        _valid_fixed_trace_probe_record(
            record_id=f"fixed-probe-{replay_policy_condition}-on-full-{base_record_id}-f{f}",
            base_record_id=base_record_id,
            parent_record_id=base_record_id,
            fraction=f,
            replay_policy_condition=replay_policy_condition,
            **overrides,
        )
        for f in PROBE_FRACTIONS_ALL
    ]


def test_run_fixed_trace_analysis_aborts_on_incomplete_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _valid_base_run_record()
    full_recs = _all_fraction_records(base["record_id"], "full")
    rkv_recs = _all_fraction_records(base["record_id"], "rkv_b128")
    del rkv_recs[-1]  # drop f=1.0 -- simulates a partially-completed replay
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", full_recs)
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", rkv_recs)

    with pytest.raises(ValueError, match="selection completeness check failed"):
        run_fixed_trace_analysis(
            output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
            stage_name="test_v3_selection", settings=_SETTINGS,
            selected_base_record_ids={base["record_id"]},
        )


def test_run_fixed_trace_analysis_passes_with_complete_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = _valid_base_run_record()
    full_recs = _all_fraction_records(base["record_id"], "full")
    rkv_recs = _all_fraction_records(base["record_id"], "rkv_b128")
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", full_recs)
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", rkv_recs)

    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_selection", settings=_SETTINGS,
        selected_base_record_ids={base["record_id"]},
    )
    assert rc == 0
    decision = read_json(Path("results/decisions/test_v3_selection_fixed_trace.json"))
    assert decision["n_shared"] == 1


def test_run_fixed_trace_analysis_rejects_superset_probe_files_under_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Two complete problems on disk; selection only names one. 2026-07-19
    # review: this must now ABORT (probe files contain a base_record_id
    # outside the selection) rather than silently scoping down to n_shared=1
    # -- a superset mismatch (e.g. the wrong, larger probe file passed in)
    # must never be quietly discarded.
    base_a = _valid_base_run_record(record_id="base-a", dataset={"dataset_name": "gsm8k", "source_row_index": 0, "question_hash": "b" * 64, "normalized_gold": "42"})
    base_b = _valid_base_run_record(record_id="base-b", dataset={"dataset_name": "gsm8k", "source_row_index": 1, "question_hash": "c" * 64, "normalized_gold": "42"})
    full_recs = _all_fraction_records("base-a", "full") + _all_fraction_records("base-b", "full")
    rkv_recs = _all_fraction_records("base-a", "rkv_b128") + _all_fraction_records("base-b", "rkv_b128")
    _write(tmp_path / "full.jsonl", [base_a, base_b])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", full_recs)
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", rkv_recs)

    with pytest.raises(ValueError, match="NOT in the selection"):
        run_fixed_trace_analysis(
            output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
            stage_name="test_v3_selection_scoped", settings=_SETTINGS,
            selected_base_record_ids={"base-a"},
        )


def test_run_fixed_trace_analysis_accepts_selection_matching_probe_files_exactly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Same as above, but the probe files contain ONLY the selected example --
    # this is the clean, correctly-scoped case and must succeed with
    # n_shared == n_selected == 1.
    base_a = _valid_base_run_record(record_id="base-a", dataset={"dataset_name": "gsm8k", "source_row_index": 0, "question_hash": "b" * 64, "normalized_gold": "42"})
    full_recs = _all_fraction_records("base-a", "full")
    rkv_recs = _all_fraction_records("base-a", "rkv_b128")
    _write(tmp_path / "full.jsonl", [base_a])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", full_recs)
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", rkv_recs)

    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_selection_scoped_clean", settings=_SETTINGS,
        selected_base_record_ids={"base-a"},
    )
    assert rc == 0
    decision = read_json(Path("results/decisions/test_v3_selection_scoped_clean_fixed_trace.json"))
    assert decision["n_shared"] == 1


def test_run_fixed_trace_analysis_without_selection_file_is_unaffected(tmp_path, monkeypatch):
    # selected_base_record_ids=None (every pre-2026-07-18 caller) must
    # preserve the exact prior behavior -- no completeness requirement.
    monkeypatch.chdir(tmp_path)
    base = _valid_base_run_record()
    full_rec = _valid_fixed_trace_probe_record(replay_policy_condition="full")
    rkv_rec = _valid_fixed_trace_probe_record(record_id="fixed-probe-rkv-f1", replay_policy_condition="rkv_b128")
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", [full_rec])
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", [rkv_rec])

    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_no_selection", settings=_SETTINGS,
    )
    assert rc == 0


# --- Duplicate (base_record_id, fraction) rows (§ external review 2026-07-16) ---

def test_load_fixed_trace_records_rejects_duplicate_base_and_fraction(tmp_path):
    rec_a = _valid_fixed_trace_probe_record(record_id="fixed-probe-a")
    rec_b = _valid_fixed_trace_probe_record(record_id="fixed-probe-b")  # same base_record_id, same fraction=1.0
    _write(tmp_path / "dup.jsonl", [rec_a, rec_b])
    with pytest.raises(ValueError):
        load_fixed_trace_records(tmp_path / "dup.jsonl", "rkv_b128")


# --- replay_policy_condition role validation (§ external review 2026-07-16) ---
#
# `load_fixed_trace_records` previously only checked `trace_source_condition`
# consistency/agreement -- it never checked a row's own `replay_policy_
# condition` against the `replay_condition` the caller passed in based on
# which file (by filename convention) is being read. A swapped or renamed
# pair of fixed-trace probe files -- e.g. full_on_full_fixed_trace_probes.
# jsonl actually containing rkv_b128-policy records, and vice versa -- was
# silently accepted and could flip which curve gets called FullKV vs R-KV.

def test_load_fixed_trace_records_rejects_replay_policy_condition_mismatch(tmp_path):
    # A record declaring replay_policy_condition="rkv_b128" saved into a file
    # being read as the "full" replay policy's probes -- exactly what an
    # accidental file swap/rename would produce.
    rec = _valid_fixed_trace_probe_record(replay_policy_condition="rkv_b128")
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", [rec])
    with pytest.raises(ValueError, match="replay_policy_condition"):
        load_fixed_trace_records(tmp_path / "full_on_full_fixed_trace_probes.jsonl", "full")


def test_load_fixed_trace_records_rejects_mixed_replay_policy_condition_within_one_file(tmp_path):
    rec_a = _valid_fixed_trace_probe_record(record_id="fixed-probe-a", replay_policy_condition="rkv_b128")
    rec_b = _valid_fixed_trace_probe_record(
        record_id="fixed-probe-b", fraction=0.5, replay_policy_condition="full",
    )
    _write(tmp_path / "mixed.jsonl", [rec_a, rec_b])
    with pytest.raises(ValueError, match="replay_policy_condition"):
        load_fixed_trace_records(tmp_path / "mixed.jsonl", "rkv_b128")


def test_run_fixed_trace_analysis_rejects_swapped_full_and_rkv_probe_files(tmp_path):
    # End-to-end version of the adversarial case above: full_on_full declares
    # replay_policy_condition="rkv_b128" and rkv_b128_on_full declares "full"
    # -- as if the two files were swapped or renamed. Must be rejected, not
    # silently analyzed with the curves' identities flipped.
    base = _valid_base_run_record()
    full_rec = _valid_fixed_trace_probe_record(replay_policy_condition="rkv_b128")
    rkv_rec = _valid_fixed_trace_probe_record(record_id="fixed-probe-rkv-f1", replay_policy_condition="full")
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", [full_rec])
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", [rkv_rec])

    with pytest.raises(ValueError, match="replay_policy_condition"):
        run_fixed_trace_analysis(
            output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
            stage_name="early_gap_v2_b128", settings=_SETTINGS,
        )


# --- Strict natural-accuracy gate wiring in run_fixed_trace_analysis
# (2026-07-18 external review: the accuracy screen was built from the
# SELECTION-FILTERED base records, so FullKV "accuracy" on the selected
# population was 1.0 by construction -- e.g. 10/10 instead of the true 33/50) ---

_V3_SETTINGS = FixedTraceSettings(
    min_eligible_examples=1,
    min_actual_compression_rate=0.0,
    max_mean_f1_retention_ratio=1.0,
    require_meaningful_compression=True,
    meaningful_retention_ceiling=0.7,
    min_meaningfully_compressed_scored_fractions=0,
    min_meaningful_compression_rate=0.0,
    max_pilot_accuracy_drop=0.10,
)


def _natural_population(n: int, condition: str, incorrect_indices: set[int] = frozenset()) -> list[dict]:
    from kvcot.schemas import MethodConfig

    records = []
    for i in range(n):
        overrides = dict(
            record_id=f"base-{condition}-gsm8k_calibration_50-{i}-seed42",
            condition=condition,
            dataset={"dataset_name": "gsm8k", "source_row_index": i, "question_hash": "b" * 64, "normalized_gold": "42"},
            is_correct=i not in incorrect_indices,
        )
        if condition != "full":
            overrides["method_config"] = MethodConfig(method="rkv").model_dump(mode="json")
        records.append(_valid_base_run_record(**overrides))
    return records


def _v3_selected_run(tmp_path, n_total: int = 50, n_selected: int = 10, *,
                     rkv_natural_records: list[dict] | None = None,
                     full_incorrect_indices: set[int] = frozenset()) -> tuple[list[str], list[dict]]:
    """Write a complete v3 fixture: n_total natural FullKV records (the
    canonical trace source), a natural R-KV file, and complete fixed-trace
    probe files covering exactly the first n_selected (correct) examples."""
    full_natural = _natural_population(n_total, "full", incorrect_indices=full_incorrect_indices)
    if rkv_natural_records is None:
        rkv_natural_records = _natural_population(n_total, "rkv_b128")
    # Select the first n_selected FullKV-correct examples -- mirroring the
    # real selection's correctness requirement.
    selected = [r for r in full_natural if r["is_correct"]][:n_selected]
    selected_ids = [r["record_id"] for r in selected]
    full_probe_recs = []
    rkv_probe_recs = []
    for r in selected:
        full_probe_recs.extend(_all_fraction_records(r["record_id"], "full"))
        rkv_probe_recs.extend(_all_fraction_records(r["record_id"], "rkv_b128"))
    _write(tmp_path / "full.jsonl", full_natural)
    _write(tmp_path / "rkv_b128.jsonl", rkv_natural_records)
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", full_probe_recs)
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", rkv_probe_recs)
    return selected_ids, full_natural


def test_selected_analysis_uses_all_natural_records_for_accuracy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # 17 of 50 FullKV records incorrect (the true accuracy is 33/50) but the
    # 10 SELECTED examples are all correct -- the defect under test reported
    # full_accuracy=1.0 from the selected subset.
    incorrect = set(range(40, 50))  # keep the first 40 correct so selection finds 10
    incorrect |= {33, 34, 35, 36, 37, 38, 39}
    selected_ids, _ = _v3_selected_run(tmp_path, full_incorrect_indices=incorrect)
    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_accuracy_population", settings=_V3_SETTINGS,
        selected_base_record_ids=set(selected_ids), expected_accuracy_n=50,
    )
    assert rc == 0
    decision = read_json(Path("results/decisions/test_v3_accuracy_population_fixed_trace.json"))
    assert decision["n_shared"] == 10
    assert decision["strict_accuracy_gate"]["expected_n"] == 50
    assert decision["strict_accuracy_gate"]["gate_passed"] is True
    assert decision["strict_accuracy_gate"]["accuracy_screen"]["n_accuracy_pairs"] == 50
    # THE regression: accuracy must reflect the full 50-record population
    # (33/50 = 0.66), never the artificially-all-correct selected 10.
    assert decision["accuracy_screen"]["n_accuracy_pairs"] == 50
    assert decision["accuracy_screen"]["full_accuracy"] == pytest.approx(33 / 50)


def test_selected_analysis_rejects_partial_natural_rkv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Only 10 natural R-KV records exist where 50 are expected --
    # build_accuracy_screen alone would happily intersect down to 10 pairs
    # and report pilot_accuracy_plausible=True; the strict gate must fail.
    partial_rkv = _natural_population(10, "rkv_b128")
    selected_ids, _ = _v3_selected_run(tmp_path, rkv_natural_records=partial_rkv)
    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_partial_rkv", settings=_V3_SETTINGS,
        selected_base_record_ids=set(selected_ids), expected_accuracy_n=50,
    )
    assert rc == 1
    decision = read_json(Path("results/decisions/test_v3_partial_rkv_fixed_trace.json"))
    assert decision["strict_accuracy_gate"]["gate_passed"] is False
    assert any("rkv record count" in r for r in decision["strict_accuracy_gate"]["reasons"])
    assert decision["hypothesis_status"] == "not_tested"
    assert decision["screen_valid"] is False


def test_final_analysis_rejects_wrong_natural_rkv_condition(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # rkv_b128.jsonl actually contains condition="full" records (the wrong
    # file copied/renamed) -- counts, keys, and identities all match, so only
    # the expected_rkv_condition check can catch it.
    mislabeled = _natural_population(50, "full")
    selected_ids, _ = _v3_selected_run(tmp_path, rkv_natural_records=mislabeled)
    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_wrong_condition", settings=_V3_SETTINGS,
        selected_base_record_ids=set(selected_ids), expected_accuracy_n=50,
    )
    assert rc == 1
    decision = read_json(Path("results/decisions/test_v3_wrong_condition_fixed_trace.json"))
    assert decision["strict_accuracy_gate"]["gate_passed"] is False
    assert any("this gate was asked to check" in r for r in decision["strict_accuracy_gate"]["reasons"])
    assert decision["hypothesis_status"] == "not_tested"


def test_final_analysis_rejects_natural_rkv_identity_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mismatched = [
        dict(r, model_revision="some-other-model-revision")
        for r in _natural_population(50, "rkv_b128")
    ]
    selected_ids, _ = _v3_selected_run(tmp_path, rkv_natural_records=mismatched)
    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_identity_mismatch", settings=_V3_SETTINGS,
        selected_base_record_ids=set(selected_ids), expected_accuracy_n=50,
    )
    assert rc == 1
    decision = read_json(Path("results/decisions/test_v3_identity_mismatch_fixed_trace.json"))
    assert decision["strict_accuracy_gate"]["gate_passed"] is False
    assert any("do not share one" in r for r in decision["strict_accuracy_gate"]["reasons"])


def test_failed_strict_accuracy_gate_prevents_screened_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Every pair is fully eligible and the intersected accuracy screen alone
    # would be plausible -- ONLY the strict gate fails (missing natural R-KV
    # records). hypothesis_status must be "not_tested", never "screened".
    partial_rkv = _natural_population(10, "rkv_b128")
    selected_ids, _ = _v3_selected_run(tmp_path, rkv_natural_records=partial_rkv)
    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_gate_blocks_screened", settings=_V3_SETTINGS,
        selected_base_record_ids=set(selected_ids), expected_accuracy_n=50,
    )
    assert rc == 1
    decision = read_json(Path("results/decisions/test_v3_gate_blocks_screened_fixed_trace.json"))
    # Eligibility itself was fine -- the pairs are complete and valid.
    assert decision["n_eligible"] >= 1
    # The intersected screen would have looked plausible on its own.
    assert decision["strict_accuracy_gate"]["accuracy_screen"]["pilot_accuracy_plausible"] is True
    # But the gate failed, so the screen must not report "screened".
    assert decision["hypothesis_status"] == "not_tested"
    assert any("strict_accuracy_gate failed" in r for r in decision["screen_invalid_reasons"])


def test_v3_analysis_missing_natural_rkv_file_fails_gate_not_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selected_ids, _ = _v3_selected_run(tmp_path)
    (tmp_path / "rkv_b128.jsonl").unlink()
    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_missing_natural", settings=_V3_SETTINGS,
        selected_base_record_ids=set(selected_ids), expected_accuracy_n=50,
    )
    assert rc == 1
    decision = read_json(Path("results/decisions/test_v3_missing_natural_fixed_trace.json"))
    assert decision["strict_accuracy_gate"]["gate_passed"] is False
    assert any("not found" in r for r in decision["strict_accuracy_gate"]["reasons"])
    assert decision["hypothesis_status"] == "not_tested"


def test_v2_analysis_without_strict_gate_is_unchanged(tmp_path, monkeypatch):
    # Protocol-v2 stages (require_meaningful_compression=False) never had a
    # natural R-KV run: no strict gate is built, no natural file is read,
    # rc stays 0, and both gate fields stay None in the decision JSON.
    monkeypatch.chdir(tmp_path)
    base = _valid_base_run_record()
    full_rec = _valid_fixed_trace_probe_record(replay_policy_condition="full")
    rkv_rec = _valid_fixed_trace_probe_record(record_id="fixed-probe-rkv-f1", replay_policy_condition="rkv_b128")
    _write(tmp_path / "full.jsonl", [base])
    _write(tmp_path / "full_on_full_fixed_trace_probes.jsonl", [full_rec])
    _write(tmp_path / "rkv_b128_on_full_fixed_trace_probes.jsonl", [rkv_rec])

    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v2_unchanged", settings=_SETTINGS,
    )
    assert rc == 0
    decision = read_json(Path("results/decisions/test_v2_unchanged_fixed_trace.json"))
    assert decision["accuracy_screen"] is None
    assert decision["strict_accuracy_gate"] is None


def test_decision_json_records_analysis_provenance_and_input_hashes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    selected_ids, _ = _v3_selected_run(tmp_path)
    rc = run_fixed_trace_analysis(
        output_dir=tmp_path, trace_condition="full", replay_condition="rkv_b128",
        stage_name="test_v3_provenance", settings=_V3_SETTINGS,
        selected_base_record_ids=set(selected_ids), expected_accuracy_n=50,
    )
    assert rc == 0
    decision = read_json(Path("results/decisions/test_v3_provenance_fixed_trace.json"))
    prov = decision["analysis_provenance"]
    assert isinstance(prov["git_commit"], str) and prov["git_commit"]
    assert isinstance(prov["git_dirty"], bool)
    hashes = decision["input_sha256"]
    assert hashes["full_base"] is not None and len(hashes["full_base"]) == 64
    assert hashes["natural_rkv"] is not None
    assert hashes["full_fixed_trace_probes"] is not None
    assert hashes["rkv_fixed_trace_probes"] is not None
    assert hashes["selection"] is None  # no selection file path passed in this direct call
