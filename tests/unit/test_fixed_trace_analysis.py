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
from kvcot.utils.io import JsonlWriter, read_jsonl


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
        _fixed_probe_record(base["record_id"], f, trace_source_condition="patched_noop", matches_anchor=True)
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
    _validate_base_records([_valid_base_run_record()], tmp_path / "full.jsonl")  # must not raise


def test_validate_base_records_rejects_stale_schema_version_dict(tmp_path):
    # A raw dict shaped like a protocol-v1 (schema 1.1.0) record -- missing
    # the protocol-v2-only fields entirely, and carrying the old version
    # string. model_validate must reject it, not silently accept it as a
    # plain dict the way analysis used to.
    stale_row = _valid_base_run_record()
    stale_row["schema_version"] = "1.1.0"
    with pytest.raises(ValueError):
        _validate_base_records([stale_row], tmp_path / "full.jsonl")


def test_validate_base_records_rejects_mismatched_identity(tmp_path):
    row_a = _valid_base_run_record(record_id="base-a")
    row_b = _valid_base_run_record(record_id="base-b", config_sha256="f" * 64)
    with pytest.raises(ValueError):
        _validate_base_records([row_a, row_b], tmp_path / "full.jsonl")


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
