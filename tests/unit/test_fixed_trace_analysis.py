"""Unit tests for kvcot.analysis.fixed_trace — the secondary, additive
prefix-sufficiency (PSS/Delta_PSS) screen. Runs entirely from synthetic JSONL
dicts on the CPU, mirroring tests/unit/test_pipeline.py's/test_metrics.py's
style for the frozen EAS/Delta_EAS pipeline. Covers the plan's ten mandatory
fixed-trace tests (numbered in each test's docstring).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kvcot.analysis.fixed_trace import (
    _assert_shared_trace_source,
    build_fixed_trace_decision,
    build_fixed_trace_pairs,
    compute_delta_pss,
    compute_pss,
    load_fixed_trace_records,
    match_rate_delta_rkv_minus_full,
)
from kvcot.config import PROBE_FRACTIONS_ALL, PROBE_FRACTIONS_SCORED
from kvcot.utils.io import JsonlWriter, read_jsonl


def _base_record(
    src_idx: int, seed: int, *, cap_hit: bool = False, parse_status: str = "generation_prompt_preopened_ok"
) -> dict:
    return {
        "record_id": f"base-full-ds-{src_idx}-seed{seed}",
        "global_seed": seed,
        "dataset": {"source_row_index": src_idx},
        "cap_hit": cap_hit,
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
    src_idx: int = 0,
    seed: int = 42,
    cap_hit: bool = False,
    parse_status: str = "generation_prompt_preopened_ok",
    rkv_base_record_id: str | None = None,
):
    """Build one problem's worth of fixed-trace fixtures: one canonical base
    record plus one probe record per fraction present in each of
    `full_matches`/`rkv_matches` (a fraction simply absent from the dict
    means "no probe record was ever written for it," e.g. to simulate a
    missing f=1 anchor or a missing scored fraction)."""
    base = _base_record(src_idx, seed, cap_hit=cap_hit, parse_status=parse_status)
    full_recs = [
        _fixed_probe_record(base["record_id"], f, matches_anchor=full_matches[f], f1_anchor_is_correct=full_f1_correct)
        for f in PROBE_FRACTIONS_ALL
        if f in full_matches
    ]
    rkv_base_id = rkv_base_record_id if rkv_base_record_id is not None else base["record_id"]
    rkv_recs = [
        _fixed_probe_record(
            rkv_base_id, f, matches_anchor=rkv_matches[f], f1_anchor_is_correct=rkv_f1_correct, compaction_count=rkv_compaction
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
    matches = {f: True for f in PROBE_FRACTIONS_ALL}
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


# --- 5. Missing f=1 makes a pair ineligible ---

def test_missing_f1_anchor_makes_pair_ineligible(tmp_path):
    matches_no_f1 = {f: True for f in PROBE_FRACTIONS_ALL if f != 1.0}
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=matches_no_f1, rkv_matches=_ALL_MATCH,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert len(pairs) == 1
    assert pairs[0].eligibility.eligible is False
    assert "full_f1_anchor_incorrect" in pairs[0].eligibility.failure_reasons
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
    assert "rkv_missing_scored_probe" in pairs[0].eligibility.failure_reasons
    assert pairs[0].delta_pss is None


# --- 7. R-KV with zero replay compactions is ineligible ---

def test_rkv_zero_replay_compactions_makes_pair_ineligible(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, rkv_compaction=0,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert len(pairs) == 1
    assert pairs[0].eligibility.eligible is False
    assert "rkv_no_replay_compaction" in pairs[0].eligibility.failure_reasons


def test_cap_hit_canonical_trace_makes_pair_ineligible(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH, cap_hit=True,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    assert pairs[0].eligibility.eligible is False
    assert "canonical_trace_cap_hit" in pairs[0].eligibility.failure_reasons


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


# --- 8. FullKV and R-KV computed from different canonical traces raise an error ---

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


# --- 9. Agreement-rate delta is RKV minus FullKV ---

def test_match_rate_delta_is_rkv_minus_full():
    full_curve = {0.5: 0.4, 0.75: 0.6}
    rkv_curve = {0.5: 0.9, 0.75: 0.6}
    delta = match_rate_delta_rkv_minus_full(full_curve, rkv_curve)
    assert delta[0.5] == pytest.approx(0.5)  # rkv higher match rate -> positive -> less sensitive
    assert delta[0.75] == pytest.approx(0.0)


# --- 10. Analysis produces no p-value at n=10 (or any n) ---

def test_build_fixed_trace_decision_has_no_pvalue_or_ci(tmp_path):
    base, base_records, full_probes, rkv_probes = _make_fixed_trace_run(
        tmp_path, full_matches=_ALL_MATCH, rkv_matches=_ALL_MATCH,
    )
    pairs = build_fixed_trace_pairs(base_records, full_probes, rkv_probes)
    decision = build_fixed_trace_decision(len(pairs), pairs, full_curve={}, rkv_curve={})
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
    decision = build_fixed_trace_decision(len(pairs), pairs, full_curve={0.5: 1.0}, rkv_curve={0.5: 1.0})
    json.dumps(decision)  # must not raise
