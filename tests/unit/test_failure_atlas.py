"""Unit tests for kvcot.failure_atlas (Phase A2, GSM8K protocol-v3 failure
atlas). CPU-only, runs entirely from synthetic JSONL dicts, mirroring
tests/unit/test_fixed_trace_analysis.py's style.

The prompt-offset regression test (`test_prompt_offset_regression_*`) is
mandatory: an earlier manual analysis compared a raw generated-relative
divergence index directly against an absolute compaction position, which
silently misclassified an after-compaction divergence as before-compaction.
"""
from __future__ import annotations

import gzip
import json
import random

import pytest

from kvcot.failure_atlas import (
    EXPECTED_PAIR_COUNT,
    FailureAtlasIntegrityError,
    build_atlas_row,
    build_atlas_summary,
    build_failure_atlas,
    common_prefix_and_divergence,
    divergence_relation_to_compaction,
    pair_and_validate_records,
    reasoning_region_category,
    write_atlas_csv,
    write_atlas_markdown,
)
from kvcot.utils.io import read_jsonl_auto, write_json


# --------------------------------------------------------------- fixtures


def _record(
    src_idx: int = 1,
    seed: int = 42,
    *,
    condition: str = "full",
    record_id: str | None = None,
    question_hash: str = "qh1",
    prompt_token_ids=(1, 2, 3),
    generated_token_ids=(10, 20, 30),
    decoded_output: str = "reasoning text </think> answer text",
    think_parse_status: str = "ok",
    think_end_index: int | None = 1,
    extracted_answer: str | None = "42",
    extraction_method: str = "boxed",
    is_correct: bool | None = True,
    cap_hit: bool = False,
    compaction_count: int = 0,
    compaction_event_steps=(),
    retention: dict | None = None,
    config_sha256: str = "cfg1",
    model_revision: str = "rev1",
    tokenizer_revision: str = "tok1",
    upstream_rkv_commit: str = "up1",
) -> dict:
    return {
        "record_id": record_id or f"base-{condition}-ds-{src_idx}-seed{seed}",
        "condition": condition,
        "global_seed": seed,
        "config_sha256": config_sha256,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "provenance": {"upstream_rkv_commit": upstream_rkv_commit},
        "dataset": {"source_row_index": src_idx, "question_hash": question_hash},
        "prompt_token_ids": list(prompt_token_ids),
        "generated_token_ids": list(generated_token_ids),
        "decoded_output": decoded_output,
        "think_span": {"think_parse_status": think_parse_status, "think_end_index": think_end_index},
        "extracted_answer": extracted_answer,
        "extraction_method": extraction_method,
        "is_correct": is_correct,
        "cap_hit": cap_hit,
        "compaction_count": compaction_count,
        "compaction_event_steps": list(compaction_event_steps),
        "retention": retention,
    }


def _retention(ratio: float) -> dict:
    return {
        "fullkv_equivalent_slots": 1000,
        "physical_cache_slots_per_layer": [int(1000 * ratio)],
        "instantaneous_retention_ratio": ratio,
        "post_compaction_budget_tokens": 128,
        "tokens_since_last_compaction": 5,
    }


def _valid_pairs(n: int = EXPECTED_PAIR_COUNT) -> tuple[list[dict], list[dict]]:
    """n independent, mutually valid (full, rkv) pairs with no divergence,
    used as a known-good baseline that individual tests mutate."""
    full_records, rkv_records = [], []
    for i in range(n):
        prompt = list(range(1000 + i, 1000 + i + 10))
        gen = [i, i + 1, i + 2, i + 3]
        full_records.append(
            _record(
                src_idx=i, condition="full", question_hash=f"qh{i}",
                prompt_token_ids=prompt, generated_token_ids=gen,
                think_end_index=1, is_correct=True, retention=None,
            )
        )
        rkv_records.append(
            _record(
                src_idx=i, condition="rkv_b128", question_hash=f"qh{i}",
                prompt_token_ids=prompt, generated_token_ids=gen,
                think_end_index=1, is_correct=True,
                compaction_count=2, compaction_event_steps=[5, 6],
                retention=_retention(0.4),
            )
        )
    return full_records, rkv_records


# ----------------------------------------------- prompt-offset regression


def test_prompt_offset_regression_pure_function():
    """The mandatory regression case from the A2 brief: prompt=200,
    first_divergence_generated_index=40, first_compaction_absolute_position=230
    must classify as AFTER, not before, first compaction."""
    abs_div = 200 + 40
    assert abs_div == 240
    assert divergence_relation_to_compaction(abs_div, 230) == "after_first_compaction"


def test_prompt_offset_regression_end_to_end():
    prompt = list(range(200))
    common = list(range(9000, 9040))  # 40 identical tokens
    full_gen = common + [1]
    rkv_gen = common + [2]  # diverges at generated index 40
    full = _record(prompt_token_ids=prompt, generated_token_ids=full_gen, think_end_index=1)
    rkv = _record(
        condition="rkv_b128", prompt_token_ids=prompt, generated_token_ids=rkv_gen,
        think_end_index=1, compaction_event_steps=[230], compaction_count=1, retention=_retention(0.3),
    )
    row = build_atlas_row(full, rkv, analysis_code_commit="deadbeef", full_artifact_path="f", rkv_artifact_path="r")
    assert row.first_divergence_generated_index == 40
    assert row.first_divergence_absolute_position == 240
    assert row.first_compaction_absolute_position == 230
    assert row.divergence_relation_to_first_compaction == "after_first_compaction"
    # naive (buggy) comparison would have used the raw generated index (40) against
    # the absolute compaction position (230) and wrongly concluded "before".
    assert 40 < 230
    assert row.divergence_relation_to_first_compaction != "before_first_compaction"


def test_apparent_before_becomes_after_once_offset_applied():
    """Raw generated-index divergence (10) is numerically less than the raw
    compaction position (50) -- but once prompt offset (100) is added, the
    absolute divergence position (110) is actually AFTER it."""
    assert divergence_relation_to_compaction(100 + 10, 50) == "after_first_compaction"


# ------------------------------------------------- divergence arithmetic


def test_divergence_at_generated_index_zero():
    common, div = common_prefix_and_divergence([9, 1, 2], [8, 1, 2])
    assert (common, div) == (0, 0)


def test_divergence_exactly_at_first_compaction():
    assert divergence_relation_to_compaction(15, 15) == "at_first_compaction"


def test_divergence_one_after_first_compaction():
    assert divergence_relation_to_compaction(16, 15) == "after_first_compaction"


def test_divergence_before_first_compaction():
    assert divergence_relation_to_compaction(14, 15) == "before_first_compaction"


def test_completely_identical_sequences():
    common, div = common_prefix_and_divergence([1, 2, 3], [1, 2, 3])
    assert div is None
    assert common == 3


def test_fully_identical_flag_on_row():
    full = _record(generated_token_ids=[1, 2, 3])
    rkv = _record(condition="rkv_b128", generated_token_ids=[1, 2, 3], retention=_retention(0.5))
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.fully_identical_sequences is True
    assert row.identical_until_shorter_terminates is False
    assert row.divergence_exists is False
    assert row.divergence_relation_to_first_compaction == "no_divergence"


def test_full_terminates_first():
    full = _record(generated_token_ids=[1, 2, 3])
    rkv = _record(condition="rkv_b128", generated_token_ids=[1, 2, 3, 4, 5], retention=_retention(0.5))
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.identical_until_shorter_terminates is True
    assert row.fully_identical_sequences is False
    assert row.divergence_exists is False


def test_rkv_terminates_first():
    full = _record(generated_token_ids=[1, 2, 3, 4, 5])
    rkv = _record(condition="rkv_b128", generated_token_ids=[1, 2, 3], retention=_retention(0.5))
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.identical_until_shorter_terminates is True
    assert row.divergence_exists is False


def test_empty_generated_sequences_both_sides():
    full = _record(generated_token_ids=[])
    rkv = _record(condition="rkv_b128", generated_token_ids=[], retention=_retention(0.5))
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.fully_identical_sequences is True
    assert row.common_prefix_token_count == 0


def test_empty_generated_sequence_one_side():
    full = _record(generated_token_ids=[])
    rkv = _record(condition="rkv_b128", generated_token_ids=[1, 2], retention=_retention(0.5))
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.identical_until_shorter_terminates is True
    assert row.common_prefix_token_count == 0


def test_no_compaction_relation():
    assert divergence_relation_to_compaction(10, None) == "no_compaction"


def test_no_divergence_relation():
    assert divergence_relation_to_compaction(None, 10) == "no_divergence"


def test_first_compaction_position_uses_min_regardless_of_list_order():
    full = _record(generated_token_ids=[1, 2, 9])
    rkv = _record(
        condition="rkv_b128", generated_token_ids=[1, 2, 8],
        compaction_event_steps=[50, 12, 30], compaction_count=3, retention=_retention(0.5),
    )
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.first_compaction_absolute_position == 12


# ------------------------------------------------------- </think> region


def test_region_reasoning_before_marker():
    assert reasoning_region_category(5, 20, 20, True, True) == "reasoning"


def test_region_at_marker_boundary():
    assert reasoning_region_category(20, 20, 20, True, True) == "reasoning_to_answer_transition"


def test_region_immediately_after_marker():
    assert reasoning_region_category(21, 20, 20, True, True) == "post_think_answer"


def test_region_identical_through_think_then_diverge():
    full = _record(generated_token_ids=[1, 2, 3, 4, 5], think_end_index=3)
    rkv = _record(
        condition="rkv_b128", generated_token_ids=[1, 2, 3, 4, 9], think_end_index=3, retention=_retention(0.5),
    )
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.identical_through_think is True
    assert row.reasoning_region_category == "post_think_answer"
    assert row.correctness_pair in {"correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong"}


def test_region_one_output_missing_close_marker():
    assert reasoning_region_category(5, 20, None, True, False) == "malformed_or_missing_think_boundary"


def test_region_both_outputs_missing_close_marker():
    assert reasoning_region_category(5, None, None, False, False) == "malformed_or_missing_think_boundary"


def test_region_different_closing_marker_locations_band():
    # full closes at 10, rkv closes at 15; divergence at 12 falls strictly between them.
    assert reasoning_region_category(12, 10, 15, True, True) == "reasoning_to_answer_transition"
    assert reasoning_region_category(9, 10, 15, True, True) == "reasoning"
    assert reasoning_region_category(16, 10, 15, True, True) == "post_think_answer"


def test_region_no_divergence():
    assert reasoning_region_category(None, 10, 10, True, True) == "no_divergence"


def test_row_malformed_think_boundary_full_missing_close_marker():
    full = _record(think_parse_status="no_close_marker", think_end_index=None, generated_token_ids=[1, 2, 3])
    rkv = _record(
        condition="rkv_b128", generated_token_ids=[1, 2, 4], think_end_index=1, retention=_retention(0.5),
    )
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.reasoning_region_category == "malformed_or_missing_think_boundary"
    assert row.identical_through_think is False


def test_marker_repeats_after_first_true():
    from kvcot.failure_atlas import _marker_repeats_after

    tokens = [1, 2, 99, 3, 99, 4]
    assert _marker_repeats_after(tokens, 2) is True


def test_marker_repeats_after_first_false():
    from kvcot.failure_atlas import _marker_repeats_after

    tokens = [1, 2, 99, 3, 4]
    assert _marker_repeats_after(tokens, 2) is False


def test_decoded_suffix_missing_marker_returns_none():
    from kvcot.failure_atlas import _decoded_suffix_after_think

    assert _decoded_suffix_after_think("no marker here at all") is None


def test_decoded_suffix_empty_string():
    from kvcot.failure_atlas import _decoded_suffix_after_think

    assert _decoded_suffix_after_think("") is None


# --------------------------------------------------------- correctness


@pytest.mark.parametrize(
    "full_correct,rkv_correct,pair,flip",
    [
        (True, True, "correct_correct", "correct_to_correct"),
        (True, False, "correct_wrong", "correct_to_wrong"),
        (False, True, "wrong_correct", "wrong_to_correct"),
        (False, False, "wrong_wrong", "wrong_to_wrong"),
    ],
)
def test_correctness_labels(full_correct, rkv_correct, pair, flip):
    full = _record(is_correct=full_correct)
    rkv = _record(condition="rkv_b128", is_correct=rkv_correct, retention=_retention(0.5))
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.correctness_pair == pair
    assert row.flip_direction == flip


def test_null_extraction_is_not_coerced_to_correct():
    full = _record(is_correct=None, extracted_answer=None, extraction_method="none")
    rkv = _record(condition="rkv_b128", is_correct=True, retention=_retention(0.5))
    row = build_atlas_row(full, rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
    assert row.full_is_correct is None
    assert row.correctness_pair == "wrong_correct"  # None is_correct treated as not-correct, never True
    assert row.malformed_answer_full is True


# ------------------------------------------------------------ pairing


def test_pairing_shuffled_order_still_pairs_correctly():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    shuffled_full = full_records[:]
    shuffled_rkv = rkv_records[:]
    random.Random(0).shuffle(shuffled_full)
    random.Random(1).shuffle(shuffled_rkv)
    rows = build_failure_atlas(
        shuffled_full, shuffled_rkv, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r",
    )
    assert [r.source_row_index for r in rows] == list(range(EXPECTED_PAIR_COUNT))
    for row in rows:
        assert row.full_record_id == f"base-full-ds-{row.source_row_index}-seed42"
        assert row.rkv_record_id == f"base-rkv_b128-ds-{row.source_row_index}-seed42"


def test_pairing_duplicate_keys_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    full_records[1] = dict(full_records[1])
    full_records[1]["dataset"] = dict(full_records[1]["dataset"])
    full_records[1]["dataset"]["source_row_index"] = full_records[0]["dataset"]["source_row_index"]
    with pytest.raises(FailureAtlasIntegrityError, match="duplicate"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_missing_rkv_counterpart_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    # rkv loses index 49's key and gains a disjoint one, keeping count at 50
    # but breaking the key-set intersection.
    rkv_records[-1] = dict(rkv_records[-1])
    rkv_records[-1]["dataset"] = dict(rkv_records[-1]["dataset"])
    rkv_records[-1]["dataset"]["source_row_index"] = 999
    rkv_records[-1]["question_hash"] = "qh999"
    with pytest.raises(FailureAtlasIntegrityError, match="not identical"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_missing_full_counterpart_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    full_records[-1] = dict(full_records[-1])
    full_records[-1]["dataset"] = dict(full_records[-1]["dataset"])
    full_records[-1]["dataset"]["source_row_index"] = 999
    with pytest.raises(FailureAtlasIntegrityError, match="not identical"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_mismatched_question_hash_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    rkv_records[0] = dict(rkv_records[0])
    rkv_records[0]["dataset"] = dict(rkv_records[0]["dataset"])
    rkv_records[0]["dataset"]["question_hash"] = "some-other-hash"
    with pytest.raises(FailureAtlasIntegrityError, match="question_hash"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_record_count_mismatch_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    with pytest.raises(FailureAtlasIntegrityError, match="expected exactly 50"):
        pair_and_validate_records(full_records[:-1], rkv_records)


def test_pairing_wrong_count_both_sides_raises():
    full_records, rkv_records = _valid_pairs(10)
    with pytest.raises(FailureAtlasIntegrityError, match="expected exactly 50"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_mismatched_provenance_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    full_records[0] = dict(full_records[0])
    full_records[0]["config_sha256"] = "a-different-config-hash"
    with pytest.raises(FailureAtlasIntegrityError, match="run identity"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_cross_condition_provenance_mismatch_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    for r in rkv_records:
        r["config_sha256"] = "rkv-side-different-hash"
    with pytest.raises(FailureAtlasIntegrityError, match="different provenance"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_mismatched_prompt_token_ids_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    rkv_records[0] = dict(rkv_records[0])
    rkv_records[0]["prompt_token_ids"] = [999, 998, 997]
    with pytest.raises(FailureAtlasIntegrityError, match="prompt_token_ids"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_missing_required_field_raises():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    del full_records[0]["cap_hit"]
    with pytest.raises(FailureAtlasIntegrityError, match="missing required field"):
        pair_and_validate_records(full_records, rkv_records)


def test_pairing_no_compaction_case():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    rkv_records[0] = dict(rkv_records[0])
    rkv_records[0]["compaction_event_steps"] = []
    rkv_records[0]["compaction_count"] = 0
    rkv_records[0]["generated_token_ids"] = [999]  # force a divergence so relation is meaningful
    rows = build_failure_atlas(
        full_records, rkv_records, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r",
    )
    row0 = next(r for r in rows if r.source_row_index == 0)
    assert row0.first_compaction_absolute_position is None
    assert row0.divergence_relation_to_first_compaction == "no_compaction"


def test_pairing_output_is_stably_sorted_by_source_row_index():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    rows = build_failure_atlas(
        full_records, rkv_records, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r",
    )
    assert [r.source_row_index for r in rows] == sorted(r.source_row_index for r in rows)


# ------------------------------------------------------------- summary


def test_build_atlas_summary_invariants_hold_for_valid_set():
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    rows = build_failure_atlas(
        full_records, rkv_records, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r",
    )
    summary = build_atlas_summary(
        rows, full_artifact_path="f", rkv_artifact_path="r",
        full_artifact_sha256="sha_f", rkv_artifact_sha256="sha_r", analysis_code_commit="c",
    )
    assert summary["n_pairs"] == EXPECTED_PAIR_COUNT
    assert all(summary["invariant_checks"].values())
    assert summary["operating_point_valid"] is False
    assert summary["hypothesis_status"] == "not_tested"
    assert summary["diagnostic_label"] == "post_hoc_diagnostic"


def test_build_atlas_summary_flags_wrong_row_count():
    # build_atlas_summary's own invariant/warning logic is independent of
    # pair_and_validate_records's hard 50-count gate -- exercise it directly
    # against a short row list (as could happen if it were ever called on a
    # different-sized atlas) rather than through the strict pairing path.
    full_records, rkv_records = _valid_pairs(10)
    rows = [
        build_atlas_row(f, r, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
        for f, r in zip(full_records, rkv_records)
    ]
    summary = build_atlas_summary(
        rows, full_artifact_path="f", rkv_artifact_path="r",
        full_artifact_sha256="sha_f", rkv_artifact_sha256="sha_r", analysis_code_commit="c",
    )
    assert summary["invariant_checks"]["row_count_is_50"] is False
    assert any("expected 50" in w for w in summary["warnings"])


# -------------------------------------------------------------- writers


def test_write_atlas_csv_row_count(tmp_path):
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    rows = build_failure_atlas(
        full_records, rkv_records, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r",
    )
    out = tmp_path / "atlas.csv"
    write_atlas_csv(rows, out)
    import csv as _csv

    with open(out, encoding="utf-8") as f:
        data_rows = list(_csv.DictReader(f))
    assert len(data_rows) == EXPECTED_PAIR_COUNT


def test_write_atlas_markdown_has_required_sections(tmp_path):
    full_records, rkv_records = _valid_pairs(EXPECTED_PAIR_COUNT)
    rows = build_failure_atlas(
        full_records, rkv_records, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r",
    )
    summary = build_atlas_summary(
        rows, full_artifact_path="f", rkv_artifact_path="r",
        full_artifact_sha256="sha_f", rkv_artifact_sha256="sha_r", analysis_code_commit="c",
    )
    out = tmp_path / "atlas.md"
    write_atlas_markdown(rows, summary, out)
    text = out.read_text(encoding="utf-8")
    assert "Identical reasoning with final-answer flips" in text
    assert "Methodology and coordinate conventions" in text
    assert "Limitations and claim boundary" in text
    assert "post-hoc diagnostic" in text


def test_write_atlas_markdown_handles_missing_special_rows(tmp_path):
    """source rows 30/271/1115 need not exist in an arbitrary synthetic set --
    the markdown writer must say so, not crash."""
    full_records, rkv_records = _valid_pairs(5)
    rows = [
        build_atlas_row(f, r, analysis_code_commit="c", full_artifact_path="f", rkv_artifact_path="r")
        for f, r in zip(full_records, rkv_records)
    ]
    summary = build_atlas_summary(
        rows, full_artifact_path="f", rkv_artifact_path="r",
        full_artifact_sha256="sha_f", rkv_artifact_sha256="sha_r", analysis_code_commit="c",
    )
    out = tmp_path / "atlas.md"
    write_atlas_markdown(rows, summary, out)
    text = out.read_text(encoding="utf-8")
    assert "not present among the 50 paired examples" in text


# --------------------------------------------------------------- io


def test_read_jsonl_auto_reads_plain_jsonl(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl_auto(p)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_auto_reads_gzip_jsonl(tmp_path):
    p = tmp_path / "x.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write('{"a": 1}\n{"a": 2}\n')
    assert list(read_jsonl_auto(p)) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_auto_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_jsonl_auto(tmp_path / "does_not_exist.jsonl.gz"))
