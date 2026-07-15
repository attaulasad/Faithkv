"""Unit tests for the record -> primary-result glue (kvcot.analysis.pipeline).

Runs entirely from synthetic JSONL dicts on the CPU — no model, no torch.
Validates the join (probe -> base, per condition/seed), match/EAS/Delta_EAS
wiring, §8.3 eligibility, the paired-accuracy inputs, and the attrition-funnel
row-inputs. The statistical tests themselves are covered by test_metrics.py /
the stats unit tests; here we only check the wiring that feeds them.
"""
from __future__ import annotations

from pathlib import Path

from kvcot.analysis.pipeline import (
    build_pair_results,
    count_answer_changed_at_any_scored_fraction,
    funnel_records,
    load_condition_records,
    paired_accuracy_inputs,
    problem_level_delta_eas,
)
from kvcot.config import PROBE_FRACTIONS_ALL, PROBE_FRACTIONS_SCORED
from kvcot.utils.io import JsonlWriter


def _base_record(condition: str, src_idx: int, seed: int, *, answer: str, correct: bool,
                 parse_status: str = "generation_prompt_preopened_ok", compaction_count: int = 3) -> dict:
    return {
        "record_id": f"base-{condition}-ds-{src_idx}-seed{seed}",
        "global_seed": seed,
        "condition": condition,
        "dataset": {"source_row_index": src_idx},
        "extracted_answer": answer,
        "is_correct": correct,
        "cap_hit": False,
        "compaction_count": compaction_count,
        "think_span": {"think_parse_status": parse_status},
    }


def _probe_record(base_record_id: str, fraction: float, probe_answer: str | None) -> dict:
    return {
        "record_id": f"probe-{base_record_id}-f{fraction}",
        "base_record_id": base_record_id,
        "fraction": fraction,
        "normalized_probe_answer": probe_answer,
    }


def _write(path: Path, rows: list[dict]) -> None:
    w = JsonlWriter(path, validator=None)
    for r in rows:
        w.append(r)


def _make_run(tmp_path: Path, *, full_probe_answers, rkv_probe_answers,
              base_answer="42", full_correct=True, rkv_correct=True,
              rkv_compaction=3, seeds=(13, 42)):
    """Build a one-problem, N-seed run. `*_probe_answers` maps fraction->answer
    (applied to every seed) for f in PROBE_FRACTIONS_ALL."""
    full_base, rkv_base, full_probes, rkv_probes = [], [], [], []
    for seed in seeds:
        fb = _base_record("full", 0, seed, answer=base_answer, correct=full_correct)
        rb = _base_record("rkv_b128", 0, seed, answer=base_answer, correct=rkv_correct,
                          compaction_count=rkv_compaction)
        full_base.append(fb)
        rkv_base.append(rb)
        for f in PROBE_FRACTIONS_ALL:
            full_probes.append(_probe_record(fb["record_id"], f, full_probe_answers.get(f)))
            rkv_probes.append(_probe_record(rb["record_id"], f, rkv_probe_answers.get(f)))
    _write(tmp_path / "full.jsonl", full_base)
    _write(tmp_path / "full_probes.jsonl", full_probes)
    _write(tmp_path / "rkv_b128.jsonl", rkv_base)
    _write(tmp_path / "rkv_b128_probes.jsonl", rkv_probes)
    full = load_condition_records(tmp_path / "full.jsonl", tmp_path / "full_probes.jsonl", "full")
    rkv = load_condition_records(tmp_path / "rkv_b128.jsonl", tmp_path / "rkv_b128_probes.jsonl", "rkv_b128")
    return full, rkv


def test_positive_delta_when_rkv_less_sensitive(tmp_path):
    # FullKV changes its answer at 3 of 7 scored fractions; R-KV never does.
    # EAS_full = 3/7, EAS_rkv = 0 -> Delta_EAS = 3/7 (the hypothesized sign).
    full_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    for f in (0.125, 0.375, 0.625):
        full_ans[f] = "99"  # mismatch base "42"
    rkv_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}

    full, rkv = _make_run(tmp_path, full_probe_answers=full_ans, rkv_probe_answers=rkv_ans)
    pairs = build_pair_results(full, rkv)
    assert len(pairs) == 2  # two seeds
    for p in pairs:
        assert p.eligibility.eligible
        assert abs(p.delta_eas - 3 / 7) < 1e-9

    aggregates, primary_values = problem_level_delta_eas(pairs)
    assert len(primary_values) == 1  # one problem, >=2 eligible seeds
    assert abs(primary_values[0] - 3 / 7) < 1e-9


def test_ineligible_when_rkv_never_compacts(tmp_path):
    full_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    rkv_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    full, rkv = _make_run(tmp_path, full_probe_answers=full_ans, rkv_probe_answers=rkv_ans,
                          rkv_compaction=0)
    pairs = build_pair_results(full, rkv)
    for p in pairs:
        assert not p.eligibility.eligible
        assert "rkv_no_compaction" in p.eligibility.failure_reasons
        assert p.delta_eas is None
    _aggs, primary = problem_level_delta_eas(pairs)
    assert primary == []  # nothing enters the primary tests


def test_missing_scored_probe_makes_pair_ineligible(tmp_path):
    full_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    rkv_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    full, rkv = _make_run(tmp_path, full_probe_answers=full_ans, rkv_probe_answers=rkv_ans)
    # Drop one scored probe from the R-KV side by rewriting its file without f=0.5.
    kept = [
        r for r in _read(tmp_path / "rkv_b128_probes.jsonl")
        if float(r["fraction"]) != 0.5
    ]
    _rewrite(tmp_path / "rkv_b128_probes.jsonl", kept)
    rkv = load_condition_records(tmp_path / "rkv_b128.jsonl", tmp_path / "rkv_b128_probes.jsonl", "rkv_b128")
    pairs = build_pair_results(full, rkv)
    for p in pairs:
        assert not p.eligibility.eligible
        assert "missing_scored_probe" in p.eligibility.failure_reasons


def test_extraction_failure_makes_match_undefined_and_eas_undefined(tmp_path):
    full_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    full_ans[0.5] = None  # extraction failure at a scored fraction
    rkv_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    full, rkv = _make_run(tmp_path, full_probe_answers=full_ans, rkv_probe_answers=rkv_ans)
    pairs = build_pair_results(full, rkv)
    for p in pairs:
        # all scored probes are present, but one match is undefined -> EAS_full
        # undefined -> Delta undefined even though the pair is otherwise eligible.
        assert p.delta_eas is None


def test_paired_accuracy_inputs_average_over_seeds(tmp_path):
    full_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    rkv_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    # full correct on both seeds; rkv correct on one of two -> 1.0 vs 0.5.
    full, rkv = _make_run(tmp_path, full_probe_answers=full_ans, rkv_probe_answers=rkv_ans,
                          seeds=(13, 42))
    # Flip one rkv seed to incorrect by rewriting.
    rows = _read(tmp_path / "rkv_b128.jsonl")
    rows[0]["is_correct"] = False
    _rewrite(tmp_path / "rkv_b128.jsonl", rows)
    rkv = load_condition_records(tmp_path / "rkv_b128.jsonl", tmp_path / "rkv_b128_probes.jsonl", "rkv_b128")
    full_acc, rkv_acc = paired_accuracy_inputs(full, rkv)
    assert full_acc == [1.0]
    assert rkv_acc == [0.5]


def test_funnel_and_stage1a_counts(tmp_path):
    full_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    for f in (0.25,):
        full_ans[f] = "7"  # one change -> this base "changed at some fraction"
    rkv_ans = {f: "42" for f in PROBE_FRACTIONS_ALL}
    full, rkv = _make_run(tmp_path, full_probe_answers=full_ans, rkv_probe_answers=rkv_ans)

    frecs = funnel_records(full)
    assert len(frecs) == 2
    assert all(r["base_generated"] and r["correct"] and r["f1_stable"] for r in frecs)

    n_eligible, n_changed = count_answer_changed_at_any_scored_fraction(full)
    assert n_eligible == 2
    assert n_changed == 2  # both seeds changed at f=0.25


# --- tiny JSONL helpers used only by the rewrite-based tests above ---
def _read(path: Path) -> list[dict]:
    from kvcot.utils.io import read_jsonl
    return list(read_jsonl(path))


def _rewrite(path: Path, rows: list[dict]) -> None:
    path.unlink()
    w = JsonlWriter(path, validator=None)
    for r in rows:
        w.append(r)
