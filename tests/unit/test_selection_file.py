"""CPU-only tests for protocol-v3's --selection-file support (2026-07-18
review): loading/verifying a results/selections/{stage}.json before trusting
it (kvcot.cli._load_fixed_trace_selection) and the analysis completeness
guard that aborts rather than silently treating a partial replay as ordinary
attrition (kvcot.analysis.fixed_trace._verify_selection_completeness /
run_fixed_trace_analysis's selected_base_record_ids parameter).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kvcot.cli import SelectionFileMismatchError, _load_fixed_trace_selection
from kvcot.config import FixedTraceSettings, StageConfig, load_lock_config
from kvcot.utils.io import JsonlWriter, write_json

_LOCK_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "lock.yaml")


def _write_base_file(path, rows: list[dict]) -> None:
    w = JsonlWriter(path, validator=None)
    for r in rows:
        w.append(r)


def _base_row(idx: int) -> dict:
    return {"record_id": f"base-full-x-{idx}"}


def _stage(tmp_path):
    return StageConfig(
        stage_name="test_stage", dataset_manifest="data/manifests/gsm8k_calibration_50.jsonl",
        conditions=["full", "rkv_b128"], rkv_budgets=[128], output_dir=str(tmp_path),
        fixed_trace=FixedTraceSettings(min_eligible_examples=1),
    )


def _valid_selection(tmp_path, stage, lock, base_path) -> dict:
    from kvcot.config import config_identity
    from kvcot.utils.hashing import sha256_file

    return {
        "stage_name": stage.stage_name,
        "config_path": _LOCK_PATH,
        "config_sha256": config_identity(_LOCK_PATH),
        "base_path": str(base_path),
        "base_file_sha256": sha256_file(base_path),
        "budget": stage.rkv_budgets[0],
        "divide_length": lock.rkv.divide_length,
        "meaningful_retention_ceiling": 0.7,
        "min_meaningfully_compressed_scored_fractions": 2,
        "max_selected": None,
        "n_candidates_considered": 2,
        "n_rejected_invalid_base": 0,
        "n_ranked": 2,
        "n_predicted_eligible": 2,
        "n_selected": 2,
        "selected_source_row_indices": [0, 1],
        "selected_base_record_ids": ["base-full-x-0", "base-full-x-1"],
        "candidates": [
            {"source_row_index": 0, "base_record_id": "base-full-x-0", "predicted_eligible": True},
            {"source_row_index": 1, "base_record_id": "base-full-x-1", "predicted_eligible": True},
        ],
    }


def test_load_fixed_trace_selection_accepts_matching_selection(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    loaded = _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)
    assert loaded["selected_base_record_ids"] == ["base-full-x-0", "base-full-x-1"]


def test_load_fixed_trace_selection_rejects_config_sha256_mismatch(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["config_sha256"] = "0" * 64
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="config_sha256"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_stale_base_file(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    # Base file changed AFTER the selection was written (e.g. regenerated).
    base_path.unlink()
    _write_base_file(base_path, [_base_row(0), _base_row(1), _base_row(2)])

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="base_file_sha256"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_budget_mismatch(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["budget"] = 999999
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="budget"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_divide_length_mismatch(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["divide_length"] = 1
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="divide_length"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_stage_name_mismatch(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["stage_name"] = "some_other_stage"
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="stage_name"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


# --- 2026-07-19 review: full identity-tuple cross-validation ---
# (previously only the top-level config/base/budget/divide_length/stage_name
# fields were checked; selected_base_record_ids/selected_source_row_indices/
# candidates could silently disagree with each other.)

def test_load_fixed_trace_selection_rejects_row_index_id_disagreement(tmp_path):
    # The exact repro from the review: selected_source_row_indices and
    # selected_base_record_ids disagree with what the candidate itself
    # records for that base_record_id.
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    # base-full-x-0's own candidate says source_row_index=0, but the
    # top-level selected_source_row_indices list now claims 5 for it.
    selection["selected_source_row_indices"] = [5, 1]
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="disagree"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_selected_id_absent_from_candidates(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["selected_base_record_ids"] = ["base-full-x-0", "base-full-x-999"]
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="does not appear in candidates"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_non_eligible_selected_candidate(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["candidates"][0]["predicted_eligible"] = False  # base-full-x-0, but still selected
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="predicted_eligible"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_duplicate_selected_ids(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["selected_base_record_ids"] = ["base-full-x-0", "base-full-x-0"]
    selection["selected_source_row_indices"] = [0, 0]
    selection["n_selected"] = 2
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="duplicates"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_n_selected_mismatch(tmp_path):
    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = _stage(tmp_path)
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["n_selected"] = 5  # actual list only has 2 entries
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="n_selected"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)


def test_load_fixed_trace_selection_rejects_max_selected_not_matching_preregistered_config(tmp_path):
    from kvcot.config import FixedTraceSettings, StageConfig

    _write_base_file(tmp_path / "full.jsonl", [_base_row(0), _base_row(1)])
    stage = StageConfig(
        stage_name="test_stage", dataset_manifest="data/manifests/gsm8k_calibration_50.jsonl",
        conditions=["full", "rkv_b128"], rkv_budgets=[128], output_dir=str(tmp_path),
        fixed_trace=FixedTraceSettings(min_eligible_examples=1, max_selected_examples=20),
    )
    lock = load_lock_config("configs/lock.yaml")
    base_path = tmp_path / "full.jsonl"
    selection = _valid_selection(tmp_path, stage, lock, base_path)
    selection["max_selected"] = 5  # e.g. written under a CLI --max-selected override
    selection_path = tmp_path / "selection.json"
    write_json(selection_path, selection)

    args = SimpleNamespace(config=_LOCK_PATH)
    with pytest.raises(SelectionFileMismatchError, match="max_selected"):
        _load_fixed_trace_selection(str(selection_path), args, stage, lock, base_path)
