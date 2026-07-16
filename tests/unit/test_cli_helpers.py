"""Unit tests for the CPU-safe helper functions in kvcot.cli (never the
torch-dependent `cmd_*` bodies themselves — those are exercised on GPU).
Covers the resume-identity verification (§13), condition resolution
(rkv_selected -> rkv_b{budget}), and the Stage 1B candidate-budget globber —
all pure/IO-only logic that was previously only smoke-tested manually.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kvcot.cli import (
    ResumeIdentityMismatchError,
    _get_dotted,
    _load_manifest_filtered,
    _resolve_condition,
    _stage1b_candidate_budgets,
    _verify_resumable_record_ids,
)
from kvcot.config import StageConfig
from kvcot.schemas import (
    BaseRunRecord,
    DatasetProvenance,
    MethodConfig,
    ProvenanceState,
    ThinkSpanInfo,
    VersionInfo,
)
from kvcot.utils.io import JsonlWriter


def _valid_base_record(**overrides) -> dict:
    defaults = dict(
        record_id="base-full-gsm8k_smoke_20-0-seed42",
        config_path="configs/stage0_smoke.yaml",
        config_sha256="a" * 64,
        provenance=ProvenanceState(upstream_rkv_commit="c" * 40, git_commit="x", git_dirty=False),
        versions=VersionInfo(python="3.10.0"),
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        model_revision="r" * 40,
        tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tokenizer_revision="r" * 40,
        dataset=DatasetProvenance(dataset_name="gsm8k_smoke_20", source_row_index=0, question_hash="x", normalized_gold="1"),
        condition="full",
        method_config=MethodConfig(method="fullkv"),
        global_seed=42,
        derived_seed=1,
        prompt_text="p",
        prompt_token_ids=[1],
        generated_token_ids=[2],
        decoded_output="d",
        think_span=ThinkSpanInfo(think_start_index=0, think_end_index=1, think_parse_status="ok", generation_prompt_preopened_think=False),
        extracted_answer="1",
        extraction_method="boxed",
        is_correct=True,
        cap_hit=False,
        wall_time_seconds=1.0,
        generated_token_count=1,
        compaction_count=0,
        compaction_event_steps=[],
        cache_length_final_per_layer=[1],
    )
    defaults.update(overrides)
    return BaseRunRecord(**defaults).model_dump(mode="json")


EXPECTED_IDENTITY = {
    "config_sha256": "a" * 64,
    "model_revision": "r" * 40,
    "tokenizer_revision": "r" * 40,
    "provenance.upstream_rkv_commit": "c" * 40,
}


def test_verify_resumable_accepts_matching_identity(tmp_path):
    path = tmp_path / "full.jsonl"
    JsonlWriter(path, validator=None).append(_valid_base_record())
    ids = _verify_resumable_record_ids(path, BaseRunRecord, EXPECTED_IDENTITY)
    assert ids == {"base-full-gsm8k_smoke_20-0-seed42"}


def test_verify_resumable_rejects_config_hash_mismatch(tmp_path):
    path = tmp_path / "full.jsonl"
    JsonlWriter(path, validator=None).append(_valid_base_record(config_sha256="b" * 64))
    with pytest.raises(ResumeIdentityMismatchError, match="config_sha256"):
        _verify_resumable_record_ids(path, BaseRunRecord, EXPECTED_IDENTITY)


def test_verify_resumable_rejects_upstream_commit_mismatch(tmp_path):
    path = tmp_path / "full.jsonl"
    bad_record = _valid_base_record()
    bad_record["provenance"]["upstream_rkv_commit"] = "d" * 40
    JsonlWriter(path, validator=None).append(bad_record)
    with pytest.raises(ResumeIdentityMismatchError, match="upstream_rkv_commit"):
        _verify_resumable_record_ids(path, BaseRunRecord, EXPECTED_IDENTITY)


def test_verify_resumable_rejects_schema_invalid_row(tmp_path):
    path = tmp_path / "full.jsonl"
    JsonlWriter(path, validator=None).append({"record_id": "x", "config_sha256": "a" * 64})
    with pytest.raises(ResumeIdentityMismatchError, match="schema validation"):
        _verify_resumable_record_ids(path, BaseRunRecord, EXPECTED_IDENTITY)


def test_verify_resumable_empty_file_returns_empty_set(tmp_path):
    path = tmp_path / "full.jsonl"
    path.write_text("")
    assert _verify_resumable_record_ids(path, BaseRunRecord, EXPECTED_IDENTITY) == set()


def test_get_dotted_traverses_nested_dict():
    row = {"provenance": {"upstream_rkv_commit": "abc"}}
    assert _get_dotted(row, "provenance.upstream_rkv_commit") == "abc"
    assert _get_dotted(row, "provenance") == {"upstream_rkv_commit": "abc"}


def _stage(conditions, **overrides):
    defaults = dict(
        stage_name="stage0_smoke",
        dataset_manifest="data/manifests/gsm8k_smoke_20.jsonl",
        conditions=conditions,
        output_dir="results/raw/stage0_smoke",
    )
    defaults.update(overrides)
    return StageConfig(**defaults)


def test_resolve_condition_passthrough_for_concrete_condition():
    stage = _stage(["full", "patched_noop", "rkv_b96"])
    args = SimpleNamespace(condition="rkv_b96")
    assert _resolve_condition(stage, args) == "rkv_b96"


def test_resolve_condition_rejects_unknown_condition():
    stage = _stage(["full", "patched_noop", "rkv_b96"])
    args = SimpleNamespace(condition="rkv")  # the Makefile's old, invalid name
    with pytest.raises(SystemExit):
        _resolve_condition(stage, args)


def test_resolve_condition_resolves_rkv_selected_placeholder(monkeypatch):
    stage = _stage(["full", "rkv_selected"], stage_name="stage2_main")
    args = SimpleNamespace(condition="rkv_selected")
    monkeypatch.setattr("kvcot.cli.resolve_conditions", lambda s: ["full", "rkv_b256"])
    assert _resolve_condition(stage, args) == "rkv_b256"


def test_stage1b_candidate_budgets_globs_and_sorts(tmp_path):
    for name in ["stage1b_budget_512.yaml", "stage1b_budget_128.yaml", "stage1b_budget_256.yaml", "not_a_budget.yaml"]:
        (tmp_path / name).write_text("stage_name: x\n")
    assert _stage1b_candidate_budgets(tmp_path) == [128, 256, 512]


def test_stage1b_candidate_budgets_empty_dir(tmp_path):
    assert _stage1b_candidate_budgets(tmp_path) == []


# --- _load_manifest_filtered / stage.limit (§ external review 2026-07-16) ---
#
# StageConfig.limit (e.g. early_gap_v2_b128.yaml's `limit: 10`) previously
# had NO effect here -- only `--limit` on the CLI was ever consulted, so
# every fixed-trace stage config's documented "ten-example screen" silently
# ran against the full manifest whenever a command was invoked without an
# explicit `--limit`.

def _write_manifest(path, n_rows: int) -> None:
    w = JsonlWriter(path, validator=None)
    for i in range(n_rows):
        w.append({"source_row_index": i, "question": f"q{i}", "question_hash": f"h{i}", "normalized_gold": str(i)})


def test_load_manifest_filtered_uses_stage_limit_when_cli_limit_omitted(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, 50)
    stage = _stage(["full"], dataset_manifest=str(manifest_path), limit=10)
    args = SimpleNamespace(problem_index=None, limit=None)
    rows = _load_manifest_filtered(stage, args)
    assert len(rows) == 10


def test_load_manifest_filtered_cli_limit_overrides_stage_limit(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, 50)
    stage = _stage(["full"], dataset_manifest=str(manifest_path), limit=10)
    args = SimpleNamespace(problem_index=None, limit=1)
    rows = _load_manifest_filtered(stage, args)
    assert len(rows) == 1


def test_load_manifest_filtered_no_limit_anywhere_returns_all_rows(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, 50)
    stage = _stage(["full"], dataset_manifest=str(manifest_path))  # no limit set
    args = SimpleNamespace(problem_index=None, limit=None)
    rows = _load_manifest_filtered(stage, args)
    assert len(rows) == 50
