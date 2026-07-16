import pytest
from pydantic import ValidationError

from kvcot.schemas import (
    BaseRunRecord,
    ProbeRunRecord,
    ProvenanceState,
    VersionInfo,
    DatasetProvenance,
    MethodConfig,
    ThinkSpanInfo,
    RunManifest,
)


def _provenance():
    return ProvenanceState(upstream_rkv_commit="45eaa7d69d20b7388321f077020a610d9afb65bd", git_commit="deadbeef", git_dirty=False)


def _versions():
    return VersionInfo(python="3.10.0")


def _dataset():
    return DatasetProvenance(
        dataset_name="gsm8k",
        source_row_index=3,
        question_hash="a" * 64,
        normalized_gold="42",
    )


def _think_span():
    return ThinkSpanInfo(
        think_start_index=0,
        think_end_index=120,
        think_parse_status="generation_prompt_preopened_ok",
        generation_prompt_preopened_think=True,
    )


def test_base_run_record_valid_construction():
    rec = BaseRunRecord(
        record_id="r1",
        config_path="configs/stage0_smoke.yaml",
        config_sha256="b" * 64,
        provenance=_provenance(),
        versions=_versions(),
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        model_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        tokenizer_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        dataset=_dataset(),
        condition="full",
        method_config=MethodConfig(method="fullkv"),
        global_seed=42,
        derived_seed=123456,
        prompt_text="hello",
        prompt_token_ids=[1, 2, 3],
        generated_token_ids=[4, 5, 6],
        decoded_output="the answer is 42",
        think_span=_think_span(),
        extracted_answer="42",
        extraction_method="boxed",
        is_correct=True,
        cap_hit=False,
        wall_time_seconds=1.23,
        generated_token_count=3,
        compaction_count=0,
        compaction_event_steps=[],
        cache_length_final_per_layer=[3, 3],
    )
    assert rec.schema_version == "1.2.0"  # bumped for the fixed-trace protocol v2 fields
    assert rec.record_type == "base_generation"


def test_schema_version_literal_rejects_stale_value():
    with pytest.raises(ValidationError):
        BaseRunRecord(
            record_id="r1",
            schema_version="1.1.0",
            config_path="configs/stage0_smoke.yaml",
            config_sha256="b" * 64,
            provenance=_provenance(),
            versions=_versions(),
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            model_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
            tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            tokenizer_revision="ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
            dataset=_dataset(),
            condition="full",
            method_config=MethodConfig(method="fullkv"),
            global_seed=42,
            derived_seed=123456,
            prompt_text="hello",
            prompt_token_ids=[1, 2, 3],
            generated_token_ids=[4, 5, 6],
            decoded_output="the answer is 42",
            think_span=_think_span(),
            extracted_answer="42",
            extraction_method="boxed",
            is_correct=True,
            cap_hit=False,
            wall_time_seconds=1.23,
            generated_token_count=3,
            compaction_count=0,
            compaction_event_steps=[],
            cache_length_final_per_layer=[3, 3],
        )


def test_base_run_record_rejects_missing_required_field():
    with pytest.raises(ValidationError):
        BaseRunRecord(
            record_id="r1",
            config_path="x",
            config_sha256="b" * 64,
            provenance=_provenance(),
            versions=_versions(),
            model_name="m",
            model_revision="r",
            tokenizer_name="m",
            tokenizer_revision="r",
            dataset=_dataset(),
            condition="full",
            method_config=MethodConfig(method="fullkv"),
            global_seed=42,
            derived_seed=1,
            prompt_text="x",
            prompt_token_ids=[1],
            generated_token_ids=[1],
            decoded_output="x",
            think_span=_think_span(),
            extracted_answer="1",
            extraction_method="boxed",
            is_correct=True,
            cap_hit=False,
            wall_time_seconds=1.0,
            generated_token_count=1,
            compaction_count=0,
            compaction_event_steps=[],
            # cache_length_final_per_layer intentionally omitted
        )


def test_is_correct_allows_none_on_extraction_failure():
    rec = BaseRunRecord(
        record_id="r2",
        config_path="x",
        config_sha256="b" * 64,
        provenance=_provenance(),
        versions=_versions(),
        model_name="m",
        model_revision="r",
        tokenizer_name="m",
        tokenizer_revision="r",
        dataset=_dataset(),
        condition="full",
        method_config=MethodConfig(method="fullkv"),
        global_seed=42,
        derived_seed=1,
        prompt_text="x",
        prompt_token_ids=[1],
        generated_token_ids=[1],
        decoded_output="no answer here",
        think_span=_think_span(),
        extracted_answer=None,
        extraction_method="none",
        extraction_failure_reason="no_answer_found",
        is_correct=None,
        cap_hit=False,
        wall_time_seconds=1.0,
        generated_token_count=1,
        compaction_count=0,
        compaction_event_steps=[],
        cache_length_final_per_layer=[1],
    )
    assert rec.is_correct is None


def test_method_config_rejects_unknown_method():
    with pytest.raises(ValidationError):
        MethodConfig(method="snapkv")  # out of scope per §1, must not validate


def test_probe_run_record_valid_construction():
    rec = ProbeRunRecord(
        record_id="p1",
        parent_record_id="r1",
        config_path="x",
        config_sha256="b" * 64,
        provenance=_provenance(),
        versions=_versions(),
        base_record_id="r1",
        condition="rkv_b1024",
        fraction=0.5,
        think_span_length=100,
        cut_index=50,
        control_suffix_token_ids=[1, 2, 3],
        probe_decoding_max_new_tokens=48,
        probe_output_token_ids=[9, 9],
        probe_output_text="Final answer: \\boxed{42}",
        normalized_probe_answer="42",
        probe_extraction_status="boxed",
        matches_own_condition_base_answer=True,
        is_f1_stability_probe=False,
        snapshot_cache_hash="c" * 64,
        snapshot_provenance_hash="d" * 64,
        snapshot_state_hash="e" * 64,
    )
    assert rec.fraction == 0.5


def test_run_manifest_defaults():
    m = RunManifest(
        command="kvcot generate",
        config_path="configs/stage0_smoke.yaml",
        config_sha256="b" * 64,
        git_commit="deadbeef",
        git_dirty=False,
        versions=_versions(),
        start_time_utc="2026-07-15T00:00:00+00:00",
    )
    assert m.n_attempted == 0
    assert m.n_completed == 0
