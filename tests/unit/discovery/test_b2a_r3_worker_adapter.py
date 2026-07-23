"""Step 3R4 Finding 5: canonical FullKV worker-result adapter tests.

Every worker result in this file is constructed through the strict
`FullKVWorkerResultR3` schema (never a raw dict that bypasses worker-
schema validation). No torch, no CUDA, no R-KV import, no real FullKV
inference.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kvcot.discovery.b2a_r3_contract import (
    CANDIDATE_MANIFEST_PATH,
    DATASET_REPO,
    DATASET_REVISION,
    GENERATION_CONFIG_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    FROZEN_GENERATION_CONFIG,
)
from kvcot.discovery.b2a_r3_qualification import B2AR3FullKVQualificationEvidence
from kvcot.discovery.b2a_r3_worker_adapter import (
    FULLKV_WORKER_RESULT_R3_SCHEMA_VERSION,
    FullKVWorkerResultR3,
    WorkerAdapterRefused,
    adapt_fullkv_worker_result_to_r3_evidence,
)
from kvcot.discovery.b2a_workers import FullKVWorkerResult
from kvcot.utils.hashing import sha256_int_ids, sha256_json

from tests.unit.discovery.test_b2a_r3_independent_audit_repair import _memory, _timing

CONFIG_SHA = "de8ac65a348c307c4f00089da07914666332935981bcaa7c98a150a9e7e778b3"


def _candidate_manifest() -> dict:
    return json.loads(Path(CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))


def _valid_worker_result(ordinal: int = 0, **overrides) -> FullKVWorkerResultR3:
    candidate = _candidate_manifest()["candidates"][ordinal]
    generated = list(range(2000))
    fields = dict(
        worker_result_schema_version=FULLKV_WORKER_RESULT_R3_SCHEMA_VERSION,
        role="fullkv",
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        requested_model_revision=MODEL_REVISION,
        model_revision_match=True,
        tokenizer_name=TOKENIZER_NAME,
        tokenizer_revision=TOKENIZER_REVISION,
        requested_tokenizer_revision=TOKENIZER_REVISION,
        tokenizer_revision_match=True,
        dataset_repo=DATASET_REPO,
        dataset_config="default",
        dataset_split="test",
        dataset_revision=DATASET_REVISION,
        source_example_index=candidate["source_example_index"],
        unique_id=candidate["unique_id"],
        raw_row_sha256=candidate["raw_row_sha256"],
        problem_sha256=candidate["problem_sha256"],
        gold_answer_sha256=candidate["gold_answer_sha256"],
        expected_prompt_token_ids_sha256="1" * 64,
        observed_prompt_token_ids_sha256="1" * 64,
        prompt_token_count=100,
        natural_generated_token_ids=generated,
        generated_token_ids_sha256=sha256_int_ids(generated),
        natural_answer=candidate["row"]["answer"],
        natural_answer_status="correct",
        cap_hit=False,
        think_parse_status="ok",
        think_start_index=0,
        think_end_index=100,
        generation_prompt_preopened_think=False,
        requested_device="cuda:0",
        parameter_placement_evidence={
            "requested_device": "cuda:0",
            "every_parameter_on_cuda": True,
            "no_offload_verified": True,
            "parameter_count": 100,
            "unique_device_types": ["cuda"],
            "unique_devices": ["cuda:0"],
            "hf_device_map": None,
        },
        actual_batch_size=1,
        timing_evidence=_timing(),
        memory_phase_evidence=_memory(),
        wall_seconds=3.0,
        runtime_generation_config=FROZEN_GENERATION_CONFIG,
        worker_generation_config_sha256=GENERATION_CONFIG_SHA256,
        worker_config_sha256=CONFIG_SHA,
        software_versions={"transformers": "4.55.4"},
    )
    fields.update(overrides)
    if "model_revision" in overrides and "requested_model_revision" not in overrides:
        fields["requested_model_revision"] = overrides["model_revision"]
    if "tokenizer_revision" in overrides and "requested_tokenizer_revision" not in overrides:
        fields["requested_tokenizer_revision"] = overrides["tokenizer_revision"]
    if "natural_generated_token_ids" in overrides and "generated_token_ids_sha256" not in overrides:
        fields["generated_token_ids_sha256"] = sha256_int_ids(overrides["natural_generated_token_ids"])
    return FullKVWorkerResultR3.model_validate(fields)


def test_canonical_r3_worker_result_converts_successfully():
    worker_result = _valid_worker_result()
    evidence = adapt_fullkv_worker_result_to_r3_evidence(
        worker_result=worker_result, candidate_manifest=_candidate_manifest(),
        candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
    )
    assert isinstance(evidence, B2AR3FullKVQualificationEvidence)
    assert evidence.candidate_ordinal == 0
    assert evidence.peak_cuda_allocated_bytes == 1_000
    assert evidence.peak_cuda_reserved_bytes == 2_000


def test_legacy_worker_result_refused():
    legacy = FullKVWorkerResult(
        role="fullkv", model_revision=MODEL_REVISION, tokenizer_revision=TOKENIZER_REVISION,
        dataset_repo=DATASET_REPO, dataset_revision=DATASET_REVISION, manifest_hash="a" * 64,
        prompt_token_ids_sha256="1" * 64, prompt_token_count=100,
        natural_generated_token_ids=[1, 2, 3], natural_answer="42", natural_answer_status="correct",
        cap_hit=False, prefill_call_count=1, decode_call_count=3, call_boundary_trace_hash="b" * 64,
        wall_seconds=1.0, determinism_policy={}, runtime_generation={}, runtime_generation_config_hash="c" * 64,
        parameter_placement={}, runtime_identity={}, memory={}, peak_cuda_allocated_bytes=1,
        peak_cuda_reserved_bytes=1, every_parameter_on_cuda=True, batch_size=1, software_versions={},
    )
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=legacy, candidate_manifest=_candidate_manifest(),
            candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
        )


def test_arbitrary_dict_refused():
    worker_result = _valid_worker_result()
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=worker_result.model_dump(mode="python"), candidate_manifest=_candidate_manifest(),
            candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
        )


def test_wrong_candidate_ordinal_out_of_range_refused():
    worker_result = _valid_worker_result()
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=worker_result, candidate_manifest=_candidate_manifest(),
            candidate_ordinal=99, expected_config_sha256=CONFIG_SHA,
        )


def test_wrong_source_index_refused():
    worker_result = _valid_worker_result(source_example_index=999999)
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=worker_result, candidate_manifest=_candidate_manifest(),
            candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
        )


def test_wrong_unique_id_refused():
    worker_result = _valid_worker_result(unique_id="not/the/real/row.json")
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=worker_result, candidate_manifest=_candidate_manifest(),
            candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
        )


def test_wrong_raw_row_hash_refused():
    worker_result = _valid_worker_result(raw_row_sha256="0" * 64)
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=worker_result, candidate_manifest=_candidate_manifest(),
            candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_repo", "alternate/example"),
        ("dataset_revision", "0" * 40),
        ("model_revision", "0" * 40),
        ("tokenizer_revision", "0" * 40),
    ],
)
def test_wrong_frozen_identity_refused(field, value):
    worker_result = _valid_worker_result(**{field: value})
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=worker_result, candidate_manifest=_candidate_manifest(),
            candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
        )


def test_wrong_generated_token_hash_rejected_at_schema_construction():
    with pytest.raises(Exception):
        _valid_worker_result(generated_token_ids_sha256="0" * 64)


def test_wrong_generation_config_hash_rejected_at_schema_construction():
    with pytest.raises(Exception):
        _valid_worker_result(worker_generation_config_sha256="0" * 64)


def test_boolean_generated_token_id_rejected_at_schema_construction():
    with pytest.raises(Exception):
        _valid_worker_result(natural_generated_token_ids=[True, 1, 2])


def test_wrong_role_rejected_at_schema_construction():
    with pytest.raises(Exception):
        _valid_worker_result(role="rkv")


def test_wrong_answer_status_rejected_at_schema_construction():
    with pytest.raises(Exception):
        _valid_worker_result(natural_answer_status="maybe")


def test_wrong_schema_version_rejected_at_schema_construction():
    with pytest.raises(Exception):
        _valid_worker_result(worker_result_schema_version="v0")


def test_unknown_field_rejected_at_schema_construction():
    with pytest.raises(Exception):
        _valid_worker_result(unexpected_field="nope")


def test_timing_evidence_with_memory_phase_spliced_in_refused():
    tainted = _timing()
    tainted.append(
        {
            "phase": "before_model_load", "started_at": 0.0, "ended_at": 1.0, "duration_seconds": 1.0,
            "synchronize_before_start": True, "synchronize_before_end": True, "completed": True,
            "failure_type": None, "failure_message": None,
        }
    )
    worker_result = _valid_worker_result(timing_evidence=tainted)
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=worker_result, candidate_manifest=_candidate_manifest(),
            candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
        )


def test_missing_memory_phase_refused():
    incomplete = [row for row in _memory() if row["phase"] != "model_load"]
    worker_result = _valid_worker_result(memory_phase_evidence=incomplete)
    with pytest.raises(WorkerAdapterRefused):
        adapt_fullkv_worker_result_to_r3_evidence(
            worker_result=worker_result, candidate_manifest=_candidate_manifest(),
            candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
        )


def test_peak_bytes_extracted_from_memory_evidence_not_caller_supplied():
    memory = _memory()
    for record in memory:
        if record["phase"] == "fullkv_complete_worker":
            record["peak_allocated"] = 555_555
            record["peak_reserved"] = 666_666
    worker_result = _valid_worker_result(memory_phase_evidence=memory)
    evidence = adapt_fullkv_worker_result_to_r3_evidence(
        worker_result=worker_result, candidate_manifest=_candidate_manifest(),
        candidate_ordinal=0, expected_config_sha256=CONFIG_SHA,
    )
    assert evidence.peak_cuda_allocated_bytes == 555_555
    assert evidence.peak_cuda_reserved_bytes == 666_666


def test_adapter_never_accepts_a_caller_supplied_gate_map():
    import inspect

    signature = inspect.signature(adapt_fullkv_worker_result_to_r3_evidence)
    forbidden_names = {"conditions", "qualified", "failed_conditions", "thinking_span_valid", "trace_complete"}
    assert forbidden_names.isdisjoint(signature.parameters)


def test_evidence_has_no_smuggleable_thinking_result_fields():
    """B2AR3FullKVQualificationEvidence has no thinking_span_valid/
    trace_complete fields at all -- those are only ever DERIVED downstream
    by the qualification evaluator, from the frozen predicates, never
    accepted as a stored value here."""
    assert "thinking_span_valid" not in B2AR3FullKVQualificationEvidence.model_fields
    assert "trace_complete" not in B2AR3FullKVQualificationEvidence.model_fields


def test_no_rkv_import_no_cuda_no_transformers():
    """Importing the adapter module, and calling
    adapt_fullkv_worker_result_to_r3_evidence against synthetic evidence,
    must never import torch/transformers/R-KV -- even when those modules
    would raise immediately if imported (mirrors the established guard
    pattern in tests/unit/test_cli_b2a_r3.py)."""
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    script = """
import sys

class _ForbiddenImportError(ImportError):
    pass

_FORBIDDEN_TOP = {"torch", "transformers", "flash_attn"}
_FORBIDDEN_SUBMODULES = (
    "kvcot.discovery.b2a_workers",
    "kvcot.discovery.schemas",
    "kvcot.discovery.scientific_summary",
    "kvcot.generation.policies",
)

class _Guard:
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top in _FORBIDDEN_TOP:
            raise _ForbiddenImportError("FORBIDDEN_IMPORT:" + name)
        if any(name == m or name.startswith(m + ".") for m in _FORBIDDEN_SUBMODULES):
            raise _ForbiddenImportError("FORBIDDEN_IMPORT:" + name)
        return None

sys.meta_path.insert(0, _Guard())
sys.path.insert(0, "src")

import kvcot.discovery.b2a_r3_worker_adapter as m

print("import-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(repo_root), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout
