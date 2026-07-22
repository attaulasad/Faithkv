"""B1B-R4 §9/§10/§11 tests for `kvcot.discovery.runtime_evidence` -- every
function is duck-typed against plain fake objects, no torch/transformers
import required.
"""
from __future__ import annotations

from types import SimpleNamespace

from kvcot.discovery.runtime_evidence import (
    build_runtime_generation_record,
    derive_batch_size_from_input_ids,
    derive_parameter_placement,
    derive_runtime_identity,
    read_resolved_model_revision,
    read_resolved_tokenizer_revision,
)


class _FakeDevice:
    def __init__(self, type_: str):
        self.type = type_


class _FakeParam:
    def __init__(self, device_type: str):
        self.device = _FakeDevice(device_type)


class _FakeModel:
    def __init__(self, device_types, hf_device_map=None, commit_hash=None):
        self._params = [(f"p{i}", _FakeParam(t)) for i, t in enumerate(device_types)]
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map
        self.config = SimpleNamespace(_commit_hash=commit_hash)

    def named_parameters(self):
        return iter(self._params)


class _FakeInputIds:
    def __init__(self, shape):
        self.shape = shape


# --------------------------------------------------------------------------
# RuntimeGenerationRecord
# --------------------------------------------------------------------------


def test_runtime_generation_record_is_always_greedy_never_sampling():
    record = build_runtime_generation_record(
        batch_size=1, max_new_tokens=48, eos_token_id=99, attention_backend="flash_attention_2",
        framework_seed=13, prompt_token_count=200,
    )
    assert record.generation_mode == "greedy"
    assert record.do_sample is False
    assert record.temperature is None
    assert record.top_p is None


def test_runtime_generation_record_hash_changes_with_batch_size():
    a = build_runtime_generation_record(
        batch_size=1, max_new_tokens=48, eos_token_id=99, attention_backend="flash_attention_2",
        framework_seed=13, prompt_token_count=200,
    )
    b = build_runtime_generation_record(
        batch_size=2, max_new_tokens=48, eos_token_id=99, attention_backend="flash_attention_2",
        framework_seed=13, prompt_token_count=200,
    )
    assert a.canonical_hash() != b.canonical_hash()


def test_identical_inputs_produce_identical_hash():
    a = build_runtime_generation_record(
        batch_size=1, max_new_tokens=48, eos_token_id=99, attention_backend="flash_attention_2",
        framework_seed=13, prompt_token_count=200,
    )
    b = build_runtime_generation_record(
        batch_size=1, max_new_tokens=48, eos_token_id=99, attention_backend="flash_attention_2",
        framework_seed=13, prompt_token_count=200,
    )
    assert a.canonical_hash() == b.canonical_hash()


# --------------------------------------------------------------------------
# ParameterPlacementEvidence
# --------------------------------------------------------------------------


def test_every_parameter_on_cuda_true_when_all_params_cuda():
    model = _FakeModel(["cuda", "cuda", "cuda"])
    evidence = derive_parameter_placement(model)
    assert evidence.every_parameter_on_cuda is True
    assert evidence.no_offload_verified is True
    assert evidence.unique_device_types == ("cuda",)
    assert evidence.parameter_count == 3


def test_every_parameter_on_cuda_false_when_one_param_is_cpu():
    model = _FakeModel(["cuda", "cpu", "cuda"])
    evidence = derive_parameter_placement(model)
    assert evidence.every_parameter_on_cuda is False
    assert evidence.no_offload_verified is False
    assert evidence.unique_device_types == ("cpu", "cuda")


def test_zero_parameters_is_not_a_vacuous_pass():
    model = _FakeModel([])
    evidence = derive_parameter_placement(model)
    assert evidence.every_parameter_on_cuda is False
    assert evidence.parameter_count == 0


def test_device_map_offload_entry_fails_no_offload_even_if_params_report_cuda():
    model = _FakeModel(["cuda", "cuda"], hf_device_map={"layer.0": "cuda:0", "layer.1": "cpu"})
    evidence = derive_parameter_placement(model)
    assert evidence.every_parameter_on_cuda is True  # named_parameters() itself all say cuda
    assert evidence.no_offload_verified is False  # but the device map reveals a cpu-assigned entry
    assert evidence.hf_device_map == {"layer.0": "cuda:0", "layer.1": "cpu"}


# --------------------------------------------------------------------------
# batch size
# --------------------------------------------------------------------------


def test_batch_size_derived_from_actual_tensor_shape():
    assert derive_batch_size_from_input_ids(_FakeInputIds((1, 50))) == 1
    assert derive_batch_size_from_input_ids(_FakeInputIds((4, 50))) == 4


# --------------------------------------------------------------------------
# resolved revision readback + runtime identity
# --------------------------------------------------------------------------


def test_resolved_model_revision_read_from_config_commit_hash():
    model = _FakeModel(["cuda"], commit_hash="abc123")
    assert read_resolved_model_revision(model) == "abc123"


def test_resolved_model_revision_none_when_attribute_absent():
    model = SimpleNamespace(config=SimpleNamespace())
    assert read_resolved_model_revision(model) is None


def test_resolved_tokenizer_revision_read_from_commit_hash_attribute():
    tokenizer = SimpleNamespace(_commit_hash="deadbeef")
    assert read_resolved_tokenizer_revision(tokenizer) == "deadbeef"


def test_runtime_identity_matches_when_resolved_equals_requested():
    model = _FakeModel(["cuda"], commit_hash="rev-a")
    tokenizer = SimpleNamespace(_commit_hash="rev-a")
    identity = derive_runtime_identity(
        model=model, tokenizer=tokenizer, requested_model_revision="rev-a", requested_tokenizer_revision="rev-a"
    )
    assert identity.model_revision_match is True
    assert identity.tokenizer_revision_match is True


def test_runtime_identity_fails_closed_when_resolved_is_none_never_true():
    model = _FakeModel(["cuda"], commit_hash=None)
    tokenizer = SimpleNamespace(_commit_hash=None)
    identity = derive_runtime_identity(
        model=model, tokenizer=tokenizer, requested_model_revision="rev-a", requested_tokenizer_revision="rev-a"
    )
    assert identity.resolved_model_revision is None
    assert identity.model_revision_match is False  # never silently True just because it's unresolved
    assert identity.tokenizer_revision_match is False


def test_runtime_identity_fails_when_resolved_disagrees_with_requested():
    model = _FakeModel(["cuda"], commit_hash="rev-b")
    tokenizer = SimpleNamespace(_commit_hash="rev-a")
    identity = derive_runtime_identity(
        model=model, tokenizer=tokenizer, requested_model_revision="rev-a", requested_tokenizer_revision="rev-a"
    )
    assert identity.model_revision_match is False
