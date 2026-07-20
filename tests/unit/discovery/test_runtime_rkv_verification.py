"""CPU tests for `kvcot.discovery.runtime_rkv_verification` (B1B-R3 §4/§18).
Builds a minimal fake "loaded model" object (per-layer `self_attn.kv_cluster`
+ `self_attn.config.update_kv` + `model.config.{divide_method,divide_length,
compression_content}`) -- never imports torch/transformers/rkv, matching this
repository's existing discipline for CPU-only discovery tests."""
from __future__ import annotations

import types

import pytest

from kvcot.discovery.discovery_config import DiscoveryRkvLock
from kvcot.discovery.runtime_rkv_verification import (
    RuntimeRkvConfigError,
    frozen_rkv_config_hash,
    frozen_runtime_comparable_fields,
    read_runtime_rkv_config,
    runtime_rkv_config_hash,
    verify_runtime_matches_frozen,
)

PINNED_UPSTREAM_REVISION = "45eaa7d69d20b7388321f077020a610d9afb65bd"


def _frozen_rkv(**overrides) -> DiscoveryRkvLock:
    fields = dict(budget=1024, upstream_revision=PINNED_UPSTREAM_REVISION, kernel_size=3)
    fields.update(overrides)
    return DiscoveryRkvLock(**fields)


class _KvCluster:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _AttnConfig:
    def __init__(self, update_kv=True):
        self.update_kv = update_kv


class _SelfAttn:
    def __init__(self, kv_cluster, update_kv=True):
        self.kv_cluster = kv_cluster
        self.config = _AttnConfig(update_kv=update_kv)


class _Layer:
    def __init__(self, kv_cluster, update_kv=True):
        self.self_attn = _SelfAttn(kv_cluster, update_kv=update_kv)


class _ModelConfig:
    def __init__(self, divide_method="step_length", divide_length=128, compression_content="all"):
        self.divide_method = divide_method
        self.divide_length = divide_length
        self.compression_content = compression_content


class _ModelModel:
    def __init__(self, layers):
        self.layers = layers


class _FakeModel:
    def __init__(self, num_layers=3, config=None, **kv_kwargs):
        config = config or {}
        self.config = _ModelConfig(**config)
        layer_kv_kwargs = dict(
            budget=1024, window_size=8, kernel_size=3, mix_lambda=0.1, retain_ratio=0.2,
            retain_direction="last", record_kept_token_indices=True,
        )
        layer_kv_kwargs.update(kv_kwargs)
        self.model = _ModelModel([_Layer(_KvCluster(**layer_kv_kwargs)) for _ in range(num_layers)])


def test_matching_runtime_and_frozen_config_passes():
    frozen = _frozen_rkv()
    model = _FakeModel()
    result = verify_runtime_matches_frozen(frozen, model)
    assert result.passed
    assert result.mismatched_fields == ()
    assert result.frozen_hash == result.runtime_hash


def test_kernel_size_mismatch_is_reported_and_hashes_differ():
    frozen = _frozen_rkv(kernel_size=3)
    model = _FakeModel(kernel_size=7)  # R1KV's own upstream default, not the frozen value
    result = verify_runtime_matches_frozen(frozen, model)
    assert not result.passed
    assert result.mismatched_fields == ("kernel_size",)
    assert result.frozen_hash != result.runtime_hash


def test_budget_mismatch_is_reported():
    frozen = _frozen_rkv(budget=1024)
    model = _FakeModel(budget=999)
    result = verify_runtime_matches_frozen(frozen, model)
    assert not result.passed
    assert "budget" in result.mismatched_fields


def test_missing_kv_cluster_raises_hard_error():
    model = _FakeModel()
    model.model.layers[1].self_attn.kv_cluster = None
    with pytest.raises(RuntimeRkvConfigError, match="no self_attn.kv_cluster"):
        read_runtime_rkv_config(model)


def test_empty_layers_raises_hard_error():
    model = types.SimpleNamespace(model=_ModelModel([]), config=_ModelConfig())
    with pytest.raises(RuntimeRkvConfigError, match="empty"):
        read_runtime_rkv_config(model)


def test_cross_layer_disagreement_raises_hard_error():
    model = _FakeModel(num_layers=3)
    model.model.layers[2].self_attn.kv_cluster.budget = 512
    with pytest.raises(RuntimeRkvConfigError, match="disagrees with layer 0"):
        read_runtime_rkv_config(model)


def test_missing_update_kv_attribute_raises_hard_error():
    model = _FakeModel()
    del model.model.layers[0].self_attn.config.update_kv
    with pytest.raises(RuntimeRkvConfigError, match="update_kv"):
        read_runtime_rkv_config(model)


def test_missing_model_config_field_raises_hard_error():
    model = _FakeModel()
    del model.config.divide_method
    with pytest.raises(RuntimeRkvConfigError, match="divide_method"):
        read_runtime_rkv_config(model)


def test_frozen_fields_include_update_kv_and_record_kept_token_indices():
    frozen = _frozen_rkv()
    fields = frozen_runtime_comparable_fields(frozen)
    assert fields["update_kv"] is True
    assert fields["record_kept_token_indices"] is True
    assert fields["kernel_size"] == 3


def test_frozen_hash_stable_and_sensitive_to_kernel_size():
    a = frozen_rkv_config_hash(_frozen_rkv(kernel_size=3))
    b = frozen_rkv_config_hash(_frozen_rkv(kernel_size=3))
    c = frozen_rkv_config_hash(_frozen_rkv(kernel_size=5))
    assert a == b
    assert a != c


def test_runtime_hash_deterministic_for_identical_model_state():
    model_a = _FakeModel()
    model_b = _FakeModel()
    assert runtime_rkv_config_hash(model_a) == runtime_rkv_config_hash(model_b)
