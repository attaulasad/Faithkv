"""Policy-level mock tests for Blocker 1 (unconditional no-offload
assertion, `docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`).

Proves that `FullKVPolicy.load` and `_PatchedPolicyBase.load` (via
`RKVPolicy`) both call `assert_no_offloaded_parameters` unconditionally --
never gated behind `model.device.type == "cuda"` -- using fully mocked
`transformers`/`rkv.monkeypatch` call sites, exactly like
`tests/unit/test_policies_architecture_dispatch.py`. No network request or
real model loading occurs in any test in this file.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch
import transformers

from kvcot.discovery.no_offload import ModelOffloadError
from kvcot.generation import state as kvcot_state
from kvcot.generation.policies import FullKVPolicy, RKVPolicy


@pytest.fixture(autouse=True)
def _reset_process_mode():
    kvcot_state.reset_active_mode_for_testing()
    yield
    kvcot_state.reset_active_mode_for_testing()


class _FakeParam:
    def __init__(self, device_type: str):
        self.device = types.SimpleNamespace(type=device_type)


class _FakeConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self._updates = []

    def update(self, d):
        self._updates.append(dict(d))


class _FakeModel:
    """`model.device` is DELIBERATELY always "cpu" here, regardless of what
    the real per-parameter map says -- the whole point of these tests is
    that the assertion must never consult this misleading, unreliable
    property (Blocker 1's required correction)."""

    def __init__(self, param_devices: dict[str, str], hf_device_map: dict | None = None):
        self.eval_called = False
        self.config = _FakeConfig("irrelevant-post-construction")
        self.device = types.SimpleNamespace(type="cpu")
        self._fake_params = {name: _FakeParam(dt) for name, dt in param_devices.items()}
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map

    def eval(self):
        self.eval_called = True
        return self

    def named_parameters(self):
        return iter(self._fake_params.items())


@pytest.fixture
def fake_rkv_monkeypatch(monkeypatch):
    calls = []
    fake_module = types.ModuleType("rkv")
    fake_monkeypatch_module = types.ModuleType("rkv.monkeypatch")

    def _make(name):
        def _fn(compression_config):
            calls.append((name, compression_config))

        return _fn

    fake_monkeypatch_module.replace_qwen2 = _make("replace_qwen2")
    fake_monkeypatch_module.replace_llama = _make("replace_llama")
    fake_monkeypatch_module.replace_qwen3 = _make("replace_qwen3")
    fake_module.monkeypatch = fake_monkeypatch_module

    monkeypatch.setitem(sys.modules, "rkv", fake_module)
    monkeypatch.setitem(sys.modules, "rkv.monkeypatch", fake_monkeypatch_module)
    return calls


def _mock_transformers(monkeypatch, model_factory, model_type="qwen2"):
    calls = {"config": [], "model": [], "tokenizer": []}

    def fake_config_from_pretrained(model_name, revision=None, **kwargs):
        calls["config"].append({"model_name": model_name, "revision": revision, **kwargs})
        return _FakeConfig(model_type)

    def fake_model_from_pretrained(model_name, **kwargs):
        calls["model"].append({"model_name": model_name, **kwargs})
        return model_factory()

    class _FakeTokenizer:
        def encode(self, text):
            return [1, 2, 3]

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        calls["tokenizer"].append({"model_name": model_name, **kwargs})
        return _FakeTokenizer()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(fake_config_from_pretrained))
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", staticmethod(fake_model_from_pretrained)
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tokenizer_from_pretrained))
    return calls


# --- 1. FullKV calls the assertion ---------------------------------------


def test_fullkv_load_calls_assertion_and_passes_on_all_cuda(monkeypatch):
    calls = _mock_transformers(monkeypatch, lambda: _FakeModel({"w": "cuda"}))
    model = FullKVPolicy().load("m", "rev", torch.bfloat16, "flash_attention_2")
    assert model.eval_called is True
    assert len(calls["model"]) == 1  # exactly one construction, no retry/second load


def test_fullkv_load_raises_when_a_parameter_is_offloaded(monkeypatch):
    _mock_transformers(monkeypatch, lambda: _FakeModel({"w0": "cuda", "w1": "cpu"}))
    with pytest.raises(ModelOffloadError):
        FullKVPolicy().load("m", "rev", torch.bfloat16, "flash_attention_2")


# --- 2. R-KV calls the assertion ------------------------------------------


def test_rkv_load_calls_assertion_and_passes_on_all_cuda(fake_rkv_monkeypatch, monkeypatch):
    _mock_transformers(monkeypatch, lambda: _FakeModel({"w": "cuda"}))
    model = RKVPolicy(budget=128).load("m", "rev", torch.bfloat16, "flash_attention_2")
    assert model.eval_called is True


def test_rkv_load_raises_when_a_parameter_is_offloaded(fake_rkv_monkeypatch, monkeypatch):
    _mock_transformers(monkeypatch, lambda: _FakeModel({"w0": "cuda", "w1": "cpu"}))
    with pytest.raises(ModelOffloadError):
        RKVPolicy(budget=128).load("m", "rev", torch.bfloat16, "flash_attention_2")


# --- 3/6. model.device reports cpu while parameters include cuda/cpu: fails, never bypassed via model.device ---


def test_fullkv_fails_despite_model_device_reporting_cpu_while_params_mixed(monkeypatch):
    # _FakeModel.device is always "cpu" (see class docstring); real params
    # are mixed cuda/cpu here. If the assertion (wrongly) consulted
    # model.device first it would never even attempt the per-parameter walk
    # (the old `if model.device.type == "cuda":` guard would have made this
    # load SUCCEED). It must fail.
    _mock_transformers(monkeypatch, lambda: _FakeModel({"w0": "cuda", "w1": "cpu"}))
    with pytest.raises(ModelOffloadError):
        FullKVPolicy().load("m", "rev", torch.bfloat16, "flash_attention_2")


# --- 4. first parameter cuda, a later parameter cpu: fails ---------------


def test_rkv_fails_when_first_param_cuda_but_later_param_cpu(fake_rkv_monkeypatch, monkeypatch):
    _mock_transformers(
        monkeypatch, lambda: _FakeModel({"layer0.w": "cuda", "layer1.w": "cuda", "layer2.w": "cpu"})
    )
    with pytest.raises(ModelOffloadError) as excinfo:
        RKVPolicy(budget=128).load("m", "rev", torch.bfloat16, "flash_attention_2")
    assert "layer2.w" in str(excinfo.value)


# --- 5. hf_device_map containing disk: fails -------------------------------


def test_fullkv_fails_when_hf_device_map_contains_disk(monkeypatch):
    _mock_transformers(
        monkeypatch,
        lambda: _FakeModel({"w0": "cuda", "w1": "cuda"}, hf_device_map={"w0": "cuda:0", "w1": "disk"}),
    )
    with pytest.raises(ModelOffloadError) as excinfo:
        FullKVPolicy().load("m", "rev", torch.bfloat16, "flash_attention_2")
    assert "disk" in str(excinfo.value)


def test_rkv_fails_when_hf_device_map_contains_disk(fake_rkv_monkeypatch, monkeypatch):
    _mock_transformers(
        monkeypatch,
        lambda: _FakeModel({"w0": "cuda"}, hf_device_map={"w0": "disk"}),
        model_type="llama",
    )
    with pytest.raises(ModelOffloadError):
        RKVPolicy(budget=1024).load(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "rev", torch.bfloat16, "flash_attention_2"
        )


# --- 7. no network request or real model loading occurs during tests ------


def test_no_real_network_request_or_real_model_loading(fake_rkv_monkeypatch, monkeypatch):
    calls = _mock_transformers(monkeypatch, lambda: _FakeModel({"w": "cuda"}), model_type="llama")
    RKVPolicy(budget=1024).load(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "rev", torch.bfloat16, "flash_attention_2"
    )
    # Every entry point that could touch the network is the mocked
    # staticmethod above; asserting the call counts confirms the mocked
    # path (not any real transformers/huggingface_hub code) is what ran.
    assert len(calls["config"]) == 1
    assert len(calls["model"]) == 1
    assert len(calls["tokenizer"]) == 1
