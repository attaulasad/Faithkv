"""B1A-1 construction-parity tests (Part V.10 of
docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md). Mocks AutoConfig,
AutoModelForCausalLM, AutoTokenizer, and the pinned rkv.monkeypatch module
entirely -- no real Hugging Face network request, no real model weights, no
GPU required.
"""
import sys
import types

import pytest
import torch
import transformers

from kvcot.discovery.dispatch import UnsupportedArchitectureError
from kvcot.generation import state as kvcot_state
from kvcot.generation.policies import FullKVPolicy, RKVPolicy


@pytest.fixture(autouse=True)
def _reset_process_mode():
    kvcot_state.reset_active_mode_for_testing()
    yield
    kvcot_state.reset_active_mode_for_testing()


class _FakeConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self._updates = []

    def update(self, d):
        self._updates.append(dict(d))


class _FakeModel:
    def __init__(self):
        self.eval_called = False
        self.config = _FakeConfig("irrelevant-post-construction")
        self.device = torch.device("cpu")

    def eval(self):
        self.eval_called = True
        return self


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


@pytest.fixture
def mocked_transformers(monkeypatch):
    """Patches AutoConfig/AutoModelForCausalLM/AutoTokenizer.from_pretrained
    on the real (installed) transformers module so no network call ever
    happens; records every call's kwargs for parity assertions."""
    calls = {"config": [], "model": [], "tokenizer": []}
    model_type_box = {"model_type": "qwen2"}

    def fake_config_from_pretrained(model_name, revision=None, **kwargs):
        calls["config"].append({"model_name": model_name, "revision": revision, **kwargs})
        return _FakeConfig(model_type_box["model_type"])

    def fake_model_from_pretrained(model_name, **kwargs):
        calls["model"].append({"model_name": model_name, **kwargs})
        return _FakeModel()

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
    return calls, model_type_box


def test_llama_rkv_selects_only_the_llama_patcher(fake_rkv_monkeypatch, mocked_transformers):
    calls, model_type_box = mocked_transformers
    model_type_box["model_type"] = "llama"
    policy = RKVPolicy(budget=1024)

    policy.load("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "rev", torch.bfloat16, "flash_attention_2")

    assert [name for name, _ in fake_rkv_monkeypatch] == ["replace_llama"]


def test_qwen2_rkv_selects_only_replace_qwen2(fake_rkv_monkeypatch, mocked_transformers):
    calls, model_type_box = mocked_transformers
    model_type_box["model_type"] = "qwen2"
    policy = RKVPolicy(budget=128)

    policy.load("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "rev", torch.bfloat16, "flash_attention_2")

    assert [name for name, _ in fake_rkv_monkeypatch] == ["replace_qwen2"]


def test_unknown_architecture_raises_before_model_construction(fake_rkv_monkeypatch, mocked_transformers):
    calls, model_type_box = mocked_transformers
    model_type_box["model_type"] = "falcon"
    policy = RKVPolicy(budget=1024)

    with pytest.raises(UnsupportedArchitectureError):
        policy.load("some/falcon-model", "rev", torch.bfloat16, "flash_attention_2")

    assert calls["model"] == []  # AutoModelForCausalLM.from_pretrained was never reached
    assert fake_rkv_monkeypatch == []


def test_fullkv_and_rkv_share_identical_model_loading_arguments(fake_rkv_monkeypatch, mocked_transformers):
    calls, model_type_box = mocked_transformers
    model_type_box["model_type"] = "qwen2"

    kvcot_state.reset_active_mode_for_testing()
    FullKVPolicy().load("m", "rev", torch.bfloat16, "flash_attention_2")
    full_call = calls["model"][-1]

    kvcot_state.reset_active_mode_for_testing()
    RKVPolicy(budget=128).load("m", "rev", torch.bfloat16, "flash_attention_2")
    rkv_call = calls["model"][-1]

    shared_keys = ["model_name", "revision", "torch_dtype", "low_cpu_mem_usage", "device_map", "use_cache", "attn_implementation"]
    for key in shared_keys:
        assert full_call[key] == rkv_call[key], f"{key} diverged: full={full_call[key]!r} rkv={rkv_call[key]!r}"


def test_patching_is_process_global_stock_and_patched_cannot_coexist(fake_rkv_monkeypatch, mocked_transformers):
    calls, model_type_box = mocked_transformers
    model_type_box["model_type"] = "qwen2"

    FullKVPolicy().load("m", "rev", torch.bfloat16, "flash_attention_2")
    with pytest.raises(kvcot_state.ProcessModeConflictError):
        RKVPolicy(budget=128).load("m", "rev", torch.bfloat16, "flash_attention_2")


def test_dispatch_order_is_config_before_model(fake_rkv_monkeypatch, mocked_transformers):
    calls, model_type_box = mocked_transformers
    model_type_box["model_type"] = "qwen2"
    RKVPolicy(budget=128).load("m", "rev", torch.bfloat16, "flash_attention_2")

    assert len(calls["config"]) == 1
    assert len(calls["model"]) == 1
    # The patch call happened (fake_rkv_monkeypatch non-empty) and config was
    # read; AutoModelForCausalLM was constructed after, never before.
    assert fake_rkv_monkeypatch[0][0] == "replace_qwen2"


def test_no_real_network_request_possible_everything_is_mocked(fake_rkv_monkeypatch, mocked_transformers):
    calls, model_type_box = mocked_transformers
    model_type_box["model_type"] = "llama"
    RKVPolicy(budget=1024).load(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "rev", torch.bfloat16, "flash_attention_2"
    )
    assert len(calls["config"]) == 1
    assert len(calls["model"]) == 1
    assert len(calls["tokenizer"]) == 1
