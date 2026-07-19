import sys
import types

import pytest

from kvcot.discovery.dispatch import (
    MODEL_TYPE_TO_PATCHER_NAME,
    UnsupportedArchitectureError,
    resolve_patcher,
    resolve_patcher_name,
)


def test_resolve_patcher_name_known_architectures():
    assert resolve_patcher_name("qwen2") == "replace_qwen2"
    assert resolve_patcher_name("llama") == "replace_llama"
    assert resolve_patcher_name("qwen3") == "replace_qwen3"


def test_resolve_patcher_name_unknown_architecture_raises_before_any_import():
    with pytest.raises(UnsupportedArchitectureError):
        resolve_patcher_name("gpt2")
    with pytest.raises(UnsupportedArchitectureError):
        resolve_patcher_name("mistral")


def test_no_default_fallback_patcher_exists():
    # There must be no wildcard/default entry in the dispatch table.
    assert "*" not in MODEL_TYPE_TO_PATCHER_NAME
    assert "default" not in MODEL_TYPE_TO_PATCHER_NAME


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


def test_resolve_patcher_invokes_only_the_matching_patcher_llama(fake_rkv_monkeypatch):
    cfg = {"method": "rkv"}
    resolve_patcher("llama", cfg)
    assert fake_rkv_monkeypatch == [("replace_llama", cfg)]


def test_resolve_patcher_invokes_only_the_matching_patcher_qwen2(fake_rkv_monkeypatch):
    cfg = {"method": "rkv"}
    resolve_patcher("qwen2", cfg)
    assert fake_rkv_monkeypatch == [("replace_qwen2", cfg)]


def test_resolve_patcher_unsupported_architecture_never_touches_rkv_module(fake_rkv_monkeypatch):
    with pytest.raises(UnsupportedArchitectureError):
        resolve_patcher("falcon", {"method": "rkv"})
    assert fake_rkv_monkeypatch == []


def test_resolve_patcher_missing_function_on_module_raises_no_fallback(monkeypatch):
    fake_module = types.ModuleType("rkv")
    fake_monkeypatch_module = types.ModuleType("rkv.monkeypatch")
    # Deliberately missing replace_llama, simulating a submodule drift.
    fake_monkeypatch_module.replace_qwen2 = lambda cfg: None
    fake_module.monkeypatch = fake_monkeypatch_module
    monkeypatch.setitem(sys.modules, "rkv", fake_module)
    monkeypatch.setitem(sys.modules, "rkv.monkeypatch", fake_monkeypatch_module)

    with pytest.raises(UnsupportedArchitectureError):
        resolve_patcher("llama", {"method": "rkv"})
