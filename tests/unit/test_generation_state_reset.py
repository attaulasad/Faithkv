"""Part V.11: reset_patched_state must be architecture-generic, verified
here with both Qwen2-shaped and Llama-shaped fake model objects (identical
attribute names, per third_party/R-KV/HuggingFace/rkv/modeling.py's three
*Attention_init functions) -- no real model or GPU required.
"""
from kvcot.generation.state import reset_patched_state


class _FakeConfig:
    def __init__(self):
        self.compression = "stale"


class _FakeKVCluster:
    def __init__(self, record_kept_token_indices=True):
        self.record_kept_token_indices = record_kept_token_indices
        self.evicted_token_num = 7
        self.kept_token_indices = ["stale"]
        self.kept_attention_scores = ["stale"]
        self.kept_similarity_scores = ["stale"]
        self.kept_final_scores = ["stale"]


class _FakeSelfAttn:
    def __init__(self, record_kept_token_indices=True):
        self.config = _FakeConfig()
        self.kv_cluster = _FakeKVCluster(record_kept_token_indices)


class _FakeLayer:
    def __init__(self, record_kept_token_indices=True):
        self.self_attn = _FakeSelfAttn(record_kept_token_indices)


class _FakeInnerModel:
    def __init__(self, n_layers, record_kept_token_indices=True):
        self.layers = [_FakeLayer(record_kept_token_indices) for _ in range(n_layers)]


class _FakeCausalLM:
    """Duck-typed exactly like a real Qwen2ForCausalLM OR LlamaForCausalLM
    patched by rkv.monkeypatch -- the attribute names are identical across
    both architectures (kvcot.discovery.dispatch's two verified targets)."""

    def __init__(self, n_layers=3, with_length=True, with_after_think=True, record_kept_token_indices=True):
        self.model = _FakeInnerModel(n_layers, record_kept_token_indices)
        if with_length:
            self.length = 12345
        if with_after_think:
            self.after_think = True


def _assert_layers_reset(model, expect_kv_cluster_reset: bool):
    for layer in model.model.layers:
        assert layer.self_attn.config.compression is None
        if expect_kv_cluster_reset:
            kv = layer.self_attn.kv_cluster
            assert kv.evicted_token_num == 0
            assert kv.kept_token_indices == []
            assert kv.kept_attention_scores == []
            assert kv.kept_similarity_scores == []
            assert kv.kept_final_scores == []


def test_reset_generalizes_to_qwen2_shaped_model():
    model = _FakeCausalLM(n_layers=4)
    fresh_cache = reset_patched_state(model, fresh_cache_factory=lambda: object())
    assert not hasattr(model, "length")
    assert not hasattr(model, "after_think")
    _assert_layers_reset(model, expect_kv_cluster_reset=True)
    assert fresh_cache is not None


def test_reset_generalizes_to_llama_shaped_model():
    # Llama's attribute shape (self.model.layers[i].self_attn.config /
    # .kv_cluster, model.length, model.after_think) is identical to
    # Qwen2's under the pinned R-KV patchers -- same fake class, different
    # instantiation, proving the reset logic makes no Qwen2-specific
    # assumption.
    model = _FakeCausalLM(n_layers=6, with_length=True, with_after_think=False)
    reset_patched_state(model, fresh_cache_factory=lambda: object())
    assert not hasattr(model, "length")
    assert not hasattr(model, "after_think")
    _assert_layers_reset(model, expect_kv_cluster_reset=True)


def test_reset_is_a_no_op_safe_when_length_and_after_think_absent():
    model = _FakeCausalLM(n_layers=2, with_length=False, with_after_think=False)
    reset_patched_state(model, fresh_cache_factory=lambda: object())
    assert not hasattr(model, "length")
    assert not hasattr(model, "after_think")


def test_reset_skips_kv_cluster_bookkeeping_when_not_recording():
    model = _FakeCausalLM(n_layers=2, record_kept_token_indices=False)
    reset_patched_state(model, fresh_cache_factory=lambda: object())
    for layer in model.model.layers:
        assert layer.self_attn.config.compression is None
        # Bookkeeping lists are untouched (still "stale") when the kv
        # cluster was never configured to record them -- this repository's
        # frozen setting always enables recording, but the reset must not
        # assume that and crash/misbehave if it were ever False.
        assert layer.self_attn.kv_cluster.kept_token_indices == ["stale"]


def test_reset_returns_a_fresh_cache_via_the_factory():
    model = _FakeCausalLM(n_layers=1)
    sentinel = object()
    result = reset_patched_state(model, fresh_cache_factory=lambda: sentinel)
    assert result is sentinel
