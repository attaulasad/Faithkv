"""B1B-R4 §20: CPU tests that execute the REAL `run_fullkv_worker`/
`run_rkv_worker` control-flow bodies (`kvcot.discovery.b2a_workers`)
end-to-end, via the internal dependency-injection seams
(`_load_model`/`_load_tokenizer`/`_fresh_cache_factory`/`_cuda`/`_device`)
those two functions expose -- never a preconstructed
`FullKVWorkerResult`/`RKVWorkerResult`. No real torch CUDA, no real model
download, no real HTTP call: every backend is a small, deterministic, real-
torch-CPU-tensor fake.

Scope note (documented, not silently omitted -- see
`docs/B1B_R4_FINAL_B2A_CLOSURE.md` §"Adversarial self-review"): the fake
model used here never naturally emits EOS (its logits are deterministic
zeros, so argmax always selects token 0), so Pass 1 always hits
`max_new_tokens` (`cap_hit=True`) rather than reaching a valid, scoreable
example. This still exercises the ENTIRE worker-control body up through
Pass 1's natural-run loop, seed application, model/tokenizer loading seam,
call-boundary trace, runtime identity, memory observation, and
`FullKVWorkerResult`/`RKVWorkerResult` construction from REAL (not
preconstructed) values -- it does not exercise the full 12-real+1-no-op
success path, which requires a substantially larger fixture (a fake model
that emits a controlled EOS spike after enough compaction events) not
built in this pass.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kvcot.discovery.b2a_workers import FullKVWorkerResult, RKVWorkerResult, run_fullkv_worker, run_rkv_worker

NUM_LAYERS = 2
NUM_HEADS = 1
HEAD_DIM = 4
VOCAB_SIZE = 8
BUDGET = 1_000_000  # never reached -- no eviction noise for this scoped test
EOS_TOKEN_ID = 999  # never emitted by the deterministic all-zero-logit fake model


class _FakeCudaFacade:
    """A real, stateful (but CPU-backed) fake of the small `torch.cuda`
    surface these workers use -- `is_available()` always `True` so the
    worker's CUDA precondition check passes, every memory counter is a
    plain Python int the test can assert on."""

    def __init__(self):
        self._peak_allocated = 12345
        self._peak_reserved = 23456
        self.reset_peak_memory_stats_call_count = 0

    def is_available(self):
        return True

    def synchronize(self):
        pass

    def memory_allocated(self):
        return 1000

    def memory_reserved(self):
        return 2000

    def reset_peak_memory_stats(self):
        self.reset_peak_memory_stats_call_count += 1

    def max_memory_allocated(self):
        return self._peak_allocated

    def max_memory_reserved(self):
        return self._peak_reserved


class _FakeRkvConfig:
    def __init__(self, rkv_lock):
        self.compression = None
        self.divide_method = rkv_lock.divide_method
        self.divide_length = rkv_lock.divide_length
        self.compression_content = rkv_lock.compression_content
        self.update_kv = True


class _FakeKVCluster:
    def __init__(self, rkv_lock):
        self.budget = rkv_lock.budget
        self.window_size = rkv_lock.window_size
        self.kernel_size = rkv_lock.kernel_size
        self.mix_lambda = rkv_lock.mix_lambda
        self.retain_ratio = rkv_lock.retain_ratio
        self.retain_direction = rkv_lock.retain_direction
        self.record_kept_token_indices = True
        self.evicted_token_num = 0
        self.kept_token_indices: list = []
        self.kept_attention_scores: list = []
        self.kept_similarity_scores: list = []
        self.kept_final_scores: list = []


class _FakeAttn:
    def __init__(self, rkv_lock):
        self.config = _FakeRkvConfig(rkv_lock)
        self.kv_cluster = _FakeKVCluster(rkv_lock)


class _FakeLayer:
    def __init__(self, rkv_lock):
        self.self_attn = _FakeAttn(rkv_lock)


class _FakeCache:
    def __init__(self, num_layers: int):
        self.key_cache = [torch.empty(1, NUM_HEADS, 0, HEAD_DIM) for _ in range(num_layers)]
        self.value_cache = [torch.empty(1, NUM_HEADS, 0, HEAD_DIM) for _ in range(num_layers)]

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class _FakeModel:
    """Deterministic, never-evicting (budget effectively infinite), never-EOS
    fake CausalLM -- real torch CPU tensors, real forward-call shape
    (`input_ids`/`position_ids`/`past_key_values`/`use_cache`/
    `cache_position`), duck-typed exactly like `real_model_adapter`'s real
    adapters expect."""

    def __init__(self, rkv_lock=None, num_layers: int = NUM_LAYERS):
        if rkv_lock is not None:
            layers = [_FakeLayer(rkv_lock) for _ in range(num_layers)]
        else:
            layers = [SimpleNamespace(self_attn=SimpleNamespace(config=SimpleNamespace(compression=None))) for _ in range(num_layers)]
        self.model = SimpleNamespace(layers=layers)
        config_kwargs = dict(num_key_value_heads=NUM_HEADS)
        if rkv_lock is not None:
            config_kwargs.update(
                divide_method=rkv_lock.divide_method, divide_length=rkv_lock.divide_length,
                compression_content=rkv_lock.compression_content,
            )
        self.config = SimpleNamespace(**config_kwargs)
        self._params = [("weight", SimpleNamespace(device=SimpleNamespace(type="cuda")))]

    def named_parameters(self):
        return iter(self._params)

    def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=None, cache_position=None):
        cache = past_key_values
        seq_len = input_ids.shape[1]
        for i in range(len(self.model.layers)):
            new_k = torch.zeros(1, NUM_HEADS, seq_len, HEAD_DIM)
            new_v = new_k.clone()
            cache.key_cache[i] = torch.cat([cache.key_cache[i], new_k], dim=-2)
            cache.value_cache[i] = torch.cat([cache.value_cache[i], new_v], dim=-2)
        logits = torch.zeros(1, seq_len, VOCAB_SIZE)  # argmax always selects token 0
        return SimpleNamespace(logits=logits)


class _FakeTokenizer:
    eos_token_id = EOS_TOKEN_ID

    def decode(self, ids, skip_special_tokens=True):
        return "no boxed answer here"


def _build_fake_discovery_config():
    """A REAL `kvcot.discovery.discovery_config.DiscoveryConfig` (pydantic),
    not a hand-rolled stand-in -- `canonical_config_hash`/`NaturalRunProvenance
    .config_sha256` call `.model_dump(mode="json")` on the nested locks, so
    a plain fake class without that method would fail; using the real
    schema also means every field-level validator (full-40-hex revisions,
    the pinned R-KV upstream_revision, etc.) genuinely runs."""
    from kvcot.discovery.discovery_config import (
        PINNED_RKV_UPSTREAM_REVISION,
        DiscoveryConfig,
        DiscoveryDatasetLock,
        DiscoveryGenerationLock,
        DiscoveryModelLock,
        DiscoveryRkvLock,
    )

    return DiscoveryConfig(
        model=DiscoveryModelLock(
            name="fake/model", revision="a1" * 20, tokenizer_name="fake/tokenizer", tokenizer_revision="b2" * 20,
            model_type="qwen2", dtype="bfloat16",
        ),
        dataset=DiscoveryDatasetLock(name="MATH-500", config="default", split="test", revision="c3" * 20),
        rkv=DiscoveryRkvLock(budget=1_000_000, upstream_revision=PINNED_RKV_UPSTREAM_REVISION),
        generation=DiscoveryGenerationLock(max_new_tokens=12, framework_seed=13),
    )


class _FakeManifest:
    dataset_repo = "HuggingFaceH4/MATH-500"
    dataset_revision = "d" * 40
    example_index = 0
    unique_id = "test/1.json"
    gold_answer = "0"
    prompt_token_ids = [1, 2, 3]
    prompt_token_ids_sha256 = "p" * 64
    prompt_token_count = 3

    def manifest_hash(self):
        return "h" * 64


def test_run_fullkv_worker_executes_the_real_body_end_to_end():
    config = _build_fake_discovery_config()
    manifest = _FakeManifest()
    model = _FakeModel()
    fake_cuda = _FakeCudaFacade()

    result_dict = run_fullkv_worker(
        config, manifest,
        _load_model=lambda: model, _load_tokenizer=lambda: _FakeTokenizer(),
        _fresh_cache_factory=lambda: _FakeCache(NUM_LAYERS), _cuda=fake_cuda, _device="cpu",
    )

    result = FullKVWorkerResult.model_validate(result_dict)
    assert result.role == "fullkv"
    assert result.cap_hit is True  # deterministic fake never emits EOS
    assert result.prefill_call_count == 1
    assert result.decode_call_count == config.generation.max_new_tokens
    assert len(result.natural_generated_token_ids) == config.generation.max_new_tokens
    assert result.batch_size == 1
    assert result.every_parameter_on_cuda is True  # derived from the fake param's device.type == "cuda"
    assert result.peak_cuda_allocated_bytes == 12345
    assert result.peak_cuda_reserved_bytes == 23456
    assert result.determinism_policy["framework_seed"] == 13
    assert result.runtime_generation["generation_mode"] == "greedy"
    assert result.runtime_generation["do_sample"] is False
    # B1 execution-boundary closure §4: `reset_patched_state` (called here
    # to build the natural-generation state) must not add a second, hidden
    # peak-memory reset on top of the worker's own explicit one.
    assert fake_cuda.reset_peak_memory_stats_call_count == 1


def test_run_fullkv_worker_requires_cuda_when_no_fake_backend_injected():
    """Never silently uses a fake in production -- omitting the injection
    seams entirely (the production call shape) must still hard-require real
    CUDA."""
    from kvcot.discovery.b2a_workers import WorkerFailedError

    config = _build_fake_discovery_config()
    manifest = _FakeManifest()
    with pytest.raises(WorkerFailedError, match="requires CUDA"):
        run_fullkv_worker(config, manifest)


def test_run_rkv_worker_executes_the_real_body_through_pass1():
    config = _build_fake_discovery_config()
    manifest = _FakeManifest()
    rkv_lock = config.rkv
    model = _FakeModel(rkv_lock=rkv_lock)
    fake_cuda = _FakeCudaFacade()

    result_dict = run_rkv_worker(
        config, manifest,
        _load_model=lambda: model, _load_tokenizer=lambda: _FakeTokenizer(),
        _fresh_cache_factory=lambda: _FakeCache(NUM_LAYERS), _cuda=fake_cuda, _device="cpu",
    )

    result = RKVWorkerResult.model_validate(result_dict)
    assert result.role == "rkv"
    # The runtime R-KV config verification (a REAL function,
    # kvcot.discovery.runtime_rkv_verification.verify_runtime_matches_frozen)
    # ran against the fake kv_cluster's real attribute values and matched
    # the frozen config -- proving this seam actually executed, not skipped.
    assert result.rkv_config_hash_match is True
    # cap_hit invalidates the example before Pass 2 is ever reached --
    # documented scope limitation (module docstring) -- but the REAL
    # worker-result construction still ran to completion from real (not
    # preconstructed) values.
    assert result.example_valid is False
    assert result.token_identical_replay is False  # Pass 2 never attempted
    assert result.selected_event_count_exact is False
    assert result.attempted_real_pair_count == 0
    assert result.determinism_policy["framework_seed"] == 13
    # B1 execution-boundary closure §4: the worker's own explicit reset
    # (once, before any state construction) must be the ONLY peak-memory
    # reset -- `reset_patched_state`, called here to build Pass 1's initial
    # state, must not add a second, hidden one now that it owns model/cache
    # state only.
    assert fake_cuda.reset_peak_memory_stats_call_count == 1


def test_run_rkv_worker_requires_cuda_when_no_fake_backend_injected():
    from kvcot.discovery.b2a_workers import WorkerFailedError

    config = _build_fake_discovery_config()
    manifest = _FakeManifest()
    with pytest.raises(WorkerFailedError, match="requires CUDA"):
        run_rkv_worker(config, manifest)


def test_run_rkv_worker_fails_closed_when_fake_kv_cluster_disagrees_with_frozen_config():
    """A runtime R-KV mismatch (a fake `kv_cluster.budget` that disagrees
    with the frozen config) must raise, never silently proceed."""
    from kvcot.discovery.b2a_workers import WorkerFailedError

    config = _build_fake_discovery_config()
    manifest = _FakeManifest()
    rkv_lock = config.rkv
    model = _FakeModel(rkv_lock=rkv_lock)
    # Mutate EVERY layer consistently (never just one -- a per-layer split
    # would instead raise the unrelated cross-layer-agreement error) so this
    # test isolates the intended frozen-vs-runtime mismatch path.
    for layer in model.model.layers:
        layer.self_attn.kv_cluster.budget = 999999999  # disagrees with config.rkv.budget

    with pytest.raises(WorkerFailedError, match="runtime R-KV configuration disagrees"):
        run_rkv_worker(
            config, manifest,
            _load_model=lambda: model, _load_tokenizer=lambda: _FakeTokenizer(),
            _fresh_cache_factory=lambda: _FakeCache(NUM_LAYERS), _cuda=_FakeCudaFacade(), _device="cpu",
        )


def test_production_call_shape_never_passes_injection_kwargs():
    """B1B-R4 §19: the production CLI/subprocess entry point
    (`kvcot.discovery.b2a_worker_entry`) calls `run_fullkv_worker(config,
    manifest)`/`run_rkv_worker(config, manifest)` with no extra keyword
    arguments -- verified by source inspection, so a future edit that
    accidentally threads a fake-backend option into the production path is
    caught here."""
    import inspect

    from kvcot.discovery import b2a_worker_entry

    source = inspect.getsource(b2a_worker_entry)
    assert "run_fullkv_worker(config, manifest)" in source
    assert "run_rkv_worker(config, manifest)" in source
    assert "_load_model" not in source
    assert "_cuda" not in source
