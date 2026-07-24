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
from kvcot.discovery.final_contract import (
    FULLKV_REQUIRED_TIMING_PHASES,
    PAIR_REQUIRED_TIMING_SUBPHASES,
    RKV_REQUIRED_TIMING_PHASES,
)

NUM_LAYERS = 2
CONTROLLED_NUM_LAYERS = 3
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
        config_kwargs = dict(num_key_value_heads=NUM_HEADS, vocab_size=VOCAB_SIZE)
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


class _ControlledTokenizer:
    eos_token_id = 7

    def decode(self, ids, skip_special_tokens=True):
        return r"The computation is complete. \boxed{0}"


class _ControlledModel(_FakeModel):
    """CPU causal-LM fake that drives the complete R-KV success path."""

    def __init__(self, rkv_lock, *, stop_length=130):
        from _fake_rkv_fixtures import FakeR1KV

        super().__init__(rkv_lock=rkv_lock, num_layers=CONTROLLED_NUM_LAYERS)
        self.stop_length = stop_length
        for layer in self.model.layers:
            layer.self_attn.kv_cluster = FakeR1KV(
                budget=rkv_lock.budget,
                window_size=rkv_lock.window_size,
                kernel_size=rkv_lock.kernel_size,
                mix_lambda=rkv_lock.mix_lambda,
                retain_ratio=rkv_lock.retain_ratio,
                retain_direction=rkv_lock.retain_direction,
            )

    def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=None, cache_position=None):
        cache = past_key_values
        seq_len = input_ids.shape[1]
        if not hasattr(self, "length"):
            self.length = 0
        fed_positions = torch.arange(self.length, self.length + seq_len, dtype=torch.float32)
        self.length += seq_len
        for layer_index, layer in enumerate(self.model.layers):
            offsets = torch.arange(HEAD_DIM, dtype=torch.float32).view(1, 1, 1, -1)
            new_k = fed_positions.view(1, 1, -1, 1) + offsets + layer_index * 0.01
            new_v = new_k + 0.5
            key = torch.cat([cache.key_cache[layer_index], new_k], dim=-2)
            value = torch.cat([cache.value_cache[layer_index], new_v], dim=-2)
            # Compact at a controlled cadence.  Several positions accumulate
            # between events, yielding at least two genuine evictions per
            # event and therefore a valid 2x2 candidate/donor pool.
            if self.length % 8 == 0 and key.shape[-2] >= layer.self_attn.kv_cluster.budget:
                key, value = layer.self_attn.kv_cluster.update_kv(key, new_k, value)
            cache.key_cache[layer_index] = key
            cache.value_cache[layer_index] = value
        logits = torch.full((1, seq_len, VOCAB_SIZE), -20.0)
        logits[:, :, 7 if self.length >= self.stop_length else 1] = 20.0
        return SimpleNamespace(logits=logits)


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


def _build_controlled_discovery_config():
    config = _build_fake_discovery_config()
    return config.model_copy(
        update={
            "rkv": config.rkv.model_copy(update={"budget": 12, "window_size": 4, "divide_length": 8}),
            "generation": config.generation.model_copy(update={"max_new_tokens": 160}),
        }
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
    assert result.actual_batch_size_verified is True
    assert result.actual_call_evidence[0]["call_kind"] == "prefill"
    assert result.actual_call_evidence[0]["input_ids_shape"] == [1, 3]
    assert all(event["batch_size"] == 1 for event in result.actual_call_evidence)
    assert result.every_parameter_on_cuda is True  # derived from the fake param's device.type == "cuda"
    assert result.peak_cuda_allocated_bytes == 12345
    assert result.peak_cuda_reserved_bytes == 23456
    assert result.determinism_policy["framework_seed"] == 13
    assert result.runtime_generation["generation_mode"] == "greedy"
    assert result.runtime_generation["do_sample"] is False
    # B1 execution-boundary closure §4: `reset_patched_state` (called here
    # to build the natural-generation state) must not add a second, hidden
    # peak-memory reset on top of the worker's explicit phase-owned resets.
    assert fake_cuda.reset_peak_memory_stats_call_count == len(result.memory_phase_evidence)
    assert set(FULLKV_REQUIRED_TIMING_PHASES).issubset(
        {record["phase"] for record in result.timing_evidence}
    )
    assert {phase["phase"] for phase in result.memory_phase_evidence} >= {
        "before_model_load", "model_load", "post_load_baseline",
        "fullkv_complete_natural_generation", "fullkv_complete_worker",
    }


def test_run_fullkv_worker_requires_cuda_when_no_fake_backend_injected(monkeypatch):
    """Never silently uses a fake in production -- omitting the injection
    seams entirely (the production call shape) must still hard-require real
    CUDA. Host-neutral (repaired 2026-07-24, see
    docs/B2A_R3_GPU_HOST_NEUTRAL_PREFLIGHT_TEST_REPAIR_2026-07-24.md):
    only `torch.cuda.is_available` is monkeypatched to a deterministic
    `False` so this passes identically whether the real physical host has
    a visible CUDA device or not -- `_cuda` stays omitted, so the worker
    still selects real `torch.cuda` (`cuda = _cuda if _cuda is not None
    else torch.cuda`), exercising the actual production default-selection
    call shape rather than a fake CUDA facade."""
    from kvcot.discovery.b2a_workers import WorkerFailedError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _fail_if_reached(*args, **kwargs):
        raise AssertionError("snapshot resolution must not be reached after clean no-CUDA refusal")

    monkeypatch.setattr("kvcot.discovery.snapshot_boundary.resolve_local_snapshot", _fail_if_reached)

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
    assert fake_cuda.reset_peak_memory_stats_call_count == len(result.memory_phase_evidence)


def test_run_rkv_worker_executes_complete_twelve_pair_success_path(monkeypatch):
    from _fake_rkv_fixtures import install_fake_rkv_compression_module

    install_fake_rkv_compression_module(monkeypatch)
    config = _build_controlled_discovery_config()
    manifest = _FakeManifest()
    model = _ControlledModel(config.rkv)
    fake_cuda = _FakeCudaFacade()

    # The production branch restorer intentionally creates DynamicCache
    # itself.  Replace that dependency with the same real-tensor CPU cache
    # used by the injected initial-state factory; orchestration is untouched.
    import sys
    import types

    fake_transformers = types.ModuleType("transformers")
    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = lambda: _FakeCache(CONTROLLED_NUM_LAYERS)
    fake_transformers.cache_utils = fake_cache_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)
    result = RKVWorkerResult.model_validate(
        run_rkv_worker(
            config,
            manifest,
            _load_model=lambda: model,
            _load_tokenizer=lambda: _ControlledTokenizer(),
            _fresh_cache_factory=lambda: _FakeCache(CONTROLLED_NUM_LAYERS),
            _cuda=fake_cuda,
            _device="cpu",
        )
    )
    assert result.example_valid is True
    assert result.natural_answer_status == "correct"
    assert result.eligible_compaction_events >= 3
    assert result.selected_compaction_events == 3
    assert result.attempted_real_pair_count == 12
    assert result.completed_real_pair_count == 12
    assert result.failed_real_pair_count == 0
    assert result.unique_completed_real_pair_count == 12
    assert result.events_with_exactly_four_unique_real_pairs == 3
    assert result.has_duplicate_real_pair_identity is False
    assert result.attempted_no_op_pair_count == 1
    assert result.completed_no_op_pair_count == 1
    assert result.has_duplicate_no_op_pair_identity is False
    assert result.semantic_swap_checks_required == 12
    assert result.semantic_swap_checks_attempted == 12
    assert result.semantic_swap_checks_passed == 12
    assert result.semantic_swap_checks_failed == 0
    assert result.no_op_numerical_parity is True
    assert len(result.selected_event_evidence) == 3
    assert len({item["compaction_event_id"] for item in result.selected_event_evidence}) == 3
    real_attempts = [item for item in result.attempted_pair_identities if item["pair_kind"] == "real"]
    assert len(real_attempts) == 12
    assert len({tuple(sorted(item.items())) for item in real_attempts}) == 12
    assert len(result.completed_pair_identities) == 13
    assert result.failed_pair_identities == []
    assert result.no_op_identity["pair_kind"] == "no_op"
    assert len(result.semantic_mutation_reports) == 13
    assert result.no_op_evidence["baseline_nll"] == result.no_op_evidence["no_op_nll"]
    assert result.no_op_evidence["baseline_nll_sha256"] == result.no_op_evidence["no_op_nll_sha256"]
    assert result.no_op_evidence["maximum_absolute_per_token_difference"] == 0.0
    assert result.no_op_evidence["starting_snapshot_sha256"]
    assert result.replay_evidence["pass1_token_sha256"] == result.replay_evidence["pass2_token_sha256"]
    assert result.replay_evidence["complete_compaction_trace_match"] is True
    memory_phases = [item["phase"] for item in result.memory_phase_evidence]
    assert "model_load" in memory_phases
    assert "rkv_complete_pass1" in memory_phases
    assert "rkv_complete_pass2" in memory_phases
    assert sum(phase.startswith("real_pair:") for phase in memory_phases) == 12
    assert sum(phase.startswith("no_op_pair:") for phase in memory_phases) == 1
    timing_phases = {item["phase"] for item in result.timing_evidence}
    assert set(RKV_REQUIRED_TIMING_PHASES).issubset(timing_phases)
    complete_pairs = [
        phase for phase in timing_phases
        if (phase.startswith("real_pair:") or phase.startswith("no_op_pair:")) and phase.count(":") == 3
    ]
    assert len(complete_pairs) == 13
    for complete in complete_pairs:
        assert all(f"{complete}:{subphase}" in timing_phases for subphase in PAIR_REQUIRED_TIMING_SUBPHASES)
    assert result.actual_batch_size_verified is True
    assert all(event["batch_size"] == 1 for event in result.actual_call_evidence)
    assert any(event["call_kind"] == "prefill" for event in result.actual_call_evidence)
    assert any(event["call_kind"] == "decode" for event in result.actual_call_evidence)
    assert all(value > 0 for value in result.real_pair_wall_seconds)
    assert all(value > 0 for value in result.no_op_pair_wall_seconds)


def test_run_rkv_worker_requires_cuda_when_no_fake_backend_injected(monkeypatch):
    """Host-neutral (repaired 2026-07-24, see
    docs/B2A_R3_GPU_HOST_NEUTRAL_PREFLIGHT_TEST_REPAIR_2026-07-24.md):
    only `torch.cuda.is_available` is monkeypatched to a deterministic
    `False`; `_cuda` stays omitted so the worker still selects real
    `torch.cuda`, exercising the actual production default-selection call
    shape rather than a fake CUDA facade."""
    from kvcot.discovery.b2a_workers import WorkerFailedError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def _fail_if_reached(*args, **kwargs):
        raise AssertionError("snapshot resolution must not be reached after clean no-CUDA refusal")

    monkeypatch.setattr("kvcot.discovery.snapshot_boundary.resolve_local_snapshot", _fail_if_reached)

    config = _build_fake_discovery_config()
    manifest = _FakeManifest()
    with pytest.raises(WorkerFailedError, match="requires CUDA"):
        run_rkv_worker(config, manifest)


def test_run_rkv_worker_fails_closed_when_fake_kv_cluster_disagrees_with_frozen_config():
    """A runtime R-KV mismatch (a fake `kv_cluster.budget` that disagrees
    with the frozen config) must raise, never silently proceed. Gate H1:
    every exception raised inside the worker body -- including this
    deliberate `WorkerFailedError` refusal -- is now wrapped in a
    `WorkerBodyFailure` carrying partial evidence, with the original
    `WorkerFailedError` preserved as `__cause__` (never swallowed)."""
    from kvcot.discovery.b2a_workers import WorkerFailedError
    from kvcot.discovery.worker_partial_evidence import WorkerBodyFailure

    config = _build_fake_discovery_config()
    manifest = _FakeManifest()
    rkv_lock = config.rkv
    model = _FakeModel(rkv_lock=rkv_lock)
    # Mutate EVERY layer consistently (never just one -- a per-layer split
    # would instead raise the unrelated cross-layer-agreement error) so this
    # test isolates the intended frozen-vs-runtime mismatch path.
    for layer in model.model.layers:
        layer.self_attn.kv_cluster.budget = 999999999  # disagrees with config.rkv.budget

    with pytest.raises(WorkerBodyFailure, match="runtime R-KV configuration disagrees") as exc_info:
        run_rkv_worker(
            config, manifest,
            _load_model=lambda: model, _load_tokenizer=lambda: _FakeTokenizer(),
            _fresh_cache_factory=lambda: _FakeCache(NUM_LAYERS), _cuda=_FakeCudaFacade(), _device="cpu",
        )
    assert isinstance(exc_info.value.__cause__, WorkerFailedError)
    # F1: the FAILING stage is the stage that was executing when the
    # exception was raised -- never the last COMPLETED stage.
    assert exc_info.value.evidence.failing_stage == "runtime R-KV config verification"
    assert exc_info.value.evidence.last_completed_stage == "model_load"
    assert exc_info.value.evidence.is_oom is False
    # Partial evidence genuinely accumulated before this failure (model
    # load, determinism policy) survives on the typed evidence object.
    assert exc_info.value.evidence.determinism_policy is not None
    assert len(exc_info.value.evidence.timing_evidence) > 0


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


def _run_controlled_coordinator(monkeypatch, tmp_path):
    """CPU end-to-end harness: canonical bodies -> entry envelopes ->
    coordinator -> artifact. Returns `(artifact, attempt)`."""
    import contextlib
    import io
    import sys
    import types
    from types import SimpleNamespace

    from _fake_rkv_fixtures import install_fake_rkv_compression_module
    from kvcot.discovery import b2a_execute, b2a_worker_entry, b2a_workers
    from kvcot.discovery.attempt_artifacts import atomic_write_json, create_attempt_directory

    install_fake_rkv_compression_module(monkeypatch)
    config = _build_controlled_discovery_config()
    manifest = _FakeManifest()
    manifest.dataset_config = "default"
    manifest.dataset_split = "test"
    manifest.raw_content_hash = "r" * 64
    manifest.rendered_user_message_sha256 = "u" * 64
    manifest.chat_template_source_sha256 = "t" * 64

    fake_transformers = types.ModuleType("transformers")
    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = lambda: _FakeCache(CONTROLLED_NUM_LAYERS)
    fake_transformers.cache_utils = fake_cache_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)

    real_full = run_fullkv_worker
    real_rkv = run_rkv_worker

    def tokenizer():
        value = _ControlledTokenizer()
        value._commit_hash = config.model.tokenizer_revision
        return value

    def model():
        value = _ControlledModel(config.rkv)
        value.config._commit_hash = config.model.revision
        return value

    monkeypatch.setattr(
        b2a_workers, "run_fullkv_worker",
        lambda cfg, man: real_full(
            cfg, man, _load_model=model, _load_tokenizer=tokenizer,
            _fresh_cache_factory=lambda: _FakeCache(CONTROLLED_NUM_LAYERS), _cuda=_FakeCudaFacade(), _device="cpu",
        ),
    )
    monkeypatch.setattr(
        b2a_workers, "run_rkv_worker",
        lambda cfg, man: real_rkv(
            cfg, man, _load_model=model, _load_tokenizer=tokenizer,
            _fresh_cache_factory=lambda: _FakeCache(CONTROLLED_NUM_LAYERS), _cuda=_FakeCudaFacade(), _device="cpu",
        ),
    )
    monkeypatch.setattr("kvcot.discovery.discovery_config.load_discovery_config", lambda path: config)
    monkeypatch.setattr("kvcot.discovery.manifest.load_b2a_one_example_manifest", lambda path: manifest)
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda cfg, man: None)

    def runner(argv, **kwargs):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = b2a_worker_entry.main(argv[3:])
        return SimpleNamespace(returncode=returncode, stdout=stdout.getvalue(), stderr=stderr.getvalue())

    attempt = create_attempt_directory(root=tmp_path, attempt_id="cpu-e2e")
    atomic_write_json(attempt.path / "invocation.json", {
        "attempt_id": attempt.attempt_id, "started_at": "2026-01-01T00:00:00+00:00",
        "argv": ["python", "-m", "kvcot", "b2a-calibrate", "--execute"],
        "config_path": "controlled.yaml", "manifest_path": "controlled-manifest.json",
    })
    atomic_write_json(attempt.path / "preflight.json", {"passed": True})
    atomic_write_json(attempt.path / "provenance.json", {"git": {"dirty": False, "rkv_submodule_match": True}})
    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path="controlled.yaml", manifest_path="controlled-manifest.json",
        python_executable="fake-python", subprocess_runner=runner, attempt_directory=attempt.path,
    )
    return artifact, attempt


def test_actual_worker_bodies_flow_through_envelopes_coordinator_and_artifact(monkeypatch, tmp_path):
    artifact, attempt = _run_controlled_coordinator(monkeypatch, tmp_path)
    assert artifact.gate_result.passed is True
    assert artifact.artifact_path == attempt.path / "final.json"
    assert artifact.artifact_path.is_file()
    assert (attempt.path / "fullkv" / "envelope.json").is_file()
    assert (attempt.path / "rkv" / "envelope.json").is_file()
    assert (attempt.path / "rkv" / "pair_identities.json").is_file()
    assert (attempt.path / "process_outcome.json").is_file()
    # CPU fakes are deliberately not accepted as RTX-3090/snapshot evidence.
    assert artifact.final_gate_result.passed is False
    assert "single_rtx3090_verified" in artifact.final_gate_result.failed_conditions
    assert "no_offload_and_placement_verified" in artifact.final_gate_result.failed_conditions


def test_coordinator_lifecycle_writes_completion_before_final_and_references_it(monkeypatch, tmp_path):
    """F5: completion.json exists before final.json is written, so the final
    reference manifest includes it; final.json never references itself."""
    import json

    artifact, attempt = _run_controlled_coordinator(monkeypatch, tmp_path)
    completion = json.loads((attempt.path / "completion.json").read_text(encoding="utf-8"))
    assert completion["attempt_id"] == attempt.attempt_id
    assert completion["outcome"] == "gate_failed"  # CPU fakes never pass the final device gates
    assert completion["exit_code"] == 2
    assert completion["intended_final_relative_path"] == "final.json"

    final = json.loads((attempt.path / "final.json").read_text(encoding="utf-8"))
    listed = [item["relative_path"] for item in final["attempt_artifacts"]["files"]]
    assert "completion.json" in listed
    assert "process_outcome.json" in listed
    assert "final.json" not in listed
    assert all(item["semantic_role"] != "unknown" for item in final["attempt_artifacts"]["files"])

    from kvcot.discovery.attempt_verification import verify_final_reference_manifest

    ok, reasons = verify_final_reference_manifest(attempt.path)
    assert reasons == ()
    assert ok is True


def test_coordinator_final_write_failure_preserves_prefinal_artifacts(monkeypatch, tmp_path):
    """F5: a failed final.json write preserves every pre-final artifact,
    writes final_write_failure.json, never overwrites completion.json, and
    is never reported as a completed successful attempt."""
    import json

    import pytest as pytest_module

    from kvcot.discovery import attempt_artifacts as artifacts_module
    from kvcot.discovery.b2a_execute import B2AFinalWriteError

    real_write = artifacts_module.atomic_write_json

    def failing_write(path, payload):
        if path.name == "final.json":
            raise OSError("disk full while writing final.json")
        return real_write(path, payload)

    monkeypatch.setattr(artifacts_module, "atomic_write_json", failing_write)
    with pytest_module.raises(B2AFinalWriteError):
        _run_controlled_coordinator(monkeypatch, tmp_path)
    attempt_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(attempt_dirs) == 1
    attempt_path = attempt_dirs[0]
    assert not (attempt_path / "final.json").exists()
    assert (attempt_path / "completion.json").is_file()
    assert (attempt_path / "final_write_failure.json").is_file()
    failure = json.loads((attempt_path / "final_write_failure.json").read_text(encoding="utf-8"))
    assert failure["failure_type"] == "OSError"
    for preserved in ("invocation.json", "preflight.json", "provenance.json", "process_outcome.json",
                      "fullkv/result.json", "rkv/result.json"):
        assert (attempt_path / preserved).is_file()
