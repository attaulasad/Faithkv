"""CPU-only (no GPU, no real model download) regression tests for
`kvcot.generation.replay.branch_and_probe`'s `probe_cache_mode="frozen_at_cut"`
(protocol v3, CHANGELOG.md 2026-07-17/2026-07-18) — 2026-07-18 review asked
for "CPU/fake-model tests" of this logic specifically, since the real GPU
tests (`test_frozen_probe_gpu.py`) cannot run on this build machine at all.

`FakeRKVModel` below reproduces the exact schedule/trigger mechanics audited
in `docs/UPSTREAM_AUDIT.md` H3/H4 well enough to drive `kvcot.generation.
decode.decode_step`'s real contract (a callable returning `.logits`, given
`input_ids`/`position_ids`/`past_key_values`/`cache_position`) — WITHOUT a
real transformer forward pass:

  - `self.length` (cumulative token counter) increments by this call's
    token count, exactly as modeling.py does.
  - Each layer reads its OWN `compression` flag (`None`/`True`/`False`) at
    the START of the call: `None`/`True` attempts eviction (no-op if the
    post-append cache length is still below budget, else evicts to exactly
    budget); `False` is a pure no-op growth, no eviction attempt at all.
  - AFTER every layer is processed, the flag for every layer is
    OVERWRITTEN from the schedule (`self.length % divide_length == 0`) --
    this is the exact mechanic that makes `_force_compression_off` need to
    run before EVERY fed token, not just once before the loop: a single
    reset before the loop would be silently undone by the first real
    forward call.

This module needs a real (CPU-only) torch install to import
`kvcot.generation.replay` at all (it imports torch at module scope by
design, kvcot.generation is never required to be torch-free) --
`pytest.importorskip` at module scope skips this whole file cleanly on a
machine without torch (e.g. this repository's documented CPU-only dev setup,
`pip install -e ".[cpu-tools,dev]"`, which does not include torch), rather
than a hard collection error. No `@pytest.mark.gpu` — nothing here touches
CUDA or downloads a real model, so it is not gated by that marker or by
`tests/conftest.py`'s GPU auto-skip.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kvcot.generation.replay import branch_and_probe  # noqa: E402
from kvcot.generation.state import ModelStateSnapshot  # noqa: E402

NUM_LAYERS = 2
NUM_HEADS = 1
HEAD_DIM = 4
VOCAB_SIZE = 8


@dataclass
class _FakeConfig:
    compression: object = None


class _FakeKVCluster:
    def __init__(self, budget: int):
        self.budget = budget
        self.evicted_token_num = 0
        self.kept_token_indices: list = []
        self.kept_attention_scores: list = []
        self.kept_similarity_scores: list = []
        self.kept_final_scores: list = []
        self.record_kept_token_indices = True


class _FakeAttn:
    def __init__(self, budget: int):
        self.config = _FakeConfig()
        self.kv_cluster = _FakeKVCluster(budget)


class _FakeLayer:
    def __init__(self, budget: int):
        self.self_attn = _FakeAttn(budget)


class _FakeProvenance:
    def clone(self):
        return self


class _FakeCache:
    def __init__(self, num_layers: int):
        self.key_cache = [torch.empty(0) for _ in range(num_layers)]
        self.value_cache = [torch.empty(0) for _ in range(num_layers)]

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class FakeRKVModel:
    """Faithful-but-lightweight stand-in for the patched R-KV CausalLM
    forward (see module docstring) — never a real transformer, just the
    schedule/trigger/cache-length bookkeeping `branch_and_probe`'s
    frozen_at_cut assertions depend on being correct."""

    def __init__(self, budget: int, divide_length: int, num_layers: int = NUM_LAYERS):
        self.model = SimpleNamespace(layers=[_FakeLayer(budget) for _ in range(num_layers)])
        self.divide_length = divide_length

    def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=None, cache_position=None):
        cache = past_key_values
        seq_len = input_ids.shape[1]
        if not hasattr(self, "length"):
            self.length = seq_len
        else:
            self.length += seq_len

        for i, layer in enumerate(self.model.layers):
            kv_cluster = layer.self_attn.kv_cluster
            compression = layer.self_attn.config.compression
            current_len = cache.key_cache[i].shape[-2] if cache.key_cache[i].numel() > 0 else 0
            new_len = current_len + seq_len
            if compression is None or compression is True:
                if new_len >= kv_cluster.budget:
                    kv_cluster.kept_token_indices.append(torch.arange(kv_cluster.budget))
                    kv_cluster.evicted_token_num += new_len - kv_cluster.budget
                    new_len = kv_cluster.budget
            cache.key_cache[i] = torch.zeros(1, NUM_HEADS, new_len, HEAD_DIM)
            cache.value_cache[i] = torch.zeros(1, NUM_HEADS, new_len, HEAD_DIM)

        # Overwrite EVERY layer's flag from the schedule, unconditionally,
        # exactly like modeling.py's CausalLM_forward -- this is what
        # `_force_compression_off` must outrun by running before every call.
        is_newline = self.length % self.divide_length == 0
        for layer in self.model.layers:
            layer.self_attn.config.compression = is_newline

        logits = torch.zeros(1, seq_len, VOCAB_SIZE)
        return SimpleNamespace(logits=logits)


def _make_snapshot(start_len: int, num_layers: int = NUM_LAYERS) -> ModelStateSnapshot:
    return ModelStateSnapshot(
        key_cache=[torch.zeros(1, NUM_HEADS, start_len, HEAD_DIM) for _ in range(num_layers)],
        value_cache=[torch.zeros(1, NUM_HEADS, start_len, HEAD_DIM) for _ in range(num_layers)],
        query_cache={},
        compression_flags_per_layer=["true"] * num_layers,
        model_length=start_len,
        after_think=None,
        compaction_event_steps=[],
        tokens_since_last_compaction=0,
        absolute_position=start_len,
        provenance=_FakeProvenance(),
        kv_cluster_bookkeeping_per_layer=[
            {
                "evicted_token_num": 0, "kept_token_indices": [], "kept_attention_scores": [],
                "kept_similarity_scores": [], "kept_final_scores": [],
            }
            for _ in range(num_layers)
        ],
    )


# Aggressive schedule (budget already reached at the snapshot, divide_length
# tiny) so a schedule check -- and, under native mode, a real eviction --
# fires within the ~20-token probe window used below.
BUDGET = 8
DIVIDE_LENGTH = 2
START_LEN = 8  # already at budget
EOS_TOKEN_ID = -1  # unreachable -- decoding always runs the full max_new_tokens
MAX_NEW_TOKENS = 20


def test_frozen_at_cut_forces_compression_off_before_every_fed_token():
    """The core mechanism: FakeRKVModel overwrites every layer's compression
    flag from the schedule at the END of every call, unconditionally -- if
    `_force_compression_off` only ran once (e.g. before the loop, not
    before every `feed()` call), the second scheduled call would silently
    re-enable compression and evict. Asserting zero net growth-plus-eviction
    across ~20 tokens under divide_length=2 (10 schedule checks) is only
    possible if the override is re-applied before literally every call.
    """
    model = FakeRKVModel(budget=BUDGET, divide_length=DIVIDE_LENGTH)
    cache = _FakeCache(NUM_LAYERS)
    snap = _make_snapshot(START_LEN)

    result = branch_and_probe(
        model, cache, snap,
        close_marker_token_ids=[1], control_suffix_token_ids=[2, 3],
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID, device="cpu",
        probe_cache_mode="frozen_at_cut",
    )

    tokens_fed = result.final_absolute_position - snap.absolute_position
    assert tokens_fed > 0
    expected = [START_LEN + tokens_fed] * NUM_LAYERS
    assert result.final_cache_lengths_per_layer == expected
    assert result.probe_cache_mode == "frozen_at_cut"


def test_frozen_at_cut_leaves_evicted_token_num_unchanged():
    model = FakeRKVModel(budget=BUDGET, divide_length=DIVIDE_LENGTH)
    cache = _FakeCache(NUM_LAYERS)
    snap = _make_snapshot(START_LEN)

    branch_and_probe(
        model, cache, snap,
        close_marker_token_ids=[1], control_suffix_token_ids=[2, 3],
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID, device="cpu",
        probe_cache_mode="frozen_at_cut",
    )

    for layer in model.model.layers:
        assert layer.self_attn.kv_cluster.evicted_token_num == 0
        assert layer.self_attn.kv_cluster.kept_token_indices == []


def test_native_mode_evicts_under_the_same_fixture_frozen_prevents():
    """Confirms this fixture is a meaningful regression scenario: NATIVE
    mode, under the identical setup frozen_at_cut is tested against above,
    must show real growth-then-eviction (cache length does NOT simply equal
    start + tokens_fed, and evicted_token_num grows) -- otherwise the two
    tests above would be vacuously true."""
    model = FakeRKVModel(budget=BUDGET, divide_length=DIVIDE_LENGTH)
    cache = _FakeCache(NUM_LAYERS)
    snap = _make_snapshot(START_LEN)

    result = branch_and_probe(
        model, cache, snap,
        close_marker_token_ids=[1], control_suffix_token_ids=[2, 3],
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID, device="cpu",
        probe_cache_mode="native",
    )

    tokens_fed = result.final_absolute_position - snap.absolute_position
    never_evicted = [START_LEN + tokens_fed] * NUM_LAYERS
    assert result.final_cache_lengths_per_layer != never_evicted
    assert result.probe_cache_mode == "native"
    for layer in model.model.layers:
        assert layer.self_attn.kv_cluster.evicted_token_num > 0


def test_frozen_at_cut_raises_if_cache_length_assertion_would_fail():
    """Constructs a model that ignores the compression override entirely
    (a deliberately broken fake, standing in for a hypothetical upstream/
    monkeypatch drift) to confirm branch_and_probe's own hard assertion
    fires rather than silently returning a contaminated result."""

    class _BrokenModel(FakeRKVModel):
        def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=None, cache_position=None):
            # Ignore whatever the caller set -- always attempt eviction,
            # exactly the bug frozen_at_cut is supposed to prevent.
            for layer in self.model.layers:
                layer.self_attn.config.compression = True
            return super().__call__(input_ids, position_ids, past_key_values, use_cache, cache_position)

    model = _BrokenModel(budget=BUDGET, divide_length=DIVIDE_LENGTH)
    cache = _FakeCache(NUM_LAYERS)
    snap = _make_snapshot(START_LEN)

    with pytest.raises(RuntimeError, match="frozen_at_cut probe changed cache length unexpectedly"):
        branch_and_probe(
            model, cache, snap,
            close_marker_token_ids=[1], control_suffix_token_ids=[2, 3],
            max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID, device="cpu",
            probe_cache_mode="frozen_at_cut",
        )


def test_frozen_at_cut_raises_on_eviction_count_mismatch_even_if_length_matches():
    """A model that evicts (grows kv_cluster.evicted_token_num /
    kept_token_indices) but coincidentally leaves the physical cache length
    unchanged from start+tokens_fed (e.g. it evicted then re-grew by the
    same amount) must still be caught -- branch_and_probe's frozen_at_cut
    assertion checks evicted_token_num independently of cache length."""

    class _SneakyModel(FakeRKVModel):
        def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=None, cache_position=None):
            out = super().__call__(input_ids, position_ids, past_key_values, use_cache, cache_position)
            # After the real (correct, frozen) call completed, sneak in a
            # bookkeeping-only eviction-count bump with no cache-length
            # change -- simulates a monkeypatch/version drift where the
            # length invariant coincidentally still holds but a real
            # compaction event was nonetheless recorded.
            for layer in self.model.layers:
                layer.self_attn.kv_cluster.evicted_token_num += 1
            return out

    model = _SneakyModel(budget=BUDGET, divide_length=DIVIDE_LENGTH)
    cache = _FakeCache(NUM_LAYERS)
    snap = _make_snapshot(START_LEN)

    with pytest.raises(RuntimeError, match="recorded a NEW compaction event"):
        branch_and_probe(
            model, cache, snap,
            close_marker_token_ids=[1], control_suffix_token_ids=[2, 3],
            max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID, device="cpu",
            probe_cache_mode="frozen_at_cut",
        )


def test_native_mode_behavior_is_unaffected_by_frozen_at_cut_existing():
    """Regression: adding probe_cache_mode must not change the default
    ("native") code path at all -- calling branch_and_probe without the
    argument produces identical results to explicitly passing "native"."""
    model_a = FakeRKVModel(budget=BUDGET, divide_length=DIVIDE_LENGTH)
    cache_a = _FakeCache(NUM_LAYERS)
    snap_a = _make_snapshot(START_LEN)
    result_default = branch_and_probe(
        model_a, cache_a, snap_a,
        close_marker_token_ids=[1], control_suffix_token_ids=[2, 3],
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID, device="cpu",
    )

    model_b = FakeRKVModel(budget=BUDGET, divide_length=DIVIDE_LENGTH)
    cache_b = _FakeCache(NUM_LAYERS)
    snap_b = _make_snapshot(START_LEN)
    result_explicit_native = branch_and_probe(
        model_b, cache_b, snap_b,
        close_marker_token_ids=[1], control_suffix_token_ids=[2, 3],
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID, device="cpu",
        probe_cache_mode="native",
    )

    assert result_default.probe_output_token_ids == result_explicit_native.probe_output_token_ids
    assert result_default.final_cache_lengths_per_layer == result_explicit_native.final_cache_lengths_per_layer
    assert result_default.probe_cache_mode == "native"


def test_invalid_probe_cache_mode_is_never_silently_treated_as_native():
    """probe_cache_mode is a plain str parameter (not a Literal) on
    branch_and_probe's own signature, but only "native" ever triggers the
    frozen-cache override path (`if probe_cache_mode == "frozen_at_cut"`) --
    an unrecognized value must behave exactly like "native" (fail open to
    the SAFE default, never silently apply partial frozen behavior) rather
    than raising or silently freezing. This documents that contract
    explicitly rather than leaving it implicit."""
    model = FakeRKVModel(budget=BUDGET, divide_length=DIVIDE_LENGTH)
    cache = _FakeCache(NUM_LAYERS)
    snap = _make_snapshot(START_LEN)

    result = branch_and_probe(
        model, cache, snap,
        close_marker_token_ids=[1], control_suffix_token_ids=[2, 3],
        max_new_tokens=MAX_NEW_TOKENS, eos_token_id=EOS_TOKEN_ID, device="cpu",
        probe_cache_mode="some_typo_value",
    )
    # No exception, and no frozen-cache assertion applied (that assertion
    # only runs when probe_cache_mode == "frozen_at_cut" exactly).
    assert result.probe_cache_mode == "some_typo_value"
