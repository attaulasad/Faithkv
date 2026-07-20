import contextlib

import pytest
import torch

from kvcot.discovery.capture import _recomputed_kept_physical_indices, capture_update_kv
from kvcot.generation.provenance import LayerProvenance

from _fake_rkv_fixtures import FakeR1KV, install_fake_rkv_compression_module

BUDGET = 12
WINDOW = 4
NUM_HEADS = 2
HEAD_DIM = 8
SEQ_LEN = 20  # >= BUDGET -> triggers compaction


def _make_tensors(seed: int, seq_len: int = SEQ_LEN, window: int = WINDOW):
    g = torch.Generator().manual_seed(seed)
    key_states = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM, generator=g)
    value_states = torch.randn(1, NUM_HEADS, seq_len, HEAD_DIM, generator=g)
    query_states = torch.randn(1, NUM_HEADS, window, HEAD_DIM, generator=g)
    return key_states, value_states, query_states


def _identity_position_map(seq_len: int) -> torch.Tensor:
    return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(NUM_HEADS, -1).clone()


def test_wrapper_captures_gather_parity_on_real_compaction(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=1)

    sink = []
    with capture_update_kv(kv, sink, pre_event_position_map_fn=lambda: _identity_position_map(SEQ_LEN)):
        k_out, v_out = kv.update_kv(key_states, query_states, value_states)

    assert len(sink) == 1
    record = sink[0]
    assert record.had_compaction is True
    assert record.gather_parity_passed is True
    assert record.parity_check_passed is True
    assert record.parity_failure_reason is None
    assert torch.equal(record.returned_key_states, k_out)
    assert torch.equal(record.returned_value_states, v_out)


def test_no_compaction_below_budget_skips_recomputation(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=2, seq_len=8)  # < budget

    sink = []
    with capture_update_kv(kv, sink):
        kv.update_kv(key_states, query_states, value_states)

    assert len(sink) == 1
    record = sink[0]
    assert record.had_compaction is False
    assert record.recomputed_final_score is None
    assert record.parity_check_passed is True
    assert record.observed_kept_indices_parity_passed is None  # not applicable -- no compaction


def test_wrapper_is_per_instance_not_class_level(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv_a = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    kv_b = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=3)

    sink = []
    with capture_update_kv(kv_a, sink):
        kv_a.update_kv(key_states.clone(), query_states.clone(), value_states.clone())
        # kv_b is a completely separate instance -- never wrapped.
        assert "update_kv" not in kv_b.__dict__
        kv_b.update_kv(key_states.clone(), query_states.clone(), value_states.clone())

    assert len(sink) == 1  # only kv_a's call was captured


def test_restore_on_normal_exit(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=4)

    sink = []
    with capture_update_kv(kv, sink):
        kv.update_kv(key_states.clone(), query_states.clone(), value_states.clone())
    assert "update_kv" not in kv.__dict__

    # Further calls work exactly like the unwrapped original, no capturing.
    kv.update_kv(key_states.clone(), query_states.clone(), value_states.clone())
    assert len(sink) == 1


def test_restore_on_exception(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)

    class _Boom(Exception):
        pass

    sink = []
    try:
        with capture_update_kv(kv, sink):
            raise _Boom("simulated failure mid-capture")
    except _Boom:
        pass

    assert "update_kv" not in kv.__dict__


def test_captured_returned_tensors_are_clones_not_aliases(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=5)

    sink = []
    with capture_update_kv(kv, sink):
        k_out, v_out = kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    before = record.returned_key_states.clone()
    k_out.fill_(0.0)  # mutate the real returned tensor after the fact
    assert torch.equal(record.returned_key_states, before)  # record's clone is unaffected


def test_wrapper_mutates_no_input_tensor_and_no_config_attribute(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW, mix_lambda=0.1, retain_ratio=0.2)
    key_states, value_states, query_states = _make_tensors(seed=6)
    key_before = key_states.clone()
    value_before = value_states.clone()

    sink = []
    with capture_update_kv(kv, sink):
        kv.update_kv(key_states, query_states, value_states)

    assert torch.equal(key_states, key_before)
    assert torch.equal(value_states, value_before)
    assert kv.budget == BUDGET
    assert kv.window_size == WINDOW
    assert kv.mix_lambda == 0.1
    assert kv.retain_ratio == 0.2


def test_hook_off_vs_hook_on_bit_exact_returned_tensors(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    key_states, value_states, query_states = _make_tensors(seed=7)

    kv_off = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    k_off, v_off = kv_off.update_kv(key_states.clone(), query_states.clone(), value_states.clone())

    kv_on = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    sink = []
    with capture_update_kv(kv_on, sink):
        k_on, v_on = kv_on.update_kv(key_states.clone(), query_states.clone(), value_states.clone())

    assert torch.equal(k_off, k_on)
    assert torch.equal(v_off, v_on)


def test_pre_call_tensors_are_retained_as_clones_with_correct_shape(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=10)

    sink = []
    with capture_update_kv(kv, sink):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.pre_call_key_shape == (1, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    assert record.pre_call_value_shape == (1, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    assert record.pre_call_dtype == str(key_states.dtype)
    assert record.pre_call_device == str(key_states.device)
    assert torch.equal(record.pre_call_key_states, key_states)


def test_broken_update_kv_detected_by_gather_parity_check(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)

    class BrokenR1KV(FakeR1KV):
        def update_kv(self, key_states, query_states, value_states):
            k, v = super().update_kv(key_states, query_states, value_states)
            if k.shape[-2] == self.budget:
                # Corrupt the result: roll the compressed cache by one slot,
                # simulating an aliasing/off-by-one eviction bug.
                k = torch.roll(k, shifts=1, dims=2)
                v = torch.roll(v, shifts=1, dims=2)
            return k, v

    kv = BrokenR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=9)

    sink = []
    with capture_update_kv(kv, sink, pre_event_position_map_fn=lambda: _identity_position_map(SEQ_LEN)):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.gather_parity_passed is False
    assert record.parity_check_passed is False
    assert "gather_parity_failed" in record.parity_failure_reason


# --------------------------------------------------------------------------
# Blocker 2: absolute survivor parity at EVERY compaction event (repaired)
# --------------------------------------------------------------------------


def test_observed_kept_indices_parity_passes_at_first_event(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=True)
    key_states, value_states, query_states = _make_tensors(seed=8)

    sink = []
    with capture_update_kv(kv, sink, pre_event_position_map_fn=lambda: _identity_position_map(SEQ_LEN)):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.observed_kept_indices_parity_passed is True
    assert torch.equal(record.recomputed_kept_absolute_positions, record.observed_kept_absolute_positions)


class _TwoEventHarness:
    """Drives a real `FakeR1KV` through exactly two compaction events using
    the REAL `kvcot.generation.provenance.LayerProvenance` adapter to build
    the pre-event absolute-position map at each event -- never a shadow
    reconstruction, exactly the contract `capture_update_kv` requires.
    Event 2's pre-event map is provably non-identity: it is seeded from
    event 1's own (per-head, order-shuffling) survivor selection."""

    def __init__(self, budget=8, window=3, num_heads=2, head_dim=6, seed=100):
        self.budget = budget
        self.window = window
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gen = torch.Generator().manual_seed(seed)
        self.provenance = LayerProvenance.empty(num_heads)
        self.full_keys: torch.Tensor | None = None
        self.full_values: torch.Tensor | None = None

    def _rand(self, seq_len):
        return torch.randn(1, self.num_heads, seq_len, self.head_dim, generator=self.gen)

    def prefill(self, kv, capture_sink, n_tokens: int):
        self.full_keys = self._rand(n_tokens)
        self.full_values = self._rand(n_tokens)
        query = self._rand(self.window)
        self.provenance.append_new_tokens_prefill(list(range(n_tokens)))
        with capture_update_kv(kv, capture_sink, pre_event_position_map_fn=lambda: self.provenance.positions):
            k_out, v_out = kv.update_kv(self.full_keys, query, self.full_values)
        self._maybe_adopt(kv)
        self.full_keys, self.full_values = k_out, v_out
        return capture_sink[-1]

    def append_and_step(self, kv, capture_sink, absolute_position: int):
        new_k = self._rand(1)
        new_v = self._rand(1)
        query = self._rand(self.window)
        self.full_keys = torch.cat([self.full_keys, new_k], dim=2)
        self.full_values = torch.cat([self.full_values, new_v], dim=2)
        self.provenance.append_new_token(absolute_position)
        with capture_update_kv(kv, capture_sink, pre_event_position_map_fn=lambda: self.provenance.positions):
            k_out, v_out = kv.update_kv(self.full_keys, query, self.full_values)
        self._maybe_adopt(kv)
        self.full_keys, self.full_values = k_out, v_out
        return capture_sink[-1]

    def _maybe_adopt(self, kv):
        if getattr(kv, "kept_token_indices", None):
            self.provenance.adopt_upstream_kept_indices(kv.kept_token_indices[-1])


BUDGET_2 = 8
WINDOW_2 = 3


def _build_two_event_harness(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    harness = _TwoEventHarness(budget=BUDGET_2, window=WINDOW_2, num_heads=2, head_dim=6, seed=100)
    kv = FakeR1KV(budget=BUDGET_2, window_size=WINDOW_2, record_kept_token_indices=True)
    sink = []
    event1_record = harness.prefill(kv, sink, n_tokens=10)  # 10 >= budget(8) -> event 1 fires immediately
    assert event1_record.had_compaction is True
    return harness, kv, sink, event1_record


def test_multi_event_first_and_second_event_absolute_parity(monkeypatch):
    harness, kv, sink, event1_record = _build_two_event_harness(monkeypatch)

    # Event 1's pre-event map is the fresh prefill append -- identity by
    # construction (no eviction has happened yet).
    assert torch.equal(event1_record.pre_event_absolute_position_map, _identity_position_map(10))
    assert event1_record.observed_kept_indices_parity_passed is True

    # Post-event-1 provenance is now the (per-head, order-shuffling)
    # survivor selection -- provably non-identity.
    post_event1_positions = harness.provenance.positions.clone()
    assert not torch.equal(post_event1_positions, _identity_position_map(post_event1_positions.shape[1]))

    # Drive to event 2: one appended token pushes length to budget(8)+1=9 >= 8.
    event2_record = harness.append_and_step(kv, sink, absolute_position=10)
    assert event2_record.had_compaction is True
    assert event2_record.observed_kept_indices_parity_passed is True
    assert torch.equal(event2_record.recomputed_kept_absolute_positions, event2_record.observed_kept_absolute_positions)

    # Event 2's pre-event map itself is non-identity (its first BUDGET_2
    # columns come straight from event 1's shuffled survivor selection).
    assert not torch.equal(
        event2_record.pre_event_absolute_position_map,
        torch.arange(event2_record.pre_event_absolute_position_map.shape[1]).unsqueeze(0).expand(2, -1),
    )
    assert len(sink) == 2


def test_same_set_wrong_order_fails_ordered_parity(monkeypatch):
    # capture.py's OWN independent recomputation (from key/query states plus
    # the pre-event position map) is left correct; only R-KV's real,
    # observed `kept_token_indices[-1]` bookkeeping is corrupted, by
    # swapping two of its columns -- same SET of absolute positions per
    # head, different order. If parity used set equality this would still
    # "pass"; ordered `torch.equal` must fail it.
    install_fake_rkv_compression_module(monkeypatch)

    class SwappedOrderR1KV(FakeR1KV):
        def update_kv(self, key_states, query_states, value_states):
            k, v = super().update_kv(key_states, query_states, value_states)
            if self.record_kept_token_indices and self.kept_token_indices:
                last = self.kept_token_indices[-1].clone()
                last[:, -1], last[:, -2] = last[:, -2].clone(), last[:, -1].clone()
                self.kept_token_indices[-1] = last
            return k, v

    kv = SwappedOrderR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=True)
    key_states, value_states, query_states = _make_tensors(seed=20)

    sink = []
    with capture_update_kv(kv, sink, pre_event_position_map_fn=lambda: _identity_position_map(SEQ_LEN)):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.observed_kept_indices_parity_passed is False
    assert "observed_kept_indices_parity_failed" in record.parity_failure_reason
    # Same SET, confirming this is genuinely an ordering failure, not a
    # missing/extra-value failure.
    for h in range(NUM_HEADS):
        assert set(record.recomputed_kept_absolute_positions[h].tolist()) == set(
            record.observed_kept_absolute_positions[h].tolist()
        )


def test_one_wrong_absolute_position_fails(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)

    class OneWrongPositionR1KV(FakeR1KV):
        def update_kv(self, key_states, query_states, value_states):
            k, v = super().update_kv(key_states, query_states, value_states)
            if self.record_kept_token_indices and self.kept_token_indices:
                last = self.kept_token_indices[-1].clone()
                last[0, 0] = last[0, 0] + 999  # one wrong absolute position, head 0
                self.kept_token_indices[-1] = last
            return k, v

    kv = OneWrongPositionR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=True)
    key_states, value_states, query_states = _make_tensors(seed=21)

    sink = []
    with capture_update_kv(kv, sink, pre_event_position_map_fn=lambda: _identity_position_map(SEQ_LEN)):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.observed_kept_indices_parity_passed is False
    assert "observed_kept_indices_parity_failed" in record.parity_failure_reason


def test_missing_pre_event_map_is_a_hard_failure(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=True)
    key_states, value_states, query_states = _make_tensors(seed=11)

    sink = []
    with capture_update_kv(kv, sink):  # no pre_event_position_map_fn supplied at all
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.had_compaction is True
    assert record.observed_kept_indices_parity_passed is False
    assert record.parity_check_passed is False
    assert "missing_pre_event_absolute_position_map" in record.parity_failure_reason


def test_pre_event_map_shape_mismatch_is_a_hard_failure(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=True)
    key_states, value_states, query_states = _make_tensors(seed=12)

    wrong_shape_map = torch.arange(SEQ_LEN - 1).unsqueeze(0).expand(NUM_HEADS, -1).clone()  # wrong length
    sink = []
    with capture_update_kv(kv, sink, pre_event_position_map_fn=lambda: wrong_shape_map):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.observed_kept_indices_parity_passed is False
    assert "shape_mismatch" in record.parity_failure_reason


def test_pre_event_position_map_and_kept_positions_are_clones_not_aliases(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=True)
    key_states, value_states, query_states = _make_tensors(seed=13)

    live_map = _identity_position_map(SEQ_LEN)
    sink = []
    with capture_update_kv(kv, sink, pre_event_position_map_fn=lambda: live_map):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    before = record.pre_event_absolute_position_map.clone()
    live_map.fill_(-1)  # mutate the live map the caller still holds
    assert torch.equal(record.pre_event_absolute_position_map, before)  # unaffected

    observed_before = record.observed_kept_absolute_positions.clone()
    kv.kept_token_indices[-1].fill_(-1)  # mutate R-KV's own live bookkeeping
    assert torch.equal(record.observed_kept_absolute_positions, observed_before)  # unaffected


def test_no_compaction_call_parity_remains_non_applicable_even_with_map_fn(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=True)
    key_states, value_states, query_states = _make_tensors(seed=14, seq_len=6)  # < budget

    sink = []
    with capture_update_kv(kv, sink, pre_event_position_map_fn=lambda: _identity_position_map(6)):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.had_compaction is False
    assert record.observed_kept_indices_parity_passed is None
    assert record.parity_check_passed is True


def test_bookkeeping_unavailable_parity_stays_none_not_a_hard_failure(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=False)
    key_states, value_states, query_states = _make_tensors(seed=15)

    sink = []
    with capture_update_kv(kv, sink):  # no map, but no bookkeeping either -- genuinely not evaluable
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.had_compaction is True
    assert record.observed_kept_indices_parity_passed is None
    assert record.parity_check_passed is True


# --------------------------------------------------------------------------
# B1B-R2 Blocker: absolute-position device/dtype parity
# (`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md`)
# --------------------------------------------------------------------------


def test_recomputed_kept_physical_indices_normalizes_dtype_and_device():
    """`recomputed_topk_indices` arrives in whatever dtype `.topk()` produces
    (int64) but this test deliberately feeds a DIFFERENT dtype (int32) than
    the target provenance-map dtype (int64) to prove normalization actually
    happens rather than accidentally already matching."""
    num_heads = 2
    kv_cache_len = 10
    window_size = 3
    topk_indices = torch.tensor([[[0, 2, 4], [1, 3, 5]]], dtype=torch.int32)  # shape (1, heads, k)

    target_dtype = torch.int64
    target_device = torch.device("cpu")
    result = _recomputed_kept_physical_indices(
        topk_indices, kv_cache_len, window_size, device=target_device, dtype=target_dtype
    )

    assert result.dtype == target_dtype
    assert result.device == target_device
    # order preserved: recomputed top-k first, then the protected recent window
    expected = torch.tensor([[0, 2, 4, 7, 8, 9], [1, 3, 5, 7, 8, 9]], dtype=target_dtype)
    assert torch.equal(result, expected)


def test_recomputed_kept_physical_indices_preserves_shape_and_ordering_when_already_matching():
    num_heads = 2
    kv_cache_len = 12
    window_size = 4
    topk_indices = torch.tensor([[[3, 1], [0, 2]]], dtype=torch.int64)
    result = _recomputed_kept_physical_indices(
        topk_indices, kv_cache_len, window_size, device=torch.device("cpu"), dtype=torch.int64
    )
    assert tuple(result.shape) == (num_heads, 2 + window_size)
    expected = torch.tensor([[3, 1, 8, 9, 10, 11], [0, 2, 8, 9, 10, 11]], dtype=torch.int64)
    assert torch.equal(result, expected)


@pytest.mark.gpu
def test_recomputed_kept_physical_indices_cuda_topk_cpu_provenance_matches_cpu_reference():
    """Mechanical device-placement test (§3, B1B-R2): top-k indices
    originate on CUDA, the provenance map stays on CPU -- normalization and
    gather must complete without a device-mismatch error, and the resulting
    absolute positions must match a pure-CPU reference computation exactly.
    Skips cleanly (repo's existing `gpu` marker convention,
    `tests/conftest.py`) whenever CUDA is unavailable -- never executed on
    this CPU-only build machine."""
    kv_cache_len = 10
    window_size = 3
    topk_indices_cpu = torch.tensor([[[0, 2, 4], [1, 3, 5]]], dtype=torch.int64)
    topk_indices_cuda = topk_indices_cpu.to("cuda")
    provenance_map = torch.arange(kv_cache_len, dtype=torch.int64).unsqueeze(0).expand(2, -1).clone()  # CPU

    result = _recomputed_kept_physical_indices(
        topk_indices_cuda, kv_cache_len, window_size, device=provenance_map.device, dtype=provenance_map.dtype
    )
    assert result.device == provenance_map.device

    reference = _recomputed_kept_physical_indices(
        topk_indices_cpu, kv_cache_len, window_size, device=provenance_map.device, dtype=provenance_map.dtype
    )
    assert torch.equal(result, reference)

    gathered = provenance_map.gather(dim=-1, index=result)
    assert gathered.device == provenance_map.device


# --------------------------------------------------------------------------
# B1B-R2 §4: target-only, memory-bounded capture
# --------------------------------------------------------------------------


def test_should_capture_false_skips_clone_and_capture_entirely(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    # seq_len < BUDGET -> FakeR1KV.update_kv itself takes the early-return
    # branch and never calls .clone() internally, so any clone call
    # observed below is attributable ONLY to the capture wrapper.
    key_states, value_states, query_states = _make_tensors(seed=30, seq_len=BUDGET - 1)

    clone_calls = {"count": 0}
    original_clone = torch.Tensor.clone

    def counting_clone(self, *args, **kwargs):
        clone_calls["count"] += 1
        return original_clone(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", counting_clone)

    sink = []
    with capture_update_kv(
        kv, sink, layer_idx=0, current_position_fn=lambda: 0, should_capture=lambda pos, layer: False
    ):
        k_out, v_out = kv.update_kv(key_states, query_states, value_states)

    assert len(sink) == 0
    assert clone_calls["count"] == 0
    # original method's behavior is unchanged: below-budget FakeR1KV.update_kv
    # returns its inputs verbatim (same values, no compression).
    assert torch.equal(k_out, key_states)
    assert torch.equal(v_out, value_states)


def test_should_capture_true_still_captures_exactly_as_before(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=31)

    sink = []
    with capture_update_kv(
        kv, sink, layer_idx=2, current_position_fn=lambda: 7, should_capture=lambda pos, layer: (pos, layer) == (7, 2)
    ):
        kv.update_kv(key_states, query_states, value_states)

    assert len(sink) == 1
    assert sink[0].had_compaction is True


def test_should_capture_selects_exactly_the_target_event_layer_pairs(monkeypatch):
    """Drives several calls across different (position, layer) pairs and
    proves the sink only ever grows for the exact preselected targets --
    never for any other call, regardless of how many non-target calls
    happen in between."""
    install_fake_rkv_compression_module(monkeypatch)
    targets = {(5, 0), (12, 1)}

    def should_capture(pos, layer):
        return (pos, layer) in targets

    current_position = {"pos": 0}
    kv_by_layer = {0: FakeR1KV(budget=BUDGET, window_size=WINDOW), 1: FakeR1KV(budget=BUDGET, window_size=WINDOW)}
    sink = []
    with contextlib.ExitStack() as stack:
        for layer_idx, kv in kv_by_layer.items():
            stack.enter_context(
                capture_update_kv(
                    kv, sink, layer_idx=layer_idx, current_position_fn=lambda: current_position["pos"],
                    should_capture=should_capture,
                )
            )
        for pos in range(15):
            current_position["pos"] = pos
            for layer_idx, kv in kv_by_layer.items():
                key_states, value_states, query_states = _make_tensors(seed=100 + pos * 2 + layer_idx)
                kv.update_kv(key_states, query_states, value_states)

    # Exactly 2 records captured -- the target-count bound, never the
    # 15 positions x 2 layers = 30 total calls that actually happened.
    assert len(sink) == len(targets) == 2


def test_repeated_non_target_calls_do_not_grow_retained_state(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    sink = []
    with capture_update_kv(
        kv, sink, layer_idx=0, current_position_fn=lambda: 0, should_capture=lambda pos, layer: False
    ):
        for i in range(50):
            key_states, value_states, query_states = _make_tensors(seed=200 + i)
            kv.update_kv(key_states, query_states, value_states)
    assert len(sink) == 0


def test_should_capture_none_preserves_capture_everything_default(monkeypatch):
    """Backward compatibility: omitting `should_capture` entirely (the
    default `None`) must behave EXACTLY as before this section's addition
    -- every real call captured, unconditionally."""
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=32)

    sink = []
    with capture_update_kv(kv, sink):
        kv.update_kv(key_states, query_states, value_states)

    assert len(sink) == 1
