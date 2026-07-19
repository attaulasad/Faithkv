import torch

from kvcot.discovery.capture import capture_update_kv

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


def test_wrapper_captures_gather_parity_on_real_compaction(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW)
    key_states, value_states, query_states = _make_tensors(seed=1)

    sink = []
    with capture_update_kv(kv, sink):
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


def test_observed_kept_indices_parity_passes_at_first_event(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    kv = FakeR1KV(budget=BUDGET, window_size=WINDOW, record_kept_token_indices=True)
    key_states, value_states, query_states = _make_tensors(seed=8)

    sink = []
    with capture_update_kv(kv, sink):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.observed_kept_indices_parity_passed is True


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
    with capture_update_kv(kv, sink):
        kv.update_kv(key_states, query_states, value_states)

    record = sink[0]
    assert record.gather_parity_passed is False
    assert record.parity_check_passed is False
    assert "gather_parity_failed" in record.parity_failure_reason


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
