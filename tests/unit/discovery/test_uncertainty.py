import math

import pytest
import torch

from kvcot.discovery.uncertainty import (
    NON_FINITE_MISSING_REASON,
    POSITION_ZERO_MISSING_REASON,
    UncertaintySignal,
    compute_entropy_nats,
    compute_logit_margin,
    compute_pair_uncertainty_signals,
    resolve_prediction_logit_source,
)


def test_uncertainty_signal_requires_exactly_one_of_value_or_reason():
    UncertaintySignal(value=1.0, missing_reason=None)
    UncertaintySignal(value=None, missing_reason="x")
    with pytest.raises(ValueError):
        UncertaintySignal(value=1.0, missing_reason="x")
    with pytest.raises(ValueError):
        UncertaintySignal(value=None, missing_reason=None)


def test_entropy_uniform_two_class_hand_computed():
    logits = torch.tensor([0.0, 0.0])
    result = compute_entropy_nats(logits)
    assert result.is_available
    assert result.value == pytest.approx(math.log(2), abs=1e-6)


def test_entropy_peaked_distribution_near_zero():
    logits = torch.tensor([50.0, 0.0, 0.0])
    result = compute_entropy_nats(logits)
    assert result.is_available
    assert result.value < 1e-6


def test_entropy_never_normalized_by_vocab_size():
    # Two hand-computable uniform distributions of different size must NOT
    # be rescaled to the same value -- entropy grows with support size.
    small = compute_entropy_nats(torch.zeros(2))
    large = compute_entropy_nats(torch.zeros(8))
    assert small.value == pytest.approx(math.log(2), abs=1e-6)
    assert large.value == pytest.approx(math.log(8), abs=1e-6)


def test_entropy_empty_tensor_is_missing_not_zero():
    result = compute_entropy_nats(torch.empty(0))
    assert not result.is_available
    assert result.missing_reason is not None


def test_logit_margin_hand_computed():
    logits = torch.tensor([5.0, 2.0, 1.0])
    result = compute_logit_margin(logits)
    assert result.is_available
    assert result.value == pytest.approx(3.0, abs=1e-6)


def test_logit_margin_independent_of_which_token_was_sampled():
    # Margin depends only on the top-2 raw logits, never on which token id
    # was actually generated/sampled.
    logits = torch.tensor([5.0, 2.0, 1.0])
    result = compute_logit_margin(logits)
    assert result.value == pytest.approx(3.0, abs=1e-6)


def test_logit_margin_requires_at_least_two_vocab_entries():
    result = compute_logit_margin(torch.tensor([1.0]))
    assert not result.is_available


def test_logit_margin_negative_logits_hand_computed():
    logits = torch.tensor([-1.0, -4.0, -10.0])
    result = compute_logit_margin(logits)
    assert result.value == pytest.approx(3.0, abs=1e-6)


def test_computed_in_float32_even_from_float64_input():
    logits = torch.tensor([5.0, 2.0], dtype=torch.float64)
    entropy = compute_entropy_nats(logits)
    margin = compute_logit_margin(logits)
    assert entropy.is_available
    assert margin.is_available


def test_pair_uncertainty_signals_hand_computed_diff():
    e_entropy = UncertaintySignal(value=1.2, missing_reason=None)
    r_entropy = UncertaintySignal(value=0.4, missing_reason=None)
    e_margin = UncertaintySignal(value=3.0, missing_reason=None)
    r_margin = UncertaintySignal(value=5.5, missing_reason=None)

    pair = compute_pair_uncertainty_signals(e_entropy, r_entropy, e_margin, r_margin)

    assert pair.entropy_diff == pytest.approx(0.8, abs=1e-9)
    assert pair.logit_margin_diff == pytest.approx(-2.5, abs=1e-9)
    assert pair.uncertainty_signal_source == "raw_next_token_logits_at_token_prediction_time"


def test_pair_uncertainty_signals_diff_is_none_when_either_side_missing():
    missing = UncertaintySignal(value=None, missing_reason="x")
    present = UncertaintySignal(value=1.0, missing_reason=None)

    pair_missing_e = compute_pair_uncertainty_signals(missing, present, present, present)
    assert pair_missing_e.entropy_diff is None

    pair_missing_r = compute_pair_uncertainty_signals(present, missing, present, present)
    assert pair_missing_r.entropy_diff is None


def test_resolve_prediction_logit_source_position_zero_unavailable():
    source = resolve_prediction_logit_source(absolute_position=0, prompt_length=10)
    assert source.call_kind == "unavailable"
    assert source.sequence_index is None
    assert source.missing_reason == POSITION_ZERO_MISSING_REASON


def test_resolve_prediction_logit_source_prompt_token():
    # prompt_length=10: token at position 5 is predicted by the prefill
    # call's output at sequence index 4.
    source = resolve_prediction_logit_source(absolute_position=5, prompt_length=10)
    assert source.call_kind == "prefill"
    assert source.sequence_index == 4


def test_resolve_prediction_logit_source_first_generated_token_uses_prefill_tail():
    # The first generated token (absolute position == prompt_length) is
    # predicted by the prefill call's LAST output logit.
    source = resolve_prediction_logit_source(absolute_position=10, prompt_length=10)
    assert source.call_kind == "prefill"
    assert source.sequence_index == 9


def test_resolve_prediction_logit_source_later_generated_token_uses_decode_call():
    # absolute_position=12, prompt_length=10 -> prev=11 -> decode_call_index = 11-10 = 1
    source = resolve_prediction_logit_source(absolute_position=12, prompt_length=10)
    assert source.call_kind == "decode"
    assert source.sequence_index == 1


def test_resolve_prediction_logit_source_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        resolve_prediction_logit_source(absolute_position=-1, prompt_length=10)
    with pytest.raises(ValueError):
        resolve_prediction_logit_source(absolute_position=0, prompt_length=0)


# --------------------------------------------------------------------------
# Blocker 5: tensor-shape contracts -- raw_logits.ndim must be exactly 1
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(2, 2), (1, 2, 3), ()],
    ids=["shape_2x2", "shape_1x2x3", "scalar_tensor"],
)
def test_entropy_rejects_malformed_rank(shape):
    logits = torch.zeros(shape)
    with pytest.raises(ValueError):
        compute_entropy_nats(logits)


@pytest.mark.parametrize(
    "shape",
    [(2, 2), (1, 2, 3), ()],
    ids=["shape_2x2", "shape_1x2x3", "scalar_tensor"],
)
def test_logit_margin_rejects_malformed_rank(shape):
    logits = torch.zeros(shape)
    with pytest.raises(ValueError):
        compute_logit_margin(logits)


def test_entropy_never_flattens_or_sums_a_malformed_rank_input():
    # A (2, 3) tensor superficially "looks like" it could be flattened to a
    # valid 6-vocab distribution -- it must be rejected outright, never
    # silently reshaped/summed on the caller's behalf.
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(ValueError):
        compute_entropy_nats(logits)


def test_entropy_accepts_one_element_vocabulary():
    result = compute_entropy_nats(torch.tensor([5.0]))
    assert result.is_available
    assert result.value == pytest.approx(0.0, abs=1e-6)  # single-outcome distribution -> zero entropy


def test_logit_margin_one_element_vocabulary_is_missing_not_an_error():
    result = compute_logit_margin(torch.tensor([5.0]))
    assert not result.is_available
    assert result.missing_reason is not None


def test_entropy_non_finite_result_is_missing_with_explicit_reason():
    result = compute_entropy_nats(torch.tensor([float("inf"), 0.0]))
    assert not result.is_available
    assert result.missing_reason == NON_FINITE_MISSING_REASON


def test_logit_margin_non_finite_result_is_missing_with_explicit_reason():
    result = compute_logit_margin(torch.tensor([float("inf"), float("inf"), 0.0]))
    assert not result.is_available
    assert result.missing_reason == NON_FINITE_MISSING_REASON


def test_uncertainty_signal_stores_only_cloned_scalar_never_the_logits_tensor():
    logits = torch.tensor([5.0, 2.0, 1.0])
    entropy = compute_entropy_nats(logits)
    margin = compute_logit_margin(logits)
    assert isinstance(entropy.value, float)
    assert isinstance(margin.value, float)
    # Mutating the original logits after the fact cannot retroactively
    # change an already-extracted Python float.
    logits.fill_(0.0)
    assert margin.value == pytest.approx(3.0, abs=1e-6)


def test_source_and_missing_reason_propagate_into_pair_signals():
    e_entropy = compute_entropy_nats(torch.empty(0))  # missing
    r_entropy = compute_entropy_nats(torch.tensor([1.0, 2.0]))  # present
    e_margin = compute_logit_margin(torch.tensor([1.0]))  # missing (vocab too small)
    r_margin = compute_logit_margin(torch.tensor([5.0, 2.0]))  # present

    pair = compute_pair_uncertainty_signals(e_entropy, r_entropy, e_margin, r_margin)

    assert pair.entropy_e.is_available is False
    assert pair.entropy_e.missing_reason is not None
    assert pair.entropy_r.is_available is True
    assert pair.entropy_diff is None  # one side missing
    assert pair.logit_margin_e.is_available is False
    assert pair.logit_margin_r.is_available is True
    assert pair.logit_margin_diff is None
