import pytest
import torch

from kvcot.discovery.branch_eval import (
    SCORED_HORIZON,
    assert_timing_invariants,
    evaluate_branch,
    evaluate_swap_branches,
)

VOCAB_SIZE = 9
HIDDEN = 5

_g = torch.Generator().manual_seed(1234)
_EMBED = torch.randn(VOCAB_SIZE, HIDDEN, generator=_g)
_OUT = torch.randn(HIDDEN, VOCAB_SIZE, generator=_g)


def toy_step_fn(hidden_state: torch.Tensor, token_id: int):
    """Deterministic toy causal step: no real model, no download. Same
    (hidden_state, token_id) always produces the same (logits, new_hidden)."""
    new_hidden = hidden_state + _EMBED[token_id]
    logits = new_hidden @ _OUT
    return logits, new_hidden


def _make_reference_tokens(seed: int, n: int = SCORED_HORIZON) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (n,), generator=g).tolist()


def test_evaluate_branch_scores_exactly_48_and_bridge_absent():
    ref_tokens = _make_reference_tokens(seed=1)
    result = evaluate_branch(toy_step_fn, torch.zeros(HIDDEN), bridge_token_id=3, reference_token_ids=ref_tokens)
    assert len(result.per_token_nll) == SCORED_HORIZON
    assert len(result.per_token_logits) == SCORED_HORIZON


def test_evaluate_branch_rejects_wrong_length_reference():
    with pytest.raises(ValueError):
        evaluate_branch(toy_step_fn, torch.zeros(HIDDEN), bridge_token_id=0, reference_token_ids=[1] * 47)
    with pytest.raises(ValueError):
        evaluate_branch(toy_step_fn, torch.zeros(HIDDEN), bridge_token_id=0, reference_token_ids=[1] * 49)


def test_evaluate_branch_never_truncates_or_pads():
    ref_tokens = _make_reference_tokens(seed=2)
    result = evaluate_branch(toy_step_fn, torch.zeros(HIDDEN), bridge_token_id=1, reference_token_ids=ref_tokens)
    assert len(result.per_token_nll) == len(ref_tokens) == SCORED_HORIZON


def test_evaluate_branch_nll_hand_computed_first_step():
    # First scored NLL must come from the logits produced by feeding the
    # BRIDGE token alone (teacher-forced), never anything else.
    ref_tokens = _make_reference_tokens(seed=3)
    bridge = 5
    result = evaluate_branch(toy_step_fn, torch.zeros(HIDDEN), bridge_token_id=bridge, reference_token_ids=ref_tokens)

    expected_logits, _ = toy_step_fn(torch.zeros(HIDDEN), bridge)
    expected_nll = -torch.log_softmax(expected_logits.float(), dim=-1)[ref_tokens[0]]
    assert result.per_token_nll[0] == pytest.approx(expected_nll.item(), abs=1e-6)


def test_branches_evolve_cache_independently():
    ref_tokens = _make_reference_tokens(seed=4)
    baseline_init = torch.zeros(HIDDEN)
    swapped_init = torch.ones(HIDDEN) * 5.0  # a genuinely different starting cache

    comparison = evaluate_swap_branches(toy_step_fn, baseline_init, swapped_init, bridge_token_id=2, reference_token_ids=ref_tokens)

    assert not torch.equal(comparison.baseline_final_cache_state, comparison.swapped_final_cache_state)
    assert comparison.baseline_per_token_nll != comparison.swapped_per_token_nll


def test_swap_gain_sign_convention():
    ref_tokens = _make_reference_tokens(seed=5)
    baseline_init = torch.zeros(HIDDEN)
    swapped_init = torch.zeros(HIDDEN)
    comparison = evaluate_swap_branches(toy_step_fn, baseline_init, swapped_init, bridge_token_id=4, reference_token_ids=ref_tokens)
    # identical starting caches -> identical everything -> swap_gain == 0
    assert comparison.swap_gain == pytest.approx(0.0, abs=1e-9)
    assert comparison.swap_gain == pytest.approx(comparison.baseline_mean_nll - comparison.swapped_mean_nll, abs=1e-9)


# ---------------------------------------------------------------------------
# Mandatory strengthened no-op control (Part IX.20): complete branch-output
# comparison, not just a single-tensor self-assignment check.
# ---------------------------------------------------------------------------


def test_noop_control_complete_branch_output_equality():
    ref_tokens = _make_reference_tokens(seed=42)
    bridge_token_id = 7
    # A no-op swap (e := r) means the swapped cache is bit-identical to the
    # baseline cache before any branch evaluation begins.
    baseline_init = torch.randn(HIDDEN, generator=torch.Generator().manual_seed(99))
    swapped_init = baseline_init.clone()

    comparison = evaluate_swap_branches(
        toy_step_fn, baseline_init, swapped_init, bridge_token_id=bridge_token_id, reference_token_ids=ref_tokens
    )

    # torch.equal(baseline logits at every step, no-op logits at every step)
    assert len(comparison.baseline_per_token_logits) == len(comparison.swapped_per_token_logits) == SCORED_HORIZON
    for baseline_logits, swapped_logits in zip(comparison.baseline_per_token_logits, comparison.swapped_per_token_logits):
        assert torch.equal(baseline_logits, swapped_logits)

    # torch.equal(baseline per-token NLL tensor, no-op per-token NLL tensor)
    baseline_nll_tensor = torch.tensor(comparison.baseline_per_token_nll)
    swapped_nll_tensor = torch.tensor(comparison.swapped_per_token_nll)
    assert torch.equal(baseline_nll_tensor, swapped_nll_tensor)

    # baseline_per_token_nll list == no-op_per_token_nll list
    assert comparison.baseline_per_token_nll == comparison.swapped_per_token_nll

    # baseline_mean_nll == no-op_mean_nll
    assert comparison.baseline_mean_nll == comparison.swapped_mean_nll

    # swap_gain == 0.0
    assert comparison.swap_gain == 0.0

    # final cache states are bit-exact
    assert torch.equal(comparison.baseline_final_cache_state, comparison.swapped_final_cache_state)

    # both outputs contain exactly 48 scored NLLs
    assert len(comparison.baseline_per_token_nll) == SCORED_HORIZON
    assert len(comparison.swapped_per_token_nll) == SCORED_HORIZON


def test_timing_invariants_accept_correct_t_plus_one_t_plus_two():
    assert_timing_invariants(
        event_token_absolute_position=100, bridge_token_absolute_position=101, first_scored_absolute_position=102
    )


@pytest.mark.parametrize(
    "event,bridge,first_scored",
    [
        (100, 102, 103),  # bridge != event + 1
        (100, 101, 101),  # first_scored != bridge + 1
        (100, 101, 103),  # first_scored skips a position
    ],
)
def test_timing_invariants_reject_broken_offsets(event, bridge, first_scored):
    with pytest.raises(ValueError):
        assert_timing_invariants(event, bridge, first_scored)
