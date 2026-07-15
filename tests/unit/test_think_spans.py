from kvcot.probes.early_answering import (
    find_subsequence,
    find_think_span,
    compute_cut_index,
)

OPEN = [151648]  # <think>, single token for the pinned tokenizer revision
CLOSE = [151649]  # </think>
# Some tests use a multi-token marker to prove the parser doesn't assume a
# single-token marker (§3.5).
OPEN_MULTI = [10, 11]
CLOSE_MULTI = [20, 21]


def test_find_subsequence_basic():
    assert find_subsequence([1, 2, 3, 4, 5], [3, 4]) == 2
    assert find_subsequence([1, 2, 3], [9]) is None
    assert find_subsequence([1, 2, 3], []) == 0


def test_find_subsequence_respects_start():
    assert find_subsequence([1, 1, 1], [1], start=1) == 1


def test_preopened_prompt_generation_starts_inside_think():
    # Prompt ends with ...<｜Assistant｜> <think> \n  (the actual empirical
    # shape for this model/template, confirmed in docs/PROBE_PROTOCOL.md).
    prompt = [10, 20, 30, 151645, 151648, 198]
    generated = [100, 101, 102, 151649, 200, 201]  # thinks for 3 tokens then closes
    result = find_think_span(prompt, generated, OPEN, CLOSE)
    assert result.generation_prompt_preopened_think is True
    assert result.think_start_index == 0
    assert result.think_end_index == 3
    assert result.think_token_count == 3
    assert result.think_parse_status == "generation_prompt_preopened_ok"


def test_preopened_prompt_never_closes_within_generation():
    prompt = [151648, 198]
    generated = [1, 2, 3, 4]  # no close marker anywhere
    result = find_think_span(prompt, generated, OPEN, CLOSE)
    assert result.generation_prompt_preopened_think is True
    assert result.think_end_index is None
    assert result.think_parse_status == "no_close_marker"
    assert result.think_token_count == 0  # not trustworthy when status != ok


def test_non_preopened_prompt_model_opens_and_closes_itself():
    prompt = [10, 20, 30]  # no <think> at the tail
    generated = [151648, 198, 1, 2, 3, 151649, 4, 5]
    result = find_think_span(prompt, generated, OPEN, CLOSE)
    assert result.generation_prompt_preopened_think is False
    assert result.think_start_index == 1  # right after the 1-token open marker at index 0
    assert result.think_end_index == 5
    assert result.think_token_count == 4
    assert result.think_parse_status == "ok"


def test_non_preopened_and_never_opens():
    prompt = [10, 20, 30]
    generated = [1, 2, 3, 4]
    result = find_think_span(prompt, generated, OPEN, CLOSE)
    assert result.think_parse_status == "no_open_marker"
    assert result.think_start_index is None
    assert result.think_end_index is None


def test_multi_token_markers_not_assumed_single_token():
    prompt = [1, 2, 10, 11]  # ends with the 2-token open marker
    generated = [100, 101, 20, 21, 200]
    result = find_think_span(prompt, generated, OPEN_MULTI, CLOSE_MULTI)
    assert result.generation_prompt_preopened_think is True
    assert result.think_end_index == 2  # index where the 2-token close marker begins
    assert result.think_token_count == 2


def test_open_marker_earlier_in_prompt_but_closed_before_end_is_not_preopened():
    # open marker appears mid-prompt but is properly closed before the
    # prompt ends -> generation prompt does NOT end inside an open block.
    prompt = [151648, 198, 1, 2, 151649, 3]
    generated = [151648, 198, 9, 151649]
    result = find_think_span(prompt, generated, OPEN, CLOSE)
    assert result.generation_prompt_preopened_think is False
    assert result.think_parse_status == "ok"


def test_compute_cut_index_zero_and_full():
    assert compute_cut_index(100, 0.0) == 0
    assert compute_cut_index(100, 1.0) == 100


def test_compute_cut_index_floors():
    assert compute_cut_index(101, 0.125) == 12  # floor(12.625)
    assert compute_cut_index(7, 0.5) == 3  # floor(3.5)


def test_compute_cut_index_rejects_out_of_range_fraction():
    import pytest

    with pytest.raises(ValueError):
        compute_cut_index(10, 1.5)
    with pytest.raises(ValueError):
        compute_cut_index(10, -0.1)
