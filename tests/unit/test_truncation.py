import pytest

from kvcot.probes.early_answering import (
    find_think_span,
    absolute_cut_position,
    truncate_generated_tokens,
)
from kvcot.config import PROBE_FRACTIONS_ALL, PROBE_FRACTIONS_SCORED

OPEN = [151648]
CLOSE = [151649]

# A generation-prompt-preopened trace: 40 think tokens then close, then a
# short "final answer" tail. Think tokens are 1000..1039, closer is CLOSE,
# tail is 2000..2009.
PROMPT = [1, 2, 3, 151648, 198]
THINK_TOKENS = list(range(1000, 1040))  # 40 tokens
TAIL = list(range(2000, 2010))
GENERATED = THINK_TOKENS + CLOSE + TAIL


@pytest.fixture
def span():
    return find_think_span(PROMPT, GENERATED, OPEN, CLOSE)


def test_span_fixture_sane(span):
    assert span.think_parse_status == "generation_prompt_preopened_ok"
    assert span.think_start_index == 0
    assert span.think_end_index == 40
    assert span.think_token_count == 40


def test_f0_cut_is_exactly_think_start(span):
    pos = absolute_cut_position(span, 0.0)
    assert pos == span.think_start_index == 0
    truncated = truncate_generated_tokens(GENERATED, span, 0.0)
    assert truncated == []  # zero think tokens kept


def test_f1_cut_is_exactly_think_end(span):
    pos = absolute_cut_position(span, 1.0)
    assert pos == span.think_end_index == 40
    truncated = truncate_generated_tokens(GENERATED, span, 1.0)
    assert truncated == THINK_TOKENS  # the full think span, nothing from the tail


def test_all_frozen_fractions_produce_monotonically_nondecreasing_cuts(span):
    positions = [absolute_cut_position(span, f) for f in PROBE_FRACTIONS_ALL]
    assert positions == sorted(positions)
    assert positions[0] == 0
    assert positions[-1] == 40


def test_scored_fractions_are_strictly_between_f0_and_f1_cuts(span):
    f0_pos = absolute_cut_position(span, 0.0)
    f1_pos = absolute_cut_position(span, 1.0)
    for f in PROBE_FRACTIONS_SCORED:
        pos = absolute_cut_position(span, f)
        assert f0_pos < pos < f1_pos, f"fraction {f} produced boundary cut {pos}"


def test_truncated_tokens_are_always_a_prefix_of_the_think_span(span):
    for f in PROBE_FRACTIONS_ALL:
        truncated = truncate_generated_tokens(GENERATED, span, f)
        assert truncated == THINK_TOKENS[: len(truncated)]


def test_cannot_compute_cut_position_for_unparsed_span():
    bad_span = find_think_span([1, 2, 3], [1, 2, 3], OPEN, CLOSE)  # never opens
    assert bad_span.think_parse_status == "no_open_marker"
    with pytest.raises(ValueError):
        absolute_cut_position(bad_span, 0.5)


def test_known_fraction_produces_known_cut_count(span):
    # L=40, fraction=0.375 -> floor(15.0) = 15
    pos = absolute_cut_position(span, 0.375)
    assert pos - span.think_start_index == 15
