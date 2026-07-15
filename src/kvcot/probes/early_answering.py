"""Think-span parsing and early-answering probe mechanics (§3.5, §6, §8.1).

CLAIM BOUNDARY (§1) — repeated here because every consumer of this module's
output (records, summaries, plots, reports) must carry it:

  Allowed conclusion: lower sensitivity to truncating visible reasoning
  under R-KV.

  Forbidden conclusions: that the chain is fake, decorative, unfaithful to
  the model's "true thoughts," or that we observed internal cognition. This
  measures counterfactual behavioral dependence on visible generated
  tokens. Nothing else.

Think delimiters are token-ID subsequences, never assumed to be a single
token (§3.5) — even though, empirically, `<think>`/`</think>` happen to
each be exactly one token (151648 / 151649) for this pinned tokenizer
revision (see docs/PROBE_PROTOCOL.md), the parsing logic here treats them as
general subsequences so it stays correct if that ever changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

CLAIM_BOUNDARY_NOTICE = (
    "This measures counterfactual behavioral dependence on the visible, "
    "generated chain-of-thought under truncation — NOT internal faithfulness, "
    "NOT whether reasoning is 'real', and NOT a claim about the model's "
    "internal cognition. See docs/EXPERIMENT.md and build brief §1."
)


def find_subsequence(haystack: Sequence[int], needle: Sequence[int], start: int = 0) -> int | None:
    """First index >= start at which `needle` occurs as a contiguous
    subsequence of `haystack`, or None. O(len(haystack) * len(needle))
    worst case, which is fine here since needle is a handful of tokens."""
    n = len(needle)
    if n == 0:
        return start if start <= len(haystack) else None
    last_start = len(haystack) - n
    for i in range(max(start, 0), last_start + 1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return None


def _last_subsequence_index(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    n = len(needle)
    if n == 0 or len(haystack) < n:
        return None
    for i in range(len(haystack) - n, -1, -1):
        if list(haystack[i : i + n]) == list(needle):
            return i
    return None


def _ends_inside_open_block(
    token_ids: Sequence[int], open_marker: Sequence[int], close_marker: Sequence[int]
) -> bool:
    last_open = _last_subsequence_index(token_ids, open_marker)
    if last_open is None:
        return False
    close_after = find_subsequence(token_ids, close_marker, start=last_open + len(open_marker))
    return close_after is None


@dataclass(frozen=True)
class ThinkSpanResult:
    think_start_index: int | None  # index into generated_token_ids, inclusive
    think_end_index: int | None  # index into generated_token_ids, exclusive (start of the close-marker subsequence)
    think_parse_status: str
    generation_prompt_preopened_think: bool

    @property
    def think_token_count(self) -> int:
        """L: number of generated tokens strictly inside the think span.
        Only meaningful when think_parse_status indicates success — callers
        must check status before trusting this for eligibility (§8.3)."""
        if self.think_start_index is None or self.think_end_index is None:
            return 0
        return max(0, self.think_end_index - self.think_start_index)


def find_think_span(
    prompt_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    open_marker_ids: Sequence[int],
    close_marker_ids: Sequence[int],
) -> ThinkSpanResult:
    """Determine the think span over `generated_token_ids`.

    The chat template's generation prompt may already end inside an open
    think block (confirmed true for this model/tokenizer/template — see
    docs/PROBE_PROTOCOL.md — but detected here from the actual prompt
    tokens, never assumed). In that case the entire generated sequence
    starts already "inside" thinking (think_start_index = 0), and only the
    close marker needs to be found among the generated tokens.
    """
    preopened = _ends_inside_open_block(prompt_token_ids, open_marker_ids, close_marker_ids)

    if preopened:
        think_start = 0
        close_idx = find_subsequence(generated_token_ids, close_marker_ids, start=0)
        if close_idx is None:
            return ThinkSpanResult(
                think_start_index=think_start,
                think_end_index=None,
                think_parse_status="no_close_marker",
                generation_prompt_preopened_think=True,
            )
        return ThinkSpanResult(
            think_start_index=think_start,
            think_end_index=close_idx,
            think_parse_status="generation_prompt_preopened_ok",
            generation_prompt_preopened_think=True,
        )

    open_idx = find_subsequence(generated_token_ids, open_marker_ids, start=0)
    if open_idx is None:
        return ThinkSpanResult(
            think_start_index=None,
            think_end_index=None,
            think_parse_status="no_open_marker",
            generation_prompt_preopened_think=False,
        )
    think_start = open_idx + len(open_marker_ids)
    close_idx = find_subsequence(generated_token_ids, close_marker_ids, start=think_start)
    if close_idx is None:
        return ThinkSpanResult(
            think_start_index=think_start,
            think_end_index=None,
            think_parse_status="no_close_marker",
            generation_prompt_preopened_think=False,
        )
    return ThinkSpanResult(
        think_start_index=think_start,
        think_end_index=close_idx,
        think_parse_status="ok",
        generation_prompt_preopened_think=False,
    )


def compute_cut_index(think_token_count: int, fraction: float) -> int:
    """floor(fraction * L), per §6 step 7. fraction=0.0 is the state just
    before the first thinking token (cut_index == 0)."""
    if think_token_count < 0:
        raise ValueError("think_token_count must be >= 0")
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    return math.floor(fraction * think_token_count)


def absolute_cut_position(think_span: ThinkSpanResult, fraction: float) -> int:
    """Index into `generated_token_ids` (not relative to the think span) at
    which to snapshot for a given probe fraction: `think_start_index +
    floor(fraction * L)`. f=0.0 -> exactly `think_start_index` (the state
    just before the first thinking token, §6 step 7). f=1.0 -> exactly
    `think_end_index` (the full think span, kept in its entirety).

    Raises if the think span failed to parse — callers must check
    `think_span.think_parse_status` before calling this (an unparsed span
    has no valid cut position, and this repository never guesses one, per
    §3.5: "Record think_parse_status; never guess.").
    """
    if think_span.think_start_index is None:
        raise ValueError("cannot compute a cut position for an unparsed think span")
    cut = compute_cut_index(think_span.think_token_count, fraction)
    return think_span.think_start_index + cut


def truncate_generated_tokens(
    generated_token_ids: Sequence[int], think_span: ThinkSpanResult, fraction: float
) -> list[int]:
    """The prefix of `generated_token_ids` that would be teacher-forced
    during replay before branching at this probe fraction — i.e. everything
    up to (not including) the tokens dropped by truncation. This is a pure
    function over already-generated tokens for testing the truncation
    arithmetic in isolation; the real replay path
    (`kvcot.generation.replay`) performs the equivalent cut against live
    model/cache state, not against a plain Python list.
    """
    cut = absolute_cut_position(think_span, fraction)
    return list(generated_token_ids[:cut])
