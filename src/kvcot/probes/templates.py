"""The single source of truth for the base user prompt and the early-
answering control suffix (§7 of the build brief). Every other module —
generation, replay, docs generation, tests — imports these constants rather
than re-typing the strings. Never duplicate these in configs or scripts.

`BASE_USER_TEMPLATE` is verbatim (not paraphrased) from the pinned upstream
source, `HuggingFace/run_math.py:35` at commit
45eaa7d69d20b7388321f077020a610d9afb65bd (see docs/UPSTREAM_AUDIT.md §5):
byte-identical, including the leading spaces before "You need to..." /
"Provide the final answer" and the double space before `\\boxed{{}}`. This
repository calls it "official HF runner wording" and that means upstream's
literal bytes, not a cleaned-up paraphrase.
"""
from __future__ import annotations

BASE_USER_TEMPLATE = (
    "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the "
    "problem step by step. First, you need to provide the chain-of-thought, then provide "
    "the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"
)


def render_base_user_message(question: str) -> str:
    """Fill in the one frozen template. `question` is inserted verbatim
    (no stripping/normalizing) — any whitespace irregularity in the source
    dataset row is part of what gets hashed and reproduced exactly."""
    return BASE_USER_TEMPLATE.format(question=question)


# Fed AFTER the closing-think token sequence has been teacher-forced onto a
# branched snapshot (§6, step 8). This text does NOT itself contain the
# `</think>` marker — the exact closing-marker token id subsequence is
# resolved from the live tokenizer (never assumed to be a single token; see
# kvcot.probes.early_answering.find_think_span) and injected as tokens
# separately, before this suffix is tokenized and fed.
#
# Requirements this string must satisfy (§7): close over the think block
# exactly once (achieved structurally — this text contains no think tags of
# its own, so it cannot reopen or double-close one); ask for only the final
# answer, in the same format as the base prompt; never ask for more
# reasoning; be coherent whether branched from f=0 (almost no visible
# reasoning happened yet) or f=1 (the full natural chain already happened —
# this is the stability control, and this suffix must reproduce the model's
# own already-given answer here).
CONTROL_SUFFIX_TEXT = (
    "\n\nStop reasoning now. Based on everything above, give only the final "
    "answer, in the format: Final answer: \\boxed{}"
)


def render_control_suffix() -> str:
    return CONTROL_SUFFIX_TEXT


# Used only by the secondary, additive fixed-trace probe
# (kvcot.cli.cmd_replay_fixed_trace / kvcot.analysis.fixed_trace) — never by
# the frozen replay-probe/EAS pipeline above, which always uses
# CONTROL_SUFFIX_TEXT.
#
# Protocol v2 (2026-07-16, CHANGELOG.md): the original design left this
# empty, relying on the closing `</think>` marker alone to start greedy
# answer decoding in the model's normal answer mode. In practice R1-Distill's
# answer mode is a verbose structured write-up that essentially never reaches
# a `\boxed{...}` (or even `Final answer:`) within the frozen 48-token probe
# budget, so extraction fell through to the conservative final-number
# fallback on nearly every probe — an incidental mid-sentence number, not an
# intended answer. That silently contaminated every fixed-trace match/PSS
# value (kvcot.utils.answers.extract_answer's fallback tier is documented as
# "conservative", not "reliable enough to anchor a metric on").
#
# The fix is a teacher-forced FORMAT prefix, not a recomputation instruction:
# "\n\nFinal answer: \\boxed{" forces the model directly into the box, with
# no natural-language content that could cue it to re-derive the answer from
# the question rather than continue from the (possibly-compressed) reasoning
# prefix actually in its cache. This is identical across conditions (fed as
# plain teacher-forced tokens before probe decoding begins, exactly like the
# closing marker), so it cannot itself introduce a policy-dependent confound.
# Never add instructions such as "solve again" / "recalculate" / "use the
# question" / "explain your answer" here — those would encourage
# recomputation, defeating the fixed-trace design's whole point (prefix
# sufficiency under a SHARED reasoning prefix, kvcot.analysis.fixed_trace's
# module docstring).
FIXED_TRACE_SUFFIX_TEXT = "\n\nFinal answer: \\boxed{"


def render_fixed_trace_suffix() -> str:
    """Teacher-forced format prefix only (see FIXED_TRACE_SUFFIX_TEXT) — no
    natural-language instruction. The replay branch feeds the model's native
    </think> marker, then this prefix, then allows greedy answer decoding to
    continue writing the box's contents (kvcot.cli.cmd_replay_fixed_trace
    reconstructs the full answer text as this prefix + the generated tokens,
    kvcot.utils.answers.has_complete_boxed_answer/extract_answer operate on
    that reconstructed text, never on the generated tokens alone)."""
    return FIXED_TRACE_SUFFIX_TEXT
