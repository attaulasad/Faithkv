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
# CONTROL_SUFFIX_TEXT. Deliberately empty: unlike the early-answering probe
# (which must work when branched from f=0, almost no visible reasoning yet,
# and so needs an explicit "stop and answer" instruction), the fixed-trace
# probe's whole point is prefix-sufficiency under a shared canonical trace —
# injecting "Stop reasoning now. Based on everything above..." risks cueing
# the model to recompute the answer directly from the question rather than
# from the (possibly-compressed) reasoning prefix actually in its cache,
# which would confound the two conditions' cache states with two different
# answer-elicitation strategies. The closing </think> marker alone supplies
# the next-token logits needed to start greedy answer decoding (see
# kvcot.generation.replay.branch_and_probe: it only requires the closing
# marker and suffix not both be empty).
FIXED_TRACE_SUFFIX_TEXT = ""


def render_fixed_trace_suffix() -> str:
    """No natural-language intervention. The replay branch feeds the model's
    native </think> marker and then allows greedy answer generation to begin
    in the model's normal answer mode."""
    return FIXED_TRACE_SUFFIX_TEXT
