"""Real MATH-500 answer verification, wired into Pass 1's `AnswerFn`
contract (B1B-R3 §7). Reuses the repository's existing extraction and
symbolic-equivalence machinery directly -- `kvcot.utils.answers.extract_answer`
(boxed / final-answer-marker / conservative-number-fallback priority) and
`kvcot.utils.math_verifier.verify_math_equivalence` (subprocess-isolated
`math-verify` symbolic comparison) -- never a second, independently-written
string-equality verifier.

## Defect repaired

`kvcot.discovery.b2a_execute`'s `_answer_fn` previously labeled EVERY
natural generation `"unverifiable"` unconditionally, while
`kvcot.discovery.orchestrator.run_example` rejects every status other than
`"correct"` -- so the real execution path could never reach Pass 2, no
matter what the model actually generated. `Math500AnswerVerifier` below is
a real, three-outcome verifier: it decodes the generated tokens, extracts
the model's final answer, compares it against the frozen gold answer from
the resolved one-example manifest's dataset row, and returns exactly one of
`"correct"` / `"incorrect"` / `"unverifiable"` -- never a fourth silent
default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcot.discovery.pass1 import AnswerFn, NaturalAnswerStatus
from kvcot.utils.answers import ExtractedAnswer, extract_answer
from kvcot.utils.math_verifier import MathVerificationResult, verify_math_equivalence


@dataclass(frozen=True)
class AnswerVerificationDetail:
    """Everything `docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md` §7 requires
    preserved in the artifact: the raw decoded text, the extraction result,
    the symbolic-verification result, and the final three-way status --
    never just the status alone."""

    decoded_text: str
    extracted: ExtractedAnswer
    verification: MathVerificationResult | None  # None iff extraction itself failed -- nothing to verify against
    gold_answer: str
    status: NaturalAnswerStatus


class Math500AnswerVerifier:
    """A stateful `AnswerFn` implementation (matches
    `kvcot.discovery.pass1.AnswerFn`'s `Callable[[list[int]], tuple[str |
    None, NaturalAnswerStatus]]` contract exactly via `__call__`) that also
    records the full verification detail on `self.last_result` after being
    called, so a caller (`kvcot.discovery.b2a_execute`) can preserve it in
    the B2A artifact without widening the shared `AnswerFn` type itself."""

    def __init__(self, tokenizer: Any, gold_answer: str):
        self.tokenizer = tokenizer
        self.gold_answer = gold_answer
        self.last_result: AnswerVerificationDetail | None = None

    def __call__(self, generated_ids: list[int]) -> tuple[str | None, NaturalAnswerStatus]:
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        extracted = extract_answer(text)

        if extracted.normalized_value is None:
            detail = AnswerVerificationDetail(
                decoded_text=text, extracted=extracted, verification=None,
                gold_answer=self.gold_answer, status="unverifiable",
            )
            self.last_result = detail
            return None, "unverifiable"

        # B2A-R1 answer-verifier repair (2026-07-22): `extracted.normalized_value`
        # (the model's own last valid `\boxed{...}` content, already stripped of
        # its wrapper by `extract_answer`) and `self.gold_answer` (the bare
        # MATH-500 dataset `answer` field, e.g. `"\\left( 3, \\frac{\\pi}{2}
        # \\right)"`) are both BARE LaTeX, never wrapped in `\boxed{}`.
        # `math_verify.parse`'s non-anchored fallback extraction (used when no
        # `\boxed{}`/"final answer" anchor is present in the text handed to it)
        # is unreliable for compound expressions -- empirically, it drops an
        # ordered pair's second component entirely for one bare string and
        # returns no candidate at all for a whitespace-only variant of the
        # SAME pair, even though both sides are trivially, symbolically
        # equivalent. Re-wrapping each already-extracted final answer in its
        # own `\boxed{...}` before verification routes both through
        # `math_verify`'s well-tested, well-anchored boxed-extraction path
        # instead (confirmed double-wrap-safe: `\boxed{\boxed{x}}` normalizes
        # identically to `\boxed{x}`) -- this is a general parsing-boundary
        # fix, not special-cased to tuples or to this specific problem's
        # answer.
        verification = verify_math_equivalence(
            f"\\boxed{{{extracted.normalized_value}}}", f"\\boxed{{{self.gold_answer}}}"
        )
        if verification.is_equivalent is True:
            status: NaturalAnswerStatus = "correct"
        elif verification.is_equivalent is False:
            status = "incorrect"
        else:
            status = "unverifiable"

        detail = AnswerVerificationDetail(
            decoded_text=text, extracted=extracted, verification=verification,
            gold_answer=self.gold_answer, status=status,
        )
        self.last_result = detail
        return extracted.normalized_value, status


def build_math500_answer_fn(tokenizer: Any, gold_answer: str) -> Math500AnswerVerifier:
    return Math500AnswerVerifier(tokenizer, gold_answer)
