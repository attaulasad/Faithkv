"""B2A-R1 answer-verifier repair (2026-07-22).

The consumed B2A-R1 attempt
(`results/decisions/b2a_attempt_20260722T072823470986Z_...`, preserved in
`docs/evidence/B2A_R1_ATTEMPT_INDEX_2026-07-22.json`) reported FullKV's
natural answer as `"unverifiable"` even though the model's generated answer
was mathematically identical to the frozen gold answer:

  gold (manifest.gold_answer):        \\left( 3, \\frac{\\pi}{2} \\right)
  model (fullkv.natural_answer):      \\left(3, \\frac{\\pi}{2}\\right)

Root cause (confirmed by directly exercising the installed `math-verify`
0.9.0 package, not assumed): both strings are BARE LaTeX -- neither
`extract_answer`'s stripped `\\boxed{...}` content nor the MATH-500 dataset's
raw `answer` field is ever wrapped in `\\boxed{}`. `math_verify.parse`'s
non-anchored fallback extraction (used whenever no `\\boxed{}`/"final
answer" anchor is present) is unreliable for compound expressions: it
parsed the gold string above as `[3, '3']` (silently dropping the second
tuple component) and the model string as `[]` (no candidate at all), purely
because of an incidental internal-whitespace difference the anchored path
does not care about. Wrapping the SAME two strings in `\\boxed{}` before
parsing correctly recovers `(3, pi/2)` on both sides and verifies them
equivalent.

This is a defect in `math500_verification.Math500AnswerVerifier`'s calling
convention into the existing, already-correct symbolic verifier
(`kvcot.utils.math_verifier.verify_math_equivalence`) -- not a defect in
`math_verify` itself, and not specific to ordered tuples: any bare compound
expression (tuple, interval, set) hits the same unreliable fallback path.
The fix re-wraps both the extracted model answer and the gold answer in
`\\boxed{...}` at the one production call site, which routes both through
`math_verify`'s well-tested boxed-extraction path -- confirmed double-wrap-
safe (`\\boxed{\\boxed{x}}` normalizes identically to `\\boxed{x}}`), so it
is correct regardless of whether either string happens to already carry a
`\\boxed{}` wrapper.
"""
from __future__ import annotations

from kvcot.discovery.math500_verification import Math500AnswerVerifier, build_math500_answer_fn


class _FakeTokenizer:
    """Maps a single integer id to a fixed decoded string -- these tests
    exercise verification logic, not real tokenization."""

    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text

    def decode(self, generated_ids, skip_special_tokens=True):
        return self.decoded_text


def _verify(decoded_text: str, gold_answer: str) -> Math500AnswerVerifier:
    verifier = build_math500_answer_fn(_FakeTokenizer(decoded_text), gold_answer)
    verifier([1, 2, 3])  # generated_ids content is irrelevant -- the fake tokenizer ignores it
    return verifier


def test_actual_b2a_r1_observed_case_is_now_correct_not_unverifiable():
    """The exact strings from the preserved B2A-R1 attempt artifacts."""
    verifier = _verify(
        decoded_text=r"final answer: \boxed{\left(3, \frac{\pi}{2}\right)}",
        gold_answer=r"\left( 3, \frac{\pi}{2} \right)",
    )
    assert verifier.last_result.status == "correct"
    assert verifier.last_result.verification.is_equivalent is True


def test_exact_gold_against_itself_is_correct():
    gold = r"\left( 3, \frac{\pi}{2} \right)"
    verifier = _verify(decoded_text=f"\\boxed{{{gold}}}", gold_answer=gold)
    assert verifier.last_result.status == "correct"


def test_whitespace_variant_is_correct():
    verifier = _verify(
        decoded_text=r"\boxed{\left(   3   ,   \frac{\pi}{2}   \right)}",
        gold_answer=r"(3,\frac{\pi}{2})",
    )
    assert verifier.last_result.status == "correct"


def test_left_right_vs_plain_parens_is_correct():
    verifier = _verify(
        decoded_text=r"\boxed{(3, \frac{\pi}{2})}",
        gold_answer=r"\left(3, \frac{\pi}{2}\right)",
    )
    assert verifier.last_result.status == "correct"


def test_unequal_tuple_component_is_incorrect():
    verifier = _verify(
        decoded_text=r"\boxed{(3, \frac{\pi}{3})}",
        gold_answer=r"(3, \frac{\pi}{2})",
    )
    assert verifier.last_result.status == "incorrect"


def test_swapped_tuple_order_is_incorrect():
    """Ordered-pair component order matters -- (a, b) != (b, a)."""
    verifier = _verify(
        decoded_text=r"\boxed{(\frac{\pi}{2}, 3)}",
        gold_answer=r"(3, \frac{\pi}{2})",
    )
    assert verifier.last_result.status == "incorrect"


def test_malformed_tuple_never_counts_as_correct():
    verifier = _verify(
        decoded_text=r"\boxed{(3, \frac{\pi}{2}}",  # unbalanced paren
        gold_answer=r"(3, \frac{\pi}{2})",
    )
    assert verifier.last_result.status != "correct"


def test_scalar_fraction_equivalence_still_correct():
    verifier = _verify(decoded_text=r"\boxed{0.5}", gold_answer=r"\frac{1}{2}")
    assert verifier.last_result.status == "correct"


def test_scalar_fraction_inequivalence_still_incorrect():
    verifier = _verify(decoded_text=r"\boxed{\frac{1}{3}}", gold_answer=r"\frac{1}{2}")
    assert verifier.last_result.status == "incorrect"


def test_symbolic_constant_pi_equivalence_still_correct():
    verifier = _verify(decoded_text=r"\boxed{3.14159265358979}", gold_answer=r"\pi")
    assert verifier.last_result.status == "correct"


def test_gold_answer_already_boxed_is_still_correct_double_wrap_safe():
    """If a future dataset variant's `answer` field already carries its own
    `\\boxed{}`, double-wrapping must not break verification."""
    verifier = _verify(decoded_text=r"\boxed{42}", gold_answer=r"\boxed{42}")
    assert verifier.last_result.status == "correct"


def test_unextractable_model_answer_remains_unverifiable():
    verifier = _verify(decoded_text="no boxed answer anywhere in this text", gold_answer=r"\frac{1}{2}")
    assert verifier.last_result.status == "unverifiable"
    assert verifier.last_result.verification is None
