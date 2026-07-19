"""B1A-2 MATH-500 verifier test matrix (Part VI of
docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md). Every case below was run
against the actually-installed math-verify==0.9.0 package and its real
output recorded before being asserted here (per the mission's "inspect the
installed API rather than guessing" instruction) -- no dataset download, no
network access; each comparison launches one real, short-lived, isolated
child process via `python -m kvcot.utils._math_verify_worker`.
"""
import pytest

from kvcot.utils.math_verifier import (
    MathVerificationResult,
    VERIFIER_TIMEOUT_SECONDS,
    _verify_math_equivalence_raw,
    verify_math_equivalence,
)


def test_result_post_init_rejects_inconsistent_status_and_is_equivalent():
    with pytest.raises(ValueError):
        MathVerificationResult(is_equivalent=True, status="not_equivalent", failure_reason=None)
    with pytest.raises(ValueError):
        MathVerificationResult(is_equivalent=False, status="equivalent", failure_reason=None)
    with pytest.raises(ValueError):
        MathVerificationResult(is_equivalent=False, status="timeout", failure_reason="x")


@pytest.mark.parametrize(
    "gold,prediction",
    [
        ("\\boxed{1/2}", "\\boxed{0.5}"),
        ("\\boxed{\\frac{1}{2}}", "\\boxed{0.5}"),
        ("\\boxed{2x}", "\\boxed{x+x}"),
        ("\\boxed{-3}", "\\boxed{-(3)}"),
        ("\\boxed{(1,2)}", "\\boxed{(1, 2)}"),
        ("\\boxed{\\{2,1\\}}", "\\boxed{\\{1,2\\}}"),
        ("\\boxed{[0, 1]}", "\\boxed{[0,1]}"),
        ("\\boxed{2}", "x=2"),
    ],
)
def test_accepted_equivalences(gold, prediction):
    result = verify_math_equivalence(prediction, gold)
    assert result.status == "equivalent"
    assert result.is_equivalent is True


def test_negative_infinity_bare_oo_is_not_treated_as_infinity_by_this_library():
    # Deviation from the naive spec example (-oo) recorded here rather than
    # silently asserted as passing: math-verify 0.9.0's expression grammar
    # parses bare "oo" as an implicit product of two symbols named "o", not
    # sympy's infinity -- verified directly (Part VI.13 instructs inspecting
    # the installed API rather than guessing). "-oo" is therefore correctly
    # classified NOT_EQUIVALENT against "-\infty", never silently accepted.
    result = verify_math_equivalence("\\boxed{(-oo, 3)}", "\\boxed{(-\\infty, 3)}")
    assert result.status in ("not_equivalent", "verifier_error")
    assert result.is_equivalent is not True


def test_interval_open_notation_equivalence_with_real_infinity_spelling():
    result = verify_math_equivalence("\\boxed{(-\\infty, 3.0)}", "\\boxed{(-\\infty, 3)}")
    assert result.status == "equivalent"
    assert result.is_equivalent is True


@pytest.mark.parametrize(
    "gold,prediction",
    [
        ("\\boxed{1/2}", "\\boxed{2/3}"),
        ("\\boxed{-3}", "\\boxed{3}"),
        ("\\boxed{(1,2)}", "\\boxed{(2,1)}"),
        ("\\boxed{\\{1,2\\}}", "\\boxed{\\{1,3\\}}"),
        ("\\boxed{[0,1]}", "\\boxed{(0,1)}"),
    ],
)
def test_rejected_non_equivalences(gold, prediction):
    result = verify_math_equivalence(prediction, gold)
    assert result.status == "not_equivalent"
    assert result.is_equivalent is False


def test_malformed_latex_never_counts_as_correct():
    result = verify_math_equivalence("\\boxed{\\frac{1}{}}", "\\boxed{5}")
    assert result.is_equivalent is not True


def test_empty_prediction_output_is_unparseable():
    result = verify_math_equivalence("", "\\boxed{5}")
    assert result.status == "prediction_unparseable"
    assert result.is_equivalent is None


def test_no_extracted_answer_is_unparseable():
    result = verify_math_equivalence("no answer here at all", "\\boxed{5}")
    assert result.status == "prediction_unparseable"
    assert result.is_equivalent is None


def test_malformed_unbalanced_box_is_unparseable():
    result = verify_math_equivalence("\\boxed{1/2", "\\boxed{5}")
    assert result.status == "prediction_unparseable"
    assert result.is_equivalent is None


def test_multiple_boxed_expressions_last_final_box_wins():
    prediction = "intermediate \\boxed{1} more text final \\boxed{2}"
    result = verify_math_equivalence(prediction, "\\boxed{2}")
    assert result.status == "equivalent"
    assert result.is_equivalent is True


def test_earlier_intermediate_box_does_not_override_later_final_box():
    prediction = "intermediate \\boxed{1} more text final \\boxed{2}"
    # If the earlier box (1) had incorrectly won, this would be "equivalent".
    result = verify_math_equivalence(prediction, "\\boxed{1}")
    assert result.status == "not_equivalent"


def test_unparseable_prediction_never_counts_correct():
    result = verify_math_equivalence("", "\\boxed{5}")
    assert result.is_equivalent is not True


def test_unparseable_gold_never_counts_correct():
    result = verify_math_equivalence("\\boxed{5}", "")
    assert result.status == "gold_unparseable"
    assert result.is_equivalent is not True


def test_forced_timeout_is_isolated_and_reported():
    result = _verify_math_equivalence_raw(
        "\\boxed{5}", "\\boxed{5}", timeout_seconds=0.3, extra_payload={"sleep_seconds": 2}
    )
    assert result.status == "timeout"
    assert result.is_equivalent is None


def test_production_entry_point_uses_frozen_five_second_timeout():
    assert VERIFIER_TIMEOUT_SECONDS == 5.0


def test_worker_exit_failure_is_reported_as_verifier_error(monkeypatch):
    import subprocess

    from kvcot.utils import math_verifier

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(math_verifier.subprocess, "run", fake_run)
    result = math_verifier.verify_math_equivalence("\\boxed{5}", "\\boxed{5}")
    assert result.status == "verifier_error"
    assert result.is_equivalent is None
    assert "boom" in result.failure_reason


def test_malformed_worker_stdout_is_reported_as_verifier_error(monkeypatch):
    import subprocess

    from kvcot.utils import math_verifier

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="not json", stderr="")

    monkeypatch.setattr(math_verifier.subprocess, "run", fake_run)
    result = math_verifier.verify_math_equivalence("\\boxed{5}", "\\boxed{5}")
    assert result.status == "verifier_error"
    assert result.is_equivalent is None
