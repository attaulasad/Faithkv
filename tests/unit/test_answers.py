from kvcot.utils.answers import (
    answers_match,
    answers_match_or_none,
    extract_answer,
    has_complete_boxed_answer,
    normalize_numeric_string,
)


def test_simple_boxed_integer():
    r = extract_answer("The answer is \\boxed{42}.")
    assert r.normalized_value == "42"
    assert r.method == "boxed"
    assert r.failure_reason is None


def test_nested_braces_fraction():
    r = extract_answer("So x = \\boxed{\\frac{1}{2}}.")
    assert r.method == "boxed"
    assert r.raw_match == "\\frac{1}{2}"
    # not numeric-normalizable, but still an explicit extracted answer
    assert r.normalized_value == "\\frac{1}{2}"
    assert r.failure_reason is None


def test_negative_number():
    r = extract_answer("\\boxed{-17}")
    assert r.normalized_value == "-17"


def test_decimal_number():
    r = extract_answer("\\boxed{3.14}")
    assert r.normalized_value == "3.14"


def test_integer_equivalent_decimal_normalizes_to_integer():
    r = extract_answer("\\boxed{5.0}")
    assert r.normalized_value == "5"


def test_commas_in_boxed_number():
    r = extract_answer("\\boxed{1,000}")
    assert r.normalized_value == "1000"


def test_dollar_sign_in_boxed():
    r = extract_answer("\\boxed{\\$5}")
    # "\$5" -> strip currency-ish leading noise via normalize; backslash-dollar
    # is not a plain $, so normalize_numeric_string must still recover "5".
    assert r.normalized_value == "5"


def test_plain_dollar_sign():
    assert normalize_numeric_string("$1,234.50") == "1234.5"


def test_multiple_boxes_takes_last():
    r = extract_answer("First \\boxed{1} then reconsider: \\boxed{2}")
    assert r.normalized_value == "2"
    assert r.raw_match == "2"


def test_missing_box_falls_back_to_final_answer_marker():
    r = extract_answer("I worked it out.\nFinal answer: 99")
    assert r.method == "final_answer_marker"
    assert r.normalized_value == "99"


def test_final_answer_marker_stops_at_newline():
    r = extract_answer("Final answer: 7\nWait, let me double check that.")
    assert r.normalized_value == "7"


def test_missing_box_and_marker_falls_back_to_number_scan():
    r = extract_answer("After all the steps, we get 123 as the result.")
    assert r.method == "final_number_fallback"
    assert r.normalized_value == "123"


def test_malformed_box_no_closing_brace():
    r = extract_answer("The answer is \\boxed{42 but I never closed it")
    assert r.normalized_value is None
    assert r.failure_reason == "malformed_box"


def test_no_answer_at_all():
    r = extract_answer("I am still thinking about this problem.")
    assert r.normalized_value is None
    assert r.method == "none"
    assert r.failure_reason == "no_answer_found"


def test_empty_generation():
    r = extract_answer("")
    assert r.normalized_value is None
    assert r.failure_reason == "empty_generation"


def test_explicit_marker_is_never_overridden_by_intermediate_numbers():
    text = "We tried 5, then 10, then 15 before settling.\nFinal answer: 15"
    r = extract_answer(text)
    assert r.method == "final_answer_marker"
    assert r.normalized_value == "15"


def test_boxed_takes_priority_over_final_answer_marker_text():
    # A model that writes both; boxed (method 1) must win.
    text = "Final answer: 5\n\\boxed{7}"
    r = extract_answer(text)
    assert r.method == "boxed"
    assert r.normalized_value == "7"


def test_answers_match_requires_both_present():
    assert answers_match("5", "5") is True
    assert answers_match(None, "5") is False
    assert answers_match(None, None) is False
    assert answers_match("5", "6") is False


def test_normalize_numeric_string_rejects_non_numeric():
    assert normalize_numeric_string("banana") is None
    assert normalize_numeric_string("") is None


def test_answers_match_or_none_distinguishes_extraction_failure_from_mismatch():
    assert answers_match_or_none(None, "5") is None
    assert answers_match_or_none("5", None) is None
    assert answers_match_or_none(None, None) is None
    assert answers_match_or_none("5", "5") is True
    assert answers_match_or_none("5", "6") is False


def test_has_complete_boxed_answer_requires_closed_brace():
    assert has_complete_boxed_answer("Final answer: \\boxed{42}")
    assert not has_complete_boxed_answer("Final answer: \\boxed{42")
    assert not has_complete_boxed_answer("no box here at all")
