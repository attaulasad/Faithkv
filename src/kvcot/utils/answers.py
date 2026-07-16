"""Answer extraction and normalization (§7 of the build brief).

Priority order, strictly enforced:
  1. last valid `\\boxed{...}` in the text (balanced-brace scan, not regex —
     regex cannot correctly match nested braces like `\\boxed{\\frac{1}{2}}`)
  2. explicit `Final answer:` marker
  3. conservative final-number fallback

"Never accept an arbitrary number from intermediate reasoning when an
explicit final-answer marker exists" is enforced structurally: the fallback
method only runs when methods 1 and 2 both fail to find anything, not as a
tiebreak among candidates.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*:\s*(.+)", re.IGNORECASE)
# A "number" here allows an optional leading currency symbol, optional minus
# sign, digit groups with optional comma separators, and an optional decimal
# part. This intentionally does not attempt full LaTeX/fraction parsing —
# the fallback tier is documented as conservative.
_NUMBER_RE = re.compile(
    r"[-+]?\$?\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\$?\s*\d+(?:\.\d+)?"
)


@dataclass(frozen=True)
class ExtractedAnswer:
    normalized_value: str | None
    raw_match: str | None
    method: str  # "boxed" | "final_answer_marker" | "final_number_fallback" | "none"
    failure_reason: str | None  # None iff normalized_value is not None


def _find_all_boxed(text: str) -> list[str]:
    """Balanced-brace scan for every `\\boxed{...}` occurrence, in order of
    appearance. A `\\boxed{` with no matching close brace before the string
    ends is skipped (malformed), not treated as a crash.
    """
    marker = "\\boxed{"
    results: list[str] = []
    search_from = 0
    while True:
        start = text.find(marker, search_from)
        if start == -1:
            break
        content_start = start + len(marker)
        depth = 1
        i = content_start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[content_start : i - 1])
            search_from = i
        else:
            # Unbalanced from this occurrence onward; stop scanning rather
            # than mis-pairing braces from a later, unrelated occurrence.
            search_from = content_start
            break
    return results


def normalize_numeric_string(raw: str) -> str | None:
    """Normalize a candidate answer string to a canonical numeric form, or
    return None if it cannot be parsed as a number at all (e.g. it's a
    LaTeX fraction or free text — callers treat that as extraction failure
    for the fallback tier, but `\\boxed{...}` content that isn't a plain
    number is still returned as normalized text by `extract_answer`, since a
    boxed non-numeric answer is still a valid, explicit final answer).
    """
    s = raw.strip()
    if not s:
        return None
    # strip currency symbols and surrounding markdown/punctuation noise
    s = s.strip().strip("*").strip()
    s = re.sub(r"^\\?[\$€£]\s*", "", s)
    s = s.replace(",", "")
    s = s.strip()
    if s == "":
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", s):
            return str(int(s))
        if re.fullmatch(r"[-+]?\d*\.\d+", s) or re.fullmatch(r"[-+]?\d+\.\d*", s):
            f = float(s)
            if f == int(f):
                return str(int(f))
            # Trim trailing zeros without losing precision-as-written intent.
            return repr(f)
    except ValueError:
        return None
    return None


def _normalize_boxed_content(raw: str) -> str:
    """Boxed content may be a plain number, or may be arbitrary math text
    (e.g. `\\frac{1}{2}`, `x=5`). Try numeric normalization first; fall back
    to a lightly-cleaned literal string so non-numeric boxed answers are
    still returned (extraction succeeded — the *comparison* logic upstream
    of this module decides what to do with a non-numeric answer).
    """
    numeric = normalize_numeric_string(raw)
    if numeric is not None:
        return numeric
    return raw.strip()


def extract_answer(text: str) -> ExtractedAnswer:
    if text is None or text.strip() == "":
        return ExtractedAnswer(None, None, "none", "empty_generation")

    boxed_matches = _find_all_boxed(text)
    if boxed_matches:
        last_raw = boxed_matches[-1]
        if last_raw.strip() == "":
            return ExtractedAnswer(None, last_raw, "none", "empty_boxed_content")
        normalized = _normalize_boxed_content(last_raw)
        return ExtractedAnswer(normalized, last_raw, "boxed", None)

    if "\\boxed{" in text:
        # A `\boxed{` with no matching close brace is an explicit (broken)
        # final-answer attempt. Per the brief, an explicit marker must never
        # be silently overridden by a weaker heuristic — including a raw
        # number that merely happens to appear inside the broken box's own
        # unclosed content. Fail outright rather than guessing.
        return ExtractedAnswer(None, None, "none", "malformed_box")

    final_answer_matches = _FINAL_ANSWER_RE.findall(text)
    if final_answer_matches:
        last_raw = final_answer_matches[-1].strip()
        # Truncate at the first newline: "Final answer: 42\nsome trailing
        # commentary" should not swallow the commentary into the answer.
        last_raw = last_raw.split("\n", 1)[0].strip()
        if last_raw != "":
            normalized = normalize_numeric_string(last_raw)
            if normalized is not None:
                return ExtractedAnswer(normalized, last_raw, "final_answer_marker", None)
            # Marker present but content isn't numeric and isn't boxed:
            # still return it as an explicit (non-numeric) answer rather than
            # falling through to the number-scan fallback, since the brief
            # forbids the fallback from overriding an explicit marker.
            return ExtractedAnswer(last_raw, last_raw, "final_answer_marker", None)

    # Conservative fallback: only reached when there is no explicit marker.
    number_matches = _NUMBER_RE.findall(text)
    if number_matches:
        last_raw = number_matches[-1]
        normalized = normalize_numeric_string(last_raw)
        if normalized is not None:
            return ExtractedAnswer(normalized, last_raw, "final_number_fallback", None)

    if final_answer_matches:
        # We found a marker but its content was unusable, and there is no
        # fallback number either.
        return ExtractedAnswer(None, None, "none", "marker_found_but_unparseable")
    return ExtractedAnswer(None, None, "none", "no_answer_found")


def answers_match(normalized_a: str | None, normalized_b: str | None) -> bool:
    """Equality on already-normalized values. Two None values are NOT a
    match — an extraction failure never counts as agreement with anything,
    including another extraction failure (§8: match_i,c,s(f) requires both
    sides to have a real extracted answer)."""
    if normalized_a is None or normalized_b is None:
        return False
    return normalized_a == normalized_b


def answers_match_or_none(normalized_a: str | None, normalized_b: str | None) -> bool | None:
    """Three-valued match, for call sites that must not conflate "the model
    produced a valid but different answer" (False) with "we could not
    extract a valid answer at all" (None) — the two are scientifically
    different failure modes and coercing the second into the first silently
    (as plain `==` comparison would) hides extraction breakage inside what
    looks like a normal disagreement rate. `answers_match` above is kept
    separate (and still used by the frozen primary replay-probe path, whose
    §8 match_i,c,s(f) definition is explicitly two-valued) — this function
    is for newer call sites (fixed-trace probe matching) that need the
    third value."""
    if normalized_a is None or normalized_b is None:
        return None
    return normalized_a == normalized_b


def has_complete_boxed_answer(text: str) -> bool:
    """True iff `text` contains at least one well-formed (balanced-brace)
    `\\boxed{...}` occurrence — used as a stopping predicate during greedy
    fixed-trace probe decoding (kvcot.generation.replay.branch_and_probe's
    `stop_predicate`) so decoding halts the instant the box closes, rather
    than continuing to `max_new_tokens` and risking a second solution
    attempt inside the same generation. An unclosed `\\boxed{` does not
    count (mirrors `_find_all_boxed`'s own malformed-box handling)."""
    return bool(_find_all_boxed(text))
