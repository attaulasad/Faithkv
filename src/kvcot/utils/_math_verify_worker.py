"""Child-process worker for `kvcot.utils.math_verifier`.

Runs in complete isolation (`python -m kvcot.utils._math_verify_worker`),
launched fresh per comparison by the parent via `subprocess`. This is the
"otherwise isolate verification in a child process and terminate it on
timeout" fallback (`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md` Part
VI.14): `math_verify`'s own built-in `parsing_timeout`/`timeout_seconds`
mechanism uses `multiprocessing`, which was found to raise
`OSError: [WinError 6] The handle is invalid` on this Windows development
host — a real, observed failure of the "package's verified timeout
mechanism," not a hypothetical one — so both are disabled here
(`parsing_timeout=None`, `timeout_seconds=None`) and the *parent's*
`subprocess.run(..., timeout=...)` is the sole, authoritative timeout: it
genuinely kills this whole process (not just a thread) on expiry, so no
symbolic work can keep running past the deadline.

Reads one JSON object from stdin: `{"gold_text": str, "prediction_text":
str, "sleep_seconds": float | None}`. `sleep_seconds` is test-only (never
sent by the real parent path) — lets tests force a deterministic timeout
without depending on real symbolic-solver latency. Writes exactly one JSON
object to stdout: `{"status": ..., "failure_reason": str | None}`.
"""
from __future__ import annotations

import json
import sys
import time


def _run(payload: dict) -> dict:
    sleep_seconds = payload.get("sleep_seconds")
    if sleep_seconds:
        time.sleep(float(sleep_seconds))

    gold_text = payload["gold_text"]
    prediction_text = payload["prediction_text"]

    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    except Exception as exc:  # pragma: no cover - import-time environment failure
        return {"status": "verifier_error", "failure_reason": f"import failure: {type(exc).__name__}: {exc}"}

    extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]

    try:
        gold_parsed = parse(gold_text, extraction_config=extraction_config, parsing_timeout=None)
    except Exception as exc:
        return {"status": "verifier_error", "failure_reason": f"gold parse exception: {type(exc).__name__}: {exc}"}
    if not gold_parsed:
        return {"status": "gold_unparseable", "failure_reason": "math_verify.parse returned no candidates for gold_text"}

    try:
        prediction_parsed = parse(prediction_text, extraction_config=extraction_config, parsing_timeout=None)
    except Exception as exc:
        return {"status": "verifier_error", "failure_reason": f"prediction parse exception: {type(exc).__name__}: {exc}"}
    if not prediction_parsed:
        return {
            "status": "prediction_unparseable",
            "failure_reason": "math_verify.parse returned no candidates for prediction_text",
        }

    try:
        is_equivalent = bool(verify(gold_parsed, prediction_parsed, timeout_seconds=None))
    except Exception as exc:
        return {"status": "verifier_error", "failure_reason": f"verify exception: {type(exc).__name__}: {exc}"}

    return {"status": "equivalent" if is_equivalent else "not_equivalent", "failure_reason": None}


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except Exception as exc:
        print(json.dumps({"status": "verifier_error", "failure_reason": f"malformed worker payload: {exc!r}"}))
        return
    result = _run(payload)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
