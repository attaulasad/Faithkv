"""Content-verified attempt artifact lifecycle (independent-audit Gate H6).

`kvcot.discovery.b2a_execute`'s `attempt_files_verified`/
`worker_envelopes_verified` final-gate conditions used to be pure
file-existence checks (`required_attempt_files.issubset(existing)`,
`(path / "envelope.json").is_file()`). A worker (or a bug, or tampering)
could satisfy those checks with internally-inconsistent, mutually-
contradictory, or malformed content -- e.g. an envelope whose
`result_sha256` does not actually match its neighboring `result.json`,
or a `timing.json` copy that silently diverged from the result it was
supposed to mirror.

`verify_attempt_artifacts` parses every required pre-final artifact and
cross-validates its CONTENT against the worker results and against each
other, returning every reason it found wrong rather than stopping at the
first (so a caller can report a complete list, never just "verification
failed")."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kvcot.utils.hashing import sha256_json

REQUIRED_ATTEMPT_FILES: frozenset[str] = frozenset({
    "invocation.json", "preflight.json", "provenance.json",
    "fullkv/command.json", "fullkv/stdout.log", "fullkv/stderr.log", "fullkv/progress.jsonl",
    "fullkv/envelope.json", "fullkv/result.json", "fullkv/timing.json", "fullkv/memory.json",
    "rkv/command.json", "rkv/stdout.log", "rkv/stderr.log", "rkv/progress.jsonl",
    "rkv/envelope.json", "rkv/result.json", "rkv/timing.json", "rkv/memory.json",
    "rkv/pair_identities.json", "rkv/semantic_swaps.json", "rkv/replay_evidence.json",
})


def verify_attempt_artifacts(
    attempt_directory: Path, *, fullkv_result: dict[str, Any], rkv_result: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    """Returns `(verified, reasons)`. `reasons` is empty iff `verified`.
    Every required file is parsed and cross-checked for content
    consistency against the corresponding worker result and against its
    sibling artifacts -- never only checked for existence."""
    reasons: list[str] = []
    existing = {
        path.relative_to(attempt_directory).as_posix()
        for path in attempt_directory.rglob("*") if path.is_file()
    }
    missing = REQUIRED_ATTEMPT_FILES - existing
    if missing:
        reasons.append(f"missing required attempt files: {sorted(missing)}")
        return False, tuple(reasons)

    def load_json(relative_path: str) -> Any | None:
        try:
            return json.loads((attempt_directory / relative_path).read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 -- report every malformed file, never stop at the first
            reasons.append(f"{relative_path} does not parse as valid JSON: {type(exc).__name__}: {exc}")
            return None

    def load_jsonl(relative_path: str) -> list[dict[str, Any]] | None:
        events: list[dict[str, Any]] = []
        try:
            text = (attempt_directory / relative_path).read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"cannot read {relative_path}: {type(exc).__name__}: {exc}")
            return None
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"{relative_path}:{line_number} does not parse as valid JSON: {exc}")
                return None
        return events

    results_by_role = {"fullkv": fullkv_result, "rkv": rkv_result}
    envelope_attempt_ids: set[str] = set()

    for role, result in results_by_role.items():
        command = load_json(f"{role}/command.json")
        if command is not None:
            argv = command.get("argv")
            if not isinstance(argv, list) or "--role" not in argv or role not in argv:
                reasons.append(f"{role}/command.json argv does not name role {role!r}")
            if command.get("check") is not False:
                reasons.append(f"{role}/command.json 'check' must be False")
            if command.get("capture_output") is not True:
                reasons.append(f"{role}/command.json 'capture_output' must be True")

        result_json = load_json(f"{role}/result.json")
        if result_json is not None and result_json.get("role") != role:
            reasons.append(f"{role}/result.json role field does not match {role!r}")

        envelope = load_json(f"{role}/envelope.json")
        if envelope is not None:
            if envelope.get("role") != role:
                reasons.append(f"{role}/envelope.json role field does not match {role!r}")
            if envelope.get("success") is not True:
                reasons.append(f"{role}/envelope.json is not a success envelope")
            if envelope.get("error_type") is not None or envelope.get("error_message") is not None:
                reasons.append(f"{role}/envelope.json reports an error despite success=True")
            attempt_id = envelope.get("attempt_id")
            if isinstance(attempt_id, str):
                envelope_attempt_ids.add(attempt_id)
            if result_json is not None:
                expected_hash = sha256_json(result_json)
                if envelope.get("result_sha256") != expected_hash:
                    reasons.append(
                        f"{role}/envelope.json result_sha256 does not match the content of {role}/result.json"
                    )

        timing = load_json(f"{role}/timing.json")
        if timing is not None and result_json is not None and timing != result_json.get("timing_evidence"):
            reasons.append(f"{role}/timing.json does not match {role}/result.json's timing_evidence")

        memory = load_json(f"{role}/memory.json")
        if memory is not None and result_json is not None and memory != result_json.get("memory_phase_evidence"):
            reasons.append(f"{role}/memory.json does not match {role}/result.json's memory_phase_evidence")

        progress = load_jsonl(f"{role}/progress.jsonl")
        if progress is not None:
            if len(progress) == 0:
                reasons.append(f"{role}/progress.jsonl contains no events")
            elif any(event.get("worker_role") not in (role, None) for event in progress):
                reasons.append(f"{role}/progress.jsonl contains an event for a different worker_role")

    if len(envelope_attempt_ids) > 1:
        reasons.append(f"fullkv/rkv envelopes disagree on attempt_id: {sorted(envelope_attempt_ids)}")

    rkv_result_json = load_json("rkv/result.json")
    pair_identities = load_json("rkv/pair_identities.json")
    if pair_identities is not None and rkv_result_json is not None:
        expected_pair_identities = {
            "attempted": rkv_result_json.get("attempted_pair_identities"),
            "completed": rkv_result_json.get("completed_pair_identities"),
            "failed": rkv_result_json.get("failed_pair_identities"),
            "no_op": rkv_result_json.get("no_op_identity"),
        }
        if pair_identities != expected_pair_identities:
            reasons.append("rkv/pair_identities.json does not match rkv/result.json's pair identity fields")

    semantic_swaps = load_json("rkv/semantic_swaps.json")
    if (
        semantic_swaps is not None and rkv_result_json is not None
        and semantic_swaps != rkv_result_json.get("semantic_mutation_reports")
    ):
        reasons.append("rkv/semantic_swaps.json does not match rkv/result.json's semantic_mutation_reports")

    replay_evidence = load_json("rkv/replay_evidence.json")
    if (
        replay_evidence is not None and rkv_result_json is not None
        and replay_evidence != rkv_result_json.get("replay_evidence")
    ):
        reasons.append("rkv/replay_evidence.json does not match rkv/result.json's replay_evidence")

    return (len(reasons) == 0), tuple(reasons)


def verify_worker_envelopes(attempt_directory: Path) -> bool:
    """A narrower, envelope-only content check (distinct from
    `verify_attempt_artifacts`'s full-attempt scope) -- both FullKV and
    R-KV envelopes must exist, parse, name the correct role, and report
    `success=True` with no error recorded. Existence alone
    (`.is_file()`) is insufficient; a present-but-failed or malformed
    envelope must not satisfy this."""
    for role in ("fullkv", "rkv"):
        path = attempt_directory / role / "envelope.json"
        if not path.is_file():
            return False
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return False
        if envelope.get("role") != role or envelope.get("success") is not True:
            return False
        if envelope.get("error_type") is not None or envelope.get("error_message") is not None:
            return False
    return True
