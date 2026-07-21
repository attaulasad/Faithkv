"""Content-verified attempt artifact lifecycle (independent-audit Gate H6,
extended by the F4 final independent-audit repair into the ONE authoritative
successful-attempt verifier).

`kvcot.discovery.b2a_execute`'s `attempt_files_verified`/
`worker_envelopes_verified` final-gate conditions used to be pure
file-existence checks. `verify_attempt_artifacts` parses every required
pre-final artifact and cross-validates its CONTENT against the worker
results and against each other, returning every reason it found wrong
rather than stopping at the first.

F4 additions on top of the Gate H6 content checks:

- top-level artifact PARSING and cross-validation (`invocation.json`,
  `preflight.json`, `provenance.json`, `process_outcome.json`, and --
  when present -- `completion.json`): shared attempt ID, valid timestamps,
  start <= finish, sanitized argv, config/manifest path and hash agreement,
  preflight hardware schema, completion outcome/exit-code/gate agreement,
  provenance Git/submodule evidence;
- exact worker COMMAND identity (module, role, config/manifest/output
  paths, attempt ID, timeout, check/capture_output/text flags; duplicated,
  reordered-invalid, or contradictory argv is rejected);
- saved `result.json` == the coordinator-supplied result, plus typed
  `WorkerEnvelope` (always) and `FullKVWorkerResult`/`RKVWorkerResult`
  (`typed_results=True`) validation;
- coordinator process outcome (`process_outcome.json`: return codes 0,
  no timeout, `partial_success=False`);
- full progress-journal validation (known stages/statuses, per-event
  attempt-ID/role/timestamp validity, mapping-shaped counters, monotonic
  singleton-stage ordering, no duplicate singleton completion, start
  before completion where a start event exists, exactly 12 unique
  real-pair completions and exactly 1 no-op completion, no failure
  events, result construction before envelope construction);
- `verify_final_reference_manifest` recomputes every reference hash in
  `final.json` and rejects a changed byte, a missing/duplicate/unknown
  reference, a manifest that omits `completion.json`, or one that lists
  `final.json` itself.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from kvcot.utils.hashing import sha256_json

REQUIRED_ATTEMPT_FILES: frozenset[str] = frozenset({
    "invocation.json", "preflight.json", "provenance.json", "process_outcome.json",
    "fullkv/command.json", "fullkv/stdout.log", "fullkv/stderr.log", "fullkv/progress.jsonl",
    "fullkv/envelope.json", "fullkv/result.json", "fullkv/timing.json", "fullkv/memory.json",
    "rkv/command.json", "rkv/stdout.log", "rkv/stderr.log", "rkv/progress.jsonl",
    "rkv/envelope.json", "rkv/result.json", "rkv/timing.json", "rkv/memory.json",
    "rkv/pair_identities.json", "rkv/semantic_swaps.json", "rkv/replay_evidence.json",
})

# Independent-audit Gate H7.4: the exact durable progress-journal stage
# names BOTH worker bodies and the worker entry point actually emit in
# production -- derived directly from that code, never an aspirational
# broader list.
_COMMON_REQUIRED_PROGRESS_STAGES: tuple[str, ...] = (
    "startup", "config validation", "manifest validation", "snapshot resolution",
    "tokenizer load", "model-load start", "model-load completion", "runtime verification",
    "result construction", "envelope construction",
)
FULLKV_REQUIRED_PROGRESS_STAGES: tuple[str, ...] = _COMMON_REQUIRED_PROGRESS_STAGES
RKV_REQUIRED_PROGRESS_STAGES: tuple[str, ...] = _COMMON_REQUIRED_PROGRESS_STAGES + (
    "Pass 1", "Pass 2", "compact-target conversion", "each real pair", "no-op",
)

# F4.5: singleton stages that must complete EXACTLY once and in this order
# (first-completion indexes strictly increasing). "model-load start" is a
# started-status event, ordered separately below.
FULLKV_ORDERED_SINGLETON_STAGES: tuple[str, ...] = (
    "startup", "config validation", "manifest validation", "snapshot resolution",
    "tokenizer load", "model-load completion", "runtime verification",
    "result construction", "envelope construction",
)
RKV_ORDERED_SINGLETON_STAGES: tuple[str, ...] = (
    "startup", "config validation", "manifest validation", "snapshot resolution",
    "tokenizer load", "model-load completion", "runtime verification",
    "Pass 1", "Pass 2", "compact-target conversion",
    "result construction", "envelope construction",
)
_COMMON_KNOWN_EXTRA_STAGES: frozenset[str] = frozenset({
    "before_model_load", "post_load_baseline", "model-load start", "failed",
})
FULLKV_KNOWN_PROGRESS_STAGES: frozenset[str] = (
    frozenset(FULLKV_REQUIRED_PROGRESS_STAGES)
    | _COMMON_KNOWN_EXTRA_STAGES
    | {"fullkv_complete_natural_generation"}
)
RKV_KNOWN_PROGRESS_STAGES: frozenset[str] = frozenset(RKV_REQUIRED_PROGRESS_STAGES) | _COMMON_KNOWN_EXTRA_STAGES
KNOWN_PROGRESS_STATUSES: frozenset[str] = frozenset({"started", "completed", "failed"})

_ARGV_SECRET_MARKERS = ("token", "secret", "password", "authorization")


def verify_progress_stage_completeness(
    events: list[dict[str, Any]], *, role: str
) -> tuple[bool, tuple[str, ...]]:
    """At least one `status="completed"` event per required stage (the
    "model-load start" stage requires a `started` event instead)."""
    required = RKV_REQUIRED_PROGRESS_STAGES if role == "rkv" else FULLKV_REQUIRED_PROGRESS_STAGES
    completed_stages = {event.get("stage") for event in events if event.get("status") == "completed"}
    started_stages = {event.get("stage") for event in events if event.get("status") == "started"}
    missing = tuple(
        stage for stage in required
        if (stage not in started_stages if stage == "model-load start" else stage not in completed_stages)
    )
    return (len(missing) == 0), missing


def _parseable_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def _verify_progress_journal(
    events: list[dict[str, Any]], *, role: str, attempt_id: str | None, reasons: list[str]
) -> None:
    """F4.5: full per-event and cross-event progress-journal validation."""
    prefix = f"{role}/progress.jsonl"
    known = RKV_KNOWN_PROGRESS_STAGES if role == "rkv" else FULLKV_KNOWN_PROGRESS_STAGES
    for line_number, event in enumerate(events, start=1):
        if not _parseable_timestamp(event.get("timestamp")):
            reasons.append(f"{prefix}:{line_number} has no parseable ISO timestamp")
        if attempt_id is not None and event.get("attempt_id") != attempt_id:
            reasons.append(f"{prefix}:{line_number} attempt_id does not match invocation.json")
        if event.get("worker_role") != role:
            reasons.append(f"{prefix}:{line_number} worker_role does not match {role!r}")
        if event.get("stage") not in known:
            reasons.append(f"{prefix}:{line_number} has unknown stage {event.get('stage')!r}")
        if event.get("status") not in KNOWN_PROGRESS_STATUSES:
            reasons.append(f"{prefix}:{line_number} has unknown status {event.get('status')!r}")
        if not isinstance(event.get("counters"), dict):
            reasons.append(f"{prefix}:{line_number} counters is not mapping-shaped")

    if any(event.get("status") == "failed" or event.get("stage") == "failed" for event in events):
        reasons.append(f"{prefix} contains a failure event despite claimed success")

    ordered = RKV_ORDERED_SINGLETON_STAGES if role == "rkv" else FULLKV_ORDERED_SINGLETON_STAGES
    completion_indexes: dict[str, list[int]] = {}
    start_indexes: dict[str, list[int]] = {}
    for index, event in enumerate(events):
        if event.get("status") == "completed":
            completion_indexes.setdefault(str(event.get("stage")), []).append(index)
        elif event.get("status") == "started":
            start_indexes.setdefault(str(event.get("stage")), []).append(index)

    previous_index = -1
    for stage in ordered:
        indexes = completion_indexes.get(stage, [])
        if len(indexes) != 1:
            reasons.append(f"{prefix} singleton stage {stage!r} completed {len(indexes)} times (exactly 1 required)")
            continue
        if indexes[0] <= previous_index:
            reasons.append(f"{prefix} stage {stage!r} completed out of the required monotonic order")
        previous_index = indexes[0]

    for stage, starts in start_indexes.items():
        completions = completion_indexes.get(stage, [])
        if completions and min(completions) < min(starts):
            reasons.append(f"{prefix} stage {stage!r} completed before its own start event")
    if "model-load start" in start_indexes and "model-load completion" in completion_indexes:
        if min(completion_indexes["model-load completion"]) < min(start_indexes["model-load start"]):
            reasons.append(f"{prefix} model-load completion precedes model-load start")

    if role == "rkv":
        real_pair_phases = [
            (event.get("counters") or {}).get("timing_phase")
            for event in events
            if event.get("stage") == "each real pair" and event.get("status") == "completed"
        ]
        if len(real_pair_phases) != 12 or len(set(real_pair_phases)) != 12 or None in real_pair_phases:
            reasons.append(
                f"{prefix} requires exactly 12 unique real-pair completions, observed "
                f"{len(real_pair_phases)} events / {len(set(real_pair_phases))} unique"
            )
        no_op_completions = [
            event for event in events if event.get("stage") == "no-op" and event.get("status") == "completed"
        ]
        if len(no_op_completions) != 1:
            reasons.append(f"{prefix} requires exactly one no-op completion, observed {len(no_op_completions)}")


def _verify_command_identity(
    command: dict[str, Any],
    *,
    role: str,
    attempt_directory: Path,
    attempt_id: str | None,
    invocation: dict[str, Any] | None,
    python_executable: str | None,
    reasons: list[str],
) -> None:
    """F4.2: exact worker command identity from the recorded argv."""
    prefix = f"{role}/command.json"
    argv = command.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        reasons.append(f"{prefix} argv is not a list of strings")
        return
    if "--role" not in argv or role not in argv:
        reasons.append(f"{prefix} argv does not name role {role!r}")
    if len(argv) < 3 or argv[1:3] != ["-m", "kvcot.discovery.b2a_worker_entry"]:
        reasons.append(f"{prefix} argv does not invoke the b2a_worker_entry module")
        return
    if not argv[0]:
        reasons.append(f"{prefix} argv has an empty interpreter path")
    if python_executable is not None and argv[0] != python_executable:
        reasons.append(f"{prefix} interpreter {argv[0]!r} does not match the coordinator's {python_executable!r}")

    flag_values: dict[str, str] = {}
    tail = argv[3:]
    if len(tail) % 2 != 0:
        reasons.append(f"{prefix} argv flags are not flag/value pairs")
        return
    for flag, value in zip(tail[0::2], tail[1::2]):
        if not flag.startswith("--"):
            reasons.append(f"{prefix} argv contains a non-flag token {flag!r} in flag position")
            return
        if flag in flag_values:
            reasons.append(f"{prefix} argv duplicates flag {flag!r}")
            return
        flag_values[flag] = value
    expected_flags = {"--role", "--config", "--manifest", "--output", "--attempt-id"}
    if set(flag_values) != expected_flags:
        reasons.append(f"{prefix} argv flags {sorted(flag_values)} do not match required {sorted(expected_flags)}")
        return
    if flag_values["--role"] != role:
        reasons.append(f"{prefix} argv --role {flag_values['--role']!r} contradicts directory role {role!r}")
    if attempt_id is not None and flag_values["--attempt-id"] != attempt_id:
        reasons.append(f"{prefix} argv --attempt-id does not match the attempt ID")
    expected_output = str(attempt_directory / role / "result.json")
    if flag_values["--output"] != expected_output:
        reasons.append(f"{prefix} argv --output {flag_values['--output']!r} != {expected_output!r}")
    if invocation is not None:
        if invocation.get("config_path") is not None and flag_values["--config"] != invocation.get("config_path"):
            reasons.append(f"{prefix} argv --config contradicts invocation.json's config_path")
        if invocation.get("manifest_path") is not None and flag_values["--manifest"] != invocation.get("manifest_path"):
            reasons.append(f"{prefix} argv --manifest contradicts invocation.json's manifest_path")

    from kvcot.discovery.constants import B2A_WORKER_TIMEOUT_SECONDS

    if command.get("timeout_seconds") != B2A_WORKER_TIMEOUT_SECONDS:
        reasons.append(f"{prefix} timeout_seconds must be {B2A_WORKER_TIMEOUT_SECONDS}")
    if command.get("check") is not False:
        reasons.append(f"{prefix} 'check' must be False")
    if command.get("capture_output") is not True:
        reasons.append(f"{prefix} 'capture_output' must be True")
    if command.get("text") is not True:
        reasons.append(f"{prefix} 'text' must be True")


def verify_attempt_artifacts(
    attempt_directory: Path,
    *,
    fullkv_result: dict[str, Any],
    rkv_result: dict[str, Any],
    expected_config_hash: str | None = None,
    expected_manifest_hash: str | None = None,
    python_executable: str | None = None,
    typed_results: bool = False,
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

    # ---- F4.1: top-level artifact content ------------------------------
    invocation = load_json("invocation.json")
    attempt_id: str | None = None
    invocation_started_at: str | None = None
    if invocation is not None:
        attempt_id = invocation.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            reasons.append("invocation.json has no attempt_id")
            attempt_id = None
        directory_name = attempt_directory.name
        if attempt_id is not None and directory_name.startswith("b2a_attempt_"):
            if not directory_name.endswith(f"_{attempt_id}"):
                reasons.append("invocation.json attempt_id does not match the attempt directory name")
        invocation_started_at = invocation.get("started_at")
        if not _parseable_timestamp(invocation_started_at):
            reasons.append("invocation.json has no parseable started_at timestamp")
        argv = invocation.get("argv")
        if not isinstance(argv, list):
            reasons.append("invocation.json has no argv list")
        else:
            for item in argv:
                lowered = str(item).lower()
                if any(marker in lowered for marker in _ARGV_SECRET_MARKERS):
                    reasons.append("invocation.json argv is not sanitized (contains a credential-like token)")
                    break
        for name in ("config_path", "manifest_path"):
            if not isinstance(invocation.get(name), str) or not invocation.get(name):
                reasons.append(f"invocation.json has no {name}")

    preflight = load_json("preflight.json")
    if preflight is not None:
        if preflight.get("passed") is not True:
            reasons.append("preflight.json does not attest passed=True")
        if expected_config_hash is not None and preflight.get("config_hash") != expected_config_hash:
            reasons.append("preflight.json config_hash does not match the coordinator's config hash")
        if expected_manifest_hash is not None and preflight.get("manifest_hash") != expected_manifest_hash:
            reasons.append("preflight.json manifest_hash does not match the coordinator's manifest hash")
        device = preflight.get("device")
        if not isinstance(device, dict):
            reasons.append("preflight.json has no device hardware observation")
        else:
            required_device_keys = {
                "visible_gpu_count", "gpu_name", "device_index", "requested_device",
                "total_vram_bytes", "compute_capability", "driver_version", "cuda_runtime",
                "cudnn_version",
            }
            missing_keys = required_device_keys - set(device)
            if missing_keys:
                reasons.append(f"preflight.json device observation missing keys: {sorted(missing_keys)}")

    provenance = load_json("provenance.json")
    if provenance is not None:
        git_evidence = provenance.get("git")
        if not isinstance(git_evidence, dict):
            reasons.append("provenance.json has no git evidence")
        else:
            if git_evidence.get("dirty") is not False:
                reasons.append("provenance.json git evidence does not attest a clean working tree")
            if git_evidence.get("rkv_submodule_match") is not True:
                reasons.append("provenance.json git evidence does not attest the pinned R-KV submodule")
            head = git_evidence.get("head")
            if not isinstance(head, str) or len(head) != 40:
                reasons.append("provenance.json git evidence has no 40-hex HEAD SHA")

    process_outcome = load_json("process_outcome.json")
    if process_outcome is not None:
        if attempt_id is not None and process_outcome.get("attempt_id") != attempt_id:
            reasons.append("process_outcome.json attempt_id does not match invocation.json")
        return_codes = process_outcome.get("return_codes")
        if return_codes != {"fullkv": 0, "rkv": 0}:
            reasons.append(f"process_outcome.json return codes are not both 0: {return_codes!r}")
        timeout_state = process_outcome.get("timeout_state")
        if timeout_state != {"fullkv": False, "rkv": False}:
            reasons.append(f"process_outcome.json timeout state is not both False: {timeout_state!r}")
        if process_outcome.get("partial_success") is not False:
            reasons.append("process_outcome.json partial_success must be False for a successful attempt")
        durations = process_outcome.get("coordinator_observed_process_seconds")
        if not isinstance(durations, dict) or set(durations) != {"fullkv", "rkv"} or any(
            not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
            for value in durations.values()
        ):
            reasons.append("process_outcome.json coordinator-observed durations are missing or non-positive")

    completion = load_json("completion.json") if "completion.json" in existing else None
    if completion is not None:
        finished_at = completion.get("finished_at")
        if not _parseable_timestamp(finished_at):
            reasons.append("completion.json has no parseable finished_at timestamp")
        elif _parseable_timestamp(invocation_started_at):
            if datetime.fromisoformat(invocation_started_at) > datetime.fromisoformat(finished_at):
                reasons.append("completion.json finished_at precedes invocation.json started_at")
        if attempt_id is not None and completion.get("attempt_id") not in (None, attempt_id):
            reasons.append("completion.json attempt_id does not match invocation.json")
        outcome = completion.get("outcome")
        exit_code = completion.get("exit_code")
        gate_passed = completion.get("gate_passed")
        agreement = {
            "gate_passed": (exit_code == 0 and gate_passed is True),
            "gate_failed": (exit_code == 2 and gate_passed is False),
            "exception": (exit_code not in (0,) and gate_passed is not True),
        }
        if outcome not in agreement:
            reasons.append(f"completion.json has unknown outcome {outcome!r}")
        elif not agreement[outcome]:
            reasons.append(
                f"completion.json outcome {outcome!r} disagrees with exit_code={exit_code!r}/gate_passed={gate_passed!r}"
            )

    # ---- per-worker artifacts ------------------------------------------
    results_by_role = {"fullkv": fullkv_result, "rkv": rkv_result}
    envelope_attempt_ids: set[str] = set()

    for role, result in results_by_role.items():
        command = load_json(f"{role}/command.json")
        if command is not None:
            _verify_command_identity(
                command, role=role, attempt_directory=attempt_directory, attempt_id=attempt_id,
                invocation=invocation, python_executable=python_executable, reasons=reasons,
            )

        result_json = load_json(f"{role}/result.json")
        if result_json is not None:
            if result_json.get("role") != role:
                reasons.append(f"{role}/result.json role field does not match {role!r}")
            # F4.3: the saved result must BE the coordinator-supplied result.
            if result_json != result:
                reasons.append(f"saved {role}/result.json does not match the coordinator-supplied {role} result")
            if typed_results:
                from kvcot.discovery.b2a_workers import FullKVWorkerResult, RKVWorkerResult

                model = FullKVWorkerResult if role == "fullkv" else RKVWorkerResult
                try:
                    model.model_validate(result_json)
                except Exception as exc:  # noqa: BLE001
                    reasons.append(f"{role}/result.json does not validate as a typed {model.__name__}: {exc}")

        envelope = load_json(f"{role}/envelope.json")
        if envelope is not None:
            from kvcot.discovery.worker_envelope import WorkerEnvelope

            try:
                typed_envelope = WorkerEnvelope.model_validate(envelope)
            except Exception as exc:  # noqa: BLE001
                typed_envelope = None
                reasons.append(f"{role}/envelope.json does not validate as a typed WorkerEnvelope: {exc}")
            if envelope.get("role") != role:
                reasons.append(f"{role}/envelope.json role field does not match {role!r}")
            if envelope.get("success") is not True:
                reasons.append(f"{role}/envelope.json is not a success envelope")
            if envelope.get("error_type") is not None or envelope.get("error_message") is not None:
                reasons.append(f"{role}/envelope.json reports an error despite success=True")
            if typed_envelope is not None and (
                typed_envelope.failure_stage is not None or typed_envelope.is_oom or typed_envelope.is_timeout
            ):
                reasons.append(f"{role}/envelope.json carries failure fields despite success=True")
            envelope_id = envelope.get("attempt_id")
            if isinstance(envelope_id, str):
                envelope_attempt_ids.add(envelope_id)
            if attempt_id is not None and envelope_id != attempt_id:
                reasons.append(f"{role}/envelope.json attempt_id does not match invocation.json")
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
            else:
                stage_complete, missing_stages = verify_progress_stage_completeness(progress, role=role)
                if not stage_complete:
                    reasons.append(f"{role}/progress.jsonl is missing required stage(s): {list(missing_stages)}")
                _verify_progress_journal(progress, role=role, attempt_id=attempt_id, reasons=reasons)

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


def verify_final_reference_manifest(attempt_directory: Path) -> tuple[bool, tuple[str, ...]]:
    """F4.6/F5: recompute every hash in `final.json`'s reference manifest.
    Rejects a changed byte anywhere, a missing reference, a duplicate
    reference, an unknown (unlisted) file, a manifest that lists
    `final.json` itself, or one that omits `completion.json`."""
    from kvcot.discovery.attempt_artifacts import semantic_role_for, sha256_file

    reasons: list[str] = []
    final_path = attempt_directory / "final.json"
    if not final_path.is_file():
        return False, ("final.json is missing",)
    try:
        final_payload = json.loads(final_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, (f"final.json does not parse as valid JSON: {exc}",)
    manifest = final_payload.get("attempt_artifacts")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return False, ("final.json has no attempt_artifacts reference manifest",)

    references = manifest["files"]
    listed_paths = [reference.get("relative_path") for reference in references]
    if len(listed_paths) != len(set(listed_paths)):
        reasons.append("reference manifest lists a duplicate relative path")
    if "final.json" in listed_paths:
        reasons.append("reference manifest must not reference final.json itself")
    if "completion.json" not in listed_paths:
        reasons.append("reference manifest does not reference completion.json")

    on_disk = {
        path.relative_to(attempt_directory).as_posix()
        for path in attempt_directory.rglob("*") if path.is_file()
    } - {"final.json"}
    unknown = on_disk - set(listed_paths)
    if unknown:
        reasons.append(f"attempt directory contains unreferenced files: {sorted(unknown)}")

    for reference in references:
        relative = reference.get("relative_path")
        target = attempt_directory / str(relative)
        if not target.is_file():
            reasons.append(f"referenced file {relative!r} is missing")
            continue
        if reference.get("semantic_role") in (None, "unknown"):
            reasons.append(f"referenced file {relative!r} has no known semantic role")
        elif reference.get("semantic_role") != semantic_role_for(str(relative)):
            reasons.append(f"referenced file {relative!r} declares a conflicting semantic role")
        if reference.get("size_bytes") != target.stat().st_size:
            reasons.append(f"referenced file {relative!r} size changed")
        if reference.get("sha256") != sha256_file(target):
            reasons.append(f"referenced file {relative!r} content hash changed")

    return (len(reasons) == 0), tuple(reasons)
