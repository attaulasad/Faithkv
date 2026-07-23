"""FullKV / R-KV process separation for B2A, and the two canonical worker
bodies (B1B-R4 §5/§6/§7/§8/§9/§10/§11/§12/§19, superseding B1B-R3's version
of this module). The repository already prohibits loading stock (FullKV)
and patched (R-KV) models in one Python process
(`kvcot.generation.state.declare_process_mode`, `ProcessModeConflictError`)
-- this module is a coordinator that launches two SEPARATE OS processes
(via `subprocess`, never `multiprocessing` or in-process mode switching)
and combines their independently-produced, schema-validated JSON results,
PLUS the two worker bodies themselves.

```
Coordinator process
├── FullKV worker process   (kvcot.discovery.b2a_worker_entry --role fullkv)
└── R-KV worker process     (kvcot.discovery.b2a_worker_entry --role rkv)
```

`run_both_workers_via_subprocess` is the coordinator's own entry point.
`subprocess_runner` is dependency-injected (defaults to `subprocess.run`)
specifically so the COMPLETE coordination flow -- unique temp directories,
launching both workers, reading back and schema-validating their JSON
output, checking shared-identity agreement, cleanup -- is exercised by CPU
tests with a fake runner that writes synthetic worker output instead of
actually invoking Python/torch/CUDA (`tests/unit/discovery
/test_b2a_workers.py`).

## B1B-R4 §19: one canonical worker API

`run_fullkv_worker`/`run_rkv_worker` are the ONLY two functions
`kvcot.discovery.b2a_worker_entry`'s real `__main__` block calls -- the
prior version of this module had `run_rkv_worker` raise
`NotImplementedError` while the real R-KV body lived in
`kvcot.discovery.b2a_execute.run_rkv_worker_body` and the worker entry
point called THAT function directly for the "rkv" role, a misleading split
this repair removes. Both are GPU-only in production and are exercised by
tests only via injected fake backends
(`tests/unit/discovery/test_b2a_workers_real_bodies.py`, B1B-R4 §20) --
every `import torch`/`transformers` reference stays deferred, matching this
repository's existing discipline.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, Field

from kvcot.discovery.attrition import PairFailureDetail
from kvcot.discovery.call_trace import (
    ActualModelCallRecorder,
    CallBoundaryEvent,
    CallTraceRecorder,
    compare_call_boundary_traces,
)
from kvcot.discovery.constants import B2A_REAL_PAIR_EVALUATIONS_TOTAL, B2A_SELECTED_EVENTS
from kvcot.discovery.schemas import SwapPairRecord
from kvcot.discovery.worker_partial_evidence import WorkerExecutionState

SubprocessRunner = Callable[..., "subprocess.CompletedProcess"]


class WorkerFailedError(RuntimeError):
    pass


def _production_progress_callback(role: str):
    """Recover the durable attempt journal configured by the worker entry.

    This is environment-scoped because the public production worker call
    deliberately remains exactly ``run_*_worker(config, manifest)``.  Test
    dependency-injection parameters therefore cannot leak into production,
    while crashes inside the body still leave the last completed phase on
    disk rather than materializing progress only after a successful return.
    """
    attempt_id = os.environ.get("KVCOT_B2A_ATTEMPT_ID")
    progress_path = os.environ.get("KVCOT_B2A_PROGRESS_PATH")
    if not attempt_id or not progress_path:
        return None

    from kvcot.discovery.attempt_artifacts import append_progress

    def emit(stage: str, status: str, counters=None) -> None:
        append_progress(
            Path(progress_path), attempt_id=attempt_id, worker_role=role,
            stage=stage, status=status, counters=counters,
        )

    return emit


def _progress_stage_for_phase(phase: str) -> str:
    exact = {
        "snapshot_tokenizer_resolution": "snapshot resolution",
        "tokenizer_load": "tokenizer load",
        "model_load": "model-load completion",
        "post_load_validation": "runtime verification",
        "rkv_complete_pass1": "Pass 1",
        "rkv_complete_pass2": "Pass 2",
        "compact_target_conversion": "compact-target conversion",
    }
    if phase.startswith("real_pair:"):
        return "each real pair"
    if phase.startswith("no_op_pair:"):
        return "no-op"
    return exact.get(phase, phase)


# --------------------------------------------------------------------------
# Shared nested evidence shapes (B1B-R4 §6/§9/§10/§11/§14). Loose `dict`
# typing (never `Any`) so a required field cannot be silently omitted at
# the top level, while the nested shape's own dataclass
# (`kvcot.discovery.runtime_evidence`/`kvcot.discovery.framework_seed`) is
# the single source of truth for what belongs inside.
# --------------------------------------------------------------------------


class FullKVWorkerResult(BaseModel):
    """Every field B1B-R4 §5/§6/§9/§10/§11/§14 requires the FullKV worker to
    report. Every field is REQUIRED -- a worker that cannot measure one of
    these must fail outright, never omit it."""

    role: str = Field(pattern="^fullkv$")
    model_revision: str
    tokenizer_revision: str
    dataset_repo: str
    dataset_revision: str
    manifest_hash: str
    prompt_token_ids_sha256: str
    prompt_token_count: int = Field(ge=0)

    natural_generated_token_ids: list[int]
    natural_answer: str | None
    natural_answer_status: str
    cap_hit: bool

    # B1B-R4 §5/§8: real greedy-loop call-boundary evidence -- never
    # inferred from `len(natural_generated_token_ids)` alone.
    prefill_call_count: int = Field(ge=0)
    decode_call_count: int = Field(ge=0)
    call_boundary_trace_hash: str
    prefill_token_ids_sha256: str | None = None

    wall_seconds: float = Field(ge=0.0)

    # B1B-R4 §6: the complete applied determinism policy.
    determinism_policy: dict[str, Any]

    # B1B-R4 §10: the actual runtime generation configuration and its hash.
    runtime_generation: dict[str, Any]
    runtime_generation_config_hash: str

    # B1B-R4 §11: derived (never hard-coded) parameter placement.
    parameter_placement: dict[str, Any]

    # B1B-R4 §9: requested vs. resolved model/tokenizer revision.
    runtime_identity: dict[str, Any]

    # B1B-R4 §14: allocated/reserved before-and-peak, and the reset point.
    memory: dict[str, Any]

    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_cuda_reserved_bytes: int = Field(ge=0)
    every_parameter_on_cuda: bool
    batch_size: int = Field(ge=1)
    actual_batch_size_verified: bool = False
    actual_call_evidence: list[dict[str, Any]] = Field(default_factory=list)
    snapshot_evidence: dict[str, Any] = Field(default_factory=dict)
    device_evidence: dict[str, Any] = Field(default_factory=dict)
    dataset_row_identity: dict[str, Any] = Field(default_factory=dict)
    timing_evidence: list[dict[str, Any]] = Field(default_factory=list)
    memory_phase_evidence: list[dict[str, Any]] = Field(default_factory=list)
    software_versions: dict[str, str]


class RKVWorkerResultV1(BaseModel):
    """The original (pre-forensic-repair) R-KV worker result shape --
    `docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md`. Every
    field below survived to disk historically; the twelve real pair
    evaluations' `SwapPairRecord` science (`swap_gain`, per-token NLL
    arrays, recomputed scores) never did -- `ExampleResult.pair_records`
    held them in memory but no field on this model ever carried them out of
    the worker process. Kept, unmodified, as the LEGACY shape so an
    already-archived V1 result JSON blob remains parseable for historical
    integrity verification (`parse_rkv_worker_result` below) -- never
    retroactively edited, never fabricated a `pair_records` field it never
    had."""

    role: str = Field(pattern="^rkv$")
    model_revision: str
    tokenizer_revision: str
    dataset_repo: str
    dataset_revision: str
    manifest_hash: str
    prompt_token_ids_sha256: str
    prompt_token_count: int = Field(ge=0)

    rkv_upstream_revision: str
    runtime_rkv_config_hash: str
    frozen_rkv_config_hash: str
    rkv_config_hash_match: bool

    example_valid: bool
    natural_answer_status: str

    # B1B-R4 §8: five INDEPENDENT trajectory/parity conditions -- never all
    # derived from `example_valid` alone.
    token_identical_replay: bool
    prefill_decode_boundary_parity: bool
    compaction_position_equality: bool
    capture_gather_parity: bool
    absolute_position_parity: bool
    no_op_numerical_parity: bool

    pass1_call_boundary: dict[str, Any]
    pass2_call_boundary: dict[str, Any]

    # B1B-R4 §22: exact, independently-countable selection/completion
    # accounting.
    observed_total_compaction_events: int = Field(ge=0)
    eligible_compaction_events: int = Field(ge=0)
    selected_compaction_events: int = Field(ge=0)
    events_with_at_least_one_completed_real_pair: int = Field(ge=0)
    events_with_all_four_real_pairs_completed: int = Field(ge=0)
    attempted_real_pair_count: int = Field(ge=0)
    completed_real_pair_count: int = Field(ge=0)
    failed_real_pair_count: int = Field(ge=0)
    attempted_no_op_pair_count: int = Field(ge=0)
    completed_no_op_pair_count: int = Field(ge=0)
    # B1B-R4.1 §15: one structured `PairFailureDetail` per failed pair
    # attempt (event/layer/head/candidate/donor/kind/stage/detail/elapsed
    # time), built live by `kvcot.discovery.orchestrator.run_example` --
    # never an always-empty placeholder.
    pair_failure_details: list[PairFailureDetail]

    # B1 execution-boundary closure §12: POSITIVE semantic-swap-check
    # counts -- `kvcot.discovery.b2a_evidence.SemanticSwapCheckEvidence`.
    semantic_swap_checks_required: int = Field(ge=0)
    semantic_swap_checks_attempted: int = Field(ge=0)
    semantic_swap_checks_passed: int = Field(ge=0)
    semantic_swap_checks_failed: int = Field(ge=0)

    # B1 execution-boundary closure §13: exact, duplicate-detecting pair
    # IDENTITY accounting -- `kvcot.discovery.b2a_evidence.PairIdentityEvidence`.
    unique_completed_real_pair_count: int = Field(ge=0)
    events_with_exactly_four_unique_real_pairs: int = Field(ge=0)
    has_duplicate_real_pair_identity: bool
    has_duplicate_no_op_pair_identity: bool

    # B1B-R4 §7/§21: exact-count mandatory gate conditions, derived (never
    # hard-coded) from the counts above.
    selected_event_count_exact: bool
    real_pair_count_exact: bool
    no_op_count_exact: bool
    all_required_pair_evaluations_completed: bool

    observed_retention_ratio: float = Field(ge=0.0, le=1.0)

    # B1B-R4 §12: per-pair, non-overlapping timing -- never an aggregate
    # bucket.
    wall_seconds_pass1: float = Field(ge=0.0)
    wall_seconds_pass2: float = Field(ge=0.0)
    wall_seconds_targeted_capture: float = Field(ge=0.0)
    real_pair_wall_seconds: list[float]
    no_op_pair_wall_seconds: list[float]

    determinism_policy: dict[str, Any]
    runtime_generation: dict[str, Any]
    runtime_generation_config_hash: str
    parameter_placement: dict[str, Any]
    runtime_identity: dict[str, Any]
    memory: dict[str, Any]

    # B1B-R4 §18: minimized per-target evidence only -- no full-cache tensor.
    minimized_target_evidence: list[dict[str, Any]]

    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_cuda_reserved_bytes: int = Field(ge=0)
    every_parameter_on_cuda: bool
    batch_size: int = Field(ge=1)
    actual_batch_size_verified: bool = False
    actual_call_evidence: list[dict[str, Any]] = Field(default_factory=list)
    snapshot_evidence: dict[str, Any] = Field(default_factory=dict)
    device_evidence: dict[str, Any] = Field(default_factory=dict)
    selected_event_evidence: list[dict[str, Any]] = Field(default_factory=list)
    attempted_pair_identities: list[dict[str, Any]] = Field(default_factory=list)
    completed_pair_identities: list[dict[str, Any]] = Field(default_factory=list)
    failed_pair_identities: list[dict[str, Any]] = Field(default_factory=list)
    no_op_identity: dict[str, Any] | None = None
    semantic_mutation_reports: list[dict[str, Any]] = Field(default_factory=list)
    no_op_evidence: dict[str, Any] = Field(default_factory=dict)
    replay_evidence: dict[str, Any] = Field(default_factory=dict)
    dataset_row_identity: dict[str, Any] = Field(default_factory=dict)
    timing_evidence: list[dict[str, Any]] = Field(default_factory=list)
    memory_phase_evidence: list[dict[str, Any]] = Field(default_factory=list)
    software_versions: dict[str, str]


class RKVWorkerResultV2(RKVWorkerResultV1):
    """The repaired R-KV worker result shape
    (`docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md`) -- every
    field V1 already had, PLUS the complete scientific `SwapPairRecord`
    population. `pair_records` is REQUIRED (no default): a payload that
    omits the key fails validation outright, so a future successful worker
    can never silently regress to discarding this evidence the way the
    original (V1) shape did. Structural versioning, not attempt-ID/SHA
    special-casing: any payload carrying a `pair_records` key parses as V2;
    any payload without one parses as V1 (`parse_rkv_worker_result`)."""

    schema_version: Literal["rkv_worker_result.v2"] = "rkv_worker_result.v2"
    pair_records: list[SwapPairRecord]


# Every NEW production code path (worker construction, coordinator
# parsing/validation) targets V2 -- `RKVWorkerResult` is kept as the public
# name so every existing `from kvcot.discovery.b2a_workers import
# RKVWorkerResult` import site continues to resolve, now to the version that
# REQUIRES `pair_records`. A successful new run can therefore never emit the
# legacy (V1) shape: `RKVWorkerResult(...)` raises unless the caller supplies
# `pair_records` explicitly.
RKVWorkerResult = RKVWorkerResultV2


class UnknownRKVWorkerResultSchemaVersion(ValueError):
    pass


def parse_rkv_worker_result(raw: dict[str, Any]) -> "RKVWorkerResultV1 | RKVWorkerResultV2":
    """Version dispatch, explicit `schema_version` first, structural
    fallback second -- never attempt-ID/SHA-based.

    A payload carrying `schema_version="rkv_worker_result.v2"` MUST
    validate as `RKVWorkerResultV2` -- if it is missing `pair_records` (or
    any other V2-required field), validation raises outright; it never
    silently falls back to being accepted as a legacy V1 result just
    because the fallback's looser shape happens to also fit (V1 ignores
    unknown fields, so a broken V2 payload missing `pair_records` would
    otherwise pass V1 validation with `schema_version` silently dropped --
    a corrupted/incomplete V2 result must fail loudly, never be
    misclassified as an honestly-legacy one). A payload carrying any OTHER
    non-null `schema_version` is an unknown/unsupported version and is
    rejected outright, never guessed at. Only a payload with NO
    `schema_version` key at all -- a genuinely pre-versioning historical
    shape, exactly what every result committed before this repair looks
    like -- uses the structural (`"pair_records" in raw`) fallback.
    """
    schema_version = raw.get("schema_version")
    if schema_version is not None:
        if schema_version == "rkv_worker_result.v2":
            return RKVWorkerResultV2.model_validate(raw)
        raise UnknownRKVWorkerResultSchemaVersion(
            f"unrecognized RKVWorkerResult schema_version: {schema_version!r}"
        )
    if "pair_records" in raw:
        return RKVWorkerResultV2.model_validate(raw)
    return RKVWorkerResultV1.model_validate(raw)


@dataclass(frozen=True)
class PairRecordAvailability:
    """What can honestly be claimed about a parsed R-KV worker result's
    pair-level scientific evidence -- computed structurally from which
    typed model it parsed as, never from an attempt ID, SHA, or archive
    name."""

    scientific_pair_records_available: bool
    scientific_pair_artifacts_verified: bool
    legacy_pair_record_schema: bool


def classify_pair_record_availability(
    result: "RKVWorkerResultV1 | RKVWorkerResultV2", *, artifacts_verified: bool = False
) -> PairRecordAvailability:
    """`artifacts_verified` lets a caller that has independently confirmed
    `rkv/pair_records.json`/`rkv/scientific_summary.json` on disk (never
    assumed) report `scientific_pair_artifacts_verified=True` for a V2
    result; a V1 (legacy) result can never claim either -- it structurally
    has no pair-record field to verify."""
    if isinstance(result, RKVWorkerResultV2):
        return PairRecordAvailability(
            scientific_pair_records_available=True,
            scientific_pair_artifacts_verified=artifacts_verified,
            legacy_pair_record_schema=False,
        )
    return PairRecordAvailability(
        scientific_pair_records_available=False,
        scientific_pair_artifacts_verified=False,
        legacy_pair_record_schema=True,
    )


SHARED_IDENTITY_FIELDS: tuple[str, ...] = (
    "dataset_repo",
    "dataset_revision",
    "manifest_hash",
    "prompt_token_ids_sha256",
)


def validate_shared_identity(fullkv: FullKVWorkerResult, rkv: RKVWorkerResult) -> tuple[bool, list[str]]:
    """Both workers were launched against the SAME frozen manifest/config --
    this proves it, rather than assuming it from the fact that the
    coordinator passed the same paths to both. Returns `(ok, mismatches)`,
    never silently drops a mismatch."""
    mismatches = []
    for field in SHARED_IDENTITY_FIELDS:
        if getattr(fullkv, field) != getattr(rkv, field):
            mismatches.append(f"{field}: fullkv={getattr(fullkv, field)!r} rkv={getattr(rkv, field)!r}")
    return (not mismatches, mismatches)


@dataclass(frozen=True)
class WorkerCoordinationResult:
    fullkv: FullKVWorkerResult
    rkv: RKVWorkerResult
    shared_identity_ok: bool
    shared_identity_mismatches: tuple[str, ...]
    attempt_directory: str | None = None
    return_codes: dict[str, int] | None = None
    timeout_state: dict[str, bool] | None = None
    partial_success: bool = False
    # Independent-audit Gate H2.5: wall-clock duration of each `_launch_worker`
    # subprocess call as observed BY THE COORDINATOR (`time.perf_counter()`
    # around the `subprocess_runner` call) -- includes Python interpreter
    # startup/import overhead the worker's OWN internal `SynchronizedTimer`
    # never sees (its first measurement only starts once the worker process
    # is already running). Never summed into the existing runtime
    # projection -- exported purely as a separate, honestly-labeled
    # diagnostic so process-launch overhead is visible rather than silently
    # absent.
    coordinator_observed_process_seconds: dict[str, float] | None = None


def _default_python_executable() -> str:
    return sys.executable


def _framework_seed_for_env(config_path: str) -> int:
    """Reads `framework_seed` off the SAME frozen config the worker itself
    will load -- never an independently chosen value. Falls back to
    `DiscoveryGenerationLock`'s own schema default (also `13`, and the
    value every real frozen discovery config in this repository sets
    explicitly -- `configs/discovery/llama8b_math500_b1024.yaml`) only when
    `config_path` cannot be loaded at all (e.g. coordinator-level CPU tests
    that intentionally pass a non-existent path to exercise orchestration
    logic without real file I/O) -- a genuine production config load
    failure still fails loudly at the worker itself moments later, this
    fallback never masks that."""
    from kvcot.discovery.discovery_config import DiscoveryGenerationLock, load_discovery_config

    try:
        return load_discovery_config(config_path).generation.framework_seed
    except Exception:
        # Deliberately broad: this function only computes an auxiliary env
        # var value for the child process about to be launched -- the REAL,
        # authoritative config load/validation happens moments later INSIDE
        # that worker subprocess (`kvcot.discovery.b2a_worker_entry`), which
        # still fails loudly on a genuinely malformed config. Swallowing a
        # load failure here never masks that.
        return DiscoveryGenerationLock.model_fields["framework_seed"].default


def _worker_subprocess_env(config_path: str) -> dict[str, str]:
    """B1B-R4.1 §28 repair: `random.seed()` (called INSIDE the already-
    running worker process, `kvcot.discovery.framework_seed
    .apply_framework_seed`) cannot control Python's hash-randomization seed
    -- that is fixed once, at interpreter startup, from the
    `PYTHONHASHSEED` environment variable, before any of this repository's
    own code ever runs. The only place this can actually be set is on the
    ENVIRONMENT the child interpreter is launched into, here, before
    `subprocess_runner` starts it. `TOKENIZERS_PARALLELISM=false` silences a
    known tokenizers-library fork-safety warning under multi-worker
    subprocess launches; harmless and frozen, not a defect being repaired,
    but recorded here for completeness (CLAUDE.md §28's own wording)."""
    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(_framework_seed_for_env(config_path))
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _launch_worker(
    role: str,
    config_path: str,
    manifest_path: str,
    output_path: Path,
    python_executable: str,
    subprocess_runner: SubprocessRunner,
    timeout_seconds: int,
    attempt_id: str | None = None,
) -> subprocess.CompletedProcess:
    """B1B-R4 §16: every worker subprocess is launched with `capture_output
    =True, text=True, timeout=B2A_WORKER_TIMEOUT_SECONDS, check=False` --
    stdout/stderr are always captured (never lost), a hung worker raises
    `subprocess.TimeoutExpired` rather than blocking forever, and a nonzero
    exit is handled explicitly by the caller (never an uncaught
    `CalledProcessError` from `check=True`). B1B-R4.1 §28: `env` is built by
    `_worker_subprocess_env` -- `PYTHONHASHSEED` is fixed BEFORE this child
    interpreter starts, the only point at which it can take effect."""
    argv = [
        python_executable, "-m", "kvcot.discovery.b2a_worker_entry",
        "--role", role, "--config", config_path, "--manifest", manifest_path, "--output", str(output_path),
    ]
    if attempt_id is not None:
        argv.extend(["--attempt-id", attempt_id])
    return subprocess_runner(
        argv, capture_output=True, text=True, timeout=timeout_seconds, check=False,
        env=_worker_subprocess_env(config_path),
    )


def run_both_workers_via_subprocess(
    config_path: str,
    manifest_path: str,
    *,
    python_executable: str | None = None,
    subprocess_runner: SubprocessRunner = subprocess.run,
    attempt_directory: Path | None = None,
) -> WorkerCoordinationResult:
    """The coordinator's one entry point: creates a unique temp directory,
    launches the FullKV and R-KV workers as SEPARATE subprocess invocations
    (never in one process, never via `multiprocessing`), reads back and
    schema-validates each worker's JSON output, checks shared-identity
    agreement, and cleans up the temp directory only after the combined
    result is safely constructed. Raises `WorkerFailedError` (naming which
    worker and its exit status) on any worker failure -- the caller
    (`kvcot.discovery.b2a_execute`) is responsible for writing a FAILURE
    artifact from that exception, never silently swallowing it.

    Preserves partial success (B1B-R4 §16): if FullKV succeeds and R-KV
    fails, the FullKV result is still attached to the raised
    `WorkerFailedError` (`.partial_fullkv_result`) so the coordinator can
    fold it into a fail artifact rather than discarding it."""
    from kvcot.discovery.constants import B2A_WORKER_TIMEOUT_SECONDS

    python_executable = python_executable or _default_python_executable()
    tmp_dir = attempt_directory or Path(tempfile.mkdtemp(prefix="kvcot-b2a-workers-"))
    preserve = attempt_directory is not None
    attempt_id = tmp_dir.name.rsplit("_", 1)[-1] if preserve else None
    try:
        fullkv_output = tmp_dir / "fullkv" / "result.json" if preserve else tmp_dir / "fullkv_result.json"
        rkv_output = tmp_dir / "rkv" / "result.json" if preserve else tmp_dir / "rkv_result.json"

        def preserve_command(role: str, output: Path) -> None:
            if not preserve:
                return
            from kvcot.discovery.attempt_artifacts import atomic_write_json

            atomic_write_json(
                tmp_dir / role / "command.json",
                {
                    "argv": [python_executable, "-m", "kvcot.discovery.b2a_worker_entry", "--role", role,
                             "--config", config_path, "--manifest", manifest_path, "--output", str(output),
                             "--attempt-id", attempt_id],
                    "timeout_seconds": B2A_WORKER_TIMEOUT_SECONDS,
                    "check": False,
                    "capture_output": True,
                    "text": True,
                },
            )

        def _last_progress_event(role: str) -> dict[str, Any] | None:
            progress_path = tmp_dir / role / "progress.jsonl"
            if not progress_path.is_file():
                return None
            last = None
            for line in progress_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    last = json.loads(line)
            return last

        def _write_termination_record(
            role: str, *, termination_kind: str, timed_out: bool, return_code: int | None,
            command: list[str] | None, stdout: str, stderr: str,
        ) -> None:
            # H1.5: a timed-out or otherwise killed worker may never reach
            # its own `finally`/atomic-envelope-write code -- this record is
            # authored by the COORDINATOR (never fabricated as if the
            # worker itself wrote it) so a post-mortem can distinguish "the
            # worker wrote a real failure envelope" from "the worker died
            # so abruptly nothing at all survived except this record".
            if not preserve:
                return
            from kvcot.discovery.attempt_artifacts import atomic_write_json
            from kvcot.utils.hashing import sha256_text

            envelope_path = tmp_dir / role / "envelope.json"
            atomic_write_json(
                tmp_dir / role / "termination.json",
                {
                    "attestor": "coordinator",
                    "worker_role": role,
                    "attempt_id": attempt_id,
                    "termination_kind": termination_kind,
                    "timed_out": timed_out,
                    "return_code": return_code,
                    "command": command,
                    "stdout_sha256": sha256_text(stdout),
                    "stderr_sha256": sha256_text(stderr),
                    "last_durable_progress_event": _last_progress_event(role),
                    "worker_authored_envelope_present": envelope_path.is_file(),
                },
            )

        def preserve_timeout_logs(role: str, exc: subprocess.TimeoutExpired) -> tuple[str, str]:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            if not preserve:
                return stdout, stderr
            from kvcot.discovery.attempt_artifacts import atomic_write_text

            atomic_write_text(tmp_dir / role / "stdout.log", stdout)
            atomic_write_text(tmp_dir / role / "stderr.log", stderr)
            return stdout, stderr

        process_seconds: dict[str, float] = {}
        try:
            preserve_command("fullkv", fullkv_output)
            _fullkv_launch_started = time.perf_counter()
            fullkv_proc = _launch_worker(
                "fullkv", config_path, manifest_path, fullkv_output, python_executable, subprocess_runner,
                B2A_WORKER_TIMEOUT_SECONDS, attempt_id,
            )
            process_seconds["fullkv"] = time.perf_counter() - _fullkv_launch_started
        except subprocess.TimeoutExpired as exc:
            process_seconds["fullkv"] = time.perf_counter() - _fullkv_launch_started
            stdout, stderr = preserve_timeout_logs("fullkv", exc)
            _write_termination_record(
                "fullkv", termination_kind="timeout", timed_out=True, return_code=None,
                command=list(exc.cmd) if exc.cmd is not None else None, stdout=stdout, stderr=stderr,
            )
            err = WorkerFailedError(f"fullkv worker timed out after {B2A_WORKER_TIMEOUT_SECONDS}s: {exc}")
            err.partial_fullkv_result = None  # type: ignore[attr-defined]
            err.timed_out = True  # type: ignore[attr-defined]
            raise err from exc
        if preserve:
            from kvcot.discovery.attempt_artifacts import atomic_write_text
            atomic_write_text(tmp_dir / "fullkv" / "stdout.log", getattr(fullkv_proc, "stdout", "") or "")
            atomic_write_text(tmp_dir / "fullkv" / "stderr.log", getattr(fullkv_proc, "stderr", "") or "")
        if fullkv_proc.returncode != 0:
            _write_termination_record(
                "fullkv", termination_kind="nonzero_exit", timed_out=False,
                return_code=int(fullkv_proc.returncode), command=None,
                stdout=getattr(fullkv_proc, "stdout", "") or "", stderr=getattr(fullkv_proc, "stderr", "") or "",
            )
            err = WorkerFailedError(
                f"fullkv worker exited with code {fullkv_proc.returncode}: "
                f"stdout={getattr(fullkv_proc, 'stdout', '')!r} stderr={getattr(fullkv_proc, 'stderr', '')!r}"
            )
            err.partial_fullkv_result = None  # type: ignore[attr-defined]
            raise err

        if not fullkv_output.exists():
            raise WorkerFailedError(f"fullkv worker reported success but wrote no output file at {fullkv_output}")
        fullkv_result = FullKVWorkerResult.model_validate_json(fullkv_output.read_text(encoding="utf-8"))
        if preserve:
            _validate_atomic_worker_envelope(fullkv_output, "fullkv", fullkv_result.model_dump(mode="json"))

        try:
            preserve_command("rkv", rkv_output)
            _rkv_launch_started = time.perf_counter()
            rkv_proc = _launch_worker(
                "rkv", config_path, manifest_path, rkv_output, python_executable, subprocess_runner,
                B2A_WORKER_TIMEOUT_SECONDS, attempt_id,
            )
            process_seconds["rkv"] = time.perf_counter() - _rkv_launch_started
        except subprocess.TimeoutExpired as exc:
            process_seconds["rkv"] = time.perf_counter() - _rkv_launch_started
            stdout, stderr = preserve_timeout_logs("rkv", exc)
            _write_termination_record(
                "rkv", termination_kind="timeout", timed_out=True, return_code=None,
                command=list(exc.cmd) if exc.cmd is not None else None, stdout=stdout, stderr=stderr,
            )
            err = WorkerFailedError(f"rkv worker timed out after {B2A_WORKER_TIMEOUT_SECONDS}s: {exc}")
            err.partial_fullkv_result = fullkv_result  # type: ignore[attr-defined]
            err.timed_out = True  # type: ignore[attr-defined]
            raise err from exc
        if preserve:
            from kvcot.discovery.attempt_artifacts import atomic_write_text
            atomic_write_text(tmp_dir / "rkv" / "stdout.log", getattr(rkv_proc, "stdout", "") or "")
            atomic_write_text(tmp_dir / "rkv" / "stderr.log", getattr(rkv_proc, "stderr", "") or "")
        if rkv_proc.returncode != 0:
            _write_termination_record(
                "rkv", termination_kind="nonzero_exit", timed_out=False,
                return_code=int(rkv_proc.returncode), command=None,
                stdout=getattr(rkv_proc, "stdout", "") or "", stderr=getattr(rkv_proc, "stderr", "") or "",
            )
            err = WorkerFailedError(
                f"rkv worker exited with code {rkv_proc.returncode}: "
                f"stdout={getattr(rkv_proc, 'stdout', '')!r} stderr={getattr(rkv_proc, 'stderr', '')!r}"
            )
            err.partial_fullkv_result = fullkv_result  # type: ignore[attr-defined]
            raise err

        if not rkv_output.exists():
            err = WorkerFailedError(f"rkv worker reported success but wrote no output file at {rkv_output}")
            err.partial_fullkv_result = fullkv_result  # type: ignore[attr-defined]
            raise err

        rkv_result = RKVWorkerResult.model_validate_json(rkv_output.read_text(encoding="utf-8"))
        if preserve:
            _validate_atomic_worker_envelope(rkv_output, "rkv", rkv_result.model_dump(mode="json"))

            from kvcot.discovery.attempt_artifacts import atomic_write_json

            for role, output, result in (("fullkv", fullkv_output, fullkv_result), ("rkv", rkv_output, rkv_result)):
                envelope_path = output.with_suffix(output.suffix + ".envelope.json")
                atomic_write_json(tmp_dir / role / "envelope.json", __import__("json").loads(envelope_path.read_text("utf-8")))
                atomic_write_json(tmp_dir / role / "timing.json", result.timing_evidence)
                atomic_write_json(tmp_dir / role / "memory.json", result.memory_phase_evidence)
            atomic_write_json(tmp_dir / "rkv" / "pair_identities.json", {
                "attempted": rkv_result.attempted_pair_identities,
                "completed": rkv_result.completed_pair_identities,
                "failed": rkv_result.failed_pair_identities,
                "no_op": rkv_result.no_op_identity,
            })
            atomic_write_json(tmp_dir / "rkv" / "semantic_swaps.json", rkv_result.semantic_mutation_reports)
            atomic_write_json(tmp_dir / "rkv" / "replay_evidence.json", rkv_result.replay_evidence)
            # B2A-R2 forensic repair
            # (docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md):
            # `rkv_result` is always an `RKVWorkerResultV2` on this live
            # code path (a freshly-launched worker from this same updated
            # code always emits V2), so `.pair_records` always exists here --
            # this artifact is written unconditionally, exactly like
            # pair_identities.json/semantic_swaps.json above, never gated on
            # gate-passed/gate-failed (a run with zero completed pairs
            # honestly writes an empty list, never padded).
            from kvcot.discovery.scientific_summary import build_scientific_summary

            pair_records_payload = [record.model_dump(mode="json") for record in rkv_result.pair_records]
            atomic_write_json(tmp_dir / "rkv" / "pair_records.json", pair_records_payload)
            atomic_write_json(
                tmp_dir / "rkv" / "scientific_summary.json",
                build_scientific_summary(rkv_result.pair_records),
            )
            # F4.4: the coordinator's OWN durable record of the two worker
            # processes' outcomes -- never inferable from worker-authored
            # artifacts alone.
            atomic_write_json(tmp_dir / "process_outcome.json", {
                "attempt_id": attempt_id,
                "return_codes": {"fullkv": int(fullkv_proc.returncode), "rkv": int(rkv_proc.returncode)},
                "timeout_state": {"fullkv": False, "rkv": False},
                "partial_success": False,
                "coordinator_observed_process_seconds": dict(process_seconds),
            })

        shared_ok, mismatches = validate_shared_identity(fullkv_result, rkv_result)
        return WorkerCoordinationResult(
            fullkv=fullkv_result, rkv=rkv_result, shared_identity_ok=shared_ok,
            shared_identity_mismatches=tuple(mismatches),
            attempt_directory=str(tmp_dir) if preserve else None,
            return_codes={"fullkv": int(fullkv_proc.returncode), "rkv": int(rkv_proc.returncode)},
            timeout_state={"fullkv": False, "rkv": False},
            partial_success=False,
            coordinator_observed_process_seconds=dict(process_seconds),
        )
    finally:
        if not preserve:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _validate_atomic_worker_envelope(output_path: Path, role: str, result_payload: dict[str, Any]) -> None:
    from kvcot.discovery.worker_envelope import WorkerEnvelope
    from kvcot.utils.hashing import sha256_json

    envelope_path = output_path.with_suffix(output_path.suffix + ".envelope.json")
    if not envelope_path.is_file():
        raise WorkerFailedError(f"{role} worker result has no valid atomic envelope")
    envelope = WorkerEnvelope.model_validate_json(envelope_path.read_text(encoding="utf-8"))
    if not envelope.success or envelope.role != role:
        raise WorkerFailedError(f"{role} worker envelope does not attest successful role-matched completion")
    expected_hash = sha256_json(result_payload)
    if envelope.result_sha256 != expected_hash:
        raise WorkerFailedError(
            f"{role} worker envelope result hash mismatch: {envelope.result_sha256} != {expected_hash}"
        )


# --------------------------------------------------------------------------
# Pass1/Pass2 dual-recording instrumentation (B1B-R4 §8/§12), shared by the
# R-KV worker body below.
# --------------------------------------------------------------------------


class _RkvHarnessInstrumentation:
    """Wraps the real `PrefillFn`/`DecodeOneFn`/`SnapshotFn` WITHOUT
    modifying `kvcot.discovery.orchestrator`/`pass1`/`pass2` at all -- those
    modules are called exactly as before, just with these wrapped callables
    passed in as the injected functions they already accept.

    Pass 1 vs Pass 2 attribution uses the call-order structural fact
    `kvcot.discovery.orchestrator.run_example` guarantees: `prefill_fn` is
    called EXACTLY ONCE by Pass 1 and EXACTLY ONCE by Pass 2 (in that
    order, always) -- the second `prefill_fn` call marks the transition;
    every call before it is Pass 1, every call from it onward is Pass 2.
    Records BOTH wall-clock timing (B1B-R4 §12) AND the ordered call-kinds-
    and-tokens trace (B1B-R4 §8) for each pass independently, so
    `prefill_decode_boundary_parity` compares two genuinely independently-
    observed traces."""

    def __init__(self, prefill_fn, decode_one_fn, snapshot_fn, timer=None):
        self._prefill_fn = prefill_fn
        self._decode_one_fn = decode_one_fn
        self._snapshot_fn = snapshot_fn
        self.pass1_trace = CallTraceRecorder(prefill_fn, decode_one_fn)
        self.pass2_trace = CallTraceRecorder(prefill_fn, decode_one_fn)
        self._prefill_call_count = 0
        self.pass1_wall_seconds = 0.0
        self.pass2_wall_seconds = 0.0
        self.targeted_capture_wall_seconds = 0.0
        self.timer = timer

    def _timed(self, phase, operation):
        if self.timer is None:
            start = time.perf_counter()
            result = operation()
            return result, time.perf_counter() - start
        result = self.timer.measure(phase, operation)
        return result, self.timer.records[-1].duration_seconds

    def _active_recorder(self) -> CallTraceRecorder:
        return self.pass1_trace if self._prefill_call_count <= 1 else self.pass2_trace

    def prefill(self, state, prompt_token_ids):
        self._prefill_call_count += 1
        recorder = self._active_recorder()
        prompt_token_ids = list(prompt_token_ids)
        recorder.events.append(CallBoundaryEvent(kind="prefill", token_ids=tuple(prompt_token_ids)))
        pass_name = "pass1" if self._prefill_call_count <= 1 else "pass2"
        result, elapsed = self._timed(
            f"rkv_{pass_name}_prefill", lambda: self._prefill_fn(state, prompt_token_ids)
        )
        if self._prefill_call_count <= 1:
            self.pass1_wall_seconds += elapsed
        else:
            self.pass2_wall_seconds += elapsed
        return result

    def decode_one(self, state, token_id):
        recorder = self._active_recorder()
        recorder.events.append(CallBoundaryEvent(kind="decode", token_ids=(token_id,)))
        pass_name = "pass1" if self._prefill_call_count <= 1 else "pass2"
        result, elapsed = self._timed(
            f"rkv_{pass_name}_decode", lambda: self._decode_one_fn(state, token_id)
        )
        if self._prefill_call_count <= 1:
            self.pass1_wall_seconds += elapsed
        else:
            self.pass2_wall_seconds += elapsed
        return result

    def snapshot(self, state):
        # `snapshot_fn` is only ever called from inside Pass 2
        # (`kvcot.discovery.pass2.run_pass2_capture`, at each selected
        # target's event position) -- never during Pass 1. Its wall time is
        # therefore genuinely PART of Pass 2's real wall-clock total, so it
        # is added into `pass2_wall_seconds` here (B1B-R4 §12: "Pass 2 total
        # may contain score/capture work" / "not added again if already
        # contained in Pass 2 total") -- `targeted_capture_wall_seconds`
        # remains a diagnostic BREAKDOWN of that same time, never a second,
        # additional measurement the coordinator's projection also sums.
        result, elapsed = self._timed("snapshot_creation", lambda: self._snapshot_fn(state))
        self.targeted_capture_wall_seconds += elapsed
        self.pass2_wall_seconds += elapsed
        return result


# --------------------------------------------------------------------------
# Canonical worker bodies (B1B-R4 §19). GPU-only in production; CPU tests
# exercise these bodies only via injected fake backends
# (`tests/unit/discovery/test_b2a_workers_real_bodies.py`). Called only from
# `kvcot.discovery.b2a_worker_entry`'s `__main__` block, itself only ever
# launched as a subprocess by `run_both_workers_via_subprocess` above.
# --------------------------------------------------------------------------


def run_fullkv_worker(
    config: Any,
    manifest: Any,
    *,
    _load_model: Callable[[], Any] | None = None,
    _load_tokenizer: Callable[[], Any] | None = None,
    _fresh_cache_factory: Callable[[], Any] | None = None,
    _cuda: Any | None = None,
    _device: str = "cuda:0",
    _clock: Callable[[], float] | None = None,
    _progress: Callable[[str, str, dict[str, Any] | None], None] | None = None,
) -> dict:
    """Runs exactly one frozen example through stock FullKV using the
    IDENTICAL greedy natural-run loop R-KV's Pass 1 uses
    (`kvcot.discovery.pass1.run_natural_pass1` +
    `kvcot.discovery.real_model_adapter`'s real `PrefillFn`/`DecodeOneFn`)
    -- B1B-R4 §5 repair: no sampling function is ever called (no
    `temperature`/`top_p`/`generator`), argmax token selection, EOS never
    appended or fed, exactly one prefill call, one decode call per
    generated token. Reports identity/timing/memory/answer/call-boundary
    evidence (B1B-R4 §6/§9/§10/§11/§14). Requires CUDA in production;
    CPU tests use an injected fake backend.

    B1B-R4 §20: `_load_model`/`_load_tokenizer`/`_fresh_cache_factory`/
    `_cuda`/`_device` are internal, underscore-prefixed dependency-injection
    seams -- the production CLI/subprocess entry point
    (`kvcot.discovery.b2a_worker_entry`) never passes any of them, so
    production always uses the real `FullKVPolicy`/`AutoTokenizer`/
    `DynamicCache`/`torch.cuda` defaults constructed below. CPU tests pass
    fakes for all five to execute this ENTIRE function body (seed
    application, model/tokenizer loading seam, prompt tensor construction,
    the real greedy loop, answer verification, call-boundary trace, runtime
    identity construction, memory observation, worker-result construction)
    without touching a real GPU."""
    import torch

    from kvcot.discovery.discovery_config import canonical_config_hash
    from kvcot.discovery.execution_measurement import CudaMemoryMeasurer, SynchronizedTimer
    from kvcot.discovery.framework_seed import apply_framework_seed
    from kvcot.discovery.math500_verification import build_math500_answer_fn
    from kvcot.discovery.no_offload import assert_no_offloaded_parameters
    from kvcot.discovery.pass1 import NaturalRunProvenance, run_natural_pass1
    from kvcot.discovery.real_model_adapter import (
        RealModelState,
        build_real_decode_one_fn,
        build_real_prefill_fn,
    )
    from kvcot.discovery.runtime_evidence import (
        RESET_POINT_AFTER_LOAD_BEFORE_INFERENCE,
        MemoryEvidence,
        build_runtime_generation_record,
        derive_parameter_placement,
        derive_runtime_identity,
    )
    from kvcot.generation.provenance import LayerProvenance, ModelProvenance
    from kvcot.generation.replay import CompactionTracker
    from kvcot.generation.state import reset_patched_state

    cuda = _cuda if _cuda is not None else torch.cuda
    _progress = _progress or _production_progress_callback("fullkv")
    # F1: explicit typed execution-state tracking -- `current_stage` set
    # immediately before each material operation, `last_completed_stage`
    # only after success. On failure, failing_stage = current_stage.
    execution_state = WorkerExecutionState(attempt_id=os.environ.get("KVCOT_B2A_ATTEMPT_ID"))

    # R1: the CUDA-availability call itself is the earliest point this
    # function can genuinely fail -- guarded on its own so an unexpected
    # exception from it (never just a clean `False`) still produces
    # partial evidence instead of a bare, unwrapped crash. The deliberate
    # "no CUDA available" refusal immediately below is unchanged: it stays
    # a plain `WorkerFailedError`, not wrapped, since nothing has actually
    # started yet.
    execution_state.enter("cuda availability check")
    try:
        cuda_available = bool(cuda.is_available())
    except Exception as exc:
        from kvcot.discovery.worker_partial_evidence import raise_worker_body_failure

        raise_worker_body_failure(role="fullkv", execution_state=execution_state, exc=exc, scope=locals())
    execution_state.complete("cuda availability check")

    if not cuda_available and _load_model is None:
        # Only the REAL (unfaked) production path requires CUDA -- a CPU
        # test that injects `_load_model`/`_cuda` is exercising this
        # function's control flow deliberately, never claiming a real GPU
        # ran anything.
        raise WorkerFailedError("run_fullkv_worker requires CUDA; none is available.")

    try:
        timer = SynchronizedTimer(cuda, _clock or time.perf_counter)
        _raw_measure = timer.measure

        def _tracked_measure(phase, operation):
            execution_state.enter(phase)
            result = _raw_measure(phase, operation)
            execution_state.complete(phase)
            return result

        timer.measure = _tracked_measure
        memory_meter = CudaMemoryMeasurer(cuda)
        complete_worker_started_at = timer.begin_span()

        def measured(phase, operation):
            result = timer.measure(phase, lambda: memory_meter.observe(phase, operation))
            if _progress is not None:
                _progress(_progress_stage_for_phase(phase), "completed", {"timing_phase": phase})
            return result

        measured("before_model_load", lambda: None)

        # B1B-R4 §6: applied independently in THIS worker's own process, before
        # any model inference -- never assumed shared with the R-KV worker
        # process (a separate OS process, per B1B-R3 §11's process-separation
        # requirement).
        determinism_policy = timer.measure(
            "fullkv_worker_startup",
            lambda: apply_framework_seed(
                config.generation.framework_seed, config.generation.attention_backend, cuda_available=cuda_available,
            ),
        )

        model_snapshot = None
        tokenizer_snapshot = None
        device_evidence: dict[str, Any] = {"verified": False, "reason": "injected test backend"}
        if _load_model is None:
            from kvcot.discovery.snapshot_boundary import resolve_local_snapshot
            from kvcot.discovery.strict_device import verify_single_rtx3090

            device = verify_single_rtx3090(cuda, torch_module=torch)
            device_evidence = {"verified": True, **device.__dict__}

            def resolve_snapshots():
                return (
                    resolve_local_snapshot(config.model.name, config.model.revision, "model"),
                    resolve_local_snapshot(config.model.tokenizer_name, config.model.tokenizer_revision, "tokenizer"),
                )

            model_snapshot, tokenizer_snapshot = timer.measure("snapshot_tokenizer_resolution", resolve_snapshots)
            if _progress is not None:
                _progress("snapshot resolution", "completed", None)
            if model_snapshot.free_bytes < model_snapshot.total_bytes:
                raise WorkerFailedError("insufficient free disk relative to verified local model snapshot size")
        else:
            timer.measure("snapshot_tokenizer_resolution", lambda: (None, None))

        if _load_tokenizer is not None:
            tokenizer = measured("tokenizer_load", _load_tokenizer)
        else:
            from transformers import AutoTokenizer

            tokenizer = measured(
                "tokenizer_load",
                lambda: AutoTokenizer.from_pretrained(
                    tokenizer_snapshot.local_path, local_files_only=True, use_fast=True
                ),
            )
        if _progress is not None:
            _progress("model-load start", "started", None)
        if _load_model is not None:
            model = measured("model_load", _load_model)
        else:
            from kvcot.discovery.strict_device import load_fullkv_discovery_model

            model = measured(
                "model_load", lambda: load_fullkv_discovery_model(config, model_snapshot.local_path, _device)
            )
        with execution_state.track("runtime identity derivation"):
            runtime_identity = derive_runtime_identity(
                model=model, tokenizer=tokenizer, requested_model_revision=config.model.revision,
                requested_tokenizer_revision=config.model.tokenizer_revision,
                verified_model_revision=None if model_snapshot is None else model_snapshot.resolved_revision,
                verified_tokenizer_revision=(
                    None if tokenizer_snapshot is None else tokenizer_snapshot.resolved_revision
                ),
            )
        def validate_post_load():
            assert_no_offloaded_parameters(model)
            return derive_parameter_placement(model, requested_device=_device)

        parameter_placement = measured("post_load_validation", validate_post_load)

        with execution_state.track("model architecture extraction"):
            num_layers = len(model.model.layers)
            num_kv_heads = model.config.num_key_value_heads

        # B1B-R4 §14: reset peak memory stats AFTER load, BEFORE measured
        # inference -- current model allocation is therefore included in the
        # reset baseline, matching the R-KV worker's identical reset point.
        measured("post_load_baseline", lambda: None)

        if _fresh_cache_factory is not None:
            cache_factory = _fresh_cache_factory
        else:
            from transformers.cache_utils import DynamicCache

            cache_factory = lambda: DynamicCache()  # noqa: E731

        with execution_state.track("cache and state construction"):
            cache = reset_patched_state(model, cache_factory)
            provenance = ModelProvenance(layers={i: LayerProvenance.empty(num_kv_heads) for i in range(num_layers)})
            state = RealModelState(
                model=model, cache=cache, model_provenance=provenance, compaction=CompactionTracker(),
                absolute_position=0, device=_device,
            )

        prompt_token_ids = list(manifest.prompt_token_ids)
        actual_calls = ActualModelCallRecorder()
        raw_prefill = build_real_prefill_fn(_device, actual_calls)
        raw_decode = build_real_decode_one_fn(_device, actual_calls)

        def timed_prefill(state, tokens):
            return timer.measure("fullkv_prefill", lambda: raw_prefill(state, tokens))

        def timed_decode(state, token):
            return timer.measure("fullkv_decode", lambda: raw_decode(state, token))

        recorder = CallTraceRecorder(
            timed_prefill, timed_decode
        )
        with execution_state.track("answer verifier construction"):
            raw_answer_fn = build_math500_answer_fn(tokenizer, manifest.gold_answer)

        def answer_fn(ids):
            return timer.measure("answer_verification", lambda: raw_answer_fn(ids))
        provenance_record = NaturalRunProvenance(
            model_name=config.model.name, model_revision=config.model.revision,
            tokenizer_name=config.model.tokenizer_name, tokenizer_revision=config.model.tokenizer_revision,
            rkv_revision=config.rkv.upstream_revision, config_sha256=canonical_config_hash(config),
            dataset_name=manifest.dataset_repo, example_id=manifest.unique_id,
        )

        trace = measured(
            "fullkv_complete_natural_generation",
            lambda: run_natural_pass1(
                provenance_record, prompt_token_ids, state, recorder.prefill, recorder.decode_one,
                config.generation.max_new_tokens, tokenizer.eos_token_id, answer_fn,
            ),
        )
        wall_seconds = next(
            record.duration_seconds for record in timer.records
            if record.phase == "fullkv_complete_natural_generation"
        )

        execution_state.enter("result construction")
        if not actual_calls.batch_size_verified:
            raise WorkerFailedError("no valid actual model-call batch evidence was recorded")
        batch_size = actual_calls.events[0].batch_size

        memory_meter.observe("fullkv_complete_worker", lambda: None)
        timer.finish_span("fullkv_complete_worker", complete_worker_started_at)
        peak_allocated = memory_meter.maximum_peak_allocated
        peak_reserved = memory_meter.maximum_peak_reserved
        first_memory = memory_meter.records[0]
        memory = MemoryEvidence(
            allocated_before_reset_bytes=first_memory.allocated_before,
            reserved_before_reset_bytes=first_memory.reserved_before,
            peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
            reset_point="explicit_phase_owned_resets",
        )

        runtime_generation = build_runtime_generation_record(
            batch_size=batch_size, max_new_tokens=config.generation.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
            attention_backend=config.generation.attention_backend, framework_seed=config.generation.framework_seed,
            prompt_token_count=len(prompt_token_ids),
        )

        from kvcot.utils.hashing import sha256_int_ids

        return FullKVWorkerResult(
            role="fullkv",
            model_revision=config.model.revision,
            tokenizer_revision=config.model.tokenizer_revision,
            dataset_repo=manifest.dataset_repo,
            dataset_revision=manifest.dataset_revision,
            manifest_hash=manifest.manifest_hash(),
            prompt_token_ids_sha256=manifest.prompt_token_ids_sha256,
            prompt_token_count=len(prompt_token_ids),
            natural_generated_token_ids=list(trace.generated_token_ids),
            natural_answer=trace.natural_answer,
            natural_answer_status=trace.natural_answer_status,
            cap_hit=trace.cap_hit,
            prefill_call_count=recorder.prefill_call_count,
            decode_call_count=recorder.decode_call_count,
            call_boundary_trace_hash=recorder.ordered_call_kinds_and_tokens_hash(),
            prefill_token_ids_sha256=(
                None
                if recorder.prefill_call_count != 1
                else sha256_int_ids(next(e.token_ids for e in recorder.events if e.kind == "prefill"))
            ),
            wall_seconds=wall_seconds,
            determinism_policy=determinism_policy.__dict__,
            runtime_generation=runtime_generation.__dict__,
            runtime_generation_config_hash=runtime_generation.canonical_hash(),
            parameter_placement=parameter_placement.__dict__,
            runtime_identity=runtime_identity.__dict__,
            memory=memory.__dict__,
            peak_cuda_allocated_bytes=peak_allocated,
            peak_cuda_reserved_bytes=peak_reserved,
            every_parameter_on_cuda=parameter_placement.every_parameter_on_cuda,
            batch_size=batch_size,
            actual_batch_size_verified=actual_calls.batch_size_verified,
            actual_call_evidence=actual_calls.export(),
            snapshot_evidence={
                "verified": model_snapshot is not None and tokenizer_snapshot is not None,
                "model": None if model_snapshot is None else model_snapshot.__dict__,
                "tokenizer": None if tokenizer_snapshot is None else tokenizer_snapshot.__dict__,
            },
            device_evidence=device_evidence,
            dataset_row_identity={
                "dataset_repo": manifest.dataset_repo,
                "dataset_revision": manifest.dataset_revision,
                "example_index": manifest.example_index,
                "unique_id": manifest.unique_id,
                "raw_content_hash": getattr(manifest, "raw_content_hash", None),
                "manifest_canonical_hash": manifest.manifest_hash(),
                "rendered_user_message_sha256": getattr(manifest, "rendered_user_message_sha256", None),
                "chat_template_source_sha256": getattr(manifest, "chat_template_source_sha256", None),
                "prompt_token_ids_sha256": manifest.prompt_token_ids_sha256,
                "prompt_token_count": len(prompt_token_ids),
            },
            timing_evidence=timer.export(),
            memory_phase_evidence=memory_meter.export(),
            software_versions={"torch": torch.__version__},
        ).model_dump(mode="json")
    except Exception as exc:
        from kvcot.discovery.worker_partial_evidence import raise_worker_body_failure

        raise_worker_body_failure(role="fullkv", execution_state=execution_state, exc=exc, scope=locals())


def run_rkv_worker(
    config: Any,
    manifest: Any,
    *,
    _load_model: Callable[[], Any] | None = None,
    _load_tokenizer: Callable[[], Any] | None = None,
    _fresh_cache_factory: Callable[[], Any] | None = None,
    _cuda: Any | None = None,
    _device: str = "cuda:0",
    _clock: Callable[[], float] | None = None,
    _progress: Callable[[str, str, dict[str, Any] | None], None] | None = None,
) -> dict:
    """Runs Pass 1, Pass 2, targeted capture, branch evaluation, and the
    B2A single no-op calibration for exactly one example under R-KV, and
    reports the resulting evidence (B1B-R4 §19: the ONE canonical R-KV
    worker body -- supersedes the B1B-R3 split between a `NotImplementedError`
    stub here and the real body in `kvcot.discovery.b2a_execute
    .run_rkv_worker_body`). Requires CUDA in production; CPU tests use an
    injected fake backend. Delegates the actual pass/
    branch machinery entirely to `kvcot.discovery.orchestrator.run_example`
    -- never a second, independently-written execution path.

    B1B-R4 §20: same internal, underscore-prefixed dependency-injection
    seams as `run_fullkv_worker` -- never exposed by the production CLI/
    subprocess entry point. A CPU test injecting all five executes: seed
    application, policy construction, runtime R-KV verification (against a
    real, small fake `kv_cluster`), Pass 1, selection of three events,
    token-identical Pass 2, minimized targeted capture, snapshot creation,
    real pair evaluations, the single no-op evaluation, branch compaction
    restoration, timing collection, memory observation, independent parity
    evidence, and worker-result construction -- the REAL body, never a
    preconstructed `RKVWorkerResult`."""
    import torch

    from kvcot.discovery.attrition import AttritionCounters
    from kvcot.discovery.b2a_evidence import (
        derive_meaningful_compression_observed,
        derive_no_op_numerical_parity,
        derive_observed_retention_ratio,
        derive_pair_completion_evidence,
        derive_pair_identity_evidence,
        derive_semantic_swap_check_evidence,
        derive_trajectory_parity_evidence,
    )
    from kvcot.discovery.constants import B2A_NOOP_PAIR_EVALUATIONS_TOTAL, NoOpMode
    from kvcot.discovery.discovery_config import canonical_config_hash
    from kvcot.discovery.framework_seed import apply_framework_seed
    from kvcot.discovery.execution_measurement import (
        CudaMemoryMeasurer,
        SynchronizedTimer,
        check_pre_branch_memory,
    )
    from kvcot.discovery.math500_verification import build_math500_answer_fn
    from kvcot.discovery.no_offload import assert_no_offloaded_parameters
    from kvcot.discovery.orchestrator import PairExecutionPolicy, run_example
    from kvcot.discovery.pass1 import NaturalRunProvenance
    from kvcot.discovery.real_model_adapter import (
        RealModelState,
        build_real_branch_step_fn_restore_once,
        build_real_decode_one_fn,
        build_real_prefill_fn,
        build_real_snapshot_fn,
    )
    from kvcot.discovery.runtime_evidence import (
        RESET_POINT_AFTER_LOAD_BEFORE_INFERENCE,
        MemoryEvidence,
        build_runtime_generation_record,
        derive_parameter_placement,
        derive_runtime_identity,
    )
    from kvcot.discovery.runtime_rkv_verification import verify_runtime_matches_frozen
    from kvcot.discovery.sampling import IdentitySeedParts
    from kvcot.generation.provenance import LayerProvenance, ModelProvenance
    from kvcot.generation.replay import CompactionTracker
    from kvcot.generation.state import reset_patched_state

    cuda = _cuda if _cuda is not None else torch.cuda
    _progress = _progress or _production_progress_callback("rkv")
    execution_state = WorkerExecutionState(attempt_id=os.environ.get("KVCOT_B2A_ATTEMPT_ID"))

    # R1: see the identical guard in `run_fullkv_worker` -- the deliberate
    # "no CUDA available" refusal below stays a plain, unwrapped
    # `WorkerFailedError`; only a genuinely unexpected exception FROM the
    # availability check itself is captured as partial evidence.
    execution_state.enter("cuda availability check")
    try:
        cuda_available = bool(cuda.is_available())
    except Exception as exc:
        from kvcot.discovery.worker_partial_evidence import raise_worker_body_failure

        raise_worker_body_failure(role="rkv", execution_state=execution_state, exc=exc, scope=locals())
    execution_state.complete("cuda availability check")

    if not cuda_available and _load_model is None:
        raise WorkerFailedError("run_rkv_worker requires CUDA; none is available.")

    try:
        timer = SynchronizedTimer(cuda, _clock or time.perf_counter)
        _raw_measure = timer.measure

        def _tracked_measure(phase, operation):
            execution_state.enter(phase)
            result = _raw_measure(phase, operation)
            execution_state.complete(phase)
            return result

        timer.measure = _tracked_measure
        memory_meter = CudaMemoryMeasurer(cuda)
        complete_worker_started_at = timer.begin_span()

        def measured(phase, operation):
            result = timer.measure(phase, lambda: memory_meter.observe(phase, operation))
            if _progress is not None:
                _progress(_progress_stage_for_phase(phase), "completed", {"timing_phase": phase})
            return result

        measured("before_model_load", lambda: None)

        determinism_policy = timer.measure(
            "rkv_worker_startup",
            lambda: apply_framework_seed(
                config.generation.framework_seed, config.generation.attention_backend, cuda_available=cuda_available,
            ),
        )

        model_snapshot = None
        tokenizer_snapshot = None
        device_evidence: dict[str, Any] = {"verified": False, "reason": "injected test backend"}
        if _load_model is None:
            from kvcot.discovery.snapshot_boundary import resolve_local_snapshot
            from kvcot.discovery.strict_device import verify_single_rtx3090

            device = verify_single_rtx3090(cuda, torch_module=torch)
            device_evidence = {"verified": True, **device.__dict__}

            def resolve_snapshots():
                return (
                    resolve_local_snapshot(config.model.name, config.model.revision, "model"),
                    resolve_local_snapshot(config.model.tokenizer_name, config.model.tokenizer_revision, "tokenizer"),
                )

            model_snapshot, tokenizer_snapshot = timer.measure("snapshot_tokenizer_resolution", resolve_snapshots)
            if _progress is not None:
                _progress("snapshot resolution", "completed", None)
            if model_snapshot.free_bytes < model_snapshot.total_bytes:
                raise WorkerFailedError("insufficient free disk relative to verified local model snapshot size")
        else:
            timer.measure("snapshot_tokenizer_resolution", lambda: (None, None))

        if _load_tokenizer is not None:
            tokenizer = measured("tokenizer_load", _load_tokenizer)
        else:
            from transformers import AutoTokenizer

            tokenizer = measured(
                "tokenizer_load",
                lambda: AutoTokenizer.from_pretrained(
                    tokenizer_snapshot.local_path, local_files_only=True, use_fast=True
                ),
            )
        if _progress is not None:
            _progress("model-load start", "started", None)
        if _load_model is not None:
            model = measured("model_load", _load_model)
        else:
            from kvcot.discovery.strict_device import load_rkv_discovery_model

            model = measured(
                "model_load",
                lambda: load_rkv_discovery_model(
                    config, model_snapshot.local_path, tokenizer_snapshot.local_path, _device
                ),
            )
        execution_state.enter("runtime R-KV config verification")
        runtime_check = verify_runtime_matches_frozen(config.rkv, model)
        if not runtime_check.passed:
            raise WorkerFailedError(
                f"runtime R-KV configuration disagrees with the frozen config on: {runtime_check.mismatched_fields} "
                f"(frozen_hash={runtime_check.frozen_hash}, runtime_hash={runtime_check.runtime_hash})"
            )
        execution_state.complete("runtime R-KV config verification")
        with execution_state.track("runtime identity derivation"):
            runtime_identity = derive_runtime_identity(
                model=model, tokenizer=tokenizer, requested_model_revision=config.model.revision,
                requested_tokenizer_revision=config.model.tokenizer_revision,
                verified_model_revision=None if model_snapshot is None else model_snapshot.resolved_revision,
                verified_tokenizer_revision=(
                    None if tokenizer_snapshot is None else tokenizer_snapshot.resolved_revision
                ),
            )
        def validate_post_load():
            assert_no_offloaded_parameters(model)
            return derive_parameter_placement(model, requested_device=_device)

        parameter_placement = measured("post_load_validation", validate_post_load)
        with execution_state.track("model architecture extraction"):
            num_layers = len(model.model.layers)
            num_kv_heads = model.config.num_key_value_heads

        measured("post_load_baseline", lambda: None)

        if _fresh_cache_factory is not None:
            cache_factory = _fresh_cache_factory
        else:
            from transformers.cache_utils import DynamicCache

            cache_factory = lambda: DynamicCache()  # noqa: E731

        def _fresh_state() -> RealModelState:
            # R1: tracked on every call -- both the immediate Pass-1
            # invocation below and the later Pass-2 factory invocation
            # inside `run_example` (already nested inside its own
            # "rkv_complete_pass2" tracked span, exactly like the
            # prefill/decode/snapshot sub-phases already nest there).
            with execution_state.track("cache and state construction"):
                cache = reset_patched_state(model, cache_factory)
                provenance = ModelProvenance(
                    layers={i: LayerProvenance.empty(num_kv_heads) for i in range(num_layers)}
                )
                return RealModelState(
                    model=model, cache=cache, model_provenance=provenance, compaction=CompactionTracker(),
                    absolute_position=0, device=_device,
                )

        actual_calls = ActualModelCallRecorder()
        instrumented = _RkvHarnessInstrumentation(
            build_real_prefill_fn(_device, actual_calls),
            build_real_decode_one_fn(_device, actual_calls),
            build_real_snapshot_fn(),
            timer,
        )

        identity = IdentitySeedParts(
            global_seed=config.generation.framework_seed, dataset_name=manifest.dataset_repo,
            problem_index=manifest.example_index, model_revision=config.model.revision,
            rkv_revision=config.rkv.upstream_revision,
        )
        with execution_state.track("answer verifier construction"):
            answer_verifier = build_math500_answer_fn(tokenizer, manifest.gold_answer)

        def timed_answer_verifier(ids):
            return timer.measure("answer_verification", lambda: answer_verifier(ids))
        provenance_record = NaturalRunProvenance(
            model_name=config.model.name, model_revision=config.model.revision,
            tokenizer_name=config.model.tokenizer_name, tokenizer_revision=config.model.tokenizer_revision,
            rkv_revision=config.rkv.upstream_revision, config_sha256=canonical_config_hash(config),
            dataset_name=manifest.dataset_repo, example_id=manifest.unique_id,
        )

        prompt_token_ids = list(manifest.prompt_token_ids)
        assert len(prompt_token_ids) > 0, "structurally impossible: an empty prompt must never reach Pass 1"

        example_attrition = AttritionCounters()
        pair_attrition = AttritionCounters()

        vocab_size = int(getattr(model.config, "vocab_size", 0))
        if vocab_size <= 0:
            raise WorkerFailedError("model.config.vocab_size must be positive for the pre-branch memory guard")

        from kvcot.discovery.constants import BRIDGE_TOKEN_COUNT, SCORED_HORIZON

        def _pre_branch_guard(target, kind):
            # Independent-audit Gate H5 repair: `check_pre_branch_memory`
            # now derives K/V-growth-across-the-branch-horizon, reallocation
            # headroom, and position-tracking growth itself from the
            # snapshot's own real tensor shapes and the frozen
            # bridge/scored-horizon counts -- this call site only supplies
            # the real vocabulary size and those frozen counts, never a
            # pre-computed opaque byte total.
            stage = f"pre-branch admission:{kind}:{target.event_plan.compaction_event_id}"
            execution_state.enter(stage)
            guard_evidence = check_pre_branch_memory(
                phase=f"{kind}:{target.event_plan.compaction_event_id}",
                cuda=cuda,
                snapshot=target.pristine_snapshot,
                selected_vector_bytes=target.persistent_tensor_bytes,
                vocab_size=vocab_size,
                bridge_token_count=BRIDGE_TOKEN_COUNT,
                scored_token_count=SCORED_HORIZON,
            )
            execution_state.complete(stage)
            return guard_evidence

        example_result = run_example(
            example_id=manifest.unique_id, model_revision=config.model.revision,
            rkv_revision=config.rkv.upstream_revision, provenance=provenance_record,
            prompt_token_ids=prompt_token_ids, pass1_initial_state=_fresh_state(),
            pass2_initial_state_factory=_fresh_state, prefill_fn=instrumented.prefill,
            decode_one_fn=instrumented.decode_one, snapshot_fn=instrumented.snapshot,
            max_new_tokens=config.generation.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
            answer_fn=timed_answer_verifier, num_hidden_layers=num_layers, num_key_value_heads=num_kv_heads,
            identity=identity,
            branch_step_fn=build_real_branch_step_fn_restore_once(
                model, _device, actual_calls, consume_owned_snapshot=True
            ),
            example_attrition=example_attrition, pair_attrition=pair_attrition,
            # B1B-R4 §7: exactly ONE no-op pair evaluation for the whole B2A
            # example, not one per selected event.
            pair_execution_policy=PairExecutionPolicy(no_op_mode=NoOpMode.B2A_SINGLE_CALIBRATION),
            pre_branch_guard=_pre_branch_guard,
            operation_runner=measured,
            pair_phase_runner=timer.measure,
            # The generic CPU harness retains a monotonic diagnostic default;
            # execute mode overrides it. Authoritative B2A timing is the
            # synchronized `measured`/`timer.measure` evidence above.
            clock_fn=_clock or time.perf_counter,
            # Independent-audit Gate H2.2 repair: times the REAL target
            # capture gather/gather-parity/absolute-position-parity
            # computation inside `capture.capture_update_kv`, under its own
            # accurately-named `capture_gather_and_parity` phase -- never
            # merely a later, unrelated trace comparison.
            capture_timer_fn=timer.measure,
        )
        if example_result.aborted:
            # Independent-audit Gate H1: `run_example` returns (rather than
            # raises) when its own pair-evaluation loop is cut short by an
            # unexpected exception (e.g. a real CUDA OOM mid-pair) so every
            # pair completed before the abort survives on `example_result`.
            # Promoting it to a raised exception HERE (inside this
            # function's own `try` block) means the outer partial-evidence
            # capture below sees `example_result` already bound in
            # `locals()` -- every pair/attrition/mutation record the
            # aborted example accumulated is threaded into the worker's
            # failure envelope, never silently reported as if the worker
            # had merely produced a smaller-than-expected count.
            #
            # R1: `run_example`'s own `operation_runner`/`pair_phase_runner`
            # wiring (both the SAME tracked `measured`/`timer.measure`
            # callables this worker's `execution_state` uses) already leaves
            # `current_stage` at the specific pass/pair sub-phase that was
            # executing when the abort happened -- overwriting it here
            # unconditionally would DISCARD that more specific evidence.
            # Only fall back to a typed, `invalid_stage`-derived label in the
            # defensive case where that tracking somehow left
            # `current_stage` colliding with `last_completed_stage` anyway.
            if execution_state.current_stage == execution_state.last_completed_stage:
                execution_state.enter(f"pair-evaluation loop abort:{example_result.invalid_stage}")
            raise RuntimeError(
                f"pair-evaluation loop aborted at stage={example_result.invalid_stage!r}: "
                f"{example_result.abort_failure_type}: {example_result.abort_failure_message}"
            )

        # Independent-audit Gate H2 repair: this phase used to be named
        # `capture_and_parity` -- but it only ever timed the comparison
        # below (`compare_call_boundary_traces`, a POST-Pass-2 call-
        # boundary-shape comparison), never the actual target capture,
        # K/V-gathering, survivor parity, or absolute-position parity work,
        # all of which genuinely happen EARLIER, inside `run_example`'s
        # Pass 2 step -- already-real, already-synchronized, non-
        # overlapping timings under `rkv_pass2_prefill`/`rkv_pass2_decode`/
        # `snapshot_creation` (recorded by `instrumented.prefill`/
        # `.decode_one`/`.snapshot` above). Renamed to match what THIS
        # specific measurement actually is; the real capture/parity timing
        # evidence lives in those other, already-required phases instead of
        # being fabricated under this one's now-accurate name.
        call_boundary_comparison = timer.measure(
            "call_trace_comparison",
            lambda: compare_call_boundary_traces(instrumented.pass1_trace, instrumented.pass2_trace),
        )
        # R1: every derivation below reads `example_result`/`instrumented`/
        # `actual_calls` after `run_example` has already returned -- none of
        # it was previously tracked, so an unexpected exception here (e.g. a
        # malformed `example_result` from an injected test double) reported
        # `failing_stage` as "call_trace_comparison" (already completed),
        # not this derivation step. (Kept OUTSIDE this block, not nested
        # inside it -- `WorkerExecutionState.enter` has no stack, so a
        # nested tracked call would leave `current_stage` stuck at its own
        # name for the rest of the block instead of reverting to this
        # block's.)
        with execution_state.track("post-run evidence derivation"):
            if not actual_calls.batch_size_verified:
                raise WorkerFailedError("no valid actual model-call batch evidence was recorded")
            batch_size = actual_calls.events[0].batch_size
            from kvcot.discovery.b2a_evidence import (
                build_no_op_evidence,
                build_replay_evidence,
                derive_compaction_positions,
                derive_failed_pair_identities,
            )

            pass1_compaction_positions, pass2_compaction_positions = derive_compaction_positions(example_result)
            compaction_lists_match = pass1_compaction_positions == pass2_compaction_positions
            trajectory = derive_trajectory_parity_evidence(
                pass2_result_valid=example_result.valid,
                pass2_invalid_reason=example_result.pass2_invalid_reason,
                call_boundary_all_match=call_boundary_comparison.all_match,
                target_capture_gather_parities=tuple(
                    ev.gather_parity_passed for ev in example_result.minimized_target_evidence
                ),
                target_capture_absolute_parities=tuple(
                    ev.absolute_position_parity_passed for ev in example_result.minimized_target_evidence
                ),
                compaction_lists_match=compaction_lists_match,
            )
            pair_completion = derive_pair_completion_evidence(
                trace=example_result.trace, example_result=example_result
            )
            semantic_swap_checks = derive_semantic_swap_check_evidence(example_result)
            pair_identity = derive_pair_identity_evidence(example_result)
            observed_retention_ratio = derive_observed_retention_ratio(example_result)
            no_op_parity = derive_no_op_numerical_parity(example_result)

            attempted_identities = list(example_result.attempted_pair_identities)
            completed_identities = list(example_result.completed_pair_identities)
            # F2: the SAME shared helpers `capture_partial_evidence` uses on the
            # failure path -- one implementation of each derivation, never two.
            failed_identities = derive_failed_pair_identities(
                attempted_identities, completed_identities, pair_completion.pair_failure_details
            )
            no_op_evidence = build_no_op_evidence(example_result)
            actual_call_export = actual_calls.export()
            replay_evidence = build_replay_evidence(
                example_result,
                pass1_events=instrumented.pass1_trace.events,
                pass2_events=instrumented.pass2_trace.events,
                actual_call_export=actual_call_export,
            )
            pass1_synchronized_seconds = next(
                (record.duration_seconds for record in timer.records if record.phase == "rkv_complete_pass1"), 0.0
            )
            pass2_synchronized_seconds = next(
                (record.duration_seconds for record in timer.records if record.phase == "rkv_complete_pass2"), 0.0
            )
            real_pair_synchronized_seconds = [
                record.duration_seconds for record in timer.records
                if record.phase.startswith("real_pair:") and record.phase.count(":") == 3
            ]
            no_op_synchronized_seconds = [
                record.duration_seconds for record in timer.records
                if record.phase.startswith("no_op_pair:") and record.phase.count(":") == 3
            ]

        execution_state.enter("result construction")
        selected_event_count_exact = pair_completion.selected_compaction_events == B2A_SELECTED_EVENTS
        real_pair_count_exact = (
            pair_completion.attempted_real_pair_count == B2A_REAL_PAIR_EVALUATIONS_TOTAL
            and pair_completion.completed_real_pair_count == B2A_REAL_PAIR_EVALUATIONS_TOTAL
        )
        no_op_count_exact = (
            pair_completion.attempted_no_op_pair_count == B2A_NOOP_PAIR_EVALUATIONS_TOTAL
            and pair_completion.completed_no_op_pair_count == B2A_NOOP_PAIR_EVALUATIONS_TOTAL
        )
        all_required_pair_evaluations_completed = (
            real_pair_count_exact and no_op_count_exact and pair_completion.failed_real_pair_count == 0
        )

        memory_meter.observe("rkv_complete_worker", lambda: None)
        timer.finish_span("rkv_complete_worker", complete_worker_started_at)
        peak_allocated = memory_meter.maximum_peak_allocated
        peak_reserved = memory_meter.maximum_peak_reserved
        first_memory = memory_meter.records[0]
        memory = MemoryEvidence(
            allocated_before_reset_bytes=first_memory.allocated_before,
            reserved_before_reset_bytes=first_memory.reserved_before,
            peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
            reset_point="explicit_phase_owned_resets",
        )
        runtime_generation = build_runtime_generation_record(
            batch_size=batch_size, max_new_tokens=config.generation.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
            attention_backend=config.generation.attention_backend, framework_seed=config.generation.framework_seed,
            prompt_token_count=len(prompt_token_ids),
        )

        return RKVWorkerResult(
            role="rkv",
            model_revision=config.model.revision,
            tokenizer_revision=config.model.tokenizer_revision,
            dataset_repo=manifest.dataset_repo,
            dataset_revision=manifest.dataset_revision,
            manifest_hash=manifest.manifest_hash(),
            prompt_token_ids_sha256=manifest.prompt_token_ids_sha256,
            prompt_token_count=len(prompt_token_ids),
            rkv_upstream_revision=config.rkv.upstream_revision,
            runtime_rkv_config_hash=runtime_check.runtime_hash,
            frozen_rkv_config_hash=runtime_check.frozen_hash,
            rkv_config_hash_match=runtime_check.passed,
            example_valid=example_result.valid,
            natural_answer_status=(
                answer_verifier.last_result.status if answer_verifier.last_result is not None else "unverifiable"
            ),
            token_identical_replay=trajectory.token_identical_replay,
            prefill_decode_boundary_parity=trajectory.prefill_decode_boundary_parity,
            compaction_position_equality=trajectory.compaction_position_equality,
            capture_gather_parity=trajectory.capture_gather_parity,
            absolute_position_parity=trajectory.absolute_position_parity,
            no_op_numerical_parity=no_op_parity,
            pass1_call_boundary={
                "prefill_call_count": instrumented.pass1_trace.prefill_call_count,
                "prefill_token_count": instrumented.pass1_trace.prefill_token_count,
                "decode_call_count": instrumented.pass1_trace.decode_call_count,
                "ordered_trace_hash": instrumented.pass1_trace.ordered_call_kinds_and_tokens_hash(),
            },
            pass2_call_boundary={
                "prefill_call_count": instrumented.pass2_trace.prefill_call_count,
                "prefill_token_count": instrumented.pass2_trace.prefill_token_count,
                "decode_call_count": instrumented.pass2_trace.decode_call_count,
                "ordered_trace_hash": instrumented.pass2_trace.ordered_call_kinds_and_tokens_hash(),
            },
            observed_total_compaction_events=pair_completion.observed_total_compaction_events,
            eligible_compaction_events=pair_completion.eligible_compaction_events,
            selected_compaction_events=pair_completion.selected_compaction_events,
            events_with_at_least_one_completed_real_pair=pair_completion.events_with_at_least_one_completed_real_pair,
            events_with_all_four_real_pairs_completed=pair_completion.events_with_all_four_real_pairs_completed,
            attempted_real_pair_count=pair_completion.attempted_real_pair_count,
            completed_real_pair_count=pair_completion.completed_real_pair_count,
            failed_real_pair_count=pair_completion.failed_real_pair_count,
            attempted_no_op_pair_count=pair_completion.attempted_no_op_pair_count,
            completed_no_op_pair_count=pair_completion.completed_no_op_pair_count,
            pair_failure_details=list(pair_completion.pair_failure_details),
            semantic_swap_checks_required=semantic_swap_checks.checks_required,
            semantic_swap_checks_attempted=semantic_swap_checks.checks_attempted,
            semantic_swap_checks_passed=semantic_swap_checks.checks_passed,
            semantic_swap_checks_failed=semantic_swap_checks.checks_failed,
            unique_completed_real_pair_count=pair_identity.unique_completed_real_pair_count,
            events_with_exactly_four_unique_real_pairs=pair_identity.events_with_exactly_four_unique_real_pairs,
            has_duplicate_real_pair_identity=pair_identity.has_duplicate_real_pair_identity,
            has_duplicate_no_op_pair_identity=pair_identity.has_duplicate_no_op_pair_identity,
            selected_event_count_exact=selected_event_count_exact,
            real_pair_count_exact=real_pair_count_exact,
            no_op_count_exact=no_op_count_exact,
            all_required_pair_evaluations_completed=all_required_pair_evaluations_completed,
            observed_retention_ratio=observed_retention_ratio,
            wall_seconds_pass1=pass1_synchronized_seconds,
            wall_seconds_pass2=pass2_synchronized_seconds,
            wall_seconds_targeted_capture=instrumented.targeted_capture_wall_seconds,
            real_pair_wall_seconds=real_pair_synchronized_seconds,
            no_op_pair_wall_seconds=no_op_synchronized_seconds,
            determinism_policy=determinism_policy.__dict__,
            runtime_generation=runtime_generation.__dict__,
            runtime_generation_config_hash=runtime_generation.canonical_hash(),
            parameter_placement=parameter_placement.__dict__,
            runtime_identity=runtime_identity.__dict__,
            memory={
                **memory.__dict__,
                "pre_branch_guards": [evidence.__dict__ for evidence in example_result.pre_branch_memory_evidence],
            },
            minimized_target_evidence=[ev.__dict__ for ev in example_result.minimized_target_evidence],
            peak_cuda_allocated_bytes=peak_allocated,
            peak_cuda_reserved_bytes=peak_reserved,
            every_parameter_on_cuda=parameter_placement.every_parameter_on_cuda,
            batch_size=batch_size,
            actual_batch_size_verified=actual_calls.batch_size_verified,
            actual_call_evidence=actual_call_export,
            snapshot_evidence={
                "verified": model_snapshot is not None and tokenizer_snapshot is not None,
                "model": None if model_snapshot is None else model_snapshot.__dict__,
                "tokenizer": None if tokenizer_snapshot is None else tokenizer_snapshot.__dict__,
            },
            device_evidence=device_evidence,
            selected_event_evidence=list(example_result.selected_event_evidence),
            attempted_pair_identities=attempted_identities,
            completed_pair_identities=completed_identities,
            failed_pair_identities=failed_identities,
            no_op_identity=next(
                (identity for identity in attempted_identities if identity["pair_kind"] == "no_op"), None
            ),
            semantic_mutation_reports=list(example_result.semantic_mutation_reports),
            no_op_evidence=no_op_evidence,
            replay_evidence=replay_evidence,
            dataset_row_identity={
                "dataset_repo": manifest.dataset_repo,
                "dataset_revision": manifest.dataset_revision,
                "example_index": manifest.example_index,
                "unique_id": manifest.unique_id,
                "raw_content_hash": getattr(manifest, "raw_content_hash", None),
                "manifest_canonical_hash": manifest.manifest_hash(),
                "rendered_user_message_sha256": getattr(manifest, "rendered_user_message_sha256", None),
                "chat_template_source_sha256": getattr(manifest, "chat_template_source_sha256", None),
                "prompt_token_ids_sha256": manifest.prompt_token_ids_sha256,
                "prompt_token_count": len(prompt_token_ids),
            },
            timing_evidence=timer.export(),
            memory_phase_evidence=memory_meter.export(),
            software_versions={"torch": torch.__version__},
            # B2A-R2 forensic repair
            # (docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md):
            # the complete scientific SwapPairRecord population, straight
            # from ExampleResult.pair_records -- never reconstructed from
            # identities or summaries (that data exists nowhere else once
            # this function returns).
            pair_records=list(example_result.pair_records),
        ).model_dump(mode="json")
    except Exception as exc:
        from kvcot.discovery.worker_partial_evidence import raise_worker_body_failure

        raise_worker_body_failure(role="rkv", execution_state=execution_state, exc=exc, scope=locals())
