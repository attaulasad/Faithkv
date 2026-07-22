"""Durable worker-attempt envelopes (B1B-R4 §16). Every B2A worker
subprocess ALWAYS attempts to write one of these -- on success AND on
failure -- so a coordinator (or a human auditing `results/` after a crash)
never has to guess what a worker was doing when it died. Pure Python, no
torch import (the envelope itself is metadata, not a measurement).
"""
from __future__ import annotations

import platform
import os
import shutil
import sys
import traceback as traceback_module
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kvcot.utils.hashing import sha256_text
from kvcot.utils.hashing import sha256_json


class WorkerEnvelope(BaseModel):
    role: str
    attempt_id: str
    started_at: str
    finished_at: str
    success: bool
    requested_identities: dict[str, Any]
    resolved_identities: dict[str, Any]
    partial_measurements: dict[str, Any] | None
    determinism_policy: dict[str, Any] | None
    software_versions: dict[str, str]
    hardware_metadata: dict[str, Any]
    error_type: str | None
    error_message: str | None
    traceback: str | None
    result_sha256: str | None = None
    # Independent-audit Gate H1: explicit, typed, top-level fields for the
    # scientifically load-bearing scalars a post-mortem needs first --
    # never buried only inside the unconstrained `partial_measurements`
    # blob. `None`/`False` for a success envelope or for a failure that
    # never reached `kvcot.discovery.worker_partial_evidence
    # .capture_partial_evidence` at all (e.g. config load failed before any
    # worker body ran).
    failure_stage: str | None = None
    last_completed_stage: str | None = None
    is_oom: bool = False
    is_timeout: bool = False


def new_attempt_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_success_envelope(
    *,
    role: str,
    attempt_id: str,
    started_at: str,
    requested_identities: dict[str, Any],
    resolved_identities: dict[str, Any],
    result_payload: dict[str, Any],
    determinism_policy: dict[str, Any] | None,
    software_versions: dict[str, str],
    hardware_metadata: dict[str, Any],
) -> WorkerEnvelope:
    return WorkerEnvelope(
        role=role, attempt_id=attempt_id, started_at=started_at, finished_at=now_iso(), success=True,
        requested_identities=requested_identities, resolved_identities=resolved_identities,
        partial_measurements=result_payload, determinism_policy=determinism_policy,
        software_versions=software_versions, hardware_metadata=hardware_metadata,
        error_type=None, error_message=None, traceback=None,
        result_sha256=sha256_json(result_payload),
    )


def build_failure_envelope(
    *,
    role: str,
    attempt_id: str,
    started_at: str,
    requested_identities: dict[str, Any],
    resolved_identities: dict[str, Any],
    partial_measurements: dict[str, Any] | None,
    determinism_policy: dict[str, Any] | None,
    software_versions: dict[str, str],
    hardware_metadata: dict[str, Any],
    exc: BaseException,
    failure_stage: str | None = None,
    last_completed_stage: str | None = None,
    is_oom: bool = False,
    is_timeout: bool = False,
) -> WorkerEnvelope:
    return WorkerEnvelope(
        role=role, attempt_id=attempt_id, started_at=started_at, finished_at=now_iso(), success=False,
        requested_identities=requested_identities, resolved_identities=resolved_identities,
        partial_measurements=partial_measurements, determinism_policy=determinism_policy,
        software_versions=software_versions, hardware_metadata=hardware_metadata,
        error_type=type(exc).__name__, error_message=str(exc), traceback=traceback_module.format_exc(),
        result_sha256=None, failure_stage=failure_stage, last_completed_stage=last_completed_stage,
        is_oom=is_oom, is_timeout=is_timeout,
    )


def default_hardware_metadata() -> dict[str, Any]:
    """CPU-safe defaults -- GPU-specific fields (device name, CUDA
    capability) are filled in by the caller when `torch.cuda` is available;
    this function itself never imports torch."""
    memory_bytes = None
    try:
        import psutil

        memory_bytes = int(psutil.virtual_memory().total)
    except ImportError:
        pass
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "ram_bytes": memory_bytes,
        "working_directory_free_disk_bytes": shutil.disk_usage(Path.cwd()).free,
    }


def write_worker_envelope(envelope: WorkerEnvelope, output_path: Path) -> Path:
    """Write the mandatory atomic envelope beside its authoritative result.

    A successful result is invalid unless this immutable envelope exists
    and its ``result_sha256`` matches the result payload.
    """
    envelope_path = output_path.with_suffix(output_path.suffix + ".envelope.json")
    from kvcot.discovery.attempt_artifacts import atomic_write_json

    return atomic_write_json(envelope_path, envelope.model_dump(mode="json"))


def envelope_hash(envelope: WorkerEnvelope) -> str:
    return sha256_text(envelope.model_dump_json())
