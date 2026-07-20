"""Durable worker-attempt envelopes (B1B-R4 §16). Every B2A worker
subprocess ALWAYS attempts to write one of these -- on success AND on
failure -- so a coordinator (or a human auditing `results/` after a crash)
never has to guess what a worker was doing when it died. Pure Python, no
torch import (the envelope itself is metadata, not a measurement).
"""
from __future__ import annotations

import platform
import sys
import traceback as traceback_module
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kvcot.utils.hashing import sha256_text


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
) -> WorkerEnvelope:
    return WorkerEnvelope(
        role=role, attempt_id=attempt_id, started_at=started_at, finished_at=now_iso(), success=False,
        requested_identities=requested_identities, resolved_identities=resolved_identities,
        partial_measurements=partial_measurements, determinism_policy=determinism_policy,
        software_versions=software_versions, hardware_metadata=hardware_metadata,
        error_type=type(exc).__name__, error_message=str(exc), traceback=traceback_module.format_exc(),
    )


def default_hardware_metadata() -> dict[str, Any]:
    """CPU-safe defaults -- GPU-specific fields (device name, CUDA
    capability) are filled in by the caller when `torch.cuda` is available;
    this function itself never imports torch."""
    return {"platform": platform.platform(), "python_version": sys.version}


def write_worker_envelope(envelope: WorkerEnvelope, output_path: Path) -> Path:
    """Best-effort, non-atomic write (envelopes are diagnostic, never the
    authoritative worker result -- `--output` remains that) -- but still
    always attempted, even from within an `except` block handling the
    worker's own failure."""
    envelope_path = output_path.with_suffix(output_path.suffix + ".envelope.json")
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    envelope_path.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")
    return envelope_path


def envelope_hash(envelope: WorkerEnvelope) -> str:
    return sha256_text(envelope.model_dump_json())
