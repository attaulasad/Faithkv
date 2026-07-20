"""FullKV / R-KV process separation for B2A (B1B-R3 §11). The repository
already prohibits loading stock (FullKV) and patched (R-KV) models in one
Python process (`kvcot.generation.state.declare_process_mode`,
`ProcessModeConflictError`) -- B1B-R2's `b2a_execute.run_b2a_calibration`
nonetheless pretended the single R-KV run's timing was ALSO the FullKV
timing, in direct contradiction of that rule. This module is the repair:
a coordinator that launches two SEPARATE OS processes (via `subprocess`,
never `multiprocessing` or in-process mode switching) and combines their
independently-produced, schema-validated JSON results.

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
/test_b2a_workers.py`). `run_fullkv_worker`/`run_rkv_worker` (the functions
`kvcot.discovery.b2a_worker_entry`'s real `__main__` block calls) are
GPU-only and never invoked by any test or code path in this pass -- every
`import torch`/`transformers` reference is deferred, matching this
repository's existing discipline.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

SubprocessRunner = Callable[[Sequence[str]], "subprocess.CompletedProcess"]


class WorkerFailedError(RuntimeError):
    pass


class FullKVWorkerResult(BaseModel):
    """Every field B1B-R3 §11 requires the FullKV worker to report. Every
    field is REQUIRED (pydantic default: no field is `Optional` unless
    explicitly declared so) -- a worker that cannot measure one of these
    must fail outright, never omit it."""

    role: str = Field(pattern="^fullkv$")
    model_revision: str
    tokenizer_revision: str
    dataset_repo: str
    dataset_revision: str
    manifest_hash: str
    prompt_token_ids_sha256: str
    natural_generated_token_ids: list[int]
    natural_answer: str | None
    natural_answer_status: str
    wall_seconds: float = Field(ge=0.0)
    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_cuda_reserved_bytes: int = Field(ge=0)
    every_parameter_on_cuda: bool
    batch_size: int
    software_versions: dict[str, str]


class RKVWorkerResult(BaseModel):
    role: str = Field(pattern="^rkv$")
    model_revision: str
    tokenizer_revision: str
    dataset_repo: str
    dataset_revision: str
    manifest_hash: str
    prompt_token_ids_sha256: str
    rkv_upstream_revision: str
    runtime_rkv_config_hash: str
    frozen_rkv_config_hash: str
    example_valid: bool
    event_count: int
    observed_retention_ratio: float = Field(ge=0.0, le=1.0)
    no_op_numerical_parity: bool
    natural_answer_status: str
    wall_seconds_pass1: float = Field(ge=0.0)
    wall_seconds_pass2: float = Field(ge=0.0)
    wall_seconds_targeted_capture: float = Field(ge=0.0)
    wall_seconds_cache_clone_restore: float = Field(ge=0.0)
    wall_seconds_one_swap: float = Field(ge=0.0)
    wall_seconds_bridge_plus_48_scored: float = Field(ge=0.0)
    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_cuda_reserved_bytes: int = Field(ge=0)
    every_parameter_on_cuda: bool
    batch_size: int
    software_versions: dict[str, str]


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


def _default_python_executable() -> str:
    return sys.executable


def _launch_worker(
    role: str,
    config_path: str,
    manifest_path: str,
    output_path: Path,
    python_executable: str,
    subprocess_runner: SubprocessRunner,
) -> subprocess.CompletedProcess:
    argv = [
        python_executable, "-m", "kvcot.discovery.b2a_worker_entry",
        "--role", role, "--config", config_path, "--manifest", manifest_path, "--output", str(output_path),
    ]
    return subprocess_runner(argv)


def run_both_workers_via_subprocess(
    config_path: str,
    manifest_path: str,
    *,
    python_executable: str | None = None,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> WorkerCoordinationResult:
    """The coordinator's one entry point: creates a unique temp directory,
    launches the FullKV and R-KV workers as SEPARATE subprocess invocations
    (never in one process, never via `multiprocessing`), reads back and
    schema-validates each worker's JSON output, checks shared-identity
    agreement, and cleans up the temp directory only after the combined
    result is safely constructed. Raises `WorkerFailedError` (naming which
    worker and its exit status) on any worker failure -- the caller
    (`kvcot.discovery.b2a_execute`) is responsible for writing a FAILURE
    artifact from that exception, never silently swallowing it."""
    python_executable = python_executable or _default_python_executable()
    tmp_dir = Path(tempfile.mkdtemp(prefix="kvcot-b2a-workers-"))
    try:
        fullkv_output = tmp_dir / "fullkv_result.json"
        rkv_output = tmp_dir / "rkv_result.json"

        fullkv_proc = _launch_worker("fullkv", config_path, manifest_path, fullkv_output, python_executable, subprocess_runner)
        if fullkv_proc.returncode != 0:
            raise WorkerFailedError(
                f"fullkv worker exited with code {fullkv_proc.returncode}: "
                f"{getattr(fullkv_proc, 'stderr', '')!r}"
            )
        rkv_proc = _launch_worker("rkv", config_path, manifest_path, rkv_output, python_executable, subprocess_runner)
        if rkv_proc.returncode != 0:
            raise WorkerFailedError(
                f"rkv worker exited with code {rkv_proc.returncode}: {getattr(rkv_proc, 'stderr', '')!r}"
            )

        if not fullkv_output.exists():
            raise WorkerFailedError(f"fullkv worker reported success but wrote no output file at {fullkv_output}")
        if not rkv_output.exists():
            raise WorkerFailedError(f"rkv worker reported success but wrote no output file at {rkv_output}")

        fullkv_result = FullKVWorkerResult.model_validate_json(fullkv_output.read_text(encoding="utf-8"))
        rkv_result = RKVWorkerResult.model_validate_json(rkv_output.read_text(encoding="utf-8"))

        shared_ok, mismatches = validate_shared_identity(fullkv_result, rkv_result)
        return WorkerCoordinationResult(
            fullkv=fullkv_result, rkv=rkv_result, shared_identity_ok=shared_ok,
            shared_identity_mismatches=tuple(mismatches),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# GPU-only worker bodies -- never invoked by any test or code path in this
# pass. Called only from kvcot.discovery.b2a_worker_entry's __main__ block,
# which is itself only ever launched as a subprocess by
# run_both_workers_via_subprocess above (never imported/executed directly
# by CPU test code).
# --------------------------------------------------------------------------


def run_fullkv_worker(config: Any, manifest: Any) -> dict:
    """Runs exactly one frozen example through stock FullKV and reports
    identity/timing/memory/answer evidence (B1B-R3 §11). Requires CUDA;
    never invoked in this pass."""
    import time

    import torch

    from kvcot.discovery.no_offload import assert_no_offloaded_parameters
    from kvcot.generation.policies import FullKVPolicy
    from kvcot.generation.state import reset_patched_state

    if not torch.cuda.is_available():
        raise WorkerFailedError("run_fullkv_worker requires CUDA; none is available.")

    policy = FullKVPolicy()
    model = policy.load(config.model.name, config.model.revision, getattr(torch, config.model.dtype), config.generation.attention_backend)
    assert_no_offloaded_parameters(model)

    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.decode import generate_base

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name, revision=config.model.tokenizer_revision, use_fast=True)

    prompt_token_ids = list(manifest.prompt_token_ids)
    cache = reset_patched_state(model, lambda: DynamicCache())
    start = time.monotonic()
    result = generate_base(
        model, cache, prompt_token_ids, config.generation.max_new_tokens, 0.0, 1.0, None, tokenizer.eos_token_id, "cuda",
    )
    wall_seconds = time.monotonic() - start

    from kvcot.discovery.math500_verification import build_math500_answer_fn

    answer_fn = build_math500_answer_fn(tokenizer, manifest.gold_answer)
    natural_answer, natural_answer_status = answer_fn(list(result.generated_token_ids))

    return FullKVWorkerResult(
        role="fullkv",
        model_revision=config.model.revision,
        tokenizer_revision=config.model.tokenizer_revision,
        dataset_repo=manifest.dataset_repo,
        dataset_revision=manifest.dataset_revision,
        manifest_hash=manifest.manifest_hash(),
        prompt_token_ids_sha256=manifest.prompt_token_ids_sha256,
        natural_generated_token_ids=list(result.generated_token_ids),
        natural_answer=natural_answer,
        natural_answer_status=natural_answer_status,
        wall_seconds=wall_seconds,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        every_parameter_on_cuda=True,
        batch_size=1,
        software_versions={"torch": torch.__version__},
    ).model_dump(mode="json")


def run_rkv_worker(config: Any, manifest: Any) -> dict:
    """Runs Pass 1, Pass 2, targeted capture, branch evaluation, and the
    B2A no-op calibration for exactly one example under R-KV, and reports
    the resulting evidence (B1B-R3 §11). Requires CUDA; never invoked in
    this pass. Delegates the actual pass/branch machinery entirely to
    `kvcot.discovery.b2a_execute.run_b2a_calibration`'s own building
    blocks -- never a second, independently-written execution path."""
    raise NotImplementedError(
        "run_rkv_worker's real-model body is wired through kvcot.discovery.b2a_execute.run_b2a_calibration "
        "-- kvcot.discovery.b2a_worker_entry's __main__ block calls that function directly for the 'rkv' "
        "role rather than duplicating it here. This stub exists so the module has a symmetrical, documented "
        "seam matching run_fullkv_worker; b2a_execute.run_b2a_calibration is the actual GPU entry point."
    )
