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
this repair removes. Both are GPU-only and never invoked by any test in
this pass except via injected fake backends
(`tests/unit/discovery/test_b2a_workers_real_bodies.py`, B1B-R4 §20) --
every `import torch`/`transformers` reference stays deferred, matching this
repository's existing discipline.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from kvcot.discovery.attrition import PairFailureDetail
from kvcot.discovery.call_trace import CallBoundaryEvent, CallTraceRecorder, compare_call_boundary_traces
from kvcot.discovery.constants import B2A_REAL_PAIR_EVALUATIONS_TOTAL, B2A_SELECTED_EVENTS

SubprocessRunner = Callable[..., "subprocess.CompletedProcess"]


class WorkerFailedError(RuntimeError):
    pass


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
    software_versions: dict[str, str]


class RKVWorkerResult(BaseModel):
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
    tmp_dir = Path(tempfile.mkdtemp(prefix="kvcot-b2a-workers-"))
    try:
        fullkv_output = tmp_dir / "fullkv_result.json"
        rkv_output = tmp_dir / "rkv_result.json"

        try:
            fullkv_proc = _launch_worker(
                "fullkv", config_path, manifest_path, fullkv_output, python_executable, subprocess_runner,
                B2A_WORKER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            err = WorkerFailedError(f"fullkv worker timed out after {B2A_WORKER_TIMEOUT_SECONDS}s: {exc}")
            err.partial_fullkv_result = None  # type: ignore[attr-defined]
            err.timed_out = True  # type: ignore[attr-defined]
            raise err from exc
        if fullkv_proc.returncode != 0:
            err = WorkerFailedError(
                f"fullkv worker exited with code {fullkv_proc.returncode}: "
                f"stdout={getattr(fullkv_proc, 'stdout', '')!r} stderr={getattr(fullkv_proc, 'stderr', '')!r}"
            )
            err.partial_fullkv_result = None  # type: ignore[attr-defined]
            raise err

        if not fullkv_output.exists():
            raise WorkerFailedError(f"fullkv worker reported success but wrote no output file at {fullkv_output}")
        fullkv_result = FullKVWorkerResult.model_validate_json(fullkv_output.read_text(encoding="utf-8"))

        try:
            rkv_proc = _launch_worker(
                "rkv", config_path, manifest_path, rkv_output, python_executable, subprocess_runner,
                B2A_WORKER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            err = WorkerFailedError(f"rkv worker timed out after {B2A_WORKER_TIMEOUT_SECONDS}s: {exc}")
            err.partial_fullkv_result = fullkv_result  # type: ignore[attr-defined]
            err.timed_out = True  # type: ignore[attr-defined]
            raise err from exc
        if rkv_proc.returncode != 0:
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

        shared_ok, mismatches = validate_shared_identity(fullkv_result, rkv_result)
        return WorkerCoordinationResult(
            fullkv=fullkv_result, rkv=rkv_result, shared_identity_ok=shared_ok,
            shared_identity_mismatches=tuple(mismatches),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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

    def __init__(self, prefill_fn, decode_one_fn, snapshot_fn):
        self._prefill_fn = prefill_fn
        self._decode_one_fn = decode_one_fn
        self._snapshot_fn = snapshot_fn
        self.pass1_trace = CallTraceRecorder(prefill_fn, decode_one_fn)
        self.pass2_trace = CallTraceRecorder(prefill_fn, decode_one_fn)
        self._prefill_call_count = 0
        self.pass1_wall_seconds = 0.0
        self.pass2_wall_seconds = 0.0
        self.targeted_capture_wall_seconds = 0.0

    def _active_recorder(self) -> CallTraceRecorder:
        return self.pass1_trace if self._prefill_call_count <= 1 else self.pass2_trace

    def prefill(self, state, prompt_token_ids):
        self._prefill_call_count += 1
        recorder = self._active_recorder()
        prompt_token_ids = list(prompt_token_ids)
        recorder.events.append(CallBoundaryEvent(kind="prefill", token_ids=tuple(prompt_token_ids)))
        start = time.monotonic()
        result = self._prefill_fn(state, prompt_token_ids)
        elapsed = time.monotonic() - start
        if self._prefill_call_count <= 1:
            self.pass1_wall_seconds += elapsed
        else:
            self.pass2_wall_seconds += elapsed
        return result

    def decode_one(self, state, token_id):
        recorder = self._active_recorder()
        recorder.events.append(CallBoundaryEvent(kind="decode", token_ids=(token_id,)))
        start = time.monotonic()
        result = self._decode_one_fn(state, token_id)
        elapsed = time.monotonic() - start
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
        start = time.monotonic()
        result = self._snapshot_fn(state)
        elapsed = time.monotonic() - start
        self.targeted_capture_wall_seconds += elapsed
        self.pass2_wall_seconds += elapsed
        return result


# --------------------------------------------------------------------------
# Canonical worker bodies (B1B-R4 §19). GPU-only -- never invoked by any
# test or code path in this pass except via injected fake backends
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
    _device: str = "cuda",
) -> dict:
    """Runs exactly one frozen example through stock FullKV using the
    IDENTICAL greedy natural-run loop R-KV's Pass 1 uses
    (`kvcot.discovery.pass1.run_natural_pass1` +
    `kvcot.discovery.real_model_adapter`'s real `PrefillFn`/`DecodeOneFn`)
    -- B1B-R4 §5 repair: no sampling function is ever called (no
    `temperature`/`top_p`/`generator`), argmax token selection, EOS never
    appended or fed, exactly one prefill call, one decode call per
    generated token. Reports identity/timing/memory/answer/call-boundary
    evidence (B1B-R4 §6/§9/§10/§11/§14). Requires CUDA; never invoked in
    this pass except via an injected fake backend in a CPU test.

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
        derive_batch_size_from_input_ids,
        derive_parameter_placement,
        derive_runtime_identity,
    )
    from kvcot.generation.provenance import LayerProvenance, ModelProvenance
    from kvcot.generation.replay import CompactionTracker
    from kvcot.generation.state import reset_patched_state

    cuda = _cuda if _cuda is not None else torch.cuda
    cuda_available = bool(cuda.is_available())
    if not cuda_available and _load_model is None:
        # Only the REAL (unfaked) production path requires CUDA -- a CPU
        # test that injects `_load_model`/`_cuda` is exercising this
        # function's control flow deliberately, never claiming a real GPU
        # ran anything.
        raise WorkerFailedError("run_fullkv_worker requires CUDA; none is available.")

    # B1B-R4 §6: applied independently in THIS worker's own process, before
    # any model inference -- never assumed shared with the R-KV worker
    # process (a separate OS process, per B1B-R3 §11's process-separation
    # requirement).
    determinism_policy = apply_framework_seed(
        config.generation.framework_seed, config.generation.attention_backend, cuda_available=cuda_available,
    )

    if _load_model is not None:
        model = _load_model()
    else:
        from kvcot.generation.policies import FullKVPolicy

        policy = FullKVPolicy()
        model = policy.load(
            config.model.name, config.model.revision, getattr(torch, config.model.dtype),
            config.generation.attention_backend,
        )
    assert_no_offloaded_parameters(model)
    parameter_placement = derive_parameter_placement(model)

    if _load_tokenizer is not None:
        tokenizer = _load_tokenizer()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name, revision=config.model.tokenizer_revision, use_fast=True
        )
    runtime_identity = derive_runtime_identity(
        model=model, tokenizer=tokenizer, requested_model_revision=config.model.revision,
        requested_tokenizer_revision=config.model.tokenizer_revision,
    )

    num_layers = len(model.model.layers)
    num_kv_heads = model.config.num_key_value_heads

    # B1B-R4 §14: reset peak memory stats AFTER load, BEFORE measured
    # inference -- current model allocation is therefore included in the
    # reset baseline, matching the R-KV worker's identical reset point.
    cuda.synchronize()
    allocated_before = int(cuda.memory_allocated())
    reserved_before = int(cuda.memory_reserved())
    cuda.reset_peak_memory_stats()

    if _fresh_cache_factory is not None:
        cache_factory = _fresh_cache_factory
    else:
        from transformers.cache_utils import DynamicCache

        cache_factory = lambda: DynamicCache()  # noqa: E731

    cache = reset_patched_state(model, cache_factory)
    provenance = ModelProvenance(layers={i: LayerProvenance.empty(num_kv_heads) for i in range(num_layers)})
    state = RealModelState(
        model=model, cache=cache, model_provenance=provenance, compaction=CompactionTracker(),
        absolute_position=0, device=_device,
    )

    prompt_token_ids = list(manifest.prompt_token_ids)
    batch_size = derive_batch_size_from_input_ids(torch.tensor([prompt_token_ids]))

    recorder = CallTraceRecorder(build_real_prefill_fn(_device), build_real_decode_one_fn(_device))
    answer_fn = build_math500_answer_fn(tokenizer, manifest.gold_answer)
    provenance_record = NaturalRunProvenance(
        model_name=config.model.name, model_revision=config.model.revision,
        tokenizer_name=config.model.tokenizer_name, tokenizer_revision=config.model.tokenizer_revision,
        rkv_revision=config.rkv.upstream_revision, config_sha256=canonical_config_hash(config),
        dataset_name=manifest.dataset_repo, example_id=manifest.unique_id,
    )

    start = time.monotonic()
    trace = run_natural_pass1(
        provenance_record, prompt_token_ids, state, recorder.prefill, recorder.decode_one,
        config.generation.max_new_tokens, tokenizer.eos_token_id, answer_fn,
    )
    wall_seconds = time.monotonic() - start

    cuda.synchronize()
    peak_allocated = int(cuda.max_memory_allocated())
    peak_reserved = int(cuda.max_memory_reserved())
    memory = MemoryEvidence(
        allocated_before_reset_bytes=allocated_before, reserved_before_reset_bytes=reserved_before,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
        reset_point=RESET_POINT_AFTER_LOAD_BEFORE_INFERENCE,
    )

    runtime_generation = build_runtime_generation_record(
        batch_size=batch_size, max_new_tokens=config.generation.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
        attention_backend=config.generation.attention_backend, framework_seed=config.generation.framework_seed,
        prompt_token_count=len(prompt_token_ids),
    )

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
        software_versions={"torch": torch.__version__},
    ).model_dump(mode="json")


def run_rkv_worker(
    config: Any,
    manifest: Any,
    *,
    _load_model: Callable[[], Any] | None = None,
    _load_tokenizer: Callable[[], Any] | None = None,
    _fresh_cache_factory: Callable[[], Any] | None = None,
    _cuda: Any | None = None,
    _device: str = "cuda",
) -> dict:
    """Runs Pass 1, Pass 2, targeted capture, branch evaluation, and the
    B2A single no-op calibration for exactly one example under R-KV, and
    reports the resulting evidence (B1B-R4 §19: the ONE canonical R-KV
    worker body -- supersedes the B1B-R3 split between a `NotImplementedError`
    stub here and the real body in `kvcot.discovery.b2a_execute
    .run_rkv_worker_body`). Requires CUDA; never invoked in this pass except
    via an injected fake backend in a CPU test. Delegates the actual pass/
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
        derive_batch_size_from_input_ids,
        derive_parameter_placement,
        derive_runtime_identity,
    )
    from kvcot.discovery.runtime_rkv_verification import verify_runtime_matches_frozen
    from kvcot.discovery.sampling import IdentitySeedParts
    from kvcot.generation.provenance import LayerProvenance, ModelProvenance
    from kvcot.generation.replay import CompactionTracker
    from kvcot.generation.state import reset_patched_state

    cuda = _cuda if _cuda is not None else torch.cuda
    cuda_available = bool(cuda.is_available())
    if not cuda_available and _load_model is None:
        raise WorkerFailedError("run_rkv_worker requires CUDA; none is available.")

    determinism_policy = apply_framework_seed(
        config.generation.framework_seed, config.generation.attention_backend, cuda_available=cuda_available,
    )

    if _load_model is not None:
        model = _load_model()
    else:
        from kvcot.generation.policies import RKVPolicy

        policy = RKVPolicy(
            budget=config.rkv.budget, window_size=config.rkv.window_size, mix_lambda=config.rkv.mix_lambda,
            retain_ratio=config.rkv.retain_ratio, retain_direction=config.rkv.retain_direction,
            divide_method=config.rkv.divide_method, divide_length=config.rkv.divide_length,
            compression_content=config.rkv.compression_content, kernel_size=config.rkv.kernel_size,
        )
        dtype = getattr(torch, config.model.dtype)
        model = policy.load(config.model.name, config.model.revision, dtype, config.generation.attention_backend)
    assert_no_offloaded_parameters(model)
    parameter_placement = derive_parameter_placement(model)

    runtime_check = verify_runtime_matches_frozen(config.rkv, model)
    if not runtime_check.passed:
        raise WorkerFailedError(
            f"runtime R-KV configuration disagrees with the frozen config on: {runtime_check.mismatched_fields} "
            f"(frozen_hash={runtime_check.frozen_hash}, runtime_hash={runtime_check.runtime_hash})"
        )

    if _load_tokenizer is not None:
        tokenizer = _load_tokenizer()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name, revision=config.model.tokenizer_revision, use_fast=True
        )
    runtime_identity = derive_runtime_identity(
        model=model, tokenizer=tokenizer, requested_model_revision=config.model.revision,
        requested_tokenizer_revision=config.model.tokenizer_revision,
    )
    num_layers = len(model.model.layers)
    num_kv_heads = model.config.num_key_value_heads

    cuda.synchronize()
    allocated_before = int(cuda.memory_allocated())
    reserved_before = int(cuda.memory_reserved())
    cuda.reset_peak_memory_stats()

    if _fresh_cache_factory is not None:
        cache_factory = _fresh_cache_factory
    else:
        from transformers.cache_utils import DynamicCache

        cache_factory = lambda: DynamicCache()  # noqa: E731

    def _fresh_state() -> RealModelState:
        cache = reset_patched_state(model, cache_factory)
        provenance = ModelProvenance(layers={i: LayerProvenance.empty(num_kv_heads) for i in range(num_layers)})
        return RealModelState(
            model=model, cache=cache, model_provenance=provenance, compaction=CompactionTracker(),
            absolute_position=0, device=_device,
        )

    instrumented = _RkvHarnessInstrumentation(
        build_real_prefill_fn(_device), build_real_decode_one_fn(_device), build_real_snapshot_fn(),
    )

    identity = IdentitySeedParts(
        global_seed=config.generation.framework_seed, dataset_name=manifest.dataset_repo,
        problem_index=manifest.example_index, model_revision=config.model.revision,
        rkv_revision=config.rkv.upstream_revision,
    )
    answer_verifier = build_math500_answer_fn(tokenizer, manifest.gold_answer)
    provenance_record = NaturalRunProvenance(
        model_name=config.model.name, model_revision=config.model.revision,
        tokenizer_name=config.model.tokenizer_name, tokenizer_revision=config.model.tokenizer_revision,
        rkv_revision=config.rkv.upstream_revision, config_sha256=canonical_config_hash(config),
        dataset_name=manifest.dataset_repo, example_id=manifest.unique_id,
    )

    prompt_token_ids = list(manifest.prompt_token_ids)
    assert len(prompt_token_ids) > 0, "structurally impossible: an empty prompt must never reach Pass 1"
    batch_size = derive_batch_size_from_input_ids(torch.tensor([prompt_token_ids]))

    example_attrition = AttritionCounters()
    pair_attrition = AttritionCounters()

    example_result = run_example(
        example_id=manifest.unique_id, model_revision=config.model.revision,
        rkv_revision=config.rkv.upstream_revision, provenance=provenance_record,
        prompt_token_ids=prompt_token_ids, pass1_initial_state=_fresh_state(),
        pass2_initial_state_factory=_fresh_state, prefill_fn=instrumented.prefill,
        decode_one_fn=instrumented.decode_one, snapshot_fn=instrumented.snapshot,
        max_new_tokens=config.generation.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
        answer_fn=answer_verifier, num_hidden_layers=num_layers, num_key_value_heads=num_kv_heads,
        identity=identity, branch_step_fn=build_real_branch_step_fn_restore_once(model, _device),
        example_attrition=example_attrition, pair_attrition=pair_attrition,
        # B1B-R4 §7: exactly ONE no-op pair evaluation for the whole B2A
        # example, not one per selected event.
        pair_execution_policy=PairExecutionPolicy(no_op_mode=NoOpMode.B2A_SINGLE_CALIBRATION),
    )

    call_boundary_comparison = compare_call_boundary_traces(instrumented.pass1_trace, instrumented.pass2_trace)
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
    )
    pair_completion = derive_pair_completion_evidence(trace=example_result.trace, example_result=example_result)
    semantic_swap_checks = derive_semantic_swap_check_evidence(example_result)
    pair_identity = derive_pair_identity_evidence(example_result)
    observed_retention_ratio = derive_observed_retention_ratio(example_result)
    no_op_parity = derive_no_op_numerical_parity(example_result)

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

    cuda.synchronize()
    peak_allocated = int(cuda.max_memory_allocated())
    peak_reserved = int(cuda.max_memory_reserved())
    memory = MemoryEvidence(
        allocated_before_reset_bytes=allocated_before, reserved_before_reset_bytes=reserved_before,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
        reset_point=RESET_POINT_AFTER_LOAD_BEFORE_INFERENCE,
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
        wall_seconds_pass1=instrumented.pass1_wall_seconds,
        wall_seconds_pass2=instrumented.pass2_wall_seconds,
        wall_seconds_targeted_capture=instrumented.targeted_capture_wall_seconds,
        real_pair_wall_seconds=list(example_result.real_pair_wall_seconds),
        no_op_pair_wall_seconds=list(example_result.no_op_pair_wall_seconds),
        determinism_policy=determinism_policy.__dict__,
        runtime_generation=runtime_generation.__dict__,
        runtime_generation_config_hash=runtime_generation.canonical_hash(),
        parameter_placement=parameter_placement.__dict__,
        runtime_identity=runtime_identity.__dict__,
        memory=memory.__dict__,
        minimized_target_evidence=[ev.__dict__ for ev in example_result.minimized_target_evidence],
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        every_parameter_on_cuda=parameter_placement.every_parameter_on_cuda,
        batch_size=batch_size,
        software_versions={"torch": torch.__version__},
    ).model_dump(mode="json")
