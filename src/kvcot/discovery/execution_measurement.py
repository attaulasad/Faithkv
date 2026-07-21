"""Synchronized timing, phase-owned CUDA memory, and runtime projection."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

from kvcot.generation.state import ModelStateSnapshot

T = TypeVar("T")
RTX3090_SAFETY_LIMIT_BYTES = 22 * 1024**3


@dataclass(frozen=True)
class TimingEvidence:
    phase: str
    started_at: float
    ended_at: float
    duration_seconds: float
    synchronize_before_start: bool
    synchronize_before_end: bool
    completed: bool
    failure_type: str | None = None
    failure_message: str | None = None


class TimedOperationError(RuntimeError):
    def __init__(self, evidence: TimingEvidence, cause: BaseException):
        super().__init__(f"{evidence.phase} failed: {type(cause).__name__}: {cause}")
        self.evidence = evidence
        self.__cause__ = cause


class SynchronizedTimer:
    """The one timing sequence used by discovery execution."""

    def __init__(self, cuda: Any, clock: Callable[[], float]):
        self.cuda = cuda
        self.clock = clock
        self.records: list[TimingEvidence] = []

    def measure(self, phase: str, operation: Callable[[], T]) -> T:
        self.cuda.synchronize()
        start = float(self.clock())
        try:
            result = operation()
        except BaseException as exc:
            self.cuda.synchronize()
            end = float(self.clock())
            evidence = self._record(phase, start, end, False, exc)
            raise TimedOperationError(evidence, exc) from exc
        self.cuda.synchronize()
        end = float(self.clock())
        self._record(phase, start, end, True, None)
        return result

    def begin_span(self) -> float:
        """Begin a parent span using the same synchronized boundary."""
        self.cuda.synchronize()
        return float(self.clock())

    def finish_span(self, phase: str, started_at: float) -> TimingEvidence:
        """Finish a parent span and retain its synchronized evidence."""
        self.cuda.synchronize()
        ended_at = float(self.clock())
        return self._record(phase, float(started_at), ended_at, True, None)

    def _record(
        self, phase: str, start: float, end: float, completed: bool, failure: BaseException | None
    ) -> TimingEvidence:
        duration = end - start
        if not all(math.isfinite(value) for value in (start, end, duration)) or duration < 0:
            raise ValueError(f"invalid synchronized timing for {phase}: start={start}, end={end}")
        evidence = TimingEvidence(
            phase=phase,
            started_at=start,
            ended_at=end,
            duration_seconds=duration,
            synchronize_before_start=True,
            synchronize_before_end=True,
            completed=completed,
            failure_type=None if failure is None else type(failure).__name__,
            failure_message=None if failure is None else str(failure),
        )
        self.records.append(evidence)
        return evidence

    def export(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


@dataclass(frozen=True)
class MemoryPhaseEvidence:
    phase: str
    allocated_before: int
    reserved_before: int
    peak_allocated: int
    peak_reserved: int
    allocated_after: int
    reserved_after: int
    reset_point: str
    synchronized_before: bool
    synchronized_after: bool
    completed: bool
    failure_type: str | None = None
    failure_message: str | None = None


class MemoryMeasuredOperationError(RuntimeError):
    def __init__(self, evidence: MemoryPhaseEvidence, cause: BaseException):
        super().__init__(f"{evidence.phase} failed: {type(cause).__name__}: {cause}")
        self.evidence = evidence
        self.__cause__ = cause


class CudaMemoryMeasurer:
    """Exclusive owner of peak resets for explicitly named phases."""

    def __init__(self, cuda: Any):
        self.cuda = cuda
        self.records: list[MemoryPhaseEvidence] = []

    def observe(self, phase: str, operation: Callable[[], T]) -> T:
        self.cuda.synchronize()
        allocated_before = int(self.cuda.memory_allocated())
        reserved_before = int(self.cuda.memory_reserved())
        self.cuda.reset_peak_memory_stats()
        reset_point = f"immediately_before:{phase}"
        try:
            result = operation()
        except BaseException as exc:
            self.cuda.synchronize()
            evidence = self._finish(
                phase, allocated_before, reserved_before, reset_point, False, type(exc).__name__, str(exc)
            )
            raise MemoryMeasuredOperationError(evidence, exc) from exc
        self.cuda.synchronize()
        self._finish(phase, allocated_before, reserved_before, reset_point, True, None, None)
        return result

    def _finish(
        self,
        phase: str,
        allocated_before: int,
        reserved_before: int,
        reset_point: str,
        completed: bool,
        failure_type: str | None,
        failure_message: str | None,
    ) -> MemoryPhaseEvidence:
        evidence = MemoryPhaseEvidence(
            phase=phase,
            allocated_before=allocated_before,
            reserved_before=reserved_before,
            peak_allocated=int(self.cuda.max_memory_allocated()),
            peak_reserved=int(self.cuda.max_memory_reserved()),
            allocated_after=int(self.cuda.memory_allocated()),
            reserved_after=int(self.cuda.memory_reserved()),
            reset_point=reset_point,
            synchronized_before=True,
            synchronized_after=True,
            completed=completed,
            failure_type=failure_type,
            failure_message=failure_message,
        )
        self.records.append(evidence)
        return evidence

    @property
    def maximum_peak_allocated(self) -> int:
        return max((record.peak_allocated for record in self.records), default=0)

    @property
    def maximum_peak_reserved(self) -> int:
        return max((record.peak_reserved for record in self.records), default=0)

    def export(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]


def snapshot_tensor_bytes(snapshot: ModelStateSnapshot) -> int:
    tensors = list(snapshot.key_cache) + list(snapshot.value_cache) + list(snapshot.query_cache.values())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def snapshot_growth_bytes_per_token(snapshot: ModelStateSnapshot) -> int:
    """Independent-audit Gate H5 repair: bytes ADDED across every layer's
    key AND value cache for exactly one more decoded position, derived
    entirely from the snapshot's OWN real tensor shapes/dtypes (never a
    hard-coded head-count/head-dim/dtype-size constant) -- `key_cache[i]`
    and `value_cache[i]` are real `(batch, num_kv_heads, seq_len,
    head_dim)` tensors; one additional position adds
    `num_kv_heads * head_dim * element_size` bytes to EACH of them."""
    per_token = 0
    for key_tensor, value_tensor in zip(snapshot.key_cache, snapshot.value_cache):
        per_token += key_tensor.shape[1] * key_tensor.shape[-1] * key_tensor.element_size()
        per_token += value_tensor.shape[1] * value_tensor.shape[-1] * value_tensor.element_size()
    return per_token


def snapshot_position_tracking_bytes_per_token(snapshot: ModelStateSnapshot) -> int:
    """Bytes of int64 absolute-position bookkeeping
    (`kvcot.generation.provenance.LayerProvenance`/`CompactionTracker`)
    added per layer per KV head for one more decoded position -- derived
    from the snapshot's own layer/head counts, never an unexplained
    constant. `key_cache[i].shape[1]` is `num_key_value_heads` for that
    layer (the same axis `snapshot_growth_bytes_per_token` reads)."""
    per_token = 0
    for key_tensor in snapshot.key_cache:
        per_token += key_tensor.shape[1] * 8  # one int64 position per (layer, kv_head)
    return per_token


@dataclass(frozen=True)
class PreBranchMemoryEvidence:
    phase: str
    allocated_before: int
    reserved_before: int
    snapshot_clone_bytes: int
    selected_vector_bytes: int
    # Independent-audit Gate H5 repair: componentized, each derived from a
    # real shape/dtype/count rather than folded into one opaque
    # `known_temporary_bytes` the caller had to compute itself.
    bridge_token_count: int
    scored_token_count: int
    total_future_branch_tokens: int
    per_token_kv_growth_bytes: int
    complete_horizon_kv_growth_bytes: int
    append_realloc_temporary_bytes: int
    query_cache_growth_bytes: int
    logits_bytes: int
    log_softmax_bytes: int
    nll_scalar_bytes: int
    position_tracking_bytes: int
    known_temporary_bytes: int
    required_additional_bytes: int
    projected_peak_bytes: int
    safety_limit_bytes: int
    accepted: bool
    synchronized: bool
    rejection_reason: str | None


def check_pre_branch_memory(
    *,
    phase: str,
    cuda: Any,
    snapshot: ModelStateSnapshot,
    selected_vector_bytes: int,
    vocab_size: int,
    bridge_token_count: int,
    scored_token_count: int,
    safety_limit_bytes: int = RTX3090_SAFETY_LIMIT_BYTES,
) -> PreBranchMemoryEvidence:
    """Independent-audit Gate H5 repair: the prior version accounted only
    `snapshot_clone_bytes` + `selected_vector_bytes` + a caller-supplied
    opaque `known_temporary_bytes` -- omitting the K/V cache growth the
    branch itself will cause across its full bridge-plus-scored horizon,
    the temporary storage `torch.cat`-style reallocation needs while that
    growth happens, and per-token bookkeeping (position tracking) growth.
    Every new component below is derived from either the snapshot's own
    real tensor shapes (`snapshot_growth_bytes_per_token`,
    `snapshot_position_tracking_bytes_per_token`) or the frozen,
    already-real `vocab_size`/horizon counts -- never an unexplained
    constant. `append_realloc_temporary_bytes` is deliberately
    conservative (documented, not presented as exact): PyTorch's
    cache-append pattern briefly holds both the old and the newly-grown
    tensor live during the copy, so one additional full horizon's worth of
    growth is reserved as safety headroom. `query_cache_growth_bytes` is
    explicitly `0`, not omitted -- R-KV's query cache is a fixed-size
    window (`kvcot.generation.state`'s own docstring), so it does not grow
    with the branch horizon."""
    for name, value in (
        ("selected_vector_bytes", selected_vector_bytes), ("vocab_size", vocab_size),
        ("bridge_token_count", bridge_token_count), ("scored_token_count", scored_token_count),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative int for the pre-branch memory guard, got {value!r}")
    if len(snapshot.key_cache) == 0:
        raise ValueError("snapshot has no layer K/V tensors -- cannot derive shape-based memory growth")

    cuda.synchronize()
    allocated = int(cuda.memory_allocated())
    reserved = int(cuda.memory_reserved())
    # Discovery branch restore transfers tensor ownership from one working
    # clone into its fresh live cache, so only one additional complete tensor
    # set can exist beside the already-accounted pristine snapshot.
    clone_bytes = snapshot_tensor_bytes(snapshot)

    total_future_branch_tokens = int(bridge_token_count) + int(scored_token_count)
    per_token_kv_growth = snapshot_growth_bytes_per_token(snapshot)
    complete_horizon_kv_growth = per_token_kv_growth * total_future_branch_tokens
    append_realloc_temporary = complete_horizon_kv_growth
    query_cache_growth = 0
    logits_bytes = int(vocab_size) * 4
    log_softmax_bytes = int(vocab_size) * 4
    nll_scalar_bytes = int(scored_token_count) * 4
    position_tracking_bytes = snapshot_position_tracking_bytes_per_token(snapshot) * total_future_branch_tokens
    known_temporary = logits_bytes + log_softmax_bytes + nll_scalar_bytes + position_tracking_bytes

    required = (
        clone_bytes + int(selected_vector_bytes) + complete_horizon_kv_growth + append_realloc_temporary
        + query_cache_growth + known_temporary
    )
    projected = max(allocated, reserved) + required
    accepted = projected <= safety_limit_bytes
    return PreBranchMemoryEvidence(
        phase=phase,
        allocated_before=allocated,
        reserved_before=reserved,
        snapshot_clone_bytes=clone_bytes,
        selected_vector_bytes=int(selected_vector_bytes),
        bridge_token_count=int(bridge_token_count),
        scored_token_count=int(scored_token_count),
        total_future_branch_tokens=total_future_branch_tokens,
        per_token_kv_growth_bytes=per_token_kv_growth,
        complete_horizon_kv_growth_bytes=complete_horizon_kv_growth,
        append_realloc_temporary_bytes=append_realloc_temporary,
        query_cache_growth_bytes=query_cache_growth,
        logits_bytes=logits_bytes,
        log_softmax_bytes=log_softmax_bytes,
        nll_scalar_bytes=nll_scalar_bytes,
        position_tracking_bytes=position_tracking_bytes,
        known_temporary_bytes=known_temporary,
        required_additional_bytes=required,
        projected_peak_bytes=projected,
        safety_limit_bytes=safety_limit_bytes,
        accepted=accepted,
        synchronized=True,
        rejection_reason=None if accepted else "projected branch allocation exceeds the frozen safety limit",
    )


@dataclass(frozen=True)
class RuntimeProjection:
    fullkv_startup_and_model_load_seconds: float
    rkv_startup_and_model_load_seconds: float
    fullkv_natural_generation_seconds: float
    rkv_pass1_seconds: float
    rkv_pass2_seconds: float
    per_example_inference_seconds: float
    example_count: int
    conservative_real_pair_seconds: float
    real_pair_count: int
    projected_total_seconds: float


def build_runtime_projection(
    *,
    fullkv_startup_and_model_load_seconds: float,
    rkv_startup_and_model_load_seconds: float,
    fullkv_natural_generation_seconds: float,
    rkv_pass1_seconds: float,
    rkv_pass2_seconds: float,
    b2a_real_pair_seconds: list[float],
    example_count: int = 12,
    real_pair_count: int = 144,
) -> RuntimeProjection:
    if len(b2a_real_pair_seconds) != B2A_REAL_PAIR_EVALUATIONS_TOTAL:
        raise ValueError(
            f"runtime projection requires exactly {B2A_REAL_PAIR_EVALUATIONS_TOTAL} B2A real-pair durations"
        )
    values = [
        fullkv_startup_and_model_load_seconds,
        rkv_startup_and_model_load_seconds,
        fullkv_natural_generation_seconds,
        rkv_pass1_seconds,
        rkv_pass2_seconds,
        *b2a_real_pair_seconds,
    ]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("runtime projection components must be finite and non-negative")
    per_example = fullkv_natural_generation_seconds + rkv_pass1_seconds + rkv_pass2_seconds
    conservative_pair = max(b2a_real_pair_seconds)
    total = (
        fullkv_startup_and_model_load_seconds
        + rkv_startup_and_model_load_seconds
        + example_count * per_example
        + real_pair_count * conservative_pair
    )
    return RuntimeProjection(
        fullkv_startup_and_model_load_seconds=fullkv_startup_and_model_load_seconds,
        rkv_startup_and_model_load_seconds=rkv_startup_and_model_load_seconds,
        fullkv_natural_generation_seconds=fullkv_natural_generation_seconds,
        rkv_pass1_seconds=rkv_pass1_seconds,
        rkv_pass2_seconds=rkv_pass2_seconds,
        per_example_inference_seconds=per_example,
        example_count=example_count,
        conservative_real_pair_seconds=conservative_pair,
        real_pair_count=real_pair_count,
        projected_total_seconds=total,
    )


# Avoid an import cycle at module import time while keeping the projection's
# frozen constant as its sole pair-count authority.
from kvcot.discovery.constants import B2A_REAL_PAIR_EVALUATIONS_TOTAL  # noqa: E402
