"""Per-instance, read-only capture wrapper around `R1KV.update_kv` (Part
VIII of `docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`, absolute-survivor-
parity repair in `docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` Blocker 2).

Never a class-level or global patch — attached to one specific `kv_cluster`
instance for the lifetime of a `with` block, and always restored on exit,
including on exception. Reads only through: the call's own arguments
(cloned before the real call, never fed back), the target instance's own
configuration attributes (`budget`, `window_size`, ... — read from the SAME
instance being wrapped, never a second, independently-configured copy),
the real return value, the instance's own bookkeeping state after the call
(`kept_token_indices`), and a caller-supplied `pre_event_position_map_fn`
thunk sourced from the existing provenance adapter
(`kvcot.generation.provenance.LayerProvenance.positions` — read fresh
immediately before each call, never a shadow FullKV reconstruction). It
never claims to read a local variable inside `update_kv`'s function body —
not possible without editing `third_party/R-KV` (prohibited) or an
unsupported frame-trace hook (fragile, version-coupled, not used) —
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §8.

Independently recomputes R-KV's own real, decision-driving score formula
(`r1_kv.py:49-82` — never the differently-windowed `r1_kv.py:91-116`
"analysis" formula R-KV itself persists as `kept_final_scores`, which is a
different tensor with different shape/normalization, §7 of the correction
document) using the PINNED `compute_attention_scores`/`cal_similarity`
helpers imported from `rkv.compression` at call time — reused, never
reimplemented, so there is exactly one source of truth for the arithmetic
outside test fixtures. `rkv` is only ever importable on the GPU host
(`third_party/R-KV`, installed editable there); this module defers that
import to inside the one function that needs it, matching
`kvcot.generation.policies`'s existing discipline.

## Absolute survivor parity (repaired, Blocker 2)

Previously this module only checked observed-vs-recomputed survivor
identity via SET equality of *pre-storage physical indices*, and only at
the first compaction event of a run (where pre-storage index happens to
equal absolute token position trivially, no remap needed yet) — silently
returning `None` (non-evaluable) for every later event. The active
experiment deliberately excludes the first and last probed events, so a
parity check that only ever fires at the first event checked nothing that
actually matters.

The repaired check runs at EVERY compaction event a `pre_event_position_map_fn`
is supplied for: it recomputes R-KV's own top-k physical indices over the
non-recent pool, appends the protected recent-window physical indices in
the same order the real returned compressed cache uses, gathers the
corresponding ABSOLUTE source-token positions from the caller-supplied
pre-event position map (never pre-storage physical indices, which are only
meaningful at the first event), and compares the resulting ordered
absolute-position tensor against `kv_cluster.kept_token_indices[-1]` using
exact shape equality and `torch.equal` — never set equality, since storage
order matters (later physical-slot identity depends on it).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class UpdateKvCaptureRecord:
    """One captured before/after observation of a single `update_kv` call.
    `had_compaction=False` means no eviction happened on this call
    (`kv_cache_len < budget`, r1_kv.py:46-47) — nothing to recompute or
    check, by construction, not a parity failure.

    For a compaction call where survivor-identity bookkeeping is available
    on `kv_cluster` (`record_kept_token_indices=True`), absolute-survivor
    parity is ALWAYS evaluated to a concrete boolean — never `None` — and a
    missing/malformed `pre_event_absolute_position_map` is itself a hard
    parity failure, never a silent skip."""

    had_compaction: bool
    pre_call_key_states: torch.Tensor
    pre_call_value_states: torch.Tensor
    pre_call_key_shape: tuple[int, ...]
    pre_call_value_shape: tuple[int, ...]
    pre_call_dtype: str
    pre_call_device: str
    recomputed_final_score: torch.Tensor | None
    # Separate score COMPONENTS (Blocker 9/B1B §9: "capture candidate K/V
    # and score components") -- `final_score = attention_component *
    # mix_lambda - similarity_component * (1 - mix_lambda)`, r1_kv.py:82;
    # stored separately (never only the combined final_score) so a
    # downstream pair record can report `attention_component_diff`/
    # `similarity_component_diff` independently, not just their already-
    # mixed sum.
    recomputed_attention_component: torch.Tensor | None
    recomputed_similarity_component: torch.Tensor | None
    recomputed_topk_indices: torch.Tensor | None
    # `kv_cluster.window_size` at capture time -- needed by downstream
    # consumers (`kvcot.discovery.pipeline`) to correctly split
    # `pre_event_absolute_position_map`/the score-component tensors into
    # their non-recent-pool vs. protected-recent-window portions.
    # `recomputed_topk_indices.shape[-1]` is NOT the pool size (it's
    # `budget - window_size`, the number SELECTED, which only happens to
    # equal the pool size when `pre_event_len == budget`) -- storing this
    # explicitly avoids that easy-to-get-wrong derivation.
    window_size: int | None
    returned_key_states: torch.Tensor
    returned_value_states: torch.Tensor
    gather_parity_passed: bool | None
    pre_event_absolute_position_map: torch.Tensor | None
    recomputed_kept_absolute_positions: torch.Tensor | None
    observed_kept_absolute_positions: torch.Tensor | None
    observed_kept_indices_parity_passed: bool | None
    parity_check_passed: bool
    parity_failure_reason: str | None


PositionMapFn = Callable[[], "torch.Tensor | None"]


def _recompute_final_score_and_indices(
    kv_cluster, key_states: torch.Tensor, query_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicates `r1_kv.py:49-82` exactly using the pinned helpers,
    reading every hyperparameter off `kv_cluster` itself. Returns
    `(final_score, indices, attention_component, similarity_component)` --
    the two components separately, never only their already-mixed sum, so a
    downstream pair record can report each diff independently."""
    from rkv.compression import cal_similarity, compute_attention_scores  # pinned submodule, GPU host only

    window_size = kv_cluster.window_size
    attn_weights = compute_attention_scores(query_states, key_states)
    attn_weights_sum = (
        torch.nn.functional.softmax(
            attn_weights[:, :, -window_size:, :-window_size], dim=-1, dtype=torch.float32
        )
        .mean(dim=-2)
        .to(query_states.dtype)
    )
    attn_cache = torch.nn.functional.max_pool1d(
        attn_weights_sum, kernel_size=kv_cluster.kernel_size, padding=kv_cluster.kernel_size // 2, stride=1
    )
    similarity_cos = cal_similarity(
        key_states, retain_ratio=kv_cluster.retain_ratio, retain_direction=kv_cluster.retain_direction
    )[:, :, :-window_size]
    final_score = attn_cache * kv_cluster.mix_lambda - similarity_cos * (1 - kv_cluster.mix_lambda)
    indices = final_score.topk(kv_cluster.budget - window_size, dim=-1).indices
    return final_score, indices, attn_cache, similarity_cos


def _recomputed_kept_physical_indices(
    recomputed_topk_indices: torch.Tensor,
    kv_cache_len: int,
    window_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Physical-slot indices of the kept set, IN THE SAME ORDER the real
    returned compressed cache uses: recomputed top-k (non-recent pool)
    first, then the protected recent window (`r1_kv.py`'s own
    `torch.cat([k_past_compress, k_cur], dim=2)` order) — shape
    `(num_heads, budget)`.

    `device`/`dtype` are the PROVENANCE MAP's own device/dtype (the tensor
    this result is about to be `gather`-ed against), never assumed to match
    `recomputed_topk_indices`'s device (Blocker: on a real CUDA compaction,
    `recomputed_topk_indices` is a CUDA tensor produced by `.topk()` on
    CUDA score tensors, while the provenance map -- and this function's own
    `recent_window`, previously a bare `torch.arange(...)` that silently
    defaulted to CPU -- can legitimately live on CPU. Both operands are
    normalized to the provenance map's device/dtype BEFORE concatenation,
    never the other way around (the complete provenance map is never moved
    to CUDA merely to hide the mismatch, since it is not this function's
    tensor to relocate) -- shape and ordering are preserved exactly."""
    num_heads = recomputed_topk_indices.shape[1]
    physical_indices = recomputed_topk_indices.squeeze(0).to(device=device, dtype=dtype)
    recent_window = torch.arange(kv_cache_len - window_size, kv_cache_len, device=device, dtype=dtype).expand(
        num_heads, -1
    )
    return torch.cat([physical_indices, recent_window], dim=-1)


def _build_capture_record(
    kv_cluster,
    pre_key: torch.Tensor,
    pre_query: torch.Tensor,
    pre_value: torch.Tensor,
    returned_key: torch.Tensor,
    returned_value: torch.Tensor,
    pre_event_position_map: torch.Tensor | None,
) -> UpdateKvCaptureRecord:
    kv_cache_len = pre_key.shape[-2]
    had_compaction = kv_cache_len >= kv_cluster.budget

    common = dict(
        pre_call_key_states=pre_key.clone(),
        pre_call_value_states=pre_value.clone(),
        pre_call_key_shape=tuple(pre_key.shape),
        pre_call_value_shape=tuple(pre_value.shape),
        pre_call_dtype=str(pre_key.dtype),
        pre_call_device=str(pre_key.device),
        returned_key_states=returned_key.clone(),
        returned_value_states=returned_value.clone(),
    )

    if not had_compaction:
        return UpdateKvCaptureRecord(
            had_compaction=False,
            recomputed_final_score=None,
            recomputed_attention_component=None,
            recomputed_similarity_component=None,
            recomputed_topk_indices=None,
            window_size=kv_cluster.window_size,
            gather_parity_passed=None,
            pre_event_absolute_position_map=None,
            recomputed_kept_absolute_positions=None,
            observed_kept_absolute_positions=None,
            observed_kept_indices_parity_passed=None,
            parity_check_passed=True,
            parity_failure_reason=None,
            **common,
        )

    try:
        final_score, indices, attention_component, similarity_component = _recompute_final_score_and_indices(
            kv_cluster, pre_key, pre_query
        )
    except Exception as exc:
        return UpdateKvCaptureRecord(
            had_compaction=True,
            recomputed_final_score=None,
            recomputed_attention_component=None,
            recomputed_similarity_component=None,
            recomputed_topk_indices=None,
            window_size=kv_cluster.window_size,
            gather_parity_passed=None,
            pre_event_absolute_position_map=None,
            recomputed_kept_absolute_positions=None,
            observed_kept_absolute_positions=None,
            observed_kept_indices_parity_passed=None,
            parity_check_passed=False,
            parity_failure_reason=f"recomputation raised {type(exc).__name__}: {exc}",
            **common,
        )

    window_size = kv_cluster.window_size
    head_dim = pre_key.shape[-1]
    gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    # Gathered against the ORIGINAL (uncloned) pre-call tensors -- a real
    # mismatch here means the recomputation formula itself is wrong, never a
    # numerical-precision artifact, since both sides are the identical
    # deterministic computation over the same inputs.
    recomputed_k = pre_key[:, :, :-window_size, :].gather(dim=2, index=gather_indices)
    recomputed_v = pre_value[:, :, :-window_size, :].gather(dim=2, index=gather_indices)
    k_cur = pre_key[:, :, -window_size:, :]
    v_cur = pre_value[:, :, -window_size:, :]
    reconstructed_k = torch.cat([recomputed_k, k_cur], dim=2)
    reconstructed_v = torch.cat([recomputed_v, v_cur], dim=2)

    gather_parity_passed = bool(
        torch.equal(reconstructed_k, returned_key) and torch.equal(reconstructed_v, returned_value)
    )

    (
        observed_kept_indices_parity_passed,
        recomputed_kept_absolute_positions,
        observed_kept_absolute_positions,
        absolute_parity_failure_reason,
    ) = _check_observed_kept_absolute_position_parity(
        kv_cluster, indices, kv_cache_len, window_size, pre_event_position_map
    )

    reasons = []
    if not gather_parity_passed:
        reasons.append("gather_parity_failed")
    if absolute_parity_failure_reason is not None:
        reasons.append(absolute_parity_failure_reason)
    parity_check_passed = not reasons

    return UpdateKvCaptureRecord(
        had_compaction=True,
        recomputed_final_score=final_score,
        recomputed_attention_component=attention_component,
        recomputed_similarity_component=similarity_component,
        recomputed_topk_indices=indices,
        window_size=window_size,
        gather_parity_passed=gather_parity_passed,
        pre_event_absolute_position_map=(
            pre_event_position_map.clone() if pre_event_position_map is not None else None
        ),
        recomputed_kept_absolute_positions=recomputed_kept_absolute_positions,
        observed_kept_absolute_positions=observed_kept_absolute_positions,
        observed_kept_indices_parity_passed=observed_kept_indices_parity_passed,
        parity_check_passed=parity_check_passed,
        parity_failure_reason=",".join(reasons) if reasons else None,
        **common,
    )


def _check_observed_kept_absolute_position_parity(
    kv_cluster,
    recomputed_topk_indices: torch.Tensor,
    kv_cache_len: int,
    window_size: int,
    pre_event_position_map: torch.Tensor | None,
) -> tuple[bool | None, torch.Tensor | None, torch.Tensor | None, str | None]:
    """Absolute-position survivor parity for ONE compaction event, evaluated
    the same way at every event (first, middle, last) — never limited to
    the first event of a run, and never set equality.

    Returns `(observed_kept_indices_parity_passed, recomputed_kept_absolute_positions,
    observed_kept_absolute_positions, failure_reason)`. The first element is
    `None` (not evaluable, never treated as a pass) only when R-KV's own
    survivor bookkeeping (`kept_token_indices`) is unavailable on this
    instance at all -- there is nothing to compare against, so no parity
    claim of any kind is made. Whenever that bookkeeping IS available, this
    function ALWAYS returns a concrete `True`/`False` for a compaction
    event: a missing `pre_event_position_map` is itself a hard failure
    (never a silent `None` skip), per Blocker 2's required correction.
    """
    if not getattr(kv_cluster, "record_kept_token_indices", False):
        return None, None, None, None
    kept_token_indices = getattr(kv_cluster, "kept_token_indices", None)
    if not kept_token_indices:
        return None, None, None, None

    observed = kept_token_indices[-1]

    if pre_event_position_map is None:
        return False, None, observed.clone(), "missing_pre_event_absolute_position_map"

    num_heads = recomputed_topk_indices.shape[1]
    expected_map_shape = (num_heads, kv_cache_len)
    if tuple(pre_event_position_map.shape) != expected_map_shape:
        return (
            False,
            None,
            observed.clone(),
            f"pre_event_absolute_position_map_shape_mismatch: expected {expected_map_shape}, "
            f"got {tuple(pre_event_position_map.shape)}",
        )

    recomputed_physical_indices = _recomputed_kept_physical_indices(
        recomputed_topk_indices,
        kv_cache_len,
        window_size,
        device=pre_event_position_map.device,
        dtype=pre_event_position_map.dtype,
    )
    recomputed_absolute_positions = pre_event_position_map.gather(dim=-1, index=recomputed_physical_indices)

    if recomputed_absolute_positions.shape != observed.shape:
        return (
            False,
            recomputed_absolute_positions,
            observed.clone(),
            f"kept_indices_shape_mismatch: recomputed {tuple(recomputed_absolute_positions.shape)}, "
            f"observed {tuple(observed.shape)}",
        )

    ordered_equal = bool(torch.equal(recomputed_absolute_positions, observed))
    return (
        ordered_equal,
        recomputed_absolute_positions,
        observed.clone(),
        None if ordered_equal else "observed_kept_indices_parity_failed",
    )


CurrentPositionFn = Callable[[], "int | None"]
ShouldCaptureFn = Callable[[int, int], bool]


@contextlib.contextmanager
def capture_update_kv(
    kv_cluster,
    capture_sink: list[UpdateKvCaptureRecord],
    pre_event_position_map_fn: PositionMapFn | None = None,
    *,
    layer_idx: int | None = None,
    current_position_fn: CurrentPositionFn | None = None,
    should_capture: ShouldCaptureFn | None = None,
):
    """Attach the wrapper to `kv_cluster.update_kv` for the lifetime of this
    `with` block only. Restores the exact original bound method on exit,
    even if the body raises.

    ## Target-only, memory-bounded capture (B1B-R2 §4)

    `should_capture`, when supplied together with `layer_idx` and
    `current_position_fn`, is called as `should_capture(position, layer_idx)`
    immediately before each real `update_kv` call (`position` comes fresh
    from `current_position_fn()`, called every time, never cached). When it
    returns `False` (or when `current_position_fn` returns `None`, meaning
    "no position available"), this wrapper calls the ORIGINAL `update_kv`
    directly and returns its result unchanged: no input tensor is cloned, no
    capture record is built or appended, `capture_sink` is not touched.
    Behavior for a non-target call is therefore bit-for-bit identical to the
    unwrapped method — the wrapper is invisible on the non-target path, not
    merely "cheaper".

    `should_capture` defaults to `None`, which preserves this function's
    original behavior exactly: every real call is captured (used by every
    pre-existing caller/test of this function). Passing `should_capture`
    is how a caller bounds retained state to the number of TRUE evaluations
    across a whole run (e.g. Pass 2's 3 preselected event/layer targets) —
    never by the total number of decode or compaction calls, which for a
    real generation can be orders of magnitude larger.

    For a call where `should_capture` returns `True` (or is not supplied at
    all), one `UpdateKvCaptureRecord` is appended to `capture_sink`, exactly
    as before this section's addition.

    `pre_event_position_map_fn`, when supplied, is called with no arguments
    IMMEDIATELY BEFORE each real `update_kv` call and must return either
    `None` or a `(num_key_value_heads, pre_event_cache_length)` tensor
    giving the absolute source-token position occupying each physical cache
    slot at that exact moment -- sourced from the caller's own
    `kvcot.generation.provenance.LayerProvenance.positions` for this layer
    (read fresh every call, never a cached/stale snapshot, never a shadow
    FullKV reconstruction).

    Whenever `kv_cluster` has survivor-identity bookkeeping available
    (`record_kept_token_indices=True` with a non-empty `kept_token_indices`)
    AND a compaction event fires on a given call, absolute-survivor parity
    is ALWAYS evaluated to a concrete `True`/`False` for that call -- never
    silently skipped. This includes the case where `pre_event_position_map_fn`
    was never supplied at all (`None`), or was supplied but returned `None`
    for that specific call: both are treated identically, as a missing map,
    which is itself a hard parity failure (Blocker 2's required
    correction) -- never a silent `None`/"not applicable" result. Parity is
    `None` (genuinely not evaluable, never treated as a pass) only when
    `kv_cluster` has no survivor bookkeeping at all -- there is nothing to
    compare against in that case, independent of whether a position map was
    offered.
    """
    original_update_kv = kv_cluster.update_kv
    had_instance_override = "update_kv" in kv_cluster.__dict__

    def _wrapped(key_states, query_states, value_states):
        if should_capture is not None:
            position = current_position_fn() if current_position_fn is not None else None
            if position is None or layer_idx is None or not should_capture(position, layer_idx):
                # Non-target call: the ORIGINAL method, unchanged -- no
                # clone, no capture record, capture_sink untouched.
                return original_update_kv(key_states, query_states, value_states)

        pre_key = key_states.clone()
        pre_query = query_states.clone()
        pre_value = value_states.clone()
        pre_event_position_map = pre_event_position_map_fn() if pre_event_position_map_fn is not None else None

        returned_key, returned_value = original_update_kv(key_states, query_states, value_states)

        capture_sink.append(
            _build_capture_record(
                kv_cluster, pre_key, pre_query, pre_value, returned_key, returned_value, pre_event_position_map
            )
        )
        return returned_key, returned_value

    kv_cluster.update_kv = _wrapped
    try:
        yield capture_sink
    finally:
        if had_instance_override:
            kv_cluster.update_kv = original_update_kv
        else:
            del kv_cluster.update_kv
