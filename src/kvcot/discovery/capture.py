"""Per-instance, read-only capture wrapper around `R1KV.update_kv` (Part
VIII of `docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`).

Never a class-level or global patch — attached to one specific `kv_cluster`
instance for the lifetime of a `with` block, and always restored on exit,
including on exception. Reads only through: the call's own arguments
(cloned before the real call, never fed back), the target instance's own
configuration attributes (`budget`, `window_size`, ... — read from the SAME
instance being wrapped, never a second, independently-configured copy),
the real return value, and the instance's own bookkeeping state after the
call (`kept_token_indices`). It never claims to read a local variable
inside `update_kv`'s function body — not possible without editing
`third_party/R-KV` (prohibited) or an unsupported frame-trace hook (fragile,
version-coupled, not used) — `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §8.

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
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UpdateKvCaptureRecord:
    """One captured before/after observation of a single `update_kv` call.
    `had_compaction=False` means no eviction happened on this call
    (`kv_cache_len < budget`, r1_kv.py:46-47) — nothing to recompute or
    check, by construction, not a parity failure."""

    had_compaction: bool
    pre_call_key_states: torch.Tensor
    pre_call_value_states: torch.Tensor
    pre_call_key_shape: tuple[int, ...]
    pre_call_value_shape: tuple[int, ...]
    pre_call_dtype: str
    pre_call_device: str
    recomputed_final_score: torch.Tensor | None
    recomputed_topk_indices: torch.Tensor | None
    returned_key_states: torch.Tensor
    returned_value_states: torch.Tensor
    gather_parity_passed: bool | None
    observed_kept_indices_parity_passed: bool | None
    parity_check_passed: bool
    parity_failure_reason: str | None


def _recompute_final_score_and_indices(
    kv_cluster, key_states: torch.Tensor, query_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicates `r1_kv.py:49-82` exactly using the pinned helpers,
    reading every hyperparameter off `kv_cluster` itself."""
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
    return final_score, indices


def _build_capture_record(
    kv_cluster,
    pre_key: torch.Tensor,
    pre_query: torch.Tensor,
    pre_value: torch.Tensor,
    returned_key: torch.Tensor,
    returned_value: torch.Tensor,
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
            recomputed_topk_indices=None,
            gather_parity_passed=None,
            observed_kept_indices_parity_passed=None,
            parity_check_passed=True,
            parity_failure_reason=None,
            **common,
        )

    try:
        final_score, indices = _recompute_final_score_and_indices(kv_cluster, pre_key, pre_query)
    except Exception as exc:
        return UpdateKvCaptureRecord(
            had_compaction=True,
            recomputed_final_score=None,
            recomputed_topk_indices=None,
            gather_parity_passed=None,
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

    observed_kept_indices_parity_passed = _check_observed_kept_indices_parity(
        kv_cluster, indices, kv_cache_len, window_size
    )

    reasons = []
    if not gather_parity_passed:
        reasons.append("gather_parity_failed")
    if observed_kept_indices_parity_passed is False:
        reasons.append("observed_kept_indices_parity_failed")
    parity_check_passed = not reasons

    return UpdateKvCaptureRecord(
        had_compaction=True,
        recomputed_final_score=final_score,
        recomputed_topk_indices=indices,
        gather_parity_passed=gather_parity_passed,
        observed_kept_indices_parity_passed=observed_kept_indices_parity_passed,
        parity_check_passed=parity_check_passed,
        parity_failure_reason=",".join(reasons) if reasons else None,
        **common,
    )


def _check_observed_kept_indices_parity(
    kv_cluster, recomputed_topk_indices: torch.Tensor, kv_cache_len: int, window_size: int
) -> bool | None:
    """Set-equality check per KV head between the recomputed selection
    (recomputed pre-storage indices plus the always-kept recent window)
    and R-KV's own real, observed `kept_token_indices[-1]`. Meaningful only
    at the FIRST compaction event of a run, where pre-storage index ==
    absolute token position (no prior-event remap has happened yet,
    r1_kv.py:141-154's `evicted_token_num`-offset remap only applies from
    the second event onward) — multi-event absolute-position bookkeeping is
    `kvcot.generation.provenance.LayerProvenance`'s job, reused unchanged,
    never reimplemented here. Returns `None` (not evaluable, never treated
    as a pass) when bookkeeping is unavailable or this is not a first-event
    comparison.
    """
    if not getattr(kv_cluster, "record_kept_token_indices", False):
        return None
    kept_token_indices = getattr(kv_cluster, "kept_token_indices", None)
    if not kept_token_indices:
        return None
    is_first_event = len(kept_token_indices) == 1 and getattr(kv_cluster, "evicted_token_num", None) == (
        kv_cache_len - kv_cluster.budget
    )
    if not is_first_event:
        return None

    observed = kept_token_indices[-1]
    num_heads = recomputed_topk_indices.shape[1]
    recent_window = torch.arange(kv_cache_len - window_size, kv_cache_len).expand(num_heads, -1)
    recomputed_positions = torch.cat([recomputed_topk_indices.squeeze(0), recent_window], dim=-1)

    if recomputed_positions.shape != observed.shape:
        return False
    return all(
        set(recomputed_positions[h].tolist()) == set(observed[h].tolist()) for h in range(num_heads)
    )


@contextlib.contextmanager
def capture_update_kv(kv_cluster, capture_sink: list[UpdateKvCaptureRecord]):
    """Attach the wrapper to `kv_cluster.update_kv` for the lifetime of this
    `with` block only. Appends one `UpdateKvCaptureRecord` per call to
    `capture_sink`. Restores the exact original bound method on exit, even
    if the body raises."""
    original_update_kv = kv_cluster.update_kv
    had_instance_override = "update_kv" in kv_cluster.__dict__

    def _wrapped(key_states, query_states, value_states):
        pre_key = key_states.clone()
        pre_query = query_states.clone()
        pre_value = value_states.clone()

        returned_key, returned_value = original_update_kv(key_states, query_states, value_states)

        capture_sink.append(_build_capture_record(kv_cluster, pre_key, pre_query, pre_value, returned_key, returned_value))
        return returned_key, returned_value

    kv_cluster.update_kv = _wrapped
    try:
        yield capture_sink
    finally:
        if had_instance_override:
            kv_cluster.update_kv = original_update_kv
        else:
            del kv_cluster.update_kv
