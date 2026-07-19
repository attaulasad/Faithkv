"""Fixed-shape within-head swap primitive (Part IX.19 of
`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`,
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §6).

```
key_cache[L][0, h, r_slot, :]   = captured_key_e
value_cache[L][0, h, r_slot, :] = captured_value_e
```

Shape-preserving, content-substitution only: no slot is added or removed,
no other layer/head/slot changes, net physical cache bytes is always
exactly zero. Supports the degenerate no-op case (`e := r`, writing back
exactly what was already there) as a first-class input, not a special case
requiring separate code — `is_noop` on the result reports whether the
written value equals the pre-write value, for the mandatory no-op control
(Part IX.20).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


class SwapIndexError(RuntimeError):
    pass


class SwapAliasingError(RuntimeError):
    pass


@dataclass(frozen=True)
class SwapResult:
    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]
    layer_index: int
    kv_head_index: int
    retained_post_storage_position: int
    is_noop: bool


def apply_within_head_swap(
    key_cache: list[torch.Tensor],
    value_cache: list[torch.Tensor],
    layer_index: int,
    kv_head_index: int,
    retained_post_storage_position: int,
    candidate_key: torch.Tensor,
    candidate_value: torch.Tensor,
) -> SwapResult:
    """Clone the full per-layer cache list (so the caller's original tensors
    are never mutated), then write `candidate_key`/`candidate_value`
    (shape `(head_dim,)`) into exactly one `(layer, batch=0, kv_head, slot)`
    position of the cloned K and V tensors. Every other layer, head, and
    slot is byte-identical to the input, since only one indexed write
    happens on top of an otherwise-untouched clone.
    """
    if len(key_cache) != len(value_cache):
        raise SwapIndexError(f"key_cache has {len(key_cache)} layers, value_cache has {len(value_cache)}")
    if not (0 <= layer_index < len(key_cache)):
        raise SwapIndexError(f"layer_index={layer_index} out of range for {len(key_cache)} layers")

    layer_k = key_cache[layer_index]
    layer_v = value_cache[layer_index]

    if layer_k.data_ptr() == layer_v.data_ptr():
        raise SwapAliasingError(
            f"key_cache[{layer_index}] and value_cache[{layer_index}] share the same underlying "
            "tensor storage -- refusing to swap, this is almost certainly a caller bug."
        )
    if candidate_key.data_ptr() == candidate_value.data_ptr():
        raise SwapAliasingError("candidate_key and candidate_value are the same tensor object")

    if layer_k.shape != layer_v.shape:
        raise SwapIndexError(f"key/value shape mismatch at layer {layer_index}: {layer_k.shape} vs {layer_v.shape}")
    if layer_k.dim() != 4:
        raise SwapIndexError(f"expected a 4-D (batch, num_kv_heads, seq_len, head_dim) tensor, got shape {layer_k.shape}")

    _, num_kv_heads, seq_len, head_dim = layer_k.shape
    if not (0 <= kv_head_index < num_kv_heads):
        raise SwapIndexError(f"kv_head_index={kv_head_index} out of range for {num_kv_heads} heads")
    if not (0 <= retained_post_storage_position < seq_len):
        raise SwapIndexError(f"retained_post_storage_position={retained_post_storage_position} out of range for seq_len={seq_len}")

    if tuple(candidate_key.shape) != (head_dim,):
        raise SwapIndexError(f"candidate_key must have shape ({head_dim},), got {tuple(candidate_key.shape)}")
    if tuple(candidate_value.shape) != (head_dim,):
        raise SwapIndexError(f"candidate_value must have shape ({head_dim},), got {tuple(candidate_value.shape)}")

    new_key_cache = [t.clone() for t in key_cache]
    new_value_cache = [t.clone() for t in value_cache]

    target_k = new_key_cache[layer_index]
    target_v = new_value_cache[layer_index]

    pre_k = target_k[0, kv_head_index, retained_post_storage_position, :].clone()
    pre_v = target_v[0, kv_head_index, retained_post_storage_position, :].clone()

    target_k[0, kv_head_index, retained_post_storage_position, :] = candidate_key
    target_v[0, kv_head_index, retained_post_storage_position, :] = candidate_value

    is_noop = bool(torch.equal(pre_k, candidate_key) and torch.equal(pre_v, candidate_value))

    return SwapResult(
        key_cache=new_key_cache,
        value_cache=new_value_cache,
        layer_index=layer_index,
        kv_head_index=kv_head_index,
        retained_post_storage_position=retained_post_storage_position,
        is_noop=is_noop,
    )
