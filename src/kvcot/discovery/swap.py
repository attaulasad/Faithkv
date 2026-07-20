"""Fixed-shape within-head swap primitive (Part IX.19 of
`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`,
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §6; dtype/device/storage-overlap
hardening in `docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` Blocker 4).

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

## Dtype/device/aliasing hardening (Blocker 4)

A plain shape check does not stop a silent dtype/device cast on assignment
(`target[...] = candidate` silently upcasts/downcasts/copies-across-devices
in PyTorch rather than raising), and a plain `tensor.data_ptr() ==
tensor.data_ptr()` check only catches two tensors sharing the exact same
starting address — it misses two tensors that are different VIEWS into the
same underlying storage at different offsets (e.g. a candidate that is
itself a slice of the very cache being written into). Every dtype/device
mismatch is rejected before any clone or write happens, and every storage
overlap is detected via underlying-storage identity
(`tensor.untyped_storage().data_ptr()`), never object identity or a
starting-address-only comparison. Non-contiguous candidate vectors are
rejected outright rather than attempting ambiguous span analysis.
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


def _storage_id(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _assert_no_storage_overlap(named_tensors: list[tuple[str, torch.Tensor]]) -> None:
    """Conservative storage-overlap guard: rejects ANY pair of tensors in
    `named_tensors` that share the same underlying storage, regardless of
    whether they start at the same offset -- catches offset views into the
    same storage and candidate-is-a-view-into-the-cache bugs that a plain
    `data_ptr()`-on-the-tensor-itself comparison would miss."""
    seen: dict[int, str] = {}
    for name, tensor in named_tensors:
        storage_id = _storage_id(tensor)
        if storage_id in seen:
            raise SwapAliasingError(
                f"{name!r} and {seen[storage_id]!r} share the same underlying tensor storage "
                "(detected via untyped_storage().data_ptr(), not just matching starting address) "
                "-- refusing to swap, this is almost certainly a caller bug."
            )
        seen[storage_id] = name


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

    # --- storage-overlap guard: EVERY pair among the four tensors this call
    # touches, before any shape/dtype/device check (a bad alias is a bug
    # regardless of whether shapes happen to also be compatible). ---
    _assert_no_storage_overlap(
        [
            (f"key_cache[{layer_index}]", layer_k),
            (f"value_cache[{layer_index}]", layer_v),
            ("candidate_key", candidate_key),
            ("candidate_value", candidate_value),
        ]
    )

    if layer_k.shape != layer_v.shape:
        raise SwapIndexError(f"key/value shape mismatch at layer {layer_index}: {layer_k.shape} vs {layer_v.shape}")
    if layer_k.dtype != layer_v.dtype:
        raise SwapIndexError(f"key/value dtype mismatch at layer {layer_index}: {layer_k.dtype} vs {layer_v.dtype}")
    if layer_k.device != layer_v.device:
        raise SwapIndexError(f"key/value device mismatch at layer {layer_index}: {layer_k.device} vs {layer_v.device}")
    if layer_k.dim() != 4:
        raise SwapIndexError(f"expected a 4-D (batch, num_kv_heads, seq_len, head_dim) tensor, got shape {layer_k.shape}")

    batch_size, num_kv_heads, seq_len, head_dim = layer_k.shape
    if batch_size != 1:
        raise SwapIndexError(f"expected batch size exactly 1 at layer {layer_index}, got {batch_size}")
    if not (0 <= kv_head_index < num_kv_heads):
        raise SwapIndexError(f"kv_head_index={kv_head_index} out of range for {num_kv_heads} heads")
    if not (0 <= retained_post_storage_position < seq_len):
        raise SwapIndexError(f"retained_post_storage_position={retained_post_storage_position} out of range for seq_len={seq_len}")

    if tuple(candidate_key.shape) != (head_dim,):
        raise SwapIndexError(f"candidate_key must have shape ({head_dim},), got {tuple(candidate_key.shape)}")
    if tuple(candidate_value.shape) != (head_dim,):
        raise SwapIndexError(f"candidate_value must have shape ({head_dim},), got {tuple(candidate_value.shape)}")

    if not candidate_key.is_contiguous():
        raise SwapIndexError("candidate_key must be contiguous -- non-contiguous candidates are rejected outright")
    if not candidate_value.is_contiguous():
        raise SwapIndexError("candidate_value must be contiguous -- non-contiguous candidates are rejected outright")

    if candidate_key.dtype != layer_k.dtype:
        raise SwapIndexError(f"candidate_key.dtype ({candidate_key.dtype}) must equal target key-cache dtype ({layer_k.dtype})")
    if candidate_value.dtype != layer_v.dtype:
        raise SwapIndexError(f"candidate_value.dtype ({candidate_value.dtype}) must equal target value-cache dtype ({layer_v.dtype})")
    if candidate_key.device != layer_k.device:
        raise SwapIndexError(f"candidate_key.device ({candidate_key.device}) must equal target key-cache device ({layer_k.device})")
    if candidate_value.device != layer_v.device:
        raise SwapIndexError(f"candidate_value.device ({candidate_value.device}) must equal target value-cache device ({layer_v.device})")

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
