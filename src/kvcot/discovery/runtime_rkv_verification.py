"""Runtime R-KV configuration readback and verification, discovery-track
only (`docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md` §4). Never imported from a
CPU-only path -- this module reads attributes off a LIVE, loaded model's
per-layer `self_attn.kv_cluster`/`self_attn.config` objects, so it only
makes sense to call after `RKVPolicy.load(...)` has actually run on a real
GPU host. `kvcot.discovery.b2a_execute` is the only caller.

## Why this module exists

`kvcot.discovery.discovery_config.DiscoveryRkvLock` freezes and hashes
every R-KV hyperparameter the task brief cares about, but freezing a value
in a config file proves nothing about what the upstream `R1KV`/monkeypatch
machinery actually did with it at runtime -- `RKVMethodConfig` silently
defaulting a field the caller forgot to pass is exactly the kind of gap
CLAUDE.md's anti-fabrication stance exists to catch. Every field this
module reads back is verified against the pinned R-KV submodule's own
source, cited by exact line, not guessed:

- `budget`, `window_size`, `kernel_size`, `mix_lambda`, `retain_ratio`,
  `retain_direction`, `record_kept_token_indices` -- set as plain instance
  attributes in `R1KV.__init__`
  (`third_party/R-KV/HuggingFace/rkv/compression/r1_kv.py:8-24`), reached
  at `model.model.layers[i].self_attn.kv_cluster`.
- `divide_method`, `divide_length`, `compression_content` -- NOT on
  `kv_cluster` at all; set via `model.config.update(compression_config)`
  in `kvcot.generation.policies._PatchedPolicyBase.load` and read back at
  `model.config.divide_method` / `.divide_length` / `.compression_content`
  (verified against `third_party/R-KV/HuggingFace/rkv/modeling.py:557,593,605`).
- `update_kv` -- also NOT on `kv_cluster`; it is a per-layer attention-config
  flag, `self_attn.config.update_kv`, read at
  `third_party/R-KV/HuggingFace/rkv/modeling.py:137,166,295,323` (three
  architectures, six call sites, all reading the identical attribute name).

A field this module cannot find on the runtime object at all (a future
R-KV submodule revision renaming or dropping an attribute) is a hard
`RuntimeRkvConfigError`, never a silently-skipped comparison -- the whole
point of this module is that "the field was frozen and hashed in the
config" must never be confused with "the field was verified against the
runtime object".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcot.discovery.discovery_config import DiscoveryRkvLock
from kvcot.utils.hashing import sha256_json

# The exact field set compared between the frozen config and the runtime
# readback -- deliberately NOT the same set `discovery_config.rkv_config_hash`
# hashes (that one also includes `upstream_revision`, which is a submodule
# pin, not something readable off a loaded model instance).
RUNTIME_COMPARABLE_RKV_FIELDS: tuple[str, ...] = (
    "budget",
    "window_size",
    "kernel_size",
    "mix_lambda",
    "retain_ratio",
    "retain_direction",
    "divide_method",
    "divide_length",
    "compression_content",
    "record_kept_token_indices",
    "update_kv",
)


class RuntimeRkvConfigError(RuntimeError):
    pass


def frozen_runtime_comparable_fields(rkv: DiscoveryRkvLock) -> dict[str, Any]:
    """The subset of the frozen `DiscoveryRkvLock` that is actually
    runtime-observable on a loaded model -- `update_kv`/
    `record_kept_token_indices` are not represented in `DiscoveryRkvLock`
    itself (they are always `True` in this repository, `RKVMethodConfig`'s
    own defaults, CLAUDE.md's frozen `update_kv`/eviction-on` policy), so
    they are filled in here as the same constants
    `kvcot.discovery.b2a_execute` passes into `RKVPolicy`, never silently
    left out of the comparison."""
    return {
        "budget": rkv.budget,
        "window_size": rkv.window_size,
        "kernel_size": rkv.kernel_size,
        "mix_lambda": rkv.mix_lambda,
        "retain_ratio": rkv.retain_ratio,
        "retain_direction": rkv.retain_direction,
        "divide_method": rkv.divide_method,
        "divide_length": rkv.divide_length,
        "compression_content": rkv.compression_content,
        "record_kept_token_indices": True,
        "update_kv": True,
    }


def frozen_rkv_config_hash(rkv: DiscoveryRkvLock) -> str:
    return sha256_json(frozen_runtime_comparable_fields(rkv))


def read_runtime_rkv_config(model: Any) -> dict[str, Any]:
    """Read every field in `RUNTIME_COMPARABLE_RKV_FIELDS` back off the
    live model, requiring every transformer layer to agree exactly (R-KV's
    own per-step schedule applies identically to every layer -- see
    `kvcot.generation.replay._note_event_once`'s docstring for the same
    cross-layer-agreement principle applied to compaction events). Disagreement
    across layers, or a missing `kv_cluster`/expected config attribute on any
    layer, is a hard `RuntimeRkvConfigError` -- never a value silently taken
    from "whichever layer happened to be checked first"."""
    layers = model.model.layers
    if len(layers) == 0:
        raise RuntimeRkvConfigError("model.model.layers is empty -- nothing to verify.")

    per_layer_values: list[dict[str, Any]] = []
    for layer_idx, layer in enumerate(layers):
        kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
        if kv_cluster is None:
            raise RuntimeRkvConfigError(
                f"layer {layer_idx} has no self_attn.kv_cluster -- the R-KV monkeypatch was not "
                "applied to this layer, or this model was loaded as stock FullKV."
            )
        attn_config = layer.self_attn.config
        values: dict[str, Any] = {}
        for field in ("budget", "window_size", "kernel_size", "mix_lambda", "retain_ratio", "retain_direction",
                      "record_kept_token_indices"):
            if not hasattr(kv_cluster, field):
                raise RuntimeRkvConfigError(
                    f"layer {layer_idx}'s kv_cluster has no attribute {field!r} -- the pinned R-KV "
                    "submodule revision may have changed this instance's shape."
                )
            values[field] = getattr(kv_cluster, field)
        for field in ("divide_method", "divide_length", "compression_content"):
            if not hasattr(model.config, field):
                raise RuntimeRkvConfigError(
                    f"model.config has no attribute {field!r} -- RKVPolicy.load's "
                    "model.config.update(...) call may not have run for this model."
                )
            values[field] = getattr(model.config, field)
        if not hasattr(attn_config, "update_kv"):
            raise RuntimeRkvConfigError(
                f"layer {layer_idx}'s self_attn.config has no attribute 'update_kv' -- the "
                "compression_config dict passed to the R-KV monkeypatch did not include it."
            )
        values["update_kv"] = attn_config.update_kv
        per_layer_values.append(values)

    first = per_layer_values[0]
    for layer_idx, values in enumerate(per_layer_values[1:], start=1):
        for field in RUNTIME_COMPARABLE_RKV_FIELDS:
            if values[field] != first[field]:
                raise RuntimeRkvConfigError(
                    f"layer {layer_idx} disagrees with layer 0 on {field!r}: "
                    f"{values[field]!r} != {first[field]!r} -- every R-KV layer must share one "
                    "configuration; a per-layer split would mean the monkeypatch was applied "
                    "inconsistently."
                )
    return {field: first[field] for field in RUNTIME_COMPARABLE_RKV_FIELDS}


def runtime_rkv_config_hash(model: Any) -> str:
    return sha256_json(read_runtime_rkv_config(model))


@dataclass(frozen=True)
class RuntimeConfigVerificationResult:
    passed: bool
    frozen_hash: str
    runtime_hash: str
    frozen_fields: dict[str, Any]
    runtime_fields: dict[str, Any]
    mismatched_fields: tuple[str, ...]


def verify_runtime_matches_frozen(rkv: DiscoveryRkvLock, model: Any) -> RuntimeConfigVerificationResult:
    """The one function `kvcot.discovery.b2a_execute` calls: reads the
    runtime config off `model`, hashes both sides with the identical field
    set, and reports exactly which fields (if any) disagree -- never a bare
    boolean with no way to see why it failed."""
    frozen_fields = frozen_runtime_comparable_fields(rkv)
    runtime_fields = read_runtime_rkv_config(model)
    mismatched = tuple(
        field for field in RUNTIME_COMPARABLE_RKV_FIELDS if frozen_fields[field] != runtime_fields[field]
    )
    return RuntimeConfigVerificationResult(
        passed=not mismatched,
        frozen_hash=sha256_json(frozen_fields),
        runtime_hash=sha256_json(runtime_fields),
        frozen_fields=frozen_fields,
        runtime_fields=runtime_fields,
        mismatched_fields=mismatched,
    )
