"""Mutable-state inventory, reset, and single-process stock/patched guard
(§3.1, §3.2 of the build brief).

Every mutable-state item handled here is cited against
docs/UPSTREAM_AUDIT.md §3.2 — this module is the executable version of that
audit's inventory. If upstream's actual attribute names ever drift
(transformers version bump, R-KV update on `third_party/R-KV`), both places
need to change together, and `tests/integration/test_no_state_leak_gpu.py`
is the GPU-side check that would catch a silent drift.

This module imports torch at module scope by design (see pyproject.toml's
note on deferred GPU imports) — it is only ever imported from inside a
generation/replay code path, never from `kvcot.cli`'s `--dry-run` branch or
from `kvcot.analysis`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch

_ACTIVE_MODE: Literal["stock", "patched"] | None = None


class ProcessModeConflictError(RuntimeError):
    pass


def declare_process_mode(mode: Literal["stock", "patched"]) -> None:
    """§3.2: "Never load stock and patched models in the same Python
    process." The R-KV monkeypatch is a process-global class patch on
    `transformers.models.qwen2.modeling_qwen2` (docs/UPSTREAM_AUDIT.md H1)
    — there is no per-instance undo, so this cannot be enforced by the
    patch itself. Call this once, before loading any model, from every
    entry point (`kvcot.generation.policies.FullKVPolicy.load` uses
    mode="stock"; `PatchedNoopPolicy`/`RKVPolicy` use mode="patched").
    Raises if a conflicting mode was already declared in this process.
    """
    global _ACTIVE_MODE
    if _ACTIVE_MODE is not None and _ACTIVE_MODE != mode:
        raise ProcessModeConflictError(
            f"process already declared mode={_ACTIVE_MODE!r}; cannot also use mode={mode!r} "
            "in the same process. Stock (FullKV) and patched (patched_noop/R-KV) models must "
            "run in separate OS processes — see docs/UPSTREAM_AUDIT.md H1 and §3.2 of the "
            "build brief. Launch a new process per condition rather than switching modes here."
        )
    _ACTIVE_MODE = mode


def reset_active_mode_for_testing() -> None:
    """Test-only escape hatch. Production code paths never call this — a
    real process is only ever one mode for its whole lifetime."""
    global _ACTIVE_MODE
    _ACTIVE_MODE = None


def reset_patched_state(model: Any, fresh_cache_factory: Callable[[], Any]) -> Any:
    """Reset every documented mutable state item before an independent base
    generation or a fresh replay (§3.1). Must run immediately before every
    independent base generation and every fresh replay.

    `fresh_cache_factory` is a zero-arg callable returning a brand-new
    transformers Cache instance (e.g. `lambda: DynamicCache()`) — a *new*
    instance is required, not a cleared one, since `query_cache`
    (modeling.py:260-261) is a `hasattr`-gated dynamic attribute that only
    "doesn't exist" on a genuinely fresh object; clearing a dict on a reused
    Cache is not equivalent to that attribute never having been set.

    Returns the fresh cache so callers thread it through the next forward
    call explicitly — this function never holds cache state itself.
    """
    # self.length (modeling.py:560-563) — cumulative absolute token counter
    # on the CausalLM model object. Deleting it (not zeroing it) matches
    # upstream's own `if not hasattr(self, "length")` first-use check.
    if hasattr(model, "length"):
        del model.length

    # self.after_think (modeling.py:556-559, 593-596) — only ever created
    # when compression_content == "think". This repo freezes
    # compression_content=all (§4), so it should never actually be present,
    # but the reset must not assume its absence (UPSTREAM_AUDIT.md §3.2).
    if hasattr(model, "after_think"):
        del model.after_think

    # layer.self_attn.config.compression (modeling.py:609-610) — tri-state
    # (None/True/False) per-layer flag. Reset to the documented initial
    # value (None, set from compression_config["compression"] at
    # run_math.py:230) on every layer individually — upstream itself sets
    # it with an explicit per-layer loop, so a shared/aliased reset would
    # be wrong if configs are ever not literally the same object per layer.
    for layer in model.model.layers:
        layer.self_attn.config.compression = None

        # kv_cluster bookkeeping (r1_kv.py:29-35) — one R1KV instance per
        # layer (constructed inside Qwen2Attention_init, modeling.py:230-234
        # — one call per layer, so one instance per layer); only
        # present/meaningful when record_kept_token_indices=True.
        kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
        if kv_cluster is not None and getattr(kv_cluster, "record_kept_token_indices", False):
            kv_cluster.evicted_token_num = 0
            kv_cluster.kept_token_indices = []
            kv_cluster.kept_attention_scores = []
            if hasattr(kv_cluster, "kept_similarity_scores"):
                kv_cluster.kept_similarity_scores = []
            if hasattr(kv_cluster, "kept_final_scores"):
                kv_cluster.kept_final_scores = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # query_cache (modeling.py:260-261) needs no explicit reset here: it is
    # a `hasattr`-gated dict dynamically attached to the Cache object, so a
    # genuinely fresh Cache instance (returned below) naturally has none of
    # it — this is exactly why fresh_cache_factory must construct a new
    # object rather than reuse/clear one.
    return fresh_cache_factory()


@dataclass
class ModelStateSnapshot:
    """Deep-cloned model/cache state at one point in decoding (§6 step 8,
    docs/REPLAY_DESIGN.md §3). Every tensor field is `.clone()`d, never a
    view/alias, so branching from the same snapshot twice cannot let one
    branch's teacher-forced writes bleed into another
    (§6.1 hard gate: "restoring the same snapshot twice ... yields
    identical probe tokens").

    `provenance` and `compaction_event_steps` are plain Python objects,
    deep-copied via `copy.deepcopy` by the snapshot constructor in
    kvcot.generation.replay, not shared references.
    """

    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]
    query_cache: dict[int, torch.Tensor]
    compression_flags_per_layer: list[Literal["none", "true", "false"]]
    model_length: int
    after_think: bool | None
    compaction_event_steps: list[int] = field(default_factory=list)
    tokens_since_last_compaction: int = 0
    absolute_position: int = 0
    # Populated by kvcot.generation.provenance; kept as a loosely-typed
    # field here to avoid a state.py <-> provenance.py import cycle.
    provenance: Any = None
    # Deep copy of each layer's kv_cluster (R1KV) bookkeeping lists
    # (evicted_token_num, kept_token_indices, kept_attention_scores,
    # kept_similarity_scores, kept_final_scores — r1_kv.py:29-35), only
    # when record_kept_token_indices=True. Required for the §6.1 hard gate
    # "restoring the same snapshot twice ... yields identical probe
    # tokens": without restoring this too, a second branch from the same
    # snapshot would inherit the first branch's mutations to these lists
    # (they live on the model's kv_cluster objects, not the cache), and
    # upstream's own multi-event remap (r1_kv.py:141-154) reads them, so a
    # stale list would corrupt the *next* real compaction event's absolute-
    # position remapping on the second branch.
    kv_cluster_bookkeeping_per_layer: list[dict[str, Any]] | None = None
