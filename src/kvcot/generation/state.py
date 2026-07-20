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

import copy
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

    Architecture-generic by construction, not Qwen2-specific: verified
    directly against `third_party/R-KV/HuggingFace/rkv/modeling.py`'s three
    `*Attention_init` functions (Llama/Qwen2/Qwen3, dispatched by
    `kvcot.discovery.dispatch`) — all three attach state under the
    identical attribute names this function reads/resets
    (`self_attn.config.compression`, `self_attn.kv_cluster`, and the
    CausalLM-level `self.length`/`self.after_think`, set by the one shared
    `CausalLM_forward` all three patchers install). No architecture defines
    any additional mutable state beyond what is reset here — confirmed by
    inspection, not assumed; re-check this comment against the pinned
    submodule if a new architecture is ever added to the dispatch table.

    `fresh_cache_factory` is a zero-arg callable returning a brand-new
    transformers Cache instance (e.g. `lambda: DynamicCache()`) — a *new*
    instance is required, not a cleared one, since `query_cache`
    (modeling.py:260-261) is a `hasattr`-gated dynamic attribute that only
    "doesn't exist" on a genuinely fresh object; clearing a dict on a reused
    Cache is not equivalent to that attribute never having been set.

    Returns the fresh cache so callers thread it through the next forward
    call explicitly — this function never holds cache state itself.

    B1 execution-boundary closure (2026-07-20): this function owns MODEL
    and CACHE state only -- it must never reset, synchronize, or otherwise
    touch CUDA memory-measurement state. It used to call
    `torch.cuda.reset_peak_memory_stats()` as a bundled side effect, which
    is a genuine measurement-integrity defect for any caller that
    constructs more than one fresh state within a single measured window
    (`kvcot.discovery.b2a_workers.run_rkv_worker` calls this once for Pass
    1's initial state and once more, via `pass2_initial_state_factory`,
    for Pass 2's -- the SECOND call used to silently wipe whatever peak
    Pass 1 had already accumulated, moments before Pass 2 even started).
    Peak-memory reset is now owned exclusively by each caller's own
    measurement-boundary code (`kvcot.discovery.b2a_workers` already had
    its own explicit, correctly-scoped reset per worker spanning Pass 1
    through Pass 2 and branch evaluation; `kvcot.cli.cmd_generate` and
    `kvcot.generation.replay.replay_and_snapshot` gained an equivalent
    explicit reset immediately adjacent to their own `reset_patched_state`
    call, preserving their exact prior behavior).
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
        # layer (constructed inside the dispatched *Attention_init function
        # — Qwen2Attention_init, LlamaAttention_init, or Qwen3Attention_init,
        # modeling.py — one call per layer, so one instance per layer,
        # identically across all three architectures); only
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

    def clone(self) -> "ModelStateSnapshot":
        """An independent deep copy — every tensor `.clone()`d, every
        container deep-copied, never a view/alias onto `self`'s own storage
        (`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §5: "No
        storage alias may exist between pristine, baseline, and swapped
        mutable tensors"). Added for B1B-R2's discovery-harness branch
        construction (`kvcot.discovery.pipeline.build_swap_pair_record`),
        which needs two independently-mutable working copies (baseline,
        swapped) from one pristine post-event snapshot without a live
        model/cache to restore into — the pre-existing `capture_snapshot`/
        `restore_snapshot` pair in `kvcot.generation.replay` remains the
        only path for the PRIMARY pipeline, which always has a live model
        and cache to restore state into; this method fills the one gap
        that pair does not cover (a snapshot-to-snapshot clone with no live
        model in the loop at all)."""
        return ModelStateSnapshot(
            key_cache=[t.clone() for t in self.key_cache],
            value_cache=[t.clone() for t in self.value_cache],
            query_cache={i: t.clone() for i, t in self.query_cache.items()},
            compression_flags_per_layer=list(self.compression_flags_per_layer),
            model_length=self.model_length,
            after_think=self.after_think,
            compaction_event_steps=list(self.compaction_event_steps),
            tokens_since_last_compaction=self.tokens_since_last_compaction,
            absolute_position=self.absolute_position,
            provenance=self.provenance.clone() if self.provenance is not None else None,
            kv_cluster_bookkeeping_per_layer=(
                copy.deepcopy(self.kv_cluster_bookkeeping_per_layer)
                if self.kv_cluster_bookkeeping_per_layer is not None
                else None
            ),
        )
