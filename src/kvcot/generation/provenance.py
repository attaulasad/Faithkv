"""Per-layer, per-KV-head absolute source position tracking (§3.4, §6),
without altering R-KV's scores or selected indices (docs/UPSTREAM_AUDIT.md
H5: confirmed that `R1KV(record_kept_token_indices=True)` only adds
bookkeeping computed from the *same* `indices` used for the real eviction —
it does not change what gets evicted).

Design, grounded in r1_kv.py (docs/UPSTREAM_AUDIT.md H3, H5):

  - Physical cache slot -> absolute source token position is a 1:1 mapping
    that changes ONLY at an actual eviction event (a `topk`-then-`gather`,
    r1_kv.py:82,168-179) — between events the cache simply grows by
    appending one new slot per new token, with no reordering. So this
    adapter does NOT need to reimplement the remap itself: it appends new
    absolute positions between events (cheap, exact), and at an eviction
    event it adopts upstream's own already-remapped
    `kv_cluster.kept_token_indices[-1]` (r1_kv.py:141-165) as ground truth
    for the post-event mapping — that list already resolves multi-event
    remapping (its `evicted_token_num`/`prev_indices` bookkeeping,
    r1_kv.py:141-154) into absolute positions, and per H5, reading it
    changes nothing about what R1KV actually evicted.

  - Eviction happens independently per KV head (r1_kv.py's `topk` runs
    along the last dim of a `(bsz, num_kv_heads, ...)` tensor), so this
    adapter tracks positions per (layer, kv_head), not just per layer.

  - Detecting *whether* an eviction event happened on a given forward call
    is done by comparing `len(kv_cluster.kept_token_indices)` before and
    after the call — GPU-unverified assumption, flagged in
    docs/REPLAY_DESIGN.md §5 alongside the other open assumptions.

Imports torch at module scope by design — only ever imported from a real
generation/replay code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class LayerProvenance:
    """Absolute source token position per (kv_head, physical_slot) for one
    layer. `positions[h, j]` is the absolute index (into the full
    prompt+generated token stream) of whatever content currently occupies
    KV-head h's physical cache slot j."""

    positions: torch.Tensor  # shape (num_kv_heads, current_cache_len), dtype long, on CPU

    @classmethod
    def empty(cls, num_kv_heads: int) -> "LayerProvenance":
        return cls(positions=torch.empty((num_kv_heads, 0), dtype=torch.long))

    def append_new_token(self, absolute_position: int) -> None:
        """No eviction happened on this call: the cache grew by exactly one
        slot (this repository's frozen batch-1, one-token-per-decode-call
        loop, docs/REPLAY_DESIGN.md §2), holding the newly processed
        token's absolute position, identically across every KV head."""
        num_heads = self.positions.shape[0]
        new_col = torch.full((num_heads, 1), absolute_position, dtype=torch.long)
        self.positions = torch.cat([self.positions, new_col], dim=1)

    def append_new_tokens_prefill(self, absolute_positions: list[int]) -> None:
        """Prefill call: the cache grows by the entire prompt length in one
        step (docs/REPLAY_DESIGN.md §2), identically across every KV head."""
        num_heads = self.positions.shape[0]
        block = torch.tensor(absolute_positions, dtype=torch.long).unsqueeze(0).expand(num_heads, -1)
        self.positions = torch.cat([self.positions, block], dim=1)

    def adopt_upstream_kept_indices(self, kept_indices_absolute: torch.Tensor) -> None:
        """An eviction event happened. Replace this layer's tracked
        positions with upstream's own already-remapped absolute positions
        for the survivors (r1_kv.py:141-165), shape (num_kv_heads, budget).
        """
        self.positions = kept_indices_absolute.clone().to(dtype=torch.long, device="cpu")

    def clone(self) -> "LayerProvenance":
        return LayerProvenance(positions=self.positions.clone())


@dataclass
class ModelProvenance:
    """One LayerProvenance per transformer layer, plus the prompt/think
    absolute-position boundaries needed to classify survivors (§3.4)."""

    layers: dict[int, LayerProvenance] = field(default_factory=dict)
    prompt_length: int = 0
    think_start_absolute: int | None = None
    think_end_absolute: int | None = None

    def clone(self) -> "ModelProvenance":
        return ModelProvenance(
            layers={idx: lp.clone() for idx, lp in self.layers.items()},
            prompt_length=self.prompt_length,
            think_start_absolute=self.think_start_absolute,
            think_end_absolute=self.think_end_absolute,
        )


def sync_after_forward_call(
    model_provenance: ModelProvenance,
    model,
    kept_indices_before: dict[int, int],
) -> dict[int, int]:
    """Call once after every forward call (prefill or single-token decode)
    for which the caller has already appended the new token's absolute
    position(s) via `append_new_token`/`append_new_tokens_prefill` on every
    layer. Detects, per layer, whether an eviction event fired during that
    call (`len(kv_cluster.kept_token_indices)` grew) and if so, replaces
    that layer's tracked positions with upstream's own remapped result.

    `kept_indices_before` is `{layer_idx: len(kv_cluster.kept_token_indices)
    before this forward call}`, threaded in/out by the caller so this
    function stays a pure per-call step rather than owning long-lived state
    itself. Returns the updated lengths for the next call.
    """
    kept_indices_after: dict[int, int] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
        if kv_cluster is None or not getattr(kv_cluster, "record_kept_token_indices", False):
            kept_indices_after[layer_idx] = 0
            continue
        n_before = kept_indices_before.get(layer_idx, 0)
        n_after = len(kv_cluster.kept_token_indices)
        kept_indices_after[layer_idx] = n_after
        if n_after > n_before:
            # An eviction event fired on this call for this layer. Take the
            # most recent (last-appended) recorded absolute positions.
            model_provenance.layers[layer_idx].adopt_upstream_kept_indices(
                kv_cluster.kept_token_indices[-1]
            )
    return kept_indices_after


@dataclass(frozen=True)
class RetentionSummaryResult:
    prompt_tokens_total: int
    prompt_tokens_surviving_mean: float
    think_tokens_total: int
    think_tokens_surviving_mean: float


def compute_provenance_retention_summary(model_provenance: ModelProvenance) -> RetentionSummaryResult:
    """§3.4/§9: report prompt-token retention and reasoning-token retention
    separately, aggregated (mean) across layers and KV heads. `prompt`
    means absolute position < prompt_length; `think` means
    think_start_absolute <= position < think_end_absolute. Positions
    outside both ranges (e.g. already-generated non-think tail tokens, or
    an unparsed think span) are not counted in either bucket.
    """
    prompt_len = model_provenance.prompt_length
    think_start = model_provenance.think_start_absolute
    think_end = model_provenance.think_end_absolute

    prompt_counts: list[float] = []
    think_counts: list[float] = []
    for layer in model_provenance.layers.values():
        positions = layer.positions
        if positions.numel() == 0:
            prompt_counts.append(0.0)
            think_counts.append(0.0)
            continue
        # Per-KV-head counts, then averaged across heads for this layer
        # (aggregation documented here per §9's requirement to document the
        # exact method used).
        is_prompt = positions < prompt_len
        prompt_counts.append(is_prompt.sum(dim=-1).float().mean().item())
        if think_start is not None and think_end is not None:
            is_think = (positions >= think_start) & (positions < think_end)
            think_counts.append(is_think.sum(dim=-1).float().mean().item())
        else:
            think_counts.append(0.0)

    prompt_total = prompt_len
    think_total = (think_end - think_start) if (think_start is not None and think_end is not None) else 0

    return RetentionSummaryResult(
        prompt_tokens_total=prompt_total,
        prompt_tokens_surviving_mean=sum(prompt_counts) / len(prompt_counts) if prompt_counts else 0.0,
        think_tokens_total=think_total,
        think_tokens_surviving_mean=sum(think_counts) / len(think_counts) if think_counts else 0.0,
    )
