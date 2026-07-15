"""Deep-clone snapshot/restore, teacher-forced replay, and branch-suffix
advancement (§6). One function serves both FullKV and R-KV — the policy
determines whether the model's layers carry a `kv_cluster` at all; there is
no separate FullKV replay path (§6: "Both conditions use the identical
replay/snapshot/branch code path.").

Imports torch at module scope by design — only ever imported from a real
generation code path.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch

from kvcot.generation.decode import decode_step, prefill
from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.sampling import greedy_next_token
from kvcot.generation.state import ModelStateSnapshot, reset_patched_state


@dataclass
class CompactionTracker:
    event_steps: list[int] = field(default_factory=list)
    last_event_absolute_position: int = 0

    def note_event(self, absolute_position: int) -> None:
        self.event_steps.append(absolute_position)
        self.last_event_absolute_position = absolute_position

    def tokens_since_last(self, absolute_position: int) -> int:
        return absolute_position - self.last_event_absolute_position

    def clone(self) -> "CompactionTracker":
        return CompactionTracker(
            event_steps=list(self.event_steps),
            last_event_absolute_position=self.last_event_absolute_position,
        )


def _has_kv_cluster(model, layer_idx: int):
    kv_cluster = getattr(model.model.layers[layer_idx].self_attn, "kv_cluster", None)
    if kv_cluster is not None and getattr(kv_cluster, "record_kept_token_indices", False):
        return kv_cluster
    return None


def _sync_layer_after_call(
    model,
    cache,
    layer_idx: int,
    model_provenance: ModelProvenance,
    kept_indices_lengths: dict[int, int],
    expected_len_if_no_evict: int,
    compaction: CompactionTracker,
    absolute_position_after: int,
) -> None:
    """Shared post-forward-call bookkeeping step, used identically after
    the prefill call and after every single-token decode call (no separate
    "prefill sync" vs "decode sync" code paths, to keep this in exactly one
    place). Cross-checks two independent eviction signals — the physical
    cache length (always available) and upstream's own
    `kv_cluster.kept_token_indices` bookkeeping (only when
    `record_kept_token_indices=True`, which this repository always sets,
    policies.py) — and fails loudly on disagreement rather than silently
    trusting one, since a disagreement would mean our understanding of the
    upstream mechanism (docs/UPSTREAM_AUDIT.md H3-H5) is wrong.
    """
    actual_len = cache.key_cache[layer_idx].shape[-2]
    evicted_by_length = actual_len < expected_len_if_no_evict

    kv_cluster = _has_kv_cluster(model, layer_idx)
    if kv_cluster is not None:
        n_after = len(kv_cluster.kept_token_indices)
        evicted_by_bookkeeping = n_after > kept_indices_lengths.get(layer_idx, 0)
        assert evicted_by_length == evicted_by_bookkeeping, (
            f"compaction-detection signals disagree at layer {layer_idx}, "
            f"absolute_position={absolute_position_after}: "
            f"cache-length says evicted={evicted_by_length}, "
            f"kept_token_indices says evicted={evicted_by_bookkeeping}"
        )
        kept_indices_lengths[layer_idx] = n_after
        if evicted_by_bookkeeping:
            model_provenance.layers[layer_idx].adopt_upstream_kept_indices(
                kv_cluster.kept_token_indices[-1]
            )
            compaction.note_event(absolute_position_after)
    elif evicted_by_length:
        # No kv_cluster (stock FullKV) but cache shrank — should be
        # structurally impossible (stock attention never evicts), so this
        # is a hard invariant violation, not a soft warning.
        raise AssertionError(
            f"layer {layer_idx} has no kv_cluster but its cache length decreased "
            f"({actual_len} < {expected_len_if_no_evict}) — stock FullKV must never evict."
        )


def replay_and_snapshot(
    model,
    fresh_cache_factory,
    prompt_token_ids: list[int],
    generated_token_ids: list[int],
    think_span,
    snapshot_absolute_positions: dict[float, int],
    device: str,
) -> dict[float, ModelStateSnapshot]:
    """Teacher-forced replay of a previously recorded base generation,
    taking deep-cloned snapshots at each requested probe fraction's
    absolute position (§6 steps 3-7). Uses the identical prefill-then-
    single-token-decode call shape as the original base generation
    (docs/REPLAY_DESIGN.md §2) — never a bulk prefill of a truncated
    prefix. `snapshot_absolute_positions` maps probe fraction -> absolute
    index into the full (prompt + generated) token stream, as returned by
    `kvcot.probes.early_answering.absolute_cut_position(...) +
    len(prompt_token_ids)`.
    """
    cache = reset_patched_state(model, fresh_cache_factory)
    num_layers = len(model.model.layers)
    num_kv_heads = model.config.num_key_value_heads
    prompt_length = len(prompt_token_ids)

    model_provenance = ModelProvenance(
        layers={i: LayerProvenance.empty(num_kv_heads) for i in range(num_layers)},
        prompt_length=prompt_length,
        think_start_absolute=(
            prompt_length + think_span.think_start_index
            if think_span.think_start_index is not None
            else None
        ),
        think_end_absolute=(
            prompt_length + think_span.think_end_index
            if think_span.think_end_index is not None
            else None
        ),
    )
    compaction = CompactionTracker()
    kept_indices_lengths: dict[int, int] = {i: 0 for i in range(num_layers)}
    snapshots: dict[float, ModelStateSnapshot] = {}

    def maybe_snapshot(pos_reached: int) -> None:
        for fraction, target_pos in snapshot_absolute_positions.items():
            if target_pos == pos_reached and fraction not in snapshots:
                snapshots[fraction] = capture_snapshot(
                    model, cache, model_provenance, compaction, pos_reached
                )

    # --- prefill: one N-token call ---
    logits, absolute_position = prefill(model, cache, prompt_token_ids, device)
    for lp in model_provenance.layers.values():
        lp.append_new_tokens_prefill(list(range(prompt_length)))
    for layer_idx in range(num_layers):
        _sync_layer_after_call(
            model, cache, layer_idx, model_provenance, kept_indices_lengths,
            expected_len_if_no_evict=prompt_length,
            compaction=compaction, absolute_position_after=absolute_position,
        )
    maybe_snapshot(absolute_position)  # covers fraction==0 when the think span starts at the prompt boundary

    # --- decode: one single-token call per recorded generated token ---
    for token_id in generated_token_ids:
        len_before = {i: cache.key_cache[i].shape[-2] for i in range(num_layers)}
        logits = decode_step(model, cache, token_id, absolute_position, device)
        fed_position = absolute_position
        absolute_position += 1
        for lp in model_provenance.layers.values():
            lp.append_new_token(fed_position)
        for layer_idx in range(num_layers):
            _sync_layer_after_call(
                model, cache, layer_idx, model_provenance, kept_indices_lengths,
                expected_len_if_no_evict=len_before[layer_idx] + 1,
                compaction=compaction, absolute_position_after=absolute_position,
            )
        maybe_snapshot(absolute_position)

    missing = set(snapshot_absolute_positions) - set(snapshots)
    if missing:
        raise RuntimeError(
            f"replay finished without reaching requested snapshot fraction(s) {missing} "
            "(the recorded generation was shorter than expected, or a cut position was "
            "computed against the wrong think span)"
        )
    return snapshots


def _compression_flag_to_str(v) -> str:
    if v is None:
        return "none"
    return "true" if v else "false"


def _compression_flag_from_str(s: str):
    return {"none": None, "true": True, "false": False}[s]


def capture_snapshot(
    model, cache, model_provenance: ModelProvenance, compaction: CompactionTracker, absolute_position: int
) -> ModelStateSnapshot:
    """Deep-clone every field enumerated in docs/REPLAY_DESIGN.md §3.
    Every tensor is `.clone()`d; every Python container is deep-copied.
    Never a view/alias onto the live model/cache state."""
    num_layers = len(model.model.layers)

    kv_cluster_bookkeeping: list[dict] = []
    for layer_idx in range(num_layers):
        kv_cluster = _has_kv_cluster(model, layer_idx)
        if kv_cluster is None:
            kv_cluster_bookkeeping.append({})
            continue
        kv_cluster_bookkeeping.append(
            {
                "evicted_token_num": kv_cluster.evicted_token_num,
                "kept_token_indices": copy.deepcopy(kv_cluster.kept_token_indices),
                "kept_attention_scores": copy.deepcopy(kv_cluster.kept_attention_scores),
                "kept_similarity_scores": copy.deepcopy(getattr(kv_cluster, "kept_similarity_scores", [])),
                "kept_final_scores": copy.deepcopy(getattr(kv_cluster, "kept_final_scores", [])),
            }
        )

    return ModelStateSnapshot(
        key_cache=[cache.key_cache[i].clone() for i in range(num_layers)],
        value_cache=[cache.value_cache[i].clone() for i in range(num_layers)],
        query_cache={
            i: t.clone() for i, t in getattr(cache, "query_cache", {}).items()
        },
        compression_flags_per_layer=[
            _compression_flag_to_str(model.model.layers[i].self_attn.config.compression)
            for i in range(num_layers)
        ],
        model_length=getattr(model, "length", 0),
        after_think=getattr(model, "after_think", None),
        compaction_event_steps=list(compaction.event_steps),
        tokens_since_last_compaction=compaction.tokens_since_last(absolute_position),
        absolute_position=absolute_position,
        provenance=model_provenance.clone(),
        kv_cluster_bookkeeping_per_layer=kv_cluster_bookkeeping,
    )


def restore_snapshot(model, cache, snapshot: ModelStateSnapshot) -> ModelProvenance:
    """Restore live model/cache state from a deep-cloned snapshot, for
    branching (§6 step 8). Always restores from a *fresh clone* of the
    snapshot's tensors (never the snapshot's own tensors directly), so
    restoring the same snapshot twice cannot let the two branches share
    mutable storage — combined with `capture_snapshot`'s own `.clone()`
    calls at capture time, this gives two full clone boundaries around
    every reuse of a snapshot.
    """
    num_layers = len(model.model.layers)
    for i in range(num_layers):
        cache.key_cache[i] = snapshot.key_cache[i].clone()
        cache.value_cache[i] = snapshot.value_cache[i].clone()
    cache.query_cache = {i: t.clone() for i, t in snapshot.query_cache.items()}

    for i in range(num_layers):
        model.model.layers[i].self_attn.config.compression = _compression_flag_from_str(
            snapshot.compression_flags_per_layer[i]
        )
        if snapshot.kv_cluster_bookkeeping_per_layer:
            bk = snapshot.kv_cluster_bookkeeping_per_layer[i]
            kv_cluster = _has_kv_cluster(model, i)
            if kv_cluster is not None and bk:
                kv_cluster.evicted_token_num = bk["evicted_token_num"]
                kv_cluster.kept_token_indices = copy.deepcopy(bk["kept_token_indices"])
                kv_cluster.kept_attention_scores = copy.deepcopy(bk["kept_attention_scores"])
                if hasattr(kv_cluster, "kept_similarity_scores"):
                    kv_cluster.kept_similarity_scores = copy.deepcopy(bk["kept_similarity_scores"])
                if hasattr(kv_cluster, "kept_final_scores"):
                    kv_cluster.kept_final_scores = copy.deepcopy(bk["kept_final_scores"])

    model.length = snapshot.model_length
    if snapshot.after_think is not None:
        model.after_think = snapshot.after_think
    elif hasattr(model, "after_think"):
        del model.after_think

    return snapshot.provenance.clone()


@dataclass(frozen=True)
class ProbeResult:
    control_suffix_token_ids: list[int]
    probe_output_token_ids: list[int]


def branch_and_probe(
    model,
    cache,
    snapshot: ModelStateSnapshot,
    close_marker_token_ids: list[int],
    control_suffix_token_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int,
    device: str,
) -> ProbeResult:
    """§6 steps 8-9: from a deep-cloned snapshot, teacher-force the
    closing-think token sequence and then the single control suffix
    (kvcot.probes.templates.render_control_suffix, already tokenized by the
    caller — this module has no tokenizer dependency of its own), advancing
    policy and positions one token at a time (never a multi-token batch,
    per docs/REPLAY_DESIGN.md §2 on why call shape matters), then generates
    up to `max_new_tokens` answer tokens greedily (§4: probe decoding is
    always greedy/deterministic).

    Restoring the same `snapshot` and calling this twice must yield
    identical `probe_output_token_ids` (§6.1 hard gate) — guaranteed by
    `restore_snapshot`'s clone-on-restore plus this function doing no
    resampling anywhere (teacher-forced feed, then greedy argmax decode).
    """
    restore_snapshot(model, cache, snapshot)
    absolute_position = snapshot.absolute_position

    def feed(token_id: int) -> torch.Tensor:
        nonlocal absolute_position
        logits = decode_step(model, cache, token_id, absolute_position, device)
        absolute_position += 1
        return logits

    logits = None
    for token_id in close_marker_token_ids:
        logits = feed(token_id)
    for token_id in control_suffix_token_ids:
        logits = feed(token_id)

    if logits is None:
        raise ValueError(
            "branch_and_probe requires at least one token in "
            "close_marker_token_ids + control_suffix_token_ids to establish next-token logits"
        )

    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_id = greedy_next_token(logits)
        token_id = int(next_id.item())
        generated.append(token_id)
        if token_id == eos_token_id:
            break
        logits = feed(token_id)

    return ProbeResult(
        control_suffix_token_ids=control_suffix_token_ids,
        probe_output_token_ids=generated,
    )
