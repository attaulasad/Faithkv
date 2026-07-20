"""Real-model `PrefillFn`/`DecodeOneFn`/`SnapshotFn`/branch-evaluation
adapters (`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §11,
repaired by `docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md` §8/§9) -- the seam a
future, separately-authorized B2A execution plugs into. This module is
CODE, not a stub: every function below is a real, reviewable
implementation built entirely from primitives the PRIMARY pipeline already
uses and has exercised (`kvcot.generation.decode`, `kvcot.generation.state`,
`kvcot.generation.replay`, `kvcot.generation.provenance`) -- it never
reimplements the call shape or the eviction-detection logic independently.

**This module is never imported by any GPU-executing test or code path in
this repository as of this pass** -- `cmd_b2a_calibrate`'s `--execute` mode
is the only caller, and that mode requires CUDA (unavailable on this
CPU-only build) and a fully-resolved manifest before it would ever reach
here. No GPU has run this code; no model has been loaded through it. Every
`import torch`/`transformers` reference below is deferred to inside a
function body, matching this repository's existing discipline for
GPU-only code (`kvcot.generation`, `kvcot.cli`). Its pure state-bookkeeping
logic (independent of any real model forward call) IS exercised by CPU
tests using lightweight fake `model`/`cache` objects
(`tests/unit/discovery/test_real_model_adapter_state.py`).

## B1B-R3 §8 repair: one authoritative real-model state-advance lifecycle

B1B-R2's `build_real_prefill_fn`/`build_real_decode_one_fn` called
`kvcot.generation.replay._sync_layer_after_call` directly but never called
`LayerProvenance.append_new_tokens_prefill`/`append_new_token` on
`RealModelState.model_provenance` first -- unlike
`kvcot.generation.replay.replay_and_snapshot`'s own prefill/decode blocks,
which always append BEFORE syncing. Since `_sync_layer_after_call` only
WRITES `model_provenance.layers[i].positions` wholesale when an eviction
event fires (`LayerProvenance.adopt_upstream_kept_indices`), and otherwise
leaves it untouched, `RealModelState.model_provenance.layers[i].positions`
would have stayed permanently EMPTY between prefill and the first real
eviction event on this adapter's original code -- silently corrupting
`_layer_observations_for_all_layers`'s `pre_event_absolute_position_map`
(read directly off `model_provenance.layers[layer_idx].positions.clone()`)
at exactly the moment Pass 1's candidate/donor pool selection
(`kvcot.discovery.pass1._pools_for_layer_head`) depends on it being
correct. `advance_after_forward` below is now the ONE function that both
appends new-token positions AND syncs/notes compaction events, so this gap
cannot recur at a second call site -- `build_real_prefill_fn`,
`build_real_decode_one_fn`, and the restore-once branch evaluator all call
it, never duplicating the append-then-sync sequence independently.

## B1B-R3 §9 repair: restore exactly once per branch, never once per token

B1B-R2's `_build_real_branch_step_fn` restored a COMPLETE
`ModelStateSnapshot` into a fresh cache on EVERY call -- i.e. once per
scored token (49 restores for one bridge-plus-48 branch), not once per
branch. `build_real_branch_step_fn_restore_once` below restores only on
the FIRST call for a branch (when it receives the initial
`ModelStateSnapshot`) and reuses the already-restored live cache/provenance
for every subsequent call in that same branch (when it receives the
`_LiveBranchState` its own previous call returned) -- `branch_eval
.evaluate_branch`'s existing functional `(state, token_id) ->
(logits, new_state)` contract is unchanged, so no other module needs to
change to get this fix; only the concrete real-model closure does.

## B1B-R4.1 §4 repair: one authoritative provenance state, pending-position
projection instead of a Pass-2 shadow tracker

Before this pass, `kvcot.discovery.pass2.run_pass2_capture` maintained its
OWN mutable `dict[int, LayerProvenance]` for the real-model path, hand-kept
in lockstep with `advance_after_forward`'s appends purely by parallel,
independently-written code at two call sites -- a real risk of silent
divergence with no structural guarantee the two tracks agree. Pass 2 needs
this because `capture_update_kv`'s `pre_event_position_map_fn` is read
DURING the real forward call (from inside `update_kv`, called by
`state.model(...)`), i.e. strictly BEFORE `advance_after_forward` (called
AFTER the forward call returns) has appended this call's fed positions to
`state.model_provenance` -- so the map available at that instant must
reflect a PROJECTED state ("if this in-flight call's fed positions were
already appended"), not yet the committed one.

`RealModelState.projected_pre_event_position_map` is the fix:
`register_pending_fed_positions` is called by every real forward-call site
below (`prefill_fn`, `decode_one_fn`, the branch step function) BEFORE the
actual `state.model(...)`/`decode_step(...)` call; the projection method
derives the pre-event map on demand from `state.model_provenance` (the one
authoritative owner) plus the still-pending fed positions, via a disposable
`LayerProvenance.clone()` mutated with the SAME `append_new_token`/
`append_new_tokens_prefill` methods `advance_after_forward` itself uses
(never a second, independently-written position-arithmetic implementation)
-- the clone is discarded immediately after; `state.model_provenance` is
never touched by the projection. `advance_after_forward` remains the only
function that ever commits a real transition; pending state is cleared
after it succeeds, or immediately (before `advance_after_forward` is even
attempted) if the forward call itself raises -- never a partial commit.

`kvcot.discovery.pass2.run_pass2_capture` now detects (via
`hasattr(state, "projected_pre_event_position_map")`) whether it has been
handed a state that owns its own authoritative projection; when it has
(the real-model path), Pass 2 builds no parallel `LayerProvenance` dict of
its own at all and reads the projection directly. The CPU synthetic harness
state (`tests/unit/discovery/_synthetic_harness.py`'s `HarnessState`) does
not implement this method, so its code path through Pass 2 is completely
unchanged -- this repair touches only the real-model path.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from kvcot.discovery.harness_types import (
    DecodeOneFn,
    LayerStepObservation,
    NaturalStepResult,
    PrefillFn,
    PrefillStepResult,
    SnapshotFn,
)


@dataclass
class RealModelState:
    """The `state` object threaded through Pass 1/Pass 2 for a REAL model --
    bundles exactly what `kvcot.generation.replay.replay_and_snapshot`
    already bundles as local variables, so the same reset/sync helpers
    apply unchanged. `kv_cluster_for_layer` is the one method
    `kvcot.discovery.pass2.run_pass2_capture` requires of any state object.

    Single authoritative owner (B1B-R3 §8, tightened by B1B-R4.1 §4) of:
    the live model/cache, the complete `ModelProvenance` (every layer --
    Pass 2 no longer keeps any second, independently-mutated per-target
    provenance track of its own; it reads `projected_pre_event_position_map`
    below instead), the `CompactionTracker`, the absolute position, and the
    device. `kept_indices_lengths` is NOT persisted here deliberately --
    every call site recomputes it fresh from the live
    `kv_cluster.kept_token_indices` length at the start of each call
    (self-correcting from the model's own ground truth, never threaded
    state that could drift).

    `pending_fed_absolute_positions`/`pending_call_kind` (B1B-R4.1 §4) are
    the one piece of NOT-YET-COMMITTED state this class carries: the
    positions a real forward call currently in flight is about to feed,
    registered by the caller immediately before that call via
    `register_pending_fed_positions` and cleared via `clear_pending` either
    after `advance_after_forward` commits them for real, or immediately if
    the forward call itself raised -- never left set across two different
    calls, and never used to mutate `model_provenance` directly (only
    `advance_after_forward` ever does that)."""

    model: Any
    cache: Any
    model_provenance: Any  # kvcot.generation.provenance.ModelProvenance
    compaction: Any  # kvcot.generation.replay.CompactionTracker
    absolute_position: int
    device: str
    pending_fed_absolute_positions: list[int] | None = field(default=None)
    pending_call_kind: Literal["prefill", "decode"] | None = field(default=None)

    def kv_cluster_for_layer(self, layer_index: int):
        return self.model.model.layers[layer_index].self_attn.kv_cluster

    def register_pending_fed_positions(self, positions: list[int], call_kind: Literal["prefill", "decode"]) -> None:
        """Must be called immediately before the real forward call that will
        feed `positions` -- makes them visible to `projected_pre_event_position_map`
        for the duration of that call, without touching `model_provenance`."""
        self.pending_fed_absolute_positions = list(positions)
        self.pending_call_kind = call_kind

    def clear_pending(self) -> None:
        self.pending_fed_absolute_positions = None
        self.pending_call_kind = None

    def projected_pre_event_position_map(self, layer_index: int) -> "Any":
        """The pre-event absolute-position map for `layer_index`, as it
        would read AT THIS INSTANT from inside the in-flight forward call's
        `update_kv` hook -- `model_provenance`'s own committed positions plus
        any still-pending fed positions for this call, computed via a
        disposable `LayerProvenance.clone()` (never mutating
        `model_provenance` itself, and never re-deriving append semantics
        independently of `LayerProvenance.append_new_token`/
        `append_new_tokens_prefill`, the same methods `advance_after_forward`
        uses for the real commit)."""
        return _project_pre_event_position_map(self, layer_index)


def compute_kept_indices_lengths(model: Any) -> dict[int, int]:
    """Snapshot of `len(kv_cluster.kept_token_indices)` per layer, read off
    the LIVE model -- callers must call this BEFORE the forward call they
    are about to advance past (`advance_after_forward` compares against it
    AFTER that call to detect growth), exactly the same "before" snapshot
    `kvcot.generation.replay.replay_and_snapshot` threads across its own
    prefill/decode calls."""
    from kvcot.generation.replay import _has_kv_cluster

    num_layers = len(model.model.layers)
    return {
        i: (len(kv_cluster.kept_token_indices) if (kv_cluster := _has_kv_cluster(model, i)) is not None else 0)
        for i in range(num_layers)
    }


def _project_pre_event_position_map(state: Any, layer_index: int) -> "Any":
    """Module-level implementation backing `RealModelState
    .projected_pre_event_position_map` (kept free-standing, not just a
    method, so `build_real_branch_step_fn_restore_once`'s `_LiveBranchState`
    -- which also owns a `model_provenance` and can also have pending
    positions mid-call, even though nothing currently reads its projection
    -- can share the identical implementation without duplicating it)."""
    lp = state.model_provenance.layers.get(layer_index)
    if lp is None:
        return None
    pending = state.pending_fed_absolute_positions
    if not pending:
        return lp.positions
    projected = lp.clone()
    call_kind = state.pending_call_kind
    if call_kind == "prefill":
        projected.append_new_tokens_prefill(pending)
    elif call_kind == "decode":
        projected.append_new_token(pending[0])
    else:
        raise ValueError(
            f"pending_call_kind must be 'prefill' or 'decode' when pending positions are set, got {call_kind!r}"
        )
    return projected.positions


@contextlib.contextmanager
def _pending_positions_scope(state: Any, positions: list[int], call_kind: Literal["prefill", "decode"]) -> Iterator[None]:
    """B1B-R4.1 §4: registers `positions` as pending on `state` before the
    real forward call the caller is about to make, and guarantees they are
    cleared if that call raises -- so a failed forward call never leaves
    stale pending state visible to a later, unrelated call. Does NOT clear
    pending state on the successful path (the caller does that explicitly,
    after `advance_after_forward` has actually committed the transition) --
    this context manager only ever owns the exception path."""
    state.register_pending_fed_positions(positions, call_kind)
    try:
        yield
    except BaseException:
        state.clear_pending()
        raise


def advance_after_forward(
    model: Any,
    cache: Any,
    model_provenance: Any,
    compaction: Any,
    *,
    fed_absolute_positions: list[int],
    expected_cache_lengths_if_no_eviction: dict[int, int],
    kept_indices_lengths_before_call: dict[int, int],
    call_kind: Literal["prefill", "decode"],
) -> dict[int, "LayerStepObservation"]:
    """The one shared real-model state-advance step (B1B-R3 §8), called
    AFTER the actual forward call has already run. In order:

    1. Appends `fed_absolute_positions` to every layer's provenance
       (`LayerProvenance.append_new_tokens_prefill` for `call_kind=
       "prefill"`, `append_new_token` for `call_kind="decode"` --
       `fed_absolute_positions` must have exactly one element in that case).
    2. Synchronizes every layer via `kvcot.generation.replay
       ._sync_layer_after_call` (reused directly, never reimplemented) --
       this also mutates `kept_indices_lengths_before_call` in place to the
       post-call count, exactly like `replay_and_snapshot`'s own threaded
       dict, so a caller keeping the same dict object across the whole run
       and passing it back in unchanged next call gets a self-consistent
       "before" snapshot every time without recomputing it from the model.
    3. Requires cross-layer event agreement via `_note_event_once`.
    4. Returns one `LayerStepObservation` per layer, built from the same
       synced state -- callers (`build_real_prefill_fn`,
       `build_real_decode_one_fn`, the restore-once branch evaluator) use
       this directly instead of re-deriving observations independently.

    `expected_cache_lengths_if_no_eviction` and `kept_indices_lengths_before_call`
    must both be computed by the CALLER from state captured BEFORE the
    forward call (this function only ever sees post-call state) -- exactly
    the same requirement `kvcot.generation.replay._sync_layer_after_call`
    already has. `compute_kept_indices_lengths` is the helper for the
    latter.
    """
    from kvcot.generation.replay import _has_kv_cluster, _note_event_once, _sync_layer_after_call

    num_layers = len(model.model.layers)
    if call_kind == "prefill":
        for lp in model_provenance.layers.values():
            lp.append_new_tokens_prefill(fed_absolute_positions)
        absolute_position_after = len(fed_absolute_positions)
    elif call_kind == "decode":
        if len(fed_absolute_positions) != 1:
            raise ValueError(f"call_kind='decode' requires exactly one fed position, got {fed_absolute_positions!r}")
        for lp in model_provenance.layers.values():
            lp.append_new_token(fed_absolute_positions[0])
        absolute_position_after = fed_absolute_positions[0] + 1
    else:
        raise ValueError(f"call_kind must be 'prefill' or 'decode', got {call_kind!r}")

    per_layer_event_fired: list[bool] = []
    observations: dict[int, LayerStepObservation] = {}
    for layer_idx in range(num_layers):
        pre_event_map = model_provenance.layers[layer_idx].positions.clone()
        event_fired = _sync_layer_after_call(
            model, cache, layer_idx, model_provenance, kept_indices_lengths_before_call,
            expected_len_if_no_evict=expected_cache_lengths_if_no_eviction[layer_idx],
            absolute_position_after=absolute_position_after,
        )
        per_layer_event_fired.append(event_fired)
        kv_cluster = _has_kv_cluster(model, layer_idx)
        observed_kept = None
        window_size = None
        if event_fired and kv_cluster is not None:
            observed_kept = kv_cluster.kept_token_indices[-1].clone()
            window_size = kv_cluster.window_size
        observations[layer_idx] = LayerStepObservation(
            had_compaction=event_fired,
            cache_length_after=int(cache.key_cache[layer_idx].shape[-2]),
            pre_event_absolute_position_map=pre_event_map if event_fired else None,
            observed_kept_absolute_positions=observed_kept,
            window_size=window_size,
        )

    _note_event_once(per_layer_event_fired, compaction, absolute_position_after)
    return observations


def build_real_prefill_fn(device: str, call_recorder: Any | None = None) -> PrefillFn:
    """Real-model `PrefillFn` -- one opaque forward call over the complete
    prompt (B1B-R2 §6), never repeated one-token calls. `state` must be a
    `RealModelState` whose `cache` is freshly constructed
    (`kvcot.generation.state.reset_patched_state`). Position bookkeeping is
    now handled entirely by `advance_after_forward` (B1B-R3 §8) -- this
    function no longer needs to (and does not) touch
    `state.model_provenance` directly itself.

    ## Prefill logits, all positions (documented simplification vs.
    `kvcot.generation.decode.prefill`)

    `kvcot.generation.decode.prefill` (the primary/frozen pipeline's own
    prefill call) intentionally returns only the LAST position's logits --
    the primary pipeline never needs the earlier ones.
    `_prefill_forward_all_positions` issues the IDENTICAL forward call
    (same `input_ids`/`position_ids`/`cache_position`/`use_cache`
    construction, verified line-for-line against
    `kvcot.generation.decode.prefill`) but reads `out.logits[0, :, :]`
    (every position) instead of `out.logits[0, -1, :]` -- required because
    `PrefillFn`'s contract needs one logits tensor per prompt position
    (`kvcot.discovery.harness_types.PrefillStepResult`). This is additive
    (a different slice of the SAME forward call's output), never a second,
    differently-shaped forward call.

    ## Per-position compaction attribution during prefill (documented limit)

    A real transformer prefill call is opaque: R-KV's `kept_token_indices`
    bookkeeping only tells us whether at least one compaction fired
    somewhere across the whole prompt, never at which exact intra-prompt
    position. This adapter attributes any prefill-phase compaction to the
    LAST prompt position -- harmless for this harness's purposes, because
    `kvcot.discovery.pass1.eligible_event_ids` already excludes every
    prefill-phase event from target selection (B1B-R2 §6): no prefill-phase
    observation this adapter produces is ever used to build a branch.
    """

    def prefill_fn(state: RealModelState, prompt_token_ids) -> PrefillStepResult:
        import torch

        prompt_token_ids = list(prompt_token_ids)
        n = len(prompt_token_ids)
        num_layers = len(state.model.model.layers)

        kept_indices_lengths_before_call = compute_kept_indices_lengths(state.model)

        input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
        position_ids = torch.arange(0, n, device=device).unsqueeze(0)
        cache_position = torch.arange(0, n, device=device)
        if call_recorder is not None:
            call_recorder.observe("prefill", input_ids, position_ids, cache_position)
        with _pending_positions_scope(state, list(range(n)), "prefill"):
            with torch.no_grad():
                out = state.model(
                    input_ids=input_ids, position_ids=position_ids, past_key_values=state.cache,
                    use_cache=True, cache_position=cache_position,
                )
        all_position_logits = out.logits[0, :, :]  # (n, vocab) -- one row per prompt position

        observations = advance_after_forward(
            state.model, state.cache, state.model_provenance, state.compaction,
            fed_absolute_positions=list(range(n)),
            expected_cache_lengths_if_no_eviction={i: n for i in range(num_layers)},
            kept_indices_lengths_before_call=kept_indices_lengths_before_call,
            call_kind="prefill",
        )
        state.clear_pending()

        # Per-position observations: intra-prompt attribution is not
        # available from one opaque call (see docstring) -- every position
        # except the last reports had_compaction=False; the single real
        # eviction-detection pass runs once, after the whole call,
        # attributed to the LAST prompt position (returned by
        # advance_after_forward above).
        per_position_observations = [
            {i: LayerStepObservation(had_compaction=False, cache_length_after=int(state.cache.key_cache[i].shape[-2]))
             for i in range(num_layers)}
            for _ in range(n - 1)
        ]
        per_position_observations.append(observations)

        state.absolute_position = n
        return PrefillStepResult(
            new_state=state,
            per_position_logits=tuple(all_position_logits[i] for i in range(n)),
            per_position_layer_observations=tuple(per_position_observations),
        )

    return prefill_fn


def build_real_decode_one_fn(device: str, call_recorder: Any | None = None) -> DecodeOneFn:
    """Real-model `DecodeOneFn` -- one single-token forward call, matching
    `kvcot.generation.decode.decode_step`'s exact call shape (imported and
    reused directly, never reimplemented). Position bookkeeping is handled
    entirely by `advance_after_forward` (B1B-R3 §8)."""

    def decode_one_fn(state: RealModelState, token_id: int) -> NaturalStepResult:
        from kvcot.generation.decode import decode_step

        num_layers = len(state.model.model.layers)
        expected_len_if_no_evict = {
            i: int(state.cache.key_cache[i].shape[-2]) + 1 for i in range(num_layers)
        }
        kept_indices_lengths_before_call = compute_kept_indices_lengths(state.model)

        fed_position = state.absolute_position
        with _pending_positions_scope(state, [fed_position], "decode"):
            logits = decode_step(
                state.model, state.cache, token_id, state.absolute_position, device,
                None if call_recorder is None else call_recorder.observe,
            )
        state.absolute_position += 1

        observations = advance_after_forward(
            state.model, state.cache, state.model_provenance, state.compaction,
            fed_absolute_positions=[fed_position],
            expected_cache_lengths_if_no_eviction=expected_len_if_no_evict,
            kept_indices_lengths_before_call=kept_indices_lengths_before_call,
            call_kind="decode",
        )
        state.clear_pending()
        return NaturalStepResult(next_token_logits=logits, new_state=state, layer_observations=observations)

    return decode_one_fn


def build_real_snapshot_fn() -> SnapshotFn:
    """Real-model `SnapshotFn` -- delegates entirely to
    `kvcot.generation.replay.capture_snapshot`, the SAME complete-state
    capture the primary pipeline already uses (never a second,
    independently-written snapshot constructor)."""

    def snapshot_fn(state: RealModelState):
        from kvcot.generation.replay import capture_snapshot

        return capture_snapshot(
            state.model, state.cache, state.model_provenance, state.compaction, state.absolute_position
        )

    return snapshot_fn


# --------------------------------------------------------------------------
# B1B-R3 §9: restore-once branch evaluation
# --------------------------------------------------------------------------


def restore_compaction_tracker_from_snapshot(snapshot: Any) -> "CompactionTracker":
    """B1B-R4 §13 repair: reconstruct a branch's `CompactionTracker` from
    the snapshot it is restoring FROM, instead of `CompactionTracker()` (an
    always-empty tracker that silently discarded every compaction event the
    natural run recorded before this snapshot's absolute position). Uses
    `kvcot.generation.state.ModelStateSnapshot`'s own two fields
    (`compaction_event_steps`, `tokens_since_last_compaction`) -- never a
    second, independently-recomputed history.

    `last_event_absolute_position` is derived as `snapshot.absolute_position
    - snapshot.tokens_since_last_compaction`, matching
    `CompactionTracker.tokens_since_last`'s own definition exactly (so this
    is a true inverse, not an approximation) -- in the never-any-event case
    (`compaction_event_steps == []`), `tokens_since_last_compaction` at
    capture time equals `absolute_position - 0` (the class's own zero
    default), so this derivation naturally recovers `0`, the same initial
    value a fresh `CompactionTracker()` would report -- consistent with
    `CompactionTracker`'s defined initial semantics, not a special case.
    """
    from kvcot.generation.replay import CompactionTracker

    return CompactionTracker(
        event_steps=list(snapshot.compaction_event_steps),
        last_event_absolute_position=snapshot.absolute_position - snapshot.tokens_since_last_compaction,
    )


@dataclass
class _LiveBranchState:
    """A branch's already-restored, in-progress live state -- returned by
    the restore-once step function after its first call, and passed back in
    unchanged (mutated in place) on every subsequent call within the same
    branch. Never constructed by any caller outside this module; a caller
    that wants a NEW branch always starts from a `ModelStateSnapshot`.

    Carries the same pending-position fields/methods as `RealModelState`
    (B1B-R4.1 §4) purely for exception-safety uniformity across every real
    forward-call site in this module -- no capture instrumentation reads a
    branch's projection today (`capture_update_kv` is Pass-2-only), but a
    branch's own `advance_after_forward` call still needs the identical
    register-before/clear-after-or-on-exception discipline, so this class
    does not get a second, differently-shaped lifecycle."""

    cache: Any
    model_provenance: Any
    compaction: Any
    absolute_position: int
    pending_fed_absolute_positions: list[int] | None = field(default=None)
    pending_call_kind: Literal["prefill", "decode"] | None = field(default=None)

    def register_pending_fed_positions(self, positions: list[int], call_kind: Literal["prefill", "decode"]) -> None:
        self.pending_fed_absolute_positions = list(positions)
        self.pending_call_kind = call_kind

    def clear_pending(self) -> None:
        self.pending_fed_absolute_positions = None
        self.pending_call_kind = None

    def projected_pre_event_position_map(self, layer_index: int) -> "Any":
        return _project_pre_event_position_map(self, layer_index)


def build_real_branch_step_fn_restore_once(
    model: Any,
    device: str,
    call_recorder: Any | None = None,
    *,
    consume_owned_snapshot: bool = False,
):
    """Real `BranchStepFn` (B1B-R3 §9 repair): restores a complete
    `ModelStateSnapshot` into a FRESH cache EXACTLY ONCE per branch -- on
    the first call, when `state` is the branch's initial
    `ModelStateSnapshot` -- and reuses the already-restored live cache/
    provenance/compaction/position for every subsequent call in the same
    branch, when `state` is the `_LiveBranchState` this function's own
    previous call returned. `kvcot.discovery.branch_eval.evaluate_branch`'s
    existing `(state, token_id) -> (logits, new_state)` contract is
    unchanged -- this closure is the only thing that changed, not the
    generic evaluator it plugs into.
    """
    from kvcot.generation.decode import decode_step
    from kvcot.generation.replay import restore_snapshot
    from kvcot.generation.state import ModelStateSnapshot

    def branch_step_fn(state: Any, token_id: int):
        from transformers.cache_utils import DynamicCache

        if isinstance(state, ModelStateSnapshot):
            cache = DynamicCache()
            restored_provenance = restore_snapshot(
                model, cache, state, consume_owned_snapshot=consume_owned_snapshot
            )
            live = _LiveBranchState(
                cache=cache, model_provenance=restored_provenance,
                compaction=restore_compaction_tracker_from_snapshot(state),
                absolute_position=state.absolute_position,
            )
        elif isinstance(state, _LiveBranchState):
            live = state
        else:
            raise TypeError(f"branch_step_fn expects a ModelStateSnapshot or _LiveBranchState, got {type(state)}")

        num_layers = len(model.model.layers)
        expected_len_if_no_evict = {i: int(live.cache.key_cache[i].shape[-2]) + 1 for i in range(num_layers)}
        kept_indices_lengths_before_call = compute_kept_indices_lengths(model)

        fed_position = live.absolute_position
        with _pending_positions_scope(live, [fed_position], "decode"):
            logits = decode_step(
                model, live.cache, token_id, live.absolute_position, device,
                None if call_recorder is None else call_recorder.observe,
            )
        live.absolute_position += 1

        advance_after_forward(
            model, live.cache, live.model_provenance, live.compaction,
            fed_absolute_positions=[fed_position],
            expected_cache_lengths_if_no_eviction=expected_len_if_no_evict,
            kept_indices_lengths_before_call=kept_indices_lengths_before_call,
            call_kind="decode",
        )
        live.clear_pending()
        return logits, live

    return branch_step_fn


def evaluate_branch_from_snapshot(
    model: Any,
    device: str,
    initial_snapshot: Any,
    bridge_token_id: int,
    reference_token_ids: list[int],
):
    """B1B-R3 §9's named entry point: restore `initial_snapshot` exactly
    once, feed the one unscored bridge token, then sequentially feed the
    teacher-forced reference tokens, scoring exactly `len(reference_token_ids)`
    targets -- delegates entirely to `kvcot.discovery.branch_eval
    .evaluate_branch` (never a second, independently-written scoring loop),
    using `build_real_branch_step_fn_restore_once` as the injected
    `step_fn` so no snapshot restoration happens between tokens. Branch
    state (the live cache/provenance) is released for garbage collection
    once this function returns -- nothing here retains a reference beyond
    the call.
    """
    from kvcot.discovery.branch_eval import evaluate_branch

    step_fn = build_real_branch_step_fn_restore_once(model, device)
    return evaluate_branch(step_fn, initial_snapshot, bridge_token_id, reference_token_ids)
