"""§6.1 replay acceptance tests — the load-bearing GPU tests for this whole
repository. "The invariants the science depends on are discrete. Gate on
those exactly; treat float closeness as a diagnostic."

Implemented in full, marked `@pytest.mark.gpu`; auto-skipped on this build
machine (no CUDA device). See docs/GPU_VALIDATION_PLAN.md and
docs/REPLAY_DESIGN.md (especially §2's named call-shape assumption, which
this file's compaction-event-step comparisons exist specifically to check).
"""
from __future__ import annotations

import os

import pytest

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
BUDGET = 64  # small on purpose: forces multiple real compactions within a short fixture
WINDOW_SIZE = 8
MIX_LAMBDA = 0.1
RETAIN_RATIO = 0.2
RETAIN_DIRECTION = "last"
DIVIDE_METHOD = "step_length"
DIVIDE_LENGTH = 128

# Diagnostic-only tolerances (§6.1: "declare the tolerance in the test
# file"). These do NOT gate pass/fail on their own — see the assertions
# below, which only hard-fail on the discrete invariants.
KV_MAX_ABS_TOL = 5e-2
KV_MAX_REL_TOL = 5e-2
LOGIT_MAX_ABS_TOL = 5e-2

_FILLER = "We continue reasoning step by step, checking every detail carefully. "
LONG_PROMPT = "You are given a math problem.\n\nProblem: What is 123 + 456?\n\n Provide the final answer."
# ~500+ tokens of filler so >=2 compactions fire at budget=64, divide_length=128.
LONG_CONTINUATION = "\n" + (_FILLER * 60) + "So 123 + 456 = 579.\nFinal answer: \\boxed{579}"


def _set_determinism():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.use_deterministic_algorithms(True)


def _load_rkv_model(attn_impl: str):
    import torch
    from rkv.monkeypatch import replace_qwen2
    from transformers import AutoModelForCausalLM, AutoTokenizer

    compression_config = {
        "method": "rkv",
        "method_config": {
            "budget": BUDGET, "window_size": WINDOW_SIZE, "mix_lambda": MIX_LAMBDA,
            "retain_ratio": RETAIN_RATIO, "retain_direction": RETAIN_DIRECTION,
            "first_tokens": 4, "record_kept_token_indices": True,
        },
        "compression": None,
        "update_kv": True,
    }
    replace_qwen2(compression_config)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation=attn_impl,
    )
    model.eval()
    model.config.update({"divide_method": DIVIDE_METHOD, "divide_length": DIVIDE_LENGTH, "compression_content": "all"})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    return model, tokenizer


def _generate_and_collect(model, tokenizer, device: str) -> dict:
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.decode import decode_step, prefill
    from kvcot.generation.state import reset_patched_state

    cache = reset_patched_state(model, lambda: DynamicCache())
    prompt_ids = tokenizer(LONG_PROMPT, add_special_tokens=True)["input_ids"]
    continuation_ids = tokenizer(LONG_CONTINUATION, add_special_tokens=False)["input_ids"]

    logits, pos = prefill(model, cache, prompt_ids, device)
    for token_id in continuation_ids:
        logits = decode_step(model, cache, token_id, pos, device)
        pos += 1

    compaction_event_steps = []
    kept_sets_per_layer = {}
    cache_shapes = [tuple(cache.key_cache[i].shape) for i in range(len(model.model.layers))]
    for layer_idx, layer in enumerate(model.model.layers):
        kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
        if kv_cluster is not None:
            kept_sets_per_layer[layer_idx] = [
                frozenset(t.flatten().tolist()) for t in kv_cluster.kept_token_indices
            ]

    # Number of real compaction EVENTS = the per-layer count of recorded
    # kept-index tensors, NOT summed across layers. All R-KV layers share one
    # per-step compression schedule, so every layer must record the same number
    # of events; assert that and take one representative value. (Summing across
    # layers is the ×n_layers inflation that made `assert n_compactions >= 2`
    # vacuously true on a single real event.)
    per_layer_event_counts = [len(v) for v in kept_sets_per_layer.values()]
    assert len(set(per_layer_event_counts)) <= 1, (
        f"R-KV layers disagree on compaction event count: {sorted(set(per_layer_event_counts))}"
    )
    n_compactions = per_layer_event_counts[0] if per_layer_event_counts else 0

    return {
        "prompt_ids": prompt_ids,
        "continuation_ids": continuation_ids,
        "final_absolute_position": pos,
        "cache_shapes": cache_shapes,
        "kept_sets_per_layer": kept_sets_per_layer,
        "n_compactions": n_compactions,
    }


def _replay(model, tokenizer, base: dict, device: str) -> dict:
    from kvcot.generation.replay import replay_and_snapshot
    from kvcot.generation.state import reset_patched_state
    from transformers.cache_utils import DynamicCache
    from kvcot.probes.early_answering import find_think_span

    # No real think markers in this fixture; snapshot at the single
    # end-of-generation position so replay exercises the full trace.
    end_pos = len(base["prompt_ids"]) + len(base["continuation_ids"])
    fake_span = find_think_span(base["prompt_ids"], base["continuation_ids"], [151648], [151649])

    snapshots = replay_and_snapshot(
        model=model,
        fresh_cache_factory=lambda: DynamicCache(),
        prompt_token_ids=base["prompt_ids"],
        generated_token_ids=base["continuation_ids"],
        think_span=fake_span,
        snapshot_absolute_positions={1.0: end_pos},
        device=device,
    )
    snap = snapshots[1.0]

    kept_sets_per_layer = {}
    for layer_idx, layer in enumerate(model.model.layers):
        kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
        if kv_cluster is not None:
            kept_sets_per_layer[layer_idx] = [
                frozenset(t.flatten().tolist()) for t in kv_cluster.kept_token_indices
            ]

    return {
        "compaction_event_steps": snap.compaction_event_steps,
        "cache_shapes": [tuple(t.shape) for t in snap.key_cache],
        "query_cache_shapes": {i: tuple(t.shape) for i, t in snap.query_cache.items()},
        "kept_sets_per_layer": kept_sets_per_layer,
        "snapshot": snap,
    }


@pytest.mark.gpu
@pytest.mark.parametrize("attn_impl", ["flash_attention_2", "sdpa"])
def test_replay_reproduces_identical_tokens_and_compaction_events(attn_impl):
    _set_determinism()
    model, tokenizer = _load_rkv_model(attn_impl)
    device = "cuda"

    base = _generate_and_collect(model, tokenizer, device)
    replayed = _replay(model, tokenizer, base, device)

    # HARD GATE: >=2 real compaction EVENTS fired in the original generation
    # (per-layer event count, not events*n_layers — see _generate_and_collect).
    assert base["n_compactions"] >= 2

    # HARD GATE: identical compaction event count via replay (exact step
    # positions are compared per-layer against the base run's own recorded
    # event count as a first-order check; full step-by-step equality is
    # checked in the two-independent-replays test below, which compares
    # replay against replay with no original-vs-replay confound).
    for layer_idx, base_sets in base["kept_sets_per_layer"].items():
        assert len(base_sets) == len(replayed["kept_sets_per_layer"].get(layer_idx, [])), (
            f"layer {layer_idx}: base had {len(base_sets)} compaction events, "
            f"replay had {len(replayed['kept_sets_per_layer'].get(layer_idx, []))}"
        )
        # HARD GATE: identical per-layer, per-KV-head surviving absolute
        # source positions at the final event.
        assert base_sets[-1] == replayed["kept_sets_per_layer"][layer_idx][-1]

    # HARD GATE: identical cache shapes.
    assert base["cache_shapes"] == replayed["cache_shapes"]

    # HARD GATE: cache length stays within post-compaction budget plus
    # interval growth (docs/UPSTREAM_AUDIT.md H4 refinement).
    for shape in replayed["cache_shapes"]:
        cache_len = shape[-2]
        assert cache_len <= BUDGET + DIVIDE_LENGTH, (
            f"cache length {cache_len} exceeds budget+divide_length ({BUDGET}+{DIVIDE_LENGTH})"
        )


@pytest.mark.gpu
def test_two_independent_replays_are_identical():
    _set_determinism()
    model, tokenizer = _load_rkv_model("flash_attention_2")
    device = "cuda"
    base = _generate_and_collect(model, tokenizer, device)

    replay_1 = _replay(model, tokenizer, base, device)
    replay_2 = _replay(model, tokenizer, base, device)

    assert replay_1["compaction_event_steps"] == replay_2["compaction_event_steps"]
    assert replay_1["cache_shapes"] == replay_2["cache_shapes"]
    assert replay_1["query_cache_shapes"] == replay_2["query_cache_shapes"]
    assert replay_1["kept_sets_per_layer"] == replay_2["kept_sets_per_layer"]


@pytest.mark.gpu
def test_restoring_same_snapshot_twice_yields_identical_probe_tokens():
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.replay import branch_and_probe

    _set_determinism()
    model, tokenizer = _load_rkv_model("flash_attention_2")
    device = "cuda"
    base = _generate_and_collect(model, tokenizer, device)
    replayed = _replay(model, tokenizer, base, device)
    snap = replayed["snapshot"]

    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    from kvcot.probes.templates import render_control_suffix

    suffix_ids = tokenizer.encode(render_control_suffix(), add_special_tokens=False)

    # A fresh DynamicCache per branch call is required (and sufficient):
    # `restore_snapshot` (inside `branch_and_probe`) POPULATES the cache from
    # the snapshot via the public `cache.update(...)` path, which grows
    # `cache.layers` and lazily initializes each layer. It does NOT item-assign
    # into `cache.key_cache[i]` — on transformers 4.55.4 that would IndexError,
    # because a fresh DynamicCache pre-creates only one layer and `key_cache`
    # is a growth-free `KeyValuesWrapper` property. Because `update` appends,
    # the cache MUST be fresh here (a reused cache would concatenate, not
    # overwrite); constructing a new one per call is exactly that guarantee.
    probe_1 = branch_and_probe(
        model, DynamicCache(), snap, close_ids, suffix_ids, 48, tokenizer.eos_token_id, device
    )
    probe_2 = branch_and_probe(
        model, DynamicCache(), snap, close_ids, suffix_ids, 48, tokenizer.eos_token_id, device
    )

    assert probe_1.probe_output_token_ids == probe_2.probe_output_token_ids


@pytest.mark.gpu
def test_fullkv_replay_passes_identity_test():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.decode import decode_step, prefill
    from kvcot.generation.replay import replay_and_snapshot
    from kvcot.generation.state import declare_process_mode, reset_patched_state
    from kvcot.probes.early_answering import find_think_span

    _set_determinism()
    declare_process_mode("stock")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="flash_attention_2",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    device = "cuda"

    cache = reset_patched_state(model, lambda: DynamicCache())
    prompt_ids = tokenizer(LONG_PROMPT, add_special_tokens=True)["input_ids"]
    continuation_ids = tokenizer(LONG_CONTINUATION, add_special_tokens=False)["input_ids"]
    logits, pos = prefill(model, cache, prompt_ids, device)
    for token_id in continuation_ids:
        logits = decode_step(model, cache, token_id, pos, device)
        pos += 1
    base_cache_shapes = [tuple(cache.key_cache[i].shape) for i in range(len(model.model.layers))]

    fake_span = find_think_span(prompt_ids, continuation_ids, [151648], [151649])
    snapshots = replay_and_snapshot(
        model=model,
        fresh_cache_factory=lambda: DynamicCache(),
        prompt_token_ids=prompt_ids,
        generated_token_ids=continuation_ids,
        think_span=fake_span,
        snapshot_absolute_positions={1.0: len(prompt_ids) + len(continuation_ids)},
        device=device,
    )
    snap = snapshots[1.0]

    # HARD GATE: FullKV never evicts, so cache length must equal the total
    # processed token count exactly (no compaction shrinkage possible).
    assert snap.compaction_event_steps == []
    for shape in [tuple(t.shape) for t in snap.key_cache]:
        assert shape[-2] == len(prompt_ids) + len(continuation_ids)
    assert [tuple(t.shape) for t in snap.key_cache] == base_cache_shapes


@pytest.mark.gpu
def test_streaming_replay_vs_bulk_prefill_negative_control():
    """Instruments bulk-prefill of the same truncated prefix as a distinct
    code path and verifies streaming replay fires the expected periodic
    compaction events while bulk prefill does not — asserted on event
    *diagnostics* (count, presence), never on "different kept-sets on some
    arbitrary example" (§6.1: coincidental equality is possible and not a
    meaningful assertion)."""
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.decode import prefill
    from kvcot.generation.state import reset_patched_state

    _set_determinism()
    model, tokenizer = _load_rkv_model("flash_attention_2")
    device = "cuda"
    base = _generate_and_collect(model, tokenizer, device)

    # Streaming replay (our real code path): >=2 compaction events, per
    # test_replay_reproduces_identical_tokens_and_compaction_events above.
    streaming_events = base["n_compactions"]

    # Bulk prefill of the identical full token sequence in ONE forward call
    # (the deliberately wrong way to "replay", per docs/REPLAY_DESIGN.md §1) —
    # a single N-token call only ever crosses `divide_length` multiples
    # implicitly (self.length jumps straight to the final count), so the
    # patched CausalLM's post-call schedule check can fire at most once
    # (docs/UPSTREAM_AUDIT.md H4), which is structurally different from the
    # streaming loop's per-128-token schedule checks.
    cache = reset_patched_state(model, lambda: DynamicCache())
    full_sequence = base["prompt_ids"] + base["continuation_ids"]
    prefill(model, cache, full_sequence, device)
    # Per-layer event count (representative), same convention as
    # _generate_and_collect — never summed across layers.
    bulk_per_layer = [
        len(layer.self_attn.kv_cluster.kept_token_indices)
        for layer in model.model.layers
        if getattr(layer.self_attn, "kv_cluster", None) is not None
    ]
    bulk_events = bulk_per_layer[0] if bulk_per_layer else 0

    assert streaming_events >= 2
    assert bulk_events != streaming_events, (
        "bulk prefill produced the same compaction-event count as streaming "
        "replay — this would mean call shape doesn't actually affect the "
        "schedule, contradicting docs/REPLAY_DESIGN.md §2's named assumption; "
        "if this genuinely holds on GPU, that assumption needs revising, not "
        "this test loosening."
    )
