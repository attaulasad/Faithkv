"""Mandatory GPU gate for protocol v3 (CHANGELOG.md 2026-07-17): real-GPU
verification of `kvcot.generation.replay.branch_and_probe`'s
`probe_cache_mode="frozen_at_cut"` — the fix for protocol v2's actual
observed failure mode (`rkv_evicted_during_answer_probe` on 5 of 10 shared
examples in the committed `results/decisions/early_gap_v2_b128_fixed_trace.
json`). Every prior GPU test in this repository exercises `branch_and_probe`
only in its default `"native"` mode; this file specifically constructs a
scenario where R-KV's own schedule WOULD compact again while the probe
writes its answer, and confirms `"frozen_at_cut"` prevents that by
construction, not merely by post-hoc detection.

BUDGET/DIVIDE_LENGTH here are deliberately much smaller than the frozen
production values (`configs/lock.yaml`: no analogous budget is frozen there
since budget is swept; `divide_length=128`) — same rationale as
tests/integration/test_replay_gpu.py's `BUDGET = 64`: small values force
the schedule to cross a compaction checkpoint WITHIN the ~64-token probe
window, so the scenario this test exists to check is actually exercised
deterministically rather than left to chance.

Process isolation follows tests/integration/test_replay_gpu.py's pattern —
see that file's module docstring for the full rationale.

Implemented in full, marked `@pytest.mark.gpu`; auto-skipped on this build
machine (no CUDA device).
"""
from __future__ import annotations

import multiprocessing as mp
import os

import pytest

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
BUDGET = 32  # small on purpose -- see module docstring
WINDOW_SIZE = 8
MIX_LAMBDA = 0.1
RETAIN_RATIO = 0.2
RETAIN_DIRECTION = "last"
DIVIDE_METHOD = "step_length"
DIVIDE_LENGTH = 8  # small on purpose -- guarantees a schedule check within the probe's own token window
PROBE_MAX_NEW_TOKENS = 64
_SUBPROCESS_TIMEOUT_SECONDS = 1200

_FILLER = "We continue reasoning step by step, checking every detail carefully. "
LONG_PROMPT = "You are given a math problem.\n\nProblem: What is 123 + 456?\n\n Provide the final answer."
LONG_CONTINUATION = "\n" + (_FILLER * 60) + "So 123 + 456 = 579.\nFinal answer: \\boxed{579}"


def _set_determinism():  # pragma: no cover - runs in a spawned process
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.use_deterministic_algorithms(True)


def _load_rkv_model(attn_impl: str):  # pragma: no cover - runs in a spawned process
    import torch
    from rkv.monkeypatch import replace_qwen2
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvcot.generation.state import declare_process_mode

    declare_process_mode("patched")

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


def _run_in_subprocess(target, args: tuple = (), timeout: int = _SUBPROCESS_TIMEOUT_SECONDS) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))
    proc.start()
    result = queue.get(timeout=timeout)
    proc.join(timeout=timeout)
    assert proc.exitcode == 0, f"subprocess for {target.__name__} exited with {proc.exitcode}"
    if isinstance(result, dict) and result.get("__error__"):
        raise RuntimeError(f"subprocess for {target.__name__} raised: {result['__error__']}")
    return result


def _worker_frozen_vs_native(result_queue: "mp.Queue") -> None:  # pragma: no cover - runs in a spawned process
    try:
        from transformers.cache_utils import DynamicCache

        from kvcot.generation.replay import branch_and_probe, replay_and_snapshot
        from kvcot.probes.early_answering import ThinkSpanResult, absolute_cut_position
        from kvcot.probes.templates import render_control_suffix

        _set_determinism()
        model, tokenizer = _load_rkv_model("flash_attention_2")
        device = "cuda"

        prompt_ids = tokenizer(LONG_PROMPT, add_special_tokens=True)["input_ids"]
        continuation_ids = tokenizer(LONG_CONTINUATION, add_special_tokens=False)["input_ids"]

        # No real <think>/</think> markers in this synthetic fixture -- the
        # whole continuation is treated as the think span (see
        # test_rkv_schedule_prediction_gpu.py for the same convention).
        # Snapshot at the midpoint, well after the cache has already been
        # compacted at least once, so kv_cluster.evicted_token_num > 0
        # BEFORE the probe ever runs.
        span = ThinkSpanResult(
            think_start_index=0, think_end_index=len(continuation_ids),
            think_parse_status="generation_prompt_preopened_ok",
            generation_prompt_preopened_think=True,
        )
        snapshot_pos = len(prompt_ids) + absolute_cut_position(span, 0.5)
        snapshots = replay_and_snapshot(
            model=model, fresh_cache_factory=lambda: DynamicCache(),
            prompt_token_ids=prompt_ids, generated_token_ids=continuation_ids,
            think_span=span, snapshot_absolute_positions={0.5: snapshot_pos}, device=device,
        )
        snap = snapshots[0.5]
        start_cache_lengths = [int(t.shape[-2]) for t in snap.key_cache]
        start_evicted_counts = [bk.get("evicted_token_num") for bk in snap.kv_cluster_bookkeeping_per_layer]

        close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        suffix_ids = tokenizer.encode(render_control_suffix(), add_special_tokens=False)

        # --- native mode: R-KV's own schedule may compact again while this
        # probe writes its answer -- protocol v2's actual observed failure
        # mode (rkv_evicted_during_answer_probe, CHANGELOG.md 2026-07-17).
        native_result = branch_and_probe(
            model, DynamicCache(), snap, close_ids, suffix_ids, PROBE_MAX_NEW_TOKENS,
            tokenizer.eos_token_id, device, probe_cache_mode="native",
        )
        native_tokens_fed = native_result.final_absolute_position - snap.absolute_position
        native_expected_no_eviction = [s + native_tokens_fed for s in start_cache_lengths]
        native_evicted_counts_after = [
            layer.self_attn.kv_cluster.evicted_token_num
            for layer in model.model.layers
            if getattr(layer.self_attn, "kv_cluster", None) is not None
        ]

        # --- frozen_at_cut from the SAME snapshot: restore_snapshot (inside
        # branch_and_probe) resets live model/cache state fresh each call,
        # so this branch starts from IDENTICAL conditions to the native one
        # above, not from wherever the native branch left off. If frozen_
        # at_cut's own internal hard assertion were violated, this call
        # would raise RuntimeError and the subprocess would report
        # __error__ instead of reaching the lines below.
        frozen_result = branch_and_probe(
            model, DynamicCache(), snap, close_ids, suffix_ids, PROBE_MAX_NEW_TOKENS,
            tokenizer.eos_token_id, device, probe_cache_mode="frozen_at_cut",
        )
        frozen_tokens_fed = frozen_result.final_absolute_position - snap.absolute_position
        frozen_expected_no_eviction = [s + frozen_tokens_fed for s in start_cache_lengths]
        frozen_evicted_counts_after = [
            layer.self_attn.kv_cluster.evicted_token_num
            for layer in model.model.layers
            if getattr(layer.self_attn, "kv_cluster", None) is not None
        ]

        # Restoring the same snapshot twice under frozen_at_cut must still
        # yield identical output tokens (§6.1 hard gate, re-verified under
        # the new mode -- teacher-forced feed + greedy decode, no
        # resampling anywhere, regardless of probe_cache_mode).
        frozen_result_2 = branch_and_probe(
            model, DynamicCache(), snap, close_ids, suffix_ids, PROBE_MAX_NEW_TOKENS,
            tokenizer.eos_token_id, device, probe_cache_mode="frozen_at_cut",
        )

        result_queue.put(
            {
                "start_cache_lengths": start_cache_lengths,
                "start_evicted_counts": start_evicted_counts,
                "native_tokens_fed": native_tokens_fed,
                "native_final_cache_lengths": native_result.final_cache_lengths_per_layer,
                "native_expected_no_eviction": native_expected_no_eviction,
                "native_evicted_counts_after": native_evicted_counts_after,
                "native_probe_cache_mode_field": native_result.probe_cache_mode,
                "frozen_tokens_fed": frozen_tokens_fed,
                "frozen_final_cache_lengths": frozen_result.final_cache_lengths_per_layer,
                "frozen_expected_no_eviction": frozen_expected_no_eviction,
                "frozen_evicted_counts_after": frozen_evicted_counts_after,
                "frozen_probe_cache_mode_field": frozen_result.probe_cache_mode,
                "frozen_output_tokens_1": frozen_result.probe_output_token_ids,
                "frozen_output_tokens_2": frozen_result_2.probe_output_token_ids,
            }
        )
    except Exception as e:  # pragma: no cover
        result_queue.put({"__error__": repr(e)})


@pytest.mark.gpu
def test_frozen_at_cut_prevents_eviction_that_native_mode_allows():
    result = _run_in_subprocess(_worker_frozen_vs_native)

    # HARD GATE: frozen_at_cut must never evict beyond start_length +
    # tokens_fed, on any layer. branch_and_probe's own internal assertion
    # would already have raised (and aborted this subprocess) if violated;
    # re-verify independently from the returned diagnostics too.
    assert result["frozen_final_cache_lengths"] == result["frozen_expected_no_eviction"]

    # HARD GATE: evicted_token_num must be UNCHANGED after a frozen_at_cut
    # probe, on every R-KV layer -- no new compaction event recorded.
    assert result["frozen_evicted_counts_after"] == result["start_evicted_counts"]

    # Diagnostic, not a hard gate on its own: this small-budget/small-
    # divide_length fixture is only a meaningful regression scenario if
    # NATIVE mode could actually evict further during the probe. If this
    # ever comes back False, the fixture no longer exercises the failure
    # mode frozen_at_cut exists to prevent -- adjust BUDGET/DIVIDE_LENGTH,
    # don't just delete this assertion.
    assert result["native_final_cache_lengths"] != result["native_expected_no_eviction"], (
        "expected native probe_cache_mode to evict further during the probe under this small "
        "budget/divide_length setup -- if this now passes cleanly, the test fixture no longer "
        "exercises the failure mode frozen_at_cut exists to prevent"
    )
    assert result["native_evicted_counts_after"] != result["start_evicted_counts"], (
        "expected evicted_token_num to grow under native mode during the probe -- same fixture "
        "concern as above"
    )

    assert result["frozen_probe_cache_mode_field"] == "frozen_at_cut"
    assert result["native_probe_cache_mode_field"] == "native"
    assert result["frozen_output_tokens_1"] == result["frozen_output_tokens_2"]
