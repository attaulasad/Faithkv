"""Mandatory GPU gate for protocol v3 (CHANGELOG.md 2026-07-17/2026-07-18):
`kvcot.analysis.rkv_schedule`'s CPU simulator was validated on this
CPU-only machine only against hand-derived cases and the audited mechanics
(`docs/UPSTREAM_AUDIT.md` H4) — never against a real GPU-measured cache
length, since no GPU exists here. This test compares the simulator's
predicted physical cache length against a REAL replay's measured cache
length at every one of the 9 frozen probe fractions, on every layer. A
one-token disagreement means the simulator does not actually reproduce
real R-KV behavior on this build/upstream-commit/transformers-version
combination and must not be trusted for outcome-blind trace selection
(`kvcot inspect-fixed-trace --write-selection`) until re-derived — per
`docs/GPU_VALIDATION_PLAN.md`'s "Protocol v3" section, this is required to
pass BEFORE any full v3 replay.

Process isolation follows the same pattern as tests/integration/
test_replay_gpu.py — see that file's module docstring for the full
rationale (stock-vs-patched process-global monkeypatch, §3.2/H1).

Implemented in full, marked `@pytest.mark.gpu`; auto-skipped on this build
machine (no CUDA device).
"""
from __future__ import annotations

import multiprocessing as mp
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
DIVIDE_LENGTH = 128  # the frozen production value (configs/lock.yaml) -- unlike BUDGET, not shrunk here
_SUBPROCESS_TIMEOUT_SECONDS = 1200

_FILLER = "We continue reasoning step by step, checking every detail carefully. "
LONG_PROMPT = "You are given a math problem.\n\nProblem: What is 123 + 456?\n\n Provide the final answer."
# ~500+ tokens of filler so several schedule/compaction events fire across
# the 9 snapshot fractions at budget=64, divide_length=128.
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


def _worker_schedule_prediction(result_queue: "mp.Queue") -> None:  # pragma: no cover - runs in a spawned process
    try:
        from transformers.cache_utils import DynamicCache

        from kvcot.config import PROBE_FRACTIONS_ALL
        from kvcot.generation.replay import replay_and_snapshot
        from kvcot.probes.early_answering import ThinkSpanResult, absolute_cut_position

        _set_determinism()
        model, tokenizer = _load_rkv_model("flash_attention_2")
        device = "cuda"

        prompt_ids = tokenizer(LONG_PROMPT, add_special_tokens=True)["input_ids"]
        continuation_ids = tokenizer(LONG_CONTINUATION, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_ids)

        # No real <think>/</think> markers in this synthetic fixture -- treat
        # the entire continuation as the think span (generation_prompt_
        # preopened_think=True, matching how a real trace whose </think>
        # generation prompt reopens the span is recorded) so all 9 frozen
        # probe fractions map to real, distinct absolute positions.
        span = ThinkSpanResult(
            think_start_index=0, think_end_index=len(continuation_ids),
            think_parse_status="generation_prompt_preopened_ok",
            generation_prompt_preopened_think=True,
        )
        cut_positions = {f: prompt_length + absolute_cut_position(span, f) for f in PROBE_FRACTIONS_ALL}

        snapshots = replay_and_snapshot(
            model=model, fresh_cache_factory=lambda: DynamicCache(),
            prompt_token_ids=prompt_ids, generated_token_ids=continuation_ids,
            think_span=span, snapshot_absolute_positions=cut_positions, device=device,
        )

        measured_cache_lengths_per_fraction = {
            f: [int(t.shape[-2]) for t in snapshots[f].key_cache] for f in PROBE_FRACTIONS_ALL
        }

        result_queue.put(
            {
                "prompt_length": prompt_length,
                "cut_positions": cut_positions,
                "measured_cache_lengths_per_fraction": measured_cache_lengths_per_fraction,
            }
        )
    except Exception as e:  # pragma: no cover
        result_queue.put({"__error__": repr(e)})


@pytest.mark.gpu
def test_simulator_matches_real_measured_cache_length_at_every_fraction():
    from kvcot.analysis.rkv_schedule import simulate_rkv_cache_lengths
    from kvcot.config import PROBE_FRACTIONS_ALL

    result = _run_in_subprocess(_worker_schedule_prediction)
    prompt_length = result["prompt_length"]
    cut_positions = result["cut_positions"]
    measured_by_fraction = result["measured_cache_lengths_per_fraction"]

    predicted_by_position = simulate_rkv_cache_lengths(
        prompt_length=prompt_length,
        target_absolute_positions=cut_positions.values(),
        budget=BUDGET,
        divide_length=DIVIDE_LENGTH,
    )

    mismatches = []
    for f in PROBE_FRACTIONS_ALL:
        pos = cut_positions[f]
        predicted_length = predicted_by_position[pos]
        for layer_idx, measured_length in enumerate(measured_by_fraction[f]):
            if measured_length != predicted_length:
                mismatches.append(
                    f"fraction={f} layer={layer_idx} absolute_position={pos}: "
                    f"predicted {predicted_length}, measured {measured_length}"
                )

    # HARD GATE: zero disagreement, at every fraction, on every layer. This
    # is the "one-token disagreement is a failure" gate
    # docs/GPU_VALIDATION_PLAN.md's Protocol v3 section requires before
    # trusting the simulator for real trace selection.
    assert not mismatches, (
        f"kvcot.analysis.rkv_schedule disagreed with real R-KV replay at "
        f"{len(mismatches)} (fraction, layer) points (prompt_length={prompt_length}, "
        f"budget={BUDGET}, divide_length={DIVIDE_LENGTH}):\n" + "\n".join(mismatches)
    )
