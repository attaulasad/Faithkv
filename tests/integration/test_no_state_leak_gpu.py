"""§3.1: cross-example mutable-state leakage test.

Running example B before example A must not change A's token sequence,
logits, compaction event steps, or kept-position hashes, given
`kvcot.generation.state.reset_patched_state` is called between every
independent generation. Tests both orders within a single process — unlike
the stock-vs-patched parity test, multiple examples under the *same*
condition legitimately share a process (only stock vs. patched must not,
§3.2); this test's whole point is that the shared process is safe as long
as the reset function actually resets everything it claims to.

Implemented in full, marked `@pytest.mark.gpu`; auto-skipped here (no CUDA
device on this build machine). See docs/GPU_VALIDATION_PLAN.md.
"""
from __future__ import annotations

import hashlib

import pytest

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
BUDGET = 64  # small on purpose: forces >=1 real compaction within each fixture
DIVIDE_LENGTH = 128

_FILLER = "We continue reasoning step by step, checking every detail carefully. "

EXAMPLE_A_PROMPT = "You are given a math problem.\n\nProblem: What is 17 + 25?\n\n Provide the final answer."
EXAMPLE_A_CONTINUATION = "\n" + (_FILLER * 20) + "So 17 + 25 = 42.\nFinal answer: \\boxed{42}"

EXAMPLE_B_PROMPT = "You are given a math problem.\n\nProblem: What is 9 times 8?\n\n Provide the final answer."
EXAMPLE_B_CONTINUATION = "\n" + (_FILLER * 20) + "So 9 * 8 = 72.\nFinal answer: \\boxed{72}"


def _load_patched_model():
    import torch
    from rkv.monkeypatch import replace_qwen2
    from transformers import AutoModelForCausalLM, AutoTokenizer

    compression_config = {
        "method": "rkv",
        "method_config": {
            "budget": BUDGET, "window_size": 8, "mix_lambda": 0.1,
            "retain_ratio": 0.2, "retain_direction": "last", "first_tokens": 4,
            "record_kept_token_indices": True,
        },
        "compression": None,
        "update_kv": True,
    }
    replace_qwen2(compression_config)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="flash_attention_2",
    )
    model.eval()
    model.config.update({"divide_method": "step_length", "divide_length": DIVIDE_LENGTH, "compression_content": "all"})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    model.newline_token_ids = [tokenizer.encode("\n")[-1]]
    model.after_think_token_ids = [tokenizer.encode("</think>")[-1] if "</think>" in tokenizer.get_vocab() else tokenizer.eos_token_id]
    return model, tokenizer


def _run_example(model, tokenizer, prompt_text: str, continuation_text: str, device: str) -> dict:
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.decode import decode_step, prefill
    from kvcot.generation.state import reset_patched_state

    cache = reset_patched_state(model, lambda: DynamicCache())
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    continuation_ids = tokenizer(continuation_text, add_special_tokens=False)["input_ids"]

    logits, pos = prefill(model, cache, prompt_ids, device)
    steps = [(float(logits.max().item()), int(logits.argmax().item()))]
    for token_id in continuation_ids:
        logits = decode_step(model, cache, token_id, pos, device)
        pos += 1
        steps.append((float(logits.max().item()), int(logits.argmax().item())))

    compaction_events = 0
    kept_position_hashes = []
    for layer in model.model.layers:
        kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
        if kv_cluster is not None:
            compaction_events += len(kv_cluster.kept_token_indices)
            for kept in kv_cluster.kept_token_indices:
                kept_position_hashes.append(hashlib.sha256(kept.numpy().tobytes()).hexdigest())

    return {
        "prompt_ids": prompt_ids,
        "steps": steps,
        "compaction_events": compaction_events,
        "kept_position_hashes": kept_position_hashes,
    }


@pytest.mark.gpu
def test_example_results_independent_of_run_order():
    model, tokenizer = _load_patched_model()
    device = "cuda"

    # Order 1: A then B
    result_a_first = _run_example(model, tokenizer, EXAMPLE_A_PROMPT, EXAMPLE_A_CONTINUATION, device)
    result_b_after_a = _run_example(model, tokenizer, EXAMPLE_B_PROMPT, EXAMPLE_B_CONTINUATION, device)

    # Order 2: B then A
    result_b_first = _run_example(model, tokenizer, EXAMPLE_B_PROMPT, EXAMPLE_B_CONTINUATION, device)
    result_a_after_b = _run_example(model, tokenizer, EXAMPLE_A_PROMPT, EXAMPLE_A_CONTINUATION, device)

    # Sanity: both fixtures actually triggered real compaction, otherwise
    # this test would pass vacuously (nothing to leak).
    assert result_a_first["compaction_events"] >= 1
    assert result_b_first["compaction_events"] >= 1

    # HARD GATE: A's results must be identical regardless of whether B ran
    # before it.
    assert result_a_first["prompt_ids"] == result_a_after_b["prompt_ids"]
    assert result_a_first["steps"] == result_a_after_b["steps"]
    assert result_a_first["compaction_events"] == result_a_after_b["compaction_events"]
    assert result_a_first["kept_position_hashes"] == result_a_after_b["kept_position_hashes"]

    # HARD GATE: same for B.
    assert result_b_first["prompt_ids"] == result_b_after_a["prompt_ids"]
    assert result_b_first["steps"] == result_b_after_a["steps"]
    assert result_b_first["compaction_events"] == result_b_after_a["compaction_events"]
    assert result_b_first["kept_position_hashes"] == result_b_after_a["kept_position_hashes"]
