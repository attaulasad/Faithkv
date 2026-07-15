"""§3.2 stock-vs-patched no-op parity.

FullKV uses stock Transformers Qwen2; R-KV uses process-global class
monkeypatches (docs/UPSTREAM_AUDIT.md H1). "Never load stock and patched
models in the same Python process" is enforced here by literally running
each condition in its own OS process via `multiprocessing` with the
`spawn` start method (not `fork`, so neither process ever inherits an
already-imported, possibly-patched `transformers.models.qwen2` module from
the other) — mirroring how the real pipeline runs each condition as a
separate process (kvcot.generation.state.declare_process_mode is the
in-process guard for accidental same-process misuse; this test additionally
proves the two conditions really can be compared safely across processes).

Diagnostic condition `patched_noop`: R-KV patch enabled with a budget far
larger than the entire fixture, so `R1KV.update_kv`'s no-op path
(r1_kv.py:46-47, `kv_cache_len < self.budget`) always fires and compression
never actually triggers.

Everything in this file is implemented in full and marked
`@pytest.mark.gpu`; it is auto-skipped on this build machine
(tests/conftest.py) since there is no CUDA device here. See
docs/GPU_VALIDATION_PLAN.md for how to actually run it.
"""
from __future__ import annotations

import multiprocessing as mp

import pytest

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
FIXTURE_PROMPT_TEXT = "You are given a math problem.\n\nProblem: What is 2 + 2?\n\n Provide the final answer."
# A short, fixed teacher-forced continuation so both conditions process an
# identical sequence with no sampling involved (logits are compared, not
# sampled tokens).
FIXTURE_CONTINUATION_TEXT = "\nLet's think. 2 + 2 = 4.\nFinal answer: \\boxed{4}"
LOGIT_ABS_TOL = 1e-2
LOGIT_REL_TOL = 1e-2


def _run_condition(condition: str, result_queue: "mp.Queue") -> None:  # pragma: no cover - runs in a spawned process
    import torch
    from transformers import AutoTokenizer

    from kvcot.generation.decode import decode_step, prefill
    from kvcot.generation.state import declare_process_mode

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    prompt_ids = tokenizer(FIXTURE_PROMPT_TEXT, add_special_tokens=True)["input_ids"]
    continuation_ids = tokenizer(FIXTURE_CONTINUATION_TEXT, add_special_tokens=False)["input_ids"]

    if condition == "full":
        declare_process_mode("stock")
        from transformers import AutoModelForCausalLM
        from transformers.cache_utils import DynamicCache

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="flash_attention_2",
        )
        model.eval()
        cache = DynamicCache()
        compaction_count = 0
    else:
        declare_process_mode("patched")
        from rkv.monkeypatch import replace_qwen2
        from transformers import AutoModelForCausalLM
        from transformers.cache_utils import DynamicCache

        compression_config = {
            "method": "rkv",
            "method_config": {
                "budget": 1_000_000, "window_size": 8, "mix_lambda": 0.1,
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
        model.config.update({"divide_method": "step_length", "divide_length": 128, "compression_content": "all"})
        cache = DynamicCache()
        compaction_count = 0

    device = "cuda"
    logits, absolute_position = prefill(model, cache, prompt_ids, device)
    per_step_logits_summary = [(float(logits.max().item()), int(logits.argmax().item()))]
    fed_tokens = []
    for token_id in continuation_ids:
        fed_tokens.append(token_id)
        logits = decode_step(model, cache, token_id, absolute_position, device)
        absolute_position += 1
        per_step_logits_summary.append((float(logits.max().item()), int(logits.argmax().item())))

    if condition != "full":
        for layer in model.model.layers:
            kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
            if kv_cluster is not None:
                compaction_count += len(kv_cluster.kept_token_indices)

    result_queue.put(
        {
            "prompt_token_ids": prompt_ids,
            "fed_tokens": fed_tokens,
            "per_step_logits_summary": per_step_logits_summary,
            "compaction_count": compaction_count,
        }
    )


def _run_in_subprocess(condition: str) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run_condition, args=(condition, queue))
    proc.start()
    result = queue.get(timeout=600)
    proc.join(timeout=600)
    assert proc.exitcode == 0, f"subprocess for condition={condition} exited with {proc.exitcode}"
    return result


@pytest.mark.gpu
def test_stock_vs_patched_noop_parity():
    full_result = _run_in_subprocess("full")
    noop_result = _run_in_subprocess("patched_noop")

    assert full_result["prompt_token_ids"] == noop_result["prompt_token_ids"]
    assert full_result["fed_tokens"] == noop_result["fed_tokens"]

    # HARD GATE: compression must never actually fire under the no-op budget.
    assert noop_result["compaction_count"] == 0

    # HARD GATE: teacher-forced argmax token outputs must be identical.
    full_argmax = [step[1] for step in full_result["per_step_logits_summary"]]
    noop_argmax = [step[1] for step in noop_result["per_step_logits_summary"]]
    assert full_argmax == noop_argmax

    # DIAGNOSTIC: report max logit magnitude delta at every step (not a
    # hard-fail tolerance check beyond the declared bounds — see
    # docs/REPLAY_DESIGN.md and build brief §6.1: index/event agreement is
    # the hard gate, float closeness is reported).
    max_abs_deltas = []
    for (full_max, _), (noop_max, _) in zip(
        full_result["per_step_logits_summary"], noop_result["per_step_logits_summary"]
    ):
        max_abs_deltas.append(abs(full_max - noop_max))
    print(f"patched_noop parity: max abs logit-max delta across steps = {max(max_abs_deltas):.6f}")
    for delta, (full_max, _) in zip(max_abs_deltas, full_result["per_step_logits_summary"]):
        rel = delta / (abs(full_max) + 1e-8)
        assert delta <= LOGIT_ABS_TOL or rel <= LOGIT_REL_TOL, (
            f"logit-max delta {delta} exceeds declared tolerance "
            f"(abs<= {LOGIT_ABS_TOL} or rel<= {LOGIT_REL_TOL})"
        )
