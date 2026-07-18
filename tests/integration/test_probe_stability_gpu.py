"""f=1 stability probe test (§10 Stage 0 pass criteria: "at f=1, >=90% of
valid probes reproduce their own base answer").

f=1 is the stability control (§4/§8.1): branching at the *end* of the
natural think span and asking again should reproduce the model's own
already-given answer, for both FullKV and R-KV. This is a check on our own
protocol (control suffix wording, decoding config), not a scientific result
— §10 is explicit that a failure here means "inspect the control suffix and
decoding protocol first, report, ask," not tune the statistic to pass.

Validity accounting (fixed 2026-07-18, external review — two real defects
in the prior version):

  1. `N_SMOKE_EXAMPLES` was 10, but the pre-registered smoke definition
     (`data/manifests/gsm8k_smoke_20.jsonl`, docs/GPU_VALIDATION_PLAN.md §6)
     is 20 rows. Corrected to 20 — the frozen definition, not a convenience
     subset.
  2. `n_valid` was incremented BEFORE checking whether the probe's answer
     was extractable at all, so an unextractable probe counted as a VALID-
     but-unstable pair — while the failure message simultaneously described
     the denominator as "both-extractable". A valid stability pair now
     requires ALL of: base generation did not hit its cap, base answer
     extracted, think span parsed, probe did not stop on `max_new_tokens`
     (cap exhaustion means the probe never finished answering — comparing a
     truncated non-answer against the base answer measures nothing about
     stability), and probe answer extracted. Everything else is excluded
     from the denominator and reported in `per_example` diagnostics instead
     of silently disappearing into either count.

The threshold (0.90) and the control suffix are NOT tuned here, per §10.
The prior GPU run's 7/10 result predates this accounting fix and remains
unresolved — this fix changes what is COUNTED, not what passes; re-run on
the frozen 20-row definition and inspect per-example diagnostics before
drawing any Stage 0 conclusion.

Process isolation (§ Step 14, 2026-07-16): the original version ran both
parametrized conditions ("full", "rkv_b{RKV_BUDGET}") inside one pytest
process. `kvcot.generation.state.declare_process_mode` would already refuse
that (stock vs. patched in one process raises `ProcessModeConflictError`),
and even without that guard the R-KV monkeypatch on
`transformers.models.qwen2` is process-global with no per-instance undo
(docs/UPSTREAM_AUDIT.md H1) — a second condition loaded in the same process
would silently run against an already-patched class. Each condition's
`_run_condition_stability` body now runs inside its own `spawn`ed
subprocess (never `fork`, so neither process inherits an already-imported,
possibly-patched `transformers.models.qwen2` from the other), mirroring the
pattern already used successfully in
tests/integration/test_patched_noop_parity_gpu.py. The parent pytest
process receives only a small result dict back over the queue — the
model/tensors themselves never cross the process boundary. A crashed child
is detected by polling (`Process.is_alive`) rather than blocking on the
full queue timeout, and a child-side exception is propagated back as a
structured `{"error": <traceback>}` payload instead of a bare nonzero exit.

Implemented in full, marked `@pytest.mark.gpu`; auto-skipped on this build
machine (no CUDA device). See docs/GPU_VALIDATION_PLAN.md.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
import time
import traceback

import pytest

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
STABILITY_THRESHOLD = 0.90
# The frozen smoke definition is the full 20-row manifest (docs/
# GPU_VALIDATION_PLAN.md §6) — never a convenience subset of it.
N_SMOKE_EXAMPLES = 20
RKV_BUDGET = 256
WINDOW_SIZE = 8
MIX_LAMBDA = 0.1
RETAIN_RATIO = 0.2
RETAIN_DIRECTION = "last"
DIVIDE_METHOD = "step_length"
DIVIDE_LENGTH = 128
BASE_MAX_NEW_TOKENS = 6144
PROBE_MAX_NEW_TOKENS = 48
TEMPERATURE = 0.6
TOP_P = 0.95
GLOBAL_SEED = 42
SUBPROCESS_TIMEOUT_SECONDS = 2400  # 20 examples, base cap 6144 -- roughly 2x the prior 10-example budget


def _load_manifest_rows(n: int) -> list[dict]:
    from kvcot.data import read_manifest

    rows = list(read_manifest("data/manifests/gsm8k_smoke_20.jsonl"))
    return rows[:n]


def _run_condition_stability(condition: str) -> dict:  # pragma: no cover - runs in a spawned process
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.decode import generate_base, decode_step
    from kvcot.generation.policies import build_policy
    from kvcot.generation.replay import replay_and_snapshot, branch_and_probe
    from kvcot.generation.sampling import make_generator
    from kvcot.generation.state import reset_patched_state
    from kvcot.probes.early_answering import find_think_span
    from kvcot.probes.templates import render_base_user_message, render_control_suffix
    from kvcot.utils.answers import extract_answer, answers_match
    from kvcot.config import load_lock_config

    lock = load_lock_config("configs/lock.yaml")
    policy = build_policy(condition, lock)
    model = policy.load(MODEL_NAME, MODEL_REVISION, torch.bfloat16, lock.attention.primary)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_fast=True)
    device = "cuda"
    open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    suffix_ids = tokenizer.encode(render_control_suffix(), add_special_tokens=False)

    rows = _load_manifest_rows(N_SMOKE_EXAMPLES)
    n_valid = 0
    n_stable = 0
    per_example: list[dict] = []

    for row in rows:
        diag = {
            "source_row_index": row["source_row_index"],
            "base_answer": None,
            "probe_answer": None,
            "probe_stop_reason": None,
            "base_cap_hit": None,
            "base_extraction_method": None,
            "probe_extraction_method": None,
            "think_parse_status": None,
            "valid": False,
            "stable": None,
            "exclusion_reason": None,
        }
        per_example.append(diag)

        user_message = render_base_user_message(row["question"])
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}], tokenize=True, add_generation_prompt=True
        )
        generator, _seed = make_generator(GLOBAL_SEED, "gsm8k", row["source_row_index"], device)

        cache = reset_patched_state(model, lambda: DynamicCache())
        base_result = generate_base(
            model, cache, prompt_ids, BASE_MAX_NEW_TOKENS, TEMPERATURE, TOP_P, generator,
            tokenizer.eos_token_id, device,
        )
        diag["base_cap_hit"] = base_result.cap_hit
        if base_result.cap_hit:
            # excluded, not counted as unstable — cap hits are their own diagnostic (§8.3/§8.4)
            diag["exclusion_reason"] = "base_cap_hit"
            continue

        base_text = tokenizer.decode(base_result.generated_token_ids, skip_special_tokens=True)
        base_answer = extract_answer(base_text)
        diag["base_answer"] = base_answer.normalized_value
        diag["base_extraction_method"] = base_answer.method
        if base_answer.normalized_value is None:
            diag["exclusion_reason"] = "base_answer_unextractable"
            continue

        span = find_think_span(prompt_ids, base_result.generated_token_ids, open_ids, close_ids)
        diag["think_parse_status"] = span.think_parse_status
        if span.think_parse_status not in ("ok", "generation_prompt_preopened_ok"):
            diag["exclusion_reason"] = "think_parse_failed"
            continue

        end_pos = len(prompt_ids) + span.think_end_index
        snapshots = replay_and_snapshot(
            model=model,
            fresh_cache_factory=lambda: DynamicCache(),
            prompt_token_ids=prompt_ids,
            generated_token_ids=base_result.generated_token_ids,
            think_span=span,
            snapshot_absolute_positions={1.0: end_pos},
            device=device,
        )
        probe = branch_and_probe(
            model, DynamicCache(), snapshots[1.0], close_ids, suffix_ids,
            PROBE_MAX_NEW_TOKENS, tokenizer.eos_token_id, device,
        )
        diag["probe_stop_reason"] = probe.stop_reason
        if probe.stop_reason == "max_new_tokens":
            # Cap exhaustion: the probe never finished answering — invalid,
            # never "unstable" (2026-07-18 accounting fix, module docstring).
            diag["exclusion_reason"] = "probe_cap_hit"
            continue

        probe_text = tokenizer.decode(probe.probe_output_token_ids, skip_special_tokens=True)
        probe_answer = extract_answer(probe_text)
        diag["probe_answer"] = probe_answer.normalized_value
        diag["probe_extraction_method"] = probe_answer.method
        if probe_answer.normalized_value is None:
            # Unextractable probe answer: invalid, never "unstable" — the
            # prior version incremented n_valid before this check.
            diag["exclusion_reason"] = "probe_answer_unextractable"
            continue

        n_valid += 1
        diag["valid"] = True
        stable = answers_match(probe_answer.normalized_value, base_answer.normalized_value)
        diag["stable"] = stable
        if stable:
            n_stable += 1

    return {"n_valid": n_valid, "n_stable": n_stable, "per_example": per_example}


def _stability_worker(condition: str, result_queue: "mp.Queue") -> None:  # pragma: no cover - runs in a spawned process
    try:
        result_queue.put(_run_condition_stability(condition))
    except BaseException:
        # Propagate a structured error instead of dying silently — the
        # parent would otherwise block until the full queue timeout.
        result_queue.put({"error": traceback.format_exc()})
        raise


def _run_in_subprocess(condition: str) -> dict:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_stability_worker, args=(condition, queue))
    proc.start()
    deadline = time.monotonic() + SUBPROCESS_TIMEOUT_SECONDS
    result: dict | None = None
    try:
        # Poll instead of one blocking get: a child that crashes before
        # putting anything must fail the test immediately with its exit
        # code, not stall the parent for the entire timeout.
        while result is None:
            try:
                result = queue.get(timeout=10)
            except queue_module.Empty:
                if not proc.is_alive():
                    proc.join(timeout=30)
                    raise AssertionError(
                        f"subprocess for condition={condition} exited with {proc.exitcode} "
                        "before reporting any result"
                    )
                if time.monotonic() > deadline:
                    proc.terminate()
                    proc.join(timeout=30)
                    raise AssertionError(
                        f"subprocess for condition={condition} produced no result within "
                        f"{SUBPROCESS_TIMEOUT_SECONDS}s"
                    )
    finally:
        if proc.is_alive() and result is None:
            proc.terminate()
    proc.join(timeout=1200)
    if "error" in result:
        raise AssertionError(f"subprocess for condition={condition} raised:\n{result['error']}")
    assert proc.exitcode == 0, f"subprocess for condition={condition} exited with {proc.exitcode}"
    return result


def _format_diagnostics(per_example: list[dict]) -> str:
    lines = []
    for d in per_example:
        lines.append(
            f"  row={d['source_row_index']}: valid={d['valid']} stable={d['stable']} "
            f"exclusion={d['exclusion_reason']} base={d['base_answer']!r} probe={d['probe_answer']!r} "
            f"probe_stop={d['probe_stop_reason']} base_cap_hit={d['base_cap_hit']} "
            f"extraction=({d['base_extraction_method']}/{d['probe_extraction_method']})"
        )
    return "\n".join(lines)


@pytest.mark.gpu
@pytest.mark.parametrize("condition", ["full", f"rkv_b{RKV_BUDGET}"])
def test_f1_probe_stability_meets_threshold(condition):
    result = _run_in_subprocess(condition)
    diagnostics = _format_diagnostics(result.get("per_example", []))
    print(f"condition={condition}: per-example diagnostics:\n{diagnostics}")
    assert result["n_valid"] > 0, (
        "no valid (non-cap-hit, think-span-parsed, both-extractable, probe-finished) examples "
        f"to evaluate stability on. Per-example diagnostics:\n{diagnostics}"
    )
    rate = result["n_stable"] / result["n_valid"]
    print(f"condition={condition}: f=1 stability rate = {rate:.3f} ({result['n_stable']}/{result['n_valid']})")
    assert rate >= STABILITY_THRESHOLD, (
        f"condition={condition} f=1 stability rate {rate:.3f} is below the {STABILITY_THRESHOLD} "
        "threshold (§10 Stage 0). Per the build brief: do not tune this statistic — inspect the "
        "control suffix (kvcot.probes.templates.CONTROL_SUFFIX_TEXT) and the probe decoding config "
        f"first, then report and ask before changing anything. Per-example diagnostics:\n{diagnostics}"
    )
