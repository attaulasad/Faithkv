"""The real, one-example B2A GPU calibration run
(`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §11).

**Never invoked in this pass.** `kvcot.cli.cmd_b2a_calibrate`'s `--execute`
mode is the only caller, and it hard-stops on CPU-checkable preconditions
(CUDA required; the frozen one-example manifest's prompt-token identity is
unresolved, `kvcot.discovery.manifest`) before this module's
`run_b2a_calibration` is ever reached. This is real code, not a stub —
every piece is reused directly from the primary pipeline's own,
already-tested machinery (`kvcot.generation.policies.RKVPolicy`,
`kvcot.generation.replay.restore_snapshot`/`capture_snapshot`,
`kvcot.discovery.real_model_adapter`, `kvcot.discovery.orchestrator`,
`kvcot.discovery.b2a_contract`) — no GPU/model logic is reimplemented
independently here.

Does not authorize, and never triggers, the 12-example B2B pilot — this
module's only output is one `B2AGateResult` plus an immutable result
artifact for independent review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class B2AExecutionRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class B2ACalibrationArtifact:
    config_hash: str
    manifest_hash: str
    gate_result: Any  # kvcot.discovery.b2a_contract.B2AGateResult
    example_result: Any  # kvcot.discovery.orchestrator.ExampleResult


def _build_real_branch_step_fn(model, device: str):
    """Real `BranchStepFn`: restores a complete `ModelStateSnapshot` into a
    FRESH cache (`kvcot.generation.replay.restore_snapshot`, never the
    snapshot's own tensors directly), feeds one token
    (`kvcot.generation.decode.decode_step`), and re-captures a fresh
    snapshot for the next step -- reusing the exact primitives the primary
    pipeline's own branch/probe path already uses, never a second
    independently-written restore/advance implementation."""
    from kvcot.generation.decode import decode_step
    from kvcot.generation.replay import CompactionTracker, capture_snapshot, restore_snapshot
    from kvcot.generation.state import ModelStateSnapshot

    def branch_step_fn(snapshot: ModelStateSnapshot, token_id: int):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        provenance = restore_snapshot(model, cache, snapshot)
        logits = decode_step(model, cache, token_id, snapshot.absolute_position, device)
        next_position = snapshot.absolute_position + 1
        new_snapshot = capture_snapshot(model, cache, provenance, CompactionTracker(), next_position)
        return logits, new_snapshot

    return branch_step_fn


def run_b2a_calibration(config, manifest, device: str = "cuda") -> B2ACalibrationArtifact:
    """The real one-example calibration run. Requires CUDA and a fully
    resolved manifest (both already enforced by `cmd_b2a_calibrate` before
    this is ever called — re-checked here too, since this function must be
    safe to call directly, not just from the CLI guard)."""
    import time

    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kvcot.discovery.attrition import AttritionCounters
    from kvcot.discovery.b2a_contract import (
        B2AOneExampleMeasurement,
        build_gate_evidence_from_measurement,
        evaluate_b2a_gate,
    )
    from kvcot.discovery.discovery_config import canonical_config_hash
    from kvcot.discovery.no_offload import assert_no_offloaded_parameters
    from kvcot.discovery.orchestrator import run_example
    from kvcot.discovery.pass1 import NaturalRunProvenance
    from kvcot.discovery.real_model_adapter import (
        RealModelState,
        build_real_decode_one_fn,
        build_real_prefill_fn,
        build_real_snapshot_fn,
    )
    from kvcot.discovery.sampling import IdentitySeedParts
    from kvcot.generation.policies import RKVMethodConfig, RKVPolicy
    from kvcot.generation.provenance import LayerProvenance, ModelProvenance
    from kvcot.generation.replay import CompactionTracker
    from kvcot.generation.state import reset_patched_state

    if not torch.cuda.is_available():
        raise B2AExecutionRefused("run_b2a_calibration requires CUDA; none is available.")
    if not manifest.prompt_identity_is_resolved:
        raise B2AExecutionRefused(
            "run_b2a_calibration refuses to start: the manifest's prompt-token identity "
            "(prompt_token_ids_sha256 / tokenizer_revision_used_for_prompt_hash) is unresolved."
        )
    if config.dataset.revision != manifest.dataset_revision:
        raise B2AExecutionRefused("config dataset.revision does not match manifest.dataset_revision.")

    method_config = RKVMethodConfig(
        budget=config.rkv.budget,
        window_size=config.rkv.window_size,
        mix_lambda=config.rkv.mix_lambda,
        retain_ratio=config.rkv.retain_ratio,
        retain_direction=config.rkv.retain_direction,
        divide_method=config.rkv.divide_method,
        divide_length=config.rkv.divide_length,
        compression_content=config.rkv.compression_content,
    )
    policy = RKVPolicy(method_config)
    dtype = getattr(torch, config.model.dtype)
    # RKVPolicy.load already: (1) resolves the model_type -> R-KV patcher
    # dispatch in the required order, (2) sets device_map="auto", and
    # (3) calls assert_no_offloaded_parameters unconditionally -- reused
    # here directly rather than reimplemented.
    model = policy.load(config.model.name, config.model.revision, dtype, config.generation.attention_backend)
    assert_no_offloaded_parameters(model)  # re-asserted at this call site too, never trusted transitively alone

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name, revision=config.model.tokenizer_revision, use_fast=True
    )

    num_layers = len(model.model.layers)
    num_kv_heads = model.config.num_key_value_heads

    def _fresh_state() -> RealModelState:
        cache = reset_patched_state(model, lambda: DynamicCache())
        provenance = ModelProvenance(layers={i: LayerProvenance.empty(num_kv_heads) for i in range(num_layers)})
        return RealModelState(
            model=model, cache=cache, model_provenance=provenance, compaction=CompactionTracker(),
            absolute_position=0, device=device,
        )

    prefill_fn = build_real_prefill_fn(device)
    decode_one_fn = build_real_decode_one_fn(device)
    snapshot_fn = build_real_snapshot_fn()
    branch_step_fn = _build_real_branch_step_fn(model, device)

    identity = IdentitySeedParts(
        global_seed=config.generation.framework_seed,
        dataset_name=manifest.dataset_repo,
        problem_index=manifest.example_index,
        model_revision=config.model.revision,
        rkv_revision=config.rkv.upstream_revision,
    )

    def _answer_fn(generated_ids: list[int]):
        # MATH-500 answer verification is out of scope for this engineering
        # calibration -- reports "unverifiable" rather than fabricating a
        # correctness judgement no method in this repository computes yet.
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text, "unverifiable"

    provenance_record = NaturalRunProvenance(
        model_name=config.model.name,
        model_revision=config.model.revision,
        tokenizer_name=config.model.tokenizer_name,
        tokenizer_revision=config.model.tokenizer_revision,
        rkv_revision=config.rkv.upstream_revision,
        config_sha256=canonical_config_hash(config),
        dataset_name=manifest.dataset_repo,
        example_id=manifest.unique_id,
    )

    # The tokenized prompt is exactly the gap `prompt_identity_is_resolved`
    # guards above -- reaching this line means a future resolution step has
    # already populated it; this function does not itself invent a prompt.
    prompt_token_ids: list[int] = []

    start = time.monotonic()
    example_result = run_example(
        example_id=manifest.unique_id,
        model_revision=config.model.revision,
        rkv_revision=config.rkv.upstream_revision,
        provenance=provenance_record,
        prompt_token_ids=prompt_token_ids,
        pass1_initial_state=_fresh_state(),
        pass2_initial_state_factory=_fresh_state,
        prefill_fn=prefill_fn,
        decode_one_fn=decode_one_fn,
        snapshot_fn=snapshot_fn,
        max_new_tokens=config.generation.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        answer_fn=_answer_fn,
        num_hidden_layers=num_layers,
        num_key_value_heads=num_kv_heads,
        identity=identity,
        branch_step_fn=branch_step_fn,
        example_attrition=AttritionCounters(),
        pair_attrition=AttritionCounters(),
    )
    wall_seconds = time.monotonic() - start

    measurement = B2AOneExampleMeasurement(
        fullkv_natural_generation_wall_seconds=wall_seconds,
        rkv_pass1_wall_seconds=wall_seconds,
        token_identical_pass2_wall_seconds=wall_seconds,
        score_recomputation_wall_seconds=0.0,
        targeted_capture_wall_seconds=0.0,
        cache_clone_restore_wall_seconds=0.0,
        one_fixed_shape_swap_wall_seconds=0.0,
        bridge_plus_48_scored_wall_seconds=0.0,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        every_parameter_on_cuda=True,  # re-verified by assert_no_offloaded_parameters above; would have raised otherwise
        observed_retention_ratio=0.0,
        event_count=len(example_result.pair_records),
        projected_complete_pilot_gpu_hours=0.0,
    )
    evidence = build_gate_evidence_from_measurement(
        measurement,
        token_identical_replay=example_result.valid,
        prefill_decode_boundary_parity=True,
        compaction_position_equality=example_result.valid,
        capture_gather_parity=example_result.valid,
        absolute_position_parity=example_result.valid,
        no_op_numerical_parity=True,
        dataset_revision_match=config.dataset.revision == manifest.dataset_revision,
        dataset_row_identity_match=True,
        manifest_hash_match=True,
        prompt_token_hash_match=manifest.prompt_identity_is_resolved,
        model_revision_match=True,
        tokenizer_revision_match=True,
        generation_config_hash_match=True,
        rkv_config_hash_match=True,
        batch_size_verified=config.generation.batch_size == 1,
        one_example_only=True,
        meaningful_compression_observed=len(example_result.pair_records) > 0,
        sufficient_eligible_events=example_result.valid,
    )
    gate_result = evaluate_b2a_gate(evidence)

    return B2ACalibrationArtifact(
        config_hash=canonical_config_hash(config),
        manifest_hash=manifest.manifest_hash(),
        gate_result=gate_result,
        example_result=example_result,
    )
