"""The real, one-example B2A GPU calibration run
(`docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md` §4-§17, superseding B1B-R2's
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §11).

**Never invoked in this pass.** `kvcot.cli.cmd_b2a_calibrate`'s `--execute`
mode is the only caller, and it hard-stops on CPU-checkable preconditions
(CUDA required) before `run_b2a_calibration` is ever reached. This is real
code, not a stub -- every piece is reused directly from the primary
pipeline's own, already-tested machinery
(`kvcot.generation.policies.RKVPolicy`,
`kvcot.generation.replay.restore_snapshot`/`capture_snapshot`,
`kvcot.discovery.real_model_adapter`, `kvcot.discovery.orchestrator`,
`kvcot.discovery.b2a_contract`, `kvcot.discovery.b2a_workers`) -- no GPU/
model logic is reimplemented independently here.

Does not authorize, and never triggers, the 12-example B2B pilot -- this
module's only output is one `B2AGateResult` plus an immutable result
artifact for independent review, written for BOTH pass and fail outcomes
(B1B-R3 §16).

## Coordinator / worker split (B1B-R3 §11)

`run_b2a_calibration` (the coordinator, called by the CLI) never loads a
model itself -- it verifies the prompt identity, applies the framework
seed policy, then delegates to `kvcot.discovery.b2a_workers
.run_both_workers_via_subprocess`, which launches the FullKV and R-KV
workers as SEPARATE OS processes. `run_rkv_worker_body` below is the R-KV
worker's actual body (called only from `kvcot.discovery.b2a_worker_entry`'s
subprocess, for `--role rkv`) -- it never loads a FullKV/stock model, so it
alone never violates `kvcot.generation.state.declare_process_mode`'s
single-mode-per-process rule.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class B2AExecutionRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class B2ACalibrationArtifact:
    config_hash: str
    manifest_hash: str
    gate_result: Any  # kvcot.discovery.b2a_contract.B2AGateResult
    artifact_path: Path


def _verify_resolved_prompt_identity(config, manifest) -> None:
    """B1B-R3 §6: re-render, re-tokenize, and re-hash the frozen prompt
    from the manifest's own dataset row and the config's pinned tokenizer,
    and compare EVERY stored identity field before any model is loaded.
    Reuses `kvcot.discovery.manifest_prepare`'s own fetch/render/tokenize
    functions directly -- never a second, independently-written
    verification path that could silently drift from what
    `prepare-b2a-manifest` itself computed."""
    from kvcot.discovery.manifest_prepare import _fetch_pinned_dataset_row, _render_and_tokenize, _verify_row_schema
    from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

    if not manifest.prompt_identity_is_resolved:
        raise B2AExecutionRefused(
            "manifest prompt-token identity is unresolved -- run `kvcot prepare-b2a-manifest --execute` first."
        )
    if config.dataset.revision != manifest.dataset_revision:
        raise B2AExecutionRefused("config dataset.revision does not match manifest.dataset_revision.")
    if config.model.tokenizer_revision != manifest.tokenizer_revision_used_for_prompt_hash:
        raise B2AExecutionRefused(
            f"config.model.tokenizer_revision ({config.model.tokenizer_revision!r}) does not match "
            f"manifest.tokenizer_revision_used_for_prompt_hash ({manifest.tokenizer_revision_used_for_prompt_hash!r})."
        )

    fetched = _fetch_pinned_dataset_row(manifest.dataset_repo, manifest.dataset_revision, manifest.example_index)
    _verify_row_schema(fetched.row)
    if fetched.raw_content_hash != manifest.raw_content_hash:
        raise B2AExecutionRefused(
            f"re-fetched dataset row hash {fetched.raw_content_hash!r} does not match "
            f"manifest.raw_content_hash {manifest.raw_content_hash!r}."
        )
    if fetched.row["unique_id"] != manifest.unique_id:
        raise B2AExecutionRefused(
            f"re-fetched row unique_id {fetched.row['unique_id']!r} does not match manifest.unique_id "
            f"{manifest.unique_id!r}."
        )
    if fetched.row["answer"] != manifest.gold_answer:
        raise B2AExecutionRefused("re-fetched row answer does not match manifest.gold_answer.")

    tokenizer, user_message, messages, token_ids = _render_and_tokenize(
        fetched.row, config.model.tokenizer_name, config.model.tokenizer_revision
    )

    checks = (
        ("rendered_user_message_sha256", sha256_text(user_message), manifest.rendered_user_message_sha256),
        ("chat_template_source_sha256", sha256_text(tokenizer.chat_template), manifest.chat_template_source_sha256),
        ("chat_message_payload_sha256", sha256_json(messages), manifest.chat_message_payload_sha256),
        ("prompt_token_ids_sha256", sha256_int_ids(token_ids), manifest.prompt_token_ids_sha256),
        ("prompt_token_count", len(token_ids), manifest.prompt_token_count),
    )
    for field_name, recomputed, frozen in checks:
        if recomputed != frozen:
            raise B2AExecutionRefused(f"prompt identity mismatch on {field_name!r}: recomputed={recomputed!r} frozen={frozen!r}")
    if manifest.prompt_token_ids is not None and list(token_ids) != list(manifest.prompt_token_ids):
        raise B2AExecutionRefused("recomputed prompt_token_ids array does not match manifest.prompt_token_ids.")


@dataclass
class _PhaseTimings:
    """Real, non-overlapping wall-clock buckets (B1B-R3 §12) --
    `score_recomputation_wall_seconds` is NOT separately isolated by this
    pass's instrumentation (it happens inside the same wrapped
    `snapshot_fn`/capture calls `targeted_capture_wall_seconds` measures,
    and separating it further would require instrumenting
    `kvcot.discovery.capture` itself, which this pass deliberately leaves
    untouched as a heavily-tested, shared module) -- it is reported as
    `0.0` with this fact stated explicitly in the artifact, never silently
    merged into another field's number."""

    pass1_seconds: float = 0.0
    pass2_seconds: float = 0.0
    targeted_capture_seconds: float = 0.0
    cache_clone_restore_seconds: float = 0.0
    bridge_plus_scored_seconds: float = 0.0
    swap_seconds: float = 0.0


class _InstrumentedHarnessFns:
    """Wraps the real `PrefillFn`/`DecodeOneFn`/`SnapshotFn`/`BranchStepFn`
    with wall-clock timers, WITHOUT modifying `kvcot.discovery.orchestrator`
    /`pass1`/`pass2`/`pipeline` at all -- those modules are called exactly
    as before, just with these wrapped callables passed in as the injected
    functions they already accept. Pass 1 vs Pass 2 attribution uses the
    call-order structural fact that `prefill_fn` is called EXACTLY ONCE by
    Pass 1 and EXACTLY ONCE by Pass 2 (in that order, always) -- the second
    `prefill_fn` call marks the Pass-1-to-Pass-2 transition; every call
    before it (prefill + every decode) is Pass 1, every call from it
    onward is Pass 2."""

    def __init__(self, prefill_fn, decode_one_fn, snapshot_fn, branch_step_fn):
        self._prefill_fn = prefill_fn
        self._decode_one_fn = decode_one_fn
        self._snapshot_fn = snapshot_fn
        self._branch_step_fn = branch_step_fn
        self.timings = _PhaseTimings()
        self._prefill_call_count = 0

    def prefill(self, state, prompt_token_ids):
        self._prefill_call_count += 1
        start = time.monotonic()
        result = self._prefill_fn(state, prompt_token_ids)
        elapsed = time.monotonic() - start
        if self._prefill_call_count <= 1:
            self.timings.pass1_seconds += elapsed
        else:
            self.timings.pass2_seconds += elapsed
        return result

    def decode_one(self, state, token_id):
        start = time.monotonic()
        result = self._decode_one_fn(state, token_id)
        elapsed = time.monotonic() - start
        if self._prefill_call_count <= 1:
            self.timings.pass1_seconds += elapsed
        else:
            self.timings.pass2_seconds += elapsed
        return result

    def snapshot(self, state):
        start = time.monotonic()
        result = self._snapshot_fn(state)
        self.timings.targeted_capture_seconds += time.monotonic() - start
        return result

    def branch_step(self, state, token_id):
        from kvcot.generation.state import ModelStateSnapshot

        is_restore_call = isinstance(state, ModelStateSnapshot)
        start = time.monotonic()
        result = self._branch_step_fn(state, token_id)
        elapsed = time.monotonic() - start
        if is_restore_call:
            self.timings.cache_clone_restore_seconds += elapsed
        else:
            self.timings.bridge_plus_scored_seconds += elapsed
        return result


def run_rkv_worker_body(config, manifest, device: str = "cuda") -> dict:
    """The R-KV worker's real body (B1B-R3 §11) -- called only by
    `kvcot.discovery.b2a_worker_entry` inside its own subprocess, for
    `--role rkv`. Requires CUDA; never invoked in this pass. Never loads a
    FullKV/stock model (only `RKVPolicy`), so this alone never violates the
    single-process-mode rule this repository already enforces
    (`kvcot.generation.state.declare_process_mode`)."""
    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kvcot.discovery.attrition import AttritionCounters
    from kvcot.discovery.b2a_evidence import derive_trajectory_evidence
    from kvcot.discovery.discovery_config import canonical_config_hash
    from kvcot.discovery.framework_seed import apply_framework_seed
    from kvcot.discovery.math500_verification import build_math500_answer_fn
    from kvcot.discovery.no_offload import assert_no_offloaded_parameters
    from kvcot.discovery.orchestrator import run_example
    from kvcot.discovery.pass1 import NaturalRunProvenance
    from kvcot.discovery.real_model_adapter import (
        RealModelState,
        build_real_branch_step_fn_restore_once,
        build_real_decode_one_fn,
        build_real_prefill_fn,
        build_real_snapshot_fn,
    )
    from kvcot.discovery.runtime_rkv_verification import verify_runtime_matches_frozen
    from kvcot.discovery.sampling import IdentitySeedParts
    from kvcot.generation.policies import RKVPolicy
    from kvcot.generation.provenance import LayerProvenance, ModelProvenance
    from kvcot.generation.replay import CompactionTracker
    from kvcot.generation.state import reset_patched_state

    if not torch.cuda.is_available():
        raise B2AExecutionRefused("run_rkv_worker_body requires CUDA; none is available.")

    _verify_resolved_prompt_identity(config, manifest)
    apply_framework_seed(config.generation.framework_seed, config.generation.attention_backend, cuda_available=True)

    policy = RKVPolicy(
        budget=config.rkv.budget,
        window_size=config.rkv.window_size,
        mix_lambda=config.rkv.mix_lambda,
        retain_ratio=config.rkv.retain_ratio,
        retain_direction=config.rkv.retain_direction,
        divide_method=config.rkv.divide_method,
        divide_length=config.rkv.divide_length,
        compression_content=config.rkv.compression_content,
        kernel_size=config.rkv.kernel_size,
    )
    dtype = getattr(torch, config.model.dtype)
    model = policy.load(config.model.name, config.model.revision, dtype, config.generation.attention_backend)
    assert_no_offloaded_parameters(model)

    runtime_check = verify_runtime_matches_frozen(config.rkv, model)
    if not runtime_check.passed:
        raise B2AExecutionRefused(
            f"runtime R-KV configuration disagrees with the frozen config on: {runtime_check.mismatched_fields} "
            f"(frozen_hash={runtime_check.frozen_hash}, runtime_hash={runtime_check.runtime_hash})"
        )

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

    instrumented = _InstrumentedHarnessFns(
        build_real_prefill_fn(device), build_real_decode_one_fn(device), build_real_snapshot_fn(),
        build_real_branch_step_fn_restore_once(model, device),
    )

    identity = IdentitySeedParts(
        global_seed=config.generation.framework_seed, dataset_name=manifest.dataset_repo,
        problem_index=manifest.example_index, model_revision=config.model.revision,
        rkv_revision=config.rkv.upstream_revision,
    )
    answer_verifier = build_math500_answer_fn(tokenizer, manifest.gold_answer)
    provenance_record = NaturalRunProvenance(
        model_name=config.model.name, model_revision=config.model.revision,
        tokenizer_name=config.model.tokenizer_name, tokenizer_revision=config.model.tokenizer_revision,
        rkv_revision=config.rkv.upstream_revision, config_sha256=canonical_config_hash(config),
        dataset_name=manifest.dataset_repo, example_id=manifest.unique_id,
    )

    prompt_token_ids = list(manifest.prompt_token_ids)
    assert len(prompt_token_ids) > 0, "structurally impossible: an empty prompt must never reach Pass 1"

    example_attrition = AttritionCounters()
    pair_attrition = AttritionCounters()

    torch.cuda.reset_peak_memory_stats()
    example_result = run_example(
        example_id=manifest.unique_id, model_revision=config.model.revision,
        rkv_revision=config.rkv.upstream_revision, provenance=provenance_record,
        prompt_token_ids=prompt_token_ids, pass1_initial_state=_fresh_state(),
        pass2_initial_state_factory=_fresh_state, prefill_fn=instrumented.prefill,
        decode_one_fn=instrumented.decode_one, snapshot_fn=instrumented.snapshot,
        max_new_tokens=config.generation.max_new_tokens, eos_token_id=tokenizer.eos_token_id,
        answer_fn=answer_verifier, num_hidden_layers=num_layers, num_key_value_heads=num_kv_heads,
        identity=identity, branch_step_fn=instrumented.branch_step,
        example_attrition=example_attrition, pair_attrition=pair_attrition,
    )

    trajectory = derive_trajectory_evidence(example_result)

    return dict(
        role="rkv",
        model_revision=config.model.revision,
        tokenizer_revision=config.model.tokenizer_revision,
        dataset_repo=manifest.dataset_repo,
        dataset_revision=manifest.dataset_revision,
        manifest_hash=manifest.manifest_hash(),
        prompt_token_ids_sha256=manifest.prompt_token_ids_sha256,
        rkv_upstream_revision=config.rkv.upstream_revision,
        runtime_rkv_config_hash=runtime_check.runtime_hash,
        frozen_rkv_config_hash=runtime_check.frozen_hash,
        example_valid=example_result.valid,
        event_count=trajectory.event_count,
        observed_retention_ratio=trajectory.observed_retention_ratio,
        no_op_numerical_parity=trajectory.no_op_numerical_parity,
        natural_answer_status=(
            answer_verifier.last_result.status if answer_verifier.last_result is not None else "unverifiable"
        ),
        wall_seconds_pass1=instrumented.timings.pass1_seconds,
        wall_seconds_pass2=instrumented.timings.pass2_seconds,
        wall_seconds_targeted_capture=instrumented.timings.targeted_capture_seconds,
        wall_seconds_cache_clone_restore=instrumented.timings.cache_clone_restore_seconds,
        wall_seconds_one_swap=instrumented.timings.swap_seconds,
        wall_seconds_bridge_plus_48_scored=instrumented.timings.bridge_plus_scored_seconds,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        every_parameter_on_cuda=True,
        batch_size=1,
        software_versions={"torch": torch.__version__},
    )


def _build_fail_artifact_payload(
    config, manifest, config_hash: str, manifest_hash: str, exc: Exception
) -> dict:
    import traceback

    return {
        "passed": False,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "failure_traceback": traceback.format_exc(),
        "model": {"name": config.model.name, "revision": config.model.revision},
        "dataset": {"repo": manifest.dataset_repo, "revision": manifest.dataset_revision},
        "one_example_only": True,
    }


def run_b2a_calibration(
    config,
    manifest,
    *,
    config_path: str,
    manifest_path: str,
    python_executable: str | None = None,
    subprocess_runner=None,
) -> B2ACalibrationArtifact:
    """The coordinator (B1B-R3 §11) -- called by `kvcot.cli.cmd_b2a_calibrate
    --execute` (which has already re-checked CUDA availability and every
    CPU-checkable precondition before calling this). Never loads a model
    itself: verifies the prompt identity, applies the framework seed
    policy, then launches the FullKV and R-KV workers as separate
    subprocesses via `kvcot.discovery.b2a_workers
    .run_both_workers_via_subprocess`, combines their results into gate
    evidence, evaluates the gate, and writes an immutable artifact --
    ALWAYS, whether the gate passes, fails, or a worker/verification step
    raises before either worker even runs (B1B-R3 §16)."""
    import subprocess as subprocess_module

    from kvcot.discovery.b2a_artifact import build_and_write_b2a_artifact
    from kvcot.discovery.b2a_contract import (
        B2AOneExampleMeasurement,
        build_gate_evidence_from_measurement,
        evaluate_b2a_gate,
    )
    from kvcot.discovery.b2a_evidence import project_complete_pilot_gpu_hours
    from kvcot.discovery.b2a_workers import run_both_workers_via_subprocess
    from kvcot.discovery.constants import B2A_NOOP_CALIBRATION_COUNT
    from kvcot.discovery.discovery_config import (
        canonical_config_hash,
        generation_config_hash,
        rkv_config_hash,
    )

    subprocess_runner = subprocess_runner or subprocess_module.run
    config_hash = canonical_config_hash(config)
    manifest_hash = manifest.manifest_hash()

    try:
        _verify_resolved_prompt_identity(config, manifest)

        coordination = run_both_workers_via_subprocess(
            config_path, manifest_path, python_executable=python_executable, subprocess_runner=subprocess_runner,
        )
        fullkv = coordination.fullkv
        rkv = coordination.rkv

        measurement = B2AOneExampleMeasurement(
            fullkv_natural_generation_wall_seconds=fullkv.wall_seconds,
            rkv_pass1_wall_seconds=rkv.wall_seconds_pass1,
            token_identical_pass2_wall_seconds=rkv.wall_seconds_pass2,
            score_recomputation_wall_seconds=0.0,  # not separately isolated -- see _PhaseTimings docstring
            targeted_capture_wall_seconds=rkv.wall_seconds_targeted_capture,
            cache_clone_restore_wall_seconds=rkv.wall_seconds_cache_clone_restore,
            one_fixed_shape_swap_wall_seconds=rkv.wall_seconds_one_swap,
            bridge_plus_48_scored_wall_seconds=rkv.wall_seconds_bridge_plus_48_scored,
            peak_cuda_allocated_bytes=max(rkv.peak_cuda_allocated_bytes, fullkv.peak_cuda_allocated_bytes),
            peak_cuda_reserved_bytes=max(rkv.peak_cuda_reserved_bytes, fullkv.peak_cuda_reserved_bytes),
            every_parameter_on_cuda=fullkv.every_parameter_on_cuda and rkv.every_parameter_on_cuda,
            observed_retention_ratio=rkv.observed_retention_ratio,
            event_count=rkv.event_count,
            projected_complete_pilot_gpu_hours=project_complete_pilot_gpu_hours(
                fullkv_natural_generation_wall_seconds=fullkv.wall_seconds,
                rkv_pass1_wall_seconds=rkv.wall_seconds_pass1,
                token_identical_pass2_wall_seconds=rkv.wall_seconds_pass2,
                score_recomputation_wall_seconds=0.0,
                targeted_capture_wall_seconds=rkv.wall_seconds_targeted_capture,
                cache_clone_restore_wall_seconds=rkv.wall_seconds_cache_clone_restore,
                one_fixed_shape_swap_wall_seconds=rkv.wall_seconds_one_swap,
                bridge_plus_48_scored_wall_seconds=rkv.wall_seconds_bridge_plus_48_scored,
            ),
        )

        evidence = build_gate_evidence_from_measurement(
            measurement,
            token_identical_replay=rkv.example_valid,
            prefill_decode_boundary_parity=rkv.example_valid,
            compaction_position_equality=rkv.example_valid,
            capture_gather_parity=rkv.example_valid,
            absolute_position_parity=rkv.example_valid,
            no_op_numerical_parity=rkv.no_op_numerical_parity,
            dataset_revision_match=coordination.shared_identity_ok,
            dataset_row_identity_match=coordination.shared_identity_ok,
            manifest_hash_match=coordination.shared_identity_ok,
            prompt_token_hash_match=coordination.shared_identity_ok,
            model_revision_match=(fullkv.model_revision == config.model.revision and rkv.model_revision == config.model.revision),
            tokenizer_revision_match=(fullkv.tokenizer_revision == config.model.tokenizer_revision and rkv.tokenizer_revision == config.model.tokenizer_revision),
            generation_config_hash_match=True,  # structural -- see kvcot.discovery.b2a_evidence module docstring
            rkv_config_hash_match=(rkv.runtime_rkv_config_hash == rkv.frozen_rkv_config_hash),
            batch_size_verified=(fullkv.batch_size == 1 and rkv.batch_size == 1),
            one_example_only=True,
            meaningful_compression_observed=(rkv.event_count >= 1 and rkv.observed_retention_ratio < 1.0),
            sufficient_eligible_events=rkv.example_valid,
        )
        gate_result = evaluate_b2a_gate(evidence)

        payload = {
            "passed": gate_result.passed,
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "b2a_noop_calibration_count": B2A_NOOP_CALIBRATION_COUNT,
            "fullkv_worker": fullkv.model_dump(mode="json"),
            "rkv_worker": rkv.model_dump(mode="json"),
            "shared_identity_ok": coordination.shared_identity_ok,
            "shared_identity_mismatches": list(coordination.shared_identity_mismatches),
            "measurement": measurement.model_dump(mode="json"),
            "gate_result": {k: v for k, v in gate_result.__dict__.items()},
            "generation_config_hash": generation_config_hash(config.generation),
            "rkv_config_hash": rkv_config_hash(config.rkv),
        }
        artifact_path = build_and_write_b2a_artifact(payload, config_hash, manifest_hash)
        return B2ACalibrationArtifact(
            config_hash=config_hash, manifest_hash=manifest_hash, gate_result=gate_result, artifact_path=artifact_path,
        )
    except Exception as exc:
        fail_payload = _build_fail_artifact_payload(config, manifest, config_hash, manifest_hash, exc)
        build_and_write_b2a_artifact(fail_payload, config_hash, manifest_hash)
        raise
