"""The real B2A-R3 FullKV qualification worker (Step 3R4-Repair-2 Finding
1, `docs/B2A_R3_STEP3R4_REPAIR2_2026-07-23.md` §1).

Before this module, no production path existed from an R3 candidate row to
`kvcot.discovery.b2a_r3_worker_adapter.FullKVWorkerResultR3` -- the adapter
only ever converted an already-hand-constructed `FullKVWorkerResultR3`
built by test fixtures. `run_fullkv_r3_qualification_worker` closes that
gap: it reuses the canonical, already-frozen
`kvcot.discovery.b2a_workers.run_fullkv_worker` natural-generation loop
completely unmodified (never a second, independently-written FullKV
loop), renders the candidate's prompt through the canonical renderer
(`kvcot.discovery.manifest_prepare.render_with_loaded_tokenizer`, the same
chat-template call `prepare-b2a-manifest` uses), and derives the
candidate's think-span via `kvcot.probes.early_answering.find_think_span`
using the EXACT SAME tokenizer instance the worker itself ran with -- never
a second, independently-loaded tokenizer.

GPU-only in real production use (`run_fullkv_worker` itself hard-requires
CUDA unless a fake backend is injected); every CPU test exercises this
function's ENTIRE body via the same internal dependency-injection seams
`run_fullkv_worker` already exposes (`_load_model`/`_load_tokenizer`/
`_fresh_cache_factory`/`_cuda`/`_device`/`_clock`/`_progress`), so
"the real, injected worker body produced this evidence" is a genuine,
testable claim -- never a hand-built `FullKVWorkerResultR3`.

No line in this module imports R-KV or initializes CUDA itself (whatever
CUDA use happens is entirely inside the reused `run_fullkv_worker` body).
"""
from __future__ import annotations

from typing import Any, Callable

from kvcot.discovery.b2a_r3_candidates import CandidateRowR3
from kvcot.utils.hashing import sha256_int_ids

__all__ = ["run_fullkv_r3_qualification_worker"]


class _CandidateNaturalRunManifest:
    """The minimal shape `kvcot.discovery.b2a_workers.run_fullkv_worker`
    needs from its `manifest` argument, built from one already-verified R3
    candidate row plus a freshly-rendered prompt -- never a second,
    independently-defined manifest schema. `manifest_hash()` echoes the
    candidate's own already-verified `raw_row_sha256` rather than inventing
    a new hash formula; nothing downstream in the R3 qualification path
    reads this field (it exists only because `FullKVWorkerResult` -- the
    historical B2A-R1/R2 shape `run_fullkv_worker` still constructs
    internally -- requires it)."""

    __slots__ = (
        "dataset_repo",
        "dataset_revision",
        "example_index",
        "unique_id",
        "gold_answer",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "prompt_token_count",
        "_manifest_hash",
    )

    def __init__(
        self,
        *,
        dataset_repo: str,
        dataset_revision: str,
        example_index: int,
        unique_id: str,
        gold_answer: str,
        prompt_token_ids: list[int],
        prompt_token_ids_sha256: str,
        manifest_hash: str,
    ) -> None:
        self.dataset_repo = dataset_repo
        self.dataset_revision = dataset_revision
        self.example_index = example_index
        self.unique_id = unique_id
        self.gold_answer = gold_answer
        self.prompt_token_ids = prompt_token_ids
        self.prompt_token_ids_sha256 = prompt_token_ids_sha256
        self.prompt_token_count = len(prompt_token_ids)
        self._manifest_hash = manifest_hash

    def manifest_hash(self) -> str:
        return self._manifest_hash


def _worker_observed_generation_config(config: Any) -> dict[str, Any]:
    """Reconstructs the qualification-comparison-shaped generation-config
    payload from the SAME `config.generation` the worker actually ran
    under -- never the worker's own `runtime_generation` dict
    (`kvcot.discovery.runtime_evidence.build_runtime_generation_record`'s
    output), which serves a different, call-shape-focused purpose and is
    structurally incompatible with
    `kvcot.discovery.b2a_r3_contract.FROZEN_GENERATION_CONFIG` (different
    field set entirely). A real qualification run against the frozen
    `configs/discovery/llama8b_math500_b1024.yaml` config reproduces
    `FROZEN_GENERATION_CONFIG` byte-for-byte here; a config that has
    drifted from that frozen file legitimately fails
    `generation_config_hash_match` instead of silently passing."""
    gen = config.generation
    return {
        "generation_mode": gen.generation_mode,
        "do_sample": gen.do_sample,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "batch_size": gen.batch_size,
        "max_new_tokens": gen.max_new_tokens,
        "framework_seed": gen.framework_seed,
        "attention_backend": gen.attention_backend,
        "cache_implementation": gen.cache_implementation,
        "no_offload_required": gen.no_offload_required,
    }


def run_fullkv_r3_qualification_worker(
    config: Any,
    candidate: CandidateRowR3,
    *,
    _load_model: Callable[[], Any] | None = None,
    _load_tokenizer: Callable[[], Any] | None = None,
    _fresh_cache_factory: Callable[[], Any] | None = None,
    _cuda: Any | None = None,
    _device: str = "cuda:0",
    _clock: Callable[[], float] | None = None,
    _progress: Callable[[str, str, dict[str, Any] | None], None] | None = None,
) -> Any:
    """Runs ONE R3 candidate through the canonical FullKV natural-run loop
    and returns a typed object validated against
    `kvcot.discovery.b2a_r3_worker_adapter.FullKVWorkerResultR3`.

    Same internal, underscore-prefixed dependency-injection seams as
    `run_fullkv_worker` itself -- never exposed by a production CLI/
    subprocess entry point (none exists yet; Stage B execution remains
    unauthorized). Production (no seams injected) hard-requires CUDA via
    the reused `run_fullkv_worker` body; CPU tests inject fakes for all of
    them, exactly like `tests/unit/discovery/test_b2a_workers_real_bodies.py`
    already does for `run_fullkv_worker` directly.

    Dataset identity is sourced from the frozen R3 contract
    (`kvcot.discovery.b2a_r3_contract.DATASET_REPO`/`DATASET_CONFIG`/
    `DATASET_SPLIT`/`DATASET_REVISION`) rather than `config.dataset`,
    because every R3 candidate is, by construction
    (`kvcot.discovery.b2a_r3_candidates.CandidateManifestR3`), already
    pinned to that one frozen dataset revision -- never a second,
    independently-named dataset identity."""
    from kvcot.discovery.b2a_r3_contract import (
        DATASET_CONFIG,
        DATASET_REPO,
        DATASET_REVISION,
        DATASET_SPLIT,
    )
    from kvcot.discovery.b2a_r3_worker_adapter import (
        FULLKV_WORKER_RESULT_R3_SCHEMA_VERSION,
        FullKVWorkerResultR3,
    )
    from kvcot.discovery.discovery_config import canonical_config_hash
    from kvcot.discovery.b2a_workers import run_fullkv_worker
    from kvcot.discovery.manifest_prepare import _render_and_tokenize, render_with_loaded_tokenizer
    from kvcot.probes.early_answering import find_think_span
    from kvcot.utils.hashing import sha256_json

    if _load_tokenizer is not None:
        tokenizer = _load_tokenizer()
        user_message, messages, prompt_token_ids = render_with_loaded_tokenizer(tokenizer, candidate.row)
    else:
        from kvcot.discovery.snapshot_boundary import resolve_local_snapshot

        tokenizer_snapshot = resolve_local_snapshot(
            config.model.tokenizer_name, config.model.tokenizer_revision, "tokenizer"
        )
        tokenizer, user_message, messages, prompt_token_ids = _render_and_tokenize(
            candidate.row,
            config.model.tokenizer_name,
            config.model.tokenizer_revision,
            local_only_path=tokenizer_snapshot.local_path,
        )

    expected_prompt_token_ids_sha256 = sha256_int_ids(prompt_token_ids)

    manifest = _CandidateNaturalRunManifest(
        dataset_repo=DATASET_REPO,
        dataset_revision=DATASET_REVISION,
        example_index=candidate.source_example_index,
        unique_id=candidate.unique_id,
        gold_answer=candidate.row["answer"],
        prompt_token_ids=list(prompt_token_ids),
        prompt_token_ids_sha256=expected_prompt_token_ids_sha256,
        manifest_hash=candidate.raw_row_sha256,
    )

    # `_load_tokenizer` is always re-threaded here (even when the caller
    # already loaded one above) so `run_fullkv_worker` reuses the EXACT
    # SAME tokenizer instance we just rendered the prompt with -- never a
    # second, independently-resolved tokenizer that could silently diverge
    # from the one `find_think_span` below is called with.
    result = run_fullkv_worker(
        config,
        manifest,
        _load_model=_load_model,
        _load_tokenizer=lambda: tokenizer,
        _fresh_cache_factory=_fresh_cache_factory,
        _cuda=_cuda,
        _device=_device,
        _clock=_clock,
        _progress=_progress,
    )

    observed_prompt_token_ids_sha256 = result.get("prefill_token_ids_sha256")
    if observed_prompt_token_ids_sha256 is None:
        raise ValueError("canonical FullKV worker did not export observed prefill_token_ids_sha256")
    runtime_identity = result.get("runtime_identity")
    if not isinstance(runtime_identity, dict):
        raise ValueError("canonical FullKV worker did not export runtime_identity")
    resolved_model_revision = runtime_identity.get("resolved_model_revision")
    resolved_tokenizer_revision = runtime_identity.get("resolved_tokenizer_revision")
    if not isinstance(resolved_model_revision, str) or not isinstance(resolved_tokenizer_revision, str):
        raise ValueError("canonical FullKV worker did not resolve model/tokenizer revisions")

    open_marker_ids = tokenizer.encode("<think>", add_special_tokens=False)
    close_marker_ids = tokenizer.encode("</think>", add_special_tokens=False)
    generated_token_ids = list(result["natural_generated_token_ids"])
    think_span = find_think_span(prompt_token_ids, generated_token_ids, open_marker_ids, close_marker_ids)

    runtime_generation_config = _worker_observed_generation_config(config)

    payload: dict[str, Any] = {
        "worker_result_schema_version": FULLKV_WORKER_RESULT_R3_SCHEMA_VERSION,
        "role": "fullkv",
        "model_name": config.model.name,
        "model_revision": resolved_model_revision,
        "requested_model_revision": runtime_identity["requested_model_revision"],
        "model_revision_match": runtime_identity["model_revision_match"],
        "tokenizer_name": config.model.tokenizer_name,
        "tokenizer_revision": resolved_tokenizer_revision,
        "requested_tokenizer_revision": runtime_identity["requested_tokenizer_revision"],
        "tokenizer_revision_match": runtime_identity["tokenizer_revision_match"],
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "source_example_index": candidate.source_example_index,
        "unique_id": candidate.unique_id,
        "raw_row_sha256": candidate.raw_row_sha256,
        "problem_sha256": candidate.problem_sha256,
        "gold_answer_sha256": candidate.gold_answer_sha256,
        "expected_prompt_token_ids_sha256": expected_prompt_token_ids_sha256,
        "observed_prompt_token_ids_sha256": observed_prompt_token_ids_sha256,
        "prompt_token_count": result["prompt_token_count"],
        "natural_generated_token_ids": generated_token_ids,
        "generated_token_ids_sha256": sha256_int_ids(generated_token_ids),
        "natural_answer": result["natural_answer"],
        "natural_answer_status": result["natural_answer_status"],
        "cap_hit": result["cap_hit"],
        "think_parse_status": think_span.think_parse_status,
        "think_start_index": think_span.think_start_index,
        "think_end_index": think_span.think_end_index,
        "generation_prompt_preopened_think": think_span.generation_prompt_preopened_think,
        "requested_device": _device,
        "parameter_placement_evidence": result["parameter_placement"],
        "actual_batch_size": result["batch_size"],
        "timing_evidence": result["timing_evidence"],
        "memory_phase_evidence": result["memory_phase_evidence"],
        "wall_seconds": result["wall_seconds"],
        "runtime_generation_config": runtime_generation_config,
        "worker_generation_config_sha256": sha256_json(runtime_generation_config),
        "worker_config_sha256": canonical_config_hash(config),
        "software_versions": result["software_versions"],
    }

    return FullKVWorkerResultR3.model_validate(payload)
