"""Runtime-derived evidence for the B2A gate (B1B-R4 §9/§10/§11). Every
field here is read back from an ACTUAL loaded model/tokenizer/input tensor
-- never a hard-coded literal (`every_parameter_on_cuda=True`,
`batch_size=1`, `generation_config_hash_match=True`) standing in for a
runtime observation. Where a genuine runtime read-back is unavailable, the
corresponding field is `None`/unresolved and the caller must treat the
matching gate condition as failed, never silently substitute a literal
`True`.

Pure Python -- imports nothing from torch/transformers at module scope, so
this module is safe to import from CPU tests; the functions that DO touch a
live model/tokenizer only duck-type against `.named_parameters()`/
`.config`/`._commit_hash`, matching `kvcot.discovery.no_offload`'s existing
duck-typed discipline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcot.utils.hashing import sha256_json


@dataclass(frozen=True)
class RuntimeGenerationRecord:
    """The ACTUAL generation configuration and call-shape policy used by a
    worker's natural-generation loop -- every field is either a frozen
    constant of this repository's greedy-decoding architecture (never a
    sampling parameter) or read back from the real call (`batch_size`,
    `prompt_token_count`)."""

    generation_mode: str
    do_sample: bool
    temperature: float | None
    top_p: float | None
    batch_size: int
    max_new_tokens: int
    eos_token_id: int | None
    eos_append_feed_policy: str
    one_prefill_policy: str
    single_token_decode_policy: str
    attention_backend: str
    cache_implementation: str
    framework_seed: int
    prompt_token_count: int

    def canonical_hash(self) -> str:
        return sha256_json(self.__dict__)


def build_runtime_generation_record(
    *,
    batch_size: int,
    max_new_tokens: int,
    eos_token_id: int | None,
    attention_backend: str,
    framework_seed: int,
    prompt_token_count: int,
) -> RuntimeGenerationRecord:
    """The one place a `RuntimeGenerationRecord` is constructed -- FullKV
    and R-KV workers both call this with their own observed `batch_size`/
    `prompt_token_count`, so `generation_config_hash_match` (B1B-R4 §10) is
    a real comparison of two independently-built hashes, never a literal
    `True`."""
    return RuntimeGenerationRecord(
        generation_mode="greedy",
        do_sample=False,
        temperature=None,
        top_p=None,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        eos_append_feed_policy="eos_never_appended_to_full_token_ids_or_fed_to_the_next_forward_call",
        one_prefill_policy="exactly_one_prefill_call_for_the_complete_prompt",
        single_token_decode_policy="exactly_one_decode_call_per_generated_non_eos_token",
        attention_backend=attention_backend,
        cache_implementation="DynamicCache",
        framework_seed=framework_seed,
        prompt_token_count=prompt_token_count,
    )


@dataclass(frozen=True)
class ParameterPlacementEvidence:
    unique_device_types: tuple[str, ...]
    every_parameter_on_cuda: bool
    hf_device_map: dict[str, str] | None
    no_offload_verified: bool
    parameter_count: int


def derive_parameter_placement(model: Any) -> ParameterPlacementEvidence:
    """Walks every named parameter (never reads `model.device`, which
    cannot detect a partially-offloaded `device_map="auto"` load -- see
    `kvcot.discovery.no_offload`'s docstring for why) and records the
    resulting device-type set plus, when present, `hf_device_map` --
    `every_parameter_on_cuda`/`no_offload_verified` are DERIVED from this
    walk, never asserted independently."""
    devices: set[str] = set()
    n_params = 0
    for _, param in model.named_parameters():
        devices.add(param.device.type)
        n_params += 1

    device_map = getattr(model, "hf_device_map", None)
    device_map_str = {str(k): str(v) for k, v in device_map.items()} if device_map else None

    every_cuda = n_params > 0 and devices == {"cuda"}
    no_offload = every_cuda and (
        device_map is None or all(str(v).lower() not in ("cpu", "disk", "meta") for v in device_map.values())
    )
    return ParameterPlacementEvidence(
        unique_device_types=tuple(sorted(devices)),
        every_parameter_on_cuda=every_cuda,
        hf_device_map=device_map_str,
        no_offload_verified=no_offload,
        parameter_count=n_params,
    )


def derive_batch_size_from_input_ids(input_ids: Any) -> int:
    """Read the ACTUAL batch dimension off a real input tensor -- never a
    hard-coded `1`. `input_ids.shape[0]` is the batch axis for every call
    shape this repository's decode loop uses (`kvcot.generation.decode
    .prefill`/`decode_step`, `kvcot.discovery.real_model_adapter`)."""
    return int(input_ids.shape[0])


def read_resolved_model_revision(model: Any) -> str | None:
    """`transformers.PretrainedConfig.from_pretrained` sets `config
    ._commit_hash` to the ACTUAL resolved commit SHA extracted from the
    downloaded/cached file path (`transformers.utils.hub
    .extract_commit_hash`) whenever the model was loaded from the Hub --
    read back here, never re-derived from the requested `revision=`
    argument. `None` (genuinely unresolved) if the attribute is absent,
    e.g. a config that did not come from `from_pretrained` at all."""
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def read_resolved_tokenizer_revision(tokenizer: Any) -> str | None:
    """Same mechanism as `read_resolved_model_revision`, for
    `transformers.PreTrainedTokenizerBase._commit_hash`."""
    return getattr(tokenizer, "_commit_hash", None)


@dataclass(frozen=True)
class RuntimeIdentityEvidence:
    requested_model_revision: str
    resolved_model_revision: str | None
    model_revision_match: bool
    requested_tokenizer_revision: str
    resolved_tokenizer_revision: str | None
    tokenizer_revision_match: bool


def derive_runtime_identity(
    *,
    model: Any,
    tokenizer: Any,
    requested_model_revision: str,
    requested_tokenizer_revision: str,
    verified_model_revision: str | None = None,
    verified_tokenizer_revision: str | None = None,
) -> RuntimeIdentityEvidence:
    """`*_match` is `True` ONLY when a resolved revision was genuinely read
    back AND it equals the requested one -- an unresolved (`None`) readback
    is `False`, never silently treated as a match (B1B-R4 §9: "If a
    resolved runtime revision cannot be established, the corresponding
    evidence must be false or unresolved and the gate must fail")."""
    resolved_model = verified_model_revision or read_resolved_model_revision(model)
    resolved_tokenizer = verified_tokenizer_revision or read_resolved_tokenizer_revision(tokenizer)
    return RuntimeIdentityEvidence(
        requested_model_revision=requested_model_revision,
        resolved_model_revision=resolved_model,
        model_revision_match=resolved_model is not None and resolved_model == requested_model_revision,
        requested_tokenizer_revision=requested_tokenizer_revision,
        resolved_tokenizer_revision=resolved_tokenizer,
        tokenizer_revision_match=resolved_tokenizer is not None and resolved_tokenizer == requested_tokenizer_revision,
    )


@dataclass(frozen=True)
class MemoryEvidence:
    """B1B-R4 §14: gates on `max(peak_allocated, peak_reserved)`, and both
    workers reset peak-memory stats at the SAME point (after model/tokenizer
    load, before measured inference) -- `reset_point` records which
    convention was actually used, so a future change to the reset point is
    visible in the artifact rather than silently assumed identical."""

    allocated_before_reset_bytes: int
    reserved_before_reset_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    reset_point: str

    @property
    def peak_tracked_bytes(self) -> int:
        return max(self.peak_allocated_bytes, self.peak_reserved_bytes)


RESET_POINT_AFTER_LOAD_BEFORE_INFERENCE = "after_model_and_tokenizer_load_before_measured_inference"
