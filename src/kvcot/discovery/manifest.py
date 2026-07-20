"""The frozen B2A one-example manifest (`docs/B1B_R2_REAL_MODEL_BOUNDARY_
AND_B2A_PREFLIGHT.md` §8). Loads and validates
`configs/discovery/b2a_one_example_manifest.json` -- pure Python (pydantic
+ json only), no torch, usable from `kvcot plan-discovery --dry-run` and
`kvcot b2a-calibrate --dry-run`.

Every identity field that CAN be resolved without a live tokenizer/model
(dataset repo/config/split/revision, example index, the dataset row's own
unique id, a content hash of the raw row) is frozen here, verified directly
against the Hugging Face Hub API. The prompt-identity fields
(`prompt_token_ids_sha256`, `tokenizer_revision_used_for_prompt_hash`, and
the rendering-provenance fields alongside them) require a live tokenizer's
chat template -- CPU-only, no model weights, so they ARE resolvable by
CPU-side code (B1B-R3 §5, `kvcot.discovery.manifest_prepare`), just not by
this module itself, which stays pure Python (pydantic + json only, no
`transformers` import) so it remains usable from every `--dry-run` path.
`prompt_identity_is_resolved` reports whether that resolution has happened
yet; a manifest with `None` there has not, and `kvcot.discovery.b2a_execute`
refuses to run against it.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

from kvcot.discovery.discovery_config import (
    MATH500_DATASET_REPO,
    MATH500_DATASET_REVISION,
    validate_full_revision,
)
from kvcot.utils.hashing import sha256_json

DEFAULT_MANIFEST_PATH = Path("configs/discovery/b2a_one_example_manifest.json")


class ChatTemplateRenderingConfig(BaseModel):
    """Every argument that affects `tokenizer.apply_chat_template`'s output,
    frozen explicitly (B1B-R3 §5.7) -- reused verbatim from the exact call
    shape `kvcot.cli.cmd_generate` already uses for the primary pipeline
    (`tokenizer.apply_chat_template([{"role": "user", "content": ...}],
    tokenize=True, add_generation_prompt=True)`), never a second,
    independently-invented rendering convention for the discovery track."""

    message_roles: tuple[str, ...]
    add_generation_prompt: bool
    tokenize: bool
    add_special_tokens_note: str


class B2AOneExampleManifest(BaseModel):
    dataset_repo: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str
    example_index: int
    unique_id: str
    raw_content_hash: str
    # The dataset row's own gold answer text (MATH-500's `answer` column),
    # frozen at the same time as `raw_content_hash` (B1B-R3 §7) -- the ONLY
    # source `kvcot.discovery.math500_verification.Math500AnswerVerifier`
    # compares generated answers against; never a second, independently
    # re-fetched or hand-typed gold value.
    gold_answer: str

    # Genuinely unresolved without a live tokenizer -- `None` is the
    # accurate, honest state, not a bug in this schema. A future B2A
    # execution must refuse to run while either is `None`
    # (`kvcot.discovery.b2a_contract`'s `prompt_token_hash_match` /
    # `tokenizer_revision_match` conditions).
    prompt_token_ids_sha256: str | None = None
    tokenizer_revision_used_for_prompt_hash: str | None = None

    # Resolved together with the two fields above, by
    # `kvcot.discovery.manifest_prepare.prepare_manifest` only -- every one
    # of these is `None` until the prompt identity is (B1B-R3 §5.9): the
    # rendered user-message hash, the tokenizer's own chat-template source
    # hash, the serialized chat-message payload hash, the exact rendering
    # configuration used, and the total token count. `prompt_token_ids`
    # optionally carries the exact array (B1B-R3 §5: "Prefer storing the
    # exact prompt token-ID array... so the GPU path does not have to
    # reconstruct its primary input from an implicit convention").
    rendered_user_message_sha256: str | None = None
    chat_template_source_sha256: str | None = None
    chat_message_payload_sha256: str | None = None
    prompt_rendering_config: ChatTemplateRenderingConfig | None = None
    prompt_token_count: int | None = None
    prompt_token_ids: tuple[int, ...] | None = None

    @field_validator("dataset_revision")
    @classmethod
    def _dataset_revision_full(cls, v: str) -> str:
        return validate_full_revision(v, "dataset_revision")

    @field_validator(
        "raw_content_hash",
        "prompt_token_ids_sha256",
        "rendered_user_message_sha256",
        "chat_template_source_sha256",
        "chat_message_payload_sha256",
    )
    @classmethod
    def _sha256_hex_if_present(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(f"{info.field_name} must be 64 lowercase hex characters, got {v!r}")
        return v

    @model_validator(mode="after")
    def _prompt_identity_fields_resolve_together(self) -> "B2AOneExampleManifest":
        """Either every prompt-identity field is set, or every one of them
        is `None` -- a manifest with, say, a token hash but no recorded
        rendering config (or vice versa) is not a valid resolved OR
        unresolved state, it is a corrupted partial write (exactly what
        B1B-R3 §5's atomic-write requirement exists to prevent)."""
        fields = (
            self.prompt_token_ids_sha256,
            self.tokenizer_revision_used_for_prompt_hash,
            self.rendered_user_message_sha256,
            self.chat_template_source_sha256,
            self.chat_message_payload_sha256,
            self.prompt_rendering_config,
            self.prompt_token_count,
        )
        n_set = sum(f is not None for f in fields)
        if n_set not in (0, len(fields)):
            raise ValueError(
                "prompt-identity fields must resolve all-together or not-at-all "
                f"({n_set}/{len(fields)} set) -- a manifest cannot be partially resolved."
            )
        if self.prompt_token_ids is not None and self.prompt_token_count is not None:
            if len(self.prompt_token_ids) != self.prompt_token_count:
                raise ValueError(
                    f"prompt_token_ids has {len(self.prompt_token_ids)} entries but "
                    f"prompt_token_count={self.prompt_token_count} -- these must agree."
                )
        return self

    @property
    def prompt_identity_is_resolved(self) -> bool:
        return self.prompt_token_ids_sha256 is not None and self.tokenizer_revision_used_for_prompt_hash is not None

    def manifest_hash(self) -> str:
        """Hash over every field of this manifest (including the `None`
        prompt-identity fields, so a manifest that later gets those fields
        filled in hashes differently from this unresolved one -- the hash
        itself is evidence of exactly which fields were frozen at the
        time)."""
        return sha256_json(self.model_dump(mode="json"))


def load_b2a_one_example_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> B2AOneExampleManifest:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    manifest = B2AOneExampleManifest.model_validate(raw)
    if manifest.dataset_repo != MATH500_DATASET_REPO:
        raise ValueError(
            f"manifest dataset_repo {manifest.dataset_repo!r} does not match the frozen discovery-track "
            f"dataset repo {MATH500_DATASET_REPO!r}"
        )
    if manifest.dataset_revision != MATH500_DATASET_REVISION:
        raise ValueError(
            f"manifest dataset_revision {manifest.dataset_revision!r} does not match the frozen "
            f"discovery-track dataset revision {MATH500_DATASET_REVISION!r} -- refusing to trust a "
            "manifest that disagrees with the config's own frozen dataset identity."
        )
    return manifest
