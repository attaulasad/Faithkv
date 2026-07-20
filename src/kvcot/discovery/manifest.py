"""The frozen B2A one-example manifest (`docs/B1B_R2_REAL_MODEL_BOUNDARY_
AND_B2A_PREFLIGHT.md` §8). Loads and validates
`configs/discovery/b2a_one_example_manifest.json` -- pure Python (pydantic
+ json only), no torch, usable from `kvcot plan-discovery --dry-run` and
`kvcot b2a-calibrate --dry-run`.

Every identity field that CAN be resolved without a live tokenizer/model
(dataset repo/config/split/revision, example index, the dataset row's own
unique id, a content hash of the raw row) is frozen here, verified directly
against the Hugging Face Hub API on 2026-07-20 -- never guessed. Two fields
genuinely CANNOT be resolved by CPU-only code in this repository:
`prompt_token_ids_sha256` (requires the live tokenizer's chat template) and
`tokenizer_revision_used_for_prompt_hash` -- both are `None` here, and
`manifest_is_fully_resolved` reports that gap explicitly rather than
fabricating a placeholder hash. A future, separately-authorized B2A
execution is the only thing that can resolve them (by actually running the
tokenizer once), at which point this file would be updated with the real
values before that run is allowed to proceed.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, field_validator

from kvcot.discovery.discovery_config import (
    MATH500_DATASET_REPO,
    MATH500_DATASET_REVISION,
    validate_full_revision,
)
from kvcot.utils.hashing import sha256_json

DEFAULT_MANIFEST_PATH = Path("configs/discovery/b2a_one_example_manifest.json")


class B2AOneExampleManifest(BaseModel):
    dataset_repo: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str
    example_index: int
    unique_id: str
    raw_content_hash: str

    # Genuinely unresolved without a live tokenizer -- `None` is the
    # accurate, honest state, not a bug in this schema. A future B2A
    # execution must refuse to run while either is `None`
    # (`kvcot.discovery.b2a_contract`'s `prompt_token_hash_match` /
    # `tokenizer_revision_match` conditions).
    prompt_token_ids_sha256: str | None = None
    tokenizer_revision_used_for_prompt_hash: str | None = None

    @field_validator("dataset_revision")
    @classmethod
    def _dataset_revision_full(cls, v: str) -> str:
        return validate_full_revision(v, "dataset_revision")

    @field_validator("raw_content_hash", "prompt_token_ids_sha256")
    @classmethod
    def _sha256_hex_if_present(cls, v: str | None, info) -> str | None:
        if v is None:
            return None
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(f"{info.field_name} must be 64 lowercase hex characters, got {v!r}")
        return v

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
