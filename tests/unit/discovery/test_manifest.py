import json

import pytest
from pydantic import ValidationError

from kvcot.discovery.discovery_config import MATH500_DATASET_REPO, MATH500_DATASET_REVISION
from kvcot.discovery.manifest import (
    DEFAULT_MANIFEST_PATH,
    B2AOneExampleManifest,
    ChatTemplateRenderingConfig,
    load_b2a_one_example_manifest,
)

_RESOLVED_PROMPT_KWARGS = dict(
    prompt_token_ids_sha256="1" * 64,
    tokenizer_revision_used_for_prompt_hash="a" * 40,
    rendered_user_message_sha256="2" * 64,
    chat_template_source_sha256="3" * 64,
    chat_message_payload_sha256="4" * 64,
    prompt_rendering_config=ChatTemplateRenderingConfig(
        message_roles=("user",), add_generation_prompt=True, tokenize=True, add_special_tokens_note="n/a",
    ),
    prompt_token_count=3,
    prompt_token_ids=(1, 2, 3),
)

VALID_KWARGS = dict(
    dataset_repo=MATH500_DATASET_REPO,
    dataset_config="default",
    dataset_split="test",
    dataset_revision=MATH500_DATASET_REVISION,
    example_index=0,
    unique_id="test/precalculus/807.json",
    raw_content_hash="0" * 64,
    gold_answer="42",
)


def test_frozen_manifest_file_loads_and_validates():
    manifest = load_b2a_one_example_manifest()
    assert manifest.dataset_repo == MATH500_DATASET_REPO
    assert manifest.dataset_revision == MATH500_DATASET_REVISION
    assert manifest.example_index == 0
    assert len(manifest.raw_content_hash) == 64


def test_frozen_manifest_file_is_at_the_documented_default_path():
    assert DEFAULT_MANIFEST_PATH.exists()


def test_prompt_identity_is_resolved_by_prepare_b2a_manifest():
    """B1B-R3 §5: `kvcot prepare-b2a-manifest --execute` resolved this
    repository's committed manifest for real (live tokenizer, pinned
    revision) -- the honest state changed from B1B-R2's `None` to a real,
    reproducible hash, not a fabricated placeholder."""
    manifest = load_b2a_one_example_manifest()
    assert manifest.prompt_identity_is_resolved is True
    assert manifest.prompt_token_ids_sha256 is not None
    assert manifest.tokenizer_revision_used_for_prompt_hash is not None
    assert manifest.prompt_token_count == len(manifest.prompt_token_ids)
    assert manifest.prompt_token_count > 0


def test_manifest_hash_is_stable_and_sensitive_to_changes():
    a = B2AOneExampleManifest(**VALID_KWARGS)
    b = B2AOneExampleManifest(**VALID_KWARGS)
    assert a.manifest_hash() == b.manifest_hash()

    changed = B2AOneExampleManifest(**{**VALID_KWARGS, "example_index": 1})
    assert changed.manifest_hash() != a.manifest_hash()


def test_manifest_hash_changes_when_prompt_identity_gets_resolved():
    unresolved = B2AOneExampleManifest(**VALID_KWARGS)
    resolved = B2AOneExampleManifest(**VALID_KWARGS, **_RESOLVED_PROMPT_KWARGS)
    assert unresolved.manifest_hash() != resolved.manifest_hash()
    assert resolved.prompt_identity_is_resolved is True


def test_prompt_identity_fields_must_resolve_all_together():
    with pytest.raises(ValidationError, match="all-together"):
        B2AOneExampleManifest(
            **VALID_KWARGS,
            prompt_token_ids_sha256="1" * 64,
            tokenizer_revision_used_for_prompt_hash="a" * 40,
        )


def test_prompt_token_ids_length_must_match_prompt_token_count():
    with pytest.raises(ValidationError, match="must agree"):
        B2AOneExampleManifest(
            **VALID_KWARGS,
            **{**_RESOLVED_PROMPT_KWARGS, "prompt_token_count": 99},
        )


def test_dataset_revision_rejects_non_full_hash():
    with pytest.raises(ValidationError):
        B2AOneExampleManifest(**{**VALID_KWARGS, "dataset_revision": "main"})


def test_raw_content_hash_rejects_malformed_hex():
    with pytest.raises(ValidationError):
        B2AOneExampleManifest(**{**VALID_KWARGS, "raw_content_hash": "not-hex"})


def test_load_rejects_manifest_disagreeing_with_frozen_dataset_repo(tmp_path):
    bad = {**VALID_KWARGS, "dataset_repo": "someone-else/MATH-500"}
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_b2a_one_example_manifest(path)


def test_load_rejects_manifest_disagreeing_with_frozen_dataset_revision(tmp_path):
    bad = {**VALID_KWARGS, "dataset_revision": "a" * 40}
    path = tmp_path / "bad_manifest.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_b2a_one_example_manifest(path)


def test_manifest_never_imports_torch():
    import kvcot.discovery.manifest as mod

    assert "torch" not in mod.__dict__
