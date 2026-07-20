import json

import pytest
from pydantic import ValidationError

from kvcot.discovery.discovery_config import MATH500_DATASET_REPO, MATH500_DATASET_REVISION
from kvcot.discovery.manifest import (
    DEFAULT_MANIFEST_PATH,
    B2AOneExampleManifest,
    load_b2a_one_example_manifest,
)

VALID_KWARGS = dict(
    dataset_repo=MATH500_DATASET_REPO,
    dataset_config="default",
    dataset_split="test",
    dataset_revision=MATH500_DATASET_REVISION,
    example_index=0,
    unique_id="test/precalculus/807.json",
    raw_content_hash="0" * 64,
)


def test_frozen_manifest_file_loads_and_validates():
    manifest = load_b2a_one_example_manifest()
    assert manifest.dataset_repo == MATH500_DATASET_REPO
    assert manifest.dataset_revision == MATH500_DATASET_REVISION
    assert manifest.example_index == 0
    assert len(manifest.raw_content_hash) == 64


def test_frozen_manifest_file_is_at_the_documented_default_path():
    assert DEFAULT_MANIFEST_PATH.exists()


def test_prompt_identity_is_not_yet_resolved():
    """Honest, expected state for this CPU-only pass: the tokenized-prompt
    hash requires a live tokenizer, which is never loaded here."""
    manifest = load_b2a_one_example_manifest()
    assert manifest.prompt_identity_is_resolved is False
    assert manifest.prompt_token_ids_sha256 is None
    assert manifest.tokenizer_revision_used_for_prompt_hash is None


def test_manifest_hash_is_stable_and_sensitive_to_changes():
    a = B2AOneExampleManifest(**VALID_KWARGS)
    b = B2AOneExampleManifest(**VALID_KWARGS)
    assert a.manifest_hash() == b.manifest_hash()

    changed = B2AOneExampleManifest(**{**VALID_KWARGS, "example_index": 1})
    assert changed.manifest_hash() != a.manifest_hash()


def test_manifest_hash_changes_when_prompt_identity_gets_resolved():
    unresolved = B2AOneExampleManifest(**VALID_KWARGS)
    resolved = B2AOneExampleManifest(
        **VALID_KWARGS,
        prompt_token_ids_sha256="1" * 64,
        tokenizer_revision_used_for_prompt_hash="a" * 40,
    )
    assert unresolved.manifest_hash() != resolved.manifest_hash()
    assert resolved.prompt_identity_is_resolved is True


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
