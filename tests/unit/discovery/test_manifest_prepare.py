"""CPU-only, network-free tests for `kvcot.discovery.manifest_prepare`
(B1B-R3 §5/§18). `_fetch_pinned_dataset_row` and `_render_and_tokenize` are
monkeypatched at the module level -- no real HTTP request or tokenizer load
happens in this test file, so it runs offline and without `transformers`
installed."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvcot.discovery import manifest_prepare as mp
from kvcot.discovery.discovery_config import load_discovery_config
from kvcot.discovery.manifest import B2AOneExampleManifest

CONFIG_PATH = "configs/discovery/llama8b_math500_b1024.yaml"

VALID_ROW = {
    "problem": "What is 2+2?",
    "solution": "4",
    "answer": "4",
    "subject": "Algebra",
    "level": 1,
    "unique_id": "test/algebra/1.json",
}


class _FakeTokenizer:
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3, 4, 5]


def _patch_network(monkeypatch, row=None, unique_id_override=None):
    row = dict(row or VALID_ROW)
    if unique_id_override:
        row["unique_id"] = unique_id_override

    def fake_fetch(dataset_repo, dataset_revision, example_index):
        return mp.FetchedDatasetRow(row=row, raw_content_hash="a" * 64)

    def fake_render_and_tokenize(row_dict, tokenizer_name, tokenizer_revision):
        tokenizer = _FakeTokenizer()
        user_message = f"Problem: {row_dict['problem']}"
        messages = [{"role": "user", "content": user_message}]
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return tokenizer, user_message, messages, token_ids

    monkeypatch.setattr(mp, "_fetch_pinned_dataset_row", fake_fetch)
    monkeypatch.setattr(mp, "_render_and_tokenize", fake_render_and_tokenize)


def _config():
    return load_discovery_config(CONFIG_PATH)


def test_dry_run_plan_reports_existing_manifest_state():
    plan = mp.build_plan(_config())
    assert plan.dataset_repo == "HuggingFaceH4/MATH-500"
    assert plan.example_index == 0
    assert plan.existing_manifest_is_prompt_resolved is True  # this repo's committed manifest is resolved


def test_resolve_prompt_identity_produces_nonempty_tokens(monkeypatch):
    _patch_network(monkeypatch)
    plan = mp.build_plan(_config())
    resolved = mp.resolve_prompt_identity(plan)
    assert len(resolved.prompt_token_ids) > 0
    assert resolved.prompt_token_count == len(resolved.prompt_token_ids)
    assert len(resolved.prompt_token_ids_sha256) == 64


def test_prepare_manifest_refuses_wrong_schema_columns(monkeypatch, tmp_path):
    bad_row = {**VALID_ROW}
    del bad_row["level"]
    _patch_network(monkeypatch, row=bad_row)
    manifest_path = tmp_path / "manifest.json"
    with pytest.raises(mp.ManifestPreparationError, match="unexpected columns"):
        mp.prepare_manifest(_config(), manifest_path, force=True)


def test_prepare_manifest_rejects_missing_chat_template(monkeypatch, tmp_path):
    def fake_fetch(dataset_repo, dataset_revision, example_index):
        return mp.FetchedDatasetRow(row=VALID_ROW, raw_content_hash="a" * 64)

    def fake_render_and_tokenize_no_template(row_dict, tokenizer_name, tokenizer_revision):
        raise mp.ManifestPreparationError(
            f"tokenizer {tokenizer_name}@{tokenizer_revision} has no chat_template -- refusing to invent one."
        )

    monkeypatch.setattr(mp, "_fetch_pinned_dataset_row", fake_fetch)
    monkeypatch.setattr(mp, "_render_and_tokenize", fake_render_and_tokenize_no_template)

    manifest_path = tmp_path / "manifest.json"
    with pytest.raises(mp.ManifestPreparationError, match="no chat_template"):
        mp.prepare_manifest(_config(), manifest_path, force=True)


def test_prepare_manifest_writes_atomically_and_validates(monkeypatch, tmp_path):
    _patch_network(monkeypatch)
    manifest_path = tmp_path / "manifest.json"

    result = mp.prepare_manifest(_config(), manifest_path, force=True)

    assert manifest_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert result.prompt_identity_is_resolved is True
    on_disk = B2AOneExampleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert on_disk.manifest_hash() == result.manifest_hash()


def test_prepare_manifest_refuses_overwrite_without_force(monkeypatch, tmp_path):
    _patch_network(monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    mp.prepare_manifest(_config(), manifest_path, force=True)

    with pytest.raises(mp.ManifestPreparationRefused):
        mp.prepare_manifest(_config(), manifest_path, force=False)


def test_prepare_manifest_force_refuses_to_change_unique_id(monkeypatch, tmp_path):
    _patch_network(monkeypatch)
    manifest_path = tmp_path / "manifest.json"
    mp.prepare_manifest(_config(), manifest_path, force=True)

    _patch_network(monkeypatch, unique_id_override="test/algebra/DIFFERENT.json")
    with pytest.raises(mp.ManifestPreparationError, match="unique_id"):
        mp.prepare_manifest(_config(), manifest_path, force=True)


def test_prepare_manifest_force_allows_correcting_raw_content_hash(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"

    def fake_fetch_v1(dataset_repo, dataset_revision, example_index):
        return mp.FetchedDatasetRow(row=VALID_ROW, raw_content_hash="a" * 64)

    def fake_render_and_tokenize(row_dict, tokenizer_name, tokenizer_revision):
        tokenizer = _FakeTokenizer()
        messages = [{"role": "user", "content": "x"}]
        return tokenizer, "x", messages, [1, 2, 3]

    monkeypatch.setattr(mp, "_fetch_pinned_dataset_row", fake_fetch_v1)
    monkeypatch.setattr(mp, "_render_and_tokenize", fake_render_and_tokenize)
    mp.prepare_manifest(_config(), manifest_path, force=True)

    def fake_fetch_v2(dataset_repo, dataset_revision, example_index):
        return mp.FetchedDatasetRow(row=VALID_ROW, raw_content_hash="b" * 64)  # corrected hash

    monkeypatch.setattr(mp, "_fetch_pinned_dataset_row", fake_fetch_v2)
    result = mp.prepare_manifest(_config(), manifest_path, force=True)
    assert result.raw_content_hash == "b" * 64


def test_snapshot_weight_shaped_files_empty_for_absent_cache():
    # B1B-R4 §15: the split guard's pure observation function, exercised
    # against an environment with no matching cache entries at all.
    assert mp._snapshot_weight_shaped_files("nonexistent/model", "0" * 40) == frozenset()


def test_assert_no_new_weight_files_passes_when_nothing_changed():
    before = frozenset({"model.safetensors"})  # pre-existing weight file
    after = frozenset({"model.safetensors"})  # unchanged
    mp._assert_no_new_weight_files_introduced(before, after)  # should not raise


def test_assert_no_new_weight_files_fails_only_on_a_genuinely_new_file():
    before = frozenset()
    after = frozenset({"model.safetensors"})
    with pytest.raises(mp.ManifestPreparationError, match="introduced new weight-shaped"):
        mp._assert_no_new_weight_files_introduced(before, after)


def test_render_and_tokenize_does_not_inspect_or_reject_pre_existing_weights(monkeypatch):
    """B1B-R4 §15 regression: `_render_and_tokenize` is shared by generic
    B2A prompt-identity re-verification -- it must never call the
    weight-cache guard at all, so a GPU host with pre-existing weights for
    the pinned revision never fails generic prompt rendering."""
    called = {"snapshot": False}

    def _boom(*args, **kwargs):
        called["snapshot"] = True
        raise AssertionError("_render_and_tokenize must never call the weight-cache snapshot guard")

    monkeypatch.setattr(mp, "_snapshot_weight_shaped_files", _boom)

    class _FakeTok:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [1, 2, 3]

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _FakeTok())
    mp._render_and_tokenize({"problem": "2+2=?"}, "some/model", "0" * 40)
    assert called["snapshot"] is False


def test_resolve_prompt_identity_detects_a_genuinely_new_weight_file(monkeypatch):
    """The guard DOES fire when `resolve_prompt_identity`'s own call
    introduces a new weight-shaped file -- proven by faking the before/after
    snapshots to differ."""
    snapshots = iter([frozenset(), frozenset({"new_model.safetensors"})])
    monkeypatch.setattr(mp, "_snapshot_weight_shaped_files", lambda *a, **k: next(snapshots))
    monkeypatch.setattr(
        mp, "_fetch_pinned_dataset_row",
        lambda *a, **k: mp.FetchedDatasetRow(row=VALID_ROW, raw_content_hash="a" * 64),
    )

    class _FakeTok:
        chat_template = "template"

    monkeypatch.setattr(
        mp, "_render_and_tokenize", lambda row, name, rev: (_FakeTok(), "x", [{"role": "user", "content": "x"}], [1, 2, 3])
    )
    plan = mp.PreparedManifestPlan(
        dataset_repo="HuggingFaceH4/MATH-500", dataset_config="default", dataset_split="test",
        dataset_revision="a" * 40, example_index=0, tokenizer_name="some/model", tokenizer_revision="0" * 40,
        existing_manifest_path=mp.DEFAULT_MANIFEST_PATH, existing_manifest_is_prompt_resolved=False, force=False,
    )
    with pytest.raises(mp.ManifestPreparationError, match="introduced new weight-shaped"):
        mp.resolve_prompt_identity(plan)
