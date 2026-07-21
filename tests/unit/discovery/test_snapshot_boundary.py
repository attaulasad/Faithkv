from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvcot.discovery.snapshot_boundary import (
    SnapshotBoundaryError,
    resolve_local_snapshot,
    verify_snapshot_evidence_raw,
)
from kvcot.discovery.strict_device import StrictDeviceError, verify_single_rtx3090

SHA = "a" * 40


def _mock_cache(monkeypatch, tmp_path: Path, repo="org/model"):
    revision = SimpleNamespace(commit_hash=SHA, snapshot_path=tmp_path)
    cache = SimpleNamespace(repos=[SimpleNamespace(repo_id=repo, revisions=[revision])])
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", lambda cache_dir=None: cache)


def test_tokenizer_snapshot_requires_exact_public_cache_identity(monkeypatch, tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    _mock_cache(monkeypatch, tmp_path)
    result = resolve_local_snapshot("org/model", SHA, "tokenizer", cache_dir=tmp_path.parent)
    assert result.resolved_revision == SHA
    assert result.local_files_only is True


def test_model_snapshot_index_fails_on_missing_shard(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": "model-00001-of-00002.safetensors"}}), encoding="utf-8"
    )
    _mock_cache(monkeypatch, tmp_path)
    with pytest.raises(SnapshotBoundaryError, match="missing shards"):
        resolve_local_snapshot("org/model", SHA, "model", cache_dir=tmp_path.parent)


def test_floating_revision_and_incomplete_files_fail_closed(monkeypatch, tmp_path):
    with pytest.raises(SnapshotBoundaryError, match="full lowercase immutable"):
        resolve_local_snapshot("org/model", "main", "tokenizer", cache_dir=tmp_path.parent)
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json.incomplete").write_text("", encoding="utf-8")
    _mock_cache(monkeypatch, tmp_path)
    with pytest.raises(SnapshotBoundaryError, match="incomplete"):
        resolve_local_snapshot("org/model", SHA, "tokenizer", cache_dir=tmp_path.parent)


class FakeCuda:
    class cudnn:
        @staticmethod
        def version(): return 90100

    def __init__(self, count=1, name="NVIDIA GeForce RTX 3090"):
        self.count = count
        self.name = name

    def device_count(self): return self.count
    def current_device(self): return 0
    def get_device_properties(self, index): return SimpleNamespace(name=self.name, total_memory=24 * 1024**3)
    def get_device_capability(self, index): return (8, 6)


def test_strict_device_preflight_records_complete_single_3090_identity():
    evidence = verify_single_rtx3090(
        FakeCuda(), torch_module=SimpleNamespace(version=SimpleNamespace(cuda="12.8")),
        driver_version_fn=lambda: "570.00",
    )
    assert evidence.policy_satisfied is True
    assert evidence.visible_gpu_count == 1
    assert evidence.compute_capability == (8, 6)


@pytest.mark.parametrize("cuda,match", [(FakeCuda(count=2), "exactly one"), (FakeCuda(name="A100"), "RTX 3090")])
def test_strict_device_preflight_rejects_wrong_hardware(cuda, match):
    with pytest.raises(StrictDeviceError, match=match):
        verify_single_rtx3090(
            cuda, torch_module=SimpleNamespace(version=SimpleNamespace(cuda="12.8")),
            driver_version_fn=lambda: "570.00",
        )


# --------------------------------------------------------------------------
# Independent-audit Gate H4.4/H4.6: `verify_snapshot_evidence_raw` must
# re-validate the CONTENT of a worker-reported snapshot dict, never trust a
# bare `verified`/`resolved_revision` pair.
# --------------------------------------------------------------------------

MODEL_SHA = "b" * 40


def _valid_model_evidence(**overrides) -> dict:
    base = dict(
        repository_id="org/model", requested_revision=MODEL_SHA, resolved_revision=MODEL_SHA,
        asset_type="model", local_path="/fake/path", files=["config.json", "model.safetensors"],
        total_bytes=1024, required_free_bytes=0, free_bytes=1_000_000_000, local_files_only=True,
    )
    base.update(overrides)
    return base


def _valid_tokenizer_evidence(**overrides) -> dict:
    base = dict(
        repository_id="org/tokenizer", requested_revision=MODEL_SHA, resolved_revision=MODEL_SHA,
        asset_type="tokenizer", local_path="/fake/path", files=["tokenizer_config.json", "tokenizer.json"],
        total_bytes=512, required_free_bytes=0, free_bytes=1_000_000_000, local_files_only=True,
    )
    base.update(overrides)
    return base


def test_snapshot_evidence_raw_passes_for_a_valid_model_report():
    assert verify_snapshot_evidence_raw(
        _valid_model_evidence(), expected_repository_id="org/model", expected_revision=MODEL_SHA,
        asset_type="model",
    ) is True


def test_snapshot_evidence_raw_passes_for_a_valid_tokenizer_report():
    assert verify_snapshot_evidence_raw(
        _valid_tokenizer_evidence(), expected_repository_id="org/tokenizer", expected_revision=MODEL_SHA,
        asset_type="tokenizer",
    ) is True


def test_snapshot_evidence_raw_fails_on_none_or_non_dict():
    assert verify_snapshot_evidence_raw(
        None, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False
    assert verify_snapshot_evidence_raw(
        "not a dict", expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_wrong_repository():
    bad = _valid_model_evidence(repository_id="org/wrong-model")
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_wrong_asset_type():
    bad = _valid_model_evidence(asset_type="tokenizer")
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_requested_resolved_mismatch():
    """A `requested_revision` != `resolved_revision` is a FLOATING revision
    -- the whole point of pinning is that the request and the resolution
    agree exactly."""
    bad = _valid_model_evidence(resolved_revision="c" * 40)
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_requested_not_matching_expected():
    bad = _valid_model_evidence(requested_revision="d" * 40, resolved_revision="d" * 40)
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_non_full_sha_revision():
    bad = _valid_model_evidence(requested_revision="main", resolved_revision="main")
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision="main", asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_when_local_files_only_is_not_true():
    bad = _valid_model_evidence(local_files_only=False)
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_empty_file_inventory():
    bad = _valid_model_evidence(files=[])
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_incomplete_or_lock_files():
    bad = _valid_model_evidence(files=["config.json", "model.safetensors.incomplete"])
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_missing_model_config():
    bad = _valid_model_evidence(files=["model.safetensors"])
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_missing_model_weights():
    bad = _valid_model_evidence(files=["config.json"])
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False


def test_snapshot_evidence_raw_fails_on_missing_tokenizer_files():
    bad = _valid_tokenizer_evidence(files=["tokenizer_config.json"])  # no vocabulary file
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/tokenizer", expected_revision=MODEL_SHA, asset_type="tokenizer",
    ) is False

    bad2 = _valid_tokenizer_evidence(files=["tokenizer.json"])  # no tokenizer_config.json
    assert verify_snapshot_evidence_raw(
        bad2, expected_repository_id="org/tokenizer", expected_revision=MODEL_SHA, asset_type="tokenizer",
    ) is False


def test_snapshot_evidence_raw_fails_on_non_positive_total_bytes():
    bad = _valid_model_evidence(total_bytes=0)
    assert verify_snapshot_evidence_raw(
        bad, expected_repository_id="org/model", expected_revision=MODEL_SHA, asset_type="model",
    ) is False
