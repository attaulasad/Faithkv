from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvcot.discovery.snapshot_boundary import SnapshotBoundaryError, resolve_local_snapshot
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
