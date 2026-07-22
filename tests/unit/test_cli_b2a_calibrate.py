"""Structural + smoke tests for `kvcot b2a-calibrate`
(`docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md` §20, superseding
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §11/§12).

Proves `--dry-run` prints the required planning numbers/hashes, creates no
result files, and -- structurally -- can never reach `from_pretrained`,
`load_dataset`, a CUDA API, or the real R-KV patcher. Proves `--execute`
requires the flag, refuses without CUDA, and refuses on an unresolved
manifest prompt-token identity -- all without ever importing torch.cuda
machinery or a real model.

B1B-R3 resolved this repository's committed manifest for real
(`kvcot prepare-b2a-manifest --execute`) -- `--dry-run` now reports no
blockers by default; the unresolved-manifest refusal path is exercised by
monkeypatching `prompt_identity_is_resolved` to simulate that state,
rather than depending on the manifest genuinely being unresolved.
"""
from __future__ import annotations

import builtins
from argparse import Namespace
from pathlib import Path

import pytest

from kvcot.cli import build_parser, cmd_b2a_calibrate
from kvcot.discovery.manifest import B2AOneExampleManifest

DISCOVERY_CONFIG = "configs/discovery/llama8b_math500_b1024.yaml"

_DISALLOWED_MODULES = {"transformers", "datasets", "huggingface_hub", "rkv", "rkv.monkeypatch"}


@pytest.fixture(autouse=True)
def isolate_execute_attempt_artifacts(tmp_path, monkeypatch):
    """Execute-mode refusal tests must exercise artifacts without dirtying the repo."""
    from kvcot.discovery import attempt_artifacts

    original = attempt_artifacts.create_attempt_directory
    monkeypatch.setattr(
        attempt_artifacts,
        "create_attempt_directory",
        lambda: original(root=tmp_path / "decisions"),
    )


@pytest.fixture
def block_disallowed_imports(monkeypatch):
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name in _DISALLOWED_MODULES or name.split(".")[0] in _DISALLOWED_MODULES:
            raise AssertionError(f"b2a-calibrate --dry-run must never import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def _dry_run_args(**overrides):
    defaults = dict(config=DISCOVERY_CONFIG, dry_run=True, execute=False, problem_index=None, limit=None)
    defaults.update(overrides)
    return Namespace(**defaults)


def test_dry_run_prints_required_planning_information(capsys, block_disallowed_imports):
    exit_code = cmd_b2a_calibrate(_dry_run_args())
    out = capsys.readouterr().out
    assert "B2A is a one-example engineering calibration. It does not authorize the 12-example pilot." in out
    assert "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" in out
    assert "HuggingFaceH4/MATH-500" in out
    assert "one_example_only=True" in out
    assert "144 real pair evaluations" in out
    assert "canonical_config_hash=" in out
    assert "generation_config_hash=" in out
    assert "rkv_config_hash=" in out
    assert "prompt_template_hash=" in out
    assert "manifest_hash=" in out
    assert "no result files created; no model loaded; no CUDA required" in out
    # B1B-R3 resolved the committed manifest's prompt-token identity for
    # real -- the repository's own manifest is expected to be fully
    # resolved now, so dry-run reports no blockers and exits 0.
    assert "no unresolved/inconsistent identity fields detected" in out
    assert exit_code == 0


def test_dry_run_creates_no_result_files(tmp_path, monkeypatch, block_disallowed_imports):
    before = set(tmp_path.rglob("*"))
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    cmd_b2a_calibrate(_dry_run_args())
    after = set(tmp_path.rglob("*"))
    assert before == after


def test_dry_run_never_touches_cuda(monkeypatch, block_disallowed_imports):
    import torch

    def _boom(*a, **k):
        raise AssertionError("b2a-calibrate --dry-run must never call a CUDA API")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    cmd_b2a_calibrate(_dry_run_args())


def test_registered_in_build_parser():
    parser = build_parser()
    args = parser.parse_args(
        ["b2a-calibrate", "--config", DISCOVERY_CONFIG, "--dry-run"]
    )
    assert args.func is cmd_b2a_calibrate
    assert args.dry_run is True
    assert args.execute is False


def test_execute_without_flag_refuses_before_touching_cuda(monkeypatch):
    import torch

    def _boom(*a, **k):
        raise AssertionError("must not reach a CUDA check without --execute")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    with pytest.raises(SystemExit, match="requires exactly one of --dry-run or --execute"):
        cmd_b2a_calibrate(Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=False, problem_index=None, limit=None))


def test_execute_rejects_problem_index_override():
    with pytest.raises(SystemExit, match="no --problem-index/--limit override"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=5, limit=None)
        )


def test_execute_rejects_limit_override():
    with pytest.raises(SystemExit, match="no --problem-index/--limit override"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=10)
        )


def test_execute_refuses_on_unresolved_manifest_before_touching_cuda(monkeypatch):
    """Simulates an unresolved manifest (monkeypatching the property,
    rather than depending on the repository's actual manifest state, which
    B1B-R3 resolved for real) -- --execute must refuse on that BEFORE ever
    calling a CUDA API, since the identity check is cheaper and more
    fundamental than device availability."""
    import torch

    monkeypatch.setattr(B2AOneExampleManifest, "prompt_identity_is_resolved", property(lambda self: False))

    def _boom(*a, **k):
        raise AssertionError("must not reach a CUDA check while blockers are still outstanding")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    with pytest.raises(SystemExit, match="prompt-token identity is unresolved"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)
        )


def test_execute_requires_cuda_when_manifest_is_resolved(monkeypatch):
    """Proves the CUDA guard fires once the identity blockers are clear --
    the repository's committed manifest is genuinely resolved (B1B-R3), so
    no monkeypatching of the manifest itself is needed here."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="requires CUDA"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)
        )


def _find_attempt_dir(tmp_path) -> Path:
    matches = list((tmp_path / "decisions").glob("b2a_attempt_*"))
    assert len(matches) == 1, f"expected exactly one attempt directory, found {matches}"
    return matches[0]


def test_execute_writes_real_device_preflight_evidence_and_threads_it_to_coordinator(monkeypatch, tmp_path):
    """Independent-audit Gate H4.3: `preflight.json` used to be a trivial
    `{"passed": True, ...}` literal written BEFORE the CUDA check even ran,
    with no device verification at all. `--execute` must now call the same
    raw-evidence producer the workers use (`verify_single_rtx3090`), write
    its REAL output into `preflight.json`, and pass it to the coordinator
    as `cli_device_preflight` so the final gate can cross-check three
    independent observations (CLI, FullKV, R-KV), never just two."""
    import json
    from types import SimpleNamespace

    import torch

    from kvcot.discovery import strict_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    fake_evidence = strict_device.StrictDeviceEvidence(
        visible_gpu_count=1, gpu_name="NVIDIA GeForce RTX 3090", device_index=0,
        requested_device="cuda:0", total_vram_bytes=24 * 1024**3, compute_capability=(8, 6),
        driver_version="555.42", cuda_runtime="12.1", cudnn_version="8900", policy_satisfied=True,
    )
    monkeypatch.setattr(strict_device, "verify_single_rtx3090", lambda *a, **k: fake_evidence)

    captured_kwargs = {}

    def fake_run_b2a_calibration(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            gate_result=SimpleNamespace(passed=True, failed_conditions=()),
            final_gate_result=SimpleNamespace(passed=True, failed_conditions=()),
            artifact_path=tmp_path / "final.json",
            # B2A-R2 forensic repair (audit round 3): the CLI now reads
            # `overall_passed` directly off the coordinator's returned
            # artifact instead of recomputing it -- this fake result must
            # supply it, matching a genuinely fully-passing attempt.
            overall_passed=True, scientific_pair_artifacts_verified=True,
            pair_record_verification_reasons=(),
        )

    monkeypatch.setattr("kvcot.discovery.b2a_execute.run_b2a_calibration", fake_run_b2a_calibration)

    cmd_b2a_calibrate(
        Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)
    )

    assert captured_kwargs["cli_device_preflight"] == {
        "verified": True, "visible_gpu_count": 1, "gpu_name": "NVIDIA GeForce RTX 3090", "device_index": 0,
        "requested_device": "cuda:0", "total_vram_bytes": 24 * 1024**3, "compute_capability": (8, 6),
        "driver_version": "555.42", "cuda_runtime": "12.1", "cudnn_version": "8900", "policy_satisfied": True,
    }

    preflight_path = _find_attempt_dir(tmp_path) / "preflight.json"
    assert preflight_path.is_file()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    assert preflight["passed"] is True
    assert preflight["device"]["gpu_name"] == "NVIDIA GeForce RTX 3090"
    assert preflight["device"]["visible_gpu_count"] == 1


def test_execute_writes_completion_record_on_gate_failure(monkeypatch, tmp_path):
    """Independent-audit Gate H7.2: `invocation.json` is immutable and is
    never rewritten with an end timestamp -- a separate `completion.json`
    must exist recording the outcome and exit code, even when the gate
    itself fails (not just on a clean pass)."""
    import json
    from types import SimpleNamespace

    import torch

    from kvcot.discovery import strict_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fake_evidence = strict_device.StrictDeviceEvidence(
        visible_gpu_count=1, gpu_name="NVIDIA GeForce RTX 3090", device_index=0,
        requested_device="cuda:0", total_vram_bytes=24 * 1024**3, compute_capability=(8, 6),
        driver_version="555.42", cuda_runtime="12.1", cudnn_version="8900", policy_satisfied=True,
    )
    monkeypatch.setattr(strict_device, "verify_single_rtx3090", lambda *a, **k: fake_evidence)

    def fake_run_b2a_calibration(*args, **kwargs):
        return SimpleNamespace(
            gate_result=SimpleNamespace(passed=False, failed_conditions=("some_condition",)),
            final_gate_result=SimpleNamespace(passed=False, failed_conditions=("single_rtx3090_verified",)),
            artifact_path=tmp_path / "final.json",
            overall_passed=False, scientific_pair_artifacts_verified=True,
            pair_record_verification_reasons=(),
        )

    monkeypatch.setattr("kvcot.discovery.b2a_execute.run_b2a_calibration", fake_run_b2a_calibration)

    exit_code = cmd_b2a_calibrate(
        Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)
    )
    assert exit_code == 2

    completion_path = _find_attempt_dir(tmp_path) / "completion.json"
    assert completion_path.is_file()
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["outcome"] == "gate_failed"
    assert completion["exit_code"] == 2
    assert completion["gate_passed"] is False
    assert "finished_at" in completion


def _preflight_passing(monkeypatch):
    import torch

    from kvcot.discovery import strict_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fake_evidence = strict_device.StrictDeviceEvidence(
        visible_gpu_count=1, gpu_name="NVIDIA GeForce RTX 3090", device_index=0,
        requested_device="cuda:0", total_vram_bytes=24 * 1024**3, compute_capability=(8, 6),
        driver_version="555.42", cuda_runtime="12.1", cudnn_version="8900", policy_satisfied=True,
    )
    monkeypatch.setattr(strict_device, "verify_single_rtx3090", lambda *a, **k: fake_evidence)


def _execute_args():
    return Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)


def test_cli_returns_2_for_an_isolated_pair_artifact_failure(monkeypatch, tmp_path, capsys):
    """B2A-R2 forensic repair (audit round 3,
    docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md §11): the
    legacy AND final gates both genuinely pass -- pair-artifact
    verification is the ONLY failing factor. The CLI must still return 2
    and report `passed=False`, never silently succeed by only checking the
    two pre-existing gates."""
    import json
    from types import SimpleNamespace

    _preflight_passing(monkeypatch)

    def fake_run_b2a_calibration(*args, **kwargs):
        return SimpleNamespace(
            gate_result=SimpleNamespace(passed=True, failed_conditions=()),
            final_gate_result=SimpleNamespace(passed=True, failed_conditions=()),
            artifact_path=tmp_path / "final.json",
            overall_passed=False, scientific_pair_artifacts_verified=False,
            pair_record_verification_reasons=("rkv/pair_records.json is missing",),
        )

    monkeypatch.setattr("kvcot.discovery.b2a_execute.run_b2a_calibration", fake_run_b2a_calibration)

    exit_code = cmd_b2a_calibrate(_execute_args())
    assert exit_code == 2

    out = capsys.readouterr().out
    assert "passed=False" in out
    assert "rkv/pair_records.json is missing" in out

    completion_path = _find_attempt_dir(tmp_path) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["outcome"] == "gate_failed"
    assert completion["exit_code"] == 2
    assert completion["gate_passed"] is False


def test_cli_returns_0_when_all_three_factors_pass(monkeypatch, tmp_path, capsys):
    import json
    from types import SimpleNamespace

    _preflight_passing(monkeypatch)

    def fake_run_b2a_calibration(*args, **kwargs):
        return SimpleNamespace(
            gate_result=SimpleNamespace(passed=True, failed_conditions=()),
            final_gate_result=SimpleNamespace(passed=True, failed_conditions=()),
            artifact_path=tmp_path / "final.json",
            overall_passed=True, scientific_pair_artifacts_verified=True,
            pair_record_verification_reasons=(),
        )

    monkeypatch.setattr("kvcot.discovery.b2a_execute.run_b2a_calibration", fake_run_b2a_calibration)

    exit_code = cmd_b2a_calibrate(_execute_args())
    assert exit_code == 0
    assert "passed=True" in capsys.readouterr().out

    completion_path = _find_attempt_dir(tmp_path) / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["outcome"] == "gate_passed"
    assert completion["exit_code"] == 0
    assert completion["gate_passed"] is True


@pytest.mark.parametrize(
    "overall_passed,expected_exit_code",
    [(True, 0), (False, 2)],
    ids=["overall_passed=True", "overall_passed=False"],
)
def test_cli_return_code_and_printed_passed_value_never_disagree_with_coordinator(
    monkeypatch, tmp_path, capsys, overall_passed, expected_exit_code
):
    """Cross-surface invariant: the CLI now does nothing but relay
    `artifact.overall_passed` (`overall_passed = artifact.overall_passed`,
    no recomputation) -- whatever the coordinator decided, the CLI's return
    code and printed `passed=` value must match it exactly, regardless of
    what the individual `gate_result`/`final_gate_result` booleans happen
    to be (deliberately held constant at True here, isolating that the CLI
    reads `overall_passed` and NOT some recomputed subset)."""
    import json
    from types import SimpleNamespace

    _preflight_passing(monkeypatch)

    def fake_run_b2a_calibration(*args, **kwargs):
        return SimpleNamespace(
            gate_result=SimpleNamespace(passed=True, failed_conditions=()),
            final_gate_result=SimpleNamespace(passed=True, failed_conditions=()),
            artifact_path=tmp_path / "final.json",
            overall_passed=overall_passed, scientific_pair_artifacts_verified=overall_passed,
            pair_record_verification_reasons=() if overall_passed else ("simulated pair-artifact failure",),
        )

    monkeypatch.setattr("kvcot.discovery.b2a_execute.run_b2a_calibration", fake_run_b2a_calibration)

    exit_code = cmd_b2a_calibrate(_execute_args())
    assert exit_code == expected_exit_code

    out = capsys.readouterr().out
    assert f"passed={overall_passed}" in out

    completion = json.loads((_find_attempt_dir(tmp_path) / "completion.json").read_text(encoding="utf-8"))
    assert completion["gate_passed"] is overall_passed
    assert completion["exit_code"] == expected_exit_code
    assert (completion["outcome"] == "gate_passed") is overall_passed


def test_execute_writes_completion_record_even_on_uncaught_exception(monkeypatch, tmp_path):
    """The completion record must exist even when the coordinator itself
    raises (a genuine crash) -- never only on a clean return path."""
    import json

    import torch

    from kvcot.discovery import strict_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fake_evidence = strict_device.StrictDeviceEvidence(
        visible_gpu_count=1, gpu_name="NVIDIA GeForce RTX 3090", device_index=0,
        requested_device="cuda:0", total_vram_bytes=24 * 1024**3, compute_capability=(8, 6),
        driver_version="555.42", cuda_runtime="12.1", cudnn_version="8900", policy_satisfied=True,
    )
    monkeypatch.setattr(strict_device, "verify_single_rtx3090", lambda *a, **k: fake_evidence)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated coordinator crash")

    monkeypatch.setattr("kvcot.discovery.b2a_execute.run_b2a_calibration", boom)

    with pytest.raises(RuntimeError, match="simulated coordinator crash"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)
        )

    completion_path = _find_attempt_dir(tmp_path) / "completion.json"
    assert completion_path.is_file()
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["outcome"] == "exception"
    assert completion["exit_code"] is None
    assert completion["gate_passed"] is None


def test_execute_refuses_and_writes_failure_when_device_preflight_fails(monkeypatch, tmp_path):
    """A real device-verification failure (wrong GPU, wrong count, etc.)
    must refuse BEFORE launching any worker, and record why."""
    import json

    import torch

    from kvcot.discovery import strict_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(*a, **k):
        raise strict_device.StrictDeviceError("B2A requires an RTX 3090, observed 'NVIDIA A100'")

    monkeypatch.setattr(strict_device, "verify_single_rtx3090", boom)

    with pytest.raises(SystemExit, match="device preflight failed"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)
        )

    attempt_dir = _find_attempt_dir(tmp_path)
    failure_path = attempt_dir / "failure.json"
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["stage"] == "device_preflight"
    assert "RTX 3090" in failure["reason"]
    assert not (attempt_dir / "preflight.json").is_file()


def test_rejects_bad_config_before_printing_anything(tmp_path, block_disallowed_imports):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        "model:\n"
        "  name: x\n"
        "  revision: main\n"
        "  tokenizer_name: x\n"
        "  tokenizer_revision: main\n"
        "  model_type: llama\n"
        "  dtype: bfloat16\n"
        "dataset:\n"
        "  name: MATH-500\n"
        "rkv:\n"
        "  budget: 1024\n"
        "  upstream_revision: 45eaa7d69d20b7388321f077020a610d9afb65bd\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        cmd_b2a_calibrate(Namespace(config=str(bad_config), dry_run=True, execute=False, problem_index=None, limit=None))
