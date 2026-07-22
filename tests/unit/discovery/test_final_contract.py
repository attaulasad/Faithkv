import pytest
from argparse import Namespace
from pathlib import Path

from kvcot.discovery.final_contract import (
    FINAL_MANDATORY_GATE_CONDITIONS,
    evaluate_final_gates,
    memory_contract_satisfied,
    timing_contract_satisfied,
)


def test_final_gate_contract_is_exact_and_fail_closed():
    values = {name: True for name in FINAL_MANDATORY_GATE_CONDITIONS}
    result = evaluate_final_gates(values)
    assert result.passed
    for name in FINAL_MANDATORY_GATE_CONDITIONS:
        failed = dict(values, **{name: False})
        result = evaluate_final_gates(failed)
        assert not result.passed
        assert name in result.failed_conditions


def test_final_gate_contract_rejects_missing_extra_and_non_boolean_values():
    values = {name: True for name in FINAL_MANDATORY_GATE_CONDITIONS}
    missing = dict(values)
    missing.pop(FINAL_MANDATORY_GATE_CONDITIONS[0])
    with pytest.raises(ValueError, match="missing"):
        evaluate_final_gates(missing)
    with pytest.raises(ValueError, match="extra"):
        evaluate_final_gates(dict(values, invented_gate=True))
    with pytest.raises(TypeError):
        evaluate_final_gates(dict(values, git_clean_verified=1))


def test_contract_names_span_dry_run_docs_worker_schemas_and_final_artifact(capsys):
    import inspect

    from kvcot.cli import cmd_b2a_calibrate
    from kvcot.discovery.b2a_execute import B2ACalibrationArtifact, run_b2a_calibration
    from kvcot.discovery.b2a_workers import FullKVWorkerResult, RKVWorkerResult

    code = cmd_b2a_calibrate(Namespace(
        config="configs/discovery/llama8b_math500_b1024.yaml", dry_run=True, execute=False,
        problem_index=None, limit=None,
    ))
    assert code == 0
    dry_run = capsys.readouterr().out
    documentation = Path("docs/B1_FINAL_CPU_CLOSURE.md").read_text(encoding="utf-8")
    for name in FINAL_MANDATORY_GATE_CONDITIONS:
        assert name in dry_run
        assert name in documentation
    for schema in (FullKVWorkerResult, RKVWorkerResult):
        assert {"timing_evidence", "memory_phase_evidence", "snapshot_evidence", "device_evidence"}.issubset(
            schema.model_fields
        )
    assert "final_gate_result" in B2ACalibrationArtifact.__dataclass_fields__
    coordinator_source = inspect.getsource(run_b2a_calibration)
    for name in FINAL_MANDATORY_GATE_CONDITIONS:
        assert name in coordinator_source


def test_missing_timing_and_malformed_memory_fail_closed():
    assert timing_contract_satisfied([], []) is False
    malformed = [{"phase": "model_load", "completed": True, "peak_allocated": "unknown"}]
    assert memory_contract_satisfied(malformed, malformed) is False
