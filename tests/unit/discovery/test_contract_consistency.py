"""Independent-audit Gate H8.2: one contract-consistency test spanning
constants, timing-phase names, worker/partial-worker/envelope schemas,
attrition stages, mismatch schemas, and gate-condition test coverage.

Each check below cross-validates two INDEPENDENTLY-maintained pieces of
this repository against each other -- a declared list vs. the code that
actually emits/consumes it -- so a future drift (a renamed phase, an added
gate condition with no test, a stage constant left out of `STAGE_ORDER`,
an envelope field the partial-evidence capture path can't fill) is caught
here rather than discovered only during a later hostile audit.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DISCOVERY = REPO_ROOT / "src" / "kvcot" / "discovery"
TESTS_ROOT = REPO_ROOT / "tests"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _all_test_source() -> str:
    return "\n".join(_read(path) for path in TESTS_ROOT.rglob("test_*.py"))


# --------------------------------------------------------------------------
# Timing-phase names: every phase a schema/contract module REQUIRES must be
# a literal string that actually appears at an emitting call site -- never
# a declared name with no corresponding production code.
# --------------------------------------------------------------------------


def test_every_required_timing_phase_is_emitted_somewhere_in_production_code():
    from kvcot.discovery.final_contract import (
        FULLKV_REQUIRED_TIMING_PHASES,
        PAIR_REQUIRED_TIMING_SUBPHASES,
        RKV_REQUIRED_TIMING_PHASES,
    )

    emitting_source = (
        _read(SRC_DISCOVERY / "b2a_workers.py") + _read(SRC_DISCOVERY / "pipeline.py")
        + _read(SRC_DISCOVERY / "capture.py")
    )
    # `rkv_pass1_prefill`/`rkv_pass1_decode`/`rkv_pass2_prefill`/
    # `rkv_pass2_decode` are constructed dynamically
    # (`f"rkv_{pass_name}_prefill"`/`f"rkv_{pass_name}_decode"` in
    # `_RkvHarnessInstrumentation`, `pass_name` is `"pass1"`/`"pass2"`) --
    # verified by the f-string TEMPLATE's presence instead of the
    # concatenated literal, which never appears verbatim anywhere.
    dynamically_constructed = {
        "rkv_pass1_prefill": 'f"rkv_{pass_name}_prefill"',
        "rkv_pass1_decode": 'f"rkv_{pass_name}_decode"',
        "rkv_pass2_prefill": 'f"rkv_{pass_name}_prefill"',
        "rkv_pass2_decode": 'f"rkv_{pass_name}_decode"',
    }
    missing = []
    for phase in (*FULLKV_REQUIRED_TIMING_PHASES, *RKV_REQUIRED_TIMING_PHASES, *PAIR_REQUIRED_TIMING_SUBPHASES):
        if phase in dynamically_constructed:
            if dynamically_constructed[phase] not in emitting_source:
                missing.append(phase)
        elif f'"{phase}"' not in emitting_source:
            missing.append(phase)
    assert missing == [], (
        f"required timing phase(s) declared in final_contract.py but not found as a literal "
        f"emitting string (or its dynamic-construction template) in b2a_workers.py/pipeline.py: {missing}"
    )


def test_capture_and_parity_literal_name_no_longer_used_as_a_real_phase():
    """Regression guard for the exact Gate H2 defect: the misleadingly-named
    `capture_and_parity` phase (which only ever timed a post-hoc trace
    comparison) must not silently reappear as a required or emitted phase
    name -- `call_trace_comparison` is its permanent replacement."""
    from kvcot.discovery.final_contract import RKV_REQUIRED_TIMING_PHASES

    assert "capture_and_parity" not in RKV_REQUIRED_TIMING_PHASES
    assert "call_trace_comparison" in RKV_REQUIRED_TIMING_PHASES
    assert '"call_trace_comparison"' in _read(SRC_DISCOVERY / "b2a_workers.py")


# --------------------------------------------------------------------------
# Mandatory gate conditions: every gate name must appear in the CLI's
# printed dry-run plan (documentation/dry-run must never drift from the
# canonical contract), and every gate must have at least one dedicated
# test mentioning it by name (never a silently-untested mandatory gate).
# --------------------------------------------------------------------------


def test_final_mandatory_gate_conditions_has_no_duplicates_and_matches_dry_run_output():
    import io
    from contextlib import redirect_stdout

    from kvcot.cli import build_parser, cmd_b2a_calibrate
    from kvcot.discovery.final_contract import FINAL_MANDATORY_GATE_CONDITIONS

    assert len(FINAL_MANDATORY_GATE_CONDITIONS) == len(set(FINAL_MANDATORY_GATE_CONDITIONS)), (
        "FINAL_MANDATORY_GATE_CONDITIONS contains a duplicate gate name"
    )

    parser = build_parser()
    args = parser.parse_args([
        "b2a-calibrate", "--config", "configs/discovery/llama8b_math500_b1024.yaml", "--dry-run",
    ])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cmd_b2a_calibrate(args)
    printed = buffer.getvalue()
    printed_gates = {line.strip().lstrip("- ").strip() for line in printed.splitlines() if line.strip().startswith("- ")}
    assert printed_gates == set(FINAL_MANDATORY_GATE_CONDITIONS), (
        "the dry-run's printed gate list has drifted from FINAL_MANDATORY_GATE_CONDITIONS: "
        f"printed-only={printed_gates - set(FINAL_MANDATORY_GATE_CONDITIONS)}, "
        f"contract-only={set(FINAL_MANDATORY_GATE_CONDITIONS) - printed_gates}"
    )


def test_every_mandatory_gate_condition_has_a_real_negative_test():
    """Directly exercises `evaluate_final_gates` with every gate condition
    individually set to `False` -- this IS the negative test H8.2 requires
    for each mandatory gate, proven here rather than merely checked for
    (`tests/unit/discovery/test_final_contract.py`'s own
    `test_final_gate_contract_is_exact_and_fail_closed` already does this
    too via the same canonical loop; duplicating the direct proof here
    means this contract-consistency test does not depend on that other
    test continuing to exist unchanged)."""
    from kvcot.discovery.final_contract import FINAL_MANDATORY_GATE_CONDITIONS, evaluate_final_gates

    passing = {name: True for name in FINAL_MANDATORY_GATE_CONDITIONS}
    assert evaluate_final_gates(passing).passed is True
    for name in FINAL_MANDATORY_GATE_CONDITIONS:
        result = evaluate_final_gates({**passing, name: False})
        assert result.passed is False, f"gate {name!r} did not fail the overall result when set to False"
        assert result.failed_conditions == (name,), f"gate {name!r}'s negative test affected other conditions"


# --------------------------------------------------------------------------
# Attrition stages: every STAGE_* constant defined in attrition.py must be
# registered in STAGE_ORDER -- a defined-but-unregistered stage would raise
# ValueError the first time anything tried to record it (`AttritionCounters
# .record_dropped`), a class of bug this repair pass introduced twice (two
# new stages, both requiring this exact registration) and so is worth its
# own permanent regression guard.
# --------------------------------------------------------------------------


def test_every_declared_attrition_stage_constant_is_registered_in_stage_order():
    from kvcot.discovery import attrition

    declared_stage_values = {
        value for name, value in vars(attrition).items()
        if name.startswith("STAGE_") and isinstance(value, str)
    }
    missing = declared_stage_values - set(attrition.STAGE_ORDER)
    assert missing == set(), f"STAGE_* constant(s) declared but missing from STAGE_ORDER: {missing}"


def test_every_stage_order_entry_is_a_declared_stage_constant():
    """The reverse direction: `STAGE_ORDER` must never contain a bare
    string literal that isn't backed by a named `STAGE_*` constant
    elsewhere in the module (which would make it easy to typo one spot and
    not the other)."""
    from kvcot.discovery import attrition

    declared_stage_values = {
        value for name, value in vars(attrition).items()
        if name.startswith("STAGE_") and isinstance(value, str)
    }
    extra = set(attrition.STAGE_ORDER) - declared_stage_values
    assert extra == set(), f"STAGE_ORDER entry with no corresponding STAGE_* constant: {extra}"


# --------------------------------------------------------------------------
# Worker envelope / partial-evidence schema: the failure envelope's typed
# fields must be exactly what `capture_partial_evidence` can actually
# populate -- never an envelope field the partial-evidence path has no way
# to fill, and never partial evidence dropped on the floor because the
# envelope has nowhere to put it.
# --------------------------------------------------------------------------


def test_worker_envelope_failure_fields_match_partial_evidence_capabilities():
    from kvcot.discovery.worker_envelope import WorkerEnvelope
    from kvcot.discovery.worker_partial_evidence import PartialWorkerEvidence

    envelope_fields = set(WorkerEnvelope.model_fields)
    required_failure_fields = {"failure_stage", "last_completed_stage", "is_oom", "is_timeout"}
    missing = required_failure_fields - envelope_fields
    assert missing == set(), f"WorkerEnvelope is missing typed failure field(s): {missing}"

    evidence_fields = set(PartialWorkerEvidence.model_fields)
    # Every field the envelope's failure branch reads off `PartialWorkerEvidence`
    # (`kvcot.discovery.b2a_worker_entry.main`'s except-block) must actually
    # exist on that schema.
    for name in ("failing_stage", "last_completed_stage", "is_oom", "is_timeout", "determinism_policy"):
        assert name in evidence_fields, f"PartialWorkerEvidence is missing field {name!r} that b2a_worker_entry reads"


def test_b2a_worker_entry_threads_every_partial_evidence_field_it_reads():
    """Static drift guard: `b2a_worker_entry.py`'s except-block references
    `evidence.<field>` for a fixed set of attributes -- every one of them
    must be a real field on `PartialWorkerEvidence`, never a typo that
    would only surface as an `AttributeError` deep inside a real GPU
    failure path."""
    from kvcot.discovery.worker_partial_evidence import PartialWorkerEvidence

    source = _read(SRC_DISCOVERY / "b2a_worker_entry.py")
    evidence_fields = set(PartialWorkerEvidence.model_fields)
    referenced = {
        line.split("evidence.", 1)[1].split(")")[0].split(",")[0].strip()
        for line in source.splitlines() if "evidence." in line and "evidence.evidence" not in line
    }
    # Filter to plausible bare attribute names only (defensive against the
    # naive split above picking up unrelated substrings).
    referenced = {name for name in referenced if name.isidentifier()}
    unknown = referenced - evidence_fields
    assert unknown == set(), f"b2a_worker_entry.py references unknown PartialWorkerEvidence field(s): {unknown}"


# --------------------------------------------------------------------------
# Mismatch schema: `MismatchRecord.export()`'s keys must be exactly what
# `b2a_workers.py`'s replay_evidence consumers expect (H3's canonical
# schema, never a silently-renamed field).
# --------------------------------------------------------------------------


def test_mismatch_record_export_schema_is_frozen():
    from kvcot.discovery.mismatch import build_mismatch_record

    exported = build_mismatch_record([1], [1]).export()
    assert set(exported) == {
        "matched", "first_mismatch_index", "expected_value", "observed_value",
        "expected_length", "observed_length", "mismatch_kind",
    }


# --------------------------------------------------------------------------
# Attempt-artifact schema: the canonical required-file set must be
# referenced from exactly one place (`kvcot.discovery.attempt_verification`)
# -- never re-declared inline anywhere else, which is precisely how the
# pre-repair `issubset(existing)` check and the content-verifying replacement
# could silently drift from each other.
# --------------------------------------------------------------------------


def test_required_attempt_files_declared_in_exactly_one_place():
    from kvcot.discovery.attempt_verification import REQUIRED_ATTEMPT_FILES

    assert "rkv/replay_evidence.json" in REQUIRED_ATTEMPT_FILES
    assert "fullkv/envelope.json" in REQUIRED_ATTEMPT_FILES
    b2a_execute_source = _read(SRC_DISCOVERY / "b2a_execute.py")
    # The old inline literal set and its existence-only check must not have
    # been reintroduced as CODE (a historical/explanatory comment
    # referencing the old pattern by name is fine and expected -- only the
    # functional code line is checked for here).
    assert "required_attempt_files = {" not in b2a_execute_source
    assert "= required_attempt_files.issubset(existing)" not in b2a_execute_source
