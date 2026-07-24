"""B2A-R3 CLI tests (Step 3 Stage-A). Functional coverage for the seven
new CPU-only planning/verification commands, a check that no forbidden
command was added, and subprocess-based import-safety tests proving every
dry-run command never imports torch/transformers/R-KV, even when those
modules would raise immediately if imported."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kvcot.cli import build_parser, main
from kvcot.discovery.b2a_r3_contract import CANDIDATE_MANIFEST_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]

FORBIDDEN_COMMANDS = {"run-b2a-r3-qualification", "execute-b2a-r3", "claim-b2a-r3-authorization"}
EXPECTED_COMMANDS = {
    "prepare-b2a-r3-candidates",
    "verify-b2a-r3-candidates",
    "plan-b2a-r3-qualification",
    "verify-b2a-r3-qualification",
    "freeze-b2a-r3-selected-row",
    "verify-b2a-r3-selection",
    "verify-b2a-r3-authorization",
}


def _registered_commands() -> set[str]:
    parser = build_parser()
    for action in parser._subparsers._group_actions:  # noqa: SLF001 -- introspecting argparse's own structure
        return set(action.choices.keys())
    return set()


def test_all_expected_b2a_r3_commands_are_registered():
    assert EXPECTED_COMMANDS <= _registered_commands()


def test_no_forbidden_b2a_r3_commands_are_registered():
    assert not (FORBIDDEN_COMMANDS & _registered_commands())


def test_prepare_b2a_r3_candidates_dry_run(capsys):
    rc = main(["prepare-b2a-r3-candidates", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no network fetch, no write performed" in out


def test_verify_b2a_r3_candidates_against_committed_manifest(capsys):
    rc = main(["verify-b2a-r3-candidates"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "structural verification PASSED" in out


def test_plan_b2a_r3_qualification_dry_run_against_committed_manifest(capsys):
    rc = main(["plan-b2a-r3-qualification", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "qualification_candidate_limit = 8" in out
    assert "absolute_timeout_envelope_seconds = 57600" in out
    assert "qualification_phase_wall_time_limit = null" in out
    assert "gpu_qualification_authorized = false" in out
    assert "would_import_rkv = false" in out


def test_plan_b2a_r3_qualification_refuses_without_dry_run():
    with pytest.raises(SystemExit):
        main(["plan-b2a-r3-qualification"])


def test_freeze_b2a_r3_selected_row_refuses_without_dry_run():
    with pytest.raises(SystemExit):
        main(["freeze-b2a-r3-selected-row"])


def test_freeze_b2a_r3_selected_row_rejects_both_dry_run_and_execute():
    with pytest.raises(SystemExit):
        main(["freeze-b2a-r3-selected-row", "--dry-run", "--execute"])


def test_freeze_b2a_r3_selected_row_rejects_neither_dry_run_nor_execute():
    with pytest.raises(SystemExit):
        main(["freeze-b2a-r3-selected-row"])


@pytest.mark.parametrize("flag,value", [("--candidates", "some/other/path.json"),
                                         ("--artifact", "some/other/path.json"),
                                         ("--config", "some/other/path.yaml")])
def test_freeze_execute_rejects_alternate_paths(flag, value):
    with pytest.raises(SystemExit):
        main(["freeze-b2a-r3-selected-row", "--execute", flag, value])


def test_freeze_execute_rejects_alternate_candidate_path_even_if_it_equals_the_fixed_path():
    from kvcot.discovery.b2a_r3_contract import CANDIDATE_MANIFEST_PATH as FIXED_PATH

    with pytest.raises(SystemExit):
        main(["freeze-b2a-r3-selected-row", "--execute", "--candidates", FIXED_PATH])


def test_freeze_execute_has_no_output_path_override_flags():
    parser = build_parser()
    freeze_parser = None
    for action in parser._subparsers._group_actions:  # noqa: SLF001
        freeze_parser = action.choices.get("freeze-b2a-r3-selected-row")
        break
    assert freeze_parser is not None
    option_strings = {opt for action in freeze_parser._actions for opt in action.option_strings}  # noqa: SLF001
    assert "--output" not in option_strings
    assert "--manifest-output" not in option_strings
    assert "--provenance-output" not in option_strings
    assert "--force" not in option_strings


def test_freeze_execute_delegates_to_production_function_and_prints_output(capsys, monkeypatch):
    import kvcot.discovery.b2a_r3_freeze as freeze_module
    import kvcot.discovery.b2a_r3_production_tokenizer as tokenizer_module

    class _FakeSnapshot:
        resolved_revision = "6a6f4aa4197940add57724a7707d069478df56b1"
        local_path = "/fake/local/path"

    fake_result = {
        "selected_unique_id": "test/number_theory/631.json",
        "selected_ordinal": 1,
        "selected_manifest_path": "configs/discovery/b2a_one_example_manifest.json",
        "selected_manifest_sha256": "a" * 64,
        "selection_provenance_path": "results/decisions/b2a_r3_selection_provenance.json",
        "selection_provenance_canonical_sha256": "b" * 64,
        "tokenizer_snapshot_revision": "6a6f4aa4197940add57724a7707d069478df56b1",
        "publication_state_before": "state_a_initial",
        "publication_state_after": "state_b_complete",
        "already_frozen": False,
        "verification_passed": True,
    }

    monkeypatch.setattr(tokenizer_module, "resolve_production_tokenizer_snapshot", lambda **_kw: _FakeSnapshot())
    monkeypatch.setattr(freeze_module, "construct_production_freeze_plan", lambda **_kw: object())
    monkeypatch.setattr(freeze_module, "publish_production_freeze", lambda _plan: fake_result)

    rc = main(["freeze-b2a-r3-selected-row", "--execute"])
    assert rc == 0
    out = capsys.readouterr().out
    for key, value in fake_result.items():
        assert f"{key} = {value}" in out


def test_freeze_execute_returns_nonzero_when_verification_fails(monkeypatch):
    import kvcot.discovery.b2a_r3_freeze as freeze_module
    import kvcot.discovery.b2a_r3_production_tokenizer as tokenizer_module

    class _FakeSnapshot:
        resolved_revision = "x"
        local_path = "/fake"

    fake_result = {
        "selected_unique_id": "row", "selected_ordinal": 1,
        "selected_manifest_path": "p1", "selected_manifest_sha256": "a" * 64,
        "selection_provenance_path": "p2", "selection_provenance_canonical_sha256": "b" * 64,
        "tokenizer_snapshot_revision": "x", "publication_state_before": "state_e_invalid",
        "publication_state_after": "state_e_invalid", "already_frozen": False, "verification_passed": False,
    }
    monkeypatch.setattr(tokenizer_module, "resolve_production_tokenizer_snapshot", lambda **_kw: _FakeSnapshot())
    monkeypatch.setattr(freeze_module, "construct_production_freeze_plan", lambda **_kw: object())
    monkeypatch.setattr(freeze_module, "publish_production_freeze", lambda _plan: fake_result)

    rc = main(["freeze-b2a-r3-selected-row", "--execute"])
    assert rc != 0


def _skip_unless_full_git_history_available():
    """The real dry-run path replays the committed Stage-B authorization
    claim, which requires `authorized_code_commit_sha` (an ancestor commit,
    not necessarily the tip) to exist in the local git object database.
    On a full clone (any normal local checkout) it always does; on a
    shallow CI checkout (`actions/checkout@v4`'s default `fetch-depth: 1`,
    unchanged by this phase -- `.github` is out of scope) it does not.
    Skip cleanly rather than fail on an environment property this test
    cannot control and this phase is not authorized to fix."""
    from kvcot.discovery.b2a_r3_contract import QUALIFICATION_ARTIFACT_PATH

    with open(QUALIFICATION_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        qualification_artifact = json.load(f)
    claim_path = Path(
        f"results/decisions/b2a_r3_authorization_claims/{qualification_artifact['stage_b_authorization_id']}.json"
    )
    with open(claim_path, "r", encoding="utf-8") as f:
        claim = json.load(f)
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{claim['authorized_code_commit_sha']}^{{commit}}"],
        cwd=REPO_ROOT, capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("full git history for the Stage-B authorized code commit is unavailable (shallow checkout)")


def test_freeze_dry_run_still_reports_no_write_or_tokenizer_action(capsys):
    _skip_unless_full_git_history_available()
    rc = main(["freeze-b2a-r3-selected-row", "--dry-run"])
    out = capsys.readouterr().out
    assert "would_load_tokenizer_for_execution = False" in out
    assert "would_write_selected_manifest = False" in out
    assert "would_write_selection_provenance = False" in out


def test_verify_b2a_r3_authorization_end_to_end(tmp_path, capsys, monkeypatch):
    from kvcot.discovery.b2a_r3_contract import (
        AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
        REQUIRED_REPOSITORY,
        global_claim_path,
    )
    from kvcot.utils.hashing import sha256_file, sha256_json
    from tests.unit.discovery.test_b2a_r3_provenance import FakeGitState

    from kvcot.discovery.b2a_r3_authorization_document import (
        AUTHORIZATION_DOCUMENT_BEGIN_MARKER,
        AUTHORIZATION_DOCUMENT_END_MARKER,
        AUTHORIZATION_DOCUMENT_SCHEMA_VERSION,
    )

    document_rel = "docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-08-01.md"
    document_path = tmp_path / document_rel
    document_path.parent.mkdir(parents=True)
    candidate_manifest = json.loads(Path("configs/discovery/b2a_r3_candidate_manifest.json").read_text())
    document_body = {
        "authorization_document_schema_version": AUTHORIZATION_DOCUMENT_SCHEMA_VERSION,
        "authorization_id": "cli-smoke-test",
        "authorization_stage": "fullkv_qualification",
        "authorized_repository": REQUIRED_REPOSITORY,
        "authorized_branch": "research/b2a-r3-runtime-qualified-calibration",
        "authorized_code_commit_sha": "b" * 40,
        "required_ancestor_shas": ["c" * 40],
        "required_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
        "candidate_manifest_canonical_sha256": candidate_manifest["canonical_sha256"],
        "maximum_candidates": 8,
        "phase_wall_time_limit_seconds": 3600,
        "qualification_artifact_canonical_sha256": None,
        "selected_manifest_sha256": None,
        "selected_manifest_hash_algorithm": None,
        "created_at_utc": "2026-08-01T00:00:00+00:00",
    }
    document_path.write_text(
        "# CLI smoke-test authorization document\n\n"
        f"{AUTHORIZATION_DOCUMENT_BEGIN_MARKER}\n```json\n{json.dumps(document_body, indent=2)}\n```\n"
        f"{AUTHORIZATION_DOCUMENT_END_MARKER}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kvcot.discovery.b2a_r3_provenance.SubprocessGitStateProvider",
        lambda _root: FakeGitState(
            commit_sha="d" * 40,
            ancestors=frozenset({"b" * 40, "c" * 40}),
            repository_root=_root,
            changed_paths=(document_rel,),
        ),
    )

    payload = {
        "artifact_schema_version": AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
        "authorization_id": "cli-smoke-test",
        "authorization_stage": "fullkv_qualification",
        "authorization_document_path": document_rel,
        "authorization_document_sha256": sha256_file(document_path),
        "authorized_repository": REQUIRED_REPOSITORY,
        "authorized_branch": "research/b2a-r3-runtime-qualified-calibration",
        "authorized_code_commit_sha": "b" * 40,
        "observed_repository": REQUIRED_REPOSITORY,
        "observed_branch": "research/b2a-r3-runtime-qualified-calibration",
        "observed_execution_commit_sha": "d" * 40,
        "required_ancestor_shas": ["c" * 40],
        "required_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
        "observed_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
        "candidate_manifest_canonical_sha256": candidate_manifest["canonical_sha256"],
        "qualification_artifact_canonical_sha256": None,
        "selected_manifest_sha256": None,
        "selected_manifest_hash_algorithm": None,
        "attempt_id": "deadbeef",
        "global_claim_path": global_claim_path("cli-smoke-test"),
        "attempt_directory_path": "results/decisions/b2a_r3_attempt_20260801T000000000000Z_deadbeef",
        "claimed_at_utc": "2026-08-01T00:00:00+00:00",
    }
    payload["canonical_sha256"] = sha256_json(payload)
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(json.dumps(payload), encoding="utf-8")

    rc = main([
        "verify-b2a-r3-authorization", "--claim", str(claim_path),
        "--document", str(document_path), "--repository-root", str(tmp_path),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "verification PASSED" in out


def test_verify_b2a_r3_authorization_rejects_tampered_claim(tmp_path, capsys):
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(json.dumps({"not": "a valid claim"}), encoding="utf-8")
    rc = main(["verify-b2a-r3-authorization", "--claim", str(claim_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "VERIFICATION FAILED" in out


# --------------------------------------------------------------------- import safety


_GUARD_PREAMBLE = """
import sys

class _ForbiddenImportError(ImportError):
    pass

_FORBIDDEN_TOP = {"torch", "transformers", "flash_attn"}
_FORBIDDEN_SUBMODULES = (
    "kvcot.discovery.b2a_workers",
    "kvcot.discovery.schemas",
    "kvcot.discovery.scientific_summary",
    "kvcot.generation.policies",
    "kvcot.discovery.b2a_r3_production_tokenizer",
)

class _Guard:
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top in _FORBIDDEN_TOP:
            raise _ForbiddenImportError("FORBIDDEN_IMPORT:" + name)
        if any(name == m or name.startswith(m + ".") for m in _FORBIDDEN_SUBMODULES):
            raise _ForbiddenImportError("FORBIDDEN_IMPORT:" + name)
        return None

sys.meta_path.insert(0, _Guard())
sys.path.insert(0, "src")
from kvcot.cli import main
"""


def _run_guarded(argv: list[str]) -> subprocess.CompletedProcess:
    script = _GUARD_PREAMBLE + f"\nrc = main({argv!r})\nsys.exit(rc)\n"
    return subprocess.run(
        [sys.executable, "-c", script], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["prepare-b2a-r3-candidates", "--dry-run"],
        ["verify-b2a-r3-candidates"],
        ["plan-b2a-r3-qualification", "--dry-run"],
    ],
)
def test_dry_run_commands_never_import_forbidden_modules(argv):
    result = _run_guarded(argv)
    assert "FORBIDDEN_IMPORT" not in result.stdout
    assert "FORBIDDEN_IMPORT" not in result.stderr
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"


def test_verify_qualification_dry_run_never_imports_forbidden_modules(tmp_path):
    # No real qualification artifact exists under Stage A -- this proves
    # the COMMAND ITSELF (its import graph) is safe even when it fails
    # closed on a missing file, not just on the happy path.
    missing = tmp_path / "does-not-exist.json"
    result = _run_guarded(
        ["verify-b2a-r3-qualification", "--artifact", str(missing), "--candidates", str(CANDIDATE_MANIFEST_PATH)]
    )
    assert "FORBIDDEN_IMPORT" not in result.stdout
    assert "FORBIDDEN_IMPORT" not in result.stderr


def test_freeze_dry_run_never_imports_forbidden_modules(tmp_path):
    missing = tmp_path / "does-not-exist.json"
    result = _run_guarded(
        ["freeze-b2a-r3-selected-row", "--dry-run", "--artifact", str(missing)]
    )
    assert "FORBIDDEN_IMPORT" not in result.stdout
    assert "FORBIDDEN_IMPORT" not in result.stderr


def test_freeze_dry_run_against_real_committed_artifact_never_imports_forbidden_modules():
    """Same import-safety guarantee, but against the REAL committed
    candidate/qualification artifacts (would_freeze=True) -- proves the
    dry-run path never imports torch/transformers/the production tokenizer
    module even on the happy path, not only when it fails closed on a
    missing file."""
    _skip_unless_full_git_history_available()
    result = _run_guarded(["freeze-b2a-r3-selected-row", "--dry-run"])
    assert "FORBIDDEN_IMPORT" not in result.stdout
    assert "FORBIDDEN_IMPORT" not in result.stderr
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "would_freeze = True" in result.stdout


# --------------------------------------------------------------------- real-artifact read-only integration


def test_real_committed_evidence_selects_ordinal_1_row_631_and_dry_run_would_freeze(capsys):
    """Read-only integration test against the actual committed
    `configs/discovery/b2a_r3_candidate_manifest.json` and
    `results/decisions/b2a_r3_qualification.json` -- confirms the accepted
    Stage-B evidence still selects candidate ordinal 1
    (`test/number_theory/631.json`), that `--dry-run` reports
    `would_freeze=True`, and that neither a tokenizer is loaded nor any
    production path is written by this test."""
    import os

    from kvcot.discovery.b2a_r3_contract import QUALIFICATION_ARTIFACT_PATH, SELECTED_MANIFEST_PATH

    _skip_unless_full_git_history_available()

    with open(QUALIFICATION_ARTIFACT_PATH, "r", encoding="utf-8") as f:
        qualification_artifact = json.load(f)
    assert qualification_artifact["selection_status"] == "selected"
    assert qualification_artifact["first_passing_candidate_ordinal"] == 1
    assert qualification_artifact["selected_unique_id"] == "test/number_theory/631.json"

    manifest_before = Path(SELECTED_MANIFEST_PATH).read_bytes()
    provenance_exists_before = os.path.exists("results/decisions/b2a_r3_selection_provenance.json")

    rc = main(["freeze-b2a-r3-selected-row", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "would_freeze = True" in out
    assert "selected_ordinal = 1" in out
    assert "selected_unique_id = test/number_theory/631.json" in out
    assert "would_load_tokenizer_for_execution = False" in out
    assert "would_write_selected_manifest = False" in out
    assert "would_write_selection_provenance = False" in out

    # No production writes occurred as a side effect of this test.
    assert Path(SELECTED_MANIFEST_PATH).read_bytes() == manifest_before
    assert os.path.exists("results/decisions/b2a_r3_selection_provenance.json") == provenance_exists_before


# --------------------------------------------------------------------- source scan


def test_source_scan_new_modules_never_import_forbidden_modules_anywhere():
    """Static, source-level scan (never trusts runtime import order alone)
    of every new B2A-R3 pure module for a forbidden import ANYWHERE in the
    file (module-level or deferred inside a function body)."""
    import ast

    forbidden_modules = {"torch", "transformers", "flash_attn"}
    forbidden_dotted_prefixes = (
        "kvcot.discovery.b2a_workers",
        "kvcot.discovery.schemas",
        "kvcot.discovery.scientific_summary",
        "kvcot.generation.policies",
    )
    checked_files = [
        "src/kvcot/discovery/b2a_r3_contract.py",
        "src/kvcot/discovery/b2a_r3_candidates.py",
        "src/kvcot/discovery/b2a_r3_runtime.py",
        "src/kvcot/discovery/b2a_r3_qualification.py",
        "src/kvcot/discovery/b2a_r3_artifacts.py",
        "src/kvcot/discovery/b2a_r3_freeze.py",
        "src/kvcot/discovery/b2a_r3_authorization.py",
        "src/kvcot/discovery/b2a_r3_provenance.py",
    ]
    for relative_path in checked_files:
        path = REPO_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found_names.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found_names.add(node.module)

        top_level_hits = {n for n in found_names if n.split(".")[0] in forbidden_modules}
        dotted_hits = {
            n for n in found_names
            if any(n == prefix or n.startswith(prefix + ".") for prefix in forbidden_dotted_prefixes)
        }
        assert not top_level_hits, f"{relative_path} imports forbidden module(s): {top_level_hits}"
        assert not dotted_hits, f"{relative_path} imports forbidden submodule(s): {dotted_hits}"
