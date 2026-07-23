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
