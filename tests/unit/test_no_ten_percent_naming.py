"""§9: "No condition may be named 'R-KV 10%'." The condition is
`RKV-B{budget}` (schema/config spelling: `rkv_b{budget}`). This test both
enforces the validator that rejects it structurally and greps the places
conditions are actually named (YAML `conditions:` lists, Python source
constructing condition-name strings) for a stray percent-style name — a
docstring or an audit document *talking about* the ban (like this file, or
docs/UPSTREAM_AUDIT.md §H7) is not itself a condition-name position and is
intentionally not flagged.
"""
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kvcot.config import StageConfig

REPO_ROOT = Path(__file__).resolve().parents[2]

# Structural places a condition name is actually assigned/listed.
CONDITION_STRING_ASSIGNMENT_RE = re.compile(
    r"""condition\s*=\s*f?["']([^"']*)["']""", re.IGNORECASE
)


def test_validator_rejects_percent_condition_name():
    with pytest.raises(ValidationError):
        StageConfig(
            stage_name="bad",
            dataset_manifest="x.jsonl",
            conditions=["R-KV 10%"],
            output_dir="out",
        )


def test_no_percent_in_any_stage_yaml_conditions_list():
    for path in sorted(REPO_ROOT.glob("configs/stage*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        conditions = raw.get("conditions", [])
        for c in conditions:
            assert "%" not in c, f"{path}: condition {c!r} contains '%'"


def test_no_percent_in_python_condition_string_assignments():
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in CONDITION_STRING_ASSIGNMENT_RE.finditer(text):
            if "%" in match.group(1):
                offenders.append((path, match.group(0)))
    assert not offenders, f"condition-name string literals containing '%': {offenders}"


def test_no_literal_rkv_ten_percent_phrase_outside_the_audit_and_this_test():
    # The exact banned phrase from the brief, as a hard backstop. Excludes
    # docs/UPSTREAM_AUDIT.md (which quotes it while explaining the ban) and
    # this test file itself (which names it to test for it).
    banned = "R-KV 10%"
    exempt = {
        REPO_ROOT / "docs" / "UPSTREAM_AUDIT.md",
        REPO_ROOT / "CLAUDE.md",
        Path(__file__).resolve(),
    }
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if path in exempt or not path.is_file():
            continue
        if any(part in {".git", ".venv", "third_party", "results", ".hf_cache_tmp"} for part in path.parts):
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if banned in text:
            offenders.append(path)
    assert not offenders, f"banned phrase {banned!r} found in: {offenders}"
