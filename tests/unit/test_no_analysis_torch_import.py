"""kvcot.analysis and kvcot.utils must run on a laptop from JSONL alone,
with no model or GPU involved (§11). This is checked statically via `ast`
(not by trying to import torch and see what breaks) so the test is
meaningful even in environments where torch happens to be installed for
other reasons.
"""
import ast
from pathlib import Path

import pytest

FORBIDDEN_MODULES = {"torch", "transformers", "flash_attn"}

CHECKED_PACKAGES = ["src/kvcot/analysis", "src/kvcot/utils"]


def _iter_py_files():
    for pkg in CHECKED_PACKAGES:
        for path in Path(pkg).rglob("*.py"):
            yield path


def _imported_top_level_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


@pytest.mark.parametrize("path", list(_iter_py_files()), ids=lambda p: str(p))
def test_no_forbidden_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = _imported_top_level_modules(tree) & FORBIDDEN_MODULES
    assert not found, f"{path} imports forbidden module(s) {found} — kvcot.analysis/kvcot.utils must stay GPU-free"


def test_at_least_one_file_was_checked():
    # Guards against the parametrize collecting zero files silently passing.
    assert len(list(_iter_py_files())) >= 5
