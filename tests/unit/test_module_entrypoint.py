import runpy

import pytest


def test_python_module_entrypoint_delegates_to_cli_main(monkeypatch):
    monkeypatch.setattr("kvcot.cli.main", lambda: 17)
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("kvcot", run_name="__main__")
    assert caught.value.code == 17
