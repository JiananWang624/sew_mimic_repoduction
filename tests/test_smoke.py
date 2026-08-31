import importlib

import pytest


@pytest.mark.parametrize("module_name", ["numpy", "scipy", "mujoco"])
def test_required_runtime_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__version__
