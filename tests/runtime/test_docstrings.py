import inspect

from reug_runtime.config import _getenv
from reug_runtime.router import execute_turn
from reug_runtime.router_tools import reug_start_turn


def _assert_google_doc(func, sections):
    doc = inspect.getdoc(func)
    assert doc is not None, f"{func.__name__} missing docstring"
    for sec in sections:
        assert f"{sec}:" in doc, f"{func.__name__} docstring missing '{sec}:' section"


def test_config_getenv_docstring():
    _assert_google_doc(_getenv, ["Args", "Returns"])


def test_router_execute_turn_docstring():
    _assert_google_doc(execute_turn, ["Args", "Yields"])


def test_router_tools_start_turn_docstring():
    _assert_google_doc(reug_start_turn, ["Args", "Returns"])
