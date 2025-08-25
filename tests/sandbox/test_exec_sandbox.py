from __future__ import annotations

import pytest

from src.sandbox.exec_sandbox import evaluate_expression, execute_statements


def test_eval_math_ok():
    assert evaluate_expression("1 + 2 + max(3,4)") == 1 + 2 + max(3, 4)


def test_eval_blocks_imports_and_attr():
    with pytest.raises(Exception):
        evaluate_expression("__import__('os').system('echo x')")
    with pytest.raises(Exception):
        evaluate_expression("(1).bit_length")


def test_exec_allows_allowlisted_call():
    # `pow` is allowlisted with a guard
    execute_statements("pow(2, 8)")


def test_exec_blocks_unlisted_calls():
    with pytest.raises(Exception):
        execute_statements("open('x','w')")


def test_exec_blocks_imports():
    with pytest.raises(Exception):
        execute_statements("import os")
