from __future__ import annotations

import warnings
from typing import Any

from src.sandbox.exec_sandbox import evaluate_expression, execute_statements


def safe_eval(expr: str) -> Any:
    warnings.warn(
        "dynamic_eval.safe_eval is deprecated; use sandbox.evaluate_expression",
        DeprecationWarning,
        stacklevel=2,
    )
    return evaluate_expression(expr)


def safe_exec(code: str) -> None:
    warnings.warn(
        "dynamic_eval.safe_exec is deprecated; use sandbox.execute_statements",
        DeprecationWarning,
        stacklevel=2,
    )
    execute_statements(code)
