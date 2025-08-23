from __future__ import annotations

import ast
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from src.sandbox.registry import ALLOWLIST

SAFE_BUILTINS: Mapping[str, Any] = MappingProxyType(
    {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
    }
)


def _forbid_nodes(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise ValueError("Imports are not allowed in sandbox.")
        if isinstance(node, ast.Attribute):
            raise ValueError("Attribute access is not allowed in sandbox.")
        if isinstance(node, ast.Subscript):
            raise ValueError("Subscript is not allowed in sandbox.")


def evaluate_expression(expr: str) -> Any:
    """
    Evaluate a simple expression with a locked-down environment.
    Only allow safe builtins and allowlisted functions by name.
    """
    tree = ast.parse(expr, mode="eval")
    _forbid_nodes(tree)
    code = compile(tree, "<sandbox-expr>", "eval")
    env: dict[str, Any] = {"__builtins__": {}}
    env.update(dict(ALLOWLIST))
    env.update(SAFE_BUILTINS)
    return eval(code, env, {})


def execute_statements(code_str: str) -> None:
    """
    Execute statements with strict restrictions (no import/attr/subscript).
    Function calls must be to allowlisted names only.
    """
    tree = ast.parse(code_str, mode="exec")
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Import | ast.ImportFrom | ast.Attribute | ast.Subscript
        ):
            raise ValueError("Prohibited syntax in sandboxed code.")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in ALLOWLIST
        ):
            raise ValueError("Calls only allowed to allowlisted functions.")
    code = compile(tree, "<sandbox-exec>", "exec")
    env: dict[str, Any] = {"__builtins__": {}}
    env.update(dict(ALLOWLIST))
    env.update(SAFE_BUILTINS)
    exec(code, env, {})
