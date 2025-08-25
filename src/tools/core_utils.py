from __future__ import annotations

import ast
import operator as op
from collections.abc import Callable
from typing import Any, Union

_Number = Union[int, float]

_ALLOWED_BIN: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}
_ALLOWED_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


class CoreUtils:
    """Core utilities: safe arithmetic and simple string ops."""

    @staticmethod
    def _eval_node(node: ast.AST) -> _Number:
        if isinstance(node, ast.Expression):
            return CoreUtils._eval_node(node.body)

        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, int | float):
                return val
            raise ValueError("only numeric literals allowed")

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            operand_result = CoreUtils._eval_node(node.operand)
            return _ALLOWED_UNARY[type(node.op)](operand_result)  # type: ignore[misc]

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN:
            left = CoreUtils._eval_node(node.left)
            right = CoreUtils._eval_node(node.right)
            # Handle division errors explicitly
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("division by zero")
            return _ALLOWED_BIN[type(node.op)](left, right)  # type: ignore[misc]

        raise ValueError(f"unsupported expression: {type(node).__name__}")

    @staticmethod
    def calculate(expression: str) -> float:
        """
        Safely evaluate arithmetic with +, -, *, /, parentheses, and unary +/-.
        Examples:
            '10 + 5 * 2' -> 20
            '(2+3)*4'    -> 20
            '-3.5 + 2'   -> -1.5
        """
        expr = expression.strip()
        if not expr:
            raise ValueError("empty expression")

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"invalid expression: {e}") from e

        result = CoreUtils._eval_node(tree)
        # normalize ints-as-floats to float for consistent API
        return float(result)

    @staticmethod
    def reverse_string(text: str) -> str:
        """Reverse a string. Non-strings are coerced to str."""
        return str(text)[::-1]
