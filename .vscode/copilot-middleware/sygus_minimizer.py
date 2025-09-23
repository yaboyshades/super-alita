"""SyGuS-inspired minimiser middleware for Copilot Agent mode."""

from __future__ import annotations

import ast
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import sympy as sp
    from sympy.printing.pycode import pycode
except ImportError:  # pragma: no cover - optional dependency
    sp = None
    pycode = None

_MAX_SAMPLES = 8
_RANDOM_SEED = 1337


def _find_targets(source: str) -> List[int]:
    targets: List[int] = []
    for idx, line in enumerate(source.splitlines(), start=1):
        if '# sygus:minimize' in line:
            targets.append(idx)
    return targets


def _functions_after(lines: List[str], markers: List[int]) -> List[str]:
    targets: List[str] = []
    marker_set = set(markers)
    for idx, line in enumerate(lines, start=1):
        if idx in marker_set:
            # grab next function def name
            for future in lines[idx:]:
                future_strip = future.strip()
                if future_strip.startswith('def '):
                    name = future_strip.split('def ', 1)[1].split('(')[0]
                    targets.append(name)
                    break
    return targets


@dataclass
class SimplificationResult:
    function: str
    original_expr: str
    simplified_expr: str
    applied: bool


def _ast_to_sympy(expr: ast.AST, symbols: Dict[str, 'sp.Symbol']) -> Optional['sp.Expr']:
    if sp is None:
        return None
    if isinstance(expr, ast.BinOp):
        left = _ast_to_sympy(expr.left, symbols)
        right = _ast_to_sympy(expr.right, symbols)
        if left is None or right is None:
            return None
        op = expr.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.Pow):
            return left ** right
        return None
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        operand = _ast_to_sympy(expr.operand, symbols)
        if operand is None:
            return None
        return -operand
    if isinstance(expr, ast.Name):
        return symbols.get(expr.id)
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
        return sp.Integer(expr.value) if isinstance(expr.value, int) else sp.Float(expr.value)
    return None


def _simplify_return(function: ast.FunctionDef, source: str) -> Optional[SimplificationResult]:
    if sp is None or pycode is None:
        return None
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return) and node.value is not None]
    if len(returns) != 1:
        return None
    return_node = returns[0]
    symbols = {arg.arg: sp.symbols(arg.arg) for arg in function.args.args if isinstance(arg, ast.arg)}
    sym_expr = _ast_to_sympy(return_node.value, symbols)
    if sym_expr is None:
        return None
    simplified = sp.simplify(sym_expr)
    simplified_code = pycode(simplified)
    original_code = ast.unparse(return_node.value)
    if len(simplified_code) >= len(original_code):
        return SimplificationResult(function.name, original_code, simplified_code, False)
    new_expr = ast.parse(simplified_code, mode='eval').body
    ast.copy_location(new_expr, return_node.value)
    return_node.value = new_expr
    return SimplificationResult(function.name, original_code, simplified_code, True)


def minimise_with_sygus(text: str, *, file_path: str | None = None) -> dict:
    markers = _find_targets(text)
    if not markers:
        return {"passed": True, "skipped": "no sygus markers"}
    lines = text.splitlines()
    target_names = set(_functions_after(lines, markers))
    tree = ast.parse(text)
    applied: List[Dict[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_names:
            result = _simplify_return(node, text)
            if result and result.applied:
                applied.append({
                    "function": result.function,
                    "original": result.original_expr,
                    "simplified": result.simplified_expr,
                })
    if not applied:
        return {"passed": True, "skipped": "no simplifications applied"}
    new_source = ast.unparse(tree)
    if file_path:
        Path(file_path).write_text(new_source, encoding='utf-8')
    return {"passed": True, "applied": applied, "updated_source": new_source}


def main(argv: List[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv:
        print('Usage: sygus_minimizer.py <python-file>', file=sys.stderr)
        return 2
    target = Path(argv[0])
    if not target.exists():
        print(f'File {target} not found', file=sys.stderr)
        return 2
    source = target.read_text(encoding='utf-8')
    report = minimise_with_sygus(source, file_path=str(target))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':  # pragma: no cover - CLI hook
    sys.exit(main())
