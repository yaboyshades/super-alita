"""Mutation-score enforcement middleware for Copilot Agent mode.

This module generates a bounded set of mutants for the target Python file,
executes the project's pytest suite against each mutant, and reports whether
the minimum mutation score threshold has been achieved.

The implementation intentionally keeps the mutation operator set small so the
runtime cost remains manageable inside an editor workflow while still catching
underdetermined tests:

* Comparison operator negation (== ↔ !=, < ↔ <=, > ↔ >=)
* Boolean constant toggling (True ↔ False)
* Arithmetic operator flipping (+ ↔ -)
* Numeric literal nudging (+1)

The entry points expose two shapes:

check_mutation_gate
    Lightweight callable returning a dictionary suitable for higher level
    quality orchestration code. This mirrors the structure used by the other
    quality extension checkers so the agent can consume it uniformly.

main
    Command line hook that can be invoked from tooling (e.g. staged from a
    Copilot Agent post-write command). It exits with a non-zero status when the
    mutation score drops below the threshold.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

_MUTATION_THRESHOLD = float(os.getenv("MUTANT_GATE_THRESHOLD", "0.9"))
_MAX_MUTANTS = int(os.getenv("MUTANT_GATE_MAX", "20"))
_PYTEST_ARGS = os.getenv("MUTANT_GATE_PYTEST_ARGS", "").split()
_PYTEST_TIMEOUT = int(os.getenv("MUTANT_GATE_TIMEOUT", "60"))  # seconds per test run
_COMPLEX_FILE_THRESHOLD = int(os.getenv("MUTANT_GATE_COMPLEX_LINES", "300"))  # lines
_EARLY_TERMINATION_SAMPLES = int(
    os.getenv("MUTANT_GATE_EARLY_SAMPLES", "5")
)  # mutants to check before early termination


@dataclass(slots=True)
class MutationReport:
    """Result of the mutation analysis."""

    score: float
    total_mutants: int
    killed_mutants: int
    survivors: List[dict]
    pytest_output: str | None = None
    early_terminated: bool = False
    timeout_occurred: bool = False
    file_complexity: str = "normal"  # "normal", "complex", "very_complex"

    @property
    def passed(self) -> bool:
        return self.total_mutants == 0 or self.score >= _MUTATION_THRESHOLD

    def as_dict(self) -> dict:
        result = {
            "passed": self.passed,
            "score": self.score,
            "total_mutants": self.total_mutants,
            "killed_mutants": self.killed_mutants,
            "survivors": self.survivors,
            "threshold": _MUTATION_THRESHOLD,
        }
        if self.early_terminated:
            result["early_terminated"] = True
        if self.timeout_occurred:
            result["timeout_occurred"] = True
        if self.file_complexity != "normal":
            result["file_complexity"] = self.file_complexity
        return result


def _assign_ids(tree: ast.AST) -> None:
    for idx, node in enumerate(ast.walk(tree)):
        setattr(node, "_mutation_id", idx)


def _clone(node: ast.AST) -> ast.AST:
    return copy.deepcopy(node)


def _comparison_mutations(node: ast.Compare) -> Iterable[ast.Compare]:
    replacements = {
        ast.Eq: ast.NotEq,
        ast.NotEq: ast.Eq,
        ast.Lt: ast.LtE,
        ast.LtE: ast.Lt,
        ast.Gt: ast.GtE,
        ast.GtE: ast.Gt,
    }
    for idx, op in enumerate(node.ops):
        replacement = replacements.get(type(op))
        if replacement is None:
            continue
        mutated = _clone(node)
        mutated.ops[idx] = replacement()
        yield mutated


def _bool_mutations(node: ast.Constant) -> Iterable[ast.Constant]:
    if isinstance(node.value, bool):
        mutated = _clone(node)
        mutated.value = not node.value
        yield mutated


def _numeric_mutations(node: ast.Constant) -> Iterable[ast.Constant]:
    if isinstance(node.value, (int, float)):
        mutated = _clone(node)
        mutated.value = node.value + 1
        yield mutated


def _binop_mutations(node: ast.BinOp) -> Iterable[ast.BinOp]:
    replacements = {
        ast.Add: ast.Sub,
        ast.Sub: ast.Add,
    }
    replacement = replacements.get(type(node.op))
    if replacement is None:
        return []
    mutated = _clone(node)
    mutated.op = replacement()
    return [mutated]


class _MutationApplier(ast.NodeTransformer):
    def __init__(self, target_id: int, replacement: ast.AST) -> None:
        self._target_id = target_id
        self._replacement = replacement

    def generic_visit(self, node: ast.AST) -> ast.AST:
        if getattr(node, "_mutation_id", None) == self._target_id:
            return copy.deepcopy(self._replacement)
        return super().generic_visit(node)


def _build_mutants(tree: ast.Module, max_mutants: int) -> List[str]:
    mutants: List[str] = []
    _assign_ids(tree)
    for node in ast.walk(tree):
        target_id = getattr(node, "_mutation_id", None)
        if target_id is None:
            continue
        mutated_nodes: Iterable[ast.AST]
        if isinstance(node, ast.Compare):
            mutated_nodes = _comparison_mutations(node)
        elif isinstance(node, ast.Constant):
            mutated_nodes = list(_bool_mutations(node)) + list(_numeric_mutations(node))
        elif isinstance(node, ast.BinOp):
            mutated_nodes = _binop_mutations(node)
        else:
            mutated_nodes = []
        for mutated_node in mutated_nodes:
            mutated_tree = copy.deepcopy(tree)
            _MutationApplier(target_id, mutated_node).visit(mutated_tree)
            ast.fix_missing_locations(mutated_tree)
            try:
                mutant_source = ast.unparse(mutated_tree)
            except Exception:  # pragma: no cover - defensive
                continue
            mutants.append(mutant_source)
            if len(mutants) >= max_mutants:
                return mutants
    return mutants


def _detect_file_complexity(path: Path) -> str:
    """Detect if file is complex based on size and AST complexity."""
    try:
        source = path.read_text(encoding="utf-8")
        line_count = len(source.splitlines())

        if line_count < _COMPLEX_FILE_THRESHOLD:
            return "normal"
        elif line_count < _COMPLEX_FILE_THRESHOLD * 2:
            return "complex"
        else:
            return "very_complex"
    except Exception:
        return "normal"


def _find_relevant_tests(target_file: Path) -> List[str]:
    """Find test files that likely test the target file."""
    test_patterns = []

    # Convert file path to module-like patterns
    rel_path = target_file.relative_to(Path.cwd())
    parts = rel_path.with_suffix("").parts

    # Look for direct test file matches
    if parts[0] == "src":
        # src/core/events.py -> tests/test_events.py, tests/core/test_events.py
        module_name = parts[-1]
        test_patterns.extend(
            [
                f"tests/test_{module_name}.py",
                f"tests/{'/'.join(parts[1:])}/test_{module_name}.py",
                f"tests/test_{'/'.join(parts[1:])}.py",
            ]
        )

    # Find existing test files that match patterns
    existing_tests = []
    for pattern in test_patterns:
        test_path = Path(pattern)
        if test_path.exists():
            existing_tests.append(str(test_path))

    return existing_tests


def _run_pytest(
    path: Path, extra_args: Sequence[str], timeout: int = _PYTEST_TIMEOUT
) -> tuple[bool, str, bool]:
    """Run pytest with timeout. Returns (success, output, timeout_occurred)."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path.cwd()))
    cmd = [sys.executable, "-m", "pytest", *extra_args]

    try:
        result = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        output = result.stdout + "\n" + result.stderr
        return result.returncode == 0, output, False
    except subprocess.TimeoutExpired:
        return False, "pytest execution timed out", True


def analyse_file(path: Path, *, max_mutants: int = _MAX_MUTANTS) -> MutationReport:
    source = path.read_text(encoding="utf-8")
    base_tree = ast.parse(source)
    mutants = _build_mutants(base_tree, max_mutants)
    if not mutants:
        return MutationReport(
            score=1.0, total_mutants=0, killed_mutants=0, survivors=[]
        )

    killed = 0
    survivors: List[dict] = []
    pytest_output = ""
    for idx, mutant_source in enumerate(mutants, start=1):
        try:
            path.write_text(mutant_source, encoding="utf-8")
            success, output = _run_pytest(path, _PYTEST_ARGS)
            pytest_output = output
        finally:
            path.write_text(source, encoding="utf-8")
        if not success:
            killed += 1
        else:
            survivors.append({"mutant_index": idx, "output": output})
    score = killed / len(mutants)
    return MutationReport(
        score=score,
        total_mutants=len(mutants),
        killed_mutants=killed,
        survivors=survivors,
        pytest_output=pytest_output,
    )


def check_mutation_gate(text: str, *, file_path: str | None = None) -> dict:
    """Return a structured report for the orchestrator.

    Parameters
    ----------
    text:
        The latest buffer content (unused – the check reads from file_path).
    file_path:
        Absolute or relative path to the file under evaluation. If omitted the
        check is a no-op that always passes.
    """

    if file_path is None:
        return {
            "passed": True,
            "score": 1.0,
            "total_mutants": 0,
            "killed_mutants": 0,
            "survivors": [],
            "threshold": _MUTATION_THRESHOLD,
            "skipped": "no file path provided",
        }

    target = Path(file_path)
    if not target.exists():
        return {
            "passed": True,
            "score": 1.0,
            "total_mutants": 0,
            "killed_mutants": 0,
            "survivors": [],
            "threshold": _MUTATION_THRESHOLD,
            "skipped": f"file {file_path} does not exist",
        }

    report = analyse_file(target)
    payload = report.as_dict()
    if not report.passed:
        payload["message"] = (
            "Mutation score below threshold; strengthen unit tests or add "
            "additional assertions."
        )
    return payload


def noop_inject(text: str) -> str:
    """Return *text* unchanged – kept for parity with other injectors."""

    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _PYTEST_ARGS  # Move to top of function
    argv = list(argv or sys.argv[1:])
    if not argv:
        print(
            "Usage: mutant_gate.py <path-to-python-file> [pytest args...]",
            file=sys.stderr,
        )
        return 2
    target = Path(argv[0]).resolve()
    extra_args = list(argv[1:]) or _PYTEST_ARGS
    if not target.exists():
        print(f"Target file {target} does not exist", file=sys.stderr)
        return 2
    if extra_args:
        _PYTEST_ARGS = extra_args
    report = analyse_file(target)
    print(json.dumps(report.as_dict(), indent=2))
    if report.passed:
        return 0
    print(
        f"Mutation score {report.score:.2%} below threshold {_MUTATION_THRESHOLD:.0%}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI hook
    sys.exit(main())
