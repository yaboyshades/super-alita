"""AST-driven constitutional compliance checks."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List

from pydantic import BaseModel, Field  # type: ignore[import-not-found]


class ConstitutionalViolation(BaseModel):
    """A violation of Super Alita constitutional principles."""

    article: str
    rule: str
    line_number: int = Field(0, ge=0)
    message: str
    severity: str


class ASTConstitutionalValidator:
    """Validate source code against constitutional rules via AST analysis."""

    def __init__(self, constitution_path: Path) -> None:
        self._constitution_path = constitution_path
        try:
            self._constitution_text = constitution_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._constitution_text = ""
        self._violations: list[ConstitutionalViolation] = []

    def validate(self, code: str, *, filename: str = "<string>") -> list[ConstitutionalViolation]:
        """Run the validator and return collected violations."""

        self._violations = []
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as exc:
            self._violations.append(
                ConstitutionalViolation(
                    article="Syntax",
                    rule="valid_python",
                    line_number=getattr(exc, "lineno", 0) or 0,
                    message=f"Syntax error: {exc.msg}",
                    severity="error",
                )
            )
            return self._violations

        self._check_article_i(tree)
        self._check_article_ii(tree)
        self._check_article_iv(tree)
        self._check_article_v(tree)
        self._check_article_vi(tree)
        return self._violations

    def calculate_constitutional_score(self) -> float:
        """Calculate the compliance score based on collected violations."""

        error_penalty = sum(0.15 for item in self._violations if item.severity == "error")
        warning_penalty = sum(0.05 for item in self._violations if item.severity == "warning")
        score = 1.0 - (error_penalty + warning_penalty)
        return max(0.0, round(score, 4))

    def _check_article_i(self, tree: ast.AST) -> None:
        """Article I: guard against mocks and placeholders."""

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "mock" in node.module.lower():
                self._violations.append(
                    ConstitutionalViolation(
                        article="Article I",
                        rule="no_mocks",
                        line_number=node.lineno,
                        message=f"Mock import '{node.module}' violates Article I",
                        severity="error",
                    )
                )

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and "mock" in func.id.lower():
                    self._violations.append(
                        ConstitutionalViolation(
                            article="Article I",
                            rule="no_mocks",
                            line_number=node.lineno,
                            message=f"Mock usage '{func.id}' violates Article I",
                            severity="error",
                        )
                    )
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                exc_func = node.exc.func
                if isinstance(exc_func, ast.Name) and exc_func.id == "NotImplementedError":
                    self._violations.append(
                        ConstitutionalViolation(
                            article="Article I",
                            rule="no_placeholders",
                            line_number=node.lineno,
                            message="NotImplementedError violates Article I",
                            severity="error",
                        )
                    )
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                value = node.value.value
                if isinstance(value, str):
                    upper_value = value.upper()
                    if "TODO" in upper_value or "FIXME" in upper_value:
                        self._violations.append(
                            ConstitutionalViolation(
                                article="Article I",
                                rule="no_placeholders",
                                line_number=node.lineno,
                                message="TODO/FIXME comments violate Article I",
                                severity="warning",
                            )
                        )

    def _check_article_ii(self, tree: ast.AST) -> None:
        """Article II: ensure mandatory type annotations."""

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in {"__init__", "__new__"}:
                    continue
                if node.returns is None:
                    self._violations.append(
                        ConstitutionalViolation(
                            article="Article II",
                            rule="type_annotations",
                            line_number=node.lineno,
                            message=f"Function '{node.name}' missing return annotation",
                            severity="error",
                        )
                    )
                for arg in node.args.args:
                    if arg.arg in {"self", "cls"}:
                        continue
                    if arg.annotation is None:
                        self._violations.append(
                            ConstitutionalViolation(
                                article="Article II",
                                rule="type_annotations",
                                line_number=node.lineno,
                                message=f"Parameter '{arg.arg}' in '{node.name}' missing type annotation",
                                severity="error",
                            )
                        )

    def _check_article_iv(self, tree: ast.AST) -> None:
        """Article IV: catch broad exception handling."""

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    self._violations.append(
                        ConstitutionalViolation(
                            article="Article IV",
                            rule="error_handling",
                            line_number=node.lineno,
                            message="Bare 'except:' is disallowed by Article IV",
                            severity="error",
                        )
                    )
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    self._violations.append(
                        ConstitutionalViolation(
                            article="Article IV",
                            rule="error_handling",
                            line_number=node.lineno,
                            message="Silent exception handling violates Article IV",
                            severity="error",
                        )
                    )

    def _check_article_v(self, tree: ast.AST) -> None:
        """Article V: ensure sandboxed dynamic execution."""

        unsafe_calls = {"eval", "exec", "compile", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in unsafe_calls:
                    self._violations.append(
                        ConstitutionalViolation(
                            article="Article V",
                            rule="sandbox_required",
                            line_number=node.lineno,
                            message=f"Use of '{func.id}' requires sandbox enforcement",
                            severity="error",
                        )
                    )

    def _check_article_vi(self, tree: ast.AST) -> None:
        """Article VI: enforce documentation expectations."""

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue
                if ast.get_docstring(node) is None:
                    self._violations.append(
                        ConstitutionalViolation(
                            article="Article VI",
                            rule="documentation",
                            line_number=node.lineno,
                            message=f"Public function '{node.name}' missing docstring",
                            severity="warning",
                        )
                    )
            if isinstance(node, ast.ClassDef):
                if ast.get_docstring(node) is None:
                    self._violations.append(
                        ConstitutionalViolation(
                            article="Article VI",
                            rule="documentation",
                            line_number=node.lineno,
                            message=f"Class '{node.name}' missing docstring",
                            severity="warning",
                        )
                    )

    @property
    def constitution_text(self) -> str:
        """Expose loaded constitution text for downstream priming."""

        return self._constitution_text

    @property
    def violations(self) -> List[ConstitutionalViolation]:
        """Return a copy of the latest violations list."""

        return list(self._violations)
