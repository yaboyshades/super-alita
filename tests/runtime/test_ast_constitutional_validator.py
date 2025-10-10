"""Unit tests for the AST-based constitutional validator."""

from __future__ import annotations

from pathlib import Path

from src.quality_gauntlet.constitutional import ASTConstitutionalValidator


def test_validator_flags_not_implemented_error(tmp_path: Path) -> None:
    constitution = tmp_path / "CONSTITUTION.md"
    constitution.write_text("Article I: No placeholders", encoding="utf-8")
    validator = ASTConstitutionalValidator(constitution)

    code = """
class Sample:
    def feature(self) -> None:
        raise NotImplementedError
"""

    violations = validator.validate(code)
    assert any(v.rule == "no_placeholders" for v in violations)
    assert validator.calculate_constitutional_score() <= 0.85


def test_validator_warnings_for_missing_docstrings(tmp_path: Path) -> None:
    constitution = tmp_path / "CONSTITUTION.md"
    constitution.write_text("Article VI: Document your code", encoding="utf-8")
    validator = ASTConstitutionalValidator(constitution)

    code = """
def add_numbers(a: int, b: int) -> int:
    return a + b
"""

    violations = validator.validate(code)
    assert any(v.rule == "documentation" for v in violations)
    score = validator.calculate_constitutional_score()
    assert 0.9 <= score <= 1.0
