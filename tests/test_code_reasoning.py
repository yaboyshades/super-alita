"""
Tests for Code Reasoning Module

Tests the mangle-style code analysis capabilities including:
- Code ingestion and fact extraction
- Rule engine execution
- Integration with unified intelligence pipeline
"""

import os
import tempfile

import pytest

from src.unified_intelligence.code_reasoning.ingester import CodeIngester
from src.unified_intelligence.code_reasoning.models import (
    CodeAnalysisRequest,
    CodeAnalysisResponse,
    Finding,
)
from src.unified_intelligence.code_reasoning.rules import RuleEngine


class TestCodeIngester:
    """Test code ingestion functionality."""

    def test_ingest_simple_function(self):
        """Test ingestion of a simple Python function."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test file
            test_file = os.path.join(tmp_dir, "test.py")
            with open(test_file, "w") as f:
                f.write(
                    """
def simple_function():
    return 42

def complex_function(x):
    for i in range(x):
        if i % 2 == 0:
            x += i
    return x
"""
                )

            # Use file-based database for testing since in-memory closes
            db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
            db_file.close()

            ingester = CodeIngester(db_file.name)
            stats = ingester.ingest_repository(tmp_dir)

            print(f"Stats: {stats}")

            # Debug: Check what calls were extracted
            with ingester.get_db_connection() as conn:
                calls = conn.execute("SELECT * FROM calls").fetchall()
                print(f"Calls found: {calls}")
                symbols = conn.execute("SELECT * FROM symbol").fetchall()
                print(f"Symbols found: {symbols}")

            assert stats["files_processed"] == 1
            assert stats["symbols_extracted"] == 2  # Two functions
            # Note: range() is a function call, so we expect 1 call
            assert stats["calls_extracted"] == 1  # range() function call
        """Test ingestion with import statements."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files
            main_file = os.path.join(tmp_dir, "main.py")
            with open(main_file, "w") as f:
                f.write(
                    """
import json
from typing import List

def process_data(data: List[str]) -> str:
    return json.dumps(data)
"""
                )

            ingester = CodeIngester(":memory:")
            stats = ingester.ingest_repository(tmp_dir)

            assert stats["files_processed"] == 1
            assert stats["imports_extracted"] == 2  # json and typing

    def test_ingest_with_tests(self):
        """Test ingestion of test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create source file
            source_file = os.path.join(tmp_dir, "calc.py")
            with open(source_file, "w") as f:
                f.write(
                    """
def add(a, b):
    return a + b
"""
                )

            # Create test file
            test_file = os.path.join(tmp_dir, "tests", "test_calc.py")
            os.makedirs(os.path.dirname(test_file))
            with open(test_file, "w") as f:
                f.write(
                    """
from calc import add

def test_add():
    assert add(1, 2) == 3
"""
                )

            ingester = CodeIngester(":memory:")
            stats = ingester.ingest_repository(tmp_dir, include_tests=True)

            assert stats["files_processed"] == 2
            assert stats["symbols_extracted"] == 2  # add function and test_add
            assert stats["tests_extracted"] == 1

    def test_complexity_calculation(self):
        """Test complexity score calculation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_file = os.path.join(tmp_dir, "complex.py")
            with open(test_file, "w") as f:
                f.write(
                    """
def simple():
    return 1

def complex_function(x):
    # Multiple control structures = higher complexity
    for i in range(x):
        if i % 2 == 0:
            if i % 3 == 0:
                try:
                    x += i
                except:
                    pass
            else:
                x -= i
    return x
"""
                )

            db_path = os.path.join(tmp_dir, "facts.db")
            ingester = CodeIngester(db_path)
            ingester.ingest_repository(tmp_dir)

            # Check that complexity scores are reasonable
            with ingester.get_db_connection() as conn:
                complexities = conn.execute(
                    "SELECT score FROM complexity"
                ).fetchall()
                scores = [row["score"] for row in complexities]

                assert len(scores) == 2
                # Simple function should have low complexity
                # Complex function should have higher complexity
                assert all(0 <= score <= 1 for score in scores)


class TestRuleEngine:
    """Test rule engine functionality."""

    def test_default_rules_loaded(self):
        """Test that default rules are loaded."""
        rule_engine = RuleEngine(":memory:")

        expected_rules = [
            "untested_complex_functions",
            "orphan_complex",
            "circular_dependencies",
            "hot_paths",
            "config_cascade_breaks",
            "reinvention_json",
        ]

        for rule in expected_rules:
            assert rule in rule_engine.rules

    def test_legacy_rule_aliases_exposed(self):
        """Ensure legacy rule aliases remain available for compatibility."""
        rule_engine = RuleEngine(":memory:")

        for alias in ["untested_function", "cycle", "hot_path"]:
            assert alias in rule_engine.rules

    def test_add_custom_rule(self):
        """Test adding a custom rule."""
        rule_engine = RuleEngine(":memory:")

        custom_sql = "SELECT * FROM symbol WHERE kind = 'function'"
        rule_engine.add_rule("custom_rule", custom_sql)

        assert "custom_rule" in rule_engine.rules
        assert rule_engine.rules["custom_rule"] == custom_sql

    def test_run_rule_on_empty_db(self):
        """Test running rules on empty database."""
        rule_engine = RuleEngine(":memory:")

        findings = rule_engine.run_all_rules()

        assert isinstance(findings, dict)
        # Should have entries for all rules, even if empty
        canonical_count = len(
            {
                rule_engine._canonical_rule_name(name)
                for name in rule_engine.rules
            }
        )
        assert len(findings) == canonical_count

    def test_run_specific_rule(self):
        """Test running a specific rule."""
        rule_engine = RuleEngine(":memory:")

        findings = rule_engine.run_rule("untested_complex_functions")

        assert isinstance(findings, list)

    def test_run_specific_rule_alias(self):
        """Legacy alias should produce identical findings."""
        rule_engine = RuleEngine(":memory:")

        alias_findings = rule_engine.run_rule("untested_function")
        canonical_findings = rule_engine.run_rule("untested_complex_functions")

        assert alias_findings == canonical_findings

    def test_remove_rule(self):
        """Test removing a rule."""
        rule_engine = RuleEngine(":memory:")

        initial_count = len(rule_engine.rules)
        rule_engine.remove_rule("untested_complex_functions")

        assert "untested_complex_functions" not in rule_engine.rules
        assert len(rule_engine.rules) == initial_count - 1


class TestCodeAnalysisIntegration:
    """Test integration between ingester and rule engine."""

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test repository
            main_file = os.path.join(tmp_dir, "main.py")
            with open(main_file, "w") as f:
                f.write(
                    """
import json

def untested_complex_function(x):
    # Complex but untested
    for i in range(x):
        if i % 2 == 0:
            if i % 3 == 0:
                x += i
    return x

def tested_simple_function():
    return 42
"""
                )

            test_file = os.path.join(tmp_dir, "tests", "test_main.py")
            os.makedirs(os.path.dirname(test_file))
            with open(test_file, "w") as f:
                f.write(
                    """
from main import tested_simple_function

def test_simple():
    assert tested_simple_function() == 42
"""
                )

            # Run full pipeline
            db_path = os.path.join(tmp_dir, "facts.db")
            ingester = CodeIngester(db_path)
            stats = ingester.ingest_repository(tmp_dir, include_tests=True)

            rule_engine = RuleEngine(db_path)
            findings = rule_engine.run_all_rules()

            # Verify results
            assert stats["files_processed"] == 2
            assert isinstance(findings, dict)

            # Should find untested complex function
            untested = findings.get("untested_complex_functions", [])
            assert len(untested) >= 1

    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create circular dependency
            a_file = os.path.join(tmp_dir, "a.py")
            with open(a_file, "w") as f:
                f.write("import b\ndef func_a(): pass")

            b_file = os.path.join(tmp_dir, "b.py")
            with open(b_file, "w") as f:
                f.write("import a\ndef func_b(): pass")

            # Run analysis
            db_path = os.path.join(tmp_dir, "facts.db")
            ingester = CodeIngester(db_path)
            ingester.ingest_repository(tmp_dir)

            rule_engine = RuleEngine(db_path)
            findings = rule_engine.run_all_rules()

            # Should detect circular dependency
            cycles = findings.get("circular_dependencies", [])
            assert len(cycles) >= 2  # Both directions of the cycle

    def test_config_cascade_break_detection(self):
        """Test that config cascade rule flags untested settings helpers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            settings_file = os.path.join(tmp_dir, "src", "core", "settings.py")
            os.makedirs(os.path.dirname(settings_file))
            with open(settings_file, "w", encoding="utf-8") as handle:
                handle.write(
                    """
from typing import Any


def load_settings(env: dict[str, Any]) -> dict[str, Any]:
    data = {}
    for key, value in env.items():
        data[key.lower()] = value
    return data
"""
                )

            db_path = os.path.join(tmp_dir, "facts.db")
            ingester = CodeIngester(db_path)
            ingester.ingest_repository(tmp_dir, include_tests=True)

            rule_engine = RuleEngine(db_path)
            findings = rule_engine.run_rule("config_cascade_breaks")

            assert any(f.file.endswith("settings.py") for f in findings)


class TestModels:
    """Test data models."""

    def test_finding_model(self):
        """Test Finding model creation."""
        finding = Finding(
            rule_name="test_rule",
            symbol="test::function",
            file="test.py",
            complexity=0.75,
            indegree=5,
            file_a="a.py",
            file_b="b.py",
            metadata={"extra": "data"},
        )

        assert finding.rule_name == "test_rule"
        assert finding.symbol == "test::function"
        assert finding.complexity == 0.75

    def test_code_analysis_request(self):
        """Test CodeAnalysisRequest model."""
        request = CodeAnalysisRequest(
            repo_path="/test/path",
            include_tests=True,
            rules_to_run=[
                "untested_complex_functions",
                "circular_dependencies",
            ],
        )

        assert request.repo_path == "/test/path"
        assert request.include_tests is True
        if request.rules_to_run:
            assert len(request.rules_to_run) == 2

    def test_code_analysis_response(self):
        """Test CodeAnalysisResponse model."""
        findings = {
            "untested_complex_functions": [
                Finding(
                    rule_name="untested_complex_functions",
                    symbol="test::func",
                    file="test.py",
                    complexity=0.8,
                    indegree=0,
                    file_a=None,
                    file_b=None,
                )
            ]
        }

        response = CodeAnalysisResponse(
            repo_path="/test/path",
            total_files=5,
            total_symbols=20,
            findings=findings,
            summary={"untested_complex_functions": 1},
            analysis_time=1.5,
            success=True,
            error_message=None,
        )

        assert response.total_files == 5
        assert response.success is True
        assert response.summary["untested_complex_functions"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
