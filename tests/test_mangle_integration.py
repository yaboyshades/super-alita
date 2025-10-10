"""
Comprehensive test suite for Mangle integration components.

Tests all aspects of the Mangle-enhanced SDD framework:
- MangleFactGenerator
- MangleReasoner
- Enhanced SDD Framework
- CLI functionality
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.sdd.enhanced_sdd_framework import EnhancedSDDFramework
from src.sdd.mangle_integration import MangleFactGenerator
from src.sdd.mangle_reasoner import MangleQueryResult, MangleReasoner
from src.sdd.mangle_rules import get_available_queries, get_query_for_question
from src.sdd.models import SpecifyRequest


class TestMangleFactGenerator:
    """Test the Mangle fact generator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = MangleFactGenerator(str(self.temp_dir))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test fact generator initialization."""
        assert self.generator.workspace_root == self.temp_dir
        assert self.generator.specs_dir == self.temp_dir / "specs"
        assert self.generator.src_dir == self.temp_dir / "src"
        assert len(self.generator._generated_facts) == 0

    def test_generate_code_facts_empty_directory(self):
        """Test code fact generation with empty source directory."""
        facts = self.generator.generate_code_facts()
        assert facts == ""

    def test_generate_code_facts_with_python_file(self):
        """Test code fact generation with Python files."""
        # Create source directory and file
        src_dir = self.temp_dir / "src"
        src_dir.mkdir()

        python_file = src_dir / "example.py"
        python_file.write_text(
            '''
def hello_world():
    """A simple hello world function."""
    return "Hello, World!"

class ExampleClass:
    def method(self):
        pass
'''
        )

        facts = self.generator.generate_code_facts()

        # Check that facts were generated
        assert facts
        assert 'function("hello_world"' in facts
        assert 'class("ExampleClass"' in facts
        assert "module(" in facts

    def test_generate_spec_facts_empty_directory(self):
        """Test spec fact generation with empty specs directory."""
        facts = self.generator.generate_spec_facts()
        assert facts == ""

    def test_generate_spec_facts_with_specification(self):
        """Test spec fact generation with specification files."""
        # Create specs directory and file
        specs_dir = self.temp_dir / "specs"
        feature_dir = specs_dir / "001-example-feature"
        feature_dir.mkdir(parents=True)

        spec_file = feature_dir / "feature-spec.md"
        spec_file.write_text(
            """
# Feature Specification: Example Feature

## Acceptance Criteria

- [ ] Feature should work correctly
- [ ] Feature should be tested

## Requirements

- Must be fast
- Must be reliable
"""
        )

        facts = self.generator.generate_spec_facts()

        # Check that facts were generated
        assert facts
        assert 'spec("001-example-feature"' in facts
        assert "acceptance_criterion(" in facts
        assert "requirement(" in facts

    def test_generate_constitution_facts(self):
        """Test constitution fact generation."""
        # Create constitution file
        constitution_file = self.temp_dir / ".github" / "CONSTITUTION.md"
        constitution_file.parent.mkdir(parents=True)
        constitution_file.write_text(
            """
## Article I: Library-First Development

**Mandate**: Use existing libraries when possible.

## Article II: Test-First Development

**Mandate**: Write tests before implementation.
"""
        )

        facts = self.generator.generate_constitution_facts()

        # Check that facts were generated
        assert facts
        assert 'constitution_article("I"' in facts
        assert 'constitution_article("II"' in facts

    def test_escape_string(self):
        """Test string escaping functionality."""
        test_string = 'Hello "World" with\nnewlines'
        escaped = self.generator._escape_string(test_string)

        assert '"' not in escaped or '\\"' in escaped
        assert "\n" not in escaped or "\\n" in escaped

    def test_get_fact_statistics(self):
        """Test fact statistics generation."""
        # Generate some facts first
        self.generator._add_fact('test_fact("example")')
        self.generator._add_fact('another_fact("test")')

        stats = self.generator.get_fact_statistics()

        assert isinstance(stats, dict)
        assert "test_fact" in stats
        assert "another_fact" in stats


class TestMangleReasoner:
    """Test the Mangle reasoner."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reasoner = MangleReasoner(str(self.temp_dir))

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test reasoner initialization."""
        assert self.reasoner.workspace_root == self.temp_dir
        assert isinstance(self.reasoner.fact_generator, MangleFactGenerator)
        assert self.reasoner._facts_cache is None
        assert len(self.reasoner._query_cache) == 0

    def test_query_result_initialization(self):
        """Test MangleQueryResult initialization."""
        result = MangleQueryResult(
            query="test_query(X)",
            raw_results=[["result1"], ["result2"]],
            success=True,
            execution_time=0.5,
        )

        assert result.query == "test_query(X)"
        assert len(result.raw_results) == 2
        assert result.success is True
        assert result.execution_time == 0.5

    def test_query_result_formatting(self):
        """Test query result formatting."""
        result = MangleQueryResult(
            query="test_query(X)",
            raw_results=[["result1"], ["result2"]],
            success=True,
        )

        formatted = result.format_for_display()
        assert "test_query(X)" in formatted
        assert "result1" in formatted
        assert "result2" in formatted

    def test_ask_question_with_known_pattern(self):
        """Test asking questions with known patterns."""
        with patch.object(self.reasoner, "query") as mock_query:
            mock_query.return_value = MangleQueryResult(
                query="untested_function(Func)",
                raw_results=[["test_func"]],
                success=True,
            )

            result = self.reasoner.ask_question("what functions are untested")

            assert result.success is True
            mock_query.assert_called_once()

    def test_ask_question_with_unknown_pattern(self):
        """Test asking questions with unknown patterns."""
        result = self.reasoner.ask_question("this is an unknown question")

        assert result.success is False
        assert "Could not understand question" in result.error_message

    @patch("subprocess.run")
    def test_query_with_mangle_success(self, mock_run):
        """Test successful Mangle query execution."""
        # Mock successful subprocess execution
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = '("result1")\n("result2")\n'
        mock_run.return_value.stderr = ""

        result = self.reasoner.query("test_query(X)")

        assert result.success is True
        assert len(result.raw_results) == 2
        assert result.raw_results[0] == ["result1"]
        assert result.raw_results[1] == ["result2"]

    @patch("subprocess.run")
    def test_query_with_mangle_failure(self, mock_run):
        """Test failed Mangle query execution."""
        # Mock failed subprocess execution
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "Error: syntax error"

        result = self.reasoner.query("invalid_query")

        assert result.success is False
        assert "syntax error" in result.error_message

    def test_validate_constitutional_compliance(self):
        """Test constitutional compliance validation."""
        with patch.object(self.reasoner, "query") as mock_query:
            mock_query.return_value = MangleQueryResult(
                query="mock_query", raw_results=[], success=True
            )

            results = self.reasoner.validate_constitutional_compliance()

            assert isinstance(results, dict)
            assert len(results) > 0
            # Should have called query multiple times for different articles
            assert mock_query.call_count >= 6

    def test_find_quality_issues(self):
        """Test quality issue detection."""
        with patch.object(self.reasoner, "query") as mock_query:
            mock_query.return_value = MangleQueryResult(
                query="mock_query",
                raw_results=[["issue1"], ["issue2"]],
                success=True,
            )

            results = self.reasoner.find_quality_issues()

            assert isinstance(results, dict)
            assert "Untested Functions" in results
            assert "Complex Functions" in results

    def test_cache_functionality(self):
        """Test query caching."""
        with patch.object(
            self.reasoner, "_execute_mangle_query"
        ) as mock_execute:
            mock_execute.return_value = MangleQueryResult(
                query="test_query",
                raw_results=[["cached_result"]],
                success=True,
            )

            # First query should execute
            result1 = self.reasoner.query("test_query")
            assert mock_execute.call_count == 1

            # Second identical query should use cache
            result2 = self.reasoner.query("test_query")
            assert mock_execute.call_count == 1  # Still 1, not 2

            assert result1.raw_results == result2.raw_results

    def test_invalidate_cache(self):
        """Test cache invalidation."""
        # Add something to cache
        self.reasoner._query_cache["test"] = MangleQueryResult(
            "test", [], True
        )
        self.reasoner._facts_cache = "cached facts"
        self.reasoner._facts_cache_valid = True

        self.reasoner.invalidate_cache()

        assert len(self.reasoner._query_cache) == 0
        assert self.reasoner._facts_cache is None
        assert self.reasoner._facts_cache_valid is False


class TestMangleRules:
    """Test the Mangle rules module."""

    def test_get_query_for_question(self):
        """Test natural language to query mapping."""
        # Test known patterns
        assert (
            get_query_for_question("what functions are untested")
            == "untested_function(Func)"
        )
        assert (
            get_query_for_question("what features are incomplete")
            == "incomplete_feature(FeatureID)"
        )
        assert (
            get_query_for_question("what violates constitution")
            == "violates_constitution_article_II(Lib)"
        )

        # Test unknown pattern
        assert get_query_for_question("unknown question") is None

    def test_get_available_queries(self):
        """Test getting list of available queries."""
        queries = get_available_queries()

        assert isinstance(queries, list)
        assert len(queries) > 0
        assert "untested functions" in queries
        assert "incomplete features" in queries


class TestEnhancedSDDFramework:
    """Test the enhanced SDD framework."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.framework = EnhancedSDDFramework(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test framework initialization."""
        assert self.framework.workspace_root == self.temp_dir
        assert isinstance(self.framework.mangle_reasoner, MangleReasoner)
        assert isinstance(self.framework._last_analysis_results, dict)

    def test_ask_question(self):
        """Test natural language question processing."""
        with patch.object(
            self.framework.mangle_reasoner, "ask_question"
        ) as mock_ask:
            mock_ask.return_value = MangleQueryResult(
                query="untested_function(Func)",
                raw_results=[["func1"], ["func2"]],
                success=True,
                execution_time=0.1,
            )

            result = self.framework.ask_question("what functions are untested")

            assert result["question"] == "what functions are untested"
            assert result["success"] is True
            assert len(result["results"]) == 2

    def test_validate_constitutional_compliance(self):
        """Test constitutional compliance validation."""
        with patch.object(
            self.framework.mangle_reasoner,
            "validate_constitutional_compliance",
        ) as mock_validate:
            mock_validate.return_value = {
                "Article I": MangleQueryResult("test", [], True),
                "Article II": MangleQueryResult("test", [["violation"]], True),
            }

            result = self.framework.validate_constitutional_compliance()

            assert "mangle_analysis" in result
            assert "summary" in result
            assert "recommendations" in result

    def test_trace_code_to_spec(self):
        """Test code-to-specification traceability."""
        with patch.object(
            self.framework.mangle_reasoner, "trace_code_to_spec"
        ) as mock_trace:
            mock_trace.return_value = MangleQueryResult(
                query="spec_for_function",
                raw_results=[["feature1"], ["feature2"]],
                success=True,
                execution_time=0.2,
            )

            result = self.framework.trace_code_to_spec("test_function")

            assert result["code_element"] == "test_function"
            assert result["success"] is True
            assert result["traceability_found"] is True
            assert len(result["related_specs"]) == 2

    def test_analyze_code_quality(self):
        """Test code quality analysis."""
        with (
            patch.object(
                self.framework.mangle_reasoner, "find_quality_issues"
            ) as mock_quality,
            patch.object(
                self.framework.mangle_reasoner, "find_incomplete_work"
            ) as mock_incomplete,
        ):
            mock_quality.return_value = {
                "Untested Functions": MangleQueryResult(
                    "test", [["func1"]], True
                )
            }
            mock_incomplete.return_value = {
                "Incomplete Features": MangleQueryResult(
                    "test", [["feat1"]], True
                )
            }

            result = self.framework.analyze_code_quality()

            assert "quality_issues" in result
            assert "incomplete_work" in result
            assert "quality_metrics" in result
            assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_enhanced_specify(self):
        """Test enhanced specify phase."""
        request = SpecifyRequest(
            user_input="Create a new authentication system",
            context={"type": "Web application"},
        )

        # Mock the parent specify method
        with patch(
            "src.sdd.constitutional_pipeline.ConstitutionalSDDPipeline.specify"
        ) as mock_parent:
            from src.sdd.models import (
                ConstitutionalValidation,
                SpecifyResponse,
            )

            mock_parent.return_value = SpecifyResponse(
                feature_id="001-auth-system",
                feature_name="auth-system",
                spec_file_path="specs/001-auth-system/spec.md",
                constitutional_validation=ConstitutionalValidation(
                    overall_score=0.8, violations=[], recommendations=[]
                ),
                analysis_results={},
            )

            # Mock Mangle analysis methods
            with patch.object(
                self.framework, "_analyze_existing_solutions"
            ) as mock_existing:
                mock_existing.return_value = {"existing_functions": 5}

                with patch.object(
                    self.framework, "_find_similar_features"
                ) as mock_similar:
                    mock_similar.return_value = {"existing_specs": 2}

                    with patch.object(
                        self.framework, "_predict_compliance_issues"
                    ) as mock_compliance:
                        mock_compliance.return_value = {
                            "existing_violations": 0
                        }

                        result = await self.framework.specify(request)

                        assert result.feature_id == "001-auth-system"
                        assert (
                            result.analysis_results.get("mangle_enhanced")
                            is True
                        )

    def test_get_fact_statistics(self):
        """Test fact statistics retrieval."""
        with patch.object(
            self.framework.mangle_reasoner, "get_fact_statistics"
        ) as mock_stats:
            mock_stats.return_value = {
                "total_facts": 100,
                "fact_types": {"function": 50, "class": 25},
                "cache_valid": True,
            }

            result = self.framework.get_fact_statistics()

            assert result["total_facts"] == 100
            assert "fact_types" in result

    def test_invalidate_caches(self):
        """Test cache invalidation."""
        # Add some cached data
        self.framework._last_analysis_results["test"] = {"data": "cached"}

        with patch.object(
            self.framework.mangle_reasoner, "invalidate_cache"
        ) as mock_invalidate:
            self.framework.invalidate_caches()

            mock_invalidate.assert_called_once()
            assert len(self.framework._last_analysis_results) == 0


class TestIntegration:
    """Integration tests for the complete Mangle system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.setup_test_workspace()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def setup_test_workspace(self):
        """Create a test workspace with sample files."""
        # Create source files
        src_dir = self.temp_dir / "src"
        src_dir.mkdir()

        (src_dir / "main.py").write_text(
            '''
def main():
    """Main function."""
    print("Hello, World!")

class App:
    def run(self):
        pass
'''
        )

        # Create test files
        tests_dir = self.temp_dir / "tests"
        tests_dir.mkdir()

        (tests_dir / "test_main.py").write_text(
            """
def test_main():
    from main import main
    main()
"""
        )

        # Create specs
        specs_dir = self.temp_dir / "specs"
        feature_dir = specs_dir / "001-hello-world"
        feature_dir.mkdir(parents=True)

        (feature_dir / "feature-spec.md").write_text(
            """
# Feature Specification: Hello World

## Acceptance Criteria

- [ ] Should print hello world
- [ ] Should be testable

## Requirements

- Must be simple
"""
        )

    def test_end_to_end_fact_generation(self):
        """Test end-to-end fact generation."""
        generator = MangleFactGenerator(str(self.temp_dir))
        facts = generator.generate_all_facts()

        # Verify facts contain expected content
        assert 'function("main"' in facts
        assert 'class("App"' in facts
        assert 'spec("001-hello-world"' in facts
        assert 'test_function("test_main"' in facts
        assert "acceptance_criterion(" in facts

    def test_end_to_end_with_mock_mangle(self):
        """Test end-to-end reasoning with mocked Mangle execution."""
        framework = EnhancedSDDFramework(self.temp_dir)

        # Mock Mangle execution to return test data
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = '("main")\n("App")\n'
            mock_run.return_value.stderr = ""

            # Test question asking
            result = framework.ask_question("what functions are untested")

            # Should not fail even if Mangle is mocked
            assert isinstance(result, dict)
            assert "question" in result

    def test_fact_generator_statistics(self):
        """Test fact generator statistics on real workspace."""
        generator = MangleFactGenerator(str(self.temp_dir))
        generator.generate_all_facts()

        stats = generator.get_fact_statistics()

        # Should have generated facts of various types
        assert stats["function"] > 0
        assert stats["class"] > 0
        assert stats["spec"] > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_mangle_reasoner_with_invalid_executable(self):
        """Test reasoner with invalid Mangle executable."""
        reasoner = MangleReasoner(".", mangle_executable="nonexistent_mangle")

        result = reasoner.query("test_query(X)")

        assert result.success is False
        assert "not found" in result.error_message

    def test_fact_generator_with_invalid_python_file(self):
        """Test fact generator with syntactically invalid Python file."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            src_dir = temp_dir / "src"
            src_dir.mkdir()

            # Create invalid Python file
            (src_dir / "invalid.py").write_text(
                "def invalid_syntax(:\n    pass"
            )

            generator = MangleFactGenerator(str(temp_dir))
            facts = generator.generate_code_facts()

            # Should not crash, should handle syntax error gracefully
            assert isinstance(facts, str)

        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_enhanced_framework_with_missing_workspace(self):
        """Test enhanced framework with missing workspace directory."""
        # Create a temporary directory for testing instead
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_workspace = Path(temp_dir) / "nonexistent" / "workspace"

            # Should not crash during initialization - will create dirs
            framework = EnhancedSDDFramework(temp_workspace)
            assert framework.workspace_root == temp_workspace


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
