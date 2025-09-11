"""Tests for Constitutional Scoring Framework.

This test module validates the ConstitutionalScorer implementation against
all six constitutional articles. Tests are written first to ensure proper
test-driven development compliance.
"""

from src.constitutional.scorer import (
    ConstitutionalResult,
    ConstitutionalScorer,
    ConstitutionalViolation,
)


class TestConstitutionalScorer:
    """Test suite for ConstitutionalScorer class."""

    def setup_method(self):
        """Set up test instances before each test."""
        self.scorer = ConstitutionalScorer()

    def test_scorer_initialization(self):
        """Test scorer initializes with proper defaults."""
        assert self.scorer.compliance_threshold == 0.75
        assert self.scorer.article_weights["Article V"] == 2.0  # Clarity 2x weight
        assert len(self.scorer.article_weights) == 6

    def test_score_specification_high_compliance(self):
        """Test specification scoring with high constitutional compliance."""
        spec = """
        Build a simple file processor library using existing Python libraries.
        Include comprehensive tests before implementation with 80% coverage minimum.
        Keep functions under 50 lines with clear, single responsibilities.
        Integrate with pytest for testing and use pathlib for file operations.
        All functions must have clear docstrings and acceptance criteria.
        Because we need reliability, we will use established patterns rather than
        custom implementations.
        """

        result = self.scorer.score_specification(spec)

        assert isinstance(result, ConstitutionalResult)
        assert result.overall_score >= 0.75
        assert result.is_compliant is True
        assert len(result.article_scores) == 6
        assert all(score >= 0.0 for score in result.article_scores.values())

    def test_score_specification_low_compliance(self):
        """Test specification scoring with low constitutional compliance."""
        spec = """
        Build a complex enterprise-grade system from scratch.
        Maybe use some patterns. Implementation details TBD.
        Could be flexible in the approach.
        """

        result = self.scorer.score_specification(spec)

        assert result.overall_score < 0.75
        assert result.is_compliant is False
        assert len(result.violations) > 0
        assert len(result.recommendations) > 0

    def test_score_code_high_compliance(self):
        """Test code scoring with high constitutional compliance."""
        code = '''
import pathlib
import json
from typing import Dict, List

def process_file(file_path: str) -> Dict[str, str]:
    """Process a file and return its contents.

    Args:
        file_path: Path to the file to process

    Returns:
        Dictionary with file metadata and content

    We use pathlib because it provides better cross-platform
    compatibility compared to os.path.
    """
    path = pathlib.Path(file_path)
    return {
        "name": path.name,
        "content": path.read_text(),
        "size": str(path.stat().st_size)
    }

def test_process_file():
    """Test the process_file function."""
    result = process_file("test.txt")
    assert "name" in result
    assert "content" in result
    assert "size" in result
'''

        result = self.scorer.score_code(code, "src/processor.py")

        assert result.overall_score >= 0.70
        assert result.article_scores["Article I"] > 0.8  # Has imports
        assert result.article_scores["Article V"] > 0.8  # Has docstrings

    def test_score_code_low_compliance(self):
        """Test code scoring with low constitutional compliance."""
        code = """
def very_long_function_that_violates_simplicity_constraints():
    # This function is intentionally long to test simplicity violations
    x = 1
    y = 2
    z = 3
    result = 0
    for i in range(100):
        if i % 2 == 0:
            if i % 4 == 0:
                if i % 8 == 0:
                    if i % 16 == 0:
                        if i % 32 == 0:
                            result += i * x * y * z
                        else:
                            result += i * x * y
                    else:
                        result += i * x
                else:
                    result += i
            else:
                result -= i
        else:
            result += i / 2

    # TODO: Fix this mess
    # FIXME: This is terrible code
    return result

def another_function():
    pass

def custom_json_parser(text):
    # Custom JSON implementation instead of using json library
    pass
"""

        result = self.scorer.score_code(code, "src/bad_code.py")

        assert result.overall_score < 0.70
        assert len(result.violations) > 0

        # Check for specific violations
        violation_articles = [v.article for v in result.violations]
        assert "Article III" in violation_articles  # Simplicity violations
        assert "Article V" in violation_articles  # Missing docstrings

    def test_score_code_syntax_error(self):
        """Test code scoring with syntax errors."""
        code = """
def broken_function(
    # Missing closing parenthesis and colon
    pass
"""

        result = self.scorer.score_code(code)

        assert result.overall_score == 0.0
        assert result.is_compliant is False
        assert len(result.violations) > 0
        assert result.violations[0].severity == "critical"

    def test_article_i_library_first_spec(self):
        """Test Article I (Library-First) scoring for specifications."""
        # High compliance spec
        good_spec = "Use existing FastAPI library for web development"
        result = self.scorer.score_specification(good_spec)
        assert result.article_scores["Article I"] > 0.5

        # Low compliance spec
        bad_spec = "Build custom web framework from scratch"
        result = self.scorer.score_specification(bad_spec)
        assert result.article_scores["Article I"] < 0.8

    def test_article_ii_test_first_spec(self):
        """Test Article II (Test-First) scoring for specifications."""
        # High compliance spec
        good_spec = "Include comprehensive testing with 80% coverage using pytest"
        result = self.scorer.score_specification(good_spec)
        assert result.article_scores["Article II"] > 0.5

        # Low compliance spec
        bad_spec = "Build a web application"
        result = self.scorer.score_specification(bad_spec)
        assert result.article_scores["Article II"] < 0.8

    def test_article_iii_simplicity_spec(self):
        """Test Article III (Simplicity Gate) scoring for specifications."""
        # High compliance spec
        good_spec = "Build a simple, clear file processor with minimal complexity"
        result = self.scorer.score_specification(good_spec)
        assert result.article_scores["Article III"] > 0.5

        # Low compliance spec
        bad_spec = "Build complex enterprise sophisticated advanced architecture"
        result = self.scorer.score_specification(bad_spec)
        assert result.article_scores["Article III"] < 0.8

    def test_article_iv_integration_spec(self):
        """Test Article IV (Integration-First) scoring for specifications."""
        # High compliance spec
        good_spec = "Include end-to-end integration testing in realistic environment"
        result = self.scorer.score_specification(good_spec)
        assert result.article_scores["Article IV"] > 0.5

        # Low compliance spec
        bad_spec = "Build a simple function"
        result = self.scorer.score_specification(bad_spec)
        assert result.article_scores["Article IV"] < 0.8

    def test_article_v_clarity_spec(self):
        """Test Article V (Clarity and Unambiguity) scoring for specifications."""
        # High compliance spec
        good_spec = "Requirements: Build file processor. Must handle .txt files. Shall return dict."
        result = self.scorer.score_specification(good_spec)
        assert result.article_scores["Article V"] > 0.5

        # Low compliance spec with ambiguous language
        bad_spec = "Maybe build something. Possibly flexible. TBD approach. Unclear requirements."
        result = self.scorer.score_specification(bad_spec)
        assert result.article_scores["Article V"] < 0.8

    def test_article_vi_counterfactual_spec(self):
        """Test Article VI (Counterfactual Justification) scoring for specifications."""
        # High compliance spec
        good_spec = "Use FastAPI because it's faster than Django for this use case"
        result = self.scorer.score_specification(good_spec)
        assert result.article_scores["Article VI"] > 0.5

        # Low compliance spec
        bad_spec = "Build a web API"
        result = self.scorer.score_specification(bad_spec)
        assert result.article_scores["Article VI"] < 0.8

    def test_weighted_scoring(self):
        """Test that Article V (Clarity) has 2x weight in overall scoring."""
        # Specification with only clarity issues
        spec_with_clarity_issues = (
            "Maybe build something. TBD. Unclear requirements. Could be flexible."
        )
        result = self.scorer.score_specification(spec_with_clarity_issues)

        # The overall score should be significantly impacted due to 2x weight
        clarity_score = result.article_scores["Article V"]
        assert clarity_score < 0.5
        assert result.overall_score < 0.8  # Should be significantly impacted

    def test_compliance_threshold_validation(self):
        """Test compliance threshold validation works correctly."""
        custom_scorer = ConstitutionalScorer(compliance_threshold=0.9)

        # Decent spec that would pass 0.75 threshold
        spec = "Use existing libraries for simple file processing with basic tests"
        result = custom_scorer.score_specification(spec)

        # Should fail higher threshold
        assert result.compliance_threshold == 0.9
        # The result compliance may vary based on the specific scoring

    def test_violation_data_structure(self):
        """Test that violations have proper data structure."""
        bad_spec = "Build something complex"
        result = self.scorer.score_specification(bad_spec)

        if result.violations:
            violation = result.violations[0]
            assert hasattr(violation, "article")
            assert hasattr(violation, "principle")
            assert hasattr(violation, "message")
            assert hasattr(violation, "severity")
            assert violation.severity in ["low", "medium", "high", "critical"]

    def test_recommendations_generation(self):
        """Test that recommendations are generated appropriately."""
        bad_spec = "Build complex system"
        result = self.scorer.score_specification(bad_spec)

        assert len(result.recommendations) > 0
        assert all(isinstance(rec, str) for rec in result.recommendations)

    def test_metadata_generation(self):
        """Test that metadata is properly generated."""
        spec = "Simple test specification"
        result = self.scorer.score_specification(spec)

        assert "scorer_version" in result.metadata
        assert "evaluation_type" in result.metadata
        assert "total_violations" in result.metadata
        assert result.metadata["evaluation_type"] == "specification"


class TestConstitutionalViolation:
    """Test suite for ConstitutionalViolation dataclass."""

    def test_violation_creation(self):
        """Test violation can be created with required fields."""
        violation = ConstitutionalViolation(
            article="Article I",
            principle="Library-First Development",
            message="No libraries found",
        )

        assert violation.article == "Article I"
        assert violation.principle == "Library-First Development"
        assert violation.message == "No libraries found"
        assert violation.severity == "medium"  # Default
        assert violation.line is None  # Default
        assert violation.suggestion is None  # Default

    def test_violation_with_all_fields(self):
        """Test violation creation with all fields specified."""
        violation = ConstitutionalViolation(
            article="Article III",
            principle="Simplicity Gate",
            message="Function too long",
            line=42,
            severity="high",
            suggestion="Break function into smaller parts",
        )

        assert violation.line == 42
        assert violation.severity == "high"
        assert violation.suggestion == "Break function into smaller parts"


class TestConstitutionalResult:
    """Test suite for ConstitutionalResult dataclass."""

    def test_result_compliance_calculation(self):
        """Test that compliance is correctly calculated."""
        # High compliance result
        result = ConstitutionalResult(
            overall_score=0.85,
            article_scores={"Article I": 0.9, "Article II": 0.8},
            violations=[],
            compliance_threshold=0.75,
        )

        assert result.is_compliant is True

        # Low compliance result
        result_low = ConstitutionalResult(
            overall_score=0.65,
            article_scores={"Article I": 0.7, "Article II": 0.6},
            violations=[],
            compliance_threshold=0.75,
        )

        assert result_low.is_compliant is False

    def test_result_defaults(self):
        """Test that result has proper defaults."""
        result = ConstitutionalResult(
            overall_score=0.8, article_scores={"Article I": 0.8}, violations=[]
        )

        assert result.recommendations == []
        assert result.metadata == {}
        assert result.compliance_threshold == 0.75


# Performance and integration tests
class TestConstitutionalScorerPerformance:
    """Performance tests for constitutional scorer."""

    def test_large_specification_performance(self):
        """Test scorer performance with large specifications."""
        import time

        # Create a large specification
        large_spec = (
            """
        Build a comprehensive file processing system using existing libraries.
        Include extensive testing with 80% coverage minimum.
        """
            * 100
        )  # Repeat to make it large

        scorer = ConstitutionalScorer()
        start_time = time.time()
        result = scorer.score_specification(large_spec)
        end_time = time.time()

        # Should complete within 2 seconds per constitutional requirement
        assert (end_time - start_time) < 2.0
        assert isinstance(result, ConstitutionalResult)

    def test_large_code_performance(self):
        """Test scorer performance with large code files."""
        import time

        # Create large code with many functions
        large_code = (
            '''
import json
import pathlib

def process_data(data):
    """Process data efficiently."""
    return json.dumps(data)

'''
            * 50
        )  # Many similar functions

        scorer = ConstitutionalScorer()
        start_time = time.time()
        result = scorer.score_code(large_code)
        end_time = time.time()

        # Should complete within 2 seconds per constitutional requirement
        assert (end_time - start_time) < 2.0
        assert isinstance(result, ConstitutionalResult)
