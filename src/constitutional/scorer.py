"""Constitutional Scoring Framework.

Implements the core constitutional compliance scoring system that evaluates
artifacts against all six constitutional articles with weighted scoring.

This module provides:
- ConstitutionalScorer: Main scoring engine
- ConstitutionalResult: Structured scoring results
- Violation detection and reporting
- Constitutional compliance thresholds
"""

import ast
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConstitutionalViolation:
    """Represents a violation of constitutional principles."""

    article: str  # "Article I", "Article II", etc.
    principle: str  # Brief description of violated principle
    message: str  # Detailed violation description
    line: int | None = None  # Line number if applicable
    severity: str = "medium"  # "low", "medium", "high", "critical"
    suggestion: str | None = None  # Suggested fix


@dataclass
class ConstitutionalResult:
    """Structured result from constitutional scoring."""

    overall_score: float  # 0.0 to 1.0
    article_scores: dict[str, float]  # Score per article
    violations: list[ConstitutionalViolation]
    compliance_threshold: float = 0.75
    is_compliant: bool = False
    recommendations: list[str] = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        """Calculate compliance and set defaults."""
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}

        self.is_compliant = self.overall_score >= self.compliance_threshold


class ConstitutionalScorer:
    """Main constitutional compliance scoring engine.

    Evaluates artifacts against all six constitutional articles:
    - Article I: Library-First Development
    - Article II: Test-First Development
    - Article III: Simplicity Gate
    - Article IV: Integration-First Testing
    - Article V: Clarity and Unambiguity
    - Article VI: Counterfactual Justification

    Example:
        scorer = ConstitutionalScorer()
        result = scorer.score_specification("Build a web API using FastAPI...")
        if result.is_compliant:
            print(f"Constitutional score: {result.overall_score:.2f}")
        else:
            print(f"Violations: {len(result.violations)}")
    """

    def __init__(self, compliance_threshold: float = 0.75):
        """Initialize the constitutional scorer.

        Args:
            compliance_threshold: Minimum score for constitutional compliance
        """
        self.compliance_threshold = compliance_threshold
        self.article_weights = {
            "Article I": 1.0,  # Library-First Development
            "Article II": 1.0,  # Test-First Development
            "Article III": 1.0,  # Simplicity Gate
            "Article IV": 1.0,  # Integration-First Testing
            "Article V": 2.0,  # Clarity and Unambiguity (2x weight)
            "Article VI": 1.0,  # Counterfactual Justification
        }

    def score_specification(self, specification: str) -> ConstitutionalResult:
        """Score a specification document for constitutional compliance.

        Args:
            specification: The specification text to evaluate

        Returns:
            ConstitutionalResult with scores and violations
        """
        violations = []
        article_scores = {}

        # Article I: Library-First Development
        article_scores["Article I"] = self._score_library_first_spec(
            specification, violations
        )

        # Article II: Test-First Development
        article_scores["Article II"] = self._score_test_first_spec(
            specification, violations
        )

        # Article III: Simplicity Gate
        article_scores["Article III"] = self._score_simplicity_spec(
            specification, violations
        )

        # Article IV: Integration-First Testing
        article_scores["Article IV"] = self._score_integration_spec(
            specification, violations
        )

        # Article V: Clarity and Unambiguity
        article_scores["Article V"] = self._score_clarity_spec(
            specification, violations
        )

        # Article VI: Counterfactual Justification
        article_scores["Article VI"] = self._score_counterfactual_spec(
            specification, violations
        )

        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(article_scores)

        recommendations = self._generate_recommendations(violations)

        return ConstitutionalResult(
            overall_score=overall_score,
            article_scores=article_scores,
            violations=violations,
            compliance_threshold=self.compliance_threshold,
            recommendations=recommendations,
            metadata={
                "scorer_version": "1.0.0",
                "evaluation_type": "specification",
                "total_violations": len(violations),
            },
        )

    def score_code(
        self, code: str, file_path: str | None = None
    ) -> ConstitutionalResult:
        """Score code for constitutional compliance.

        Args:
            code: The source code to evaluate
            file_path: Optional path to the file being scored

        Returns:
            ConstitutionalResult with scores and violations
        """
        violations = []
        article_scores = {}

        # Parse the code for analysis
        try:
            parsed_ast = ast.parse(code)
        except SyntaxError as e:
            violations.append(
                ConstitutionalViolation(
                    article="Article V",
                    principle="Clarity and Unambiguity",
                    message=f"Syntax error: {e}",
                    line=e.lineno,
                    severity="critical",
                    suggestion="Fix syntax errors before constitutional review",
                )
            )
            # Return early for syntax errors
            return ConstitutionalResult(
                overall_score=0.0,
                article_scores=dict.fromkeys(self.article_weights.keys(), 0.0),
                violations=violations,
                compliance_threshold=self.compliance_threshold,
            )

        # Article I: Library-First Development
        article_scores["Article I"] = self._score_library_first_code(
            code, parsed_ast, violations
        )

        # Article II: Test-First Development
        article_scores["Article II"] = self._score_test_first_code(
            code, file_path, violations
        )

        # Article III: Simplicity Gate
        article_scores["Article III"] = self._score_simplicity_code(
            code, parsed_ast, violations
        )

        # Article IV: Integration-First Testing
        article_scores["Article IV"] = self._score_integration_code(
            code, file_path, violations
        )

        # Article V: Clarity and Unambiguity
        article_scores["Article V"] = self._score_clarity_code(
            code, parsed_ast, violations
        )

        # Article VI: Counterfactual Justification
        article_scores["Article VI"] = self._score_counterfactual_code(
            code, violations
        )

        overall_score = self._calculate_weighted_score(article_scores)
        recommendations = self._generate_recommendations(violations)

        return ConstitutionalResult(
            overall_score=overall_score,
            article_scores=article_scores,
            violations=violations,
            compliance_threshold=self.compliance_threshold,
            recommendations=recommendations,
            metadata={
                "scorer_version": "1.0.0",
                "evaluation_type": "code",
                "file_path": file_path,
                "total_violations": len(violations),
                "line_count": len(code.split("\n")),
            },
        )

    def _score_library_first_spec(
        self, specification: str, violations: list[ConstitutionalViolation]
    ) -> float:
        """Score Article I: Library-First Development for specifications."""
        score = 1.0
        spec_lower = specification.lower()

        # Check for library research mentions
        library_indicators = [
            "existing",
            "library",
            "framework",
            "package",
            "import",
            "use",
            "leverage",
            "adopt",
            "integrate",
            "available",
        ]

        library_mentions = sum(
            1 for word in library_indicators if word in spec_lower
        )
        if library_mentions == 0:
            score -= 0.4
            violations.append(
                ConstitutionalViolation(
                    article="Article I",
                    principle="Library-First Development",
                    message="No mention of existing libraries or frameworks",
                    severity="high",
                    suggestion="Research existing solutions before custom implementation",
                )
            )

        # Check for custom implementation warnings
        custom_indicators = [
            "build from scratch",
            "custom",
            "new implementation",
        ]
        if any(phrase in spec_lower for phrase in custom_indicators):
            if (
                "justification" not in spec_lower
                and "because" not in spec_lower
            ):
                score -= 0.3
                violations.append(
                    ConstitutionalViolation(
                        article="Article I",
                        principle="Library-First Development",
                        message="Custom implementation without justification",
                        severity="medium",
                        suggestion="Justify why existing solutions are insufficient",
                    )
                )

        return max(0.0, score)

    def _score_test_first_spec(
        self, specification: str, violations: list[ConstitutionalViolation]
    ) -> float:
        """Score Article II: Test-First Development for specifications."""
        score = 1.0
        spec_lower = specification.lower()

        # Check for test-related content
        test_indicators = [
            "test",
            "testing",
            "coverage",
            "pytest",
            "unittest",
            "tdd",
            "test-driven",
            "testable",
            "validation",
        ]

        test_mentions = sum(
            1 for word in test_indicators if word in spec_lower
        )
        if test_mentions == 0:
            score -= 0.5
            violations.append(
                ConstitutionalViolation(
                    article="Article II",
                    principle="Test-First Development",
                    message="No testing requirements specified",
                    severity="high",
                    suggestion="Include comprehensive testing requirements (80% coverage minimum)",
                )
            )

        # Check for coverage requirements
        if "coverage" not in spec_lower and "80%" not in specification:
            score -= 0.2
            violations.append(
                ConstitutionalViolation(
                    article="Article II",
                    principle="Test-First Development",
                    message="No coverage requirements specified",
                    severity="medium",
                    suggestion="Specify minimum 80% test coverage requirement",
                )
            )

        return max(0.0, score)

    def _score_simplicity_spec(
        self, specification: str, violations: list[ConstitutionalViolation]
    ) -> float:
        """Score Article III: Simplicity Gate for specifications."""
        score = 1.0
        spec_lower = specification.lower()

        # Check for complexity indicators
        complexity_indicators = [
            "complex",
            "complicated",
            "advanced",
            "sophisticated",
            "enterprise",
            "abstract",
            "factory",
            "pattern",
        ]

        complexity_mentions = sum(
            1 for word in complexity_indicators if word in spec_lower
        )
        if complexity_mentions > 3:
            score -= 0.3
            violations.append(
                ConstitutionalViolation(
                    article="Article III",
                    principle="Simplicity Gate",
                    message="Specification suggests complex implementation",
                    severity="medium",
                    suggestion="Break down complex features into simple components",
                )
            )

        # Check for simplicity mentions
        simplicity_indicators = [
            "simple",
            "clear",
            "straightforward",
            "minimal",
        ]
        if not any(word in spec_lower for word in simplicity_indicators):
            score -= 0.2
            violations.append(
                ConstitutionalViolation(
                    article="Article III",
                    principle="Simplicity Gate",
                    message="No emphasis on simplicity in specification",
                    severity="low",
                    suggestion="Emphasize simple, clear implementation approach",
                )
            )

        return max(0.0, score)

    def _score_integration_spec(
        self, specification: str, violations: list[ConstitutionalViolation]
    ) -> float:
        """Score Article IV: Integration-First Testing for specifications."""
        score = 1.0
        spec_lower = specification.lower()

        # Check for integration mentions
        integration_indicators = [
            "integration",
            "end-to-end",
            "e2e",
            "system",
            "workflow",
            "realistic",
            "environment",
            "deployment",
        ]

        integration_mentions = sum(
            1 for word in integration_indicators if word in spec_lower
        )
        if integration_mentions == 0:
            score -= 0.4
            violations.append(
                ConstitutionalViolation(
                    article="Article IV",
                    principle="Integration-First Testing",
                    message="No integration testing requirements specified",
                    severity="high",
                    suggestion="Include integration and end-to-end testing requirements",
                )
            )

        return max(0.0, score)

    def _score_clarity_spec(
        self, specification: str, violations: list[ConstitutionalViolation]
    ) -> float:
        """Score Article V: Clarity and Unambiguity for specifications."""
        score = 1.0

        # Check for ambiguous language
        ambiguous_phrases = [
            "maybe",
            "possibly",
            "might",
            "could be",
            "perhaps",
            "unclear",
            "tbd",
            "to be determined",
            "flexible",
        ]

        ambiguous_count = sum(
            1
            for phrase in ambiguous_phrases
            if phrase in specification.lower()
        )
        if ambiguous_count > 0:
            score -= min(0.5, ambiguous_count * 0.1)
            violations.append(
                ConstitutionalViolation(
                    article="Article V",
                    principle="Clarity and Unambiguity",
                    message=f"Contains {ambiguous_count} ambiguous phrases",
                    severity="medium",
                    suggestion="Replace ambiguous language with specific requirements",
                )
            )

        # Check for clear structure indicators
        structure_indicators = [
            "requirements:",
            "acceptance criteria",
            "definition of done",
            "must",
            "shall",
            "will",
            "should not",
        ]

        structure_count = sum(
            1
            for phrase in structure_indicators
            if phrase in specification.lower()
        )
        if structure_count == 0:
            score -= 0.3
            violations.append(
                ConstitutionalViolation(
                    article="Article V",
                    principle="Clarity and Unambiguity",
                    message="Lacks clear structure and acceptance criteria",
                    severity="medium",
                    suggestion="Add clear requirements and acceptance criteria sections",
                )
            )

        return max(0.0, score)

    def _score_counterfactual_spec(
        self, specification: str, violations: list[ConstitutionalViolation]
    ) -> float:
        """Score Article VI: Counterfactual Justification for specifications."""
        score = 1.0
        spec_lower = specification.lower()

        # Check for alternative analysis
        justification_indicators = [
            "because",
            "since",
            "due to",
            "reason",
            "alternative",
            "considered",
            "compared",
            "versus",
            "instead of",
            "rather than",
        ]

        justification_mentions = sum(
            1 for word in justification_indicators if word in spec_lower
        )
        if justification_mentions == 0:
            score -= 0.4
            violations.append(
                ConstitutionalViolation(
                    article="Article VI",
                    principle="Counterfactual Justification",
                    message="No justification for chosen approach",
                    severity="medium",
                    suggestion="Document alternative approaches and justify choices",
                )
            )

        return max(0.0, score)

    def _score_library_first_code(
        self,
        code: str,
        parsed_ast: ast.AST,
        violations: list[ConstitutionalViolation],
    ) -> float:
        """Score Article I: Library-First Development for code."""
        score = 1.0

        # Count imports
        imports = [
            node
            for node in ast.walk(parsed_ast)
            if isinstance(node, ast.Import | ast.ImportFrom)
        ]

        if not imports:
            score -= 0.5
            violations.append(
                ConstitutionalViolation(
                    article="Article I",
                    principle="Library-First Development",
                    message="No library imports found",
                    severity="high",
                    suggestion="Use existing libraries instead of implementing from scratch",
                )
            )

        # Check for custom implementations of common functionality
        function_defs = [
            node
            for node in ast.walk(parsed_ast)
            if isinstance(node, ast.FunctionDef)
        ]

        for func in function_defs:
            func_name = func.name.lower()
            if any(
                pattern in func_name
                for pattern in [
                    "json_parse",
                    "http_request",
                    "hash_",
                    "encrypt",
                ]
            ):
                score -= 0.2
                violations.append(
                    ConstitutionalViolation(
                        article="Article I",
                        principle="Library-First Development",
                        message=f"Custom implementation detected: {func.name}",
                        line=func.lineno,
                        severity="medium",
                        suggestion=f"Consider using existing library for {func.name} functionality",
                    )
                )

        return max(0.0, score)

    def _score_test_first_code(
        self,
        code: str,
        file_path: str | None,
        violations: list[ConstitutionalViolation],
    ) -> float:
        """Score Article II: Test-First Development for code."""
        score = 1.0

        # If this is a test file, score higher
        if file_path and ("test_" in file_path or "_test.py" in file_path):
            return 1.0

        # Check for test imports or test functions
        test_indicators = ["pytest", "unittest", "test_", "assert"]
        if not any(indicator in code.lower() for indicator in test_indicators):
            score -= 0.6
            violations.append(
                ConstitutionalViolation(
                    article="Article II",
                    principle="Test-First Development",
                    message="No tests found for implementation code",
                    severity="critical",
                    suggestion="Create comprehensive tests before or alongside implementation",
                )
            )

        return max(0.0, score)

    def _score_simplicity_code(
        self,
        code: str,
        parsed_ast: ast.AST,
        violations: list[ConstitutionalViolation],
    ) -> float:
        """Score Article III: Simplicity Gate for code."""
        score = 1.0

        # Check function length (max 50 lines)
        function_defs = [
            node
            for node in ast.walk(parsed_ast)
            if isinstance(node, ast.FunctionDef)
        ]

        for func in function_defs:
            func_lines = (
                func.end_lineno - func.lineno + 1 if func.end_lineno else 1
            )
            if func_lines > 50:
                score -= 0.3
                violations.append(
                    ConstitutionalViolation(
                        article="Article III",
                        principle="Simplicity Gate",
                        message=f"Function {func.name} is {func_lines} lines (max: 50)",
                        line=func.lineno,
                        severity="high",
                        suggestion=f"Break down {func.name} into smaller functions",
                    )
                )

        # Check for complex nested structures
        max_nesting = self._calculate_max_nesting(parsed_ast)
        if max_nesting > 4:
            score -= 0.2
            violations.append(
                ConstitutionalViolation(
                    article="Article III",
                    principle="Simplicity Gate",
                    message=f"Maximum nesting depth: {max_nesting} (recommended: ≤4)",
                    severity="medium",
                    suggestion="Reduce nesting by extracting functions or using early returns",
                )
            )

        return max(0.0, score)

    def _score_integration_code(
        self,
        code: str,
        file_path: str | None,
        violations: list[ConstitutionalViolation],
    ) -> float:
        """Score Article IV: Integration-First Testing for code."""
        score = 1.0

        # Check for integration test patterns
        integration_patterns = [
            "test_integration",
            "test_e2e",
            "test_workflow",
            "test_system",
        ]

        if not any(
            pattern in code.lower() for pattern in integration_patterns
        ):
            if file_path and not (
                "test_" in file_path or "_test.py" in file_path
            ):
                score -= 0.4
                violations.append(
                    ConstitutionalViolation(
                        article="Article IV",
                        principle="Integration-First Testing",
                        message="No integration tests found",
                        severity="medium",
                        suggestion="Add integration tests for realistic workflow validation",
                    )
                )

        return max(0.0, score)

    def _score_clarity_code(
        self,
        code: str,
        parsed_ast: ast.AST,
        violations: list[ConstitutionalViolation],
    ) -> float:
        """Score Article V: Clarity and Unambiguity for code."""
        score = 1.0

        # Check for docstrings
        function_defs = [
            node
            for node in ast.walk(parsed_ast)
            if isinstance(node, ast.FunctionDef)
        ]

        functions_without_docstrings = 0
        for func in function_defs:
            if not ast.get_docstring(func):
                functions_without_docstrings += 1

        if functions_without_docstrings > 0:
            score -= min(0.5, functions_without_docstrings * 0.1)
            violations.append(
                ConstitutionalViolation(
                    article="Article V",
                    principle="Clarity and Unambiguity",
                    message=f"{functions_without_docstrings} functions missing docstrings",
                    severity="medium",
                    suggestion="Add clear docstrings to all functions",
                )
            )

        # Check for TODO/FIXME comments
        todo_count = code.lower().count("todo") + code.lower().count("fixme")
        if todo_count > 0:
            score -= min(0.3, todo_count * 0.05)
            violations.append(
                ConstitutionalViolation(
                    article="Article V",
                    principle="Clarity and Unambiguity",
                    message=f"{todo_count} TODO/FIXME comments found",
                    severity="low",
                    suggestion="Replace placeholder comments with proper implementations",
                )
            )

        return max(0.0, score)

    def _score_counterfactual_code(
        self, code: str, violations: list[ConstitutionalViolation]
    ) -> float:
        """Score Article VI: Counterfactual Justification for code."""
        score = 1.0

        # Check for decision documentation in comments
        justification_patterns = [
            "because",
            "since",
            "instead of",
            "rather than",
            "chosen because",
        ]

        if not any(
            pattern in code.lower() for pattern in justification_patterns
        ):
            score -= 0.3
            violations.append(
                ConstitutionalViolation(
                    article="Article VI",
                    principle="Counterfactual Justification",
                    message="No justification comments for implementation decisions",
                    severity="low",
                    suggestion="Add comments explaining why specific approaches were chosen",
                )
            )

        return max(0.0, score)

    def _calculate_max_nesting(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth in AST."""
        max_depth = depth

        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, ast.If | ast.For | ast.While | ast.With | ast.Try
            ):
                child_depth = self._calculate_max_nesting(child, depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._calculate_max_nesting(child, depth)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def _calculate_weighted_score(
        self, article_scores: dict[str, float]
    ) -> float:
        """Calculate weighted overall score from article scores."""
        total_weight = sum(self.article_weights.values())
        weighted_sum = sum(
            score * self.article_weights[article]
            for article, score in article_scores.items()
        )
        return weighted_sum / total_weight

    def _generate_recommendations(
        self, violations: list[ConstitutionalViolation]
    ) -> list[str]:
        """Generate actionable recommendations from violations."""
        recommendations = []

        # Group by article for better organization
        by_article = {}
        for violation in violations:
            if violation.article not in by_article:
                by_article[violation.article] = []
            by_article[violation.article].append(violation)

        for article, article_violations in by_article.items():
            if len(article_violations) > 1:
                recommendations.append(
                    f"{article}: Address {len(article_violations)} violations to improve compliance"
                )
            else:
                violation = article_violations[0]
                if violation.suggestion:
                    recommendations.append(
                        f"{article}: {violation.suggestion}"
                    )

        # Add general recommendations
        if len(violations) > 5:
            recommendations.append(
                "Consider comprehensive refactoring to address multiple constitutional violations"
            )

        if not recommendations:
            recommendations.append(
                "Constitutional compliance is excellent - maintain current standards"
            )

        return recommendations
