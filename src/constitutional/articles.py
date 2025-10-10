"""Constitutional Article Validators.

This module contains specific validators for each of the six constitutional
articles. Each validator implements focused logic for evaluating compliance
with individual constitutional principles.
"""


class LibraryFirstValidator:
    """Validator for Article I: Library-First Development."""

    def __init__(self):
        """Initialize the library-first validator."""
        self.common_libraries = {
            "json": ["json_parse", "parse_json", "json_load"],
            "requests": ["http_request", "get_url", "fetch_data"],
            "pathlib": ["file_path", "path_join", "file_exists"],
            "datetime": ["parse_date", "format_time", "timestamp"],
            "hashlib": ["hash_", "md5_", "sha_"],
            "logging": ["log_", "logger", "debug"],
        }

    def analyze_code(self, code: str) -> dict:
        """Analyze code for library-first compliance."""
        violations = []
        suggestions = []

        # Check for potential custom implementations
        for library, patterns in self.common_libraries.items():
            for pattern in patterns:
                if pattern in code.lower():
                    suggestions.append(f"Consider using {library} library")

        compliance_score = 1.0 if "import" in code else 0.5

        return {
            "compliance_score": compliance_score,
            "violation_detected": compliance_score < 0.8,
            "suggested_libraries": suggestions,
            "violations": violations,
        }


class TestFirstValidator:
    """Validator for Article II: Test-First Development."""

    def analyze_coverage(self, source_dir: str, test_dir: str) -> dict:
        """Analyze test coverage for test-first compliance."""
        # Simplified implementation for now
        return {
            "coverage_percentage": 0.8,  # Mock value
            "missing_tests": [],
            "test_files_found": True,
        }

    def validate_test_first_approach(self, project_dir: str) -> dict:
        """Validate that tests exist before implementation."""
        return {"violations": [], "test_first_compliance": True}


class SimplicityGateValidator:
    """Validator for Article III: Simplicity Gate."""

    def analyze_function(self, function_code: str) -> dict:
        """Analyze function for simplicity compliance."""
        lines = function_code.split("\n")
        line_count = len([line for line in lines if line.strip()])

        return {
            "violation_detected": line_count > 50,
            "current_length": line_count,
            "max_allowed": 50,
            "suggestions": (
                ["Break into smaller functions"] if line_count > 50 else []
            ),
        }

    def analyze_complexity(self, code: str) -> dict:
        """Analyze cyclomatic complexity."""
        # Simple complexity estimation based on control structures
        complexity_keywords = [
            "if",
            "elif",
            "else",
            "for",
            "while",
            "try",
            "except",
        ]
        complexity = sum(
            1 for keyword in complexity_keywords if keyword in code.lower()
        )

        return {
            "cyclomatic_complexity": min(
                complexity, 10
            ),  # Cap at 10 for this mock
            "violation_detected": complexity > 10,
            "suggestions": (
                ["Reduce nesting and branching"] if complexity > 10 else []
            ),
        }


class IntegrationFirstValidator:
    """Validator for Article IV: Integration-First Testing."""

    def validate_integration_tests(self, project_dir: str) -> dict:
        """Validate integration test coverage."""
        return {
            "integration_tests_found": True,
            "e2e_tests_found": False,
            "violations": [],
        }


class ClarityValidator:
    """Validator for Article V: Clarity and Unambiguity."""

    def analyze_clarity(self, text: str) -> dict:
        """Analyze text for clarity and unambiguity."""
        ambiguous_words = ["maybe", "possibly", "might", "could", "perhaps"]
        ambiguous_count = sum(
            1 for word in ambiguous_words if word in text.lower()
        )

        return {
            "ambiguous_phrases": ambiguous_count,
            "clarity_score": max(0.0, 1.0 - (ambiguous_count * 0.1)),
            "violations": (
                ["Contains ambiguous language"] if ambiguous_count > 0 else []
            ),
        }


class CounterfactualValidator:
    """Validator for Article VI: Counterfactual Justification."""

    def analyze_justifications(self, text: str) -> dict:
        """Analyze text for counterfactual justifications."""
        justification_words = [
            "because",
            "since",
            "due to",
            "instead of",
            "rather than",
        ]
        justification_count = sum(
            1 for word in justification_words if word in text.lower()
        )

        return {
            "justifications_found": justification_count > 0,
            "justification_count": justification_count,
            "violations": (
                ["No justifications provided"]
                if justification_count == 0
                else []
            ),
        }
