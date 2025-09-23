"""
Mangle reasoner for executing queries against the code knowledge graph.

This module provides the core reasoning engine that:
- Executes Mangle queries against generated facts
- Parses and formats query results
- Handles temporary file management for Mangle execution
- Provides caching and optimization for repeated queries
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .mangle_integration import MangleFactGenerator
from .mangle_rules import MANGLE_RULES, get_query_for_question

logger = logging.getLogger(__name__)


class MangleQueryResult:
    """
    Represents the result of a Mangle query execution.

    Contains both raw results and formatted output for different use cases.
    """

    def __init__(
        self,
        query: str,
        raw_results: list[list[str]],
        success: bool = True,
        error_message: str | None = None,
        execution_time: float = 0.0,
    ):
        """
        Initialize query result.

        Args:
            query: The original Mangle query
            raw_results: Raw parsed results from Mangle
            success: Whether the query executed successfully
            error_message: Error message if execution failed
            execution_time: Query execution time in seconds
        """
        self.query = query
        self.raw_results = raw_results
        self.success = success
        self.error_message = error_message
        self.execution_time = execution_time

    def as_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "query": self.query,
            "results": self.raw_results,
            "success": self.success,
            "error": self.error_message,
            "execution_time": self.execution_time,
            "count": len(self.raw_results) if self.raw_results else 0,
        }

    def as_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.as_dict(), indent=2)

    def format_for_display(self) -> str:
        """Format results for human-readable display."""
        if not self.success:
            return f"Query failed: {self.error_message}"

        if not self.raw_results:
            return "No results found."

        lines = [f"Query: {self.query}", f"Results ({len(self.raw_results)}):"]

        for i, result in enumerate(self.raw_results, 1):
            if isinstance(result, list) and result:
                formatted_result = ", ".join(str(item) for item in result)
                lines.append(f"  {i}. {formatted_result}")
            else:
                lines.append(f"  {i}. {result}")

        lines.append(f"Execution time: {self.execution_time:.3f}s")
        return "\n".join(lines)


class MangleReasoner:
    """
    Core reasoning engine that executes Mangle queries against the knowledge graph.

    This class manages the execution of deductive queries, handles temporary files,
    provides caching for performance, and formats results for different consumers.
    """

    def __init__(self, workspace_root: str = ".", mangle_executable: str = "mangle"):
        """
        Initialize the Mangle reasoner.

        Args:
            workspace_root: Root directory of the workspace to analyze
            mangle_executable: Path to the Mangle executable
        """
        self.workspace_root = Path(workspace_root)
        self.mangle_executable = mangle_executable
        self.fact_generator = MangleFactGenerator(workspace_root)

        # Cache for facts to avoid regeneration
        self._facts_cache: str | None = None
        self._facts_cache_valid = False
        self._injected_facts: list[str] = []

        # Query result cache
        self._query_cache: dict[str, MangleQueryResult] = {}


    def add_fact(self, fact: str) -> None:
        """Inject an additional fact into the knowledge base."""
        if not fact:
            return
        self._injected_facts.append(fact)
        self._facts_cache_valid = False


    def clear_injected_facts(self) -> None:
        """Clear all injected facts and invalidate cache."""
        if self._injected_facts:
            self._injected_facts.clear()
            self._facts_cache_valid = False

    def query(self, query: str, use_cache: bool = True) -> MangleQueryResult:
        """
        Execute a Mangle query and return structured results.

        Args:
            query: Mangle query string to execute
            use_cache: Whether to use cached results if available

        Returns:
            MangleQueryResult containing query results and metadata
        """
        # Check cache first
        if use_cache and query in self._query_cache:
            logger.debug(f"Using cached result for query: {query}")
            return self._query_cache[query]

        logger.info(f"Executing Mangle query: {query}")

        import time

        start_time = time.time()

        try:
            # Get facts (with caching)
            facts = self._get_facts()

            # Create query result
            result = self._execute_mangle_query(query, facts)
            result.execution_time = time.time() - start_time

            # Cache successful results
            if use_cache and result.success:
                self._query_cache[query] = result

            return result

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return MangleQueryResult(
                query=query,
                raw_results=[],
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    def ask_question(self, question: str) -> MangleQueryResult:
        """
        Answer a natural language question using Mangle reasoning.

        Args:
            question: Natural language question about the codebase

        Returns:
            MangleQueryResult with the answer
        """
        # Try to map question to a known query
        mangle_query = get_query_for_question(question)

        if mangle_query:
            return self.query(mangle_query)
        else:
            return MangleQueryResult(
                query=question,
                raw_results=[],
                success=False,
                error_message=f"Could not understand question: '{question}'. "
                f"Try asking about: untested functions, incomplete features, "
                f"constitutional violations, quality issues, etc.",
            )

    def validate_constitutional_compliance(self) -> dict[str, MangleQueryResult]:
        """
        Validate code against all constitutional rules using Mangle.

        Returns:
            Dictionary mapping article names to query results
        """
        constitutional_queries = {
            "Article I (Library-First)": "violates_library_first(Module)",
            "Article II (Test-First)": "violates_test_first(Func)",
            "Article III (Simplicity)": "violates_simplicity(Func)",
            "Article IV (Integration-First)": "violates_integration_first(FeatureID)",
            "Article V (Clarity)": "violates_clarity(Func)",
            "Article VI (Counterfactual)": "violates_counterfactual(Decision)",
            "All Violations": "constitutional_violation(Article, Violator)",
        }

        results = {}
        for article, query in constitutional_queries.items():
            results[article] = self.query(query)

        return results

    def trace_code_to_spec(self, code_element: str) -> MangleQueryResult:
        """
        Trace a code element back to its specification.

        Args:
            code_element: Name of function, class, or module to trace

        Returns:
            MangleQueryResult showing related specifications
        """
        query = f'spec_for_function("{code_element}", FeatureID)'
        return self.query(query)

    def find_quality_issues(self) -> dict[str, MangleQueryResult]:
        """
        Find all code quality issues using Mangle reasoning.

        Returns:
            Dictionary mapping issue types to query results
        """
        quality_queries = {
            "Untested Functions": "untested_function(Func)",
            "Complex Functions": "complex_function(Func)",
            "Large Classes": "large_class(Class)",
            "God Files": "god_file(File)",
            "Circular Dependencies": "circular_dependency(Module1, Module2)",
            "Dependency Hotspots": "dependency_hotspot(Module)",
            "All Quality Issues": "quality_issue(Type, Entity)",
        }

        results = {}
        for issue_type, query in quality_queries.items():
            results[issue_type] = self.query(query)

        return results

    def find_incomplete_work(self) -> dict[str, MangleQueryResult]:
        """
        Find incomplete work items using Mangle reasoning.

        Returns:
            Dictionary mapping work types to query results
        """
        incomplete_queries = {
            "Incomplete Features": "incomplete_feature(FeatureID)",
            "Orphaned Specifications": "orphaned_spec(FeatureID)",
            "Poor Test Coverage": "poor_test_coverage(Module)",
            "All Incomplete Work": "incomplete_work(Type, Entity)",
        }

        results = {}
        for work_type, query in incomplete_queries.items():
            results[work_type] = self.query(query)

        return results

    def invalidate_cache(self):
        """Invalidate all caches to force regeneration of facts and queries."""
        self._facts_cache = None
        self._facts_cache_valid = False
        self._query_cache.clear()
        logger.info("Mangle reasoner caches invalidated")

    def get_fact_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the generated facts.

        Returns:
            Dictionary with fact counts and cache status
        """
        facts = self._get_facts()
        fact_stats = self.fact_generator.get_fact_statistics()

        return {
            "total_facts": len(self.fact_generator._generated_facts),
            "fact_types": fact_stats,
            "cache_valid": self._facts_cache_valid,
            "cached_queries": len(self._query_cache),
            "facts_size_bytes": len(facts.encode("utf-8")),
        }

    def _get_facts(self) -> str:
        """
        Get facts with caching for performance.

        Returns:
            String containing all Mangle facts and rules
        """
        if self._facts_cache is None or not self._facts_cache_valid:
            logger.debug("Regenerating facts cache")
            facts = self.fact_generator.generate_all_facts()
            injected = "\n".join(self._injected_facts) if self._injected_facts else ""
            if injected:
                combined = "\n".join([facts, injected])
            else:
                combined = facts
            self._facts_cache = combined + "\n\n" + MANGLE_RULES
            self._facts_cache_valid = True

        return self._facts_cache

    def _execute_mangle_query(self, query: str, facts: str) -> MangleQueryResult:
        """
        Execute a single Mangle query with error handling.

        Args:
            query: Mangle query to execute
            facts: Combined facts and rules string

        Returns:
            MangleQueryResult with parsed results
        """
        # Create temporary files for facts and query
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mangle", delete=False, encoding="utf-8"
        ) as facts_file:
            facts_file.write(facts)
            facts_path = Path(facts_file.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".query", delete=False, encoding="utf-8"
        ) as query_file:
            query_file.write(query)
            query_path = Path(query_file.name)

        try:
            # Try to execute Mangle query
            result = self._run_mangle_command(facts_path, query_path, query)
            return result

        finally:
            # Clean up temporary files
            facts_path.unlink(missing_ok=True)
            query_path.unlink(missing_ok=True)

    def _run_mangle_command(
        self, facts_path: Path, query_path: Path, original_query: str
    ) -> MangleQueryResult:
        """
        Run the actual Mangle command and parse results.

        Args:
            facts_path: Path to facts file
            query_path: Path to query file
            original_query: Original query string for result object

        Returns:
            MangleQueryResult with parsed output
        """
        try:
            # Execute Mangle query
            cmd = [
                self.mangle_executable,
                "query",
                "--file",
                str(facts_path),
                "--query",
                str(query_path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.workspace_root,
                timeout=30,  # 30 second timeout
            )

            if result.returncode == 0:
                parsed_results = self._parse_mangle_output(result.stdout)
                return MangleQueryResult(
                    query=original_query, raw_results=parsed_results, success=True
                )
            else:
                # Handle Mangle execution errors
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.warning(f"Mangle query failed: {error_msg}")

                # Try to provide helpful error messages
                if "not found" in error_msg.lower():
                    error_msg = (
                        f"Mangle executable not found. Please install Mangle and ensure "
                        f"'{self.mangle_executable}' is in your PATH."
                    )

                return MangleQueryResult(
                    query=original_query,
                    raw_results=[],
                    success=False,
                    error_message=error_msg,
                )

        except subprocess.TimeoutExpired:
            return MangleQueryResult(
                query=original_query,
                raw_results=[],
                success=False,
                error_message="Query timed out after 30 seconds",
            )
        except FileNotFoundError:
            return MangleQueryResult(
                query=original_query,
                raw_results=[],
                success=False,
                error_message=f"Mangle executable '{self.mangle_executable}' not found. "
                f"Please install Mangle and ensure it's in your PATH.",
            )
        except Exception as e:
            return MangleQueryResult(
                query=original_query,
                raw_results=[],
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    def _parse_mangle_output(self, output: str) -> list[list[str]]:
        """
        Parse Mangle query output into structured data.

        Args:
            output: Raw output from Mangle execution

        Returns:
            List of result tuples, each as a list of strings
        """
        results = []

        for line in output.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            # Parse tuple output like (value1, value2, value3)
            if line.startswith("(") and line.endswith(")"):
                # Remove parentheses and split by comma
                content = line[1:-1]
                if content:
                    # Handle quoted strings and simple values
                    values = []
                    current_value = ""
                    in_quotes = False

                    for char in content:
                        if char == '"' and (
                            not current_value or current_value[-1] != "\\"
                        ):
                            in_quotes = not in_quotes
                        elif char == "," and not in_quotes:
                            values.append(current_value.strip().strip('"'))
                            current_value = ""
                        else:
                            current_value += char

                    # Add the last value
                    if current_value:
                        values.append(current_value.strip().strip('"'))

                    if values:
                        results.append(values)

            # Handle simple single-value results
            elif line and not line.startswith("//"):
                results.append([line])

        return results
