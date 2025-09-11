"""
Mangle Reasoning Bridge

Provides integration with Mangle deductive reasoning engine:
- Code knowledge graph generation and querying
- Logical inference and fact-based reasoning
- Question answering using deductive database
- Advisory-only operation for seamless integration
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MangleBridge:
    """
    Bridge to Mangle deductive reasoning engine.

    Provides advisory-only integration with graceful degradation
    when Mangle is not available or encounters errors.
    """

    def __init__(self, workspace_path: str | None = None):
        """Initialize the Mangle bridge with optional workspace path."""
        self.workspace_path = Path(workspace_path) if workspace_path else Path.cwd()
        self.mangle_available = False
        self.fact_generator = None
        self.reasoner = None

        # Try to initialize Mangle components
        self._initialize_mangle()

    def _initialize_mangle(self) -> None:
        """Initialize Mangle components with graceful degradation."""
        try:
            # Import Mangle components
            from src.sdd.mangle_fact_generator import MangleFactGenerator
            from src.sdd.mangle_reasoner import MangleReasoner

            self.fact_generator = MangleFactGenerator(str(self.workspace_path))
            self.reasoner = MangleReasoner()
            self.mangle_available = True

            logger.info("Mangle reasoning engine initialized successfully")

        except ImportError as e:
            logger.warning(f"Mangle components not available: {e}")
            self.mangle_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize Mangle: {e}")
            self.mangle_available = False

    def is_available(self) -> bool:
        """Check if Mangle reasoning is available."""
        return self.mangle_available

    def generate_code_insights(self, question: str) -> dict[str, Any]:
        """
        Generate code insights using Mangle reasoning.

        Args:
            question: Question about the codebase

        Returns:
            Dictionary with insights, facts, and analysis
        """
        if not self.mangle_available:
            return self._unavailable_response("Mangle reasoning not available")

        try:
            # Generate facts about the codebase
            facts = self._generate_facts()

            # Query the knowledge base
            answer = self._query_knowledge_base(question, facts)

            # Generate additional insights
            insights = self._extract_insights(facts, question)

            return {
                "available": True,
                "question": question,
                "answer": answer,
                "facts_count": len(facts),
                "insights": insights,
                "metadata": {
                    "workspace_path": str(self.workspace_path),
                    "fact_types": self._analyze_fact_types(facts),
                },
            }

        except Exception as e:
            logger.warning(f"Mangle reasoning failed: {e}")
            return self._error_response(f"Reasoning failed: {str(e)}")

    def _generate_facts(self) -> list[str]:
        """Generate facts about the codebase."""
        if not self.fact_generator:
            return []

        try:
            return self.fact_generator.generate_facts()
        except Exception as e:
            logger.warning(f"Fact generation failed: {e}")
            return []

    def _query_knowledge_base(self, question: str, facts: list[str]) -> str:
        """Query the Mangle knowledge base."""
        if not self.reasoner or not facts:
            return "No reasoning available"

        try:
            # Load facts into reasoner
            for fact in facts:
                self.reasoner.add_fact(fact)

            # Query the reasoner
            return self.reasoner.query(question)

        except Exception as e:
            logger.warning(f"Knowledge base query failed: {e}")
            return f"Query failed: {str(e)}"

    def _extract_insights(self, facts: list[str], question: str) -> dict[str, Any]:
        """Extract additional insights from facts and question context."""
        insights = {
            "code_patterns": [],
            "potential_issues": [],
            "recommendations": [],
            "statistics": {},
        }

        if not facts:
            return insights

        try:
            # Basic fact analysis
            function_facts = [f for f in facts if "function(" in f]
            class_facts = [f for f in facts if "class(" in f]
            test_facts = [f for f in facts if "test_" in f or "_test" in f]

            insights["statistics"] = {
                "total_facts": len(facts),
                "functions": len(function_facts),
                "classes": len(class_facts),
                "tests": len(test_facts),
            }

            # Code pattern detection
            if "untested" in question.lower():
                insights["code_patterns"].append("Analyzing test coverage patterns")

                # Find potentially untested functions
                tested_functions = set()
                for test_fact in test_facts:
                    # Extract function names being tested
                    # This is a simplified heuristic
                    if "test_" in test_fact:
                        tested_name = test_fact.replace("test_", "").split("(")[0]
                        tested_functions.add(tested_name)

                all_functions = set()
                for func_fact in function_facts:
                    # Extract function names
                    func_name = func_fact.split("(")[1].split(",")[0].strip("\"'")
                    all_functions.add(func_name)

                untested = all_functions - tested_functions
                if untested:
                    insights["potential_issues"].extend(
                        [
                            f"Potentially untested function: {func}"
                            for func in list(untested)[:5]
                        ]
                    )

            # Question-specific analysis
            if "complex" in question.lower():
                insights["recommendations"].append(
                    "Consider simplifying complex functions"
                )

            if "dependency" in question.lower():
                insights["code_patterns"].append("Analyzing dependency patterns")

        except Exception as e:
            logger.warning(f"Insight extraction failed: {e}")
            insights["error"] = str(e)

        return insights

    def _analyze_fact_types(self, facts: list[str]) -> dict[str, int]:
        """Analyze the types of facts in the knowledge base."""
        fact_types = {}

        for fact in facts:
            if "function(" in fact:
                fact_types["functions"] = fact_types.get("functions", 0) + 1
            elif "class(" in fact:
                fact_types["classes"] = fact_types.get("classes", 0) + 1
            elif "import(" in fact:
                fact_types["imports"] = fact_types.get("imports", 0) + 1
            elif "test_" in fact:
                fact_types["tests"] = fact_types.get("tests", 0) + 1
            else:
                fact_types["other"] = fact_types.get("other", 0) + 1

        return fact_types

    def query_codebase(self, query: str) -> dict[str, Any]:
        """
        Query the codebase using Mangle reasoning.

        Args:
            query: Natural language query about the code

        Returns:
            Query results with insights and recommendations
        """
        if not self.mangle_available:
            return self._unavailable_response("Mangle querying not available")

        try:
            # Common query patterns
            if any(
                pattern in query.lower()
                for pattern in ["untested", "no test", "missing test"]
            ):
                return self._analyze_test_coverage()
            elif any(
                pattern in query.lower()
                for pattern in ["complex", "complicated", "hard to"]
            ):
                return self._analyze_complexity()
            elif any(
                pattern in query.lower()
                for pattern in ["dependency", "import", "module"]
            ):
                return self._analyze_dependencies()
            else:
                return self.generate_code_insights(query)

        except Exception as e:
            logger.warning(f"Codebase query failed: {e}")
            return self._error_response(f"Query failed: {str(e)}")

    def _analyze_test_coverage(self) -> dict[str, Any]:
        """Analyze test coverage using Mangle facts."""
        try:
            facts = self._generate_facts()

            # Extract function and test information
            functions = []
            tests = []

            for fact in facts:
                if "function(" in fact and "test_" not in fact:
                    # Extract function name from fact
                    parts = fact.split("(")
                    if len(parts) > 1:
                        func_name = parts[1].split(",")[0].strip("\"'")
                        functions.append(func_name)
                elif "test_" in fact or "_test" in fact:
                    tests.append(fact)

            # Simple heuristic for test coverage
            tested_functions = set()
            for test in tests:
                for func in functions:
                    if func.lower() in test.lower():
                        tested_functions.add(func)

            untested = [f for f in functions if f not in tested_functions]

            return {
                "available": True,
                "analysis_type": "test_coverage",
                "total_functions": len(functions),
                "tested_functions": len(tested_functions),
                "untested_functions": len(untested),
                "untested_list": untested[:10],  # Limit to 10
                "coverage_ratio": (
                    len(tested_functions) / len(functions) if functions else 0
                ),
                "recommendations": (
                    [f"Add tests for {func}" for func in untested[:3]]
                    if untested
                    else ["Good test coverage detected"]
                ),
            }

        except Exception as e:
            logger.warning(f"Test coverage analysis failed: {e}")
            return self._error_response(f"Test analysis failed: {str(e)}")

    def _analyze_complexity(self) -> dict[str, Any]:
        """Analyze code complexity using available information."""
        try:
            facts = self._generate_facts()

            # Simple complexity heuristics
            complex_indicators = []

            for fact in facts:
                if any(
                    indicator in fact.lower()
                    for indicator in [
                        "nested",
                        "loop",
                        "if",
                        "complex",
                        "many",
                        "large",
                    ]
                ):
                    complex_indicators.append(fact)

            return {
                "available": True,
                "analysis_type": "complexity",
                "complexity_indicators": len(complex_indicators),
                "sample_indicators": complex_indicators[:5],
                "recommendations": [
                    "Consider breaking down complex functions",
                    "Look for opportunities to simplify logic",
                    "Use helper functions to reduce complexity",
                ],
            }

        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            return self._error_response(f"Complexity analysis failed: {str(e)}")

    def _analyze_dependencies(self) -> dict[str, Any]:
        """Analyze dependencies using Mangle facts."""
        try:
            facts = self._generate_facts()

            # Extract import information
            imports = [f for f in facts if "import(" in f]
            external_deps = []
            internal_deps = []

            for import_fact in imports:
                if "src." in import_fact or "./" in import_fact:
                    internal_deps.append(import_fact)
                else:
                    external_deps.append(import_fact)

            return {
                "available": True,
                "analysis_type": "dependencies",
                "total_imports": len(imports),
                "external_dependencies": len(external_deps),
                "internal_dependencies": len(internal_deps),
                "sample_external": external_deps[:5],
                "sample_internal": internal_deps[:5],
                "recommendations": [
                    "Review external dependencies for necessity",
                    "Consider consolidating similar libraries",
                    "Document dependency relationships",
                ],
            }

        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
            return self._error_response(f"Dependency analysis failed: {str(e)}")

    def _unavailable_response(self, message: str) -> dict[str, Any]:
        """Return response when Mangle is unavailable."""
        return {
            "available": False,
            "message": message,
            "fallback": "Using basic heuristics instead of Mangle reasoning",
            "recommendations": [
                "Install Mangle dependencies for enhanced reasoning",
                "Check Mangle configuration and setup",
            ],
        }

    def _error_response(self, error: str) -> dict[str, Any]:
        """Return response when Mangle encounters an error."""
        return {
            "available": False,
            "error": error,
            "fallback": "Graceful degradation to basic analysis",
            "recommendations": [
                "Check Mangle configuration",
                "Verify workspace structure",
                "Review error logs for details",
            ],
        }

    def get_status(self) -> dict[str, Any]:
        """Get status information about Mangle integration."""
        return {
            "mangle_available": self.mangle_available,
            "workspace_path": str(self.workspace_path),
            "fact_generator_initialized": self.fact_generator is not None,
            "reasoner_initialized": self.reasoner is not None,
            "capabilities": (
                [
                    "code_insights",
                    "test_coverage_analysis",
                    "complexity_analysis",
                    "dependency_analysis",
                    "codebase_querying",
                ]
                if self.mangle_available
                else ["basic_fallback"]
            ),
        }

    def validate_setup(self) -> dict[str, Any]:
        """Validate Mangle setup and configuration."""
        validation = {
            "valid": False,
            "issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check workspace
        if not self.workspace_path.exists():
            validation["issues"].append(
                f"Workspace path does not exist: {self.workspace_path}"
            )
        elif not self.workspace_path.is_dir():
            validation["issues"].append(
                f"Workspace path is not a directory: {self.workspace_path}"
            )

        # Check Mangle availability
        if not self.mangle_available:
            validation["issues"].append("Mangle components not available")
            validation["recommendations"].extend(
                [
                    "Install Mangle dependencies",
                    "Check Python path includes src/sdd/",
                    "Verify Mangle setup scripts have been run",
                ]
            )

        # Try basic operation
        if self.mangle_available:
            try:
                facts = self._generate_facts()
                if not facts:
                    validation["warnings"].append("No facts generated from workspace")
                else:
                    validation["valid"] = True

            except Exception as e:
                validation["issues"].append(f"Basic operation failed: {e}")

        return validation
