"""
Mangle Reasoning Ability for GitHub Copilot Agent Mode.

This ability integrates Mangle deductive reasoning directly into the GitHub Copilot
agent workflow, providing automatic code knowledge graph analysis for any question
or request.
"""

import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import dependencies with fallbacks for missing modules
try:
    from src.constitutional.scorer import ConstitutionalScorer
except ImportError:
    try:
        from constitutional.scorer import ConstitutionalScorer
    except ImportError:
        ConstitutionalScorer = None

try:
    from src.sdd.enhanced_sdd_framework import EnhancedSDDFramework
except ImportError:
    try:
        from sdd.enhanced_sdd_framework import EnhancedSDDFramework
    except ImportError:
        EnhancedSDDFramework = None

try:
    from src.sdd.mangle_reasoner import MangleReasoner
except ImportError:
    try:
        from sdd.mangle_reasoner import MangleReasoner
    except ImportError:
        MangleReasoner = None

try:
    from src.sdd.mangle_rules import get_available_queries, get_query_for_question
except ImportError:
    try:
        from sdd.mangle_rules import get_available_queries, get_query_for_question
    except ImportError:

        def get_available_queries():
            return ["untested functions", "constitutional violations", "quality issues"]

        def get_query_for_question(question):
            return None


class MangleReasoningAbility:
    """
    Native Mangle reasoning integration for GitHub Copilot agent mode.

    This ability automatically enhances all user interactions with:
    - Code knowledge graph analysis
    - Constitutional compliance checking
    - Natural language code querying
    - Specification-to-code traceability
    """

    def __init__(self, workspace_root: str = "."):
        """Initialize the Mangle reasoning ability."""
        self.workspace_root = workspace_root

        # Initialize components with fallbacks
        self.reasoner = MangleReasoner(workspace_root) if MangleReasoner else None
        self.sdd_framework = (
            EnhancedSDDFramework(Path(workspace_root)) if EnhancedSDDFramework else None
        )
        self.constitutional_scorer = (
            ConstitutionalScorer() if ConstitutionalScorer else None
        )

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions for the agent registry."""
        return [
            {
                "name": "mangle_ask_question",
                "description": "Ask natural language questions about code quality, constitutional compliance, and codebase analysis using Mangle deductive reasoning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Natural language question about the codebase (e.g., 'what functions are untested', 'what violates constitution')",
                        }
                    },
                    "required": ["question"],
                },
            },
            {
                "name": "mangle_analyze_context",
                "description": "Automatically analyze the current code context for quality issues and constitutional compliance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context_type": {
                            "type": "string",
                            "enum": ["current_file", "current_project", "workspace"],
                            "description": "Scope of analysis",
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific areas to focus on (constitutional, quality, coverage, dependencies)",
                        },
                    },
                    "required": ["context_type"],
                },
            },
            {
                "name": "mangle_get_suggestions",
                "description": "Get contextual improvement suggestions based on Mangle analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_element": {
                            "type": "string",
                            "description": "Specific code element to analyze (function name, class name, file path)",
                        }
                    },
                    "required": ["code_element"],
                },
            },
        ]

    async def execute_tool(
        self, tool_name: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a Mangle reasoning tool."""
        try:
            if tool_name == "mangle_ask_question":
                return await self._ask_question(parameters["question"])
            elif tool_name == "mangle_analyze_context":
                return await self._analyze_context(
                    parameters["context_type"],
                    parameters.get("focus_areas", ["constitutional", "quality"]),
                )
            elif tool_name == "mangle_get_suggestions":
                return await self._get_suggestions(parameters["code_element"])
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": f"Mangle reasoning failed: {str(e)}"}

    async def _ask_question(self, question: str) -> dict[str, Any]:
        """Process natural language question using Mangle reasoning."""
        # Use enhanced SDD framework for question answering
        answer = self.sdd_framework.ask_question(question)

        # Also try to get direct Mangle query results
        query = get_query_for_question(question)
        results = []
        if query:
            try:
                results = self.reasoner.query(query)
            except Exception:
                # Fallback if Mangle binary not available
                pass

        return {
            "question": question,
            "answer": answer,
            "query_used": query,
            "raw_results": results,
            "available_patterns": get_available_queries()[:10],  # Show first 10
            "suggestion": "Try asking about specific areas like 'untested functions', 'constitutional violations', or 'quality issues'",
        }

    async def _analyze_context(
        self, context_type: str, focus_areas: list[str]
    ) -> dict[str, Any]:
        """Analyze current context automatically."""
        analysis_results = {}

        # Run constitutional validation
        if "constitutional" in focus_areas:
            try:
                constitutional_results = (
                    self.sdd_framework.validate_constitutional_compliance()
                )
                analysis_results["constitutional"] = constitutional_results
            except Exception as e:
                analysis_results["constitutional"] = {"error": str(e)}

        # Run quality analysis
        if "quality" in focus_areas:
            quality_queries = [
                "quality_issue(Type, Entity)",
                "complex_function(Func)",
                "circular_dependency(Module1, Module2)",
            ]
            quality_results = {}
            for query in quality_queries:
                try:
                    results = self.reasoner.query(query)
                    quality_results[query] = results
                except Exception:
                    quality_results[query] = []
            analysis_results["quality"] = quality_results

        # Run coverage analysis
        if "coverage" in focus_areas:
            coverage_queries = ["untested_function(Func)", "poor_test_coverage(Module)"]
            coverage_results = {}
            for query in coverage_queries:
                try:
                    results = self.reasoner.query(query)
                    coverage_results[query] = results
                except Exception:
                    coverage_results[query] = []
            analysis_results["coverage"] = coverage_results

        # Generate summary
        summary = self._generate_analysis_summary(analysis_results)

        return {
            "context_type": context_type,
            "focus_areas": focus_areas,
            "analysis": analysis_results,
            "summary": summary,
            "recommendations": self._generate_recommendations(analysis_results),
        }

    async def _get_suggestions(self, code_element: str) -> dict[str, Any]:
        """Get specific suggestions for a code element."""
        suggestions = []

        # Check if it's an untested function
        try:
            untested_results = self.reasoner.query("untested_function(Func)")
            if any(code_element in str(result) for result in untested_results):
                suggestions.append(
                    {
                        "type": "testing",
                        "severity": "high",
                        "message": f"Function '{code_element}' appears to be untested",
                        "action": "Add comprehensive test coverage",
                    }
                )
        except Exception:
            pass

        # Check constitutional compliance
        try:
            violation_results = self.reasoner.query(
                "constitutional_violation(Article, Violator)"
            )
            for result in violation_results:
                if isinstance(result, list) and len(result) >= 2:
                    if code_element in str(result[1]):
                        suggestions.append(
                            {
                                "type": "constitutional",
                                "severity": "high",
                                "message": f"Violates Constitutional Article {result[0]}",
                                "action": f"Review Article {result[0]} requirements",
                            }
                        )
        except Exception:
            pass

        # Check complexity
        try:
            complex_results = self.reasoner.query("complex_function(Func)")
            if any(code_element in str(result) for result in complex_results):
                suggestions.append(
                    {
                        "type": "complexity",
                        "severity": "medium",
                        "message": f"Function '{code_element}' may be too complex",
                        "action": "Consider breaking into smaller functions",
                    }
                )
        except Exception:
            pass

        return {
            "code_element": code_element,
            "suggestions": suggestions,
            "analysis_timestamp": "now",
        }

    def _generate_analysis_summary(self, analysis_results: dict[str, Any]) -> str:
        """Generate a human-readable summary of analysis results."""
        summary_parts = []

        # Constitutional summary
        if "constitutional" in analysis_results:
            const_results = analysis_results["constitutional"]
            violation_count = sum(
                len(results)
                for results in const_results.values()
                if isinstance(results, list)
            )
            if violation_count > 0:
                summary_parts.append(
                    f"Found {violation_count} constitutional violations"
                )
            else:
                summary_parts.append("No constitutional violations detected")

        # Quality summary
        if "quality" in analysis_results:
            quality_results = analysis_results["quality"]
            issue_count = sum(
                len(results)
                for results in quality_results.values()
                if isinstance(results, list)
            )
            if issue_count > 0:
                summary_parts.append(f"Found {issue_count} quality issues")
            else:
                summary_parts.append("No quality issues detected")

        # Coverage summary
        if "coverage" in analysis_results:
            coverage_results = analysis_results["coverage"]
            untested_count = len(coverage_results.get("untested_function(Func)", []))
            if untested_count > 0:
                summary_parts.append(f"Found {untested_count} untested functions")
            else:
                summary_parts.append("All functions appear to be tested")

        return (
            "; ".join(summary_parts)
            if summary_parts
            else "Analysis completed successfully"
        )

    def _generate_recommendations(self, analysis_results: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Constitutional recommendations
        if "constitutional" in analysis_results:
            const_results = analysis_results["constitutional"]
            for query, results in const_results.items():
                if isinstance(results, list) and len(results) > 0:
                    if "untested_function" in query:
                        recommendations.append(
                            "Implement test-first development (Article II)"
                        )
                    elif "violates_simplicity" in query:
                        recommendations.append(
                            "Simplify complex functions (Article III)"
                        )
                    elif "violates_library_first" in query:
                        recommendations.append(
                            "Use existing libraries instead of custom implementations (Article I)"
                        )

        # Quality recommendations
        if "quality" in analysis_results:
            quality_results = analysis_results["quality"]
            if quality_results.get("complex_function(Func)"):
                recommendations.append(
                    "Refactor complex functions to improve maintainability"
                )
            if quality_results.get("circular_dependency(Module1, Module2)"):
                recommendations.append(
                    "Resolve circular dependencies to improve architecture"
                )

        # Coverage recommendations
        if "coverage" in analysis_results:
            coverage_results = analysis_results["coverage"]
            if coverage_results.get("untested_function(Func)"):
                recommendations.append("Add test coverage for untested functions")
            if coverage_results.get("poor_test_coverage(Module)"):
                recommendations.append(
                    "Improve test coverage for modules with poor coverage"
                )

        return recommendations[:5]  # Limit to top 5 recommendations

    def enhance_user_input(self, user_input: str) -> dict[str, Any]:
        """
        Automatically enhance user input with Mangle context.

        This method is called by the agent framework to provide contextual
        code knowledge graph information for any user request.
        """
        # Check if this is a question that can benefit from Mangle analysis
        query = get_query_for_question(user_input)

        enhancement = {
            "original_input": user_input,
            "mangle_context": {
                "can_answer": query is not None,
                "suggested_query": query,
                "available_patterns": get_available_queries()[:5],
            },
        }

        # For code-related questions, automatically run analysis
        if any(
            keyword in user_input.lower()
            for keyword in [
                "test",
                "function",
                "code",
                "quality",
                "constitutional",
                "violation",
                "dependency",
                "complex",
                "coverage",
                "spec",
                "feature",
            ]
        ):
            try:
                # Quick analysis to provide context
                if query:
                    results = self.reasoner.query(query)
                    enhancement["mangle_context"]["auto_analysis"] = {
                        "query": query,
                        "result_count": (
                            len(results) if isinstance(results, list) else 0
                        ),
                        "results": (
                            results[:3] if isinstance(results, list) else []
                        ),  # First 3 results
                    }
            except Exception:
                enhancement["mangle_context"]["auto_analysis"] = {
                    "error": "Mangle analysis unavailable (binary not found)"
                }

        return enhancement
