"""
Enhanced GitHub Copilot Agent with native Mangle integration.

This module provides automatic Code Knowledge Graph reasoning for all user
interactions, making Mangle analysis a seamless part of the Copilot experience.
"""

from typing import Any

from src.abilities.mangle_reasoning_ability import MangleReasoningAbility
from src.sdd.enhanced_sdd_framework import EnhancedSDDFramework


class MangleEnhancedAgent:
    """
    GitHub Copilot agent enhanced with automatic Mangle reasoning.

    This agent automatically:
    - Analyzes user questions for code knowledge graph opportunities
    - Provides constitutional compliance context
    - Enhances responses with deductive reasoning insights
    - Maintains awareness of code quality and specifications
    """

    def __init__(self, workspace_root: str = "."):
        """Initialize the enhanced agent."""
        self.workspace_root = workspace_root
        self.mangle_ability = MangleReasoningAbility(workspace_root)
        self.sdd_framework = EnhancedSDDFramework(workspace_root)

    async def process_user_input(
        self, user_input: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process user input with automatic Mangle enhancement.

        This is the main entry point for GitHub Copilot interactions.
        """
        # Enhance the user input with Mangle context
        enhancement = self.mangle_ability.enhance_user_input(user_input)

        # Determine response strategy based on input type
        response_strategy = self._determine_response_strategy(
            user_input, enhancement
        )

        # Generate enhanced response
        response = await self._generate_enhanced_response(
            user_input, enhancement, response_strategy, context
        )

        return response

    def _determine_response_strategy(
        self, user_input: str, enhancement: dict[str, Any]
    ) -> str:
        """Determine the best response strategy based on input analysis."""
        user_lower = user_input.lower()

        # Direct Mangle questions
        if enhancement["mangle_context"]["can_answer"]:
            return "mangle_query"

        # Constitutional compliance questions
        if any(
            word in user_lower
            for word in [
                "constitutional",
                "compliance",
                "violation",
                "article",
            ]
        ):
            return "constitutional_analysis"

        # Code quality questions
        if any(
            word in user_lower
            for word in ["quality", "test", "complex", "coverage", "refactor"]
        ):
            return "quality_analysis"

        # Specification questions
        if any(
            word in user_lower
            for word in ["spec", "feature", "requirement", "acceptance"]
        ):
            return "specification_analysis"

        # Code implementation questions
        if any(
            word in user_lower
            for word in ["implement", "code", "function", "class", "method"]
        ):
            return "implementation_guidance"

        # General questions with Mangle context
        return "contextual_response"

    async def _generate_enhanced_response(
        self,
        user_input: str,
        enhancement: dict[str, Any],
        strategy: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate response enhanced with Mangle insights."""

        response = {
            "user_input": user_input,
            "strategy": strategy,
            "mangle_context": enhancement["mangle_context"],
            "response": "",
            "actions": [],
            "insights": [],
        }

        try:
            if strategy == "mangle_query":
                response.update(await self._handle_mangle_query(user_input))

            elif strategy == "constitutional_analysis":
                response.update(
                    await self._handle_constitutional_analysis(user_input)
                )

            elif strategy == "quality_analysis":
                response.update(
                    await self._handle_quality_analysis(user_input)
                )

            elif strategy == "specification_analysis":
                response.update(
                    await self._handle_specification_analysis(user_input)
                )

            elif strategy == "implementation_guidance":
                response.update(
                    await self._handle_implementation_guidance(
                        user_input, context
                    )
                )

            else:  # contextual_response
                response.update(
                    await self._handle_contextual_response(
                        user_input, enhancement
                    )
                )

        except Exception as e:
            response["response"] = (
                f"I encountered an issue with Mangle analysis: {str(e)}"
            )
            response["insights"].append(
                "Mangle reasoning temporarily unavailable"
            )

        return response

    async def _handle_mangle_query(self, user_input: str) -> dict[str, Any]:
        """Handle direct Mangle queries."""
        result = await self.mangle_ability.execute_tool(
            "mangle_ask_question", {"question": user_input}
        )

        return {
            "response": result.get("answer", "No answer available"),
            "insights": [
                f"Query executed: {result.get('query_used', 'N/A')}",
                f"Results found: {len(result.get('raw_results', []))}",
            ],
            "actions": ["mangle_query_executed"],
            "mangle_results": result,
        }

    async def _handle_constitutional_analysis(
        self, user_input: str
    ) -> dict[str, Any]:
        """Handle constitutional compliance analysis."""
        # Run constitutional validation
        const_results = self.sdd_framework.validate_constitutional_compliance()

        # Count violations
        total_violations = sum(
            len(results)
            for results in const_results.values()
            if isinstance(results, list)
        )

        if total_violations == 0:
            response_text = "✅ **Constitutional Analysis**: No violations detected! Your codebase appears to comply with all six constitutional articles."
        else:
            response_text = f"⚠️ **Constitutional Analysis**: Found {total_violations} violations across the six constitutional articles."

        insights = [
            "Constitutional Articles Checked: 6",
            f"Total Violations: {total_violations}",
            "Articles: Library-First, Test-First, Simplicity, Integration, Clarity, Counterfactual",
        ]

        # Add specific violation details
        for query, results in const_results.items():
            if isinstance(results, list) and len(results) > 0:
                insights.append(f"{query}: {len(results)} violations")

        return {
            "response": response_text,
            "insights": insights,
            "actions": ["constitutional_analysis_completed"],
            "constitutional_results": const_results,
        }

    async def _handle_quality_analysis(
        self, user_input: str
    ) -> dict[str, Any]:
        """Handle code quality analysis."""
        analysis_result = await self.mangle_ability.execute_tool(
            "mangle_analyze_context",
            {
                "context_type": "workspace",
                "focus_areas": ["quality", "coverage"],
            },
        )

        summary = analysis_result.get("summary", "Analysis completed")
        recommendations = analysis_result.get("recommendations", [])

        response_text = f"🔍 **Quality Analysis**: {summary}\n\n"
        if recommendations:
            response_text += "**Recommendations:**\n"
            for i, rec in enumerate(recommendations[:3], 1):
                response_text += f"{i}. {rec}\n"

        return {
            "response": response_text,
            "insights": [
                f"Analysis scope: {analysis_result.get('context_type', 'workspace')}",
                f"Recommendations generated: {len(recommendations)}",
            ],
            "actions": ["quality_analysis_completed"],
            "quality_results": analysis_result,
        }

    async def _handle_specification_analysis(
        self, user_input: str
    ) -> dict[str, Any]:
        """Handle specification-related analysis."""
        # Check for incomplete features and orphaned specs
        try:
            incomplete_features = self.mangle_ability.reasoner.query(
                "incomplete_feature(FeatureID)"
            )
            orphaned_specs = self.mangle_ability.reasoner.query(
                "orphaned_spec(FeatureID)"
            )

            response_text = "📋 **Specification Analysis**:\n\n"

            if incomplete_features:
                response_text += f"**Incomplete Features**: {len(incomplete_features)} found\n"
                for feature in incomplete_features[:3]:
                    response_text += f"- {feature}\n"

            if orphaned_specs:
                response_text += f"**Orphaned Specifications**: {len(orphaned_specs)} found\n"
                for spec in orphaned_specs[:3]:
                    response_text += f"- {spec}\n"

            if not incomplete_features and not orphaned_specs:
                response_text += "All specifications appear to be complete and connected to code!"

            insights = [
                f"Incomplete features: {len(incomplete_features)}",
                f"Orphaned specifications: {len(orphaned_specs)}",
            ]

        except Exception:
            response_text = "📋 **Specification Analysis**: Unable to analyze specifications (Mangle binary not available)"
            insights = ["Specification analysis requires Mangle binary"]

        return {
            "response": response_text,
            "insights": insights,
            "actions": ["specification_analysis_completed"],
        }

    async def _handle_implementation_guidance(
        self, user_input: str, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Handle implementation guidance with constitutional awareness."""
        # Analyze current context for constitutional compliance
        current_file = context.get("current_file") if context else None

        guidance = "🚀 **Implementation Guidance** (Constitutional Mode):\n\n"

        # Apply constitutional principles
        guidance += "**Constitutional Principles to Follow:**\n"
        guidance += "1. **Library-First**: Research existing solutions before implementing\n"
        guidance += (
            "2. **Test-First**: Write tests before implementation (TDD)\n"
        )
        guidance += "3. **Simplicity**: Keep functions under 50 lines, complexity under 10\n"
        guidance += (
            "4. **Integration**: Test with real dependencies, not mocks\n"
        )
        guidance += "5. **Clarity**: Write clear, unambiguous code with documentation\n"
        guidance += "6. **Counterfactual**: Document why you chose this approach over alternatives\n\n"

        # Add specific suggestions if we have context about current code
        if current_file:
            suggestions_result = await self.mangle_ability.execute_tool(
                "mangle_get_suggestions", {"code_element": current_file}
            )

            suggestions = suggestions_result.get("suggestions", [])
            if suggestions:
                guidance += "**Context-Specific Suggestions:**\n"
                for suggestion in suggestions:
                    guidance += (
                        f"- {suggestion['message']}: {suggestion['action']}\n"
                    )

        return {
            "response": guidance,
            "insights": [
                "Constitutional principles applied",
                f"Context analysis: {'completed' if current_file else 'no file context'}",
            ],
            "actions": ["implementation_guidance_provided"],
        }

    async def _handle_contextual_response(
        self, user_input: str, enhancement: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle general responses with Mangle context."""
        auto_analysis = enhancement["mangle_context"].get("auto_analysis")

        response_text = "I can help you with that. "

        if auto_analysis:
            if "error" in auto_analysis:
                response_text += "Note: Code knowledge graph analysis is currently unavailable."
            else:
                result_count = auto_analysis.get("result_count", 0)
                if result_count > 0:
                    response_text += f"I notice there are {result_count} related items in your codebase that might be relevant. "

        # Add available Mangle capabilities
        available_patterns = enhancement["mangle_context"].get(
            "available_patterns", []
        )
        if available_patterns:
            response_text += f"\n\nI can also help you analyze your codebase. Try asking: '{available_patterns[0]}'"

        return {
            "response": response_text,
            "insights": [
                "Contextual response with Mangle awareness",
                f"Auto-analysis: {'completed' if auto_analysis and 'error' not in auto_analysis else 'unavailable'}",
            ],
            "actions": ["contextual_response_provided"],
        }


# Global agent instance for GitHub Copilot integration
_mangle_agent = None


def get_mangle_agent(workspace_root: str = ".") -> MangleEnhancedAgent:
    """Get the global Mangle-enhanced agent instance."""
    global _mangle_agent
    if _mangle_agent is None:
        _mangle_agent = MangleEnhancedAgent(workspace_root)
    return _mangle_agent


async def process_copilot_input(
    user_input: str, context: dict[str, Any] | None = None
) -> str:
    """
    Main entry point for GitHub Copilot integration.

    This function automatically enhances all Copilot interactions with
    Mangle reasoning and constitutional awareness.
    """
    agent = get_mangle_agent()
    result = await agent.process_user_input(user_input, context)
    return result["response"]
