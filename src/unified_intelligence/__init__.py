"""
Unified Intelligence Layer for Super-Alita System

This module provides the central orchestration for integrating:
- SDD (Specification-Driven Development) methodology
- Mangle reasoning capabilities
- Copilot enhancement features
- Constitutional compliance validation

Entry point for all enhancement capabilities.
"""

from .constitutional_engine import ConstitutionalEngine
from .copilot_enhancer import CopilotEnhancer
from .mangle_bridge import MangleBridge
from .unified_middleware import UnifiedMiddleware
from .workflow_detector import WorkflowDetector

__all__ = [
    "ConstitutionalEngine",
    "CopilotEnhancer",
    "MangleBridge",
    "UnifiedMiddleware",
    "WorkflowDetector",
    "UnifiedIntelligenceEngine",
]


class UnifiedIntelligenceEngine:
    """
    Central orchestration for SDD + Mangle + Copilot enhancement.

    Combines:
    - Constitutional validation (9 SDD articles)
    - Workflow pattern detection
    - Mangle reasoning capabilities
    - Template-driven LLM constraints

    Usage:
        engine = UnifiedIntelligenceEngine()
        enhanced = await engine.enhance_interaction("Create a new user auth feature")
    """

    def __init__(self, workspace_root: str = None, mangle_executable: str = "mangle"):
        """Initialize the unified intelligence engine."""
        self.constitutional_engine = ConstitutionalEngine()
        self.workflow_detector = WorkflowDetector()
        self.mangle_bridge = MangleBridge(
            workspace_root=workspace_root, mangle_executable=mangle_executable
        )
        self.copilot_enhancer = CopilotEnhancer()

        # Track initialization state
        self._initialized = False

    async def initialize(self):
        """Initialize all components asynchronously."""
        if not self._initialized:
            await self.mangle_bridge.initialize()
            self._initialized = True

    async def enhance_interaction(self, user_input: str, context: dict = None) -> dict:
        """
        Unified enhancement combining SDD, Mangle, and Copilot capabilities.

        Args:
            user_input: User's input text
            context: Optional context including session_id, file_context, etc.

        Returns:
            Enhanced response with guidance, compliance, and recommendations
        """
        if not self._initialized:
            await self.initialize()

        context = context or {}

        # 1. Detect workflow pattern
        pattern = self.workflow_detector.detect(user_input)

        # 2. Apply constitutional validation
        compliance = await self.constitutional_engine.validate(user_input, pattern)

        # 3. Enhance with Mangle reasoning (if code-related)
        mangle_insights = await self.mangle_bridge.get_insights(user_input, pattern)

        # 4. Generate enhanced guidance
        guidance = self.copilot_enhancer.generate_guidance(
            user_input, pattern, compliance, mangle_insights
        )

        # 5. Generate recommendations
        recommendations = self._generate_recommendations(
            pattern, compliance, mangle_insights
        )

        return {
            "original_input": user_input,
            "detected_pattern": pattern,
            "constitutional_compliance": compliance,
            "mangle_insights": mangle_insights,
            "enhanced_guidance": guidance,
            "recommendations": recommendations,
            "metadata": {
                "engine_version": "1.0.0",
                "processing_time": None,  # TODO: Add timing
                "context": context,
            },
        }

    def _generate_recommendations(
        self, pattern: str, compliance: dict, mangle_insights: dict
    ) -> list:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Constitutional recommendations
        if compliance.get("score", 1.0) < 0.75:
            recommendations.append(
                {
                    "type": "constitutional",
                    "priority": "high",
                    "message": "Constitutional compliance below threshold. Review SDD principles.",
                    "actions": compliance.get("recommendations", []),
                }
            )

        # Pattern-specific recommendations
        if pattern == "new_feature":
            recommendations.append(
                {
                    "type": "workflow",
                    "priority": "medium",
                    "message": "Use SDD specification template for new features",
                    "actions": [
                        "Define WHAT and WHY first",
                        "Add [NEEDS CLARIFICATION] markers",
                    ],
                }
            )
        elif pattern == "generate_plan":
            recommendations.append(
                {
                    "type": "workflow",
                    "priority": "medium",
                    "message": "Validate implementation plan against constitutional gates",
                    "actions": [
                        "Check simplicity constraints",
                        "Verify test-first approach",
                    ],
                }
            )

        # Mangle-based recommendations
        if mangle_insights.get("code_quality_issues"):
            recommendations.append(
                {
                    "type": "code_quality",
                    "priority": "medium",
                    "message": "Code quality issues detected",
                    "actions": mangle_insights.get("quality_recommendations", []),
                }
            )

        return recommendations

    async def validate_constitutional_compliance(self, code_or_spec: str) -> dict:
        """Quick constitutional compliance check."""
        return await self.constitutional_engine.analyze_compliance(code_or_spec)

    async def ask_code_question(self, question: str) -> dict:
        """Ask a question about the codebase using Mangle reasoning."""
        if not self._initialized:
            await self.initialize()
        return await self.mangle_bridge.ask_question(question)

    def get_supported_patterns(self) -> list:
        """Get list of supported SDD workflow patterns."""
        return self.workflow_detector.get_supported_patterns()

    def get_constitutional_articles(self) -> dict:
        """Get the SDD constitutional framework."""
        return self.constitutional_engine.get_articles()
