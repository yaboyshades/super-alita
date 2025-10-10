"""
Unified Intelligence Layer for Super-Alita System

This module provides the central orchestration for integrating:
- SDD (Specification-Driven Development) methodology
- Mangle reasoning capabilities
- Copilot enhancement features
- Constitutional compliance validation
- EOS (E-UPUSF Orchestration Schema) adaptive orchestration

Entry point for all enhancement capabilities.
"""

from .constitutional_engine import ConstitutionalEngine
from .copilot_enhancer import CopilotEnhancer
from .mangle_bridge import MangleBridge
from .unified_middleware import UnifiedMiddleware
from .workflow_detector import WorkflowDetector

# Optional EOS integration
try:
    from src.eos.mangle_integration import EOSMangleConfig, EOSMangleOrchestrator
    EOS_AVAILABLE = True
except ImportError:
    EOSMangleOrchestrator = None
    EOSMangleConfig = None
    EOS_AVAILABLE = False

__all__ = [
    "ConstitutionalEngine",
    "CopilotEnhancer",
    "MangleBridge",
    "UnifiedMiddleware",
    "WorkflowDetector",
    "UnifiedIntelligenceEngine",
    "EOS_AVAILABLE",
]


class UnifiedIntelligenceEngine:
    """
    Central orchestration for SDD + Mangle + Copilot + EOS enhancement.

    Combines:
    - Constitutional validation (9 SDD articles)
    - Workflow pattern detection
    - Mangle reasoning capabilities
    - Template-driven LLM constraints
    - EOS adaptive orchestration (when available)

    Usage:
        engine = UnifiedIntelligenceEngine()
        enhanced = await engine.enhance_interaction("Create user auth feature")
    """

    def __init__(
        self, workspace_root: str | None = None, enable_eos: bool = True
    ) -> None:
        """Initialize the unified intelligence engine."""
        self.constitutional_engine = ConstitutionalEngine()
        self.workflow_detector = WorkflowDetector()
        self.mangle_bridge = MangleBridge(workspace_path=workspace_root)
        self.copilot_enhancer = CopilotEnhancer()
        
        # EOS integration
        self.enable_eos = enable_eos and EOS_AVAILABLE
        self.eos_orchestrator = None
        
        # Track initialization state
        self._initialized = False

    async def initialize(self):
        """Initialize all components asynchronously."""
        if not self._initialized:
            # MangleBridge doesn't have async initialize, just sync
            # _initialize_mangle which is already called in constructor
            self._initialized = True

    async def enhance_interaction(
        self, user_input: str, context: dict | None = None
    ) -> dict:
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
        compliance = self.constitutional_engine.analyze_compliance(user_input)

        # 3. Enhance with Mangle reasoning (if code-related)
        mangle_insights = self.mangle_bridge.generate_code_insights(user_input)

        # 5. EOS adaptive orchestration (if enabled and available)
        eos_insights = {}
        if self.enable_eos and self.eos_orchestrator:
            try:
                eos_insights = await self._analyze_with_eos(
                    user_input, context
                )
            except Exception as e:
                eos_insights = {"error": f"EOS analysis failed: {e}"}
        
        # 6. Generate enhanced guidance
        copilot_enhancement = self.copilot_enhancer.enhance_response(
            user_input
        )
        guidance = copilot_enhancement.get("enhanced_guidance", "No guidance")

        # 7. Generate recommendations
        recommendations = self._generate_recommendations(
            pattern, compliance, mangle_insights, eos_insights
        )

        return {
            "original_input": user_input,
            "detected_pattern": pattern,
            "constitutional_compliance": compliance,
            "mangle_insights": mangle_insights,
            "eos_insights": eos_insights,
            "enhanced_guidance": guidance,
            "recommendations": recommendations,
            "metadata": {
                "engine_version": "2.0.0",
                "processing_time": None,  # TODO: Add timing
                "context": context,
                "eos_enabled": self.enable_eos,
                "eos_available": EOS_AVAILABLE,
            },
        }

    def _generate_recommendations(
        self, 
        pattern: str, 
        compliance: dict, 
        mangle_insights: dict, 
        eos_insights: dict | None = None
    ) -> list:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Constitutional recommendations
        if compliance.get("score", 1.0) < 0.75:
            recommendations.append(
                {
                    "type": "constitutional",
                    "priority": "high",
                    "message": "Constitutional compliance below threshold.",
                    "actions": compliance.get("recommendations", []),
                }
            )

        # Pattern-specific recommendations
        if pattern == "new_feature":
            recommendations.append(
                {
                    "type": "workflow",
                    "priority": "medium",
                    "message": "Use SDD spec template for new features",
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
                    "message": "Validate plan against constitutional gates",
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
                    "actions": mangle_insights.get("recommendations", []),
                }
            )

        # EOS-based recommendations
        if eos_insights and eos_insights.get("orchestration_recommendations"):
            recommendations.append(
                {
                    "type": "eos_orchestration",
                    "priority": "high",
                    "message": "EOS orchestration recommendations available",
                    "actions": eos_insights["orchestration_recommendations"],
                }
            )
        
        return recommendations

    async def _analyze_with_eos(self, user_input: str, context: dict) -> dict:
        """Analyze using EOS orchestration."""
        if not self.eos_orchestrator or not EOS_AVAILABLE:
            return {"available": False, "reason": "EOS not configured"}
        
        try:
            # This is a simplified integration - would need actual EOS spec
            # For now, return basic insights
            return {
                "available": True,
                "analysis_type": "eos_adaptive_orchestration",
                "orchestration_recommendations": [
                    "Use context-adaptive problem solving",
                    "Apply semantic lifting for complex problems",
                    "Leverage expert routing for specialized tasks"
                ],
                "confidence": 0.8
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def validate_constitutional_compliance(self, code: str) -> dict:
        """Quick constitutional compliance check."""
        return self.constitutional_engine.analyze_compliance(code)

    async def ask_code_question(self, question: str) -> dict:
        """Ask a question about the codebase using Mangle reasoning."""
        if not self._initialized:
            await self.initialize()
        return self.mangle_bridge.generate_code_insights(question)

    def get_supported_patterns(self) -> list:
        """Get list of supported SDD workflow patterns."""
        return self.workflow_detector.get_supported_patterns()

    def get_constitutional_articles(self) -> list[dict]:
        """Get the SDD constitutional framework."""
        return self.constitutional_engine.get_all_articles()
