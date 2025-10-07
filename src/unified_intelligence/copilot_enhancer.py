"""
Copilot Enhancement Engine

Provides seamless integration with GitHub Copilot:
- Enhanced response generation with SDD principles
- Constitutional compliance guidance
- Mangle reasoning integration
- Workflow pattern detection and guidance
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CopilotEnhancer:
    """
    Enhanced Copilot integration with SDD and constitutional principles.

    Provides advisory-only enhancement without blocking Copilot operation.
    """

    def __init__(self):
        """Initialize the Copilot enhancer."""
        self.enhancement_enabled = True
        self.sdd_patterns = self._init_sdd_patterns()
        self.enhancement_history: list[dict[str, Any]] = []

    def _init_sdd_patterns(self) -> dict[str, dict[str, Any]]:
        """Initialize SDD workflow patterns."""
        return {
            "new_feature": {
                "template": """
## Feature Specification Template

**Feature Name**: [Clear, descriptive name]

**Problem Statement**: [What problem does this solve?]

**User Story**: As a [user type], I want to [action] so that [benefit].

**Acceptance Criteria**:
- [ ] [Specific, testable criteria]
- [ ] [Performance requirements]
- [ ] [Error handling requirements]

**Technical Requirements**:
- [ ] Library-first approach (existing solutions researched)
- [ ] Test-first development (TDD)
- [ ] Simple, maintainable design
- [ ] Integration testing strategy

**Dependencies**: [List any external dependencies]

**Risks**: [Potential risks and mitigations]
""",
                "guidance": [
                    "Research existing libraries before implementing",
                    "Write tests before implementation",
                    "Keep design simple and focused",
                    "Define clear acceptance criteria",
                ],
            },
            "generate_plan": {
                "template": """
## Implementation Plan Template

**Objective**: [Clear goal statement]

**Architecture Overview**:
- [ ] System components and interactions
- [ ] Data flow and storage
- [ ] External integrations

**Implementation Phases**:
1. **Research & Library Selection**
   - [ ] Evaluate existing solutions
   - [ ] Select appropriate libraries

2. **Test Design**
   - [ ] Unit test specifications
   - [ ] Integration test plan
   - [ ] End-to-end scenarios

3. **Core Implementation**
   - [ ] Simple, focused components
   - [ ] Clear interfaces and contracts

4. **Integration & Validation**
   - [ ] System integration testing
   - [ ] Performance validation
   - [ ] Documentation completion

**Risks & Mitigations**: [Identified risks and responses]

**Success Criteria**: [How will we know it's complete?]
""",
                "guidance": [
                    "Start with library research and selection",
                    "Design tests before implementing",
                    "Break into simple, testable phases",
                    "Include integration testing from the start",
                ],
            },
        }

    def enhance_response(
        self, user_input: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Enhance Copilot response with SDD principles and guidance.

        Args:
            user_input: User's input/question
            context: Optional context information

        Returns:
            Enhancement data with guidance and recommendations
        """
        if not self.enhancement_enabled:
            return self._disabled_response()

        try:
            enhancement = {
                "enhanced": True,
                "user_input": user_input,
                "sdd_guidance": None,
                "constitutional_notes": [],
                "recommendations": [],
                "template_suggestion": None,
                "workflow_detected": None,
                "enhancement_metadata": {
                    "input_length": len(user_input),
                    "has_context": context is not None,
                    "timestamp": self._get_timestamp(),
                },
            }

            # Detect workflow patterns
            workflow = self._detect_workflow_pattern(user_input)
            if workflow:
                enhancement["workflow_detected"] = workflow
                enhancement["sdd_guidance"] = self._get_sdd_guidance(workflow)
                enhancement["template_suggestion"] = self._get_template_suggestion(
                    workflow
                )

            # Add constitutional guidance
            constitutional_notes = self._get_constitutional_guidance(user_input)
            if constitutional_notes:
                enhancement["constitutional_notes"] = constitutional_notes

            # Generate recommendations
            recommendations = self._generate_recommendations(user_input, workflow)
            if recommendations:
                enhancement["recommendations"] = recommendations

            # Store in history
            self.enhancement_history.append(enhancement)

            return enhancement

        except Exception as e:
            logger.warning(f"Copilot enhancement failed: {e}")
            return self._error_response(str(e))

    def generate_guidance(
        self,
        user_input: str,
        pattern: str | None = None,
        compliance: dict[str, Any] | None = None,
        mangle_insights: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create consolidated guidance for the unified intelligence engine."""

        context = {
            "detected_pattern": pattern,
            "constitutional_compliance": compliance,
            "mangle_insights": mangle_insights,
        }

        enhancement = self.enhance_response(user_input, context=context)

        highlights: list[dict[str, Any]] = []
        if compliance:
            overall_score = compliance.get("overall_score")
            if overall_score is not None:
                highlights.append(
                    {
                        "type": "constitutional_score",
                        "value": overall_score,
                        "status": "ok" if overall_score >= 0.75 else "attention",
                    }
                )

        if mangle_insights:
            highlights.append(
                {
                    "type": "mangle_availability",
                    "value": mangle_insights.get("available"),
                    "detail": mangle_insights.get("analysis_type")
                    or mangle_insights.get("message"),
                }
            )

        if pattern:
            highlights.append(
                {
                    "type": "workflow_pattern",
                    "value": pattern,
                }
            )

        return {
            "enhancement": enhancement,
            "highlights": highlights,
            "pattern": pattern,
        }

    def _detect_workflow_pattern(self, user_input: str) -> str | None:
        """Detect SDD workflow pattern from user input."""
        user_lower = user_input.lower()

        # Check for explicit commands
        if "/new_feature" in user_lower or "new_feature" in user_lower:
            return "new_feature"
        elif "/generate_plan" in user_lower or "generate_plan" in user_lower:
            return "generate_plan"

        # Check for pattern indicators
        feature_indicators = [
            "create feature",
            "new feature",
            "implement feature",
            "build feature",
            "add feature",
        ]
        plan_indicators = [
            "implementation plan",
            "plan implementation",
            "architecture plan",
            "design plan",
            "how to implement",
            "approach for",
        ]

        if any(indicator in user_lower for indicator in feature_indicators):
            return "new_feature"
        elif any(indicator in user_lower for indicator in plan_indicators):
            return "generate_plan"

        return None

    def _get_sdd_guidance(self, workflow: str) -> dict[str, Any] | None:
        """Get SDD guidance for detected workflow."""
        if workflow not in self.sdd_patterns:
            return None

        pattern = self.sdd_patterns[workflow]
        return {
            "workflow": workflow,
            "guidance_points": pattern["guidance"],
            "template_available": True,
        }

    def _get_template_suggestion(self, workflow: str) -> str | None:
        """Get template suggestion for workflow."""
        if workflow in self.sdd_patterns:
            return self.sdd_patterns[workflow]["template"]
        return None

    def _get_constitutional_guidance(self, user_input: str) -> list[str]:
        """Get constitutional principle guidance."""
        guidance = []
        user_lower = user_input.lower()

        # Library-first guidance
        if any(
            word in user_lower for word in ["implement", "create", "build", "develop"]
        ):
            guidance.append(
                "Library-First: Research existing solutions before implementing from scratch"
            )

        # Test-first guidance
        if any(
            word in user_lower
            for word in ["function", "feature", "component", "module"]
        ):
            guidance.append("Test-First: Write tests before implementation (TDD)")

        # Simplicity guidance
        if any(word in user_lower for word in ["complex", "advanced", "sophisticated"]):
            guidance.append(
                "Simplicity Gate: Prefer simple solutions over complex ones"
            )

        # Integration guidance
        if any(
            word in user_lower for word in ["api", "service", "database", "external"]
        ):
            guidance.append(
                "Integration-First: Test system integration before unit details"
            )

        # Clarity guidance
        if any(word in user_lower for word in ["unclear", "ambiguous", "confusing"]):
            guidance.append("Clarity: Ensure specifications are clear and unambiguous")

        return guidance

    def _generate_recommendations(
        self, user_input: str, workflow: str | None
    ) -> list[str]:
        """Generate specific recommendations based on input and workflow."""
        recommendations = []
        user_lower = user_input.lower()

        # Workflow-specific recommendations
        if workflow == "new_feature":
            recommendations.extend(
                [
                    "Start with problem statement and user story",
                    "Define clear acceptance criteria",
                    "Research existing libraries and solutions",
                    "Plan test strategy before implementation",
                ]
            )
        elif workflow == "generate_plan":
            recommendations.extend(
                [
                    "Begin with library research and evaluation",
                    "Design test strategy first",
                    "Break into simple, testable phases",
                    "Include integration testing from start",
                ]
            )

        # Content-based recommendations
        if "test" not in user_lower:
            recommendations.append("Consider test-driven development approach")

        if "library" not in user_lower and "package" not in user_lower:
            recommendations.append("Research existing libraries before implementing")

        if any(word in user_lower for word in ["complex", "advanced", "sophisticated"]):
            recommendations.append("Look for simpler alternatives")

        if "documentation" not in user_lower and any(
            word in user_lower for word in ["feature", "component", "system"]
        ):
            recommendations.append("Include documentation in your plan")

        return recommendations

    def generate_enhanced_prompt(
        self, original_prompt: str, enhancement: dict[str, Any]
    ) -> str:
        """Generate an enhanced prompt with SDD guidance."""
        if not enhancement.get("enhanced"):
            return original_prompt

        enhanced_prompt = original_prompt

        # Add SDD guidance
        if enhancement.get("sdd_guidance"):
            sdd_section = "\n\n## SDD Guidance\n"
            guidance = enhancement["sdd_guidance"]["guidance_points"]
            for point in guidance:
                sdd_section += f"- {point}\n"
            enhanced_prompt += sdd_section

        # Add constitutional notes
        if enhancement.get("constitutional_notes"):
            const_section = "\n\n## Constitutional Principles\n"
            for note in enhancement["constitutional_notes"]:
                const_section += f"- {note}\n"
            enhanced_prompt += const_section

        # Add template if available
        if enhancement.get("template_suggestion"):
            template_section = "\n\n## Suggested Template\n"
            template_section += enhancement["template_suggestion"]
            enhanced_prompt += template_section

        # Add recommendations
        if enhancement.get("recommendations"):
            rec_section = "\n\n## Recommendations\n"
            for rec in enhancement["recommendations"]:
                rec_section += f"- {rec}\n"
            enhanced_prompt += rec_section

        return enhanced_prompt

    def get_enhancement_summary(self) -> dict[str, Any]:
        """Get summary of enhancement activity."""
        total_enhancements = len(self.enhancement_history)

        if total_enhancements == 0:
            return {
                "total_enhancements": 0,
                "enhancement_enabled": self.enhancement_enabled,
                "message": "No enhancements performed yet",
            }

        # Analyze enhancement patterns
        workflows_detected = [
            e.get("workflow_detected")
            for e in self.enhancement_history
            if e.get("workflow_detected")
        ]

        workflow_counts = {}
        for workflow in workflows_detected:
            workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1

        recent_enhancements = self.enhancement_history[-5:]  # Last 5

        return {
            "total_enhancements": total_enhancements,
            "enhancement_enabled": self.enhancement_enabled,
            "workflow_patterns": workflow_counts,
            "recent_activity": len(recent_enhancements),
            "most_common_workflow": (
                max(workflow_counts.items(), key=lambda x: x[1])[0]
                if workflow_counts
                else None
            ),
        }

    def _disabled_response(self) -> dict[str, Any]:
        """Return response when enhancement is disabled."""
        return {
            "enhanced": False,
            "message": "Copilot enhancement is disabled",
            "fallback": "Operating in basic mode",
        }

    def _error_response(self, error: str) -> dict[str, Any]:
        """Return response when enhancement encounters an error."""
        return {
            "enhanced": False,
            "error": error,
            "fallback": "Enhancement failed, using basic mode",
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        try:
            from datetime import datetime

            return datetime.now().isoformat()
        except Exception:
            return "unknown"

    def toggle_enhancement(self, enabled: bool = True) -> dict[str, Any]:
        """Toggle enhancement on/off."""
        self.enhancement_enabled = enabled
        return {
            "enhancement_enabled": self.enhancement_enabled,
            "message": f"Copilot enhancement {'enabled' if enabled else 'disabled'}",
        }

    def clear_history(self) -> dict[str, Any]:
        """Clear enhancement history."""
        cleared_count = len(self.enhancement_history)
        self.enhancement_history.clear()

        return {
            "cleared": True,
            "count": cleared_count,
            "message": f"Cleared {cleared_count} enhancement records",
        }

    def get_status(self) -> dict[str, Any]:
        """Get status of Copilot enhancer."""
        return {
            "enhancement_enabled": self.enhancement_enabled,
            "supported_workflows": list(self.sdd_patterns.keys()),
            "enhancement_history_count": len(self.enhancement_history),
            "capabilities": [
                "workflow_detection",
                "sdd_guidance",
                "constitutional_guidance",
                "template_suggestions",
                "recommendation_generation",
                "prompt_enhancement",
            ],
        }
