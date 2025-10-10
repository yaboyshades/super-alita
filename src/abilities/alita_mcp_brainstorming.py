"""
Alita MCP Brainstorming Ability.

Implements capability gap assessment and MCP tool specification generation
for Super Alita's autonomous tool creation workflow.

This ability analyzes user tasks to determine:
1. Whether existing abilities are sufficient
2. What new MCPs are needed (tool specifications)
3. Constitutional compliance of proposed MCPs
4. Confidence scores for assessment quality

Based on the Alita paper: "ALITA: GENERALIST AGENT ENABLING SCALABLE
AGENTIC REASONING WITH MINIMAL PREDEFINITION AND MAXIMAL SELF-EVOLUTION"
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from src.constitutional.scorer import ConstitutionalScorer
from src.plugins.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


@dataclass
class MCPToolSpecification:
    """Specification for a new MCP tool."""

    name: str
    purpose: str
    suggested_libraries: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    interface_description: str
    estimated_complexity: str  # "simple", "medium", "complex"
    constitutional_alignment: dict[str, float]


@dataclass
class BrainstormingResult:
    """Result from MCP brainstorming capability assessment."""

    needs_new_mcp: bool
    tool_specifications: list[MCPToolSpecification]
    capability_gaps: list[str]
    existing_abilities_sufficient: bool
    assessment_confidence: float  # 0.0 to 1.0
    constitutional_score: float  # Overall constitutional compliance
    recommendations: list[str]
    metadata: dict[str, Any]


class AlitaMCPBrainstormingAbility(PluginInterface):
    """
    MCP Brainstorming: Capability gap assessment for autonomous tool creation.

    This ability implements the first stage of Alita's MCP creation workflow:
    analyzing whether existing abilities can solve the task, or if new MCPs
    are needed. It generates detailed tool specifications with constitutional
    compliance validation.

    Integration with Super Alita:
    - Uses ConstitutionalScorer for Article I-VIII compliance
    - Accesses ability registry to evaluate existing capabilities
    - Emits events for audit trail (mcp_brainstormed)
    - Generates neural atoms for genealogy tracking
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the MCP brainstorming ability.

        Args:
            config: Configuration dictionary with optional keys:
                - model: LLM model for brainstorming (default: claude-3-7-sonnet)
                - temperature: Sampling temperature (default: 0.7)
                - max_tokens: Max response tokens (default: 2000)
                - constitutional_threshold: Min compliance score (default: 0.75)
                - complexity_threshold: Max allowed complexity (default: "complex")
        """
        super().__init__(name="alita_mcp_brainstorming")
        self.config = config or {}
        self.model_name = self.config.get("model", "claude-3-7-sonnet")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.constitutional_threshold = self.config.get(
            "constitutional_threshold", 0.75
        )
        self.complexity_threshold = self.config.get(
            "complexity_threshold", "complex"
        )
        self.constitutional_scorer = ConstitutionalScorer(
            compliance_threshold=self.constitutional_threshold
        )
        self.llm_client = None  # Set during initialization
        self.ability_registry = None  # Set during initialization

    async def initialize(self, _event_bus: Any, **kwargs: Any) -> bool:
        """Initialize the brainstorming ability."""
        self.llm_client = kwargs.get("llm_client")
        self.ability_registry = kwargs.get("ability_registry")
        logger.info(
            "🧠 Alita MCP Brainstorming initialized "
            "(model=%s, threshold=%s)",
            self.model_name,
            self.constitutional_threshold,
        )
        return True

    async def shutdown(self) -> None:
        """Shutdown the brainstorming ability."""
        logger.debug("Shutting down MCP brainstorming ability")

    async def cleanup(self) -> None:
        """Cleanup the brainstorming ability."""
        logger.debug("Cleaning up MCP brainstorming ability")

    async def process_event(
        self, _event: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Process events if needed."""
        # Could handle capability_gap_request events in future
        return None

    async def assess_capability_gap(
        self,
        task_description: str,
        current_capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Assess whether existing abilities are sufficient or new MCPs are needed.

        This is the primary entry point for MCP brainstorming. It analyzes
        the task, reviews existing capabilities, and generates tool
        specifications with constitutional validation.

        Args:
            task_description: Natural language description of the task
            current_capabilities: Dict of currently available abilities/tools

        Returns:
            Dictionary with brainstorming results including:
            - needs_new_mcp: bool
            - tool_specifications: list of MCP specs
            - capability_gaps: list of identified gaps
            - constitutional_score: compliance score
            - assessment_confidence: quality of assessment
        """
        try:
            logger.info(
                f"🧠 Brainstorming MCP requirements for task: {task_description[:100]}..."
            )

            # Get current capabilities from registry
            if current_capabilities is None:
                current_capabilities = self._get_current_capabilities()

            # Generate brainstorming prompt
            prompt = self._build_brainstorming_prompt(
                task_description, current_capabilities
            )

            # Call LLM for capability assessment
            brainstorming_response = await self._call_llm(prompt)

            # Parse and validate the response
            result = self._parse_brainstorming_response(
                brainstorming_response, task_description
            )

            # Constitutional validation
            constitutional_result = self._validate_constitutional_compliance(
                result
            )

            # Enhance result with constitutional scores
            result.constitutional_score = constitutional_result.overall_score
            result.metadata["constitutional_validation"] = {
                "is_compliant": constitutional_result.is_compliant,
                "article_scores": constitutional_result.article_scores,
                "violations": [
                    {
                        "article": v.article,
                        "message": v.message,
                        "severity": v.severity,
                    }
                    for v in constitutional_result.violations
                ],
            }

            # Emit event for audit trail
            await self._emit_brainstorming_event(task_description, result)

            # Convert to dictionary for JSON serialization
            return self._to_dict(result)

        except Exception as e:
            logger.error(f"❌ MCP brainstorming failed: {e}", exc_info=True)
            return {
                "needs_new_mcp": False,
                "tool_specifications": [],
                "capability_gaps": [],
                "existing_abilities_sufficient": True,
                "assessment_confidence": 0.0,
                "constitutional_score": 0.0,
                "recommendations": [
                    f"Brainstorming failed: {str(e)}. "
                    "Falling back to existing abilities."
                ],
                "metadata": {"error": str(e)},
            }

    def _get_current_capabilities(self) -> dict[str, Any]:
        """Extract current capabilities from ability registry.

        Returns:
            Dictionary mapping ability names to descriptions
        """
        capabilities = {}
        if self.ability_registry:
            # Access registered tools from ability registry
            try:
                tools = getattr(
                    self.ability_registry, "list_tools", lambda: []
                )()
                for tool in tools:
                    capabilities[tool.get("tool_id", "unknown")] = {
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("input_schema", {}),
                        "output_schema": tool.get("output_schema", {}),
                    }
            except Exception as e:
                logger.warning(f"⚠️  Could not access ability registry: {e}")

        return capabilities

    def _build_brainstorming_prompt(
        self,
        task_description: str,
        current_capabilities: dict[str, Any],
    ) -> str:
        """Build the LLM prompt for capability assessment.

        Args:
            task_description: User's task description
            current_capabilities: Currently available abilities

        Returns:
            Formatted prompt for LLM
        """
        capabilities_summary = "\n".join(
            [
                f"- {name}: {info.get('description', 'No description')}"
                for name, info in current_capabilities.items()
            ]
        )

        prompt = f"""You are an expert capability assessment agent for Super Alita, an AI orchestration platform.

**Task:** Analyze whether the following task can be solved with existing abilities, or if a new MCP (Model Context Protocol) tool needs to be created.

**User Task:**
{task_description}

**Currently Available Abilities:**
{capabilities_summary if capabilities_summary else "No abilities registered yet."}

**Your Analysis Must Include:**

1. **Capability Gap Assessment:**
   - Can existing abilities solve this task? (yes/no with confidence)
   - What specific capabilities are missing?

2. **MCP Tool Specification (if needed):**
   - Tool name (snake_case, descriptive)
   - Purpose (one sentence)
   - Suggested open-source libraries (Python packages)
   - Input schema (JSON Schema format)
   - Output schema (JSON Schema format)
   - Interface description (method signatures, usage patterns)
   - Estimated complexity (simple/medium/complex)

3. **Constitutional Alignment:**
   - Article I (Library-First): Does it leverage existing libraries?
   - Article II (Test-First): Is the interface testable?
   - Article III (Simplicity): Is the scope appropriately narrow?
   - Article V (Clarity): Is the specification unambiguous?

4. **Recommendations:**
   - Alternative approaches using existing abilities
   - Risks or challenges in implementing the MCP
   - Suggested testing strategy

**Output Format (JSON):**
```json
{{
  "needs_new_mcp": true|false,
  "existing_abilities_sufficient": true|false,
  "assessment_confidence": 0.0-1.0,
  "capability_gaps": ["gap1", "gap2"],
  "tool_specifications": [
    {{
      "name": "tool_name",
      "purpose": "Brief description",
      "suggested_libraries": ["library1", "library2"],
      "input_schema": {{"type": "object", "properties": {{}}}},
      "output_schema": {{"type": "object", "properties": {{}}}},
      "interface_description": "Detailed interface description",
      "estimated_complexity": "simple|medium|complex",
      "constitutional_alignment": {{
        "article_i_compliance": 0.0-1.0,
        "article_ii_compliance": 0.0-1.0,
        "article_iii_compliance": 0.0-1.0,
        "article_v_compliance": 0.0-1.0
      }}
    }}
  ],
  "recommendations": ["recommendation1", "recommendation2"]
}}
```

**Important Guidelines:**
- Prefer reusing existing abilities over creating new MCPs
- Suggest specific, well-maintained open-source libraries (check PyPI)
- Keep MCP scope narrow (Article III: Simplicity)
- Ensure testable interfaces (Article II: Test-First)
- Be honest about confidence and uncertainty

Provide your analysis as valid JSON only (no markdown fences).
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for brainstorming.

        Args:
            prompt: Formatted brainstorming prompt

        Returns:
            LLM response as string
        """
        if self.llm_client:
            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.get("text", "")
            except Exception as e:
                logger.error(f"❌ LLM call failed: {e}")
                raise

        # Fallback: return empty JSON structure
        logger.warning(
            "⚠️  No LLM client available, returning empty brainstorming result"
        )
        return json.dumps(
            {
                "needs_new_mcp": False,
                "existing_abilities_sufficient": True,
                "assessment_confidence": 0.0,
                "capability_gaps": [],
                "tool_specifications": [],
                "recommendations": [
                    "LLM client not configured. "
                    "Cannot perform capability assessment."
                ],
            }
        )

    def _parse_brainstorming_response(
        self, llm_response: str, task_description: str
    ) -> BrainstormingResult:
        """Parse LLM response into structured BrainstormingResult.

        Args:
            llm_response: Raw LLM response
            task_description: Original task description

        Returns:
            Structured BrainstormingResult
        """
        try:
            # Extract JSON from response (handle markdown fences)
            json_str = llm_response.strip()
            if json_str.startswith("```"):
                # Remove markdown code fences
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            data = json.loads(json_str)

            # Parse tool specifications
            tool_specs = []
            for spec_data in data.get("tool_specifications", []):
                tool_spec = MCPToolSpecification(
                    name=spec_data["name"],
                    purpose=spec_data["purpose"],
                    suggested_libraries=spec_data["suggested_libraries"],
                    input_schema=spec_data["input_schema"],
                    output_schema=spec_data["output_schema"],
                    interface_description=spec_data["interface_description"],
                    estimated_complexity=spec_data["estimated_complexity"],
                    constitutional_alignment=spec_data[
                        "constitutional_alignment"
                    ],
                )
                tool_specs.append(tool_spec)

            # Build result
            result = BrainstormingResult(
                needs_new_mcp=data.get("needs_new_mcp", False),
                tool_specifications=tool_specs,
                capability_gaps=data.get("capability_gaps", []),
                existing_abilities_sufficient=data.get(
                    "existing_abilities_sufficient", True
                ),
                assessment_confidence=data.get("assessment_confidence", 0.5),
                constitutional_score=0.0,  # Filled in later
                recommendations=data.get("recommendations", []),
                metadata={
                    "task_description": task_description,
                    "llm_model": self.model_name,
                    "raw_response": llm_response[:500],
                },
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
            # Return fallback result
            return BrainstormingResult(
                needs_new_mcp=False,
                tool_specifications=[],
                capability_gaps=[],
                existing_abilities_sufficient=True,
                assessment_confidence=0.0,
                constitutional_score=0.0,
                recommendations=[
                    f"Failed to parse brainstorming response: {str(e)}"
                ],
                metadata={
                    "error": str(e),
                    "raw_response": llm_response[:500],
                },
            )

    def _validate_constitutional_compliance(
        self, result: BrainstormingResult
    ) -> Any:
        """Validate MCP specifications against constitutional articles.

        Args:
            result: BrainstormingResult with tool specifications

        Returns:
            ConstitutionalResult from scorer
        """
        if not result.tool_specifications:
            # No MCPs to validate
            return self.constitutional_scorer.score_specification(
                "No new MCPs proposed. Using existing abilities."
            )

        # Build specification text for constitutional scoring
        spec_texts = []
        for spec in result.tool_specifications:
            spec_text = f"""
**MCP: {spec.name}**
Purpose: {spec.purpose}
Libraries: {', '.join(spec.suggested_libraries)}
Complexity: {spec.estimated_complexity}
Input: {json.dumps(spec.input_schema, indent=2)}
Output: {json.dumps(spec.output_schema, indent=2)}
Interface: {spec.interface_description}
"""
            spec_texts.append(spec_text)

        combined_spec = "\n\n".join(spec_texts)
        return self.constitutional_scorer.score_specification(combined_spec)

    async def _emit_brainstorming_event(
        self, task_description: str, result: BrainstormingResult
    ) -> None:
        """Emit event for audit trail and neural atom creation.

        Args:
            task_description: Original task description
            result: Brainstorming result
        """
        # TODO: Emit event through event bus when wired
        # event = create_event(
        #     event_type="mcp_brainstormed",
        #     payload={
        #         "task_description": task_description,
        #         "needs_new_mcp": result.needs_new_mcp,
        #         "tool_count": len(result.tool_specifications),
        #         "constitutional_score": result.constitutional_score,
        #         "assessment_confidence": result.assessment_confidence,
        #     },
        # )
        # await self.event_bus.publish(event)
        pass

    def _to_dict(self, result: BrainstormingResult) -> dict[str, Any]:
        """Convert BrainstormingResult to dictionary.

        Args:
            result: BrainstormingResult instance

        Returns:
            Dictionary representation
        """
        return {
            "needs_new_mcp": result.needs_new_mcp,
            "tool_specifications": [
                {
                    "name": spec.name,
                    "purpose": spec.purpose,
                    "suggested_libraries": spec.suggested_libraries,
                    "input_schema": spec.input_schema,
                    "output_schema": spec.output_schema,
                    "interface_description": spec.interface_description,
                    "estimated_complexity": spec.estimated_complexity,
                    "constitutional_alignment": spec.constitutional_alignment,
                }
                for spec in result.tool_specifications
            ],
            "capability_gaps": result.capability_gaps,
            "existing_abilities_sufficient": result.existing_abilities_sufficient,
            "assessment_confidence": result.assessment_confidence,
            "constitutional_score": result.constitutional_score,
            "recommendations": result.recommendations,
            "metadata": result.metadata,
        }
