"""
LADDER Multi-Agent Bridge with Cognitive Integration

This module provides the bridge between LADDER planning and multi-agent execution,
enhanced with cognitive modules for quality-driven code generation.

Integration includes:
- MultiModalCodeAnalyzer for code quality assessment
- ReasoningChainVerifier for explainable generation
- Quality-driven revision workflows
- Cognitive fitness scoring
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

# Import cognitive modules
from src.cognitive_modules import (
    MultiModalAnalysisResult,
    MultiModalCodeAnalyzer,
    ReasoningChain,
    ReasoningChainVerifier,
)
from src.core import yaml_utils

logger = structlog.get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("config/reug_integration_config.yaml")


def _deep_merge_dicts(
    base: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class LADDERTask:
    """Task representation from LADDER planner."""

    task_id: str
    description: str
    requirements: str
    complexity: str = "medium"
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeGenerationResult:
    """Result from code generation workflow."""

    code: str
    fitness_score: float
    cognitive_fitness: float
    quality_metrics: dict[str, float]
    reasoning_chain: ReasoningChain | None = None
    revision_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class LADDERMultiAgentBridge:
    """
    Bridge between LADDER planning and multi-agent execution with cognitive enhancements.

    This class orchestrates the workflow:
    1. LADDER planner generates task decomposition
    2. Multi-agent system executes tasks
    3. Cognitive modules assess and improve code quality
    4. Results combined with cognitive fitness scoring
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        enable_cognitive: bool = True,
    ):
        """
        Initialize the LADDER Multi-Agent Bridge.

        Args:
            config: In-memory configuration overrides.
            config_path: Optional path to a YAML configuration file.
            enable_cognitive: Whether to enable cognitive modules.
        """
        resolved_path = self._resolve_config_path(config_path)
        file_config: dict[str, Any] = {}
        if resolved_path is not None:
            file_config = self._load_config_from_path(
                resolved_path, warn_on_missing=config_path is not None
            )

        config_overrides = config or {}
        self.config_path = resolved_path
        self.config = _deep_merge_dicts(file_config, config_overrides)

        self.cognitive_enabled = enable_cognitive and self.config.get(
            "cognitive", {}
        ).get("enabled", True)

        # Initialize cognitive modules
        if self.cognitive_enabled:
            try:
                cognitive_config = self.config.get("cognitive", {})

                # Initialize MultiModalCodeAnalyzer
                self.cognitive_analyzer = MultiModalCodeAnalyzer(
                    config=cognitive_config.get("multimodal_analyzer", {})
                )

                # Initialize ReasoningChainVerifier
                self.reasoning_verifier = ReasoningChainVerifier(
                    config=cognitive_config.get("reasoning_verifier", {})
                )

                logger.info(
                    "Cognitive modules initialized successfully",
                    analyzer=type(self.cognitive_analyzer).__name__,
                    verifier=type(self.reasoning_verifier).__name__,
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize cognitive modules, disabling",
                    exc_info=e,
                )
                self.cognitive_enabled = False
                self.cognitive_analyzer = None
                self.reasoning_verifier = None
        else:
            self.cognitive_analyzer = None
            self.reasoning_verifier = None
            logger.info("Cognitive modules disabled by configuration")

        # Configuration for quality gates
        self.max_revision_attempts = (
            self.config.get("cognitive", {})
            .get("integration", {})
            .get("max_revision_attempts", 3)
        )

        # Configuration for fitness calculation
        self.darwin_weight = (
            self.config.get("cognitive", {})
            .get("integration", {})
            .get("darwin_weight", 0.6)
        )

        # Enable reasoning context
        self.enable_reasoning_context = (
            self.config.get("cognitive", {})
            .get("integration", {})
            .get("enable_reasoning_context", True)
        )

    def _resolve_config_path(
        self,
        config_path: str | Path | None,
    ) -> Path | None:
        """Resolve the configuration path, honoring repository defaults."""
        if config_path is None:
            default_path = DEFAULT_CONFIG_PATH
            return default_path if default_path.exists() else None
        return Path(config_path)

    def _load_config_from_path(
        self,
        path: Path,
        *,
        warn_on_missing: bool,
    ) -> dict[str, Any]:
        """Load configuration from disk with defensive parsing."""
        try:
            contents = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            if warn_on_missing:
                logger.warning(
                    "Cognitive integration config file not found",
                    config_path=str(path),
                )
            return {}
        except OSError as exc:  # pragma: no cover - rare filesystem failure
            logger.warning(
                "Unable to read cognitive integration config",
                config_path=str(path),
                error=str(exc),
            )
            return {}

        try:
            data = yaml_utils.safe_load(contents) or {}
        except Exception as exc:  # pragma: no cover - defensive parsing
            logger.warning(
                "Failed to parse cognitive integration config",
                config_path=str(path),
                exc_info=exc,
            )
            return {}

        if not isinstance(data, dict):
            logger.warning(
                "Cognitive integration config must be a mapping",
                config_path=str(path),
            )
            return {}

        logger.info(
            "Loaded cognitive integration config",
            config_path=str(path),
        )
        return data

    async def execute_integrated_workflow(
        self,
        ladder_task: LADDERTask,
        darwin_fitness: float = 0.5,
    ) -> CodeGenerationResult:
        """
        Execute the integrated workflow with cognitive enhancements.

        Args:
            ladder_task: Task from LADDER planner
            darwin_fitness: Base fitness score from Darwin optimizer

        Returns:
            CodeGenerationResult with code, fitness, and quality metrics
        """
        logger.info(
            "Starting integrated workflow",
            task_id=ladder_task.task_id,
            complexity=ladder_task.complexity,
            cognitive_enabled=self.cognitive_enabled,
        )

        # Generate reasoning context if enabled
        reasoning_context = None
        if self.cognitive_enabled and self.enable_reasoning_context:
            reasoning_context = await self._generate_reasoning_context(
                task_description=ladder_task.description,
                context=ladder_task.context,
            )

        # Generate initial code (placeholder - would call actual multi-agent system)
        generated_code = await self._generate_code_with_multiagent(
            task=ladder_task, reasoning_context=reasoning_context
        )

        # Assess code quality
        analysis = await self._assess_code_quality(
            code=generated_code, requirements=ladder_task.requirements
        )

        # Check quality gates and trigger revision if needed
        revision_count = 0
        if analysis and self._should_trigger_revision(analysis):
            logger.info(
                "Quality below threshold, triggering revision",
                quality_scores=analysis.quality_prediction,
            )

            for attempt in range(self.max_revision_attempts):
                generated_code = await self._trigger_quality_revision(
                    code=generated_code,
                    analysis=analysis,
                    task=ladder_task,
                )
                revision_count += 1

                # Re-assess after revision
                analysis = await self._assess_code_quality(
                    code=generated_code, requirements=ladder_task.requirements
                )

                # Check if quality is now acceptable
                if not self._should_trigger_revision(analysis):
                    logger.info(
                        "Quality improved to acceptable level",
                        revision_count=revision_count,
                    )
                    break

                logger.info(
                    "Quality still below threshold, retrying",
                    attempt=attempt + 1,
                    max_attempts=self.max_revision_attempts,
                )

        # Calculate cognitive fitness
        if analysis:
            cognitive_fitness = self._calculate_cognitive_fitness(
                darwin_fitness=darwin_fitness,
                analysis=analysis,
                chain=reasoning_context,
            )
            quality_metrics = analysis.quality_prediction
        else:
            cognitive_fitness = darwin_fitness
            quality_metrics = {}

        logger.info(
            "Workflow completed",
            task_id=ladder_task.task_id,
            darwin_fitness=darwin_fitness,
            cognitive_fitness=cognitive_fitness,
            revision_count=revision_count,
        )

        return CodeGenerationResult(
            code=generated_code,
            fitness_score=darwin_fitness,
            cognitive_fitness=cognitive_fitness,
            quality_metrics=quality_metrics,
            reasoning_chain=reasoning_context,
            revision_count=revision_count,
            metadata={
                "task_id": ladder_task.task_id,
                "complexity": ladder_task.complexity,
                "cognitive_enabled": self.cognitive_enabled,
            },
        )

    async def _assess_code_quality(
        self,
        code: str,
        requirements: str,
        enable_caching: bool = True,
    ) -> MultiModalAnalysisResult | None:
        """
        Assess generated code quality using multi-modal analysis.

        Args:
            code: Generated code to analyze
            requirements: Original requirements for alignment checking
            enable_caching: Whether to use cached results

        Returns:
            Analysis result or None if cognitive modules disabled/failed
        """
        if not self.cognitive_enabled or self.cognitive_analyzer is None:
            return None

        try:
            analysis = await self.cognitive_analyzer.analyze_code_multimodal(
                code=code, requirements=requirements
            )

            logger.info(
                "Code quality assessment completed",
                understanding_confidence=analysis.understanding_confidence,
                correctness=analysis.quality_prediction.get(
                    "correctness", 0.0
                ),
                maintainability=analysis.quality_prediction.get(
                    "maintainability", 0.0
                ),
            )

            return analysis

        except Exception as e:
            logger.warning(
                "Code quality assessment failed",
                exc_info=e,
                fallback="continuing without cognitive analysis",
            )
            return None

    async def _generate_reasoning_context(
        self,
        task_description: str,
        context: dict[str, Any] | None = None,
    ) -> ReasoningChain | None:
        """
        Generate reasoning chain for agent task context.

        Args:
            task_description: Task description for reasoning
            context: Additional context for reasoning generation

        Returns:
            Reasoning chain or None if disabled/failed
        """
        if not self.cognitive_enabled or self.reasoning_verifier is None:
            return None

        if not self.enable_reasoning_context:
            return None

        try:
            chain = (
                await self.reasoning_verifier.generate_with_reasoning_chain(
                    requirements=task_description, context=context
                )
            )

            logger.info(
                "Reasoning chain generated",
                step_count=len(chain.steps),
                overall_confidence=chain.overall_confidence,
                completeness=chain.completeness_score,
            )

            return chain

        except Exception as e:
            logger.warning(
                "Reasoning chain generation failed",
                exc_info=e,
                fallback="continuing without reasoning context",
            )
            return None

    def _calculate_cognitive_fitness(
        self,
        darwin_fitness: float,
        analysis: MultiModalAnalysisResult | None,
        chain: ReasoningChain | None,
    ) -> float:
        """
        Calculate combined fitness score with cognitive insights.

        Args:
            darwin_fitness: Base fitness from Darwin optimizer
            analysis: Multi-modal analysis result
            chain: Reasoning chain result

        Returns:
            Combined fitness score [0-1]
        """
        if not self.cognitive_enabled or analysis is None:
            return darwin_fitness

        # Extract quality dimensions
        correctness = analysis.quality_prediction.get("correctness", 0.5)
        maintainability = analysis.quality_prediction.get(
            "maintainability", 0.5
        )
        security = analysis.quality_prediction.get("security", 0.5)

        # Calculate cognitive score (weighted average)
        cognitive_score = (
            correctness * 0.4
            + maintainability * 0.3
            + security * 0.2
            + analysis.understanding_confidence * 0.1
        )

        # Add reasoning confidence if available
        if chain is not None:
            reasoning_bonus = chain.overall_confidence * 0.1
            cognitive_score = cognitive_score * 0.9 + reasoning_bonus

        # Combine with Darwin fitness (weighted)
        cognitive_weight = 1.0 - self.darwin_weight

        combined_fitness = (
            darwin_fitness * self.darwin_weight
            + cognitive_score * cognitive_weight
        )

        logger.debug(
            "Cognitive fitness calculated",
            darwin_fitness=darwin_fitness,
            cognitive_score=cognitive_score,
            combined_fitness=combined_fitness,
        )

        return combined_fitness

    async def _trigger_quality_revision(
        self,
        code: str,
        analysis: MultiModalAnalysisResult,
        task: LADDERTask,
    ) -> str:
        """
        Trigger revision workflow when quality is below threshold.

        Args:
            code: Current code to improve
            analysis: Quality analysis result
            task: Original task

        Returns:
            Improved code
        """
        # Build improvement prompt
        improvements = "\n".join(
            [
                f"- {suggestion}"
                for suggestion in analysis.improvement_suggestions[:5]
            ]
        )

        revision_prompt = f"""
Improve the following code based on quality analysis:

Current Quality Scores:
- Correctness: {analysis.quality_prediction.get('correctness', 0.0):.2%}
- Maintainability: {analysis.quality_prediction.get('maintainability', 0.0):.2%}
- Security: {analysis.quality_prediction.get('security', 0.0):.2%}

Improvement Suggestions:
{improvements}

Original Requirements:
{task.requirements}

Current Code:
```python
{code}
```

Please provide improved code that addresses these issues.
"""

        # Use multi-agent system to generate improved version
        # (Placeholder - would call actual orchestrator)
        improved_code = await self._generate_code_with_multiagent(
            task=LADDERTask(
                task_id=f"{task.task_id}-revision",
                description=revision_prompt,
                requirements=task.requirements,
                complexity="high",  # Force thorough review
            ),
            reasoning_context=None,
        )

        return improved_code

    def _should_trigger_revision(
        self, analysis: MultiModalAnalysisResult
    ) -> bool:
        """
        Check if code quality is below threshold and revision should be triggered.

        Args:
            analysis: Multi-modal analysis result

        Returns:
            True if revision should be triggered
        """
        thresholds = (
            self.config.get("cognitive", {})
            .get("multimodal_analyzer", {})
            .get("quality_thresholds", {})
        )

        correctness_min = thresholds.get("correctness_min", 0.7)
        maintainability_min = thresholds.get("maintainability_min", 0.6)
        security_min = thresholds.get("security_min", 0.75)

        quality = analysis.quality_prediction

        # Check critical dimensions
        if quality.get("correctness", 1.0) < correctness_min:
            logger.info(
                "Correctness below threshold", score=quality.get("correctness")
            )
            return True

        if quality.get("security", 1.0) < security_min:
            logger.info(
                "Security below threshold", score=quality.get("security")
            )
            return True

        if quality.get("maintainability", 1.0) < maintainability_min:
            logger.info(
                "Maintainability below threshold",
                score=quality.get("maintainability"),
            )
            return True

        return False

    async def _generate_code_with_multiagent(
        self,
        task: LADDERTask,
        reasoning_context: ReasoningChain | None = None,
    ) -> str:
        """
        Generate code using multi-agent system (placeholder).

        In a real implementation, this would call the actual multi-agent orchestrator.

        Args:
            task: Task to generate code for
            reasoning_context: Optional reasoning chain for context

        Returns:
            Generated code
        """
        # Placeholder implementation
        # In real implementation, would call:
        # result = await self.multi_agent_orchestrator.generate_python_code(
        #     prompt=task.description,
        #     complexity_level=task.complexity,
        #     reasoning_context=reasoning_context
        # )
        # return result.code

        # For now, return a simple placeholder
        return f"""
# Generated code for task: {task.task_id}
# Description: {task.description}

def placeholder_function():
    \"\"\"Placeholder implementation.\"\"\"
    pass
"""
