# src/ecosystem/master_orchestrator.py
"""
The brain that coordinates all subsystems.
This is the first production-minded implementation, focusing on the TODO
workflow with injectable dependencies and no-op fallbacks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

# Added imports for event bus and telemetry
from .eventbus import IEventBus, NoopEventBus
from .telemetry import Telemetry

# --- Enums and Data Classes ---


class WorkflowType(Enum):
    """Defines the types of development workflows the orchestrator can handle."""

    TODO_RESOLUTION = "todo_resolution"
    CODE_REVIEW = "code_review"
    FEATURE_DEVELOPMENT = "feature_development"
    PASTED_CODE_INTEGRATION = "pasted_code_integration"
    # Future workflows can be added here


@dataclass
class DeveloperContext:
    """Represents the complete developer state and preferences."""

    user_id: str
    current_task: str | None = None
    skill_level: str = "mid"  # Can be 'junior', 'mid', 'senior', 'architect'
    preferred_patterns: list[str] = field(default_factory=list)
    active_codebase: str = "default_project"
    recent_files: list[str] = field(default_factory=list)


# --- Analysis & Search Result Data Classes ---


@dataclass
class TodoAnalysisResult:
    """Data class for the result of a TODO complexity analysis."""

    complexity_score: float
    confidence: float
    estimated_effort: str  # e.g., "small", "medium", "large"
    required_context: list[str]


@dataclass
class SemanticSearchResult:
    """Represents a single result from a semantic code search."""

    path: str
    code_snippet: str
    relevance_score: float


@dataclass
class GitHubExample:
    """Represents a single code example fetched from GitHub."""

    repo: str
    path: str
    code_snippet: str
    license: str | None = None


# --- Protocols for Dependency Injection ---


class IReugEngine(Protocol):
    """Interface for the REUG Cognitive Engine."""

    async def analyze_todo_complexity(self, todo_text: str) -> TodoAnalysisResult: ...


class ISemanticCodeSearch(Protocol):
    """Interface for the Semantic Code Search module."""

    async def find_related_implementations(
        self, query: str, codebase: str
    ) -> list[SemanticSearchResult]: ...


class ICopilotContextEnhancer(Protocol):
    """Interface for enhancing Copilot context, e.g., by finding GitHub examples."""

    async def find_github_examples(self, query: str) -> list[GitHubExample]: ...


class IDynamicSnippetGenerator(Protocol):
    """Interface for generating dynamic, context-aware code snippets."""

    async def generate_for_todo(
        self,
        todo_text: str,
        related_code: list[SemanticSearchResult],
        patterns: list[str],
    ) -> list[dict[str, str]]: ...


class IMetricsCollector(Protocol):
    """Interface for a metrics collection system."""

    async def record_workflow_execution(
        self, workflow_name: str, metadata: dict[str, Any]
    ) -> None: ...


# Add these new protocols alongside the existing ones (IReugEngine, etc.)
class INamingConventionEnforcer(Protocol):
    """Interface for a naming convention validation module."""

    async def validate_code_block(
        self, code_block: str, file_path: str
    ) -> list[dict[str, Any]]: ...


class IPatternAnalyzer(Protocol):
    """Interface for comparing code against project-specific patterns."""

    async def compare_against_codebase(self, code_block: str) -> dict[str, Any]: ...


# --- No-Op Implementations for Standalone Operation ---


class NoopReugEngine(IReugEngine):
    """A default, no-operation REUG engine for standalone testing."""

    async def analyze_todo_complexity(
        self, todo_text: str  # noqa: ARG002
    ) -> TodoAnalysisResult:
        print("[Orchestrator] No-op REUG Engine: Returning default analysis.")
        return TodoAnalysisResult(
            complexity_score=0.3,
            confidence=0.5,
            estimated_effort="small",
            required_context=[],
        )


class NoopSemanticSearch(ISemanticCodeSearch):
    """A default, no-operation Semantic Search for standalone testing."""

    async def find_related_implementations(  # noqa: ARG002
        self, query: str, codebase: str  # noqa: ARG002
    ) -> list[SemanticSearchResult]:
        print("[Orchestrator] No-op Semantic Search: Returning no results.")
        return []


class NoopCopilotEnhancer(ICopilotContextEnhancer):
    """A default, no-operation Copilot Enhancer for standalone testing."""

    async def find_github_examples(
        self, query: str  # noqa: ARG002
    ) -> list[GitHubExample]:
        print("[Orchestrator] No-op Copilot Enhancer: Returning no examples.")
        return []


class NoopSnippetGenerator(IDynamicSnippetGenerator):
    """A default, no-operation Snippet Generator for standalone testing."""

    async def generate_for_todo(  # noqa: ARG002
        self,
        todo_text: str,
        related_code: list[SemanticSearchResult],  # noqa: ARG002
        patterns: list[str],  # noqa: ARG002
    ) -> list[dict[str, str]]:
        print("[Orchestrator] No-op Snippet Generator: Returning a default snippet.")
        return [
            {
                "prefix": "todo_impl",
                "body": f"// TODO: Implement based on '{todo_text}'\n$0",
            }
        ]


class NoopMetricsCollector(IMetricsCollector):
    """A default, no-operation Metrics Collector for standalone testing."""

    async def record_workflow_execution(  # noqa: ARG002
        self, workflow_name: str, metadata: dict[str, Any]
    ) -> None:
        print(
            f"[Orchestrator] No-op Metrics: Would record {workflow_name} "
            f"with metadata {metadata}"
        )
        pass


class NoopNamingEnforcer(INamingConventionEnforcer):
    async def validate_code_block(
        self, code_block: str, file_path: str
    ) -> list[dict[str, Any]]:
        print("[Orchestrator] No-op Naming Enforcer: Assuming all names are valid.")
        return []  # Return an empty list, indicating no violations


class NoopPatternAnalyzer(IPatternAnalyzer):
    async def compare_against_codebase(self, code_block: str) -> dict[str, Any]:
        print("[Orchestrator] No-op Pattern Analyzer: No patterns compared.")
        return {"compliance_score": 0.7, "suggested_refactors": []}


# --- The Master Orchestrator ---


class EcosystemOrchestrator:
    """Master orchestrator that coordinates all AI development tools."""

    def __init__(
        self,
        reug_engine: IReugEngine | None = None,
        semantic_search: ISemanticCodeSearch | None = None,
        copilot_enhancer: ICopilotContextEnhancer | None = None,
        snippet_generator: IDynamicSnippetGenerator | None = None,
        metrics_collector: IMetricsCollector | None = None,
        # ADDED: event_bus and telemetry dependencies
        event_bus: IEventBus = None,
        telemetry: Telemetry = None,
        # ADD THESE NEW DEPENDENCIES
        naming_enforcer: INamingConventionEnforcer | None = None,
        pattern_analyzer: IPatternAnalyzer | None = None,
    ):
        # Core systems are injected, allowing for easy replacement with real
        # implementations.
        self.reug_engine = reug_engine or NoopReugEngine()
        self.semantic_search = semantic_search or NoopSemanticSearch()
        self.copilot_enhancer = copilot_enhancer or NoopCopilotEnhancer()
        self.snippet_generator = snippet_generator or NoopSnippetGenerator()
        self.metrics_collector = metrics_collector or NoopMetricsCollector()

        # ADDED:
        self.event_bus = event_bus or NoopEventBus()
        self.telemetry = telemetry or Telemetry()

        # MODIFIED: Use the telemetry's counter
        self.metrics_collector = (
            telemetry  # Can be aliased for simplicity or have its own class
        )

        # ADD THESE NEW DEPENDENCIES
        self.naming_enforcer = naming_enforcer or NoopNamingEnforcer()
        self.pattern_analyzer = pattern_analyzer or NoopPatternAnalyzer()

        # State management for developer contexts.
        self.developer_contexts: dict[str, DeveloperContext] = {}

    async def _get_or_create_developer_context(self, user_id: str) -> DeveloperContext:
        """Retrieves an existing context for a developer or creates a new one."""
        if user_id not in self.developer_contexts:
            self.developer_contexts[user_id] = DeveloperContext(user_id=user_id)
        return self.developer_contexts[user_id]

    async def handle_developer_action(
        self, user_id: str, action: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Central entry point for all developer interactions with the ecosystem."""
        dev_context = await self._get_or_create_developer_context(user_id)

        # ADDED: Emit an event when an action is handled
        await self.event_bus.emit(
            "developer.action.received", {"user_id": user_id, "action": action}
        )

        # Simple routing based on the action type. This can be expanded with a more
        # sophisticated classification engine in the future.
        if action == "todo_detected":
            # ADDED: Wrap the workflow in a telemetry span
            with self.telemetry.timer(
                "workflow.todo_resolution.duration_ms", tags={"user_id": user_id}
            ):
                self.telemetry.increment_counter(
                    "workflow_runs.todo_resolution", tags={"user_id": user_id}
                )
                return await self._orchestrate_todo_workflow(dev_context, context)
        elif action == "code_pasted":
            with self.telemetry.timer(
                "workflow.integration.duration_ms", tags={"user_id": user_id}
            ):
                self.telemetry.increment_counter(
                    "workflow_runs.integration", tags={"user_id": user_id}
                )
                return await self._orchestrate_integration_workflow(
                    dev_context, context
                )

        # Default response for unknown actions.
        return {"status": "error", "message": f"Unknown action: '{action}'"}

    async def _orchestrate_todo_workflow(
        self, dev_context: DeveloperContext, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Coordinates the complete TODO resolution workflow across all
        integrated subsystems."""
        todo_text = context.get("todo_text", "")
        if not todo_text:
            return {"status": "error", "message": "todo_text not provided in context"}

        # ADDED: Emit event at the start of the workflow
        await self.event_bus.emit(
            "workflow.todo_resolution.started",
            {"user_id": dev_context.user_id, "file_path": context.get("file_path")},
        )

        # ADDED: Time individual steps of the workflow
        with self.telemetry.timer("todo.analysis.duration_ms"):
            todo_analysis = await self.reug_engine.analyze_todo_complexity(todo_text)

        with self.telemetry.timer("todo.semantic_search.duration_ms"):
            related_code = await self.semantic_search.find_related_implementations(
                todo_text, dev_context.active_codebase
            )

        with self.telemetry.timer("todo.github_search.duration_ms"):
            github_examples = await self.copilot_enhancer.find_github_examples(
                todo_text
            )

        with self.telemetry.timer("todo.snippet_generation.duration_ms"):
            snippets = await self.snippet_generator.generate_for_todo(
                todo_text, related_code, dev_context.preferred_patterns
            )

        # 5. Unified Copilot Prompt Synthesis
        copilot_prompt = self._synthesize_copilot_context(
            {
                "todo_text": todo_text,
                "todo_analysis": todo_analysis,
                "related_code": related_code,
                "github_examples": github_examples,
                "developer_preferences": dev_context.preferred_patterns,
            }
        )

        # REMOVED: The metrics collector is now handled by telemetry
        # await self.metrics_collector.record_workflow_execution(...)

        # ADDED: Emit a final event with the outcome
        await self.event_bus.emit(
            "workflow.todo_resolution.completed",
            {
                "user_id": dev_context.user_id,
                "confidence": todo_analysis.confidence,
                "related_files_found": len(related_code),
                "github_examples_found": len(github_examples),
            },
        )

        # 7. Return a unified, actionable response
        return {
            "workflow_type": "todo_resolution",
            "copilot_prompt": copilot_prompt,
            "vscode_snippets": snippets,
            "confidence": todo_analysis.confidence,
            "estimated_effort": todo_analysis.estimated_effort,
            "related_files": [item.path for item in related_code],
        }

    def _synthesize_copilot_context(self, context_data: dict[str, Any]) -> str:
        """Creates a concise, context-rich prompt engineered for high-quality
        GitHub Copilot responses."""

        todo_text: str = context_data["todo_text"]
        todo_analysis: TodoAnalysisResult = context_data["todo_analysis"]
        related_code: list[SemanticSearchResult] = context_data["related_code"]
        github_examples: list[GitHubExample] = context_data["github_examples"]

        prompt_parts = [
            f'TASK: Implement the following TODO: "{todo_text}"',
            f"This task is estimated to be of "
            f"'{todo_analysis.estimated_effort}' effort.",
        ]

        if related_code:
            prompt_parts.append("\n--- RELEVANT INTERNAL CODE EXAMPLES ---")
            for _i, item in enumerate(
                related_code[:2]
            ):  # Limit to top 2 for prompt brevity
                prompt_parts.append(
                    f"Example from `{item.path}` "
                    f"(relevance: {item.relevance_score:.2f}):"
                )
                prompt_parts.append(f"```python\n{item.code_snippet}\n```")

        if github_examples:
            prompt_parts.append("\n--- RELEVANT PUBLIC GITHUB EXAMPLES ---")
            for _i, item in enumerate(github_examples[:2]):  # Limit to top 2
                prompt_parts.append(
                    f"Example from repository `{item.repo}` in file `{item.path}`:"
                )
                prompt_parts.append(f"```python\n{item.code_snippet}\n```")

        prompt_parts.append(
            "\nINSTRUCTIONS: Based on the provided context and examples, "
            "generate a complete and robust implementation."
        )

        return "\n".join(prompt_parts)

    # Add this new method to the EcosystemOrchestrator class
    async def _orchestrate_integration_workflow(
        self, dev_context: DeveloperContext, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyzes pasted code and generates a plan for integrating it into the codebase."""
        pasted_code = context.get("pasted_code", "")
        file_path = context.get("file_path", "")
        if not pasted_code:
            return {"status": "error", "message": "pasted_code not provided"}

        # Emit event at the start of the workflow
        await self.event_bus.emit(
            "workflow.integration.started",
            {"user_id": dev_context.user_id, "file_path": file_path},
        )

        # 1. Analyze for Naming Convention Violations
        naming_violations = await self.naming_enforcer.validate_code_block(
            pasted_code, file_path
        )

        # 2. Analyze for Architectural Pattern Mismatches
        pattern_analysis = await self.pattern_analyzer.compare_against_codebase(
            pasted_code
        )

        # 3. (From Patch 0002) Find similar internal code for context
        related_internal_code = await self.semantic_search.find_related_implementations(
            pasted_code, dev_context.active_codebase
        )

        # 4. Synthesize an "Integration Plan" with refactoring prompts for Copilot
        integration_prompts = self._synthesize_integration_prompts(
            {
                "naming_violations": naming_violations,
                "pattern_analysis": pattern_analysis,
                "related_code": related_internal_code,
            }
        )

        # 5. Track metrics for this new workflow
        issues_found = len(naming_violations) + len(
            pattern_analysis.get("suggested_refactors", [])
        )

        # Emit a final event with the outcome
        await self.event_bus.emit(
            "workflow.integration.completed",
            {"user_id": dev_context.user_id, "issues_found": issues_found},
        )

        return {
            "workflow_type": "pasted_code_integration",
            "compliance_score": pattern_analysis.get("compliance_score", 0.0),
            "issues_found": issues_found,
            "refactoring_prompts": integration_prompts,  # This is the key output
            "related_files": [item.path for item in related_internal_code],
        }

    def _synthesize_integration_prompts(
        self, analysis_data: dict[str, Any]
    ) -> list[str]:
        """Generates a series of specific, actionable prompts for Copilot to refactor the code."""
        prompts = []

        # Create prompts for naming violations
        for violation in analysis_data.get("naming_violations", []):
            prompts.append(
                f"Refactor the name `{violation['name']}` to `{violation['suggestion']}` to conform to the project's {violation['rule']} naming convention."
            )

        # Create prompts for pattern mismatches
        for refactor in analysis_data.get("pattern_analysis", {}).get(
            "suggested_refactors", []
        ):
            prompts.append(refactor["prompt"])  # The analyzer provides the full prompt

        if not prompts:
            prompts.append(
                "The pasted code appears to be consistent with project standards. Review for logical correctness."
            )

        return prompts
