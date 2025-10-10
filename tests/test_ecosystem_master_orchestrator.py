# tests/test_ecosystem_master_orchestrator.py
from typing import Any

import pytest

# This assumes 'src' is in the Python path, which is a common
# practice configured via pytest.ini or conftest.py.
from src.ecosystem.master_orchestrator import (
    EcosystemOrchestrator,
    GitHubExample,
    ICopilotContextEnhancer,
    IDynamicSnippetGenerator,
    IMetricsCollector,
    IReugEngine,
    ISemanticCodeSearch,
    NoopCopilotEnhancer,
    NoopSemanticSearch,
    SemanticSearchResult,
    TodoAnalysisResult,
)

# --- Fake Implementations for Deterministic Testing ---


class FakeReugEngine(IReugEngine):
    """A fake REUG engine that returns predictable analysis results."""

    async def analyze_todo_complexity(
        self, todo_text: str
    ) -> TodoAnalysisResult:  # noqa: ARG002
        return TodoAnalysisResult(
            complexity_score=0.8,
            confidence=0.9,
            estimated_effort="medium",
            required_context=["database"],
        )


class FakeSemanticSearch(ISemanticCodeSearch):
    """A fake Semantic Search that returns a fixed list of related code."""

    async def find_related_implementations(
        self, query: str, codebase: str  # noqa: ARG002
    ) -> list[SemanticSearchResult]:
        return [
            SemanticSearchResult(
                path="src/db.py",
                code_snippet="def connect_to_db(): ...",
                relevance_score=0.95,
            )
        ]


class FakeCopilotEnhancer(ICopilotContextEnhancer):
    """A fake Copilot Enhancer that returns a fixed GitHub example."""

    async def find_github_examples(
        self, query: str
    ) -> list[GitHubExample]:  # noqa: ARG002
        return [
            GitHubExample(
                repo="test/repo",
                path="main.py",
                code_snippet="lru_cache()",
                license="MIT",
            )
        ]


class FakeSnippetGenerator(IDynamicSnippetGenerator):
    """A fake Snippet Generator that returns a predefined snippet."""

    async def generate_for_todo(
        self,
        todo_text: str,  # noqa: ARG002
        related_code: list[SemanticSearchResult],  # noqa: ARG002
        patterns: list[str],  # noqa: ARG002
    ) -> list[dict[str, str]]:
        return [
            {"prefix": "test_snippet", "body": "implementation goes here;"}
        ]


class FakeMetricsCollector(IMetricsCollector):
    """A fake Metrics Collector that records events for later assertion."""

    def __init__(self):
        self.recorded_events: list[tuple[str, dict[str, Any]]] = []

    async def record_workflow_execution(
        self, workflow_name: str, metadata: dict[str, Any]
    ) -> None:
        self.recorded_events.append((workflow_name, metadata))


# --- Pytest Fixtures and Tests ---


@pytest.fixture
def orchestrator_with_fakes():
    """Provides a fully-wired orchestrator instance with fake dependencies
    for each test."""
    return EcosystemOrchestrator(
        reug_engine=FakeReugEngine(),
        semantic_search=FakeSemanticSearch(),
        copilot_enhancer=FakeCopilotEnhancer(),
        snippet_generator=FakeSnippetGenerator(),
        metrics_collector=FakeMetricsCollector(),
    )


@pytest.mark.asyncio
async def test_orchestrate_todo_workflow_happy_path(
    orchestrator_with_fakes: EcosystemOrchestrator,
):
    """
    Tests the full end-to-end TODO workflow with successful results from all
    mocked subsystems.
    """
    user_id = "test_dev_01"
    action = "todo_detected"
    context = {
        "todo_text": "Implement an LRU cache",
        "file_path": "src/utils/cache.py",
    }

    result = await orchestrator_with_fakes.handle_developer_action(
        user_id, action, context
    )

    # Assertions on the final, unified output
    assert result is not None
    assert result["workflow_type"] == "todo_resolution"
    assert result["estimated_effort"] == "medium"
    assert "Implement the following TODO" in result["copilot_prompt"]
    assert "src/db.py" in result["copilot_prompt"]  # From FakeSemanticSearch
    assert "test/repo" in result["copilot_prompt"]  # From FakeCopilotEnhancer
    assert len(result["vscode_snippets"]) == 1
    assert result["vscode_snippets"][0]["prefix"] == "test_snippet"
    assert result["related_files"] == ["src/db.py"]

    # Assertion on metrics collection
    metrics_collector = orchestrator_with_fakes.metrics_collector
    assert len(metrics_collector.recorded_events) == 1
    event_name, metadata = metrics_collector.recorded_events[0]
    assert event_name == "todo_resolution"
    assert metadata["complexity"] == 0.8
    assert (
        metadata["context_sources"] == 2
    )  # 1 from semantic search, 1 from github
    assert metadata["file_path"] == "src/utils/cache.py"


@pytest.mark.asyncio
async def test_orchestrate_todo_workflow_no_context_found(
    orchestrator_with_fakes: EcosystemOrchestrator,
):
    """
    Tests that the workflow gracefully handles cases where search and
    enhancers find no results.
    """
    # Override fakes to return empty lists, simulating no context found.
    orchestrator_with_fakes.semantic_search = NoopSemanticSearch()
    orchestrator_with_fakes.copilot_enhancer = NoopCopilotEnhancer()

    user_id = "test_dev_02"
    action = "todo_detected"
    context = {"todo_text": "A very obscure and unique TODO"}

    result = await orchestrator_with_fakes.handle_developer_action(
        user_id, action, context
    )

    # Assert that the prompt is still generated but without the context sections.
    assert "RELEVANT INTERNAL CODE EXAMPLES" not in result["copilot_prompt"]
    assert "RELEVANT PUBLIC GITHUB EXAMPLES" not in result["copilot_prompt"]
    assert result["related_files"] == []

    # Check that metrics correctly reflect the lack of context.
    metrics_collector = orchestrator_with_fakes.metrics_collector
    assert metrics_collector.recorded_events[0][1]["context_sources"] == 0


@pytest.mark.asyncio
async def test_handle_unknown_action(
    orchestrator_with_fakes: EcosystemOrchestrator,
):
    """
    Tests that the orchestrator returns a specific error for an unsupported action.
    """
    user_id = "test_dev_03"
    action = "unsupported_action"
    context = {"data": "some_data"}

    result = await orchestrator_with_fakes.handle_developer_action(
        user_id, action, context
    )

    assert result["status"] == "error"
    assert "Unknown action" in result["message"]


@pytest.mark.asyncio
async def test_missing_todo_text():
    """
    Tests that the orchestrator handles missing todo_text gracefully.
    """
    orchestrator = EcosystemOrchestrator()
    user_id = "test_dev_04"
    action = "todo_detected"
    context = {"file_path": "src/utils/cache.py"}  # Missing todo_text

    result = await orchestrator.handle_developer_action(
        user_id, action, context
    )

    assert result["status"] == "error"
    assert "todo_text not provided" in result["message"]


@pytest.mark.asyncio
async def test_developer_context_creation_and_reuse():
    """
    Tests that developer contexts are created and reused correctly.
    """
    orchestrator = EcosystemOrchestrator()
    user_id = "test_dev_05"

    # First call should create context
    context1 = await orchestrator._get_or_create_developer_context(user_id)
    assert context1.user_id == user_id
    assert context1.skill_level == "mid"  # Default value

    # Second call should reuse same context
    context2 = await orchestrator._get_or_create_developer_context(user_id)
    assert context1 is context2  # Same object reference


@pytest.mark.asyncio
async def test_copilot_prompt_synthesis():
    """
    Tests the copilot prompt synthesis with various context scenarios.
    """
    orchestrator = EcosystemOrchestrator()

    # Test with full context
    context_data = {
        "todo_text": "Implement cache system",
        "todo_analysis": TodoAnalysisResult(0.7, 0.8, "large", ["redis"]),
        "related_code": [
            SemanticSearchResult("cache.py", "class Cache:", 0.9)
        ],
        "github_examples": [
            GitHubExample(
                "redis/redis-py", "cache.py", "Redis cache impl", "BSD"
            )
        ],
        "developer_preferences": ["clean_code", "type_hints"],
    }

    prompt = orchestrator._synthesize_copilot_context(context_data)

    assert "Implement cache system" in prompt
    assert "large" in prompt
    assert "RELEVANT INTERNAL CODE EXAMPLES" in prompt
    assert "cache.py" in prompt
    assert "RELEVANT PUBLIC GITHUB EXAMPLES" in prompt
    assert "redis/redis-py" in prompt
    assert "INSTRUCTIONS:" in prompt


@pytest.mark.asyncio
async def test_no_op_implementations():
    """
    Tests that no-op implementations work correctly for standalone operation.
    """
    orchestrator = (
        EcosystemOrchestrator()
    )  # Uses no-op implementations by default

    user_id = "test_dev_06"
    action = "todo_detected"
    context = {"todo_text": "Test TODO", "file_path": "test.py"}

    result = await orchestrator.handle_developer_action(
        user_id, action, context
    )

    # Should work with no-op implementations
    assert result["workflow_type"] == "todo_resolution"
    assert result["estimated_effort"] == "small"  # From NoopReugEngine
    assert result["related_files"] == []  # From NoopSemanticSearch
    assert len(result["vscode_snippets"]) == 1  # From NoopSnippetGenerator
    assert "todo_impl" in result["vscode_snippets"][0]["prefix"]
