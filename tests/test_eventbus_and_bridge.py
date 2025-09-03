# tests/test_eventbus_and_bridge.py
from typing import Any

import pytest

# Assumes src is in the path
from src.ecosystem.eventbus import StdoutEventBus
from src.ecosystem.github_bridge import (
    CopilotContextEnhancerFromGitHub,
    GitHubCodeSearchBridge,
)
from src.ecosystem.master_orchestrator import GitHubExample

# --- Test Data ---

FAKE_GITHUB_API_RESPONSE = {
    "total_count": 1,
    "incomplete_results": False,
    "items": [
        {
            "name": "cache.py",
            "path": "my_utils/cache.py",
            "repository": {
                "full_name": "cool-org/cool-repo",
                "license": {"spdx_id": "MIT"},
            },
            "text_matches": [
                {
                    "fragment": (
                        "from functools import lru_cache\n\n"
                        "@lru_cache(maxsize=None)\n"
                        "def get_expensive_data():\n..."
                    )
                }
            ],
        }
    ],
}

# --- Stubs for Testing ---


class StubGitHubBridge(GitHubCodeSearchBridge):
    """A stubbed bridge that returns a predefined API response without
    making network calls."""

    def __init__(self):
        # We don't call super().__init__() so we don't need a token.
        pass

    async def search(
        self, query: str, per_page: int = 3  # noqa: ARG002
    ) -> dict[str, Any]:
        return FAKE_GITHUB_API_RESPONSE


# --- Tests ---


@pytest.mark.asyncio
async def test_stdout_event_bus_emits_json():
    """Tests that the StdoutEventBus correctly formats and stores events."""
    event_bus = StdoutEventBus()
    topic = "test.event"
    payload = {"key": "value", "number": 123}

    await event_bus.emit(topic, payload)

    assert len(event_bus.events) == 1
    emitted_event = event_bus.events[0]

    assert emitted_event["topic"] == topic
    assert emitted_event["payload"] == payload
    assert "timestamp" in emitted_event


@pytest.mark.asyncio
async def test_github_enhancer_normalization():
    """
    Tests that the CopilotContextEnhancerFromGitHub correctly normalizes the
    raw API response from the bridge into structured GitHubExample objects.
    """
    stub_bridge = StubGitHubBridge()
    enhancer = CopilotContextEnhancerFromGitHub(bridge=stub_bridge)

    examples = await enhancer.find_github_examples("some query")

    assert len(examples) == 1
    example = examples[0]

    assert isinstance(example, GitHubExample)
    assert example.repo == "cool-org/cool-repo"
    assert example.path == "my_utils/cache.py"
    assert "lru_cache" in example.code_snippet
    assert example.license == "MIT"


@pytest.mark.asyncio
async def test_github_enhancer_handles_api_failure():
    """
    Tests that the enhancer gracefully returns an empty list if the
    bridge raises an exception (e.g., network error, API rate limit).
    """

    class FailingGitHubBridge(GitHubCodeSearchBridge):
        def __init__(self):
            pass

        async def search(
            self, query: str, per_page: int = 3  # noqa: ARG002
        ) -> dict[str, Any]:
            raise RuntimeError("Simulated API failure")

    enhancer = CopilotContextEnhancerFromGitHub(bridge=FailingGitHubBridge())
    examples = await enhancer.find_github_examples("some query")

    assert examples == []
