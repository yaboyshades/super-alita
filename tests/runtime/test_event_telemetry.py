from datetime import datetime
from typing import Any

import pytest

from src.core.events import create_event
from src.plugins.puter_plugin import PuterPlugin
from src.plugins.perplexica_search_plugin import (
    PerplexicaResponse,
    PerplexicaSearchPlugin,
    SearchMode,
)


class CaptureBus:
    """Minimal event bus capturing published events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_plugin_events_include_telemetry_fields(monkeypatch) -> None:
    bus = CaptureBus()

    class DummyClient:
        async def initialize(self) -> None:
            return None

        async def cleanup(self) -> None:  # pragma: no cover - unused
            return None

        async def read_file(self, path: str) -> str:
            return "dummy"

        async def write_file(self, path: str, content: str, create_dirs: bool = True) -> bool:
            return True

        async def delete_file(self, path: str) -> bool:  # pragma: no cover - unused
            return True

        async def list_directory(self, path: str):  # pragma: no cover - unused
            return []

        async def execute_command(
            self, command: str, args: list[str] | None = None, cwd: str | None = None, env: dict[str, str] | None = None
        ) -> dict[str, Any]:  # pragma: no cover - unused
            return {"stdout": "", "stderr": "", "exit_code": 0, "execution_time": 0}

    monkeypatch.setattr(
        "src.plugins.puter_plugin.PuterApiClient", lambda *args, **kwargs: DummyClient()
    )

    puter = PuterPlugin()
    await puter.setup(bus, None, {})
    puter_event = create_event(
        "puter_file_operation",
        conversation_id="session_a",
        source_plugin="client",
    )
    puter_event.metadata = {"operation": "read", "file_path": "/tmp/test.txt"}
    await puter._handle_file_operation(puter_event)
    assert bus.events, "Puter plugin did not publish events"
    puter_out = next(e for e in bus.events if e.event_type == "puter_operation_completed")
    assert puter_out.source_plugin == "puter"
    assert puter_out.conversation_id == "session_a"
    assert puter_out.correlation_id
    assert isinstance(puter_out.timestamp, datetime)

    bus.events.clear()

    perplex = PerplexicaSearchPlugin()
    await perplex.setup(bus, None, {})

    async def fake_search(*args: Any, **kwargs: Any) -> PerplexicaResponse:  # pragma: no cover - simple stub
        return PerplexicaResponse(
            query="hi",
            search_mode=SearchMode.WEB,
            summary="",
            reasoning="",
            sources=[],
            citations=[],
        )

    perplex.search = fake_search
    search_event = create_event(
        "perplexica_search",
        query="hi",
        session_id="session_b",
        conversation_id="session_b",
        source_plugin="client",
    )
    await perplex._handle_search_request(search_event)
    result_evt = next(e for e in bus.events if e.event_type == "perplexica_result")
    assert result_evt.source_plugin == "perplexica_search"
    assert result_evt.correlation_id
    assert isinstance(result_evt.timestamp, datetime)


@pytest.mark.asyncio
async def test_perplexica_offline_emits_error_event() -> None:
    bus = CaptureBus()
    perplex = PerplexicaSearchPlugin()
    await perplex.setup(bus, None, {})
    perplex.web_agent = None
    perplex.web_agent_search_url = "http://localhost:9/search"

    search_event = create_event(
        "perplexica_search",
        query="hi",
        session_id="session_c",
        conversation_id="session_c",
        source_plugin="client",
    )
    await perplex._handle_search_request(search_event)
    error_evt = next(e for e in bus.events if e.event_type == "tool_result")
    assert not error_evt.success
    assert error_evt.source_plugin == "perplexica_search"
    assert "WebAgent unavailable" in error_evt.error
