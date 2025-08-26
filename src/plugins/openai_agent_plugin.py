"""OpenAI Agent Plugin bridging conversation events with the OpenAI Agents API."""

import logging
from typing import Any
from uuid import uuid4

from openai_agents import AgentsClient

from ..core.plugin_interface import PluginInterface

logger = logging.getLogger(__name__)


class OpenAIAgentPlugin(PluginInterface):
    """Plugin that forwards conversation messages to an OpenAI Agent."""

    def __init__(self) -> None:
        super().__init__()
        self.client: AgentsClient | None = None
        self._config: dict[str, Any] = {}

    @property
    def name(self) -> str:  # type: ignore[override]
        return "openai_agent"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:  # type: ignore[override]
        await super().setup(event_bus, store, config)
        self._config = config or {}
        api_key = self._config.get("api_key")
        if api_key:
            self.client = AgentsClient(api_key=api_key)
        logger.info("OpenAI Agent plugin configured")

    async def start(self) -> None:  # type: ignore[override]
        await super().start()
        await self.subscribe("conversation_message", self._handle_message)
        logger.info("OpenAI Agent plugin started")

    async def shutdown(self) -> None:  # type: ignore[override]
        self.client = None
        logger.info("OpenAI Agent plugin shutting down")
        await super().shutdown()

    async def _handle_message(self, event: dict[str, Any]) -> None:
        """Forward user message to OpenAI Agent and emit response."""
        if not self.client:
            logger.warning("OpenAI Agent client not configured; echoing message")
            reply = event.get("user_message", "")
        else:
            user_message = event.get("user_message", "")
            try:
                response = await self.client.responses.create(prompt=user_message)
                reply = response.output_text
            except Exception as exc:  # pragma: no cover - network failures
                logger.error("Agent call failed: %s", exc)
                reply = ""

        await self.emit_event(
            "conversation_message",
            session_id=event.get("session_id", "agent"),
            user_message=reply,
            message_id=f"reply_{uuid4().hex}",
        )
