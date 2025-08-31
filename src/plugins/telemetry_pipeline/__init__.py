"""Telemetry Pipeline Plugin for Super Alita."""

from __future__ import annotations

from typing import Any

from src.core.plugin_interface import PluginInterface
from src.plugins.telemetry_pipeline.orchestrator import (
    TelemetryPipelineOrchestrator,
)


class TelemetryPipelinePlugin(PluginInterface):
    """Plugin for processing telemetry into high-signal prompts."""

    def __init__(self) -> None:
        super().__init__()
        self.orchestrator: TelemetryPipelineOrchestrator | None = None

    @property
    def name(self) -> str:
        """Return the unique name identifier for this plugin."""
        return "telemetry_pipeline"

    @property
    def version(self) -> str:
        """Return the version of this plugin."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Return a description of what this plugin does."""
        return "Process telemetry into high-signal prompts"

    async def setup(self, event_bus: Any, store: Any, config: dict[str, Any]) -> None:
        """Initialize the plugin with required dependencies."""
        await super().setup(event_bus, store, config)
        llm_provider = config.get("llm_provider")
        self.orchestrator = TelemetryPipelineOrchestrator(llm_provider)

    async def start(self) -> None:
        """Start the plugin's main operations."""
        await super().start()
        # Subscribe to telemetry events if needed
        # await self.subscribe("telemetry_data", self._handle_telemetry)

    async def shutdown(self) -> None:
        """Clean up plugin resources."""
        self.orchestrator = None
        await super().shutdown()

    def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools from this plugin."""
        return [
            {
                "name": "process_telemetry",
                "description": "Process telemetry data into high-signal prompts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task context for processing",
                        },
                        "telemetry_items": {
                            "type": "array",
                            "description": "List of telemetry data items",
                        },
                        "constraints": {
                            "type": "array",
                            "description": "List of constraint strings",
                            "items": {"type": "string"},
                        },
                        "token_budget": {
                            "type": "integer",
                            "description": "Maximum tokens for output",
                            "default": 2000,
                        },
                    },
                    "required": ["task", "telemetry_items"],
                },
            }
        ]

    async def process_telemetry(
        self,
        task: str,
        telemetry_items: list[dict[str, Any]],
        constraints: list[str] | None = None,
        token_budget: int = 2000,
    ) -> str:
        """Process telemetry through the full pipeline."""
        if not self.orchestrator:
            msg = "Plugin not initialized"
            raise RuntimeError(msg)

        return await self.orchestrator.process_telemetry(
            task=task,
            telemetry_items=telemetry_items,
            constraints=constraints,
            token_budget=token_budget,
        )


__all__ = ["TelemetryPipelinePlugin"]
