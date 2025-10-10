# Example code snippet for integrating telemetry pipeline
# This file demonstrates how to add telemetry processing to an existing
# FastAPI app
# Usage:
#   from fastapi import FastAPI
#   app = FastAPI()
#   setup_telemetry_route(app)

from typing import Any

from src.plugins.telemetry_pipeline.orchestrator import (
    TelemetryPipelineOrchestrator,
)


def setup_telemetry_route(app: Any) -> None:
    """Add telemetry processing route to an existing FastAPI app.

    Args:
        app: FastAPI application instance with ability_registry and
             llm_provider in state
    """

    @app.post("/v1/telemetry/process")
    async def process_telemetry(request: dict[str, Any]) -> dict[str, str]:
        """Process telemetry data through the pipeline"""

        orchestrator = TelemetryPipelineOrchestrator(
            app.state.ability_registry, app.state.llm_provider
        )

        prompt = await orchestrator.process_telemetry(
            task=request.get("task"),
            telemetry_items=request.get("telemetry", []),
            constraints=request.get("constraints", []),
            token_budget=request.get("token_budget", 2000),
        )

        return {"prompt": prompt}
