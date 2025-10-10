#!/usr/bin/env python3
"""Example test of telemetry pipeline.

Note: This is example code. To run, you need to:
1. Create mock registry and llm objects
2. Or integrate with actual application services
"""

import asyncio
from typing import Any

from src.plugins.telemetry_pipeline.orchestrator import (
    TelemetryPipelineOrchestrator,
)


async def main() -> None:
    """Example telemetry pipeline test."""
    # Sample telemetry
    telemetry = [
        {
            "id": "E001",
            "timestamp": "2024-01-01T12:00:00Z",
            "message": "Chat streaming p95 latency spike to 1.8s",
            "session_id": "test_session",
            "latency_ms": 1800,
        }
    ]

    # Example: Create mock objects (replace with actual implementation)
    registry: Any = None  # TODO: Replace with actual AbilityRegistry
    llm: Any = None  # TODO: Replace with actual LLMProvider

    # Process through pipeline
    orchestrator = TelemetryPipelineOrchestrator(registry, llm)

    prompt = await orchestrator.process_telemetry(
        task="Diagnose and fix chat streaming latency",
        telemetry_items=telemetry,
        constraints=["Keep p95 < 500ms"],
        token_budget=1000,
    )

    print("Generated Prompt:")
    print(prompt)


if __name__ == "__main__":
    asyncio.run(main())
