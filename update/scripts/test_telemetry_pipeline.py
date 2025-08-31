#!/usr/bin/env python3
"""Quick test of telemetry pipeline"""

import asyncio

from src.plugins.telemetry_pipeline.orchestrator import TelemetryPipelineOrchestrator


async def main():
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
