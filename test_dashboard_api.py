"""
Simple test for telemetry dashboard API endpoints
"""

import asyncio
import json
from pathlib import Path

from src.core.telemetry.collector import TelemetryCollector
from src.core.telemetry.dashboard import TelemetryDashboard
from src.core.telemetry.simple_event_bus import SimpleEventBus
from src.core.cortex import create_cortex_runtime


async def test_dashboard_api():
    """Test the telemetry dashboard API endpoints"""
    print("Testing Telemetry Dashboard API")
    
    # Create telemetry collector with some test data
    output_file = Path("test_telemetry.jsonl")
    collector = TelemetryCollector(output_file)
    
    # Create Cortex runtime and generate some test data
    runtime = create_cortex_runtime()
    event_bus = SimpleEventBus()
    await event_bus.subscribe("*", collector.collect_event)
    await runtime.setup(event_bus=event_bus)
    
    # Generate test telemetry data
    print("Generating test telemetry data...")
    for i in range(3):
        context = runtime.create_context(f"test_session_{i}", "test_user")
        result = await runtime.process_cycle(f"Test query {i}", context)
        await asyncio.sleep(0.01)  # Small delay
    
    await runtime.shutdown()
    
    # Create dashboard
    dashboard = TelemetryDashboard(collector, host="localhost", port=8002)
    app = dashboard.get_app()
    
    print("Testing API endpoints...")
    
    # Test the FastAPI app manually
    from fastapi.testclient import TestClient
    
    with TestClient(app) as client:
        # Test metrics endpoint
        print("  Testing /api/metrics...")
        response = client.get("/api/metrics")
        assert response.status_code == 200
        metrics = response.json()
        print(f"    Metrics: {metrics}")
        assert metrics["total_events"] >= 3
        
        # Test events endpoint
        print("  Testing /api/events...")
        response = client.get("/api/events")
        assert response.status_code == 200
        events = response.json()
        print(f"    Events count: {len(events)}")
        assert len(events) >= 3
        
        # Test health endpoint
        print("  Testing /api/health...")
        response = client.get("/api/health")
        assert response.status_code == 200
        health = response.json()
        print(f"    Health: {health}")
        assert health["status"] == "healthy"
        
        # Test dashboard HTML
        print("  Testing dashboard HTML...")
        response = client.get("/")
        assert response.status_code == 200
        assert "Super Alita Telemetry Dashboard" in response.text
    
    # Clean up
    if output_file.exists():
        output_file.unlink()
    
    print("All API tests passed!")


if __name__ == "__main__":
    asyncio.run(test_dashboard_api())