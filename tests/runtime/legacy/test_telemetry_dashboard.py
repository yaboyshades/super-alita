"""
Test script for telemetry dashboard
"""
import pytest
pytest.skip("legacy test", allow_module_level=True)

import asyncio
import time
from pathlib import Path

from core.telemetry import TelemetryCollector, TelemetryDashboard
from core.telemetry.simple_event_bus import SimpleEventBus
from core.cortex import create_cortex_runtime

async def test_telemetry_dashboard():
    """Test the telemetry dashboard with Cortex runtime"""
    print("Starting Telemetry Dashboard Test")
    
    # Create telemetry collector
    output_file = Path("test_telemetry.jsonl")
    collector = TelemetryCollector(output_file)
    
    # Create dashboard
    dashboard = TelemetryDashboard(collector, host="localhost", port=8001)
    
    # Create Cortex runtime with event bus
    runtime = create_cortex_runtime()
    event_bus = SimpleEventBus()
    
    # Connect telemetry collector to event bus
    await event_bus.subscribe("*", collector.collect_event)
    
    print("Setting up Cortex runtime with telemetry...")
    await runtime.setup(event_bus=event_bus)
    
    # Start dashboard server in background
    print("Starting telemetry dashboard on http://localhost:8001")
    dashboard_task = asyncio.create_task(dashboard.start_server())
    
    # Give server time to start
    await asyncio.sleep(2)
    
    print("Running test cognitive cycles...")
    
    # Run some test cycles to generate telemetry data
    test_inputs = [
        "Create a Python function to calculate fibonacci numbers",
        "Analyze this code for potential improvements", 
        "Design a REST API for user management",
        "Explain how machine learning works",
        "Write unit tests for the fibonacci function"
    ]
    
    for i, test_input in enumerate(test_inputs):
        print(f"  Cycle {i+1}: {test_input[:50]}...")
        
        context = runtime.create_context(
            session_id=f"test_session_{i}",
            user_id="test_user",
            workspace="/test/workspace"
        )
        
        result = await runtime.process_cycle(test_input, context)
        
        print(f"    Success: {result.success}, Duration: {result.total_duration_ms:.1f}ms")
        
        # Small delay between cycles
        await asyncio.sleep(1)
    
    print("Telemetry data generated!")
    print(f"   Total events: {collector.metrics.total_events}")
    print(f"   Total cycles: {collector.metrics.total_cycles}")
    print(f"   Avg cycle duration: {collector.metrics.avg_cycle_duration_ms:.1f}ms")
    print(f"   Success rate: {collector.metrics.success_rate:.1%}")
    
    print("\nDashboard is running at: http://localhost:8001")
    print("   You can view real-time telemetry data in your browser")
    print("   Press Ctrl+C to stop...")
    
    try:
        # Keep running until interrupted
        await dashboard_task
    except KeyboardInterrupt:
        print("\nStopping telemetry dashboard...")
    finally:
        await runtime.shutdown()
        
        # Clean up test file
        if output_file.exists():
            output_file.unlink()
        
        print("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_telemetry_dashboard())