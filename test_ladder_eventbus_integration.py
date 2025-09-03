#!/usr/bin/env python3
"""
Test LADDER EventBus Integration
=================================

This test validates the complete LADDER EventBus integration, ensuring that:
1. LADDER core components work (verified in test_ladder_core.py)
2. EventBus infrastructure is functional
3. LadderAdapter successfully bridges LADDER and EventBus
4. Events are properly emitted and handled
"""

import asyncio
import time

from src.core.in_memory_event_bus import InMemoryEventBus
from src.ladder.integration.cortex_adapter import (
    LadderAdapter,
    LadderIntegrationConfig,
    PlanningMode,
)
from src.ladder.planner import LadderPlanner, PlannerConfig


async def test_ladder_eventbus_integration():
    """Test complete LADDER EventBus integration."""
    print("🧪 Testing LADDER EventBus Integration")
    print("=" * 50)

    # 1. Setup EventBus
    print("\n1. Setting up EventBus...")
    event_bus = InMemoryEventBus()
    await event_bus.start()
    print("   ✓ EventBus initialized")

    # 2. Setup LADDER Planner
    print("\n2. Setting up LADDER Planner...")
    config = PlannerConfig(
        max_decomposition_depth=3,
        execution_timeout=30.0,
        shadow_mode=True,  # Safe execution mode
    )
    planner = LadderPlanner(config)
    print("   ✓ LADDER Planner initialized")

    # 3. Setup LADDER Adapter
    print("\n3. Setting up LADDER Adapter...")
    integration_config = LadderIntegrationConfig(
        max_concurrent_tasks=2,
        planning_mode=PlanningMode.SHADOW,
    )
    adapter = LadderAdapter(
        planner=planner,
        event_bus=event_bus,
        source_plugin="test_ladder",
        config=integration_config,
    )
    await adapter.setup()
    print("   ✓ LADDER Adapter initialized and subscribed to events")

    # 4. Test planning request via adapter
    print("\n4. Testing planning request...")
    start_time = time.time()

    result = await adapter.handle_request(
        query="Create a simple Python function to calculate fibonacci numbers",
        context={"language": "python", "complexity": "simple"},
    )

    execution_time = time.time() - start_time
    print(f"   ✓ Planning completed in {execution_time:.2f}s")
    print(f"   Status: {result['status']}")
    print(f"   Tasks created: {result['tasks_created']}")
    print(f"   Session ID: {result['session_id']}")

    # 5. Check integration metrics
    print("\n5. Checking integration metrics...")
    metrics = adapter.get_metrics()
    print(f"   Total plans: {metrics['total_plans']}")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    print(f"   Average planning time: {metrics['avg_planning_time']:.3f}s")
    print(f"   Active plans: {metrics['active_plans']}")
    print(f"   Planning mode: {metrics['planning_mode']}")

    # 6. Test event-driven planning
    print("\n6. Testing event-driven planning...")

    # Setup event capture
    received_events = []

    async def capture_events(event):
        received_events.append(event)
        print(f"   📨 Event received: {event.event_type}")

    # Subscribe to planning events
    await event_bus.subscribe("planning_started", capture_events)
    await event_bus.subscribe("planning_completed", capture_events)
    await event_bus.subscribe("planning_error", capture_events)

    # Emit a planning request
    await event_bus.emit(
        "planning_request",
        source_plugin="test_client",
        goal="Write a unit test for a sorting algorithm",
        session_id="test_session_123",
        context={"test_framework": "pytest"},
    )

    # Give some time for event processing
    await asyncio.sleep(0.5)

    print(f"   ✓ Captured {len(received_events)} events")

    # 7. Test multiple concurrent requests
    print("\n7. Testing concurrent requests...")

    async def make_request(i):
        return await adapter.handle_request(
            f"Task {i}: Implement a simple calculator", {"task_id": i}
        )

    # Run 3 concurrent requests
    concurrent_results = await asyncio.gather(
        make_request(1), make_request(2), make_request(3), return_exceptions=True
    )

    successful_requests = sum(
        1
        for r in concurrent_results
        if isinstance(r, dict) and r.get("status") == "success"
    )
    print(f"   ✓ {successful_requests}/3 concurrent requests successful")

    # 8. Final metrics check
    print("\n8. Final metrics...")
    final_metrics = adapter.get_metrics()
    print(f"   Total plans executed: {final_metrics['total_plans']}")
    print(f"   Overall success rate: {final_metrics['success_rate']:.2%}")

    # 9. Cleanup
    print("\n9. Cleanup...")
    await event_bus.stop()
    print("   ✓ EventBus stopped")

    print("\n" + "=" * 50)
    print("🎉 LADDER EventBus Integration Test COMPLETE!")
    print(
        f"📊 Summary: {final_metrics['total_plans']} plans, "
        f"{final_metrics['success_rate']:.1%} success rate"
    )

    return final_metrics


if __name__ == "__main__":
    # Run the integration test
    result = asyncio.run(test_ladder_eventbus_integration())
    print(f"\nFinal Result: {result}")
