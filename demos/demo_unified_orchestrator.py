#!/usr/bin/env python3
"""
Demo: Unified Orchestrator & SDD Workflow in Action

This script demonstrates the complete unified orchestration pipeline with
SDD (Spec-Driven Development) workflow, constitutional validation, and
enhanced consensus capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.main import SimpleAbilityRegistry
from src.orchestration.unified_orchestrator import UnifiedOrchestrator, UnifiedRunConfig


class DemoEventLogger:
    """Captures and displays orchestrator events in real-time."""

    def __init__(self):
        self.events = []

    async def emit(self, event):
        """Event bus emit handler."""
        self.events.append(event)
        await self._display_event(event)

    async def _display_event(self, event):
        """Pretty print events as they happen."""
        event_type = event.get("type", "Unknown")

        if event_type == "UnifiedRunStarted":
            print(f"\n🚀 {event_type}")
            print(f"   Run ID: {event.get('run_id')}")
            print(f"   Session: {event.get('session_id')}")
            print(f"   Prompt: {event.get('prompt', '')[:50]}...")

        elif event_type == "UnifiedStageStarted":
            stage = event.get("stage", "unknown")
            print(f"\n🔧 {event_type}")
            print(f"   Stage: {stage}")

        elif event_type == "UnifiedStageSucceeded":
            stage = event.get("stage", "unknown")
            duration = event.get("duration_ms", 0)
            output_summary = event.get("output_summary", {})
            print(f"\n✅ {event_type}")
            print(f"   Stage: {stage}")
            print(f"   Duration: {duration}ms")
            print(f"   Output: {output_summary}")

        elif event_type == "UnifiedStageFailed":
            stage = event.get("stage", "unknown")
            error = event.get("error", "unknown")
            duration = event.get("duration_ms", 0)
            print(f"\n❌ {event_type}")
            print(f"   Stage: {stage}")
            print(f"   Error: {error}")
            print(f"   Duration: {duration}ms")

        elif event_type == "UnifiedRunCompleted":
            print(f"\n🎯 {event_type}")
            print(f"   Success: {event.get('success', False)}")
            stages = event.get("stages", {})
            print(f"   Completed Stages: {list(stages.keys())}")
            aggregate = event.get("aggregate", {})
            if aggregate:
                print("   Aggregate Results:")
                for key, value in aggregate.items():
                    print(f"     {key}: {str(value)[:50]}...")


async def demo_basic_orchestration():
    """Demo 1: Basic orchestration without SDD."""
    print("=" * 60)
    print("DEMO 1: Basic Orchestration Pipeline")
    print("=" * 60)

    # Setup
    event_bus = DemoEventLogger()
    ability_registry = SimpleAbilityRegistry()
    orchestrator = UnifiedOrchestrator(ability_registry, event_bus)

    # Configure for basic planning and consensus
    config = UnifiedRunConfig(
        prompt="Create a simple web scraper for extracting article titles",
        run_id="demo-basic-001",
        session_id="demo-session",
        enable_specification=False,
        enable_planning=True,
        enable_tasks=False,
        enable_consensus=True,
        enable_code_generation=False,
        enable_validation=False,
        enable_scoring=False,
        timeout_s=60,
    )

    print(f"📝 Running basic orchestration for: {config.prompt}")

    # Run the orchestrator
    result = await orchestrator.run(config)

    print("\n📊 Final Result Summary:")
    print(f"   Run ID: {result.get('run_id')}")
    print(f"   Stages Completed: {len(result.get('stages', {}))}")

    return result


async def demo_sdd_workflow():
    """Demo 2: SDD Workflow with Constitutional Validation."""
    print("\n" + "=" * 60)
    print("DEMO 2: SDD Workflow with Constitutional Validation")
    print("=" * 60)

    # Setup with SDD-enhanced registry
    event_bus = DemoEventLogger()
    ability_registry = SimpleAbilityRegistry()

    # Register mock SDD abilities for demo
    async def mock_sdd_specify(args):
        content = args.get("content", "")
        return {
            "specification": f"SPECIFICATION for: {content}",
            "requirements": [
                "Library-first research completed",
                "Test-first development approach",
                "Simple, focused implementation",
                "Clear integration points",
            ],
            "constitutional_score": 0.85,
            "existing_solutions": ["requests", "BeautifulSoup", "scrapy"],
        }

    async def mock_sdd_plan(args):
        spec_content = args.get("spec_content", "")
        return {
            "plan": f"IMPLEMENTATION PLAN for: {spec_content[:50]}...",
            "phases": [
                "1. Library evaluation and selection",
                "2. Test suite design and implementation",
                "3. Core scraper implementation",
                "4. Integration testing and validation",
            ],
            "constitutional_score": 0.80,
            "library_recommendations": ["requests", "beautifulsoup4", "pytest"],
        }

    async def mock_sdd_tasks(args):
        plan_content = args.get("plan_content", "")
        return {
            "tasks": [
                "Install and configure required libraries",
                "Write comprehensive test cases",
                "Implement URL fetching with error handling",
                "Implement HTML parsing logic",
                "Add integration tests",
                "Document usage and examples",
            ],
            "constitutional_score": 0.78,
            "complexity_metrics": {"avg_task_complexity": 3.2, "max_complexity": 5},
        }

    async def mock_sdd_validate(args):
        content = args.get("content", "")
        phase = args.get("phase", "")
        threshold = args.get("constitutional_threshold", 0.75)

        # Mock constitutional validation
        score = 0.82  # Above threshold
        return {
            "constitutional_score": score,
            "validation_passed": score >= threshold,
            "phase": phase,
            "violations": [],
            "recommendations": [
                "Maintain current approach",
                "Add more integration tests",
            ],
        }

    # Register SDD abilities
    ability_registry.register_tool(
        contract={
            "tool_id": "sdd_specify",
            "description": "SDD specification generation with constitutional validation",
            "input_schema": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
            },
            "output_schema": {"type": "object"},
        },
        executor=mock_sdd_specify,
    )

    ability_registry.register_tool(
        contract={
            "tool_id": "sdd_plan",
            "description": "SDD planning with constitutional review",
            "input_schema": {
                "type": "object",
                "properties": {"spec_content": {"type": "string"}},
            },
            "output_schema": {"type": "object"},
        },
        executor=mock_sdd_plan,
    )

    ability_registry.register_tool(
        contract={
            "tool_id": "sdd_tasks",
            "description": "SDD task breakdown with validation",
            "input_schema": {
                "type": "object",
                "properties": {"plan_content": {"type": "string"}},
            },
            "output_schema": {"type": "object"},
        },
        executor=mock_sdd_tasks,
    )

    ability_registry.register_tool(
        contract={
            "tool_id": "sdd_validate",
            "description": "Constitutional compliance validation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "phase": {"type": "string"},
                    "constitutional_threshold": {"type": "number"},
                },
            },
            "output_schema": {"type": "object"},
        },
        executor=mock_sdd_validate,
    )

    orchestrator = UnifiedOrchestrator(ability_registry, event_bus)

    # Configure for full SDD workflow
    config = UnifiedRunConfig(
        prompt="Build a production-ready web scraper following constitutional principles",
        run_id="demo-sdd-001",
        session_id="sdd-session",
        enable_specification=True,
        enable_planning=True,
        enable_tasks=True,
        enable_consensus=True,
        enable_validation=False,
        sdd_mode=True,
        constitutional_threshold=0.75,
        timeout_s=120,
    )

    print(f"📋 Running SDD workflow for: {config.prompt}")
    print(f"🏛️ Constitutional threshold: {config.constitutional_threshold}")

    # Run the SDD orchestrator
    result = await orchestrator.run(config)

    print("\n📊 SDD Workflow Results:")
    print(f"   Run ID: {result.get('run_id')}")
    stages = result.get("stages", {})
    print(f"   Completed Stages: {list(stages.keys())}")

    # Show constitutional compliance scores
    for stage_name, stage_data in stages.items():
        if stage_data.get("status") == "success":
            output = stage_data.get("output", {})
            if "constitutional_score" in output:
                score = output["constitutional_score"]
                status = (
                    "✅ PASS" if score >= config.constitutional_threshold else "❌ FAIL"
                )
                print(f"   {stage_name}: Constitutional Score {score:.2f} {status}")

    return result


async def demo_streaming_orchestration():
    """Demo 3: Streaming orchestration with real-time events."""
    print("\n" + "=" * 60)
    print("DEMO 3: Streaming Orchestration with Real-time Events")
    print("=" * 60)

    # Setup
    event_bus = DemoEventLogger()
    ability_registry = SimpleAbilityRegistry()
    orchestrator = UnifiedOrchestrator(ability_registry, event_bus)

    # Configure for streaming with multiple stages
    config = UnifiedRunConfig(
        prompt="Design and implement a machine learning model for text classification",
        run_id="demo-stream-001",
        session_id="stream-session",
        enable_specification=True,
        enable_planning=True,
        enable_tasks=True,
        enable_consensus=True,
        enable_validation=True,
        sdd_mode=True,
        constitutional_threshold=0.75,
        timeout_s=90,
    )

    print(f"🌊 Streaming orchestration for: {config.prompt}")
    print("📡 Real-time event stream:")

    # Stream events in real-time
    events_captured = []
    async for event in orchestrator.run_stream(config):
        events_captured.append(event)
        # Events are automatically displayed by DemoEventLogger

    print("\n📈 Streaming Summary:")
    print(f"   Total Events: {len(events_captured)}")
    print(f"   Event Types: {set(e.get('type', 'Unknown') for e in events_captured)}")

    return events_captured


async def demo_error_handling():
    """Demo 4: Error handling and graceful degradation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Error Handling and Graceful Degradation")
    print("=" * 60)

    # Setup with intentionally failing ability
    event_bus = DemoEventLogger()
    ability_registry = SimpleAbilityRegistry()

    async def failing_ability(args):
        raise ValueError("Simulated constitutional violation - complexity too high")

    ability_registry.register_tool(
        contract={
            "tool_id": "sdd_specify",
            "description": "Failing SDD specification for demo",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        },
        executor=failing_ability,
    )

    orchestrator = UnifiedOrchestrator(ability_registry, event_bus)

    config = UnifiedRunConfig(
        prompt="Create an overly complex system (this will fail constitutional validation)",
        run_id="demo-error-001",
        session_id="error-session",
        enable_specification=True,
        enable_planning=True,
        enable_tasks=False,
        sdd_mode=True,
        constitutional_threshold=0.75,
        timeout_s=30,
    )

    print(f"💥 Testing error handling for: {config.prompt}")

    # Run with expected failures
    result = await orchestrator.run(config)

    print("\n🛡️ Error Handling Results:")
    stages = result.get("stages", {})
    for stage_name, stage_data in stages.items():
        status = stage_data.get("status", "unknown")
        if status == "failed":
            error = stage_data.get("error", "unknown")
            print(f"   {stage_name}: ❌ Failed - {error}")
        elif status == "success":
            print(f"   {stage_name}: ✅ Succeeded")
        else:
            print(f"   {stage_name}: ⏭️ {status.title()}")

    return result


async def main():
    """Run all orchestrator demos."""
    print("🎬 Super Alita Unified Orchestrator & SDD Workflow Demonstration")
    print("🏛️ Constitutional Framework Integration")
    print("🤖 Enhanced Consensus Decision-Making")
    print()

    try:
        # Demo 1: Basic orchestration
        await demo_basic_orchestration()

        # Demo 2: SDD workflow with constitutional validation
        await demo_sdd_workflow()

        # Demo 3: Streaming orchestration
        await demo_streaming_orchestration()

        # Demo 4: Error handling
        await demo_error_handling()

        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key Features Demonstrated:")
        print("✅ Stage-based pipeline orchestration")
        print("✅ SDD workflow integration (/specify → /plan → /tasks)")
        print("✅ Constitutional validation with scoring")
        print("✅ Real-time event streaming and observability")
        print("✅ Graceful error handling and degradation")
        print("✅ Enhanced consensus decision-making")
        print("✅ Unified configuration and execution")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Check if we're in the right environment
    if not Path("src/orchestration/unified_orchestrator.py").exists():
        print("❌ Please run this demo from the super-alita-clean project root")
        sys.exit(1)

    print("🔧 Initializing demo environment...")
    asyncio.run(main())
