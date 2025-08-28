#!/usr/bin/env python3
"""
Test script to validate the OaK-Bandit integration in Super Alita.

This script tests the end-to-end flow:
1. User input -> Strategic Planner -> Subgoal decomposition
2. Subgoals -> OaK Coordinator -> Option selection (using bandits)
3. Selected options -> Option Executor -> Tool calls
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.events import create_event
from src.plugins.oak_core.coordinator import OakCoordinator
from src.plugins.option_executor_plugin import OptionExecutorPlugin
from src.plugins.strategic_planner_plugin import StrategicPlannerPlugin
from src.reug_runtime.event_bus import make_event_bus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OakIntegrationTest:
    """Test harness for OaK-Bandit integration."""
    
    def __init__(self):
        self.event_bus = make_event_bus()
        self.strategic_planner = StrategicPlannerPlugin()
        self.oak_coordinator = OakCoordinator()
        self.option_executor = OptionExecutorPlugin()
        self.events_received = []
        
    async def setup(self):
        """Set up all components."""
        logger.info("Setting up OaK integration test components...")
        
        # Set up event capturing
        await self.event_bus.subscribe("subgoal_defined", self._capture_event)
        await self.event_bus.subscribe("oak.plan_proposed", self._capture_event)
        await self.event_bus.subscribe("tool_call_request", self._capture_event)
        
        # Set up plugins
        await self.strategic_planner.setup(self.event_bus, None, {})
        await self.oak_coordinator.setup(self.event_bus, None, {})
        await self.option_executor.setup(self.event_bus, None, {})
        
        # Start plugins
        await self.strategic_planner.start()
        await self.oak_coordinator.start()
        await self.option_executor.start()
        
        logger.info("All components set up successfully")
        
    async def _capture_event(self, event):
        """Capture events for analysis."""
        self.events_received.append(event)
        logger.info(f"Captured event: {type(event).__name__ if hasattr(event, '__class__') else type(event)} - {getattr(event, 'type', 'unknown')}")
        
    async def test_goal_decomposition(self):
        """Test that goals are properly decomposed into subgoals."""
        logger.info("Testing goal decomposition...")
        
        test_goal = "Analyze the codebase and create a comprehensive report"
        goal_event = create_event(
            "goal_received",
            goal=test_goal,
            session_id="test_session",
        )
        
        self.events_received.clear()
        await self.event_bus.emit(goal_event.model_dump())
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Check if subgoal_defined events were emitted
        subgoal_events = [e for e in self.events_received if getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'subgoal_defined']
        
        if subgoal_events:
            logger.info(f"✅ Goal decomposition successful! Generated {len(subgoal_events)} subgoals")
            for i, event in enumerate(subgoal_events):
                if hasattr(event, 'subgoal'):
                    desc = event.subgoal.description if hasattr(event.subgoal, 'description') else str(event.subgoal)
                elif isinstance(event, dict) and 'subgoal' in event:
                    desc = event['subgoal'].get('description', str(event['subgoal']))
                else:
                    desc = "Unknown subgoal format"
                logger.info(f"  Subgoal {i+1}: {desc}")
            return True
        else:
            logger.error("❌ No subgoal_defined events received")
            return False
            
    async def test_option_selection(self):
        """Test that subgoals trigger option selection."""
        logger.info("Testing option selection...")
        
        # Manually emit a subgoal event
        subgoal_event = create_event(
            "subgoal_defined",
            subgoal={
                "description": "Analyze code structure", 
                "subgoal_id": "test_subgoal_1",
                "parent_goal_id": "test_goal"
            },
            session_id="test_session",
        )
        
        self.events_received.clear()
        await self.event_bus.emit(subgoal_event.model_dump())
        
        # Wait for OaK processing
        await asyncio.sleep(2)
        
        # Check if oak.plan_proposed events were emitted
        plan_events = [e for e in self.events_received if getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'oak.plan_proposed']
        
        if plan_events:
            logger.info(f"✅ Option selection successful! Generated {len(plan_events)} plans")
            for event in plan_events:
                if hasattr(event, 'plan'):
                    plan = event.plan
                elif isinstance(event, dict) and 'plan' in event:
                    plan = event['plan']
                else:
                    plan = "Unknown plan format"
                logger.info(f"  Plan: {plan}")
            return True
        else:
            logger.error("❌ No oak.plan_proposed events received")
            return False
            
    async def test_tool_execution_bridge(self):
        """Test that selected options are translated to tool calls."""
        logger.info("Testing tool execution bridge...")
        
        # Manually emit a plan proposed event
        plan_event = create_event(
            "oak.plan_proposed",
            goal="Test goal",
            plan=[{"option_id": "option-echo", "step": 0}],
            session_id="test_session",
            subgoal={
                "description": "Execute test action",
                "subgoal_id": "test_subgoal",
                "parent_goal_id": "test_goal"
            }
        )
        
        self.events_received.clear()
        await self.event_bus.emit(plan_event.model_dump())
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Check if tool_call_request events were emitted
        tool_events = [e for e in self.events_received if getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'tool_call_request']
        
        if tool_events:
            logger.info(f"✅ Tool execution bridge successful! Generated {len(tool_events)} tool calls")
            for event in tool_events:
                if hasattr(event, 'tool_name'):
                    tool_name = event.tool_name
                    params = getattr(event, 'parameters', {})
                elif isinstance(event, dict):
                    tool_name = event.get('tool_name', 'unknown')
                    params = event.get('parameters', {})
                else:
                    tool_name = "unknown"
                    params = {}
                logger.info(f"  Tool call: {tool_name} with params: {params}")
            return True
        else:
            logger.error("❌ No tool_call_request events received")
            return False
            
    async def test_full_integration(self):
        """Test the complete end-to-end flow."""
        logger.info("Testing full OaK-Bandit integration...")
        
        test_goal = "Help me understand the project structure and write some code"
        
        # Simulate TaskStarted event (what the router emits)
        task_event = create_event(
            "TaskStarted",
            goal=test_goal,
            correlation_id="test_session-123",
        )
        
        self.events_received.clear()
        await self.event_bus.emit(task_event.model_dump())
        
        # Wait for full processing chain
        await asyncio.sleep(3)
        
        # Analyze the event flow
        event_types = [getattr(e, 'type', e.get('type') if isinstance(e, dict) else 'unknown') for e in self.events_received]
        
        logger.info(f"Event flow: {' -> '.join(event_types)}")
        
        # Check for expected events in the chain
        has_subgoals = any('subgoal_defined' in et for et in event_types)
        has_plans = any('oak.plan_proposed' in et for et in event_types)
        has_tools = any('tool_call_request' in et for et in event_types)
        
        success = has_subgoals and has_plans and has_tools
        
        if success:
            logger.info("✅ Full integration test successful!")
            logger.info(f"   - Subgoal decomposition: {'✅' if has_subgoals else '❌'}")
            logger.info(f"   - Option selection: {'✅' if has_plans else '❌'}")
            logger.info(f"   - Tool execution: {'✅' if has_tools else '❌'}")
        else:
            logger.error("❌ Full integration test failed")
            logger.error(f"   - Subgoal decomposition: {'✅' if has_subgoals else '❌'}")
            logger.error(f"   - Option selection: {'✅' if has_plans else '❌'}")
            logger.error(f"   - Tool execution: {'✅' if has_tools else '❌'}")
            
        return success
            
    async def run_all_tests(self):
        """Run all integration tests."""
        logger.info("🚀 Starting OaK-Bandit Integration Tests")
        logger.info("=" * 50)
        
        try:
            await self.setup()
            
            tests = [
                ("Goal Decomposition", self.test_goal_decomposition),
                ("Option Selection", self.test_option_selection),
                ("Tool Execution Bridge", self.test_tool_execution_bridge),
                ("Full Integration", self.test_full_integration),
            ]
            
            results = []
            for test_name, test_func in tests:
                logger.info(f"\n🧪 Running {test_name} Test...")
                try:
                    result = await test_func()
                    results.append((test_name, result))
                except Exception as e:
                    logger.error(f"❌ {test_name} test failed with exception: {e}")
                    results.append((test_name, False))
            
            # Summary
            logger.info("\n" + "=" * 50)
            logger.info("📊 TEST RESULTS SUMMARY")
            logger.info("=" * 50)
            
            passed = sum(1 for _, result in results if result)
            total = len(results)
            
            for test_name, result in results:
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"  {status}: {test_name}")
                
            logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("🎉 ALL TESTS PASSED - OaK-Bandit integration is working!")
                return True
            else:
                logger.error("💥 SOME TESTS FAILED - Integration needs debugging")
                return False
                
        finally:
            # Cleanup
            await self.strategic_planner.stop()
            await self.oak_coordinator.stop()
            await self.option_executor.stop()


async def main():
    """Main test runner."""
    test_harness = OakIntegrationTest()
    success = await test_harness.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())