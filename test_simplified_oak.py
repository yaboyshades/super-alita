#!/usr/bin/env python3
"""
Simplified OaK-Bandit integration test that focuses on the core flow:
Strategic Planning -> Option Selection -> Tool Execution

This test bypasses the complex neural components and focuses on the
three-layer architecture integration.
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
from src.plugins.option_executor_plugin import OptionExecutorPlugin
from src.plugins.strategic_planner_plugin import StrategicPlannerPlugin
from src.reug_runtime.event_bus import make_event_bus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimplifiedOakTest:
    """Simplified test harness for OaK-Bandit integration."""
    
    def __init__(self):
        self.event_bus = make_event_bus()
        self.strategic_planner = StrategicPlannerPlugin()
        self.option_executor = OptionExecutorPlugin()
        self.events_received = []
        
    async def setup(self):
        """Set up core components."""
        logger.info("Setting up simplified OaK integration test...")
        
        # Set up event capturing
        await self.event_bus.subscribe("subgoal_defined", self._capture_event)
        await self.event_bus.subscribe("oak.plan_proposed", self._capture_event)
        await self.event_bus.subscribe("tool_call_request", self._capture_event)
        
        # Set up plugins
        await self.strategic_planner.setup(self.event_bus, None, {})
        await self.option_executor.setup(self.event_bus, None, {})
        
        # Start plugins
        await self.strategic_planner.start()
        await self.option_executor.start()
        
        logger.info("Simplified components set up successfully")
        
    async def _capture_event(self, event):
        """Capture events for analysis."""
        self.events_received.append(event)
        event_type = getattr(event, 'type', event.get('type') if isinstance(event, dict) else 'unknown')
        # Also check event_type field
        if event_type == 'unknown' and isinstance(event, dict) and 'event_type' in event:
            event_type = event['event_type']
        logger.info(f"Captured event: {event_type} - Data: {type(event)}")
        
    async def test_strategic_to_tactical_flow(self):
        """Test the flow from strategic planning to tactical execution."""
        logger.info("Testing Strategic -> Tactical flow...")
        
        # 1. Emit a goal_received event (what router emits)
        test_goal = "Analyze the codebase and write documentation"
        goal_event = create_event(
            "goal_received",
            goal=test_goal,
            session_id="test_session",
            source_plugin="test_harness",
        )
        
        self.events_received.clear()
        await self.event_bus.emit(goal_event.model_dump())
        
        # Wait for strategic planner to decompose
        await asyncio.sleep(1)
        
        # Check if subgoals were created
        subgoal_events = [e for e in self.events_received 
                         if (getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'subgoal_defined' or
                             (isinstance(e, dict) and e.get('event_type') == 'subgoal_defined'))]
        
        if subgoal_events:
            logger.info(f"✅ Strategic planning successful! Generated {len(subgoal_events)} subgoals")
            
            # 2. Now simulate the tactical layer by manually creating a plan
            first_subgoal = subgoal_events[0]
            if hasattr(first_subgoal, 'subgoal'):
                subgoal_data = first_subgoal.subgoal
            elif isinstance(first_subgoal, dict) and 'subgoal' in first_subgoal:
                subgoal_data = first_subgoal['subgoal']
            else:
                logger.error("Cannot extract subgoal data")
                return False
            
            # Create a mock plan proposed event (simulating what OaK planning would do)
            plan_event = create_event(
                "oak.plan_proposed",
                goal=test_goal,
                plan=[{"option_id": "option-echo", "step": 0}],
                session_id="test_session",
                subgoal=subgoal_data,
                source_plugin="test_harness",
            )
            
            await self.event_bus.emit(plan_event.model_dump())
            await asyncio.sleep(1)
            
            # 3. Check if tool calls were generated
            tool_events = [e for e in self.events_received 
                          if (getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'tool_call_request' or
                              (isinstance(e, dict) and e.get('event_type') == 'tool_call_request'))]
            
            if tool_events:
                logger.info(f"✅ Tactical to operational flow successful! Generated {len(tool_events)} tool calls")
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
                    logger.info(f"   Tool call: {tool_name} with params: {params}")
                return True
            else:
                logger.error("❌ No tool calls generated from option execution")
                return False
        else:
            logger.error("❌ No subgoals generated from strategic planning")
            return False
            
    async def test_bandit_option_mapping(self):
        """Test that the option executor can handle various option types."""
        logger.info("Testing bandit option mapping...")
        
        test_options = [
            "option-echo",
            "option-analyze", 
            "option-brainstorm",
            "option-unknown-fallback"
        ]
        
        success_count = 0
        for option_id in test_options:
            logger.info(f"Testing option: {option_id}")
            
            plan_event = create_event(
                "oak.plan_proposed",
                goal="Test goal",
                plan=[{"option_id": option_id, "step": 0}],
                session_id="test_session",
                subgoal={
                    "description": f"Test {option_id}",
                    "subgoal_id": f"test_{option_id}",
                    "parent_goal_id": "test_goal"
                },
                source_plugin="test_harness",
            )
            
            self.events_received.clear()
            await self.event_bus.emit(plan_event.model_dump())
            await asyncio.sleep(0.5)
            
            tool_events = [e for e in self.events_received 
                          if (getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'tool_call_request' or
                              (isinstance(e, dict) and e.get('event_type') == 'tool_call_request'))]
            
            if tool_events:
                success_count += 1
                logger.info(f"   ✅ {option_id} -> tool call successful")
            else:
                logger.info(f"   ❌ {option_id} -> no tool call generated")
                
        success_rate = success_count / len(test_options)
        logger.info(f"Option mapping success rate: {success_count}/{len(test_options)} ({success_rate:.1%})")
        
        return success_rate >= 0.75  # Allow some options to fail for unknown mappings
        
    async def test_parameter_substitution(self):
        """Test that parameters are properly substituted from subgoal context."""
        logger.info("Testing parameter substitution...")
        
        test_subgoal = {
            "description": "Analyze main.py file",
            "subgoal_id": "subgoal_001",
            "parent_goal_id": "goal_001"
        }
        
        plan_event = create_event(
            "oak.plan_proposed",
            goal="Test parameter substitution",
            plan=[{"option_id": "option-echo", "step": 0}],
            session_id="test_session",
            subgoal=test_subgoal,
            source_plugin="test_harness",
        )
        
        self.events_received.clear()
        await self.event_bus.emit(plan_event.model_dump())
        await asyncio.sleep(0.5)
        
        tool_events = [e for e in self.events_received 
                      if (getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'tool_call_request' or
                          (isinstance(e, dict) and e.get('event_type') == 'tool_call_request'))]
        
        if tool_events:
            event = tool_events[0]
            if hasattr(event, 'parameters'):
                params = event.parameters
            elif isinstance(event, dict):
                params = event.get('parameters', {})
            else:
                params = {}
                
            # Check if the subgoal description was substituted
            if 'payload' in params and 'Analyze main.py file' in str(params['payload']):
                logger.info("✅ Parameter substitution successful")
                logger.info(f"   Substituted parameters: {params}")
                return True
            else:
                logger.error(f"❌ Parameter substitution failed. Got: {params}")
                return False
        else:
            logger.error("❌ No tool call generated for parameter substitution test")
            return False
            
    async def run_all_tests(self):
        """Run all simplified integration tests."""
        logger.info("🚀 Starting Simplified OaK-Bandit Integration Tests")
        logger.info("=" * 60)
        
        try:
            await self.setup()
            
            tests = [
                ("Strategic -> Tactical Flow", self.test_strategic_to_tactical_flow),
                ("Bandit Option Mapping", self.test_bandit_option_mapping),
                ("Parameter Substitution", self.test_parameter_substitution),
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
            logger.info("\n" + "=" * 60)
            logger.info("📊 TEST RESULTS SUMMARY")
            logger.info("=" * 60)
            
            passed = sum(1 for _, result in results if result)
            total = len(results)
            
            for test_name, result in results:
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"  {status}: {test_name}")
                
            logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("🎉 ALL TESTS PASSED - Core OaK-Bandit integration is working!")
                return True
            else:
                logger.error("💥 SOME TESTS FAILED - Integration needs debugging")
                return False
                
        finally:
            # Cleanup
            await self.strategic_planner.stop()
            await self.option_executor.stop()


async def main():
    """Main test runner."""
    test_harness = SimplifiedOakTest()
    success = await test_harness.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())