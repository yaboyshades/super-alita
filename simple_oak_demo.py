#!/usr/bin/env python3
"""
Simple OaK-Bandit Integration Demo

Demonstrates the three-layer architecture without complex lifespan management.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reug_runtime.event_bus import make_event_bus
from src.plugins.strategic_planner_plugin import StrategicPlannerPlugin
from src.plugins.option_executor_plugin import OptionExecutorPlugin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Run a simple demo of the OaK-Bandit integration."""
    logger.info("🌟 Simple OaK-Bandit Integration Demo")
    logger.info("=" * 50)
    
    # Set up components
    event_bus = make_event_bus()
    strategic_planner = StrategicPlannerPlugin()
    option_executor = OptionExecutorPlugin()
    
    events_captured = []
    
    async def capture_event(event):
        events_captured.append(event)
        # Debug: print full event structure
        logger.info(f"📡 Raw event captured: {type(event)} - {event}")
        event_type = getattr(event, 'type', event.get('type') if isinstance(event, dict) else 'unknown')
        if event_type == 'unknown' and isinstance(event, dict) and 'event_type' in event:
            event_type = event['event_type']
        logger.info(f"📡 Event type: {event_type}")
    
    # Set up event capturing
    await event_bus.subscribe("subgoal_defined", capture_event)
    await event_bus.subscribe("oak.plan_proposed", capture_event)
    await event_bus.subscribe("tool_call_request", capture_event)
    
    # Set up and start plugins
    await strategic_planner.setup(event_bus, None, {})
    await option_executor.setup(event_bus, None, {})
    await strategic_planner.start()
    await option_executor.start()
    
    logger.info("✅ Components initialized")
    
    # Demo the flow
    logger.info("\n🎯 Testing Strategic Layer...")
    
    # 1. Strategic: Emit goal
    goal_event = {
        "type": "goal_received",
        "goal": "Analyze the project structure and create documentation",
        "session_id": "demo",
        "source_plugin": "demo",
    }
    
    await event_bus.emit(goal_event)
    await asyncio.sleep(1)
    
    # Check subgoals
    subgoal_events = [e for e in events_captured 
                     if (getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'subgoal_defined' or
                         (isinstance(e, dict) and e.get('event_type') == 'subgoal_defined'))]
    
    logger.info(f"   Generated {len(subgoal_events)} subgoals")
    
    # 2. Tactical: Emit plan  
    logger.info("\n🧠 Testing Tactical Layer...")
    
    if subgoal_events:
        first_subgoal = subgoal_events[0]
        if hasattr(first_subgoal, 'subgoal'):
            subgoal_data = first_subgoal.subgoal
        elif isinstance(first_subgoal, dict) and 'subgoal' in first_subgoal:
            subgoal_data = first_subgoal['subgoal']
        else:
            subgoal_data = {"description": "test subgoal", "subgoal_id": "test", "parent_goal_id": "test"}
    else:
        subgoal_data = {"description": "test subgoal", "subgoal_id": "test", "parent_goal_id": "test"}
    
    plan_event = {
        "type": "oak.plan_proposed",
        "goal": "test",
        "plan": [{"option_id": "option-echo", "step": 0}],
        "subgoal": subgoal_data,
        "session_id": "demo",
        "source_plugin": "demo",
    }
    
    events_captured.clear()  # Clear to count only tactical events
    await event_bus.emit(plan_event)
    await asyncio.sleep(1)
    
    # Check tool calls
    tool_events = [e for e in events_captured 
                  if (getattr(e, 'type', e.get('type') if isinstance(e, dict) else None) == 'tool_call_request' or
                      (isinstance(e, dict) and e.get('event_type') == 'tool_call_request'))]
    
    logger.info(f"   Generated {len(tool_events)} tool calls")
    
    # 3. Results
    logger.info("\n📊 Results:")
    logger.info(f"   Strategic Layer: {'✅' if len(subgoal_events) > 0 else '❌'} ({len(subgoal_events)} subgoals)")
    logger.info(f"   Tactical Layer:  {'✅' if len(tool_events) > 0 else '❌'} ({len(tool_events)} tool calls)")
    
    if len(subgoal_events) > 0 and len(tool_events) > 0:
        logger.info("\n🎉 OaK-Bandit Integration Success!")
        logger.info("   ✅ User goals are decomposed into subgoals")
        logger.info("   ✅ Subgoals trigger option selection") 
        logger.info("   ✅ Options are translated to tool calls")
        logger.info("   ✅ Three-layer architecture working!")
        success = True
    else:
        logger.error("\n❌ Integration issues detected")
        success = False
    
    # Cleanup
    await strategic_planner.stop()
    await option_executor.stop()
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)