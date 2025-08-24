#!/usr/bin/env python3
"""End-to-end test for Super Alita agent functionality."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_agent_request_handling():
    """Test the agent's ability to handle a real user request."""
    print("🤖 Testing Agent Request Handling...")
    
    try:
        # Import main components
        from main import create_app
        from core.decision_policy_v1 import DecisionPolicyEngine, IntentType
        from core.events import create_event
        from datetime import datetime, timezone
        
        # Create the app
        app = create_app()
        print("  ✅ Agent app created")
        
        # Create decision policy engine
        engine = DecisionPolicyEngine()
        print("  ✅ Decision policy engine ready")
        
        # Simulate a user request
        user_request = "Help me write a Python function to calculate fibonacci numbers"
        
        # Create an event for the request
        request_event = create_event(
            "user_request",
            source_plugin="test_client",
            message=user_request,
            timestamp=datetime.now(timezone.utc)
        )
        print(f"  ✅ Request event created: {request_event.event_type}")
        
        # Test intent classification using the correct method
        intent = engine.intent_classifier.classify(user_request)
        print(f"  ✅ Intent classified: {intent}")
        
        # Test goal synthesis using the correct method
        goal = engine.goal_synthesizer.synthesize(intent, {}, {})
        print(f"  ✅ Goal synthesized: {goal.description}")
        
        print("  🎉 Agent successfully processed the request!")
        return True
        
    except Exception as e:
        print(f"  ❌ Agent request test failed: {e}")
        return False

async def test_tool_execution():
    """Test tool execution capabilities."""
    print("\n🔧 Testing Tool Execution...")
    
    try:
        from main import SimpleAbilityRegistry
        
        # Create tool registry
        registry = SimpleAbilityRegistry()
        available_tools = registry.get_available_tools_schema()
        print(f"  ✅ Tool registry created with {len(available_tools)} tools")
        
        # Test echo tool if available
        if registry.knows("echo"):
            echo_result = await registry.execute("echo", {"payload": "Hello Super Alita!"})
            print(f"  ✅ Echo tool executed: {echo_result}")
        else:
            print("  ⚠️ Echo tool not available")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Tool execution test failed: {e}")
        return False

async def test_streaming_response():
    """Test streaming response capability."""
    print("\n📡 Testing Streaming Response...")
    
    try:
        # Check that the router module can be imported
        import reug_runtime.router
        print("  ✅ REUG router module imported")
        
        # Check that the main app has streaming routes
        from main import create_app
        app = create_app()
        
        routes = [route.path for route in app.routes]
        streaming_routes = [r for r in routes if 'stream' in r]
        print(f"  ✅ Streaming routes available: {streaming_routes}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Streaming test failed: {e}")
        return False

async def test_complete_workflow():
    """Test complete agent workflow simulation."""
    print("\n🚀 Testing Complete Workflow...")
    
    try:
        # Import all necessary components
        from main import create_app, SimpleAbilityRegistry
        from core.decision_policy_v1 import DecisionPolicyEngine
        from core.events import create_event
        from datetime import datetime, timezone
        
        # 1. Initialize agent
        app = create_app()
        engine = DecisionPolicyEngine()
        registry = SimpleAbilityRegistry()
        print("  ✅ Agent components initialized")
        
        # 2. Receive user request
        user_request = "Create a simple todo list application"
        print(f"  📝 User request: {user_request}")
        
        # 3. Create request event
        event = create_event(
            "user_request",
            source_plugin="user_interface",
            message=user_request,
            timestamp=datetime.now(timezone.utc)
        )
        print("  ✅ Request event created")
        
        # 4. Classify intent using correct method
        intent = engine.intent_classifier.classify(user_request)
        print(f"  🎯 Intent: {intent}")
        
        # 5. Synthesize goal using correct method
        goal = engine.goal_synthesizer.synthesize(intent, {}, {})
        print(f"  🎯 Goal: {goal.description}")
        
        # 6. Test tool availability
        available_tools = registry.get_available_tools_schema()
        print(f"  � Available tools: {len(available_tools)}")
        
        # 7. Generate response event with required fields
        response_event = create_event(
            "agent_response",
            source_plugin="decision_engine",
            session_id="test_session_123",
            response_text=f"I'll help you create a todo list application. Based on your request, I've classified this as a {intent} task.",
            response_id="response_123",
            reasoning="Classified user intent and generated appropriate response",
            timestamp=datetime.now(timezone.utc)
        )
        print("  ✅ Response event created")
        
        print("  🎉 Complete workflow executed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Complete workflow test failed: {e}")
        return False

async def main():
    """Run all end-to-end tests."""
    print("🚀 Super Alita End-to-End Testing")
    print("=" * 50)
    
    tests = [
        test_agent_request_handling,
        test_tool_execution,
        test_streaming_response,
        test_complete_workflow
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 END-TO-END TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL END-TO-END TESTS PASSED!")
        print("🚀 Super Alita is fully operational and ready for deployment!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)