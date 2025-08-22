#!/usr/bin/env python3
"""Test Super Alita Agent Cortex Integration."""

import asyncio
from pathlib import Path

from src.vscode_integration.agent_integration import SuperAlitaAgent


async def test_agent_self_development():
    """Test the agent using its own Cortex intelligence for development."""
    print("🚀 Testing Super Alita Agent with Cortex Intelligence...\n")

    # Create and initialize agent
    agent = SuperAlitaAgent(
        workspace_folder=Path("d:/Coding_Projects/super-alita-clean")
    )
    init_result = await agent.initialize()
    print(f"✨ Agent initialization: {init_result}")
    print(f"📁 Workspace: {agent.workspace_folder}")
    print("\n" + "=" * 60 + "\n")

    # Test agent enhanced methods with proper signatures
    print("🧠 Testing Agent Development Insights...")
    try:
        query = "How can I improve the agent memory and learning capabilities?"
        insights = await agent.get_development_insights(query)
        print("✅ Development insights generated:")
        print(f"Strategy: {insights.get('strategy', 'unknown')}")
        insight_text = insights.get("insights", "none")
        print(f"Insights: {insight_text[:150]}...")
        if "actionable_recommendations" in insights:
            rec_count = len(insights["actionable_recommendations"])
            print(f"Recommendations: {rec_count} items")
    except Exception as e:
        print(f"❌ Development insights error: {e}")

    print("\n📋 Testing Development Task Planning...")
    try:
        task_desc = "Enhance agent memory persistence and cross-session learning"
        task_plan = await agent.plan_development_task(task_desc)
        print("✅ Development task plan generated:")
        print(f"Plan type: {type(task_plan)}")
        if isinstance(task_plan, dict):
            print(f"Plan keys: {list(task_plan.keys())}")
            if "tasks" in task_plan:
                print(f"Tasks count: {len(task_plan['tasks'])}")
        else:
            plan_str = str(task_plan)
            print(f"Plan preview: {plan_str[:150]}...")
    except Exception as e:
        print(f"❌ Task planning error: {e}")

    # Test development KG creation
    print("\n🎯 Testing Development KG Creation...")
    try:
        if hasattr(agent, "_create_development_kg"):
            kg = agent._create_development_kg()
            node_count = len(kg.nodes)
            edge_count = len(kg.edges)
            print(f"✅ Development KG created: {node_count} nodes, {edge_count} edges")

            # Show some sample nodes
            if node_count > 0:
                sample_nodes = list(kg.nodes)[:5]
                print(f"Sample nodes: {sample_nodes}")
        else:
            print("⚠️  Development KG method not available")
    except Exception as e:
        print(f"❌ Development KG error: {e}")

    # Test MCP tool handlers
    print("\n🔧 Testing MCP Tool Integration...")
    try:
        # Test development insights tool
        if hasattr(agent, "handle_development_insights"):
            tool_result = await agent.handle_development_insights(
                {"query": "What are the current bottlenecks in the agent system?"}
            )
            print("✅ MCP development insights tool works")
            print(f"Tool result type: {type(tool_result)}")

        # Test development planning tool
        if hasattr(agent, "handle_development_planning"):
            plan_result = await agent.handle_development_planning(
                {"task": "Implement better error handling across all plugins"}
            )
            print("✅ MCP development planning tool works")
            print(f"Plan result type: {type(plan_result)}")

    except Exception as e:
        print(f"❌ MCP tool error: {e}")

    return True


if __name__ == "__main__":
    result = asyncio.run(test_agent_self_development())
    print(f"\n🎉 Agent Cortex integration test completed: {result}")
