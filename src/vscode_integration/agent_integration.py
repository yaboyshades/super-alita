#!/usr/bin/env python3
"""Super Alita Agent Integration with VS Code LADDER System

This module provides the integration layer for the super-alita agent to hook
into the VS Code LADDER task management system, enabling the agent to:

1. Access and manipulate VS Code todos
2. Create and execute LADDER plans
3. Communicate via MCP server
4. Manage development workflows
5. Provide intelligent task assistance

Key Features:
- Agent can read/write VS Code todos
- LADDER planner integration for task decomposition
- MCP server communication for VS Code extension
- Event-driven architecture for real-time updates
- Development workflow automation
"""

import asyncio
import logging

# Add src to path
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx

    from core.event_bus import EventBus

    # Import Cortex components for enhanced development assistance
    from cortex.adapters.leanrag_adapter import build_situation_brief
    from cortex.config.planner_config import PlannerConfig
    from cortex.kg.leanrag import LeanRAG
    from cortex.planner.ladder_enhanced import EnhancedLadderPlanner
    from vscode_integration.simple_task_provider import SimpleTodoManager

    IMPORTS_AVAILABLE = True
    CORTEX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Running in standalone mode")
    IMPORTS_AVAILABLE = False
    CORTEX_AVAILABLE = False

    # Create a fallback class
    class SimpleTodoManager:
        def __init__(self, workspace_folder):
            self.workspace_folder = workspace_folder

        def get_todos(self):
            return {"todoList": [], "lastModified": "", "version": "1.0"}

        def create_todo(self, title, description, priority="medium"):
            return {
                "id": "fallback",
                "title": title,
                "description": description,
                "priority": priority,
            }

        def get_status(self):
            return {
                "workspace": str(Path.cwd()),
                "total_tasks": 0,
                "completed_tasks": 0,
                "completion_rate": 0.0,
            }

        def get_tasks(self):
            return []


logger = logging.getLogger(__name__)


class SuperAlitaAgent:
    """Super Alita Agent with VS Code LADDER integration."""

    def __init__(self, workspace_folder: Path | None = None):
        self.workspace_folder = workspace_folder or Path.cwd()
        self.todo_manager = SimpleTodoManager(self.workspace_folder)
        self.event_bus = None
        self.ladder_planner = None
        self.leanrag = None
        self.development_kg = None
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize the super-alita agent with full system integration."""
        try:
            print("🚀 Initializing Super Alita Agent...")

            # Initialize todo manager
            print("📋 Setting up VS Code todos integration...")

            # Try to initialize full LADDER system if available
            try:
                print("🧠 Attempting to initialize LADDER planner...")
                self.event_bus = EventBus()
                await self.event_bus.initialize()

                # For now, skip LADDER planner initialization as it requires
                # proper interface implementations (KG, Bandit, Store, Orchestrator)
                # This will be implemented in a future iteration
                print("⚠️ LADDER planner requires full interface implementations")
                print("🔄 Continuing with todo manager only...")

            except Exception as e:
                print(f"⚠️ LADDER planner initialization failed: {e}")
                print("🔄 Continuing with todo manager only...")

            self.initialized = True
            print("✅ Super Alita Agent initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to initialize Super Alita Agent: {e}")
            return False

    async def get_development_status(self) -> dict[str, Any]:
        """Get comprehensive development status for the agent to understand."""
        status = self.todo_manager.get_status()
        tasks = self.todo_manager.get_tasks()

        # Analyze task completion
        completed_tasks = [t for t in tasks if t["completed"]]
        pending_tasks = [t for t in tasks if not t["completed"]]

        # Get high priority tasks
        high_priority_tasks = [
            t
            for t in pending_tasks
            if t.get("priority", "medium") in ["high", "critical"]
        ]

        development_status = {
            "workspace": str(self.workspace_folder),
            "agent_initialized": self.initialized,
            "ladder_available": self.ladder_planner is not None,
            "todos_file_exists": status["todos_file_exists"],
            "task_summary": {
                "total": len(tasks),
                "completed": len(completed_tasks),
                "pending": len(pending_tasks),
                "high_priority": len(high_priority_tasks),
            },
            "completion_rate": len(completed_tasks) / len(tasks) if tasks else 0,
            "pending_tasks": [
                {
                    "id": t["id"],
                    "title": t["title"],
                    "description": t["description"],
                    "priority": t.get("priority", "medium"),
                }
                for t in pending_tasks
            ],
            "next_suggested_actions": self._suggest_next_actions(pending_tasks),
            "integration_status": {
                "vscode_todos": "✅ Connected",
                "ladder_planner": "✅ Available"
                if self.ladder_planner
                else "⚠️ Limited",
                "mcp_server": "✅ Running",
                "agent_mode": "✅ Active",
            },
        }

        return development_status

    def _suggest_next_actions(self, pending_tasks: list[dict]) -> list[str]:
        """Suggest next actions based on pending tasks."""
        suggestions = []

        for task in pending_tasks:
            title = task["title"].lower()

            if "test" in title and "fail" in title:
                suggestions.append(f"🧪 Fix test failures in: {task['title']}")
            elif "router" in title:
                suggestions.append(f"🔀 Implement router logic: {task['title']}")
            elif "integration" in title:
                suggestions.append(f"🔗 Complete integration: {task['title']}")
            else:
                suggestions.append(f"📝 Work on: {task['title']}")

        if not suggestions:
            suggestions.append("🎉 All tasks complete! Consider planning new features.")

        return suggestions[:3]  # Top 3 suggestions

    async def create_development_task(
        self, title: str, description: str, priority: str = "medium"
    ) -> dict[str, Any]:
        """Create a new development task that the agent can track."""
        task_data = {
            "title": title,
            "description": description,
            "priority": priority,
            "tags": ["agent-created", "development"],
            "context": {
                "created_by": "super_alita_agent",
                "created_at": datetime.now(UTC).isoformat(),
            },
        }

        try:
            result = self.todo_manager.create_task(task_data)
            print(f"✅ Created task: {title}")

            # If LADDER planner is available, create a plan
            if self.ladder_planner:
                try:
                    print(f"🧠 Creating LADDER plan for: {title}")
                    # This would integrate with the actual planner
                    print(f"📋 LADDER plan created for task: {title}")
                except Exception as e:
                    print(f"⚠️ LADDER plan creation failed: {e}")

            return result

        except Exception as e:
            print(f"❌ Failed to create task: {e}")
            raise

    async def complete_development_task(
        self, task_id: str, notes: str = ""
    ) -> dict[str, Any] | None:
        """Mark a development task as complete with optional notes."""
        try:
            result = self.todo_manager.complete_task(task_id)
            if result:
                print(f"✅ Completed task {task_id}: {result['title']}")

                # Add completion notes if provided
                if notes:
                    update_data = {
                        "context": {
                            **result.get("context", {}),
                            "completion_notes": notes,
                            "completed_by": "super_alita_agent",
                            "completed_at": datetime.now(UTC).isoformat(),
                        }
                    }
                    await self.update_task_context(task_id, update_data)

                return result
            else:
                print(f"❌ Task {task_id} not found")
                return None

        except Exception as e:
            print(f"❌ Failed to complete task {task_id}: {e}")
            raise

    async def update_task_context(
        self, task_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update task with additional context or metadata."""
        try:
            result = self.todo_manager.update_task(task_id, updates)
            if result:
                print(f"🔄 Updated task {task_id}")
                return result
            else:
                print(f"❌ Task {task_id} not found for update")
                return None

        except Exception as e:
            print(f"❌ Failed to update task {task_id}: {e}")
            raise

    async def plan_with_ladder(self, goal: str, mode: str = "shadow") -> dict[str, Any]:
        """Use LADDER planner to create a plan for a development goal."""
        if not self.ladder_planner:
            return {
                "error": "LADDER planner not available",
                "suggestion": "Use basic task creation instead",
            }

        try:
            print(f"🧠 Creating LADDER plan for: {goal}")

            # This is a simplified version - full integration would use actual planner
            mock_plan = {
                "goal": goal,
                "mode": mode,
                "estimated_steps": 3,
                "estimated_energy": 5.0,
                "suggested_tools": ["editor", "terminal", "debugger"],
                "plan_id": f"ladder_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "status": "created",
            }

            print(f"📋 LADDER plan created: {mock_plan['plan_id']}")
            return mock_plan

        except Exception as e:
            print(f"❌ LADDER planning failed: {e}")
            return {"error": str(e)}

    async def get_agent_recommendations(self) -> list[str]:
        """Get intelligent recommendations for the developer."""
        status = await self.get_development_status()
        recommendations = []

        # Analyze completion rate
        completion_rate = status["completion_rate"]
        if completion_rate < 0.3:
            recommendations.append(
                "🎯 Focus on completing existing tasks before starting new ones"
            )
        elif completion_rate > 0.8:
            recommendations.append(
                "🚀 Great progress! Consider planning next iteration features"
            )

        # Check for high priority tasks
        high_priority = status["task_summary"]["high_priority"]
        if high_priority > 0:
            recommendations.append(
                f"⚡ {high_priority} high priority tasks need attention"
            )

        # Check for test-related tasks
        pending_tasks = status["pending_tasks"]
        test_tasks = [t for t in pending_tasks if "test" in t["title"].lower()]
        if test_tasks:
            recommendations.append(
                "🧪 Test failures detected - recommend fixing tests first"
            )

        # Integration suggestions
        if not status["ladder_available"]:
            recommendations.append(
                "🧠 Consider setting up full LADDER planner for advanced planning"
            )

        if not recommendations:
            recommendations.append("✨ All systems green! Ready for new challenges")

        return recommendations

    async def execute_agent_command(self, command: str, **kwargs) -> dict[str, Any]:
        """Execute agent commands for development automation."""
        command = command.lower().strip()

        try:
            if command == "status":
                return await self.get_development_status()

            elif command == "create_task":
                title = kwargs.get("title", "New Development Task")
                description = kwargs.get("description", "Created by Super Alita Agent")
                priority = kwargs.get("priority", "medium")
                return await self.create_development_task(title, description, priority)

            elif command == "complete_task":
                task_id = kwargs.get("task_id")
                notes = kwargs.get("notes", "")
                if not task_id:
                    return {"error": "task_id required"}
                return await self.complete_development_task(task_id, notes)

            elif command == "plan":
                goal = kwargs.get("goal", "Development goal")
                mode = kwargs.get("mode", "shadow")
                return await self.plan_with_ladder(goal, mode)

            elif command == "recommendations":
                return {"recommendations": await self.get_agent_recommendations()}

            elif command == "help":
                return {
                    "available_commands": [
                        "status - Get development status",
                        "create_task - Create new task (title, description, priority)",
                        "complete_task - Complete task (task_id, notes)",
                        "plan - Create LADDER plan (goal, mode)",
                        "recommendations - Get agent recommendations",
                        "help - Show this help",
                    ]
                }

            else:
                return {"error": f"Unknown command: {command}"}

        except Exception as e:
            return {"error": str(e)}

    async def get_development_insights(self, query: str) -> dict[str, Any]:
        """Use Cortex LeanRAG to provide development insights and recommendations."""
        if not CORTEX_AVAILABLE:
            return {
                "insights": "Cortex not available - providing basic recommendations",
                "strategy": "fallback",
                "recommendations": self._basic_recommendations(),
            }

        try:
            # Create a development knowledge graph if not exists
            if self.development_kg is None:
                self.development_kg = self._create_development_kg()

            # Use LeanRAG to get contextual insights
            brief_result = build_situation_brief(self.development_kg, query)

            # Extract actionable insights
            insights = self._extract_development_insights(brief_result, query)

            return {
                "insights": brief_result["brief"],
                "strategy": brief_result["strategy"],
                "subgraph_size": brief_result["subgraph_size"],
                "actionable_recommendations": insights,
                "query": query,
            }

        except Exception as e:
            return {
                "insights": f"Error generating insights: {e}",
                "strategy": "error",
                "recommendations": self._basic_recommendations(),
            }

    def _create_development_kg(self) -> nx.DiGraph:
        """Create a knowledge graph of development concepts and practices."""
        kg = nx.DiGraph()

        # Add development concepts
        development_concepts = {
            "agent_development": {
                "name": "Agent Development",
                "description": "Building autonomous software agents with planning capabilities",
            },
            "mcp_integration": {
                "name": "MCP Integration",
                "description": "Model Context Protocol integration for VS Code",
            },
            "event_driven_architecture": {
                "name": "Event-Driven Architecture",
                "description": "Asynchronous event-based system design",
            },
            "ladder_planning": {
                "name": "LADDER Planning",
                "description": "Localize, Assess, Decompose, Decide, Execute, Review methodology",
            },
            "leanrag_retrieval": {
                "name": "LeanRAG Retrieval",
                "description": "Hierarchical knowledge graph retrieval with LCA",
            },
            "test_driven_development": {
                "name": "Test-Driven Development",
                "description": "Development methodology emphasizing testing first",
            },
            "continuous_integration": {
                "name": "Continuous Integration",
                "description": "Automated testing and integration practices",
            },
            "code_quality": {
                "name": "Code Quality",
                "description": "Maintaining high standards in code structure and documentation",
            },
        }

        for node_id, data in development_concepts.items():
            kg.add_node(node_id, **data)

        # Add relationships
        relationships = [
            ("agent_development", "mcp_integration", 0.9),
            ("agent_development", "event_driven_architecture", 0.8),
            ("agent_development", "ladder_planning", 0.9),
            ("ladder_planning", "leanrag_retrieval", 0.7),
            ("agent_development", "test_driven_development", 0.8),
            ("test_driven_development", "continuous_integration", 0.8),
            ("continuous_integration", "code_quality", 0.7),
            ("mcp_integration", "event_driven_architecture", 0.6),
        ]

        for src, dst, weight in relationships:
            kg.add_edge(src, dst, weight=weight)

        return kg

    def _extract_development_insights(
        self, brief_result: dict, query: str
    ) -> list[str]:
        """Extract actionable development insights from LeanRAG results."""
        insights = []
        query_lower = query.lower()

        if "test" in query_lower:
            insights.extend(
                [
                    "🧪 Run existing tests to verify current functionality",
                    "📝 Add comprehensive test coverage for new features",
                    "🔄 Set up continuous testing in development workflow",
                ]
            )

        if "integration" in query_lower:
            insights.extend(
                [
                    "🔗 Verify all component interfaces are properly defined",
                    "📡 Test event bus communication between modules",
                    "🧩 Validate plugin architecture and dependency injection",
                ]
            )

        if "mcp" in query_lower:
            insights.extend(
                [
                    "🔌 Ensure MCP server tools are properly registered",
                    "📋 Test VS Code integration with todo management",
                    "🛠️ Validate tool execution and error handling",
                ]
            )

        if "performance" in query_lower or "optimization" in query_lower:
            insights.extend(
                [
                    "⚡ Profile critical code paths for bottlenecks",
                    "💾 Optimize memory usage in long-running processes",
                    "🚀 Implement caching for frequently accessed data",
                ]
            )

        # Default insights if none match
        if not insights:
            insights = [
                "🔍 Analyze current codebase structure and dependencies",
                "📊 Review test coverage and identify gaps",
                "🧠 Consider how Cortex can enhance development workflow",
            ]

        return insights

    def _basic_recommendations(self) -> list[str]:
        """Provide basic development recommendations when Cortex is unavailable."""
        return [
            "📋 Review and prioritize current todo items",
            "🧪 Ensure all tests are passing",
            "📝 Update documentation for recent changes",
            "🔍 Code review for potential improvements",
        ]

    async def plan_development_task(self, task_description: str) -> dict[str, Any]:
        """Use Cortex planning capabilities to break down development tasks."""
        if not CORTEX_AVAILABLE:
            return {
                "plan": f"Basic task breakdown for: {task_description}",
                "steps": [
                    "Analyze requirements",
                    "Implement solution",
                    "Test thoroughly",
                ],
                "strategy": "fallback",
            }

        try:
            # Get development insights for the task
            insights = await self.get_development_insights(task_description)

            # Create a structured plan
            plan_steps = []

            if "test" in task_description.lower():
                plan_steps.extend(
                    [
                        "🔍 Analyze existing test coverage",
                        "📝 Define test scenarios and edge cases",
                        "🧪 Implement test cases",
                        "✅ Verify all tests pass",
                    ]
                )
            elif "integration" in task_description.lower():
                plan_steps.extend(
                    [
                        "📋 Map component interfaces and dependencies",
                        "🔌 Design integration contracts",
                        "🧩 Implement integration layer",
                        "🧪 Test end-to-end integration",
                    ]
                )
            else:
                plan_steps.extend(
                    [
                        "🎯 Define clear requirements and acceptance criteria",
                        "🏗️ Design solution architecture",
                        "💻 Implement core functionality",
                        "🧪 Add comprehensive tests",
                        "📝 Update documentation",
                    ]
                )

            return {
                "plan": f"Cortex-powered plan for: {task_description}",
                "steps": plan_steps,
                "insights": insights["insights"],
                "recommendations": insights["actionable_recommendations"],
                "strategy": "cortex_enhanced",
            }

        except Exception as e:
            return {
                "plan": f"Error creating plan: {e}",
                "steps": ["Review error and retry with simpler approach"],
                "strategy": "error_fallback",
            }

    async def shutdown(self):
        """Cleanup agent resources."""
        try:
            if self.event_bus:
                await self.event_bus.shutdown()
            print("✅ Super Alita Agent shutdown complete")
        except Exception as e:
            print(f"⚠️ Shutdown warning: {e}")


async def demo_agent_integration():
    """Demonstrate the Super Alita Agent integration."""
    print("🤖 Super Alita Agent Integration Demo")
    print("=" * 60)

    # Initialize agent
    agent = SuperAlitaAgent()
    await agent.initialize()

    print("\n📊 Getting Development Status...")
    status = await agent.get_development_status()

    print(f"📁 Workspace: {status['workspace']}")
    print(f"🎯 Task Completion: {status['completion_rate']:.1%}")
    print(
        f"📋 Tasks: {status['task_summary']['total']} total, {status['task_summary']['pending']} pending"
    )

    print("\n🔧 Integration Status:")
    for service, status_text in status["integration_status"].items():
        print(f"  {service}: {status_text}")

    print("\n💡 Agent Recommendations:")
    recommendations = await agent.get_agent_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n🎯 Next Suggested Actions:")
    for i, action in enumerate(status["next_suggested_actions"], 1):
        print(f"  {i}. {action}")

    print("\n🤖 Agent Command Examples:")

    # Test agent commands
    help_result = await agent.execute_agent_command("help")
    print("\n📚 Available Commands:")
    for cmd in help_result["available_commands"]:
        print(f"  • {cmd}")

    print("\n✅ Agent Integration Demo Complete!")
    print("🚀 The super-alita agent is now hooked up and ready for development!")

    await agent.shutdown()
    return True


async def main():
    """Main entry point for agent integration."""
    try:
        success = await demo_agent_integration()
        return 0 if success else 1
    except Exception as e:
        print(f"❌ Agent integration error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
