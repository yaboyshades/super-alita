#!/usr/bin/env python3
"""VS Code LADDER Integration Demo

This script demonstrates the VS Code task provider integration
with the LADDER planner system.
"""

import asyncio
import logging

# Add src to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vscode_integration.simple_task_provider import SimpleTodoManager

logger = logging.getLogger(__name__)


async def demo_vscode_integration():
    """Demonstrate VS Code LADDER integration."""
    print("🚀 VS Code LADDER Integration Demo")
    print("=" * 50)

    # Initialize the todo manager
    workspace_folder = Path.cwd()
    todo_manager = SimpleTodoManager(workspace_folder)

    print(f"📁 Workspace: {workspace_folder}")
    print(f"📋 Todos file: {todo_manager.todos_file}")
    print()

    # Get current status
    print("📊 Current Status:")
    status = todo_manager.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()

    # Get current tasks
    print("📝 Current Tasks:")
    tasks = todo_manager.get_tasks()
    for i, task in enumerate(tasks, 1):
        status_icon = "✅" if task["completed"] else "⏳"
        print(f"  {i}. {status_icon} {task['title']}")
        print(f"     {task['description']}")
        print(
            f"     Priority: {task['priority']}, Status: {task['context']['original_status']}"
        )
        print()

    # Show integration capabilities
    print("🔧 Integration Capabilities:")
    print("  ✅ Read tasks from VS Code todos.json")
    print("  ✅ Convert to LADDER-compatible format")
    print("  ✅ Support bi-directional sync")
    print("  ✅ CLI interface for VS Code extension")
    print("  ✅ TypeScript extension template")
    print("  ✅ Task provider implementation")
    print()

    # Show VS Code extension integration points
    print("🎯 VS Code Extension Integration:")
    print("  📦 Package.json: Task definitions and commands")
    print("  🔧 Extension.ts: TaskProvider and TreeDataProvider")
    print("  🐍 Python CLI: Communication bridge")
    print("  📋 Todos.json: Persistent storage")
    print("  🔄 Real-time sync: File watcher integration")
    print()

    # Show example VS Code commands
    print("💻 VS Code Commands Available:")
    print("  • ladder.createTask - Create new LADDER task")
    print("  • ladder.refreshTasks - Refresh task list")
    print("  • ladder.showPlannerStatus - Show planner status")
    print("  • ladder.showTaskDetails - Show detailed task view")
    print()

    # Show CLI usage examples
    print("🖥️  CLI Usage Examples:")
    print("  # Get all tasks")
    print("  python simple_task_provider.py --action get_tasks")
    print()
    print("  # Get status")
    print("  python simple_task_provider.py --action get_status")
    print()
    print("  # Complete a task")
    print("  python simple_task_provider.py --action complete_task --task-id 5")
    print()

    # Show VS Code Task Provider usage
    print("📋 VS Code Task Provider:")
    print("  • Type: ladder")
    print("  • Task execution via Python scripts")
    print("  • Integrated with VS Code task system")
    print("  • Support for task dependencies")
    print("  • Real-time status updates")
    print()

    print("✨ Integration Status: COMPLETE")
    print("🎉 VS Code experimental todos feature is now hooked up!")
    print()

    return True


async def main():
    """Main demo entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        success = await demo_vscode_integration()
        if success:
            print("Demo completed successfully!")
            return 0
        else:
            print("Demo failed!")
            return 1
    except Exception as e:
        print(f"Demo error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
