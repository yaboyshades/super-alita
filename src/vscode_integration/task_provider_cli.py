#!/usr/bin/env python3
"""Command Line Interface for VS Code LADDER Task Provider

This script provides a CLI interface for the VS Code extension to interact
with the LADDER planner task provider. It enables creating, updating, and
querying tasks from VS Code.

Usage:
    python task_provider_cli.py --action get_tasks
    python task_provider_cli.py --action create_task --data '{"title": "New Task", "description": "Task description"}'
    python task_provider_cli.py --action get_status
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.event_bus import EventBus
    from vscode_integration.task_provider import VSCodeTaskProvider
except ImportError:
    # Alternative import path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.core.event_bus import EventBus
    from src.vscode_integration.task_provider import VSCodeTaskProvider

logger = logging.getLogger(__name__)


class TaskProviderCLI:
    """CLI interface for the VS Code task provider."""

    def __init__(self):
        self.event_bus = None
        self.task_provider = None

    async def initialize(self):
        """Initialize the task provider."""
        try:
            # Create a minimal event bus for CLI usage
            self.event_bus = EventBus()
            await self.event_bus.initialize()

            # Initialize task provider
            self.task_provider = VSCodeTaskProvider(
                event_bus=self.event_bus,
                config={
                    "sync_interval": 60,  # Longer interval for CLI usage
                    "auto_sync": False,  # Disable auto sync for CLI
                },
            )
            await self.task_provider.initialize()

        except Exception as e:
            logger.error(f"Failed to initialize task provider: {e}")
            raise

    async def get_tasks(self) -> list[dict]:
        """Get all tasks from the task provider."""
        if not self.task_provider:
            raise RuntimeError("Task provider not initialized")

        tasks = await self.task_provider.provide_tasks()
        return tasks

    async def create_task(self, task_data: dict) -> dict:
        """Create a new task."""
        if not self.task_provider:
            raise RuntimeError("Task provider not initialized")

        task = await self.task_provider.create_task(task_data)
        return task.model_dump()

    async def update_task(self, task_id: str, updates: dict) -> dict:
        """Update an existing task."""
        if not self.task_provider:
            raise RuntimeError("Task provider not initialized")

        task = await self.task_provider.update_task(task_id, updates)
        if task:
            return task.model_dump()
        else:
            raise ValueError(f"Task not found: {task_id}")

    async def complete_task(self, task_id: str) -> dict:
        """Mark a task as completed."""
        if not self.task_provider:
            raise RuntimeError("Task provider not initialized")

        task = await self.task_provider.complete_task(task_id)
        if task:
            return task.model_dump()
        else:
            raise ValueError(f"Task not found: {task_id}")

    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if not self.task_provider:
            raise RuntimeError("Task provider not initialized")

        return await self.task_provider.delete_task(task_id)

    async def get_status(self) -> dict:
        """Get status information about the task provider."""
        if not self.task_provider:
            return {
                "planner_active": False,
                "task_count": 0,
                "last_sync": None,
                "error": "Task provider not initialized",
            }

        tasks = await self.task_provider.provide_tasks()

        status = {
            "planner_active": self.task_provider.enhanced_planner is not None,
            "task_count": len(tasks),
            "last_sync": None,  # Would need to track this
            "workspace_folder": str(self.task_provider.workspace_folder),
            "todos_file_exists": self.task_provider.todos_file.exists(),
        }

        return status

    async def shutdown(self):
        """Cleanup resources."""
        if self.task_provider:
            await self.task_provider.shutdown()

        if self.event_bus:
            await self.event_bus.shutdown()


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VS Code LADDER Task Provider CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --action get_tasks
  %(prog)s --action create_task --data '{"title": "New Task", "description": "Task description"}'
  %(prog)s --action update_task --task-id "123" --data '{"completed": true}'
  %(prog)s --action get_status
        """,
    )

    parser.add_argument(
        "--action",
        required=True,
        choices=[
            "get_tasks",
            "create_task",
            "update_task",
            "complete_task",
            "delete_task",
            "get_status",
        ],
        help="Action to perform",
    )

    parser.add_argument(
        "--data", type=str, help="JSON data for create_task or update_task actions"
    )

    parser.add_argument(
        "--task-id",
        type=str,
        help="Task ID for update_task, complete_task, or delete_task actions",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cli = TaskProviderCLI()

    try:
        await cli.initialize()

        result = None

        if args.action == "get_tasks":
            result = await cli.get_tasks()

        elif args.action == "create_task":
            if not args.data:
                parser.error("--data is required for create_task action")

            try:
                task_data = json.loads(args.data)
            except json.JSONDecodeError as e:
                parser.error(f"Invalid JSON in --data: {e}")

            result = await cli.create_task(task_data)

        elif args.action == "update_task":
            if not args.task_id:
                parser.error("--task-id is required for update_task action")
            if not args.data:
                parser.error("--data is required for update_task action")

            try:
                updates = json.loads(args.data)
            except json.JSONDecodeError as e:
                parser.error(f"Invalid JSON in --data: {e}")

            result = await cli.update_task(args.task_id, updates)

        elif args.action == "complete_task":
            if not args.task_id:
                parser.error("--task-id is required for complete_task action")

            result = await cli.complete_task(args.task_id)

        elif args.action == "delete_task":
            if not args.task_id:
                parser.error("--task-id is required for delete_task action")

            result = await cli.delete_task(args.task_id)

        elif args.action == "get_status":
            result = await cli.get_status()

        # Output result as JSON
        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        error_result = {"error": str(e), "action": args.action, "success": False}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    finally:
        await cli.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
