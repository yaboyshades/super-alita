#!/usr/bin/env python3
"""Task Runner for LADDER Planner

This script executes individual LADDER tasks from VS Code.
It provides a bridge between the VS Code task system and the LADDER planner.

Usage:
    python task_runner.py --task-id <task_id> --action <action>
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_bus import EventBus
from cortex.config.planner_config import PlannerConfig
from cortex.planner.ladder_enhanced import EnhancedLadderPlanner

logger = logging.getLogger(__name__)


class TaskRunner:
    """Runner for individual LADDER tasks."""

    def __init__(self):
        self.event_bus = None
        self.planner = None

    async def initialize(self):
        """Initialize the task runner."""
        try:
            # Create event bus
            self.event_bus = EventBus()
            await self.event_bus.initialize()

            # Initialize planner
            config = PlannerConfig()
            self.planner = EnhancedLadderPlanner(
                event_bus=self.event_bus, config=config.model_dump()
            )
            await self.planner.initialize()

        except Exception as e:
            logger.error(f"Failed to initialize task runner: {e}")
            raise

    async def execute_task(self, task_id: str) -> dict:
        """Execute a specific task."""
        if not self.planner:
            raise RuntimeError("Planner not initialized")

        try:
            # Get task details
            tasks = await self.planner.get_tasks()
            task = None
            for t in tasks:
                if t.get("task_id") == task_id:
                    task = t
                    break

            if not task:
                raise ValueError(f"Task not found: {task_id}")

            # Execute the task
            result = await self.planner.execute_task(task_id)

            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": f"Task {task_id} executed successfully",
            }

        except Exception as e:
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "message": f"Task execution failed: {e}",
            }

    async def advance_stage(self, task_id: str) -> dict:
        """Advance a task to the next stage."""
        if not self.planner:
            raise RuntimeError("Planner not initialized")

        try:
            # Advance the task stage
            result = await self.planner.advance_stage(task_id)

            return {
                "success": True,
                "task_id": task_id,
                "result": result,
                "message": f"Task {task_id} stage advanced",
            }

        except Exception as e:
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "message": f"Stage advancement failed: {e}",
            }

    async def get_task_details(self, task_id: str) -> dict:
        """Get detailed information about a task."""
        if not self.planner:
            raise RuntimeError("Planner not initialized")

        try:
            tasks = await self.planner.get_tasks()
            for task in tasks:
                if task.get("task_id") == task_id:
                    return {
                        "success": True,
                        "task_id": task_id,
                        "task": task,
                        "message": "Task details retrieved",
                    }

            return {
                "success": False,
                "task_id": task_id,
                "error": "Task not found",
                "message": f"Task {task_id} not found",
            }

        except Exception as e:
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e),
                "message": f"Failed to get task details: {e}",
            }

    async def shutdown(self):
        """Cleanup resources."""
        if self.planner:
            await self.planner.shutdown()

        if self.event_bus:
            await self.event_bus.shutdown()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LADDER Task Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --task-id "abc123" --action execute
  %(prog)s --task-id "abc123" --action advance_stage
  %(prog)s --task-id "abc123" --action get_details
        """,
    )

    parser.add_argument("--task-id", required=True, help="ID of the task to operate on")

    parser.add_argument(
        "--action",
        required=True,
        choices=["execute", "advance_stage", "get_details"],
        help="Action to perform on the task",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    runner = TaskRunner()

    try:
        await runner.initialize()

        result = None

        if args.action == "execute":
            result = await runner.execute_task(args.task_id)
        elif args.action == "advance_stage":
            result = await runner.advance_stage(args.task_id)
        elif args.action == "get_details":
            result = await runner.get_task_details(args.task_id)

        # Output result as JSON
        print(json.dumps(result, indent=2, default=str))

        # Exit with error code if operation failed
        if not result.get("success", False):
            sys.exit(1)

    except Exception as e:
        error_result = {
            "success": False,
            "task_id": args.task_id,
            "action": args.action,
            "error": str(e),
            "message": f"Runner error: {e}",
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
