#!/usr/bin/env python3
"""Standalone VS Code Task Provider CLI for LADDER Planner

This is a simplified CLI interface that manages VS Code todos without requiring
the full plugin architecture. It works directly with the todos.json file.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class SimpleTodoManager:
    """Simple todo manager that works with VS Code todos.json."""

    def __init__(self, workspace_folder: Path | None = None):
        self.workspace_folder = workspace_folder or Path.cwd()
        self.todos_file = self.workspace_folder / ".vscode" / "todos.json"

    def load_todos(self) -> dict[str, Any]:
        """Load todos from VS Code todos.json file."""
        if not self.todos_file.exists():
            return {
                "todoList": [],
                "lastModified": datetime.now(UTC).isoformat(),
                "version": "1.0",
            }

        try:
            with open(self.todos_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading todos: {e}")
            return {
                "todoList": [],
                "lastModified": datetime.now(UTC).isoformat(),
                "version": "1.0",
            }

    def save_todos(self, todos_data: dict[str, Any]) -> bool:
        """Save todos to VS Code todos.json file."""
        try:
            # Ensure .vscode directory exists
            self.todos_file.parent.mkdir(exist_ok=True)

            # Update timestamp
            todos_data["lastModified"] = datetime.now(UTC).isoformat()

            with open(self.todos_file, "w", encoding="utf-8") as f:
                json.dump(todos_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            logger.error(f"Error saving todos: {e}")
            return False

    def get_tasks(self) -> list[dict[str, Any]]:
        """Get all tasks in VS Code format."""
        todos_data = self.load_todos()
        tasks = []

        for todo in todos_data.get("todoList", []):
            task = {
                "id": str(todo.get("id", uuid4())),
                "title": todo.get("title", ""),
                "description": todo.get("description", ""),
                "completed": todo.get("status") == "completed",
                "priority": "medium",
                "createdAt": datetime.now(UTC).isoformat(),
                "updatedAt": todos_data.get(
                    "lastModified", datetime.now(UTC).isoformat()
                ),
                "tags": [],
                "context": {
                    "source": "vscode_todos",
                    "original_status": todo.get("status"),
                },
            }
            tasks.append(task)

        return tasks

    def create_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new task."""
        todos_data = self.load_todos()

        # Generate new ID
        existing_ids = {todo.get("id") for todo in todos_data.get("todoList", [])}
        new_id = 1
        while new_id in existing_ids:
            new_id += 1

        # Create new todo
        new_todo = {
            "id": new_id,
            "title": task_data.get("title", "New Task"),
            "description": task_data.get("description", ""),
            "status": "completed"
            if task_data.get("completed", False)
            else "not-started",
        }

        todos_data["todoList"].append(new_todo)

        if self.save_todos(todos_data):
            return {
                "id": str(new_id),
                "title": new_todo["title"],
                "description": new_todo["description"],
                "completed": new_todo["status"] == "completed",
                "priority": task_data.get("priority", "medium"),
                "createdAt": datetime.now(UTC).isoformat(),
                "updatedAt": datetime.now(UTC).isoformat(),
                "tags": task_data.get("tags", []),
                "context": {"source": "cli"},
            }
        else:
            raise RuntimeError("Failed to save task")

    def update_task(
        self, task_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update an existing task."""
        todos_data = self.load_todos()
        todo_list = todos_data.get("todoList", [])

        try:
            task_id_int = int(task_id)
        except ValueError:
            return None

        # Find and update the todo
        for todo in todo_list:
            if todo.get("id") == task_id_int:
                if "title" in updates:
                    todo["title"] = updates["title"]
                if "description" in updates:
                    todo["description"] = updates["description"]
                if "completed" in updates:
                    todo["status"] = (
                        "completed" if updates["completed"] else "not-started"
                    )

                if self.save_todos(todos_data):
                    return {
                        "id": task_id,
                        "title": todo["title"],
                        "description": todo["description"],
                        "completed": todo["status"] == "completed",
                        "priority": updates.get("priority", "medium"),
                        "updatedAt": datetime.now(UTC).isoformat(),
                        "tags": updates.get("tags", []),
                        "context": {"source": "cli"},
                    }
                break

        return None

    def complete_task(self, task_id: str) -> dict[str, Any] | None:
        """Mark a task as completed."""
        return self.update_task(task_id, {"completed": True})

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        todos_data = self.load_todos()
        todo_list = todos_data.get("todoList", [])

        try:
            task_id_int = int(task_id)
        except ValueError:
            return False

        # Find and remove the todo
        for i, todo in enumerate(todo_list):
            if todo.get("id") == task_id_int:
                todo_list.pop(i)
                return self.save_todos(todos_data)

        return False

    def get_status(self) -> dict[str, Any]:
        """Get status information."""
        todos_data = self.load_todos()
        todo_list = todos_data.get("todoList", [])

        return {
            "planner_active": False,  # Simple version doesn't use planner
            "task_count": len(todo_list),
            "last_sync": todos_data.get("lastModified"),
            "workspace_folder": str(self.workspace_folder),
            "todos_file_exists": self.todos_file.exists(),
            "version": todos_data.get("version", "1.0"),
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simple VS Code Todos CLI",
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

    parser.add_argument(
        "--workspace",
        type=str,
        help="Workspace folder path (defaults to current directory)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize todo manager
    workspace_folder = Path(args.workspace) if args.workspace else None
    todo_manager = SimpleTodoManager(workspace_folder)

    try:
        result = None

        if args.action == "get_tasks":
            result = todo_manager.get_tasks()

        elif args.action == "create_task":
            if not args.data:
                parser.error("--data is required for create_task action")

            try:
                task_data = json.loads(args.data)
            except json.JSONDecodeError as e:
                parser.error(f"Invalid JSON in --data: {e}")

            result = todo_manager.create_task(task_data)

        elif args.action == "update_task":
            if not args.task_id:
                parser.error("--task-id is required for update_task action")
            if not args.data:
                parser.error("--data is required for update_task action")

            try:
                updates = json.loads(args.data)
            except json.JSONDecodeError as e:
                parser.error(f"Invalid JSON in --data: {e}")

            result = todo_manager.update_task(args.task_id, updates)
            if result is None:
                raise ValueError(f"Task not found: {args.task_id}")

        elif args.action == "complete_task":
            if not args.task_id:
                parser.error("--task-id is required for complete_task action")

            result = todo_manager.complete_task(args.task_id)
            if result is None:
                raise ValueError(f"Task not found: {args.task_id}")

        elif args.action == "delete_task":
            if not args.task_id:
                parser.error("--task-id is required for delete_task action")

            result = todo_manager.delete_task(args.task_id)

        elif args.action == "get_status":
            result = todo_manager.get_status()

        # Output result as JSON
        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        error_result = {"error": str(e), "action": args.action, "success": False}
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
