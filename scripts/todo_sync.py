#!/usr/bin/env python3
"""Todo Integration Script - Syncs active todos with persistent storage."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List

logger = logging.getLogger(__name__)

DEFAULT_TODO_FILE = (
    Path(__file__).resolve().parent.parent / ".vscode" / "todos.json"
)


def update_persistent_todos(todo_file: Path, new_todos: List[Any]) -> None:
    """Update the persistent todo file with new todos."""
    if todo_file.exists():
        try:
            with open(todo_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            data["todoList"] = new_todos
            data["lastModified"] = datetime.now().isoformat()

            with open(todo_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("Updated %d todos in persistent storage", len(new_todos))
        except Exception:
            logger.exception("Error updating todos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync todos to persistent file")
    parser.add_argument(
        "todos",
        nargs="?",
        help="JSON array of todos to persist",
    )
    parser.add_argument(
        "--todo-file",
        type=Path,
        default=DEFAULT_TODO_FILE,
        help="Path to todo JSON file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.todos:
        try:
            todos = json.loads(args.todos)
            update_persistent_todos(args.todo_file, todos)
        except json.JSONDecodeError:
            logger.error("Invalid JSON provided")
    else:
        logger.info("Todo integration script ready for use")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
