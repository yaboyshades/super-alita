#!/usr/bin/env python3
"""Persistent Todo Management Script for Super Alita.

Manages todos with file-based persistence across VS Code sessions.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


DEFAULT_TODO_FILE = (
    Path(__file__).resolve().parent.parent / ".vscode" / "todos.json"
)


def load_todos(todo_file: Path) -> List[Dict[str, Any]]:
    """Load todos from persistent storage."""
    if todo_file.exists():
        try:
            with open(todo_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("todoList", [])
        except (json.JSONDecodeError, KeyError):
            logger.exception("Failed to load todos from %s", todo_file)
    return []


def save_todos(todo_file: Path, todos: List[Dict[str, Any]]) -> None:
    """Save todos to persistent storage."""
    todo_file.parent.mkdir(exist_ok=True)
    data = {
        "todoList": todos,
        "lastModified": datetime.now().isoformat(),
        "version": "1.0",
    }
    with open(todo_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def initialize_default_todos() -> List[Dict[str, Any]]:
    """Initialize with default todos for the Super Alita project."""
    return [
        {
            "id": 1,
            "title": "LADDER Planner System",
            "description": "Complete LADDER planner implementation with all stages working correctly",
            "status": "completed",
        },
        {
            "id": 2,
            "title": "MCP Server Integration",
            "description": "Set up MCP server as background task with auto-startup",
            "status": "in-progress",
        },
        {
            "id": 3,
            "title": "Router Logic Implementation",
            "description": "Implement complexity-based planner routing from git patches",
            "status": "not-started",
        },
        {
            "id": 4,
            "title": "Persistent Todo Management",
            "description": "Create persistent todo system that survives VS Code restarts",
            "status": "in-progress",
        },
    ]


def main(todo_file: Path) -> None:
    """Initialize or load the todo system."""
    existing_todos = load_todos(todo_file)

    if not existing_todos:
        logger.info("Initializing default todos for Super Alita project...")
        default_todos = initialize_default_todos()
        save_todos(todo_file, default_todos)
        logger.info("Created %d default todos", len(default_todos))
    else:
        logger.info("Loaded %d existing todos", len(existing_todos))

    logger.info("Todo system ready! Use the manage_todo_list tool to interact with todos.")
    logger.info("Todos stored in: %s", todo_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage persistent todos")
    parser.add_argument(
        "--todo-file",
        type=Path,
        default=DEFAULT_TODO_FILE,
        help="Path to todo JSON file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    main(args.todo_file)
