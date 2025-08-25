#!/usr/bin/env python3
"""
Todo Integration Script - Syncs active todos with persistent storage
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TODO_FILE = Path(__file__).parent / '.vscode' / 'todos.json'

def update_persistent_todos(new_todos):
    """Update the persistent todo file with new todos."""
    if TODO_FILE.exists():
        try:
            with open(TODO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update the todoList
            data['todoList'] = new_todos
            data['lastModified'] = datetime.now().isoformat()
            
            with open(TODO_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("📋 Updated %d todos in persistent storage", len(new_todos))
        except Exception as e:
            logger.error("❌ Error updating todos: %s", e)

if __name__ == "__main__":
    # Read todos from stdin (for integration with manage_todo_list tool)
    if len(sys.argv) > 1:
        todo_json = sys.argv[1]
        try:
            todos = json.loads(todo_json)
            update_persistent_todos(todos)
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON provided")
    else:
        logger.info("💡 Todo integration script ready for use")
