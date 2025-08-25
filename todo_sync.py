#!/usr/bin/env python3
"""
Todo Integration Script - Syncs active todos with persistent storage
"""
import json
import sys
from pathlib import Path

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
                
            print(f"📋 Updated {len(new_todos)} todos in persistent storage")
        except Exception as e:
            print(f"❌ Error updating todos: {e}")

if __name__ == "__main__":
    # Read todos from stdin (for integration with manage_todo_list tool)
    if len(sys.argv) > 1:
        todo_json = sys.argv[1]
        try:
            todos = json.loads(todo_json)
            update_persistent_todos(todos)
        except json.JSONDecodeError:
            print("❌ Invalid JSON provided")
    else:
        print("💡 Todo integration script ready for use")
