#!/usr/bin/env python3
"""
Persistent Todo Management Script for Super Alita
Manages todos with file-based persistence across VS Code sessions.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

TODO_FILE = Path(__file__).parent / '.vscode' / 'todos.json'

def load_todos() -> List[Dict[str, Any]]:
    """Load todos from persistent storage."""
    if TODO_FILE.exists():
        try:
            with open(TODO_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('todoList', [])
        except (json.JSONDecodeError, KeyError):
            pass
    return []

def save_todos(todos: List[Dict[str, Any]]) -> None:
    """Save todos to persistent storage."""
    TODO_FILE.parent.mkdir(exist_ok=True)
    data = {
        'todoList': todos,
        'lastModified': datetime.now().isoformat(),
        'version': '1.0'
    }
    with open(TODO_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def initialize_default_todos() -> List[Dict[str, Any]]:
    """Initialize with default todos for the Super Alita project."""
    return [
        {
            "id": 1,
            "title": "LADDER Planner System",
            "description": "Complete LADDER planner implementation with all stages working correctly",
            "status": "completed"
        },
        {
            "id": 2, 
            "title": "MCP Server Integration",
            "description": "Set up MCP server as background task with auto-startup",
            "status": "in-progress"
        },
        {
            "id": 3,
            "title": "Router Logic Implementation", 
            "description": "Implement complexity-based planner routing from git patches",
            "status": "not-started"
        },
        {
            "id": 4,
            "title": "Persistent Todo Management",
            "description": "Create persistent todo system that survives VS Code restarts",
            "status": "in-progress"
        }
    ]

def main():
    """Initialize or load the todo system."""
    existing_todos = load_todos()
    
    if not existing_todos:
        print("📋 Initializing default todos for Super Alita project...")
        default_todos = initialize_default_todos()
        save_todos(default_todos)
        print(f"✅ Created {len(default_todos)} default todos")
    else:
        print(f"📋 Loaded {len(existing_todos)} existing todos")
    
    print("🎯 Todo system ready! Use the manage_todo_list tool to interact with todos.")
    print(f"📁 Todos stored in: {TODO_FILE}")

if __name__ == "__main__":
    main()
