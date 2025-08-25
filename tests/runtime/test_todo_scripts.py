from __future__ import annotations

import json
import sys
from pathlib import Path

from src.core.proc import run


def test_todo_manager_and_sync(tmp_path: Path) -> None:
    todo_file = tmp_path / "todos.json"
    repo_root = Path(__file__).resolve().parents[2]

    run([
        sys.executable,
        str(repo_root / "scripts" / "todo_manager.py"),
        "--todo-file",
        str(todo_file),
    ])
    data = json.loads(todo_file.read_text())
    assert data["todoList"], "default todos not initialized"

    new_todos = [{"id": 99, "title": "x", "description": "", "status": "done"}]
    run([
        sys.executable,
        str(repo_root / "scripts" / "todo_sync.py"),
        json.dumps(new_todos),
        "--todo-file",
        str(todo_file),
    ])
    data = json.loads(todo_file.read_text())
    assert data["todoList"][0]["id"] == 99
