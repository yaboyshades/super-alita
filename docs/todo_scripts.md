# Todo Scripts

Utilities for working with the project's todo list are available in the `scripts/` directory.

## `todo_manager.py`
Initializes or loads the persistent todo list.

```bash
python scripts/todo_manager.py --todo-file path/to/todos.json
```

If the file does not exist it will be created with default entries.

## `todo_sync.py`
Updates the persistent todo file with a JSON array of todos.

```bash
python scripts/todo_sync.py '[{"id": 1, "title": "Example", "status": "not-started"}]' --todo-file path/to/todos.json
```

Both scripts log their actions and accept `--todo-file` to point to a custom location. By default the file is stored in `.vscode/todos.json` at the repository root.
