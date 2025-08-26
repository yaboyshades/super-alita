from __future__ import annotations

from typing import Any

from cortex.common.types import ToolSpec

# Minimal adapter to existing todo store (assumes cortex.todo.store API)
try:
    from cortex.todo.store import TodoStore
except Exception:

    class TodoStore:
        def __init__(self):
            self._t = []

        def add(
            self,
            title: str,
            details: str = "",
            parent_id: str | None = None,
            depends_on: list[str] | None = None,
        ) -> str:
            tid = f"T-{len(self._t) + 1}"
            self._t.append(
                {
                    "id": tid,
                    "title": title,
                    "details": details,
                    "parent": parent_id,
                    "deps": depends_on or [],
                }
            )
            return tid


todo_write_spec = ToolSpec(
    name="todo.write",
    description="Create or extend a task plan. Use for multi-step work. Provide a list of subtasks with optional dependencies.",
    args_schema={
        "type": "object",
        "properties": {
            "parent_title": {"type": "string", "description": "Top-level objective"},
            "subtasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "details": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["title"],
                },
            },
        },
        "required": ["parent_title", "subtasks"],
    },
)


def todo_write(args: dict[str, Any], store: TodoStore | None = None) -> dict[str, Any]:
    store = store or TodoStore()
    parent_id = store.add(args["parent_title"], details="(auto-created by todo.write)")
    id_map = {}
    for item in args["subtasks"]:
        tid = store.add(
            item["title"], details=item.get("details", ""), parent_id=parent_id
        )
        id_map[item["title"]] = tid
    # second pass for deps
    for item in args["subtasks"]:
        deps = item.get("depends_on") or []
        if not deps:
            continue
        this_id = id_map[item["title"]]
        dep_ids = [id_map[d] for d in deps if d in id_map]
        # minimal API to set deps if available
        try:
            store.set_dependencies(this_id, dep_ids)  # optional advanced store
        except Exception:
            pass
    return {"parent_id": parent_id, "ids_by_title": id_map}
